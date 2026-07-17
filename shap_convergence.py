"""
Tests convergence of the BiLSTM's KernelSHAP faithfulness estimate as the
coalition sample budget (nsamples) increases, addressing whether the
comparatively low BiLSTM-SHAP faithfulness (Table 1) reflects genuine model
behavior or under-converged estimation (Proposition 2 in the paper).
Uses a smaller fixed sub-sample (n=30 posts) since this repeats KernelSHAP
at 4 budgets, which is expensive.
"""
import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
import shap
import pickle
from collections import Counter

torch.manual_seed(42)
np.random.seed(42)

with open("splits.json") as f:
    data = json.load(f)
test_rows = [r for r in data["test"] if r["label"] == 1 and r["rationale_mask"] is not None]
subset = test_rows[:30]
MAX_LEN = 40

with open("vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

def encode(tokens):
    ids = [vocab.get(t, 1) for t in tokens[:MAX_LEN]]
    return ids + [0] * (MAX_LEN - len(ids))

class BiLSTMAttn(nn.Module):
    def __init__(self, vocab_size, emb_dim=100, hidden_dim=64):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(hidden_dim * 2, 1)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(0.3)
    def forward(self, x, lengths=None, return_attn=False):
        mask = (x != 0).float()
        emb = self.emb(x)
        out, _ = self.lstm(emb)
        scores = self.attn(out).squeeze(-1)
        scores = scores.masked_fill(mask == 0, -1e9)
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), out).squeeze(1)
        logit = self.fc(self.dropout(context)).squeeze(-1)
        return (logit, weights) if return_attn else logit

model = BiLSTMAttn(len(vocab), hidden_dim=64)
model.load_state_dict(torch.load("bilstm_h64.pt", map_location="cpu"))
model.eval()

def predict_proba(text_list):
    batch = [encode(t.split()) for t in text_list]
    x = torch.tensor(batch)
    with torch.no_grad():
        prob = torch.sigmoid(model(x)).numpy()
    return np.stack([1 - prob, prob], axis=1)

def get_shap_scores(tokens, nsamples):
    n = len(tokens)
    def f(mask_matrix):
        texts = []
        for row in mask_matrix:
            kept = [tok if keep else "" for tok, keep in zip(tokens, row)]
            texts.append(" ".join([t for t in kept if t]))
        return predict_proba(texts)[:, 1]
    background = np.zeros((1, n))
    explainer = shap.KernelExplainer(f, background)
    shap_vals = explainer.shap_values(np.ones((1, n)), nsamples=nsamples, silent=True)
    return np.abs(np.array(shap_vals).flatten())

def iou_and_auprc(scores, rationale_mask):
    n = min(len(scores), len(rationale_mask))
    scores = np.array(scores[:n]); mask = np.array(rationale_mask[:n])
    if mask.sum() == 0 or n == 0:
        return None, None
    k = int(mask.sum())
    top_idx = np.argsort(-scores)[:k]
    pred_mask = np.zeros(n); pred_mask[top_idx] = 1
    inter = np.logical_and(pred_mask, mask).sum(); union = np.logical_or(pred_mask, mask).sum()
    iou = inter / union if union > 0 else 0.0
    try:
        auprc = average_precision_score(mask, scores)
    except Exception:
        auprc = None
    return iou, auprc

budgets = [50, 100, 200, 400]
results = {}
for budget in budgets:
    ious, auprcs = [], []
    for r in subset:
        tokens = r["tokens"][:MAX_LEN]
        mask = r["rationale_mask"][:MAX_LEN]
        if len(mask) < len(tokens):
            tokens = tokens[:len(mask)]
        s = get_shap_scores(tokens, budget)
        iou, auprc = iou_and_auprc(s, mask)
        if iou is not None:
            ious.append(iou); auprcs.append(auprc)
    results[budget] = {"mean_iou": float(np.mean(ious)), "std_iou": float(np.std(ious)),
                        "mean_auprc": float(np.mean(auprcs)), "n": len(ious)}
    print(f"nsamples={budget}: mean_IOU={np.mean(ious):.4f} (SD={np.std(ious):.4f}) mean_AUPRC={np.mean(auprcs):.4f}  n={len(ious)}")

with open("shap_convergence_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved shap_convergence_results.json")
