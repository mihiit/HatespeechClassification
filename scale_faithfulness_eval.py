import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from lime.lime_text import LimeTextExplainer

torch.manual_seed(42)
np.random.seed(42)

with open("splits.json") as f:
    data = json.load(f)
test_rows = [r for r in data["test"] if r["label"] == 1 and r["rationale_mask"] is not None]
MAX_LEN = 40
SAMPLE_N = 80
subset = test_rows[:SAMPLE_N]

counter = {}
from collections import Counter as C
cnt = C()
for r in data["train"]:
    cnt.update(r["tokens"])
vocab = {"<pad>": 0, "<unk>": 1}
for tok, c in cnt.items():
    if c >= 2:
        vocab[tok] = len(vocab)

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
        context = self.dropout(context)
        logit = self.fc(context).squeeze(-1)
        return (logit, weights) if return_attn else logit

def iou_and_auprc(scores, rationale_mask, k=None):
    n = min(len(scores), len(rationale_mask))
    scores = np.array(scores[:n])
    mask = np.array(rationale_mask[:n])
    if mask.sum() == 0 or n == 0:
        return None, None
    k = k or int(mask.sum())
    top_idx = np.argsort(-scores)[:k]
    pred_mask = np.zeros(n)
    pred_mask[top_idx] = 1
    inter = np.logical_and(pred_mask, mask).sum()
    union = np.logical_or(pred_mask, mask).sum()
    iou = inter / union if union > 0 else 0.0
    try:
        auprc = average_precision_score(mask, scores)
    except Exception:
        auprc = None
    return iou, auprc

lime_explainer = LimeTextExplainer(class_names=["normal", "toxic"], bow=False, random_state=42)

def get_lime_scores(tokens, predict_fn):
    text = " ".join(tokens)
    try:
        exp = lime_explainer.explain_instance(text, predict_fn, num_features=len(tokens), num_samples=150, labels=(1,))
        word_scores = dict(exp.as_list(label=1))
    except Exception:
        return np.zeros(len(tokens))
    return np.array([abs(word_scores.get(tok, 0.0)) for tok in tokens])

scale_results = {}
for hd in [32, 64, 128, 256]:
    model = BiLSTMAttn(len(vocab), hidden_dim=hd)
    model.load_state_dict(torch.load(f"bilstm_h{hd}.pt", map_location="cpu"))
    model.eval()

    def predict_proba(text_list, _model=model):
        batch = [encode(t.split()) for t in text_list]
        x = torch.tensor(batch)
        with torch.no_grad():
            prob = torch.sigmoid(_model(x)).numpy()
        return np.stack([1 - prob, prob], axis=1)

    attn_ious, attn_auprcs = [], []
    lime_ious, lime_auprcs = [], []
    for r in subset:
        tokens = r["tokens"][:MAX_LEN]
        mask = r["rationale_mask"][:MAX_LEN]
        if len(mask) < len(tokens):
            tokens = tokens[:len(mask)]
        ids = encode(tokens)
        x = torch.tensor([ids])
        with torch.no_grad():
            _, w = model(x, return_attn=True)
        attn_scores = w.numpy()[0][:len(tokens)]
        iou, auprc = iou_and_auprc(attn_scores, mask)
        if iou is not None:
            attn_ious.append(iou); attn_auprcs.append(auprc)

        lime_scores = get_lime_scores(tokens, predict_proba)
        iou, auprc = iou_and_auprc(lime_scores, mask)
        if iou is not None:
            lime_ious.append(iou); lime_auprcs.append(auprc)

    scale_results[hd] = {
        "attn_iou": float(np.mean(attn_ious)), "attn_auprc": float(np.mean(attn_auprcs)),
        "lime_iou": float(np.mean(lime_ious)), "lime_auprc": float(np.mean(lime_auprcs)),
        "n": len(attn_ious)
    }
    print(f"hidden_dim={hd}: attn_IOU={np.mean(attn_ious):.4f} lime_IOU={np.mean(lime_ious):.4f}")

with open("scale_faithfulness_results.json", "w") as f:
    json.dump(scale_results, f, indent=2)
print("Saved scale_faithfulness_results.json")
