import json
import pickle
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
import shap
from lime.lime_text import LimeTextExplainer

torch.manual_seed(42)
np.random.seed(42)

with open("splits.json") as f:
    data = json.load(f)
test_rows = [r for r in data["test"] if r["label"] == 1 and r["rationale_mask"] is not None]
print("Evaluating faithfulness on", len(test_rows), "toxic test posts with rationale masks")

# ---------- Load LogReg ----------
with open("logreg_model.pkl", "rb") as f:
    lr_bundle = pickle.load(f)
vec, clf = lr_bundle["vec"], lr_bundle["clf"]

# ---------- Load BiLSTM ----------
with open("vocab.pkl", "rb") as f:
    vocab = pickle.load(f)
MAX_LEN = 40

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
        if return_attn:
            return logit, weights
        return logit

bilstm = BiLSTMAttn(len(vocab))
bilstm.load_state_dict(torch.load("bilstm_model.pt", map_location="cpu"))
bilstm.eval()

def encode(tokens):
    ids = [vocab.get(t, 1) for t in tokens[:MAX_LEN]]
    ids = ids + [0] * (MAX_LEN - len(ids))
    return ids

def bilstm_predict_proba(text_list):
    """text_list: list of raw strings (space joined tokens). Returns prob of class 1."""
    batch = []
    for t in text_list:
        toks = t.split()
        batch.append(encode(toks))
    x = torch.tensor(batch)
    with torch.no_grad():
        logit = bilstm(x)
        prob = torch.sigmoid(logit).numpy()
    return np.stack([1 - prob, prob], axis=1)

def lr_predict_proba(text_list):
    X = vec.transform(text_list)
    return clf.predict_proba(X)

# ---------- Explanation extraction ----------

def get_attention_scores(tokens):
    ids = encode(tokens)
    x = torch.tensor([ids])
    with torch.no_grad():
        _, weights = bilstm(x, return_attn=True)
    w = weights.numpy()[0][:min(len(tokens), MAX_LEN)]
    return w

lime_explainer = LimeTextExplainer(class_names=["normal", "toxic"], bow=False, random_state=42)

def get_lime_scores(tokens, predict_fn, n_features=None):
    text = " ".join(tokens)
    n_feat = n_features or len(tokens)
    try:
        exp = lime_explainer.explain_instance(text, predict_fn, num_features=n_feat, num_samples=200, labels=(1,))
        word_scores = dict(exp.as_list(label=1))
    except Exception:
        return np.zeros(len(tokens))
    scores = np.array([abs(word_scores.get(tok, 0.0)) for tok in tokens])
    return scores

def get_shap_scores_lr(tokens):
    text = " ".join(tokens)
    X = vec.transform([text])
    explainer = shap.LinearExplainer(clf, vec.transform([" ".join(r["tokens"]) for r in data["train"][:200]]))
    shap_vals = explainer.shap_values(X)
    sv = np.array(shap_vals).flatten()
    feat_names = vec.get_feature_names_out()
    nz = X.nonzero()[1]
    val_map = {feat_names[i]: sv[i] for i in nz}
    scores = np.array([abs(val_map.get(tok, 0.0)) for tok in tokens])
    return scores

def get_shap_scores_bilstm(tokens):
    # KernelSHAP over token presence (mask-based), background = all-masked (empty) input
    n = len(tokens)
    def f(mask_matrix):
        texts = []
        for row in mask_matrix:
            kept = [tok if keep else "" for tok, keep in zip(tokens, row)]
            texts.append(" ".join([t for t in kept if t]))
        return bilstm_predict_proba(texts)[:, 1]
    background = np.zeros((1, n))
    explainer = shap.KernelExplainer(f, background)
    shap_vals = explainer.shap_values(np.ones((1, n)), nsamples=100, silent=True)
    return np.abs(np.array(shap_vals).flatten())

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

# ---------- Run evaluation ----------
N_SAMPLES = 150  # subset for tractable SHAP/LIME runtime
subset = test_rows[:N_SAMPLES]

results = {
    "logreg_lime": [], "logreg_shap": [],
    "bilstm_attention": [], "bilstm_lime": [], "bilstm_shap": []
}

for i, r in enumerate(subset):
    tokens = r["tokens"][:MAX_LEN]
    mask = r["rationale_mask"][:MAX_LEN]
    if len(mask) < len(tokens):
        tokens = tokens[:len(mask)]

    # LogReg + LIME
    s = get_lime_scores(tokens, lr_predict_proba)
    iou, auprc = iou_and_auprc(s, mask)
    if iou is not None: results["logreg_lime"].append((iou, auprc))

    # LogReg + SHAP
    try:
        s = get_shap_scores_lr(tokens)
        iou, auprc = iou_and_auprc(s, mask)
        if iou is not None: results["logreg_shap"].append((iou, auprc))
    except Exception as e:
        pass

    # BiLSTM + Attention
    s = get_attention_scores(tokens)
    iou, auprc = iou_and_auprc(s, mask)
    if iou is not None: results["bilstm_attention"].append((iou, auprc))

    # BiLSTM + LIME
    s = get_lime_scores(tokens, bilstm_predict_proba)
    iou, auprc = iou_and_auprc(s, mask)
    if iou is not None: results["bilstm_lime"].append((iou, auprc))

    # BiLSTM + SHAP (kernel-based, full sample)
    try:
        s = get_shap_scores_bilstm(tokens)
        iou, auprc = iou_and_auprc(s, mask)
        if iou is not None: results["bilstm_shap"].append((iou, auprc))
    except Exception:
        pass

    if (i + 1) % 25 == 0:
        print(f"processed {i+1}/{len(subset)}")

print("\n=== Faithfulness Results (mean over samples) ===")
summary = {}
for k, v in results.items():
    if not v:
        continue
    ious = [x[0] for x in v]
    auprcs = [x[1] for x in v if x[1] is not None]
    summary[k] = {
        "n": len(v),
        "mean_iou": float(np.mean(ious)), "std_iou": float(np.std(ious)),
        "mean_auprc": float(np.mean(auprcs)) if auprcs else None,
        "std_auprc": float(np.std(auprcs)) if auprcs else None,
    }
    print(f"{k}: n={len(v)} mean_IOU={np.mean(ious):.4f} mean_AUPRC={np.mean(auprcs) if auprcs else float('nan'):.4f}")

with open("faithfulness_results.json", "w") as f:
    json.dump(summary, f, indent=2)

# Save raw per-sample scores for paired significance testing
raw = {k: v for k, v in results.items()}
with open("faithfulness_raw_scores.json", "w") as f:
    json.dump(raw, f, indent=2)
print("\nSaved faithfulness_results.json and faithfulness_raw_scores.json")
