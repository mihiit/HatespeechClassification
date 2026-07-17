import json
import pickle
import math
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
subset = test_rows[:150]  # same fixed sample as faithfulness_eval.py

MAX_LEN = 40
with open("transformer_vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

def encode(tokens):
    ids = [vocab["<cls>"]] + [vocab.get(t, 1) for t in tokens[:MAX_LEN - 1]]
    ids = ids + [0] * (MAX_LEN - len(ids))
    return ids

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=MAX_LEN):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class MiniTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_heads=4, d_ff=128, n_layers=2, dropout=0.2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.fc = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, return_attn=False):
        mask = (x == 0)
        emb = self.pos(self.emb(x))
        cls_attn = None
        if return_attn:
            attn_layer = self.encoder.layers[0].self_attn
            _, attn_weights = attn_layer(emb, emb, emb, key_padding_mask=mask, average_attn_weights=True, need_weights=True)
            cls_attn = attn_weights[:, 0, :]
        out = self.encoder(emb, src_key_padding_mask=mask)
        logit = self.fc(self.dropout(out[:, 0, :])).squeeze(-1)
        return (logit, cls_attn) if return_attn else logit

model = MiniTransformer(len(vocab))
model.load_state_dict(torch.load("transformer_model.pt", map_location="cpu"))
model.eval()

def predict_proba(text_list):
    batch = [encode(t.split()) for t in text_list]
    x = torch.tensor(batch)
    with torch.no_grad():
        prob = torch.sigmoid(model(x)).numpy()
    return np.stack([1 - prob, prob], axis=1)

def get_cls_attention_scores(tokens):
    ids = encode(tokens)
    x = torch.tensor([ids])
    with torch.no_grad():
        _, cls_attn = model(x, return_attn=True)
    # cls_attn[0] has length MAX_LEN; index 0 is CLS itself, tokens start at index 1
    w = cls_attn.numpy()[0][1:1 + len(tokens)]
    return w

lime_explainer = LimeTextExplainer(class_names=["normal", "toxic"], bow=False, random_state=42)

def get_lime_scores(tokens):
    text = " ".join(tokens)
    try:
        exp = lime_explainer.explain_instance(text, predict_proba, num_features=len(tokens), num_samples=200, labels=(1,))
        word_scores = dict(exp.as_list(label=1))
    except Exception:
        return np.zeros(len(tokens))
    return np.array([abs(word_scores.get(tok, 0.0)) for tok in tokens])

def get_shap_scores(tokens):
    n = len(tokens)
    def f(mask_matrix):
        texts = []
        for row in mask_matrix:
            kept = [tok if keep else "" for tok, keep in zip(tokens, row)]
            texts.append(" ".join([t for t in kept if t]))
        return predict_proba(texts)[:, 1]
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

results = {"transformer_attention": [], "transformer_lime": [], "transformer_shap": []}

for i, r in enumerate(subset):
    tokens = r["tokens"][:MAX_LEN - 1]
    mask = r["rationale_mask"][:MAX_LEN - 1]
    if len(mask) < len(tokens):
        tokens = tokens[:len(mask)]

    s = get_cls_attention_scores(tokens)
    iou, auprc = iou_and_auprc(s, mask)
    if iou is not None: results["transformer_attention"].append((iou, auprc))

    s = get_lime_scores(tokens)
    iou, auprc = iou_and_auprc(s, mask)
    if iou is not None: results["transformer_lime"].append((iou, auprc))

    try:
        s = get_shap_scores(tokens)
        iou, auprc = iou_and_auprc(s, mask)
        if iou is not None: results["transformer_shap"].append((iou, auprc))
    except Exception:
        pass

    if (i + 1) % 25 == 0:
        print(f"processed {i+1}/{len(subset)}")

print("\n=== Transformer Faithfulness Results ===")
summary = {}
for k, v in results.items():
    if not v:
        continue
    ious = [x[0] for x in v]
    auprcs = [x[1] for x in v if x[1] is not None]
    summary[k] = {
        "n": len(v), "mean_iou": float(np.mean(ious)), "std_iou": float(np.std(ious)),
        "mean_auprc": float(np.mean(auprcs)) if auprcs else None,
        "std_auprc": float(np.std(auprcs)) if auprcs else None,
    }
    print(f"{k}: n={len(v)} mean_IOU={np.mean(ious):.4f} mean_AUPRC={np.mean(auprcs) if auprcs else float('nan'):.4f}")

with open("transformer_faithfulness_results.json", "w") as f:
    json.dump(summary, f, indent=2)
raw = {k: v for k, v in results.items()}
with open("transformer_faithfulness_raw.json", "w") as f:
    json.dump(raw, f, indent=2)
print("\nSaved transformer_faithfulness_results.json and transformer_faithfulness_raw.json")
