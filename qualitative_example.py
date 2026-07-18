"""
Qualitative case study (Figure 6): one real HateXplain test post, with human
rationale plus every applicable explanation method (Attention where the model
has it, LIME, SHAP) for all three trained models side by side.

The example (post_id 20339304_gab, Gab) is a real, HateXplain-annotated toxic
post -- not authored for this figure -- chosen because it is short (11 tokens),
correctly classified as toxic by all three trained models, has a partial
(not full-sentence) human rationale mask (82% of tokens), and contains no
identity-based slur tokens, @user mentions, or named public figures, while
still being a genuine example of the antisemitic-conspiracy rhetoric the
HateXplain annotators labeled as hate speech.

Requires: logreg_model.pkl, bilstm_model.pt + vocab.pkl, transformer_model.pt
+ transformer_vocab.pkl, splits.json (i.e. run after the main training +
faithfulness_eval.py / transformer_faithfulness_eval.py steps).
"""
import json
import pickle
import math
import numpy as np
import torch
import torch.nn as nn
import shap
from lime.lime_text import LimeTextExplainer
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

MAX_LEN = 40
POST_ID = "20339304_gab"

with open("splits.json") as f:
    data = json.load(f)
row = next(r for r in data["test"] if r["post_id"] == POST_ID)
tokens = row["tokens"]
human_mask = np.array(row["rationale_mask"], dtype=float)
n = len(tokens)
print(f"Example post ({n} tokens): {' '.join(tokens)}")

# ---------------- LogReg ----------------
with open("logreg_model.pkl", "rb") as f:
    lr_bundle = pickle.load(f)
vec, clf = lr_bundle["vec"], lr_bundle["clf"]

def lr_predict_proba(text_list):
    return clf.predict_proba(vec.transform(text_list))

def get_shap_scores_lr(tokens):
    text = " ".join(tokens)
    X = vec.transform([text])
    bg = vec.transform([" ".join(r["tokens"]) for r in data["train"][:200]])
    explainer = shap.LinearExplainer(clf, bg)
    shap_vals = explainer.shap_values(X)
    sv = np.array(shap_vals).flatten()
    feat_names = vec.get_feature_names_out()
    nz = X.nonzero()[1]
    val_map = {feat_names[i]: sv[i] for i in nz}
    return np.array([abs(val_map.get(tok, 0.0)) for tok in tokens])

# ---------------- BiLSTM ----------------
with open("vocab.pkl", "rb") as f:
    bilstm_vocab = pickle.load(f)

class BiLSTMAttn(nn.Module):
    def __init__(self, vocab_size, emb_dim=100, hidden_dim=64):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(hidden_dim * 2, 1)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(0.3)
    def forward(self, x, return_attn=False):
        mask = (x != 0).float()
        emb = self.emb(x)
        out, _ = self.lstm(emb)
        scores = self.attn(out).squeeze(-1)
        scores = scores.masked_fill(mask == 0, -1e9)
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), out).squeeze(1)
        logit = self.fc(self.dropout(context)).squeeze(-1)
        return (logit, weights) if return_attn else logit

bilstm = BiLSTMAttn(len(bilstm_vocab))
bilstm.load_state_dict(torch.load("bilstm_model.pt", map_location="cpu"))
bilstm.eval()

def encode_bilstm(tokens):
    ids = [bilstm_vocab.get(t, 1) for t in tokens[:MAX_LEN]]
    return ids + [0] * (MAX_LEN - len(ids))

def bilstm_predict_proba(text_list):
    batch = [encode_bilstm(t.split()) for t in text_list]
    x = torch.tensor(batch)
    with torch.no_grad():
        prob = torch.sigmoid(bilstm(x)).numpy()
    return np.stack([1 - prob, prob], axis=1)

def get_attention_scores_bilstm(tokens):
    x = torch.tensor([encode_bilstm(tokens)])
    with torch.no_grad():
        _, w = bilstm(x, return_attn=True)
    return w.numpy()[0][:len(tokens)]

def get_shap_scores_bilstm(tokens):
    n = len(tokens)
    def f(mask_matrix):
        texts = []
        for r in mask_matrix:
            kept = [tok if keep else "" for tok, keep in zip(tokens, r)]
            texts.append(" ".join([t for t in kept if t]))
        return bilstm_predict_proba(texts)[:, 1]
    explainer = shap.KernelExplainer(f, np.zeros((1, n)))
    vals = explainer.shap_values(np.ones((1, n)), nsamples=100, silent=True)
    return np.abs(np.array(vals).flatten())

# ---------------- Transformer ----------------
with open("transformer_vocab.pkl", "rb") as f:
    trans_vocab = pickle.load(f)

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

transformer = MiniTransformer(len(trans_vocab))
transformer.load_state_dict(torch.load("transformer_model.pt", map_location="cpu"))
transformer.eval()

def encode_transformer(tokens):
    ids = [trans_vocab["<cls>"]] + [trans_vocab.get(t, 1) for t in tokens[:MAX_LEN - 1]]
    return ids + [0] * (MAX_LEN - len(ids))

def transformer_predict_proba(text_list):
    batch = [encode_transformer(t.split()) for t in text_list]
    x = torch.tensor(batch)
    with torch.no_grad():
        prob = torch.sigmoid(transformer(x)).numpy()
    return np.stack([1 - prob, prob], axis=1)

def get_cls_attention_scores(tokens):
    x = torch.tensor([encode_transformer(tokens)])
    with torch.no_grad():
        _, cls_attn = transformer(x, return_attn=True)
    return cls_attn.numpy()[0][1:1 + len(tokens)]

def get_shap_scores_transformer(tokens):
    n = len(tokens)
    def f(mask_matrix):
        texts = []
        for r in mask_matrix:
            kept = [tok if keep else "" for tok, keep in zip(tokens, r)]
            texts.append(" ".join([t for t in kept if t]))
        return transformer_predict_proba(texts)[:, 1]
    explainer = shap.KernelExplainer(f, np.zeros((1, n)))
    vals = explainer.shap_values(np.ones((1, n)), nsamples=100, silent=True)
    return np.abs(np.array(vals).flatten())

# ---------------- Shared LIME helper ----------------
lime_explainer = LimeTextExplainer(class_names=["normal", "toxic"], bow=False, random_state=42)

def get_lime_scores(tokens, predict_fn):
    text = " ".join(tokens)
    exp = lime_explainer.explain_instance(text, predict_fn, num_features=len(tokens), num_samples=300, labels=(1,))
    word_scores = dict(exp.as_list(label=1))
    return np.array([abs(word_scores.get(tok, 0.0)) for tok in tokens])

def norm(scores):
    scores = np.array(scores, dtype=float)
    lo, hi = scores.min(), scores.max()
    if hi - lo < 1e-12:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)

print("Predictions (P(toxic)):")
print("  LogReg:     ", round(float(lr_predict_proba([" ".join(tokens)])[0, 1]), 3))
print("  BiLSTM:     ", round(float(bilstm_predict_proba([" ".join(tokens)])[0, 1]), 3))
print("  Transformer:", round(float(transformer_predict_proba([" ".join(tokens)])[0, 1]), 3))

print("Computing explanations...")
rows = [
    ("Human rationale", norm(human_mask)),
    ("LogReg - LIME", norm(get_lime_scores(tokens, lr_predict_proba))),
    ("LogReg - SHAP", norm(get_shap_scores_lr(tokens))),
    ("BiLSTM - Attention", norm(get_attention_scores_bilstm(tokens))),
    ("BiLSTM - LIME", norm(get_lime_scores(tokens, bilstm_predict_proba))),
    ("BiLSTM - SHAP", norm(get_shap_scores_bilstm(tokens))),
    ("Transformer - CLS Attn.", norm(get_cls_attention_scores(tokens))),
    ("Transformer - LIME", norm(get_lime_scores(tokens, transformer_predict_proba))),
    ("Transformer - SHAP", norm(get_shap_scores_transformer(tokens))),
]

with open("qualitative_example_scores.json", "w") as f:
    json.dump({"post_id": POST_ID, "tokens": tokens,
               "rows": [(name, list(map(float, s))) for name, s in rows]}, f, indent=2)
print("Saved qualitative_example_scores.json")

# ---------------- Figure ----------------
fig, ax = plt.subplots(figsize=(11, 6.5))
row_height = 1.0
cell_w = 1.0
cmap = plt.cm.Oranges

for ridx, (label, scores) in enumerate(rows):
    y = len(rows) - ridx
    ax.text(-0.3, y + 0.5, label, ha="right", va="center", fontsize=10.5,
             fontweight="bold" if label == "Human rationale" else "normal")
    for cidx, (tok, s) in enumerate(zip(tokens, scores)):
        color = cmap(0.15 + 0.75 * s)
        ax.add_patch(mpatches.Rectangle((cidx * cell_w, y), cell_w * 0.94, row_height * 0.82,
                                         facecolor=color, edgecolor="none"))
        ax.text(cidx * cell_w + cell_w * 0.47, y + row_height * 0.41, tok,
                 ha="center", va="center", fontsize=8.3, rotation=38 if len(tok) > 8 else 0,
                 color="black")
    if label == "Human rationale":
        ax.plot([-0.05, n * cell_w - 0.06], [y - 0.03, y - 0.03], color="black", linewidth=1.2)

ax.set_xlim(-4.6, n * cell_w + 0.3)
ax.set_ylim(0.3, len(rows) + 1.1)
ax.axis("off")
ax.set_title(
    f'Qualitative comparison on a real HateXplain test post: "{" ".join(tokens)}"\n'
    "(darker = more important to that method; all three models correctly predict toxic)",
    fontsize=11, pad=14,
)
plt.tight_layout()
plt.savefig("figures/fig6_qualitative_example.png", dpi=300, bbox_inches="tight")
print("Saved figures/fig6_qualitative_example.png")
