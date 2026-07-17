"""
Generalization check: evaluates the three HateXplain-trained models
(LogReg, BiLSTM h=64, Transformer) on the independent Davidson et al. (2017)
hate speech / offensive language dataset, without any retraining.
No rationale annotations exist in Davidson, so this checks classification
generalization only, not faithfulness.

NOTE ON CHECKPOINTS: this script loads "bilstm_h64.pt", which is produced by
scale_sweep_train.py (Section 7.3's scaling experiment), not "bilstm_model.pt"
from train_bilstm.py (Section 7.1's main comparison). Both are trained with the
same hyperparameters and seed and should perform near-identically, but are not
guaranteed to be bit-identical due to DataLoader shuffle-order differences
between the two scripts. Run scale_sweep_train.py before this script (see
README step ordering) so bilstm_h64.pt exists.
"""
import json
import re
import pickle
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

MAX_LEN = 40

df = pd.read_csv("davidson/data/labeled_data.csv")
# class: 0=hate speech, 1=offensive, 2=neither -> binary toxic = (0 or 1)
df["label"] = (df["class"] != 2).astype(int)

def simple_tokenize(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-zA-Z0-9'\s]", " ", text)
    return text.lower().split()

df["tokens"] = df["tweet"].apply(simple_tokenize)
# Use a random sample for tractability with SHAP-free eval (accuracy only, so use full set is fine, it's fast)
rows = df[["tokens", "label"]].to_dict("records")
print(f"Davidson eval set: {len(rows)} posts, toxic={sum(r['label'] for r in rows)}, normal={len(rows)-sum(r['label'] for r in rows)}")

results = {}

# ---------------- LogReg ----------------
with open("logreg_model.pkl", "rb") as f:
    lr = pickle.load(f)
vec, clf = lr["vec"], lr["clf"]
texts = [" ".join(r["tokens"]) for r in rows]
X = vec.transform(texts)
preds = clf.predict(X)
gts = [r["label"] for r in rows]
results["logreg"] = {"acc": accuracy_score(gts, preds), "f1": f1_score(gts, preds)}
print(f"LogReg on Davidson: acc={results['logreg']['acc']:.4f} f1={results['logreg']['f1']:.4f}")

# ---------------- BiLSTM (h=64) ----------------
with open("vocab.pkl", "rb") as f:
    bilstm_vocab = pickle.load(f)

def encode_bilstm(tokens):
    ids = [bilstm_vocab.get(t, 1) for t in tokens[:MAX_LEN]]
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

bilstm = BiLSTMAttn(len(bilstm_vocab), hidden_dim=64)
bilstm.load_state_dict(torch.load("bilstm_h64.pt", map_location="cpu"))
bilstm.eval()

preds, gts2 = [], []
batch_size = 128
ids_all = [encode_bilstm(r["tokens"]) for r in rows]
with torch.no_grad():
    for i in range(0, len(ids_all), batch_size):
        batch = torch.tensor(ids_all[i:i+batch_size])
        logit = bilstm(batch)
        pred = (torch.sigmoid(logit) > 0.5).numpy()
        preds.extend(pred.tolist())
gts2 = gts
results["bilstm"] = {"acc": accuracy_score(gts2, preds), "f1": f1_score(gts2, preds)}
print(f"BiLSTM(h=64) on Davidson: acc={results['bilstm']['acc']:.4f} f1={results['bilstm']['f1']:.4f}")

# ---------------- Transformer ----------------
with open("transformer_vocab.pkl", "rb") as f:
    trans_vocab = pickle.load(f)

def encode_transformer(tokens):
    ids = [trans_vocab["<cls>"]] + [trans_vocab.get(t, 1) for t in tokens[:MAX_LEN - 1]]
    return ids + [0] * (MAX_LEN - len(ids))

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
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, dropout=dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.fc = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        mask = (x == 0)
        emb = self.pos(self.emb(x))
        out = self.encoder(emb, src_key_padding_mask=mask)
        return self.fc(self.dropout(out[:, 0, :])).squeeze(-1)

transformer = MiniTransformer(len(trans_vocab))
transformer.load_state_dict(torch.load("transformer_model.pt", map_location="cpu"))
transformer.eval()

preds = []
ids_all_t = [encode_transformer(r["tokens"]) for r in rows]
with torch.no_grad():
    for i in range(0, len(ids_all_t), batch_size):
        batch = torch.tensor(ids_all_t[i:i+batch_size])
        logit = transformer(batch)
        pred = (torch.sigmoid(logit) > 0.5).numpy()
        preds.extend(pred.tolist())
results["transformer"] = {"acc": accuracy_score(gts, preds), "f1": f1_score(gts, preds)}
print(f"Transformer on Davidson: acc={results['transformer']['acc']:.4f} f1={results['transformer']['f1']:.4f}")

with open("davidson_generalization_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved davidson_generalization_results.json")
