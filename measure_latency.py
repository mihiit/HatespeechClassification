"""
Single-sample CPU inference latency (ms/sample) for all three trained models,
measured identically: warm up, then time 500 single-sample forward passes
(batch size 1, matching a real-time single-post moderation request), report
the mean. Written to close a gap in the original submission, where Transformer
latency had not been measured under the same protocol as LogReg/BiLSTM and was
left blank in Table 2.
"""
import json
import pickle
import math
import time
import numpy as np
import torch
import torch.nn as nn

MAX_LEN = 40
N_REPEATS = 500
N_WARMUP = 20

with open("splits.json") as f:
    data = json.load(f)
sample_texts = [" ".join(r["tokens"]) for r in data["test"][:N_REPEATS + N_WARMUP]]

results = {}

# ---------------- LogReg ----------------
with open("logreg_model.pkl", "rb") as f:
    lr_bundle = pickle.load(f)
vec, clf = lr_bundle["vec"], lr_bundle["clf"]

for t in sample_texts[:N_WARMUP]:
    clf.predict_proba(vec.transform([t]))
times = []
for t in sample_texts[N_WARMUP:N_WARMUP + N_REPEATS]:
    start = time.perf_counter()
    clf.predict_proba(vec.transform([t]))
    times.append(time.perf_counter() - start)
results["logreg"] = {"mean_ms": float(np.mean(times) * 1000), "n": N_REPEATS}
print(f"LogReg: {results['logreg']['mean_ms']:.4f} ms/sample")

# ---------------- BiLSTM ----------------
with open("vocab.pkl", "rb") as f:
    bvocab = pickle.load(f)

class BiLSTMAttn(nn.Module):
    def __init__(self, vocab_size, emb_dim=100, hidden_dim=64):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(hidden_dim * 2, 1)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(0.3)
    def forward(self, x):
        mask = (x != 0).float()
        emb = self.emb(x)
        out, _ = self.lstm(emb)
        scores = self.attn(out).squeeze(-1)
        scores = scores.masked_fill(mask == 0, -1e9)
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), out).squeeze(1)
        return self.fc(self.dropout(context)).squeeze(-1)

bilstm = BiLSTMAttn(len(bvocab))
bilstm.load_state_dict(torch.load("bilstm_model.pt", map_location="cpu"))
bilstm.eval()

def encode_bilstm(tokens):
    ids = [bvocab.get(t, 1) for t in tokens[:MAX_LEN]]
    return ids + [0] * (MAX_LEN - len(ids))

texts_tok = [t.split() for t in sample_texts]
with torch.no_grad():
    for tok in texts_tok[:N_WARMUP]:
        bilstm(torch.tensor([encode_bilstm(tok)]))
    times = []
    for tok in texts_tok[N_WARMUP:N_WARMUP + N_REPEATS]:
        start = time.perf_counter()
        bilstm(torch.tensor([encode_bilstm(tok)]))
        times.append(time.perf_counter() - start)
results["bilstm"] = {"mean_ms": float(np.mean(times) * 1000), "n": N_REPEATS}
print(f"BiLSTM: {results['bilstm']['mean_ms']:.4f} ms/sample")

# ---------------- Transformer ----------------
with open("transformer_vocab.pkl", "rb") as f:
    tvocab = pickle.load(f)

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
    def forward(self, x):
        mask = (x == 0)
        emb = self.pos(self.emb(x))
        out = self.encoder(emb, src_key_padding_mask=mask)
        return self.fc(self.dropout(out[:, 0, :])).squeeze(-1)

transformer = MiniTransformer(len(tvocab))
transformer.load_state_dict(torch.load("transformer_model.pt", map_location="cpu"))
transformer.eval()

def encode_transformer(tokens):
    ids = [tvocab["<cls>"]] + [tvocab.get(t, 1) for t in tokens[:MAX_LEN - 1]]
    return ids + [0] * (MAX_LEN - len(ids))

with torch.no_grad():
    for tok in texts_tok[:N_WARMUP]:
        transformer(torch.tensor([encode_transformer(tok)]))
    times = []
    for tok in texts_tok[N_WARMUP:N_WARMUP + N_REPEATS]:
        start = time.perf_counter()
        transformer(torch.tensor([encode_transformer(tok)]))
        times.append(time.perf_counter() - start)
results["transformer"] = {"mean_ms": float(np.mean(times) * 1000), "n": N_REPEATS}
print(f"Transformer: {results['transformer']['mean_ms']:.4f} ms/sample")

with open("latency_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved latency_results.json")
