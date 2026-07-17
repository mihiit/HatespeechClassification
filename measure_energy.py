"""
Measures real training and inference energy/CO2 for both models using CodeCarbon.
CPU-only measurement (1 vCPU Intel Xeon @ 2.1GHz, no GPU in this environment).
"""
import json, time, pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from codecarbon import EmissionsTracker

torch.manual_seed(42)
np.random.seed(42)

with open("splits.json") as f:
    data = json.load(f)

results = {}

# ---------------- LogReg: training energy ----------------
def join(rows):
    return [" ".join(r["tokens"]) for r in rows], [r["label"] for r in rows]

Xtr_text, ytr = join(data["train"])
Xte_text, yte = join(data["test"])

tracker = EmissionsTracker(project_name="logreg_train", measure_power_secs=1, log_level="error", save_to_file=False)
tracker.start()
t0 = time.time()
N_REPEATS_LR = 10
for _ in range(N_REPEATS_LR):
    vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2)
    Xtr = vec.fit_transform(Xtr_text)
    clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
    clf.fit(Xtr, ytr)
train_time = time.time() - t0
emissions_kg = tracker.stop()
results["logreg_train"] = {"wall_time_s": train_time / N_REPEATS_LR, "emissions_kg_co2": emissions_kg / N_REPEATS_LR, "n_repeats": N_REPEATS_LR}
print(f"LogReg train (avg of {N_REPEATS_LR}): {train_time/N_REPEATS_LR:.2f}s, {emissions_kg/N_REPEATS_LR*1000:.6f} g CO2eq")

# ---------------- LogReg: inference energy (1000 samples) ----------------
Xte = vec.transform(Xte_text[:1000])
tracker = EmissionsTracker(project_name="logreg_infer", measure_power_secs=1, log_level="error", save_to_file=False)
tracker.start()
t0 = time.time()
N_INFER_REPEATS = 50
for _ in range(N_INFER_REPEATS):
    clf.predict_proba(Xte)
infer_time = time.time() - t0
emissions_kg = tracker.stop()
results["logreg_infer_1000samples"] = {"wall_time_s": infer_time / N_INFER_REPEATS, "emissions_kg_co2": emissions_kg / N_INFER_REPEATS, "n_repeats": N_INFER_REPEATS, "n_samples": 1000}
print(f"LogReg infer (avg per 1000 preds, {N_INFER_REPEATS} reps): {infer_time/N_INFER_REPEATS:.4f}s, {emissions_kg/N_INFER_REPEATS*1000:.6f} g CO2eq")

# ---------------- BiLSTM: training energy ----------------
MAX_LEN = 40
counter = Counter()
for r in data["train"]:
    counter.update(r["tokens"])
vocab = {"<pad>": 0, "<unk>": 1}
for tok, cnt in counter.items():
    if cnt >= 2:
        vocab[tok] = len(vocab)

def encode(tokens):
    ids = [vocab.get(t, 1) for t in tokens[:MAX_LEN]]
    return ids + [0] * (MAX_LEN - len(ids))

class HXDataset(Dataset):
    def __init__(self, rows): self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, idx):
        r = self.rows[idx]
        return torch.tensor(encode(r["tokens"])), torch.tensor(min(len(r["tokens"]), MAX_LEN)), torch.tensor(r["label"], dtype=torch.float)

train_dl = DataLoader(HXDataset(data["train"]), batch_size=64, shuffle=True)
test_dl = DataLoader(HXDataset(data["test"][:1000]), batch_size=128)

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

model = BiLSTMAttn(len(vocab), hidden_dim=64)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.BCEWithLogitsLoss()

tracker = EmissionsTracker(project_name="bilstm_train", measure_power_secs=1, log_level="error", save_to_file=False)
tracker.start()
t0 = time.time()
for epoch in range(6):
    model.train()
    for x, lengths, y in train_dl:
        opt.zero_grad()
        logit = model(x, lengths)
        loss = loss_fn(logit, y)
        loss.backward()
        opt.step()
train_time = time.time() - t0
emissions_kg = tracker.stop()
results["bilstm_train"] = {"wall_time_s": train_time, "emissions_kg_co2": emissions_kg}
print(f"BiLSTM train: {train_time:.2f}s, {emissions_kg*1000:.4f} g CO2eq")

# ---------------- BiLSTM: inference energy (1000 samples x 5 passes) ----------------
model.eval()
tracker = EmissionsTracker(project_name="bilstm_infer", measure_power_secs=1, log_level="error", save_to_file=False)
tracker.start()
t0 = time.time()
N_BILSTM_INFER_REPEATS = 20
with torch.no_grad():
    for _ in range(N_BILSTM_INFER_REPEATS):
        for x, lengths, y in test_dl:
            model(x, lengths)
infer_time = time.time() - t0
emissions_kg = tracker.stop()
results["bilstm_infer_1000samples"] = {"wall_time_s": infer_time / N_BILSTM_INFER_REPEATS, "emissions_kg_co2": emissions_kg / N_BILSTM_INFER_REPEATS, "n_repeats": N_BILSTM_INFER_REPEATS, "n_samples": 1000}
print(f"BiLSTM infer (avg per 1000 preds, {N_BILSTM_INFER_REPEATS} reps): {infer_time/N_BILSTM_INFER_REPEATS:.4f}s, {emissions_kg/N_BILSTM_INFER_REPEATS*1000:.6f} g CO2eq")

with open("energy_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved energy_results.json")
