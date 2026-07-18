"""
Trains BiLSTM+attention at several hidden sizes (32, 64, 128, 256) to test
whether faithfulness scales monotonically with model size, rather than
relying on a single size-point comparison against LogReg.
Reuses the same vocab as train_bilstm.py for consistency.
"""
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score

torch.manual_seed(42)
np.random.seed(42)

with open("splits.json") as f:
    data = json.load(f)

MAX_LEN = 40
MIN_FREQ = 2

counter = Counter()
for r in data["train"]:
    counter.update(r["tokens"])
vocab = {"<pad>": 0, "<unk>": 1}
for tok, cnt in counter.items():
    if cnt >= MIN_FREQ:
        vocab[tok] = len(vocab)

def encode(tokens):
    ids = [vocab.get(t, 1) for t in tokens[:MAX_LEN]]
    ids = ids + [0] * (MAX_LEN - len(ids))
    return ids

class HXDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, idx):
        r = self.rows[idx]
        ids = encode(r["tokens"])
        length = min(len(r["tokens"]), MAX_LEN)
        return torch.tensor(ids), torch.tensor(length), torch.tensor(r["label"], dtype=torch.float)

train_ds = HXDataset(data["train"])
val_ds = HXDataset(data["val"])
test_ds = HXDataset(data["test"])
train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=128)
test_dl = DataLoader(test_ds, batch_size=128)

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

device = "cuda" if torch.cuda.is_available() else "cpu"
HIDDEN_SIZES = [32, 64, 128, 256]

# Load any previously saved results so re-running (e.g. after an interrupted
# run) doesn't silently drop sizes that were already trained in an earlier
# invocation -- this used to only save whichever sizes were retrained in the
# CURRENT run, discarding earlier results for sizes it skipped.
import os
if os.path.exists("scale_sweep_results.json"):
    with open("scale_sweep_results.json") as f:
        results = {int(k): v for k, v in json.load(f).items()}
else:
    results = {}

for hidden_dim in HIDDEN_SIZES:
    if os.path.exists(f"bilstm_h{hidden_dim}.pt") and hidden_dim in results:
        print(f"hidden_dim={hidden_dim}: already trained, skipping")
        continue
    torch.manual_seed(42)
    model = BiLSTMAttn(len(vocab), hidden_dim=hidden_dim).to(device)
    n_params = sum(pp.numel() for pp in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(6):
        model.train()
        for x, lengths, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logit = model(x, lengths)
            loss = loss_fn(logit, y)
            loss.backward()
            opt.step()

    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for x, lengths, y in test_dl:
            x = x.to(device)
            logit = model(x, lengths)
            pred = (torch.sigmoid(logit) > 0.5).cpu().numpy()
            preds.extend(pred.tolist())
            gts.extend(y.numpy().tolist())
    acc = accuracy_score(gts, preds)
    f1 = f1_score(gts, preds)
    print(f"hidden_dim={hidden_dim}: params={n_params} test_acc={acc:.4f} test_f1={f1:.4f}")
    results[hidden_dim] = {"params": n_params, "acc": acc, "f1": f1}
    torch.save(model.state_dict(), f"bilstm_h{hidden_dim}.pt")
    with open("scale_sweep_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  (checkpoint + results saved incrementally)")

print("Saved scale_sweep_results.json")
