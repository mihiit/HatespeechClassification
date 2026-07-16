import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score
import pickle

torch.manual_seed(42)
np.random.seed(42)

with open("splits.json") as f:
    data = json.load(f)

MAX_LEN = 40
MIN_FREQ = 2

# Build vocab from training tokens
counter = Counter()
for r in data["train"]:
    counter.update(r["tokens"])
vocab = {"<pad>": 0, "<unk>": 1}
for tok, cnt in counter.items():
    if cnt >= MIN_FREQ:
        vocab[tok] = len(vocab)
print("vocab size:", len(vocab))

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

    def forward(self, x, lengths, return_attn=False):
        mask = (x != 0).float()  # (B, T)
        emb = self.emb(x)
        out, _ = self.lstm(emb)  # (B, T, 2H)
        scores = self.attn(out).squeeze(-1)  # (B, T)
        scores = scores.masked_fill(mask == 0, -1e9)
        weights = torch.softmax(scores, dim=1)  # (B, T)
        context = torch.bmm(weights.unsqueeze(1), out).squeeze(1)  # (B, 2H)
        context = self.dropout(context)
        logit = self.fc(context).squeeze(-1)
        if return_attn:
            return logit, weights
        return logit

device = "cuda" if torch.cuda.is_available() else "cpu"
model = BiLSTMAttn(len(vocab)).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.BCEWithLogitsLoss()

EPOCHS = 6
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for x, lengths, y in train_dl:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        logit = model(x, lengths)
        loss = loss_fn(logit, y)
        loss.backward()
        opt.step()
        total_loss += loss.item()

    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for x, lengths, y in val_dl:
            x = x.to(device)
            logit = model(x, lengths)
            pred = (torch.sigmoid(logit) > 0.5).cpu().numpy()
            preds.extend(pred.tolist())
            gts.extend(y.numpy().tolist())
    acc = accuracy_score(gts, preds)
    f1 = f1_score(gts, preds)
    print(f"epoch {epoch+1}: train_loss={total_loss/len(train_dl):.4f} val_acc={acc:.4f} val_f1={f1:.4f}")

# Final test eval
model.eval()
preds, gts = [], []
with torch.no_grad():
    for x, lengths, y in test_dl:
        x = x.to(device)
        logit = model(x, lengths)
        pred = (torch.sigmoid(logit) > 0.5).cpu().numpy()
        preds.extend(pred.tolist())
        gts.extend(y.numpy().tolist())
print(f"TEST: acc={accuracy_score(gts, preds):.4f} f1={f1_score(gts, preds):.4f}")

torch.save(model.state_dict(), "bilstm_model.pt")
with open("vocab.pkl", "wb") as f:
    pickle.dump(vocab, f)
print("Saved bilstm_model.pt and vocab.pkl")
