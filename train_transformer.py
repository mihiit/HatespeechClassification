"""
Trains a small Transformer encoder classifier from scratch (no pretrained
weights, since huggingface.co is unreachable in this environment) as a third
architecture family, addressing the reviewer request for a
transformer-family comparison point alongside the linear and recurrent models.
"""
import json
import math
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
vocab = {"<pad>": 0, "<unk>": 1, "<cls>": 2}
for tok, cnt in counter.items():
    if cnt >= MIN_FREQ:
        vocab[tok] = len(vocab)
print("vocab size:", len(vocab))

def encode(tokens):
    ids = [vocab["<cls>"]] + [vocab.get(t, 1) for t in tokens[:MAX_LEN - 1]]
    ids = ids + [0] * (MAX_LEN - len(ids))
    return ids

class HXDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, idx):
        r = self.rows[idx]
        return torch.tensor(encode(r["tokens"])), torch.tensor(r["label"], dtype=torch.float)

train_dl = DataLoader(HXDataset(data["train"]), batch_size=64, shuffle=True)
val_dl = DataLoader(HXDataset(data["val"]), batch_size=128)
test_dl = DataLoader(HXDataset(data["test"]), batch_size=128)

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
    """Single-layer Transformer encoder, trained from scratch, no pretrained weights."""
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
        self.n_heads = n_heads

    def forward(self, x, return_attn=False):
        mask = (x == 0)  # padding mask, True = ignore
        emb = self.pos(self.emb(x))
        if return_attn:
            # Manually run first layer's self-attention to extract weights for the CLS token
            attn_layer = self.encoder.layers[0].self_attn
            _, attn_weights = attn_layer(emb, emb, emb, key_padding_mask=mask, average_attn_weights=True, need_weights=True)
            cls_attn = attn_weights[:, 0, :]  # attention FROM cls token TO all tokens
        out = self.encoder(emb, src_key_padding_mask=mask)
        cls_repr = out[:, 0, :]
        logit = self.fc(self.dropout(cls_repr)).squeeze(-1)
        if return_attn:
            return logit, cls_attn
        return logit

device = "cpu"
model = MiniTransformer(len(vocab)).to(device)
n_params = sum(p.numel() for p in model.parameters())
print("Transformer params:", n_params)

opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.BCEWithLogitsLoss()

EPOCHS = 8
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for x, y in train_dl:
        opt.zero_grad()
        logit = model(x)
        loss = loss_fn(logit, y)
        loss.backward()
        opt.step()
        total_loss += loss.item()

    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for x, y in val_dl:
            logit = model(x)
            pred = (torch.sigmoid(logit) > 0.5).numpy()
            preds.extend(pred.tolist())
            gts.extend(y.numpy().tolist())
    acc = accuracy_score(gts, preds)
    f1 = f1_score(gts, preds)
    print(f"epoch {epoch+1}: train_loss={total_loss/len(train_dl):.4f} val_acc={acc:.4f} val_f1={f1:.4f}")

model.eval()
preds, gts = [], []
with torch.no_grad():
    for x, y in test_dl:
        logit = model(x)
        pred = (torch.sigmoid(logit) > 0.5).numpy()
        preds.extend(pred.tolist())
        gts.extend(y.numpy().tolist())
test_acc = accuracy_score(gts, preds)
test_f1 = f1_score(gts, preds)
print(f"TEST: acc={test_acc:.4f} f1={test_f1:.4f}")

torch.save(model.state_dict(), "transformer_model.pt")
import pickle
with open("transformer_vocab.pkl", "wb") as f:
    pickle.dump(vocab, f)
with open("transformer_result.json", "w") as f:
    json.dump({"params": n_params, "test_acc": test_acc, "test_f1": test_f1}, f)
print("Saved transformer_model.pt, transformer_vocab.pkl, transformer_result.json")
