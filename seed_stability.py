"""
Explanation stability under random seed changes: trains the BiLSTM (h=64) at
two additional seeds (7, 123) alongside the original (42), then measures how
much the attention-based explanation for the same posts agrees across seeds
(pairwise IOU between per-post top-k attention token sets), as a stability
metric distinct from faithfulness-vs-human-rationale.
"""
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score

with open("splits.json") as f:
    data = json.load(f)
test_rows = [r for r in data["test"] if r["label"] == 1 and r["rationale_mask"] is not None]
subset = test_rows[:60]
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

def train_seed(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    train_dl = DataLoader(HXDataset(data["train"]), batch_size=64, shuffle=True)
    val_dl = DataLoader(HXDataset(data["val"]), batch_size=128)
    model = BiLSTMAttn(len(vocab), hidden_dim=64)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    for epoch in range(6):
        model.train()
        for x, lengths, y in train_dl:
            opt.zero_grad(); logit = model(x, lengths); loss = loss_fn(logit, y); loss.backward(); opt.step()
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for x, lengths, y in val_dl:
            logit = model(x, lengths)
            preds.extend((torch.sigmoid(logit) > 0.5).numpy().tolist())
            gts.extend(y.numpy().tolist())
    print(f"seed={seed}: val_acc={accuracy_score(gts, preds):.4f} val_f1={f1_score(gts, preds):.4f}")
    return model

def get_attention_topk(model, tokens, k):
    ids = encode(tokens)
    x = torch.tensor([ids])
    with torch.no_grad():
        _, w = model(x, return_attn=True)
    scores = w.numpy()[0][:len(tokens)]
    top_idx = set(np.argsort(-scores)[:k].tolist())
    return top_idx

seeds = [42, 7, 123]
models = {}
for s in seeds:
    print(f"Training seed {s}...")
    models[s] = train_seed(s)

# Pairwise explanation agreement (IOU between top-k attention tokens) across seeds
pair_ious = {(a, b): [] for i, a in enumerate(seeds) for b in seeds[i+1:]}
for r in subset:
    tokens = r["tokens"][:MAX_LEN]
    k = max(1, len(tokens) // 4)  # top 25% of tokens as the "explanation"
    topk = {s: get_attention_topk(models[s], tokens, k) for s in seeds}
    for (a, b) in pair_ious:
        inter = len(topk[a] & topk[b])
        union = len(topk[a] | topk[b])
        pair_ious[(a, b)].append(inter / union if union > 0 else 0.0)

results = {}
for (a, b), vals in pair_ious.items():
    key = f"seed{a}_vs_seed{b}"
    results[key] = {"mean_iou": float(np.mean(vals)), "std_iou": float(np.std(vals)), "n": len(vals)}
    print(f"Attention explanation agreement, seed {a} vs seed {b}: mean_IOU={np.mean(vals):.4f} (SD={np.std(vals):.4f})")

with open("seed_stability_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved seed_stability_results.json")
