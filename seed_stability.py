"""
Explanation stability under random seed changes: trains the BiLSTM (h=64) at
two additional seeds (7, 123) alongside the original (42), then measures how
much EACH explanation method's output for the same posts agrees across seeds
(pairwise IOU between per-post top-k token sets), as a stability metric
distinct from faithfulness-vs-human-rationale.

Originally this only covered attention. Extended to also cover LIME and SHAP,
since Section 9.4 of the paper flagged "whether LIME and SHAP explanations are
similarly unstable across seeds ... remains untested" as the single
highest-value follow-up -- this script now answers that question directly.
"""
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score
import shap
from lime.lime_text import LimeTextExplainer

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

def make_predict_proba(model):
    def predict_proba(text_list):
        batch = [encode(t.split()) for t in text_list]
        x = torch.tensor(batch)
        with torch.no_grad():
            prob = torch.sigmoid(model(x)).numpy()
        return np.stack([1 - prob, prob], axis=1)
    return predict_proba

lime_explainer = LimeTextExplainer(class_names=["normal", "toxic"], bow=False, random_state=42)

def get_lime_topk(model, tokens, k):
    text = " ".join(tokens)
    predict_fn = make_predict_proba(model)
    try:
        exp = lime_explainer.explain_instance(text, predict_fn, num_features=len(tokens), num_samples=150, labels=(1,))
        word_scores = dict(exp.as_list(label=1))
    except Exception:
        return set()
    scores = np.array([abs(word_scores.get(tok, 0.0)) for tok in tokens])
    return set(np.argsort(-scores)[:k].tolist())

def get_shap_topk(model, tokens, k):
    n = len(tokens)
    predict_fn = make_predict_proba(model)
    def f(mask_matrix):
        texts = []
        for row in mask_matrix:
            kept = [tok if keep else "" for tok, keep in zip(tokens, row)]
            texts.append(" ".join([t for t in kept if t]))
        return predict_fn(texts)[:, 1]
    background = np.zeros((1, n))
    try:
        explainer = shap.KernelExplainer(f, background)
        shap_vals = explainer.shap_values(np.ones((1, n)), nsamples=100, silent=True)
        scores = np.abs(np.array(shap_vals).flatten())
    except Exception:
        return set()
    return set(np.argsort(-scores)[:k].tolist())

seeds = [42, 7, 123]
models = {}
for s in seeds:
    print(f"Training seed {s}...")
    models[s] = train_seed(s)

pairs = [(a, b) for i, a in enumerate(seeds) for b in seeds[i+1:]]
methods = {
    "attention": lambda model, tokens, k: get_attention_topk(model, tokens, k),
    "lime": lambda model, tokens, k: get_lime_topk(model, tokens, k),
    "shap": lambda model, tokens, k: get_shap_topk(model, tokens, k),
}

results = {}
for method_name, get_topk in methods.items():
    print(f"\n=== Method: {method_name} ===")
    pair_ious = {p: [] for p in pairs}
    for i, r in enumerate(subset):
        tokens = r["tokens"][:MAX_LEN]
        k = max(1, len(tokens) // 4)  # top 25% of tokens as the "explanation"
        topk = {s: get_topk(models[s], tokens, k) for s in seeds}
        for (a, b) in pairs:
            inter = len(topk[a] & topk[b])
            union = len(topk[a] | topk[b])
            pair_ious[(a, b)].append(inter / union if union > 0 else 0.0)
        if (i + 1) % 20 == 0:
            print(f"  processed {i+1}/{len(subset)}")

    for (a, b), vals in pair_ious.items():
        key = f"seed{a}_vs_seed{b}"
        results.setdefault(method_name, {})[key] = {
            "mean_iou": float(np.mean(vals)), "std_iou": float(np.std(vals)), "n": len(vals)
        }
        print(f"{method_name} agreement, seed {a} vs seed {b}: mean_IOU={np.mean(vals):.4f} (SD={np.std(vals):.4f})")

with open("seed_stability_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved seed_stability_results.json (now covers attention, lime, and shap)")

