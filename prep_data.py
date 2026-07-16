"""
Prepares HateXplain data for the faithfulness study.
- Binary task: toxic (hatespeech/offensive) vs normal
- For toxic posts with rationales, builds a majority-vote token-level rationale mask
"""
import json
from collections import Counter
import numpy as np

DATA_DIR = "HateXplain/Data"

with open(f"{DATA_DIR}/dataset.json") as f:
    dataset = json.load(f)
with open(f"{DATA_DIR}/post_id_divisions.json") as f:
    splits = json.load(f)

def majority_rationale_mask(rationales, n_tokens):
    """Average across annotators who gave a rationale, threshold at 0.5."""
    if not rationales:
        return None
    valid = [r for r in rationales if len(r) == n_tokens]
    if not valid:
        return None
    arr = np.array(valid)
    avg = arr.mean(axis=0)
    return (avg >= 0.5).astype(int).tolist()

def build_split(post_ids):
    rows = []
    for pid in post_ids:
        if pid not in dataset:
            continue
        v = dataset[pid]
        labs = [a["label"] for a in v["annotators"]]
        maj_label = Counter(labs).most_common(1)[0][0]
        binary_label = 0 if maj_label == "normal" else 1  # 1 = toxic (hatespeech or offensive)
        tokens = v["post_tokens"]
        mask = majority_rationale_mask(v["rationales"], len(tokens)) if binary_label == 1 else None
        rows.append({
            "post_id": pid,
            "tokens": tokens,
            "label": binary_label,
            "orig_label": maj_label,
            "rationale_mask": mask
        })
    return rows

train = build_split(splits["train"])
val = build_split(splits["val"])
test = build_split(splits["test"])

print(f"train={len(train)} val={len(val)} test={len(test)}")
print("train label dist:", Counter([r['label'] for r in train]))
print("test posts with rationale mask:", sum(1 for r in test if r['rationale_mask'] is not None), "/", len(test))

with open("splits.json", "w") as f:
    json.dump({"train": train, "val": val, "test": test}, f)

print("Saved splits.json")
