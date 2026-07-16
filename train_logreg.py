import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import pickle

with open("splits.json") as f:
    data = json.load(f)

def join(rows):
    return [" ".join(r["tokens"]) for r in rows], [r["label"] for r in rows]

Xtr_text, ytr = join(data["train"])
Xval_text, yval = join(data["val"])
Xte_text, yte = join(data["test"])

vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2)
Xtr = vec.fit_transform(Xtr_text)
Xval = vec.transform(Xval_text)
Xte = vec.transform(Xte_text)

clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
clf.fit(Xtr, ytr)

for name, X, y in [("val", Xval, yval), ("test", Xte, yte)]:
    pred = clf.predict(X)
    print(f"{name}: acc={accuracy_score(y, pred):.4f} f1={f1_score(y, pred):.4f}")

with open("logreg_model.pkl", "wb") as f:
    pickle.dump({"vec": vec, "clf": clf}, f)
print("Saved logreg_model.pkl")
