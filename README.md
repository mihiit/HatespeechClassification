# Faithfulness Under Efficiency: Attention, LIME, and SHAP for Hate-Speech Classification

Code accompanying the paper *"Faithfulness Under Efficiency: A Rationale-Grounded
Comparison of Attention, LIME, and SHAP for Hate-Speech Classification"*, submitted
to AI4S 2026.

We compare a lightweight TF-IDF + Logistic Regression classifier against a BiLSTM
with additive attention on the [HateXplain](https://github.com/punyajoy/HateXplain)
dataset, and measure how faithfully three explanation methods (attention, LIME,
SHAP) recover human-annotated rationale spans, using IOU and AUPRC as in the
[ERASER benchmark](https://www.eraserbenchmark.com/). All numbers in the paper
are produced directly by the scripts in this repository — none are hand-entered
or estimated.

## Reproducibility

**Every number, table, and figure reported in the paper is produced by the scripts
in this repository, run in the order below, against the public HateXplain dataset,
with fixed random seeds (`42`) wherever randomness is involved.** No results in the
paper are hand-computed or estimated outside this pipeline.

## Repository structure

```
.
├── prep_data.py            # Builds train/val/test splits + rationale masks
├── train_logreg.py          # Trains TF-IDF + Logistic Regression baseline
├── train_bilstm.py          # Trains BiLSTM with additive attention
├── faithfulness_eval.py     # Extracts attention/LIME/SHAP explanations, scores IOU/AUPRC
├── significance_test.py     # Paired t-tests for Table 3 (LogReg vs BiLSTM faithfulness)
├── faithfulness_results.json    # Saved output: mean/SD faithfulness scores (Table 1)
├── faithfulness_raw_scores.json # Saved output: per-post scores used for significance testing
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone https://github.com/mihiit/hate-speechclassification.git
cd hate-speechclassification

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Download the HateXplain dataset (not redistributed in this repo; see its own license)
git clone --depth 1 https://github.com/punyajoy/HateXplain.git
```

The scripts expect the HateXplain repository to be present alongside them at
`./HateXplain/Data/dataset.json` and `./HateXplain/Data/post_id_divisions.json`.

## Running the full pipeline

Run in this exact order — each step's output is consumed by the next:

```bash
# 1. Build binary train/val/test splits with rationale masks
python3 prep_data.py
#   -> writes splits.json
#   Expected: train=15383 val=1922 test=1924

# 2. Train the TF-IDF + Logistic Regression baseline
python3 train_logreg.py
#   -> writes logreg_model.pkl
#   Expected: test acc≈0.767, f1≈0.797

# 3. Train the BiLSTM with additive attention (trained from scratch, no pretrained embeddings)
python3 train_bilstm.py
#   -> writes bilstm_model.pt, vocab.pkl
#   Expected: test acc≈0.737, f1≈0.781

# 4. Extract attention/LIME/SHAP explanations and score faithfulness against human rationales
python3 faithfulness_eval.py
#   -> writes faithfulness_results.json, faithfulness_raw_scores.json
#   Reproduces Table 1 (mean IOU / AUPRC per model-explainer combination)

# 5. Run paired significance tests (LogReg vs BiLSTM, per explanation method)
python3 significance_test.py
#   Reproduces Table 3 (paired t-tests on IOU/AUPRC)
```

Model size and inference latency figures (Table 2) can be reproduced by loading
`logreg_model.pkl` / `bilstm_model.pt` and timing repeated forward passes; a short
snippet for this is included in the paper's Methods section and omitted here for
brevity, since it is not required to reproduce the accuracy or faithfulness numbers.

## Key results

| Model | Explanation | Mean IOU | Mean AUPRC |
|---|---|---|---|
| TF-IDF + LogReg | LIME | 0.586 | 0.737 |
| TF-IDF + LogReg | SHAP | 0.620 | 0.746 |
| BiLSTM | Attention | 0.582 | 0.723 |
| BiLSTM | LIME | 0.560 | 0.715 |
| BiLSTM | SHAP | 0.525 | 0.687 |

The smaller, faster logistic regression model is at least as faithful as the
larger BiLSTM across every explanation method tested, and significantly more
faithful under SHAP (paired t-test, p = 0.0006 for IOU, p = 0.0034 for AUPRC).
See the paper for full discussion and caveats.

## Data

This repository does not redistribute the HateXplain dataset. It is publicly
available at https://github.com/punyajoy/HateXplain under its own license; see
that repository for citation and usage terms.

## Citation

If you use this code, please cite the accompanying paper (citation details to be
added upon acceptance/publication).

## License

MIT License (see `LICENSE`). This does not extend to the HateXplain dataset,
which retains its own license.
