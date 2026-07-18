# The Faithfulness Paradox: When a Simple Classifier Out-Explains Neural Networks in Hate Speech Detection

[![CI](https://github.com/mihiit/hate-speechclassification/actions/workflows/ci.yml/badge.svg)](https://github.com/mihiit/hate-speechclassification/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

Code accompanying the paper *"The Faithfulness Paradox: When a Simple Classifier
Out-Explains Neural Networks in Hate Speech Detection,"* submitted to AI4S 2026.

We compare three model families — TF-IDF + Logistic Regression, a BiLSTM with
additive attention, and a small Transformer encoder, all trained **from scratch**
on [HateXplain](https://github.com/punyajoy/HateXplain) — and measure how
faithfully three explanation methods (attention, LIME, SHAP) recover
human-annotated rationale spans (IOU / AUPRC, per the
[ERASER benchmark](https://www.eraserbenchmark.com/) protocol). We also measure
real training/inference energy (CodeCarbon), run a controlled scaling experiment
across four BiLSTM widths with formal significance testing, and check
cross-dataset generalization on the independent
[Davidson et al.](https://github.com/t-davidson/hate-speech-and-offensive-language)
corpus.

## Reproducibility

**Every number, table, and figure reported in the paper is produced by the scripts
in this repository, run in the order below, against the public HateXplain and
Davidson et al. datasets, with fixed random seeds (`42`) wherever randomness is
involved.** No results in the paper are hand-computed or estimated outside this
pipeline. Unflattering results (e.g. weak cross-dataset generalization) are
reported as directly produced by these scripts, not adjusted or omitted.

## Repository structure

```
.
├── .github/workflows/ci.yml           # CI: compile/import checks + JSON validation on every push
├── prep_data.py                       # Builds train/val/test splits + rationale masks
├── train_logreg.py                    # Trains TF-IDF + Logistic Regression baseline
├── train_bilstm.py                    # Trains BiLSTM (h=64) with additive attention
├── train_transformer.py               # Trains a 2-layer Transformer encoder from scratch
├── faithfulness_eval.py               # Attention/LIME/SHAP faithfulness for LogReg + BiLSTM (Table 1 rows 1-5)
├── transformer_faithfulness_eval.py   # CLS-Attention/LIME/SHAP faithfulness for Transformer (Table 1 rows 6-8)
├── significance_test.py               # Paired t-tests, LogReg vs BiLSTM (Table 3 rows 1-2)
├── significance_test_transformer.py   # Paired t-tests, LogReg/BiLSTM vs Transformer (Table 3 rows 3-5)
├── bootstrap_ci.py                    # 10,000-resample bootstrap 95% CIs for all Table 3 comparisons (Table 3b)
├── shap_convergence.py                # KernelSHAP coalition-budget convergence check, m in {50,100,200,400} (Table showing convergence, Sec. 7.5)
├── seed_stability.py                  # Attention-explanation stability across 3 random seeds (Table showing stability, Sec. 7.6)
├── measure_energy.py                  # CodeCarbon training/inference energy measurement (Table 4)
├── scale_sweep_train.py               # Trains BiLSTM at hidden sizes {32,64,128,256} (Table 5 accuracy)
├── scale_faithfulness_eval.py         # Attention/LIME faithfulness across the 4 sizes + raw per-post scores (Table 5, Fig. 3)
├── davidson_generalization_eval.py    # Zero-shot evaluation on Davidson et al. corpus (Table 7)
├── make_architecture_fig.py           # Generates Figure 1 (3-pipeline diagram), 300 DPI
├── make_figures.py                    # Generates Figures 2, 3, 4, 5, 300 DPI
├── figures/                           # Output figures (300 DPI PNG)
├── faithfulness_results.json / faithfulness_raw_scores.json
├── transformer_faithfulness_results.json / transformer_faithfulness_raw.json
├── scale_sweep_results.json / scale_faithfulness_results.json
├── energy_results.json
├── davidson_generalization_results.json
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

# HateXplain (main dataset; not redistributed here, see its own license)
git clone --depth 1 https://github.com/punyajoy/HateXplain.git

# Davidson et al. (only needed for the generalization check, Section 7.4)
git clone --depth 1 https://github.com/t-davidson/hate-speech-and-offensive-language.git davidson
```

Scripts expect HateXplain at `./HateXplain/Data/` and Davidson at `./davidson/data/labeled_data.csv`,
both alongside the scripts.

## Running the full pipeline

Run in this order — later steps consume earlier steps' outputs:

```bash
# 1. Build splits + rationale masks
python3 prep_data.py                 # -> splits.json (train=15383 val=1922 test=1924)

# 2. Train the three model families
python3 train_logreg.py              # -> logreg_model.pkl (test acc~0.767)
python3 train_bilstm.py              # -> bilstm_model.pt, vocab.pkl (test acc~0.737, saved as h=64)
python3 train_transformer.py         # -> transformer_model.pt, transformer_vocab.pkl (test acc~0.722)

# 3. Faithfulness evaluation (Table 1, Figure 2)
python3 faithfulness_eval.py             # LogReg + BiLSTM: attention/LIME/SHAP vs. human rationales
python3 transformer_faithfulness_eval.py # Transformer: CLS-attention/LIME/SHAP vs. human rationales

# 4. Significance testing (Table 3)
python3 significance_test.py               # Paired t-tests, LogReg vs BiLSTM (LIME, SHAP)
python3 significance_test_transformer.py   # Paired t-tests, LogReg/BiLSTM vs Transformer (all Table 3 rows)
python3 bootstrap_ci.py                    # 10,000-resample bootstrap 95% CIs (Table 3b), corroborates t-tests

# 5. Energy measurement (Table 4, Figure 4)
python3 measure_energy.py            # CodeCarbon: real training/inference CO2eq for LogReg + BiLSTM

# 6. Scaling experiment + significance (Table 5/6, Figure 3)
python3 scale_sweep_train.py         # Trains BiLSTM at h in {32,64,128,256}
python3 scale_faithfulness_eval.py   # Attention/LIME faithfulness per size + raw per-post scores
#   ANOVA / pairwise t-tests (Table 6) are computed directly from
#   scale_faithfulness_results.json's raw_attn_iou / raw_lime_iou arrays with
#   scipy.stats.f_oneway / ttest_ind; see the paper's Section 7.3 for the exact snippet.

# 7. Cross-dataset generalization check (Table 7, Figure 5)
python3 davidson_generalization_eval.py   # Zero-shot LogReg/BiLSTM/Transformer on Davidson et al.
#   Requires bilstm_h64.pt from step 6 above (see comment at top of this script)

# 8. KernelSHAP coalition-budget convergence check (Sec. 7.5)
python3 shap_convergence.py          # -> shap_convergence_results.json; requires bilstm_h64.pt from step 6

# 9. Explanation stability across random seeds (Sec. 7.6)
python3 seed_stability.py            # Trains 2 additional BiLSTM seeds internally; -> seed_stability_results.json

# 10. Generate all figures (300 DPI)
python3 make_architecture_fig.py     # -> figures/fig1_architecture.png
python3 make_figures.py              # -> figures/fig2-5_*.png
```

## Key results

**Faithfulness (n=148, main comparison):** the logistic regression model is never
the least faithful of the three families on any of 9 model-explanation combinations,
and is significantly more faithful than both neural models specifically under SHAP
(p=0.0006 vs. BiLSTM, p=0.0020 vs. Transformer). LIME-based gaps are not significant.

**Energy (CodeCarbon, CPU-only):** BiLSTM training emits ~41x more CO2eq than
LogReg training; ~9.5x more per 1,000 inferences.

**Scaling (4 BiLSTM widths):** faithfulness peaks descriptively at h=64 and declines
at larger widths, but this trend is **not** statistically significant at our sample
size (one-way ANOVA, p=0.126 attention IOU, p=0.719 LIME IOU) - reported honestly
as suggestive, not confirmed.

**Cross-dataset generalization (Davidson et al., zero-shot):** all three models
underperform the 0.832 majority-class baseline (LogReg 0.621, BiLSTM 0.572,
Transformer 0.498 accuracy) - a genuine limitation, reported plainly.

**KernelSHAP convergence:** BiLSTM SHAP faithfulness stays in a narrow band
(IOU 0.53-0.57) across an 8x increase in coalition samples (50->400) - the
SHAP faithfulness gap vs. LogReg is not primarily an under-sampling artifact.

**Explanation stability across seeds:** three BiLSTM (h=64) retrains at different
seeds achieve near-identical accuracy (0.727-0.729) but their attention explanations
agree with each other far less (mean pairwise IOU 0.33-0.41) than any single run
agrees with human rationales (0.582) - a genuine, concerning instability, reported
plainly rather than downplayed.

See the paper for full discussion, the theoretical analysis of SHAP faithfulness
(Section 6), and a complete list of limitations (Section 9).

## Key figures

| | |
|---|---|
| ![Main results](figures/fig2_main_results.png) | ![Energy](figures/fig4_energy.png) |
| Faithfulness (IOU/AUPRC) across all nine model-explanation combinations | Measured training/inference CO₂eq (log scale) |
| ![Scaling](figures/fig3_scaling_trend.png) | ![Generalization](figures/fig5_generalization.png) |
| Accuracy/faithfulness vs. BiLSTM width | Zero-shot generalization to Davidson et al. |

## Data

This repository does not redistribute HateXplain or Davidson et al.'s dataset.
Both are publicly available at their own repositories under their own licenses;
see those repositories for citation and usage terms.

## Citation

If you use this code, please cite the accompanying paper (citation details to be
added upon acceptance/publication).

## License

MIT License (see `LICENSE`). This does not extend to the HateXplain or Davidson
et al. datasets, which retain their own licenses.
