"""
Paired Cohen's d effect sizes for the Table 3 comparisons, computed directly
from the per-post paired differences in faithfulness_raw_scores.json and
transformer_faithfulness_raw.json (same inputs as significance_test.py and
significance_test_transformer.py). Uses d = mean(diff) / std(diff, ddof=1),
i.e. the paired/repeated-measures Cohen's d, which is the correct effect size
for a paired t-test (as opposed to a pooled-SD approximation from summary
statistics alone).
"""
import json
import numpy as np

with open("faithfulness_raw_scores.json") as f:
    main_raw = json.load(f)
with open("transformer_faithfulness_raw.json") as f:
    trans_raw = json.load(f)


def paired_cohens_d(a, b):
    a, b = np.array(a), np.array(b)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    diff = a - b
    return diff.mean(), diff.std(ddof=1), diff.mean() / diff.std(ddof=1)


def magnitude(d):
    ad = abs(d)
    if ad < 0.2:
        return "negligible"
    if ad < 0.5:
        return "small"
    if ad < 0.8:
        return "moderate"
    return "large"


COMPARISONS = [
    ("LogReg-LIME vs. BiLSTM-LIME", main_raw["logreg_lime"], main_raw["bilstm_lime"]),
    ("LogReg-SHAP vs. BiLSTM-SHAP", main_raw["logreg_shap"], main_raw["bilstm_shap"]),
    ("LogReg-LIME vs. Transformer-LIME", main_raw["logreg_lime"], trans_raw["transformer_lime"]),
    ("LogReg-SHAP vs. Transformer-SHAP", main_raw["logreg_shap"], trans_raw["transformer_shap"]),
    ("BiLSTM-LIME vs. Transformer-LIME", main_raw["bilstm_lime"], trans_raw["transformer_lime"]),
    ("BiLSTM-SHAP vs. Transformer-SHAP", main_raw["bilstm_shap"], trans_raw["transformer_shap"]),
    ("BiLSTM-Attn vs. Transformer-CLS-Attn", main_raw["bilstm_attention"], trans_raw["transformer_attention"]),
]

if __name__ == "__main__":
    results = {}
    for label, a_raw, b_raw in COMPARISONS:
        a_iou = [x[0] for x in a_raw]
        b_iou = [x[0] for x in b_raw]
        mean_diff, sd_diff, d = paired_cohens_d(a_iou, b_iou)
        print(f"{label}: mean_diff={mean_diff:+.4f} d={d:.2f} ({magnitude(d)})")
        results[label] = {"mean_diff": mean_diff, "d": d, "magnitude": magnitude(d)}

    with open("effect_sizes_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved effect_sizes_results.json")
