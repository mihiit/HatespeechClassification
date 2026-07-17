"""
Paired significance tests involving the Transformer baseline (completes Table 3
alongside significance_test.py, which covers only LogReg vs BiLSTM).
Requires faithfulness_raw_scores.json (from faithfulness_eval.py) and
transformer_faithfulness_raw.json (from transformer_faithfulness_eval.py).
"""
import json
from scipy import stats
import numpy as np

with open("faithfulness_raw_scores.json") as f:
    main_raw = json.load(f)
with open("transformer_faithfulness_raw.json") as f:
    trans_raw = json.load(f)


def paired_test(a, b, idx, name_a, name_b, metric):
    av = [x[idx] for x in a]
    bv = [x[idx] for x in b]
    n = min(len(av), len(bv))
    t, pval = stats.ttest_rel(av[:n], bv[:n])
    diff = np.mean(av[:n]) - np.mean(bv[:n])
    print(f"{name_a} vs {name_b} [{metric}]: n={n} mean_diff={diff:+.4f} t={t:.3f} p={pval:.4f}")
    return diff, pval


if __name__ == "__main__":
    print("=== LogReg vs Transformer ===")
    paired_test(main_raw["logreg_lime"], trans_raw["transformer_lime"], 0, "LogReg-LIME", "Transformer-LIME", "IOU")
    paired_test(main_raw["logreg_shap"], trans_raw["transformer_shap"], 0, "LogReg-SHAP", "Transformer-SHAP", "IOU")

    print("\n=== BiLSTM(h=64) vs Transformer ===")
    paired_test(main_raw["bilstm_lime"], trans_raw["transformer_lime"], 0, "BiLSTM-LIME", "Transformer-LIME", "IOU")
    paired_test(main_raw["bilstm_shap"], trans_raw["transformer_shap"], 0, "BiLSTM-SHAP", "Transformer-SHAP", "IOU")
    paired_test(main_raw["bilstm_attention"], trans_raw["transformer_attention"], 0, "BiLSTM-Attn", "Transformer-CLS-Attn", "IOU")
