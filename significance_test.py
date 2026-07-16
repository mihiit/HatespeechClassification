"""
Paired significance tests comparing LogReg vs BiLSTM faithfulness scores
(Table 3 in the paper). Requires faithfulness_raw_scores.json produced by
faithfulness_eval.py.
"""
import json
from scipy import stats
import numpy as np

with open("faithfulness_raw_scores.json") as f:
    raw = json.load(f)


def paired_test(a_name, b_name, metric_idx, metric_name):
    a = [x[metric_idx] for x in raw[a_name]]
    b = [x[metric_idx] for x in raw[b_name]]
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    t, pval = stats.ttest_rel(a, b)
    diff = np.mean(a) - np.mean(b)
    print(f"{a_name} vs {b_name} [{metric_name}]: n={n} mean_diff={diff:+.4f} t={t:.3f} p={pval:.4f}")
    return diff, pval


if __name__ == "__main__":
    print("=== IOU (metric_idx=0) ===")
    paired_test("logreg_lime", "bilstm_lime", 0, "IOU")
    paired_test("logreg_shap", "bilstm_shap", 0, "IOU")

    print("\n=== AUPRC (metric_idx=1) ===")
    paired_test("logreg_lime", "bilstm_lime", 1, "AUPRC")
    paired_test("logreg_shap", "bilstm_shap", 1, "AUPRC")
