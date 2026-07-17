"""
Bootstrap confidence intervals for the main faithfulness comparisons (Table 3),
alongside the existing paired t-tests. Uses 10,000 bootstrap resamples of the
paired per-post score differences to construct a 95% CI on the mean difference,
which does not rely on a normality assumption the t-test implicitly makes.
"""
import json
import numpy as np

np.random.seed(42)
N_BOOT = 10000

with open("faithfulness_raw_scores.json") as f:
    main_raw = json.load(f)
with open("transformer_faithfulness_raw.json") as f:
    trans_raw = json.load(f)


def bootstrap_ci(a, b, idx, name_a, name_b, metric, n_boot=N_BOOT):
    av = np.array([x[idx] for x in a])
    bv = np.array([x[idx] for x in b])
    n = min(len(av), len(bv))
    av, bv = av[:n], bv[:n]
    diffs = av - bv
    obs_mean = diffs.mean()
    boot_means = np.empty(n_boot)
    idxs = np.arange(n)
    for i in range(n_boot):
        sample = np.random.choice(idxs, size=n, replace=True)
        boot_means[i] = diffs[sample].mean()
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    print(f"{name_a} vs {name_b} [{metric}]: mean_diff={obs_mean:+.4f}  95% CI=[{lo:+.4f}, {hi:+.4f}]"
          f"  {'(excludes 0)' if lo > 0 or hi < 0 else '(includes 0)'}")
    return obs_mean, lo, hi


if __name__ == "__main__":
    print("=== LogReg vs BiLSTM ===")
    bootstrap_ci(main_raw["logreg_lime"], main_raw["bilstm_lime"], 0, "LogReg-LIME", "BiLSTM-LIME", "IOU")
    bootstrap_ci(main_raw["logreg_shap"], main_raw["bilstm_shap"], 0, "LogReg-SHAP", "BiLSTM-SHAP", "IOU")

    print("\n=== LogReg vs Transformer ===")
    bootstrap_ci(main_raw["logreg_lime"], trans_raw["transformer_lime"], 0, "LogReg-LIME", "Transformer-LIME", "IOU")
    bootstrap_ci(main_raw["logreg_shap"], trans_raw["transformer_shap"], 0, "LogReg-SHAP", "Transformer-SHAP", "IOU")

    print("\n=== BiLSTM vs Transformer ===")
    bootstrap_ci(main_raw["bilstm_lime"], trans_raw["transformer_lime"], 0, "BiLSTM-LIME", "Transformer-LIME", "IOU")
    bootstrap_ci(main_raw["bilstm_shap"], trans_raw["transformer_shap"], 0, "BiLSTM-SHAP", "Transformer-SHAP", "IOU")
    bootstrap_ci(main_raw["bilstm_attention"], trans_raw["transformer_attention"], 0, "BiLSTM-Attn", "Transformer-CLS-Attn", "IOU")
