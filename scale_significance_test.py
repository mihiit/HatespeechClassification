"""
One-way ANOVA and pairwise t-tests across the four BiLSTM hidden sizes
(Table 6, Section 7.3). Requires raw_attn_iou / raw_lime_iou arrays in
scale_faithfulness_results.json, which scale_faithfulness_eval.py now saves
(previously it only saved the mean, so this script could not be written
against committed data until that was fixed -- see that script's docstring).

NOTE: the raw per-post arrays were not persisted by earlier runs of
scale_faithfulness_eval.py, so this script cannot reproduce Table 6 against
the scale_faithfulness_results.json currently committed in this repo. It
will work once scale_faithfulness_eval.py (fixed version) is re-run against
the trained bilstm_h{32,64,128,256}.pt checkpoints.
"""
import json
import sys
from itertools import combinations
from scipy import stats

with open("scale_faithfulness_results.json") as f:
    results = json.load(f)

SIZES = ["32", "64", "128", "256"]

missing = [s for s in SIZES if "raw_attn_iou" not in results.get(s, {})]
if missing:
    print(
        f"scale_faithfulness_results.json is missing raw_attn_iou/raw_lime_iou "
        f"for hidden size(s): {missing}. Re-run scale_faithfulness_eval.py "
        f"(this repo's current version now saves these arrays) before running "
        f"this script.",
        file=sys.stderr,
    )
    sys.exit(1)

for metric in ["raw_attn_iou", "raw_lime_iou"]:
    label = "Attention IOU" if metric == "raw_attn_iou" else "LIME IOU"
    groups = [results[s][metric] for s in SIZES]
    f_stat, p_val = stats.f_oneway(*groups)
    print(f"One-way ANOVA (4 sizes) [{label}]: F={f_stat:.2f} p={p_val:.3f}")

    for s1, s2 in combinations(SIZES, 2):
        t_stat, p_val = stats.ttest_ind(results[s1][metric], results[s2][metric])
        print(f"  h={s1} vs. h={s2} (unpaired) [{label}]: t={t_stat:.2f} p={p_val:.3f}")
