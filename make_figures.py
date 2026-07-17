import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
})

DPI = 300

# ---------- Figure 2: Main results bar chart ----------
with open("faithfulness_results.json") as f:
    main = json.load(f)

labels = ["LogReg\nLIME", "LogReg\nSHAP", "BiLSTM\nAttention", "BiLSTM\nLIME", "BiLSTM\nSHAP"]
keys = ["logreg_lime", "logreg_shap", "bilstm_attention", "bilstm_lime", "bilstm_shap"]
iou_means = [main[k]["mean_iou"] for k in keys]
iou_stds = [main[k]["std_iou"] for k in keys]
auprc_means = [main[k]["mean_auprc"] for k in keys]
auprc_stds = [main[k]["std_auprc"] for k in keys]

x = np.arange(len(labels))
width = 0.35
fig, ax = plt.subplots(figsize=(7.5, 4.2))
colors_iou = "#4C72B0"
colors_auprc = "#DD8452"
b1 = ax.bar(x - width/2, iou_means, width, yerr=iou_stds, capsize=3, label="Mean IOU", color=colors_iou)
b2 = ax.bar(x + width/2, auprc_means, width, yerr=auprc_stds, capsize=3, label="Mean AUPRC", color=colors_auprc)
ax.set_ylabel("Faithfulness Score")
ax.set_title("Explanation Faithfulness Against Human Rationales (n=148)")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, 1.0)
ax.legend(loc="upper right")
ax.grid(axis="y", linestyle="--", alpha=0.4)
fig.tight_layout()
fig.savefig("figures/fig2_main_results.png", dpi=DPI)
plt.close(fig)
print("saved fig2_main_results.png")

# ---------- Figure 3: Scaling trend ----------
with open("scale_sweep_results.json") as f:
    sweep = json.load(f)
with open("scale_faithfulness_results.json") as f:
    scale_faith = json.load(f)

sizes = [32, 64, 128, 256]
params = [sweep[str(s)]["params"] / 1e6 for s in sizes]
acc = [sweep[str(s)]["acc"] for s in sizes]
attn_iou = [scale_faith[str(s)]["attn_iou"] for s in sizes]
lime_iou = [scale_faith[str(s)]["lime_iou"] for s in sizes]

fig, ax1 = plt.subplots(figsize=(7.5, 4.5))
ax1.plot(params, acc, marker="o", color="#333333", label="Test Accuracy", linewidth=2)
ax1.set_xlabel("BiLSTM Parameters (millions)")
ax1.set_ylabel("Test Accuracy")
ax1.set_ylim(0.70, 0.78)
ax1.tick_params(axis="y")

ax2 = ax1.twinx()
ax2.plot(params, attn_iou, marker="s", color="#4C72B0", label="Attention IOU", linewidth=2)
ax2.plot(params, lime_iou, marker="^", color="#DD8452", label="LIME IOU", linewidth=2)
ax2.set_ylabel("Faithfulness (IOU)")
ax2.set_ylim(0.40, 0.65)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower center", ncol=1, fontsize=9)

for i, s in enumerate(sizes):
    ax1.annotate(f"h={s}", (params[i], acc[i]), textcoords="offset points", xytext=(0, 8), fontsize=8, ha="center")

ax1.set_title("Accuracy and Faithfulness vs. BiLSTM Width (Hidden Size)")
ax1.grid(axis="both", linestyle="--", alpha=0.3)
fig.tight_layout()
fig.savefig("figures/fig3_scaling_trend.png", dpi=DPI)
plt.close(fig)
print("saved fig3_scaling_trend.png")
