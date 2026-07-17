import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.size": 11, "axes.edgecolor": "#333333", "axes.linewidth": 0.8})
DPI = 300

# ---------- Figure 2: Main results bar chart (now 8 bars incl. Transformer) ----------
with open("faithfulness_results.json") as f:
    main = json.load(f)
with open("transformer_faithfulness_results.json") as f:
    trans = json.load(f)
all_res = {**main, **trans}

labels = ["LogReg\nLIME", "LogReg\nSHAP", "BiLSTM\nAttn", "BiLSTM\nLIME", "BiLSTM\nSHAP",
          "Transf.\nAttn", "Transf.\nLIME", "Transf.\nSHAP"]
keys = ["logreg_lime", "logreg_shap", "bilstm_attention", "bilstm_lime", "bilstm_shap",
        "transformer_attention", "transformer_lime", "transformer_shap"]
iou_means = [all_res[k]["mean_iou"] for k in keys]
iou_stds = [all_res[k]["std_iou"] for k in keys]
auprc_means = [all_res[k]["mean_auprc"] for k in keys]
auprc_stds = [all_res[k]["std_auprc"] for k in keys]

x = np.arange(len(labels))
width = 0.35
fig, ax = plt.subplots(figsize=(9.5, 4.5))
ax.bar(x - width/2, iou_means, width, yerr=iou_stds, capsize=3, label="Mean IOU", color="#4C72B0")
ax.bar(x + width/2, auprc_means, width, yerr=auprc_stds, capsize=3, label="Mean AUPRC", color="#DD8452")
ax.axvline(4.5, color="gray", linestyle=":", linewidth=1)
ax.set_ylabel("Faithfulness Score")
ax.set_title("Explanation Faithfulness Across Three Model Families (n=148)")
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
ax.set_ylim(0, 1.0)
ax.legend(loc="upper right")
ax.grid(axis="y", linestyle="--", alpha=0.4)
fig.tight_layout()
fig.savefig("figures/fig2_main_results.png", dpi=DPI)
plt.close(fig)
print("saved fig2_main_results.png")

# ---------- Figure 3: Scaling trend (unchanged) ----------
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
ax1.set_xlabel("BiLSTM Parameters (millions)"); ax1.set_ylabel("Test Accuracy")
ax1.set_ylim(0.70, 0.78)
ax2 = ax1.twinx()
ax2.plot(params, attn_iou, marker="s", color="#4C72B0", label="Attention IOU", linewidth=2)
ax2.plot(params, lime_iou, marker="^", color="#DD8452", label="LIME IOU", linewidth=2)
ax2.set_ylabel("Faithfulness (IOU)"); ax2.set_ylim(0.40, 0.65)
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

# ---------- Figure 4: Energy/CO2 comparison ----------
with open("energy_results.json") as f:
    energy = json.load(f)

fig, axes = plt.subplots(1, 2, figsize=(9.5, 4))
models = ["LogReg", "BiLSTM (h=64)"]
train_co2_mg = [energy["logreg_train"]["emissions_kg_co2"]*1e6, energy["bilstm_train"]["emissions_kg_co2"]*1e6]
infer_co2_mg = [energy["logreg_infer_1000samples"]["emissions_kg_co2"]*1e6, energy["bilstm_infer_1000samples"]["emissions_kg_co2"]*1e6]

axes[0].bar(models, train_co2_mg, color=["#4C72B0", "#DD8452"])
axes[0].set_ylabel("CO$_2$eq (mg)")
axes[0].set_title("Training Emissions")
axes[0].set_yscale("log")
for i, v in enumerate(train_co2_mg):
    axes[0].text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=9)

axes[1].bar(models, infer_co2_mg, color=["#4C72B0", "#DD8452"])
axes[1].set_ylabel("CO$_2$eq (mg) per 1000 inferences")
axes[1].set_title("Inference Emissions")
axes[1].set_yscale("log")
for i, v in enumerate(infer_co2_mg):
    axes[1].text(i, v, f"{v:.5f}", ha="center", va="bottom", fontsize=9)

fig.suptitle("Measured Energy Footprint (CodeCarbon, CPU-only)", y=1.02)
fig.tight_layout()
fig.savefig("figures/fig4_energy.png", dpi=DPI, bbox_inches="tight")
plt.close(fig)
print("saved fig4_energy.png")

# ---------- Figure 5: Cross-dataset generalization ----------
with open("davidson_generalization_results.json") as f:
    gen = json.load(f)
models = ["LogReg", "BiLSTM\n(h=64)", "Transformer"]
accs = [gen["logreg"]["acc"], gen["bilstm"]["acc"], gen["transformer"]["acc"]]
f1s = [gen["logreg"]["f1"], gen["bilstm"]["f1"], gen["transformer"]["f1"]]
majority_baseline = 20620/24783

fig, ax = plt.subplots(figsize=(7, 4.3))
x = np.arange(len(models))
width = 0.35
ax.bar(x - width/2, accs, width, label="Accuracy", color="#4C72B0")
ax.bar(x + width/2, f1s, width, label="F1", color="#DD8452")
ax.axhline(majority_baseline, color="#555555", linestyle="--", linewidth=1.3, label=f"Majority-class baseline ({majority_baseline:.3f})")
ax.set_ylabel("Score")
ax.set_title("Zero-Shot Generalization: HateXplain-Trained Models on Davidson et al.")
ax.set_xticks(x); ax.set_xticklabels(models)
ax.set_ylim(0, 1.0)
ax.legend(loc="upper right", fontsize=8)
ax.grid(axis="y", linestyle="--", alpha=0.4)
fig.tight_layout()
fig.savefig("figures/fig5_generalization.png", dpi=DPI)
plt.close(fig)
print("saved fig5_generalization.png")
