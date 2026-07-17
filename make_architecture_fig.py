import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.family": "serif", "font.size": 10})

fig, ax = plt.subplots(figsize=(9, 6.2))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")

def box(x, y, w, h, text, fc="#EAF1FB", ec="#4C72B0", fontsize=9.5, bold=False):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.08",
                        linewidth=1.2, edgecolor=ec, facecolor=fc)
    ax.add_patch(b)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fontsize,
            fontweight="bold" if bold else "normal", wrap=True)

def arrow(x1, y1, x2, y2, color="#333333"):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12,
                         linewidth=1.2, color=color)
    ax.add_patch(a)

# Title labels for the two pipelines
ax.text(2.3, 9.6, "Pipeline A: TF-IDF + Logistic Regression", ha="center", fontsize=11, fontweight="bold", color="#4C72B0")
ax.text(7.7, 9.6, "Pipeline B: BiLSTM + Additive Attention", ha="center", fontsize=11, fontweight="bold", color="#DD8452")

# Shared input
box(3.7, 8.5, 2.6, 0.7, "Input post (tokens)", fc="#F2F2F2", ec="#666666", bold=True)
arrow(4.3, 8.5, 2.3, 7.9)
arrow(5.7, 8.5, 7.7, 7.9)

# Pipeline A (LogReg)
box(0.8, 7.1, 3.0, 0.8, "TF-IDF vectorizer\n(uni+bigrams, 20k feats)", fc="#EAF1FB", ec="#4C72B0")
arrow(2.3, 7.1, 2.3, 6.2)
box(0.8, 5.4, 3.0, 0.8, "Logistic Regression\nclassifier", fc="#EAF1FB", ec="#4C72B0")
arrow(2.3, 5.4, 2.3, 4.5)
box(0.8, 3.7, 3.0, 0.8, "Toxic / Normal\nprediction", fc="#F2F2F2", ec="#666666")
arrow(2.3, 3.7, 2.3, 2.8)
box(0.3, 1.5, 4.0, 1.3, "Explanations:\nLIME (perturbation)\nSHAP (exact, linear)", fc="#FFF6E9", ec="#DD8452")

# Pipeline B (BiLSTM)
box(6.2, 7.1, 3.0, 0.8, "Learned token\nembeddings (dim=100)", fc="#FDEEE4", ec="#DD8452")
arrow(7.7, 7.1, 7.7, 6.2)
box(6.2, 5.4, 3.0, 0.8, "BiLSTM\n(hidden size h per dir.)", fc="#FDEEE4", ec="#DD8452")
arrow(7.7, 5.4, 7.7, 4.5)
box(6.2, 3.7, 3.0, 0.8, "Additive attention\n+ weighted sum", fc="#FDEEE4", ec="#DD8452")
arrow(7.7, 3.7, 7.7, 2.8)
box(6.2, 1.9, 3.0, 0.7, "Toxic / Normal\nprediction", fc="#F2F2F2", ec="#666666")
arrow(7.7, 1.9, 7.7, 1.15)
box(5.7, -0.15, 4.0, 1.15, "Explanations:\nAttention weights\nLIME  |  Kernel SHAP", fc="#FFF6E9", ec="#DD8452")

ax.set_ylim(-0.4, 10)
fig.tight_layout()
fig.savefig("figures/fig1_architecture.png", dpi=300)
print("saved fig1_architecture.png")
