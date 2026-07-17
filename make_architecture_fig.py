import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.family": "serif", "font.size": 9.5})

fig, ax = plt.subplots(figsize=(11, 6.4))
ax.set_xlim(0, 13.5)
ax.set_ylim(-0.4, 10)
ax.axis("off")

def box(x, y, w, h, text, fc="#EAF1FB", ec="#4C72B0", fontsize=9, bold=False):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.08",
                        linewidth=1.2, edgecolor=ec, facecolor=fc)
    ax.add_patch(b)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fontsize,
            fontweight="bold" if bold else "normal")

def arrow(x1, y1, x2, y2, color="#333333"):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11, linewidth=1.1, color=color)
    ax.add_patch(a)

ax.text(2.1, 9.6, "A: TF-IDF + LogReg", ha="center", fontsize=10.5, fontweight="bold", color="#4C72B0")
ax.text(6.7, 9.6, "B: BiLSTM + Attention", ha="center", fontsize=10.5, fontweight="bold", color="#DD8452")
ax.text(11.3, 9.6, "C: Transformer Encoder", ha="center", fontsize=10.5, fontweight="bold", color="#55A868")

box(3.3, 8.5, 2.6, 0.65, "Input post (tokens)", fc="#F2F2F2", ec="#666666", bold=True)
arrow(3.8, 8.5, 2.1, 7.9)
arrow(5.1, 8.5, 6.7, 7.9)
arrow(5.9, 8.5, 11.3, 7.9)

# Pipeline A
box(0.7, 7.1, 2.8, 0.75, "TF-IDF vectorizer\n(uni+bigrams, 20k)", fc="#EAF1FB", ec="#4C72B0", fontsize=8.3)
arrow(2.1, 7.1, 2.1, 6.25)
box(0.7, 5.4, 2.8, 0.75, "Logistic Regression", fc="#EAF1FB", ec="#4C72B0", fontsize=8.3)
arrow(2.1, 5.4, 2.1, 4.55)
box(0.7, 3.7, 2.8, 0.75, "Toxic / Normal", fc="#F2F2F2", ec="#666666", fontsize=8.3)
arrow(2.1, 3.7, 2.1, 2.85)
box(0.3, 1.5, 3.6, 1.25, "Explanations:\nLIME | SHAP (exact)", fc="#FFF6E9", ec="#DD8452", fontsize=8.3)

# Pipeline B
box(5.3, 7.1, 2.8, 0.75, "Learned embeddings\n(dim=100)", fc="#FDEEE4", ec="#DD8452", fontsize=8.3)
arrow(6.7, 7.1, 6.7, 6.25)
box(5.3, 5.4, 2.8, 0.75, "BiLSTM (hidden h)", fc="#FDEEE4", ec="#DD8452", fontsize=8.3)
arrow(6.7, 5.4, 6.7, 4.55)
box(5.3, 3.7, 2.8, 0.75, "Additive attention\n+ weighted sum", fc="#FDEEE4", ec="#DD8452", fontsize=8.3)
arrow(6.7, 3.7, 6.7, 2.85)
box(5.3, 1.95, 2.8, 0.65, "Toxic / Normal", fc="#F2F2F2", ec="#666666", fontsize=8.3)
arrow(6.7, 1.95, 6.7, 1.2)
box(4.9, -0.1, 3.6, 1.15, "Explanations:\nAttn | LIME | Kernel SHAP", fc="#FFF6E9", ec="#DD8452", fontsize=8.3)

# Pipeline C
box(9.9, 7.1, 2.9, 0.75, "Embeddings + positional\nencoding", fc="#EAF5EC", ec="#55A868", fontsize=8.3)
arrow(11.35, 7.1, 11.35, 6.25)
box(9.9, 5.4, 2.9, 0.75, "2-layer Transformer\nencoder (4 heads)", fc="#EAF5EC", ec="#55A868", fontsize=8.3)
arrow(11.35, 5.4, 11.35, 4.55)
box(9.9, 3.7, 2.9, 0.75, "CLS token\nrepresentation", fc="#EAF5EC", ec="#55A868", fontsize=8.3)
arrow(11.35, 3.7, 11.35, 2.85)
box(9.9, 1.95, 2.9, 0.65, "Toxic / Normal", fc="#F2F2F2", ec="#666666", fontsize=8.3)
arrow(11.35, 1.95, 11.35, 1.2)
box(9.5, -0.1, 3.7, 1.15, "Explanations:\nCLS Attn | LIME | Kernel SHAP", fc="#FFF6E9", ec="#DD8452", fontsize=8.3)

fig.tight_layout()
fig.savefig("figures/fig1_architecture.png", dpi=300)
print("saved fig1_architecture.png")
