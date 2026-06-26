"""Headline figure: naive SFT collapses AMC accuracy on matched held-out subsets
(same problems scored for base and fine-tuned). Saved to assets/sft_regression.png."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Matched held-out AMC subsets: identical problems scored for base vs fine-tuned.
variants = ["1.7B · MATH\n(n=15)", "4B · MATH\n(n=10)", "1.7B · MATH+Numina\n(n=20)"]
base = [5/15*100, 7/10*100, 6/20*100]
sft  = [0/15*100, 1/10*100, 1/20*100]

x = list(range(len(variants)))
w = 0.38
fig, ax = plt.subplots(figsize=(8.2, 4.8))
b1 = ax.bar([i - w/2 for i in x], base, w, label="Base model", color="#4C72B0")
b2 = ax.bar([i + w/2 for i in x], sft, w, label="After naive QLoRA SFT", color="#C44E52")
for bars in (b1, b2):
    for r in bars:
        ax.annotate(f"{r.get_height():.0f}%", (r.get_x()+r.get_width()/2, r.get_height()),
                    ha="center", va="bottom", fontsize=9)

ax.set_xticks(x); ax.set_xticklabels(variants, fontsize=9)
ax.set_ylabel("AMC accuracy (pass@1, %)")
ax.set_ylim(0, 80)
ax.set_title("Naive SFT degrades reasoning across sizes, datasets, and learning rates",
             fontsize=11, weight="bold")
ax.legend(frameon=False)
fig.text(0.5, -0.02,
         "Matched held-out AMC subsets. Full-set baselines: 1.7B 32.5% · 4B 47.0%. "
         "Validation loss fell the whole time (1.63→0.61).",
         ha="center", fontsize=8, color="#555")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
fig.tight_layout()

os.makedirs("/Users/YNA/amc-tutor/assets", exist_ok=True)
out = "/Users/YNA/amc-tutor/assets/sft_regression.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
