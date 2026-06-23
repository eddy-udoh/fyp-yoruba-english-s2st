import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = r"C:\Users\Admin\OneDrive\Desktop\FYP OMEGA\mos_figures"
os.makedirs(OUT, exist_ok=True)
GREEN, GREEN2, MINT, LIGHT, INK, AMBER = "#0F6E56", "#1D9E75", "#5DCAA5", "#E1F5EE", "#0A2E1F", "#D97706"

counts = {1: 6, 2: 9, 3: 36, 4: 52, 5: 36}
MOS, N_EVAL, N_RAT = 3.74, 14, 139
per_sample = [3.71, 3.79, 3.29, 3.57, 3.50, 3.64, 3.79, 4.08, 3.64, 4.43]

# ── Figure 1: scorecard + rating distribution ───────────────────────────────
fig, (axL, axR) = plt.subplots(1, 2, figsize=(10, 4.6), gridspec_kw={"width_ratios": [1, 1.5]})
fig.suptitle("TTS Mean Opinion Score (MOS) — Yorùbá Speech Naturalness",
             fontsize=14, fontweight="bold", color=INK, y=1.0)

axL.axis("off")
axL.add_patch(FancyBboxPatch((0.05, 0.08), 0.9, 0.84, boxstyle="round,pad=0.02,rounding_size=0.04",
              transform=axL.transAxes, fc=GREEN, ec="none"))
axL.text(0.5, 0.62, f"{MOS:.2f}", transform=axL.transAxes, ha="center", va="center",
         fontsize=64, fontweight="bold", color="white")
axL.text(0.5, 0.40, "out of 5", transform=axL.transAxes, ha="center", color="#BFE8D8", fontsize=14)
axL.text(0.5, 0.26, "MEAN OPINION SCORE", transform=axL.transAxes, ha="center", color="white",
         fontsize=10, fontweight="bold")
axL.text(0.5, 0.15, f"{N_EVAL} evaluators · {N_RAT} ratings", transform=axL.transAxes,
         ha="center", color="#BFE8D8", fontsize=9)

ratings = [1, 2, 3, 4, 5]
vals = [counts[r] for r in ratings]
bars = axR.bar([str(r) for r in ratings], vals, color=[ "#E9A23B" if r<3 else GREEN2 for r in ratings], edgecolor=GREEN)
for b, v in zip(bars, vals):
    axR.text(b.get_x()+b.get_width()/2, v+0.8, str(v), ha="center", fontsize=11, color=INK, fontweight="bold")
axR.set_xlabel("Rating  (1 = Bad  →  5 = Excellent)")
axR.set_ylabel("Number of ratings")
axR.set_ylim(0, max(vals)*1.18)
axR.set_title("Rating distribution", fontsize=11, color=INK, pad=8)
for s in ("top", "right"): axR.spines[s].set_visible(False)
axR.tick_params(colors=INK)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "mos_scorecard.png"), dpi=170, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("mos_scorecard.png")

# ── Figure 2: per-sample MOS ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4.6))
labels = [f"S{i}" for i in range(1, 11)]
colors = [AMBER if v == min(per_sample) else (GREEN if v == max(per_sample) else GREEN2) for v in per_sample]
bars = ax.bar(labels, per_sample, color=colors, edgecolor=GREEN)
for b, v in zip(bars, per_sample):
    ax.text(b.get_x()+b.get_width()/2, v+0.04, f"{v:.2f}", ha="center", fontsize=9, color=INK)
ax.axhline(MOS, color=INK, linestyle="--", linewidth=1.3, label=f"Overall MOS = {MOS:.2f}")
ax.set_ylim(0, 5)
ax.set_ylabel("Mean naturalness (1–5)")
ax.set_title("Per-Sample MOS — Synthesised Yorùbá Clinical Phrases", fontsize=13, fontweight="bold", color=INK, pad=10)
ax.legend(frameon=False)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.tick_params(colors=INK)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "mos_per_sample.png"), dpi=170, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("mos_per_sample.png")
print("DONE ->", OUT)
