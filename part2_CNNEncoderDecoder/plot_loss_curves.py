#!/usr/bin/env python3
"""
plot_loss_curves.py — Part 2 Training & Validation Loss Curves
==============================================================
Hardcoded from training log output. Run anywhere with matplotlib installed:
    python plot_loss_curves.py
Outputs: loss_curves_part2.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Training log data (parsed from .out file) ─────────────────────────────────
epochs = list(range(1, 31))

train_mse = [
    320400.4965, 157290.6495, 157418.6302, 157433.4015, 157312.4388,
    157651.5264, 157245.3431, 157263.6866, 157338.7979, 159242.1839,
    157821.8340, 157347.1141, 157319.7983, 157247.2795, 157934.6477,
    157255.2070, 157532.5644, 157348.7368, 157298.7838, 157324.8002,
    157342.6798, 157825.4184, 149413.4983, 102596.1774,  69473.2966,
     61091.2532,  56735.9148,  52508.6998,  51260.6065,  49653.9901,
]

val_mse = [
     67216.6719,  67602.6328,  68304.9985,  67775.3027,  67897.3833,
     67742.4805,  67490.9180,  67970.2549,  67626.6333,  66402.8447,
     68097.6572,  67812.8223,  67632.1187,  67998.3975,  67827.1606,
     67874.3867,  68105.3398,  67923.3823,  68026.0259,  68066.3203,
     68134.8242,  71404.6592,  33509.5916,  49021.9170,  41638.5894,
     25460.1030,  23893.8507,  23071.9744,  22340.0476,  23361.6183,
]

train_mape = [
    19.41, 16.42, 16.37, 16.32, 16.38, 16.42, 16.42, 16.41, 16.39, 16.47,
    16.40, 16.36, 16.38, 16.39, 16.36, 16.38, 16.38, 16.36, 16.36, 16.36,
    16.35, 16.31, 15.61, 12.53, 10.26,  9.76,  9.27,  9.08,  8.96,  8.91,
]

val_mape = [
    11.97, 11.97, 11.98, 11.97, 11.99, 11.98, 11.94, 11.96, 11.95, 11.92,
    11.95, 11.95, 11.95, 11.97, 11.96, 11.96, 11.97, 11.96, 11.96, 11.96,
    11.96, 12.06,  8.50, 10.04,  8.66,  7.19,  7.05,  7.01,  6.84,  7.03,
]

lr = [
    9.97e-4, 9.89e-4, 9.76e-4, 9.57e-4, 9.33e-4, 9.05e-4, 8.72e-4, 8.35e-4,
    7.94e-4, 7.50e-4, 7.03e-4, 6.55e-4, 6.04e-4, 5.52e-4, 5.00e-4, 4.48e-4,
    3.96e-4, 3.46e-4, 2.97e-4, 2.50e-4, 2.06e-4, 1.66e-4, 1.29e-4, 9.56e-5,
    6.71e-5, 4.33e-5, 2.46e-5, 1.10e-5, 2.84e-6, 1.00e-7,
]

# Best model epochs (from log)
best_epochs = [1, 10, 23, 26, 27, 28, 29]
best_val    = [val_mse[e-1] for e in best_epochs]

# ── Colours ───────────────────────────────────────────────────────────────────
BG       = "#0f1117"
LBL      = "#e2e8f0"
GRID_COL = "#1e293b"
C_TRAIN  = "#60a5fa"   # blue  — train
C_VAL    = "#f97316"   # orange — val
C_BEST   = "#34d399"   # green — best checkpoints
C_LR     = "#a78bfa"   # purple — LR
C_MAPE_T = "#93c5fd"   # lighter blue — train MAPE
C_MAPE_V = "#fdba74"   # lighter orange — val MAPE

# ── Figure layout: 3 rows ─────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(13, 14),
                         gridspec_kw={"height_ratios": [3, 2, 1.2]})
fig.patch.set_facecolor(BG)
fig.suptitle("Part 2 — CNN Encoder-Decoder Transformer\nTraining & Validation Loss Curves",
             color=LBL, fontsize=15, fontweight="bold", y=1.01)

def style_ax(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors=LBL, labelsize=9)
    ax.xaxis.label.set_color(LBL)
    ax.yaxis.label.set_color(LBL)
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")
    ax.grid(color=GRID_COL, linewidth=0.7, linestyle="--")
    ax.set_xlim(1, 30)

# ── Row 1: MSE loss ───────────────────────────────────────────────────────────
ax1 = axes[0]
style_ax(ax1)

ax1.plot(epochs, train_mse, color=C_TRAIN, lw=2.2, label="Train MSE", zorder=3)
ax1.plot(epochs, val_mse,   color=C_VAL,   lw=2.2, label="Val MSE",   zorder=3)

# Shade the "plateau" region (epochs 2-22)
ax1.axvspan(2, 22, color="#1e293b", alpha=0.6, zorder=1)
ax1.text(12, 210000, "Plateau", color="#64748b", fontsize=9,
         ha="center", va="center", style="italic")

# Shade the "convergence" region (epochs 23-30)
ax1.axvspan(23, 30, color="#052e16", alpha=0.5, zorder=1)
ax1.text(26.5, 210000, "Convergence", color="#34d399", fontsize=9,
         ha="center", va="center", style="italic")

# Mark best checkpoints
ax1.scatter(best_epochs, best_val, color=C_BEST, s=120, zorder=5,
            marker="*", label="Best checkpoint saved")

# Annotate epoch 23 breakthrough
ax1.annotate("Breakthrough\n(epoch 23)",
             xy=(23, val_mse[22]), xytext=(20, 95000),
             color=C_BEST, fontsize=8.5, ha="center",
             arrowprops=dict(arrowstyle="->", color=C_BEST, lw=1.3),
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#052e16",
                       edgecolor=C_BEST, alpha=0.9))

# Annotate final best
ax1.annotate(f"Best val MSE\n{min(val_mse):,.0f}",
             xy=(29, min(val_mse)), xytext=(25.5, 45000),
             color=C_BEST, fontsize=8.5, ha="center",
             arrowprops=dict(arrowstyle="->", color=C_BEST, lw=1.3),
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#052e16",
                       edgecolor=C_BEST, alpha=0.9))

ax1.set_ylabel("MSE  (MWh²)", fontsize=10)
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(
    lambda x, _: f"{x/1000:.0f}k" if x >= 1000 else f"{x:.0f}"
))
ax1.legend(facecolor="#1e293b", edgecolor="#334155",
           labelcolor=LBL, fontsize=9, loc="upper right")
ax1.set_title("MSE Loss", color=LBL, fontsize=11, pad=6)

# ── Row 2: MAPE ───────────────────────────────────────────────────────────────
ax2 = axes[1]
style_ax(ax2)

ax2.plot(epochs, train_mape, color=C_MAPE_T, lw=2.2, label="Train MAPE %", zorder=3)
ax2.plot(epochs, val_mape,   color=C_MAPE_V, lw=2.2, label="Val MAPE %",   zorder=3)
ax2.axvspan(2, 22, color="#1e293b", alpha=0.6, zorder=1)
ax2.axvspan(23, 30, color="#052e16", alpha=0.5, zorder=1)

ax2.annotate(f"Best val MAPE\n{min(val_mape):.2f}%  (epoch 29)",
             xy=(29, min(val_mape)), xytext=(24, 10.5),
             color=C_BEST, fontsize=8.5, ha="center",
             arrowprops=dict(arrowstyle="->", color=C_BEST, lw=1.3),
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#052e16",
                       edgecolor=C_BEST, alpha=0.9))

ax2.set_ylabel("MAPE  (%)", fontsize=10)
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
ax2.legend(facecolor="#1e293b", edgecolor="#334155",
           labelcolor=LBL, fontsize=9, loc="upper right")
ax2.set_title("MAPE  (logged only — not used for training)",
              color=LBL, fontsize=11, pad=6)

# ── Row 3: Learning rate ──────────────────────────────────────────────────────
ax3 = axes[2]
style_ax(ax3)

ax3.plot(epochs, lr, color=C_LR, lw=2.2, zorder=3)
ax3.fill_between(epochs, lr, alpha=0.18, color=C_LR)
ax3.set_ylabel("Learning Rate", fontsize=10)
ax3.set_xlabel("Epoch", fontsize=10)
ax3.yaxis.set_major_formatter(mticker.FuncFormatter(
    lambda x, _: f"{x:.0e}" if x < 1e-3 else f"{x:.4f}"
))
ax3.set_title("CosineAnnealingLR Schedule  (1e-3 → 1e-7 over 30 epochs)",
              color=LBL, fontsize=11, pad=6)

# Shared x-axis formatting
for ax in axes:
    ax.set_xticks(range(1, 31, 2))
    ax.tick_params(axis="x", colors=LBL)

# Vertical line at epoch 23 on all panels
for ax in (ax1, ax2, ax3):
    ax.axvline(23, color=C_BEST, lw=1.1, linestyle=":", alpha=0.6, zorder=2)

plt.tight_layout(h_pad=1.8)
out = "loss_curves_part2.png"
plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG, edgecolor="none")
plt.close()
print(f"Saved: {out}")
