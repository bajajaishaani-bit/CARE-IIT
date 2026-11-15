"""
Figure 1: Allocation Shift and Social Outcomes
- Stacked capacity bars (K = 8) for unpriced benchmark vs social optimum
- Semi-transparent, editorial aesthetic for A0 research poster
Outputs: fig1_allocation_shift.png (300 dpi) and .pdf
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np

# ----------------------------------------------------------------------
# PALETTE (consistent across both figures)
# ----------------------------------------------------------------------
NAVY    = "#1B2A41"
TEAL    = "#2A9D8F"
AMBER   = "#E9A23B"
CRIMSON = "#C1453B"
GREY    = "#6B7280"
LIGHT   = "#F4F5F7"
UNUSED  = "#C9CED6"

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "mathtext.fontset":  "dejavusans",
    "axes.edgecolor":    NAVY,
    "axes.labelcolor":   NAVY,
    "text.color":        NAVY,
})

# ----------------------------------------------------------------------
# VERIFIED BASELINE NUMBERS
# ----------------------------------------------------------------------
cA_P, cB_P = 4.500, 3.500
cA_S, cB_S = 1.815, 4.853
K          = 8.0
unused_P   = K - (cA_P + cB_P)
unused_S   = K - (cA_S + cB_S)

dA_pct     = -59.7
dB_pct     = +38.7
dTotal_pct = -16.6
dEnv_pct   = -38.6
dWel_abs   = +8.969

# ----------------------------------------------------------------------
# FIGURE SETUP
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(16.5, 20.5))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

x_positions = [0.0, 1.0]
bar_width   = 0.52

# ---- Stacked segments ------------------------------------------------
def draw_stack(x, cA, cB, unused, alpha=0.88):
    ax.bar(x, cA, width=bar_width, bottom=0,
           color=AMBER, alpha=alpha, edgecolor=NAVY, linewidth=1.6, zorder=3)
    ax.bar(x, cB, width=bar_width, bottom=cA,
           color=TEAL, alpha=alpha, edgecolor=NAVY, linewidth=1.6, zorder=3)
    if unused > 1e-6:
        ax.bar(x, unused, width=bar_width, bottom=cA + cB,
               color=UNUSED, alpha=0.55, edgecolor=NAVY,
               linewidth=1.6, linestyle="--", zorder=3)

draw_stack(x_positions[0], cA_P, cB_P, unused_P, alpha=0.90)
draw_stack(x_positions[1], cA_S, cB_S, unused_S, alpha=0.90)

# ---- Capacity ceiling indicator --------------------------------------
cap_y = K
ax.plot([-0.55, 1.55], [cap_y, cap_y], color=NAVY, lw=2.4, zorder=5)
ax.text(1.55, cap_y + 0.12, r"Capacity ceiling  $K = 8$",
        ha="right", va="bottom", fontsize=15, color=NAVY, style="italic")

# Subtle horizontal gridlines
for y in range(0, 9):
    ax.axhline(y, color=GREY, lw=0.5, alpha=0.10, zorder=0)

# ---- In-segment numeric labels ---------------------------------------
def label_segment(x, y_center, text, color=NAVY, size=15, weight="bold"):
    ax.text(x, y_center, text, ha="center", va="center",
            fontsize=size, color=color, fontweight=weight, zorder=6)

label_segment(x_positions[0], cA_P / 2,               f"{cA_P:.3f}")
label_segment(x_positions[0], cA_P + cB_P / 2,        f"{cB_P:.3f}", color="white")
label_segment(x_positions[1], cA_S / 2,               f"{cA_S:.3f}")
label_segment(x_positions[1], cA_S + cB_S / 2,        f"{cB_S:.3f}", color="white")
ax.text(x_positions[1], cA_S + cB_S + unused_S / 2, f"{unused_S:.3f}",
        ha="center", va="center", fontsize=14, color=GREY,
        fontstyle="italic", zorder=6)

# ---- Bar titles ------------------------------------------------------
ax.text(x_positions[0], -0.55, "Unpriced\nbenchmark",
        ha="center", va="top", fontsize=18, color=NAVY, fontweight="bold",
        linespacing=1.15)
ax.text(x_positions[1], -0.55, "Social\noptimum",
        ha="center", va="top", fontsize=18, color=NAVY, fontweight="bold",
        linespacing=1.15)

# ---- Inline legend ---------------------------------------------------
legend_x = -0.60
ax.add_patch(mpatches.Rectangle((legend_x, 7.5), 0.14, 0.42,
             color=AMBER, alpha=0.90, ec=NAVY))
ax.text(legend_x + 0.22, 7.71, "Task A  ($c_A$)",
        fontsize=14, va="center", color=NAVY)
ax.add_patch(mpatches.Rectangle((legend_x, 6.9), 0.14, 0.42,
             color=TEAL, alpha=0.90, ec=NAVY))
ax.text(legend_x + 0.22, 7.11, "Task B  ($c_B$)",
        fontsize=14, va="center", color=NAVY)
ax.add_patch(mpatches.Rectangle((legend_x, 6.3), 0.14, 0.42,
             color=UNUSED, alpha=0.55, ec=NAVY, ls="--"))
ax.text(legend_x + 0.22, 6.51, "Unused capacity",
        fontsize=14, va="center", color=NAVY)

# ----------------------------------------------------------------------
# NUMERICAL CALLOUTS
# ----------------------------------------------------------------------
callout_x = 1.28
callout_y0 = 7.4
line_h     = 0.72

callouts = [
    (r"$\Delta$ Task A",              f"{dA_pct:+.1f}%",   AMBER),
    (r"$\Delta$ Task B",              f"{dB_pct:+.1f}%",   TEAL),
    ("Total computation",             f"{dTotal_pct:+.1f}%", CRIMSON),
    ("Environmental burden",          f"{dEnv_pct:+.1f}%",  CRIMSON),
    ("Social welfare",                f"{dWel_abs:+.3f} units", NAVY),
]

for i, (lab, val, col) in enumerate(callouts):
    y = callout_y0 - i * line_h
    ax.add_patch(mpatches.FancyBboxPatch(
        (callout_x - 0.02, y - 0.32), 0.72, 0.56,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        facecolor=col, alpha=0.10, edgecolor=col, linewidth=1.2,
        zorder=2))
    ax.text(callout_x + 0.03, y, lab, fontsize=13, color=GREY,
            va="center", ha="left", zorder=6)
    ax.text(callout_x + 0.68, y, val, fontsize=15, color=col,
            fontweight="bold", va="center", ha="right", zorder=6)

# ----------------------------------------------------------------------
# AXES COSMETICS
# ----------------------------------------------------------------------
ax.set_xlim(-0.75, 2.05)
ax.set_ylim(-1.9, 9.1)
ax.set_ylabel("Computational allocation", fontsize=16, labelpad=10)
ax.set_yticks(range(0, 9))
ax.set_yticklabels([str(i) for i in range(0, 9)], fontsize=13)
ax.set_xticks([])
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
ax.spines["left"].set_color(NAVY)
ax.spines["bottom"].set_visible(False)

# Title
fig.text(0.06, 0.955,
         "Internalising Environmental Damage Reshapes Both Composition and Scale",
         fontsize=20, fontweight="bold", color=NAVY, ha="left")
fig.text(0.06, 0.925,
         r"Task A $\downarrow$ while Task B $\uparrow$; total computation and "
         r"environmental burden fall; social welfare rises.",
         fontsize=13, color=GREY, ha="left", style="italic")

fig.tight_layout(rect=[0, 0, 1, 0.91])

# ----------------------------------------------------------------------
# SAVE
# ----------------------------------------------------------------------
fig.savefig("fig1_allocation_shift.png", dpi=300, bbox_inches="tight",
            facecolor="white")
fig.savefig("fig1_allocation_shift.pdf", bbox_inches="tight",
            facecolor="white")
print("Figure 1 saved: fig1_allocation_shift.png / .pdf")
plt.show()