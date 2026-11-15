"""
Figure 2: The Allocation Trade-off Map  (clean layout)
======================================================
Poster: "Thinking at a Cost: Optimising AI Computation under
         Environmental Externalities"

Outputs:
    fig2_tradeoff_map.png  (300 dpi)
    fig2_tradeoff_map.pdf  (vector)

Dependencies: matplotlib, numpy  (no seaborn)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch

# ----------------------------------------------------------------------
# PALETTE
# ----------------------------------------------------------------------
NAVY    = "#1B2A41"
TEAL    = "#2A9D8F"
AMBER   = "#E9A23B"
CRIMSON = "#C1453B"
GREY    = "#6B7280"
FEASIBLE= "#EFF1F4"
CONTOUR = "#B7BEC9"   # very light neutral contours

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "mathtext.fontset": "dejavusans",
    "axes.edgecolor":   NAVY,
    "axes.labelcolor":  NAVY,
    "text.color":       NAVY,
})

# ----------------------------------------------------------------------
# VERIFIED BASELINE PARAMETERS (do not alter)
# ----------------------------------------------------------------------
K         = 8.0
e_A, e_B  = 2.0, 0.7

cA_P, cB_P = 4.500,        3.500
cA_S, cB_S = 1.81512608,   4.85294106
E_P        = e_A * cA_P + e_B * cB_P     # 11.450
E_S        = e_A * cA_S + e_B * cB_S     # 7.02731...

assert abs(E_P - 11.450) < 1e-6
assert abs(E_S - 7.02731) < 1e-4

dE_pct = (E_S - E_P) / E_P * 100.0       # ≈ -38.6 %

# ----------------------------------------------------------------------
# FIGURE
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14.5, 13.0))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# ======================================================================
# 1. FEASIBLE REGION
# ======================================================================
ax.add_patch(Polygon(
    [(0, 0), (K, 0), (0, K)],
    closed=True, facecolor=FEASIBLE, edgecolor="none", zorder=1,
))

# Capacity boundary
ax.plot([0, K], [K, 0], color=NAVY, lw=2.6, zorder=4,
        solid_capstyle="round")

# Boundary label — rotated, sits just off the line on the feasible side
ax.text(4.55, 3.05,
        r"$c_A + c_B = K$",
        fontsize=14, color=NAVY, rotation=-45,
        ha="center", va="center", style="italic",
        bbox=dict(boxstyle="round,pad=0.20",
                  fc="white", ec="none", alpha=0.85),
        zorder=6)

# ======================================================================
# 2. CONTOURS  (few, evenly spaced, all labelled on the right margin)
# ======================================================================
cA_grid = np.linspace(0, K, 400)
cB_grid = np.linspace(0, K, 400)
CA, CB  = np.meshgrid(cA_grid, cB_grid)
EE      = e_A * CA + e_B * CB

# Evenly spaced base contours + the two through each optimum
base_levels = [3, 5, 9, 13, 15]
opt_levels  = [E_S, E_P]

# Base contours — light neutral
ax.contour(CA, CB, EE, levels=base_levels,
           colors=CONTOUR, linewidths=1.1, alpha=0.9, zorder=2)

# Optimum contours — coloured
ax.contour(CA, CB, EE, levels=[E_S], colors=[TEAL],
           linewidths=1.8, alpha=0.9, zorder=2)
ax.contour(CA, CB, EE, levels=[E_P], colors=[AMBER],
           linewidths=1.8, alpha=0.9, zorder=2)

# ---- Contour labels: right-margin column ---------------------------
# Compute the (c_A, c_B) where each level leaves the feasible triangle
# (i.e. intersects c_A + c_B = K) so labels attach to the visible end.
def _label_pos(level):
    # Solve e_A*a + e_B*(K - a) = level  for a in [0, K]
    a = (level - e_B * K) / (e_A - e_B)
    a = np.clip(a, 0.05 * K, 0.95 * K)
    b = K - a
    return a, b

label_levels = [(3,  r"$E=3$",  CONTOUR),
                (5,  r"$E=5$",  CONTOUR),
                (9,  r"$E=9$",  CONTOUR),
                (13, r"$E=13$", CONTOUR),
                (15, r"$E=15$", CONTOUR),
                (E_S, f"$E={E_S:.2f}$", TEAL),
                (E_P, f"$E={E_P:.2f}$", AMBER)]

for lvl, txt, col in label_levels:
    a, b = _label_pos(lvl)
    # Push label slightly INSIDE the feasible region along the contour
    n = np.array([e_A, e_B]); n = n / np.linalg.norm(n)
    px, py = a - 0.35 * n[0], b - 0.35 * n[1]
    px = np.clip(px, 0.2, K - 0.2)
    py = np.clip(py, 0.2, K - 0.2)
    ax.text(px, py, txt, fontsize=11,
            color=col, ha="center", va="center",
            fontweight="bold" if col != CONTOUR else "normal",
            bbox=dict(boxstyle="round,pad=0.16",
                      fc="white", ec="none", alpha=0.9),
            zorder=6)

# ======================================================================
# 3. DIRECTIONAL ARROW  (unpriced -> social optimum)
# ======================================================================
ax.add_patch(FancyArrowPatch(
    (cA_P, cB_P), (cA_S, cB_S),
    arrowstyle="-|>", mutation_scale=24,
    color=NAVY, lw=2.2, alpha=0.95,
    connectionstyle="arc3,rad=0.0",
    zorder=5,
))

# Arrow label — placed in the clear wedge below the arrow
ax.text(2.15, 2.95,
        "Environmental\ninternalisation",
        fontsize=12.5, color=NAVY,
        ha="center", va="center", style="italic",
        bbox=dict(boxstyle="round,pad=0.22",
                  fc="white", ec=NAVY, lw=0.8, alpha=0.9),
        zorder=7)

# ======================================================================
# 4. TWO ALLOCATIONS  (labels live in dedicated side lanes)
# ======================================================================
# --- Unpriced benchmark (amber) ------------------------------------
ax.scatter([cA_P], [cB_P], s=260, marker="o",
           facecolor=AMBER, edgecolor=NAVY, linewidth=2.0,
           alpha=0.95, zorder=8)

ax.annotate(
    f"Unpriced benchmark\n"
    f"$(c_A^P,\\,c_B^P) = ({cA_P:.3f},\\,{cB_P:.3f})$\n"
    f"$E^P = {E_P:.3f}$",
    xy=(cA_P, cB_P),
    xytext=(5.35, 2.55),                 # right lane, below
    fontsize=12, color=NAVY,
    ha="left", va="center",
    bbox=dict(boxstyle="round,pad=0.34",
              fc="white", ec=AMBER, lw=1.4, alpha=0.96),
    arrowprops=dict(arrowstyle="-", color=AMBER,
                    lw=1.1, shrinkA=2, shrinkB=8),
    zorder=9,
)

# --- Social optimum (teal) -----------------------------------------
ax.scatter([cA_S], [cB_S], s=260, marker="o",
           facecolor=TEAL, edgecolor=NAVY, linewidth=2.0,
           alpha=0.95, zorder=8)

ax.annotate(
    f"Social optimum\n"
    f"$(c_A^S,\\,c_B^S) = ({cA_S:.3f},\\,{cB_S:.3f})$\n"
    f"$E^S = {E_S:.3f}$",
    xy=(cA_S, cB_S),
    xytext=(0.55, 6.55),                 # left lane, above
    fontsize=12, color=NAVY,
    ha="left", va="center",
    bbox=dict(boxstyle="round,pad=0.34",
              fc="white", ec=TEAL, lw=1.4, alpha=0.96),
    arrowprops=dict(arrowstyle="-", color=TEAL,
                    lw=1.1, shrinkA=2, shrinkB=8),
    zorder=9,
)

# ======================================================================
# 5. KEY NUMERIC CALLOUT  (top-right, clear of everything)
# ======================================================================
ax.text(5.55, 7.35,
        f"Environmental burden\nfalls by {abs(dE_pct):.1f}%",
        fontsize=14, color=CRIMSON, fontweight="bold",
        ha="left", va="center",
        bbox=dict(boxstyle="round,pad=0.40",
                  fc="white", ec=CRIMSON, lw=1.4, alpha=0.96),
        zorder=10)

# ======================================================================
# 6. SMALL EXPLANATORY NOTE  (bottom-left, italic, unobtrusive)
# ======================================================================
ax.text(0.35, 0.55,
        "Internalising environmental damage changes the composition\n"
        "of computational allocation.",
        fontsize=11, color=GREY, ha="left", va="bottom",
        style="italic", zorder=6)

# ======================================================================
# 7. AXES
# ======================================================================
ax.set_xlim(0, K)
ax.set_ylim(0, K)
ax.set_aspect("equal")
ax.set_xticks(range(0, 9))
ax.set_yticks(range(0, 9))

ax.set_xlabel(r"Computational allocation to Task A,  $c_A$",
              fontsize=15, labelpad=10)
ax.set_ylabel(r"Computational allocation to Task B,  $c_B$",
              fontsize=15, labelpad=10)

ax.tick_params(axis="both", labelsize=12, colors=NAVY,
               direction="out", length=4, width=1.0)
ax.grid(True, color=GREY, lw=0.4, alpha=0.10)
ax.set_axisbelow(True)

for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
for s in ["left", "bottom"]:
    ax.spines[s].set_color(NAVY)
    ax.spines[s].set_linewidth(1.2)

# ======================================================================
# 8. TITLES  (figure-level, above axes)
# ======================================================================
fig.text(0.06, 0.965, "The Allocation Trade-off Map",
         fontsize=22, fontweight="bold", color=NAVY,
         ha="left", va="top")
fig.text(0.06, 0.930,
         r"Feasible allocations $c_A + c_B \leq K$; "
         r"environmental burden $E = e_A c_A + e_B c_B$, "
         r"with $e_A = 2.0$ and $e_B = 0.7$.",
         fontsize=12.5, color=GREY, ha="left", va="top",
         style="italic")

fig.tight_layout(rect=[0, 0, 1, 0.905])

# ======================================================================
# 9. SAVE
# ======================================================================
fig.savefig("fig2_tradeoff_map.png", dpi=300,
            bbox_inches="tight", facecolor="white")
fig.savefig("fig2_tradeoff_map.pdf",
            bbox_inches="tight", facecolor="white")

print("Saved: fig2_tradeoff_map.png (300 dpi) and fig2_tradeoff_map.pdf")
print(f"  E_P = {E_P:.3f}, E_S = {E_S:.5f}, ΔE = {dE_pct:.2f}%")
plt.show()