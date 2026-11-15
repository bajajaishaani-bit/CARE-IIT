
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# 1. Model parameters (illustrative, not empirically calibrated)
aA, bA, kA, eA = 10.0, 1.2, 1.2, 2.0
aB, bB, kB, eB = 7.0, 0.8, 0.8, 0.7

K = 8.0
delta = 0.5
eta = 0.4


# 2. Performance benefits
def benefit_A(cA):
    return aA * cA - 0.5 * bA * cA**2

def benefit_B(cB):
    return aB * cB - 0.5 * bB * cB**2


# 3. Environmental burden and damage
def burden(cA, cB):
    return eA * cA + eB * cB

def damage(E):
    return delta * E + 0.5 * eta * E**2


# 4. The two welfare functions
def welfare_unpriced(c):
    cA, cB = c
    return (
        benefit_A(cA) + benefit_B(cB)
        - kA * cA - kB * cB
    )

def welfare_social(c):
    cA, cB = c
    return welfare_unpriced(c) - damage(burden(cA, cB))


# 5. Feasibility: nonnegative allocations within capacity
constraints = [
    {"type": "ineq", "fun": lambda c: K - c[0] - c[1]}
]
bounds = [(0, K), (0, K)]


# 6. Maximise welfare by minimising negative welfare
def find_optimum(welfare):
    result = minimize(
        fun=lambda c: -welfare(c),
        x0=[K / 2, K / 2],
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1000}
    )

    if not result.success:
        raise RuntimeError(result.message)

    return result.x, welfare(result.x)


cP, WP_star = find_optimum(welfare_unpriced)
cS, WS_star = find_optimum(welfare_social)

print("UNPRICED BENCHMARK")
print(f"cA = {cP[0]:.4f}, cB = {cP[1]:.4f}")
print(f"Total computation = {sum(cP):.4f}")
print(f"Unused capacity = {K - sum(cP):.4f}")
print(f"Unpriced welfare = {WP_star:.4f}")

print("\nSOCIAL PLANNER")
print(f"cA = {cS[0]:.4f}, cB = {cS[1]:.4f}")
print(f"Total computation = {sum(cS):.4f}")
print(f"Unused capacity = {K - sum(cS):.4f}")
print(f"Social welfare = {WS_star:.4f}")


# 7. Construct a grid over the feasible allocation region
n = 100
cA_values = np.linspace(0, K, n)
cB_values = np.linspace(0, K, n)
CA, CB = np.meshgrid(cA_values, cB_values)

feasible = CA + CB <= K

E = eA * CA + eB * CB

WP = (
    aA * CA - 0.5 * bA * CA**2
    + aB * CB - 0.5 * bB * CB**2
    - kA * CA - kB * CB
)

WS = WP - delta * E - 0.5 * eta * E**2

# Hide points outside the feasible triangle
WP = np.where(feasible, WP, np.nan)
WS = np.where(feasible, WS, np.nan)


# 8. Draw both welfare landscapes
fig = plt.figure(figsize=(15, 6))

surfaces = [
    (WP, cP, "Unpriced benchmark", "viridis"),
    (WS, cS, "Social planner", "magma")
]

for index, (Z, optimum, title, cmap) in enumerate(surfaces, start=1):
    ax = fig.add_subplot(1, 2, index, projection="3d")

    zmin = np.nanmin(Z)
    zmax = np.nanmax(Z)
    padding = 0.08 * (zmax - zmin)
    floor = zmin - padding

    ax.plot_surface(
        CA, CB, Z,
        cmap=cmap,
        edgecolor="none",
        alpha=0.92,
        antialiased=True
    )

    # Contours projected onto the base plane
    ax.contour(
        CA, CB, Z,
        zdir="z",
        offset=floor,
        levels=10,
        cmap=cmap,
        linewidths=0.8
    )

    # Mark the numerically obtained optimum
    ax.scatter(
        optimum[0], optimum[1],
        (welfare_unpriced(optimum)
         if index == 1 else welfare_social(optimum)),
        color="black",
        s=65,
        marker="o",
        depthshade=False
    )

    ax.set_title(title, pad=14)
    ax.set_xlabel("Task A computation")
    ax.set_ylabel("Task B computation")
    ax.set_zlabel("Welfare")
    ax.set_xlim(0, K)
    ax.set_ylim(0, K)
    ax.set_zlim(floor, zmax + padding)
    ax.view_init(elev=28, azim=-135)

plt.tight_layout()
plt.show()