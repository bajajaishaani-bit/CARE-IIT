import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

base = {
    "aA": 10.0, "bA": 1.2, "kA": 1.2, "eA": 2.0,
    "aB": 7.0, "bB": 0.8, "kB": 0.8, "eB": 0.7,
    "K": 8.0, "delta": 0.5, "eta": 0.4
}

def Wp(c, p):
    A, B = c
    return (p["aA"]*A - 0.5*p["bA"]*A**2
            + p["aB"]*B - 0.5*p["bB"]*B**2
            - p["kA"]*A - p["kB"]*B)

def Ws(c, p):
    A, B = c
    E = p["eA"]*A + p["eB"]*B
    D = p["delta"]*E + 0.5*p["eta"]*E**2
    return Wp(c, p) - D

def solve(welfare, p):
    K = p["K"]
    result = minimize(
        lambda c: -welfare(c, p),
        x0=[K/2, K/2],
        method="SLSQP",
        bounds=[(0, K), (0, K)],
        constraints=[{
            "type": "ineq",
            "fun": lambda c: K - c[0] - c[1]
        }],
        options={"ftol": 1e-12, "maxiter": 1000}
    )
    if not result.success:
        raise RuntimeError(result.message)
    return result.x
p = base.copy()

cP = solve(Wp, p)
cS = solve(Ws, p)

print("Unpriced optimum:", np.round(cP, 4))
print("Social optimum:", np.round(cS, 4))
def environmental_burden(c, p):
    return p["eA"]*c[0] + p["eB"]*c[1]

def environmental_damage(c, p):
    E = environmental_burden(c, p)
    return p["delta"]*E + 0.5*p["eta"]*E**2

results = pd.DataFrame([
    {
        "Allocation": "Unpriced optimum",
        "Task A": cP[0],
        "Task B": cP[1],
        "Total computation": sum(cP),
        "Environmental burden": environmental_burden(cP, p),
        "Environmental damage": environmental_damage(cP, p),
        "Private surplus": Wp(cP, p),
        "Social welfare": Ws(cP, p)
    },
    {
        "Allocation": "Social optimum",
        "Task A": cS[0],
        "Task B": cS[1],
        "Total computation": sum(cS),
        "Environmental burden": environmental_burden(cS, p),
        "Environmental damage": environmental_damage(cS, p),
        "Private surplus": Wp(cS, p),
        "Social welfare": Ws(cS, p)
    }
])

print(results.round(4).to_string(index=False))
comparison = {
    "Change in Task A": cS[0] - cP[0],
    "Change in Task B": cS[1] - cP[1],
    "Change in total computation": sum(cS) - sum(cP),
    "Reduction in environmental burden": (
        environmental_burden(cP, p) -
        environmental_burden(cS, p)
    ),
    "Social welfare gain": Ws(cS, p) - Ws(cP, p)
}

for key, value in comparison.items():
    print(f"{key}: {value:.4f}")
labels = ["Task A", "Task B"]
unpriced = [cP[0], cP[1]]
social = [cS[0], cS[1]]

x = np.arange(2)
width = 0.35

plt.figure(figsize=(8, 5))
plt.bar(x - width/2, unpriced, width, label="Unpriced optimum")
plt.bar(x + width/2, social, width, label="Social optimum")

plt.xticks(x, labels)
plt.ylabel("Computational allocation")
plt.title("Computational Allocation: Unpriced vs Social Optimum")
plt.legend()
plt.tight_layout()
plt.savefig("Figure_1_Allocation_Comparison.png", dpi=300)
plt.show()
burdens = [
    environmental_burden(cP, p),
    environmental_burden(cS, p)
]

damages = [
    environmental_damage(cP, p),
    environmental_damage(cS, p)
]

x = np.arange(2)
width = 0.35

plt.figure(figsize=(8, 5))
plt.bar(x - width/2, burdens, width, label="Environmental burden")
plt.bar(x + width/2, damages, width, label="Environmental damage")

plt.xticks(x, ["Unpriced optimum", "Social optimum"])
plt.ylabel("Model units")
plt.title("Environmental Burden and Damage")
plt.legend()
plt.tight_layout()
plt.savefig("Figure_2_Environmental_Impact.png", dpi=300)
plt.show()
welfare_values = [
    Ws(cP, p),
    Ws(cS, p)
]

plt.figure(figsize=(8, 5))
plt.bar(
    ["Unpriced allocation", "Social allocation"],
    welfare_values
)

plt.ylabel("Social welfare")
plt.title("Social Welfare Under Alternative Allocations")
plt.tight_layout()
plt.savefig("Figure_3_Social_Welfare.png", dpi=300)
plt.show()
results.to_csv("Baseline_Results.csv", index=False)
robustness.to_csv("Robustness_Results.csv", index=False)

print("Results and robustness tables exported.")
