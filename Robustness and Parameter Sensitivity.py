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
scenarios = {
    "Baseline": {},

    "A more productive": {
        "aA": 12.0
    },

    "B more productive": {
        "aB": 9.0
    },

    "Stronger diminishing returns": {
        "bA": 1.8, "bB": 1.2
    },

    "Higher direct costs": {
        "kA": 2.0, "kB": 1.5
    },

    "Lower environmental intensity gap": {
        "eA": 1.2, "eB": 0.9
    },

    "Higher environmental intensity gap": {
        "eA": 2.8, "eB": 0.5
    }
}
rows = []

for name, changes in scenarios.items():

    p = base.copy()
    p.update(changes)

    cP = solve(Wp, p)
    cS = solve(Ws, p)

    rows.append({
        "Scenario": name,
        "Unpriced A": cP[0],
        "Unpriced B": cP[1],
        "Unpriced total": sum(cP),
        "Social A": cS[0],
        "Social B": cS[1],
        "Social total": sum(cS),
        "Social unused capacity": p["K"] - sum(cS),
        "Social welfare at unpriced optimum": Ws(cP, p),
        "Maximum social welfare": Ws(cS, p)
    })

robustness = pd.DataFrame(rows)

print(robustness.round(4).to_string(index=False))
x = np.arange(len(robustness))
width = 0.35

plt.figure(figsize=(12, 6))

plt.bar(
    x - width/2,
    robustness["Unpriced total"],
    width,
    label="Unpriced total"
)

plt.bar(
    x + width/2,
    robustness["Social total"],
    width,
    label="Social total"
)

plt.xticks(x, robustness["Scenario"], rotation=30, ha="right")
plt.ylabel("Total computational allocation")
plt.title("Robustness: Total Computation Across Parameter Scenarios")
plt.legend()
plt.tight_layout()
plt.show()
robustness["Social uses less total computation"] = (
    robustness["Social total"] < robustness["Unpriced total"] - 1e-5
)

robustness["Social welfare improves"] = (
    robustness["Maximum social welfare"]
    >= robustness["Social welfare at unpriced optimum"] - 1e-6
)

print(robustness[[
    "Scenario",
    "Social uses less total computation",
    "Social welfare improves"
]].to_string(index=False))