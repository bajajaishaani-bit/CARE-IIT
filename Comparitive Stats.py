import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

params = {
    "aA": 10.0, "bA": 1.2, "kA": 1.2, "eA": 2.0,
    "aB": 7.0, "bB": 0.8, "kB": 0.8, "eB": 0.7,
    "K": 8.0,
    "delta": 0.5,
    "eta": 0.4
}
def welfare_unpriced(c, p):
    cA, cB = c

    BA = p["aA"]*cA - 0.5*p["bA"]*cA**2
    BB = p["aB"]*cB - 0.5*p["bB"]*cB**2

    return BA + BB - p["kA"]*cA - p["kB"]*cB


def welfare_social(c, p):
    cA, cB = c

    E = p["eA"]*cA + p["eB"]*cB
    D = p["delta"]*E + 0.5*p["eta"]*E**2

    return welfare_unpriced(c, p) - D


def solve_model(p, social=True):

    welfare = welfare_social if social else welfare_unpriced
    K = p["K"]

    result = minimize(
        lambda c: -welfare(c, p),
        x0=[K / 2, K / 2],
        method="SLSQP",
        bounds=[(0, K), (0, K)],
        constraints=[
            {"type": "ineq", "fun": lambda c: K - c[0] - c[1]}
        ],
        options={"ftol": 1e-12, "maxiter": 1000}
    )
    if not result.success:
        raise RuntimeError(result.message)

    cA, cB = result.x

    return {
        "cA": cA,
        "cB": cB,
        "total": cA + cB,
        "unused": K - cA - cB,
        "welfare": welfare(result.x, p)
    }
baseline = solve_model(params, social=True)

pd.Series(baseline)
def comparative_statics(parameter, values):

    results = []

    for value in values:

        p = params.copy()
        p[parameter] = value

        optimum = solve_model(p, social=True)

        results.append({
            parameter: value,
            **optimum
        })

    return pd.DataFrame(results)
delta_values = np.linspace(0.1, 1.5, 30)

delta_results = comparative_statics(
    "delta", delta_values
)

plt.figure(figsize=(9, 5))
plt.plot(delta_results["delta"], delta_results["cA"], label="Task A")
plt.plot(delta_results["delta"], delta_results["cB"], label="Task B")
plt.plot(delta_results["delta"], delta_results["total"],
         label="Total computation", linestyle="--")

plt.xlabel("Baseline environmental damage (delta)")
plt.ylabel("Optimal computational allocation")
plt.title("Effect of Baseline Environmental Damage")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
eta_values = np.linspace(0.05, 1.2, 30)

eta_results = comparative_statics(
    "eta", eta_values
)

plt.figure(figsize=(9, 5))
plt.plot(eta_results["eta"], eta_results["cA"], label="Task A")
plt.plot(eta_results["eta"], eta_results["cB"], label="Task B")
plt.plot(eta_results["eta"], eta_results["total"],
         label="Total computation", linestyle="--")

plt.xlabel("Environmental damage curvature (eta)")
plt.ylabel("Optimal computational allocation")
plt.title("Effect of Increasing Environmental Damage Curvature")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
eA_values = np.linspace(0.8, 3.5, 30)

eA_results = comparative_statics(
    "eA", eA_values
)

plt.figure(figsize=(9, 5))
plt.plot(eA_results["eA"], eA_results["cA"], label="Task A")
plt.plot(eA_results["eA"], eA_results["cB"], label="Task B")
plt.plot(eA_results["eA"], eA_results["total"],
         label="Total computation", linestyle="--")

plt.xlabel("Environmental intensity of Task A (eA)")
plt.ylabel("Optimal computational allocation")
plt.title("Effect of Task-Specific Environmental Intensity")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
K_values = np.linspace(3, 14, 30)

K_results = comparative_statics(
    "K", K_values
)

plt.figure(figsize=(9, 5))
plt.plot(K_results["K"], K_results["cA"], label="Task A")
plt.plot(K_results["K"], K_results["cB"], label="Task B")
plt.plot(K_results["K"], K_results["total"],
         label="Total computation", linestyle="--")

plt.xlabel("Available computational capacity (K)")
plt.ylabel("Optimal computational allocation")
plt.title("Effect of Computational Capacity")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

summary = pd.DataFrame([
    {"Scenario": "Baseline", **baseline},
    {"Scenario": "High delta",
     **solve_model({**params, "delta": 1.5})},
    {"Scenario": "High eta",
     **solve_model({**params, "eta": 1.2})},
    {"Scenario": "High eA",
     **solve_model({**params, "eA": 3.5})},
    {"Scenario": "High capacity",
     **solve_model({**params, "K": 14.0})}
])

print(summary.round(4))
for name, df in [
    ("Baseline damage (delta)", delta_results),
    ("Damage curvature (eta)", eta_results),
    ("Task A environmental intensity (eA)", eA_results),
    ("Computational capacity (K)", K_results)
]:
    print(f"\n--- {name} ---")
    print(df.iloc[[0, len(df)//2, -1]].round(4).to_string(index=False))