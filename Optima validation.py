import numpy as np
import pandas as pd
from scipy.optimize import minimize

aA, bA, kA, eA = 10.0, 1.2, 1.2, 2.0
aB, bB, kB, eB = 7.0, 0.8, 0.8, 0.7

K = 8.0
delta = 0.5
eta = 0.4
def benefit_A(cA):
    return aA*cA - 0.5*bA*cA**2

def benefit_B(cB):
    return aB*cB - 0.5*bB*cB**2

def burden(cA, cB):
    return eA*cA + eB*cB

def damage(E):
    return delta*E + 0.5*eta*E**2

def welfare_unpriced(c):
    cA, cB = c
    return benefit_A(cA) + benefit_B(cB) - kA*cA - kB*cB

def welfare_social(c):
    cA, cB = c
    return welfare_unpriced(c) - damage(burden(cA, cB))

def grad_unpriced(cA, cB):
    return np.array([
        aA - bA*cA - kA,
        aB - bB*cB - kB
    ])

def grad_social(cA, cB):
    E = burden(cA, cB)
    marginal_damage = delta + eta*E

    return np.array([
        aA - bA*cA - kA - eA*marginal_damage,
        aB - bB*cB - kB - eB*marginal_damage
    ])
constraints = [
    {"type": "ineq", "fun": lambda c: K - c[0] - c[1]}
]

bounds = [(0, K), (0, K)]

def find_optimum(welfare):
    result = minimize(
        lambda c: -welfare(c),
        x0=[K/2, K/2],
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

print("Unpriced optimum:", cP)
print("Social optimum:", cS)
def validate_kkt(c, gradient, label, tol=1e-6):
    cA, cB = c
    g = gradient(cA, cB)

    slack = K - cA - cB

    # If capacity is slack, its shadow price must be zero.
    # If capacity binds, infer the shadow price from active tasks.
    active = c > tol

    if slack > tol:
        lam = 0.0
    else:
        lam = np.mean(g[active])

    # Stationarity for positive allocations
    stationarity = np.all(np.abs(g[active] - lam) < tol)

    # At zero allocations, marginal welfare cannot exceed lambda
    zero_ok = np.all(g[~active] - lam <= tol)

    feasibility = (
        cA >= -tol and
        cB >= -tol and
        slack >= -tol
    )

    dual_feasibility = lam >= -tol
    complementary_slackness = abs(lam * slack) < tol

    checks = {
        "Primal feasibility": feasibility,
        "Stationarity": stationarity,
        "Zero-allocation condition": zero_ok,
        "Dual feasibility": dual_feasibility,
        "Complementary slackness": complementary_slackness
    }

    print(f"\n{label}")
    print("Shadow price:", lam)
    print("Capacity slack:", slack)

    for condition, passed in checks.items():
        print(f"{condition}: {'PASS' if passed else 'FAIL'}")

    return all(checks.values())

valid_P = validate_kkt(cP, grad_unpriced, "Unpriced optimum")
valid_S = validate_kkt(cS, grad_social, "Social optimum")

assert valid_P and valid_S
H_unpriced = np.array([
    [-bA, 0],
    [0, -bB]
])

e = np.array([eA, eB])

H_social = (
    H_unpriced
    - eta * np.outer(e, e)
)

print("Unpriced Hessian eigenvalues:",
      np.linalg.eigvalsh(H_unpriced))

print("Social Hessian eigenvalues:",
      np.linalg.eigvalsh(H_social))

