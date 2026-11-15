# Thinking at a Cost: Optimising AI Computation under Environmental Externalities

**A theoretical model of computational resource allocation across heterogeneous AI tasks**

[Author Name]  
[Institution / Department]  
*Conference: IIT Roorkee: CARE*

---

## 1. Overview

Artificial intelligence relies on computational infrastructure whose expansion is associated with environmental pressures, including electricity consumption, cooling requirements and water use. However, computational resources are finite, and different AI tasks generate different levels of economic benefit while imposing different computational and environmental costs.

This project develops a theoretical optimisation framework to examine how scarce computational resources should be allocated across heterogeneous AI tasks when environmental externalities are incorporated into decision-making.

We compare two allocation regimes:

1. **Unpriced allocation:** computational resources are allocated to maximise private surplus, without accounting for environmental damage.
2. **Socially optimal allocation:** environmental damage is internalised in the objective function.

The central question is:

> *How does incorporating environmental externalities change the composition and overall scale of optimal computational resource allocation?*

This repository contains the mathematical formulation, analytical derivations, numerical implementation, comparative statics and scenario-based robustness checks accompanying the research poster.

**Important:** The numerical parameters are illustrative and chosen to demonstrate the model's economic mechanisms. They are not empirically estimated or calibrated to observed AI infrastructure data.

---

## 2. Model Setup

Consider two heterogeneous AI tasks, indexed by $i \in \lbrace A,B \rbrace$.

Let $c_i \ge 0$ denote the computational resources allocated to task $i$.

### 2.1 Task-level benefits

Each task generates a benefit function with diminishing marginal returns:

$$
B_i(c_i) = a_i c_i - \frac{b_i}{2} c_i^2
$$

where:
- $a_i > 0$: initial marginal benefit of computation;
- $b_i > 0$: rate at which marginal benefits diminish.

The marginal and second derivatives are:

```math
B_i'(c_i) = a_i - b_i c_i
```

```math
B_i''(c_i) = -b_i < 0
```

Thus, each task exhibits diminishing marginal returns to computational allocation.

### 2.2 Direct computational costs

The direct cost of computation is linear:

```math
C(c_A, c_B) = k_A c_A + k_B c_B
```

where $k_i > 0$ represents the direct cost per unit of computation allocated to task $i$.

### 2.3 Environmental burden

Tasks differ in their environmental intensity. Let $e_i > 0$ denote the environmental burden associated with one unit of computation.

Aggregate environmental burden is:

```math
E = e_A c_A + e_B c_B
```

We model environmental damage as a convex function:

```math
D(E) = \delta E + \frac{\eta}{2} E^2
```

where $\delta > 0$ captures the linear component of environmental damage and $\eta > 0$ captures increasing marginal damage.

The marginal damage is:

```math
D'(E) = \delta + \eta E > 0
```

and:

```math
D''(E) = \eta > 0
```

The convex damage function represents an assumed increasing marginal environmental cost of aggregate computational activity.

### 2.4 Computational capacity

Total computational resources are constrained by a finite capacity:

```math
c_A + c_B \le K
```

where $K > 0$ is the available computational capacity.

Together with non-negativity, the feasible set is:

```math
\mathcal{C} = \{ (c_A, c_B) \in \mathbb{R}_+^2 : c_A + c_B \le K \}
```

---

## 3. The Unpriced Allocation Benchmark

The unpriced allocation maximises private surplus without accounting for environmental damage.

The objective is:

```math
\max_{c_A, c_B \ge 0} W^P = B_A(c_A) + B_B(c_B) - k_A c_A - k_B c_B
```

subject to:

```math
c_A + c_B \le K
```

The Lagrangian is:

```math
\mathcal{L}^P = W^P + \lambda^P (K - c_A - c_B)
```

For an interior allocation, the first-order conditions are:

```math
a_A - b_A c_A - k_A = \lambda^P
```

```math
a_B - b_B c_B - k_B = \lambda^P
```

The complementary slackness condition is:

```math
\lambda^P (K - c_A - c_B) = 0
```

The multiplier $\lambda^P$ represents the shadow value of computational capacity.

At this allocation, environmental burden is generated but is not incorporated into the decision-maker's objective.

---

## 4. Social Welfare Maximisation

The social planner accounts for both private surplus and environmental damage.

The social welfare function is:

```math
W^S = B_A(c_A) + B_B(c_B) - k_A c_A - k_B c_B - D(E)
```

The optimisation problem becomes:

```math
\max_{c_A, c_B \ge 0} W^S
```

subject to:

```math
c_A + c_B \le K
```

### 4.1 First-order conditions

The Lagrangian is:

```math
\mathcal{L}^S = W^S + \lambda (K - c_A - c_B)
```

For each task:

```math
\frac{\partial W^S}{\partial c_i} = a_i - b_i c_i - k_i - e_i(\delta + \eta E)
```

For an interior allocation, the first-order conditions are:

```math
a_i - b_i c_i - k_i - e_i(\delta + \eta E) = \lambda
```

for $i \in \lbrace A,B \rbrace$.

Equivalently:

```math
B_i'(c_i) - k_i - e_i D'(E) = \lambda
```

This condition states that the net marginal social benefit of allocating computation to each active task must equal the shadow value of computational capacity.

Unlike the unpriced benchmark, the social planner deducts the task-specific marginal environmental damage:

```math
e_i D'(E) = e_i(\delta + \eta E)
```

A task with higher environmental intensity therefore faces a larger marginal social cost, all else equal.

### 4.2 KKT conditions

For completeness, introduce multipliers $\mu_A, \mu_B \ge 0$ for the non-negativity constraints.

The KKT conditions are:

**Stationarity**

```math
a_i - b_i c_i - k_i - e_i(\delta + \eta E) - \lambda + \mu_i = 0
```

**Primal feasibility**

```math
c_A + c_B \le K, \qquad c_i \ge 0
```

**Dual feasibility**

```math
\lambda, \mu_A, \mu_B \ge 0
```

**Complementary slackness**

```math
\lambda(K - c_A - c_B) = 0
```

```math
\mu_i c_i = 0
```

For an active task, $c_i > 0$, so $\mu_i = 0$. If capacity is not fully utilised, $\lambda = 0$.

---

## 5. Concavity and Uniqueness of the Optimum

The Hessian of the social welfare function is:

```math
\nabla^2 W^S = - \begin{bmatrix} b_A & 0 \\ 0 & b_B \end{bmatrix} - \eta \begin{bmatrix} e_A^2 & e_A e_B \\ e_A e_B & e_B^2 \end{bmatrix}
```

Let:

```math
\mathbf{e} = \begin{bmatrix} e_A \\ e_B \end{bmatrix}
```

Then:

```math
\nabla^2 W^S = -\text{diag}(b_A, b_B) - \eta \mathbf{e}\mathbf{e}^\top
```

For any non-zero vector $\mathbf{x} \in \mathbb{R}^2$:

```math
\mathbf{x}^\top \nabla^2 W^S \mathbf{x} = -b_A x_A^2 - b_B x_B^2 - \eta(e_A x_A + e_B x_B)^2 < 0
```

Since $b_A, b_B, \eta > 0$, the Hessian is negative definite.

Therefore:
- Social welfare is strictly concave.
- The feasible set is convex.
- A feasible global maximum exists.
- The maximiser is unique.

The unpriced objective is also strictly concave because its Hessian is:

```math
\nabla^2 W^P = -\text{diag}(b_A, b_B)
```

Thus, both allocation problems have unique global optima.

---

## 6. Comparative Statics

We examine how changes in environmental parameters affect the socially optimal allocation.

Consider an interior solution with both tasks active and computational capacity slack, so that $\lambda = 0$.

The first-order conditions can be written in vector form:

```math
\mathbf{a} - \mathbf{k} - \text{diag}(b_A, b_B)\mathbf{c} - \mathbf{e}(\delta + \eta E) = 0
```

Define:

```math
M = \text{diag}(b_A, b_B) + \eta \mathbf{e}\mathbf{e}^\top
```

Because $M$ is positive definite, it is invertible.

### 6.1 Effect of the linear damage parameter

Differentiating the first-order conditions with respect to $\delta$:

```math
-M \frac{\partial \mathbf{c}^S}{\partial \delta} - \mathbf{e} = 0
```

Therefore:

```math
\frac{\partial \mathbf{c}^S}{\partial \delta} = -M^{-1}\mathbf{e}
```

### 6.2 Effect of the convexity parameter

Differentiating with respect to $\eta$:

```math
-M \frac{\partial \mathbf{c}^S}{\partial \eta} - E\mathbf{e} = 0
```

Therefore:

```math
\frac{\partial \mathbf{c}^S}{\partial \eta} = -EM^{-1}\mathbf{e}
```

Since $M$ is a positive-definite matrix with positive diagonal and non-negative off-diagonal elements, its inverse has positive diagonal and non-positive off-diagonal elements in this two-task setting. Consequently, both derivatives have negative components for positive environmental intensities.

**Interpretation:** Within this interior, slack-capacity regime, an increase in either environmental damage parameter reduces the optimal allocation to both tasks.

These comparative-static results are conditional on the assumed interior regime. They should not be interpreted as universal results for every possible capacity-constrained allocation.

---

## 7. Numerical Implementation

The model is solved numerically using the parameter values below.

### 7.1 Baseline parameters

| Parameter | Task A | Task B |
|---|---:|---:|
| Initial marginal benefit ($a_i$) | 10.0 | 7.0 |
| Benefit curvature ($b_i$) | 1.2 | 0.8 |
| Direct cost ($k_i$) | 1.2 | 0.8 |
| Environmental intensity ($e_i$) | 2.0 | 0.7 |

Common parameters:

| Parameter | Value |
|---|---:|
| Computational capacity ($K$) | 8.0 |
| Linear damage ($\delta$) | 0.5 |
| Damage convexity ($\eta$) | 0.4 |

These values are illustrative and are held constant in the baseline simulation.

### 7.2 Baseline allocation results

| Outcome | Unpriced allocation | Social optimum |
|---|---:|---:|
| Task A computation | 4.500 | 1.815 |
| Task B computation | 3.500 | 4.853 |
| Total computation | 8.000 | 6.668 |
| Unused capacity | 0.000 | 1.332 |
| Environmental burden | 11.450 | 7.027 |

The social optimum reallocates computation away from the more environmentally intensive Task A and towards Task B.

Environmental burden falls by approximately **38.6%** relative to the unpriced allocation.

The social planner also leaves some computational capacity unused. This occurs because the marginal social benefit of additional computation is no longer sufficient to justify its direct and environmental costs.

The result illustrates why maximum utilisation of computational capacity need not coincide with maximum social welfare.

### 7.3 Welfare comparison

Social welfare is evaluated using the same social objective at both allocations:

```math
W^S(\mathbf{c}^P) = 12.3045
```

```math
W^S(\mathbf{c}^S) = 21.2738
```

The social planner improves welfare by approximately:

```math
\Delta W^S = 8.9693
```

This comparison is distinct from comparing the two regimes using different objective functions: both allocations are evaluated using the social welfare function.

---

## 8. Robustness and Scenario-Based Sensitivity Analysis

Beyond the baseline calibration, the numerical implementation explores alternative parameter configurations.

The experiments examine:
- **Higher linear environmental damage ($\delta$):** how stronger baseline environmental damage affects optimal computation.
- **Higher damage convexity ($\eta$):** how increasing marginal environmental damage changes allocation.
- **Higher environmental intensity of Task A ($e_A$):** how greater task-specific environmental costs affect the allocation between tasks.
- **Higher computational capacity ($K$):** whether additional capacity is utilised or left idle.
- **Alternative task productivity and cost configurations:** whether the baseline reallocation pattern persists under different task characteristics.

The scenario checks demonstrate that the direction and magnitude of reallocation depend on the relative benefits, direct costs and environmental intensities of the tasks.

An important qualification is that **lower total computation is not a universal outcome**. When the relatively cleaner task is sufficiently productive, the social planner may utilise the full available capacity.

These are selected numerical scenarios, not probabilistic sensitivity estimates or empirical forecasts.

---

## 9. Main Findings

The model generates five central insights:

1. **Environmental internalisation changes allocation:** incorporating environmental damage shifts computation away from tasks with higher environmental intensity, all else equal.
2. **Composition matters:** the socially optimal allocation depends on the relative marginal benefits, direct costs and environmental intensities of the tasks.
3. **Capacity need not be fully utilised:** the social planner may rationally leave computational resources unused when their marginal social cost exceeds their marginal benefit.
4. **Environmental damage affects scale and composition:** stronger environmental damage can reduce optimal computation, subject to the capacity regime and task characteristics.
5. **Private efficiency and social efficiency differ:** an allocation that maximises private surplus need not maximise social welfare.

The broader implication is that evaluating AI infrastructure solely through computational output or capacity utilisation can overlook the environmental costs of resource allocation.

---

## 10. Limitations and Scope

This framework is a stylised theoretical model. Its purpose is to isolate the economic implications of environmental externalities in computational resource allocation.

Several limitations should be recognised:
- The model considers only two representative AI tasks.
- Task benefits are represented by quadratic functions.
- Direct computational costs are linear.
- Environmental burden is represented by reduced-form intensity coefficients.
- Environmental damage is assumed to be increasing and convex.
- The numerical parameters are illustrative rather than empirically estimated.
- The model does not separately estimate electricity consumption, water use or emissions.
- The framework is static and does not model technological change, dynamic investment or intertemporal environmental effects.

These assumptions provide analytical tractability but limit the direct empirical interpretation of the numerical results.

---

## 11. Repository Structure

The repository is organised around the analytical and computational stages of the project.

```text
Thinking-at-a-Cost/
│
├── README.md
│
├── notebooks/
│   ├── 01_Model_and_Welfare_Landscape.ipynb
│   ├── 02_Optimality_and_KKT_Validation.ipynb
│   ├── 03_Comparative_Statics.ipynb
│   ├── 04_Robustness_and_Parameter_Sensitivity.ipynb
│   └── 05_Final_Results_and_Figures.ipynb
│
├── figures/
│   ├── welfare_landscape.png
│   ├── allocation_comparison.png
│   └── allocation_tradeoff_map.png
│
└── requirements.txt
```

*The filenames and directory structure can be updated to match the final uploaded repository.*

---

## 12. Reproducibility

The numerical analysis is implemented in Python using Jupyter notebooks.

To reproduce the analysis:
1. Clone or download this repository.
2. Install the required Python packages.
3. Open the notebooks in the numbered order.
4. Run the cells sequentially to reproduce the optimisation results, comparative statics and figures.

Typical dependencies include:
- Python 3
- NumPy
- SciPy
- Pandas
- Matplotlib
- Jupyter Notebook

If a `requirements.txt` file is provided, dependencies can be installed using:

```bash
pip install -r requirements.txt
```

The notebooks contain the numerical implementation and visualisations accompanying the analytical framework.

---

## 13. References

Strubell, E., Ganesh, A., & McCallum, A. (2019). Energy and policy considerations for deep learning in NLP. *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics*, 3645–3650.

OECD. (2022). *Measuring the environmental impacts of artificial intelligence compute and applications: The AI footprint*. OECD Digital Economy Papers, No. 341.

---

## 14. Acknowledgements

I gratefully acknowledge Prof. [Supervisor's Full Name] for their valuable guidance, insightful feedback and continued support throughout this research. I also thank the Centre for Artificial Intelligence and Robotics (CARE), IIT Roorkee, for providing the opportunity to present this work.

---

**Research note:** This repository is intended as a technical companion to the conference poster. It documents the theoretical model, mathematical derivations, numerical experiments and assumptions underlying the results. The code and numerical outputs should be interpreted within the scope and limitations of the model described above.
