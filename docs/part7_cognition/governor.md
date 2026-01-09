## 26. Theory of Meta-Stability: The Universal Governor as Homeostatic Controller

{cite}`finn2017maml,franceschi2018bilevel,hospedales2021metalearning`

The Fragile Agent architecture relies on the strict satisfaction of information-theoretic and geometric constraints (The Sieve, Section 3). Manual tuning of the associated Lagrange multipliers is intractable due to the non-stationary coupling between the Representation ($G$), the Dynamics ($S$), and the Value ($V$). We formalize the training process as a dynamical system and introduce the **Universal Governor**, a meta-controller that regulates the learning dynamics. The Governor solves a bilevel optimization problem; convergence is characterized via a training Lyapunov function (Definition {prf:ref}`def-training-lyapunov-function`).

:::{admonition} Researcher Bridge: Automated Homeostasis vs. Hyperparameter Tuning
:class: tip
:name: rb-homeostasis
In standard RL, we spend weeks "grid-searching" for the right entropy coefficient ($\alpha$) or learning rate ($\eta$). The **Universal Governor** replaces this with a **homeostatic control loop**. It treats hyperparameters as a dynamical system that responds in real-time to the Sieve's diagnostic residuals. Instead of a static configuration, you have a meta-controller that "squeezes" the learning dynamics to stay on the stable manifold.
:::

This section **unifies and extends** the heuristic methods of Section 3.5 (Primal-Dual, PID, Learned Precisions) into a single neural meta-controller framework.

(sec-relationship-to-adaptive-multipliers)=
### 26.1 Relationship to Adaptive Multipliers (Section 3.5)

:::{prf:remark} Extending Section 3.5
:label: rem-extending-section

Section 3.5 introduces three methods for adaptive multiplier tuning:
- **3.5.A (Primal-Dual):** $\lambda_{t+1} = \Pi[\lambda_t + \eta_\lambda (C(\theta_t) - \epsilon)]$ — linear, memoryless
- **3.5.B (PID):** $\lambda_{t+1} = K_p e_t + K_i \sum e + K_d \Delta e$ — hand-tuned temporal filter
- **3.5.C (Learned Precisions):** $\lambda_i = \exp(-s_i)$ — diagonal covariance, no temporal structure

Each method addresses a specific failure mode but lacks generality. The **Universal Governor** subsumes all three as special cases of a learned temporal policy over the diagnostic stream.

:::
:::{prf:definition} The Meta-Control Problem
:label: def-the-meta-control-problem

Let $\theta_t \in \mathcal{M}_\Theta$ be the agent parameters at training step $t$. The meta-control problem is: find a policy $\pi_{\mathfrak{G}}$ that selects hyperparameters $\Lambda_t$ to minimize task loss while satisfying the Sieve constraints.

**Cross-references:** Section 3.5 (Adaptive Multipliers), Section 3.4 (Joint Optimization).

:::
(sec-formalization-of-learning-dynamics)=
### 26.2 Formalization of Learning Dynamics

Let $\mathcal{M}_\Theta$ be the parameter manifold of the agent. The state of the agent at training step $t$ is denoted by $\theta_t \in \mathcal{M}_\Theta$.

:::{prf:definition} Uncontrolled Dynamics
:label: def-uncontrolled-dynamics

Standard gradient descent defines a discrete flow on $\mathcal{M}_\Theta$:

$$
\theta_{t+1} = \theta_t - \eta \nabla \mathcal{L}_{\text{task}}(\theta_t),
$$
where $\eta > 0$ is the step size.

Units: $[\theta] = \text{parameter units}$, $[\eta] = \text{step}^{-1}$, $[\nabla\mathcal{L}] = \text{nat} \cdot [\theta]^{-1}$.

:::
:::{prf:definition} Constrained Dynamics
:label: def-constrained-dynamics

The Fragile Agent imposes $K$ constraints $\{C_k(\theta) \leq 0\}_{k=1}^K$ defined by the Sieve (Section 3.1). Each $C_k$ corresponds to a diagnostic node:

$$
C_k(\theta) = \text{Node}_k(\theta) - \epsilon_k,
$$
where $\epsilon_k$ is the tolerance threshold. The learning dynamics must satisfy these constraints throughout training.

:::
:::{prf:definition} Controlled Update Law
:label: def-controlled-update-law

The controlled update with adaptive multipliers is:

$$
\theta_{t+1} = \theta_t - \eta_t \left( G^{-1}(\theta_t) \nabla \mathcal{L}_{\text{task}}(\theta_t) + \sum_{k=1}^K \lambda_{k,t} \nabla C_k(\theta_t) \right),
$$
where:
- $G(\theta)$ is the parameter-space metric (cf. natural gradient, Section 2.5)
- $\eta_t$ is the adaptive learning rate
- $\lambda_{k,t} \geq 0$ are the constraint multipliers

Units: $[\lambda_k] = \text{dimensionless}$.

*Remark (Natural Gradient Connection).* The factor $G^{-1}$ applies preconditioning analogous to Fisher Information in natural gradient methods {cite}`amari1998natural`. This ensures updates are measured in information-geometric units rather than Euclidean units.

**Cross-references:** Section 2.5 (State-Space Metric), Section 3.1 (Diagnostic Nodes).

:::
(sec-the-universal-governor)=
### 26.3 The Universal Governor

We define the meta-controller that observes diagnostic residuals and outputs control signals.

:::{prf:definition} Diagnostic State Space
:label: def-diagnostic-state-space

The Governor observes the **Sieve Residuals** via the constraint evaluation map $\Psi: \mathcal{M}_\Theta \to \mathbb{R}^K$:

$$
s_t = \Psi(\theta_t) = [C_1(\theta_t), \ldots, C_K(\theta_t)]^\top.
$$
The components of $s_t$ are the normalized defect functionals corresponding to diagnostic nodes 1–41 (Section 3.1). Positive values indicate constraint violation.

Units: $[s_t] = \text{nat}$ (for entropy-based nodes) or dimensionless (for normalized defects).

:::
:::{prf:definition} The Universal Governor
:label: def-the-universal-governor

The Governor is a policy $\pi_{\mathfrak{G}}: \mathbb{R}^{K \times H} \to \mathbb{R}_+^{K+2}$ mapping the history of Sieve residuals to control inputs:

$$
\Lambda_t = \pi_{\mathfrak{G}}(s_t, s_{t-1}, \ldots, s_{t-H}; \phi),
$$
where:
- $\Lambda_t = (\eta_t, \lambda_{1,t}, \ldots, \lambda_{K,t}, T_{c,t}) \in \mathbb{R}_+^{K+2}$
- $\phi$ are the learnable parameters of the Governor
- $H$ is the history horizon (temporal context)

Units: $[\eta_t] = \text{step}^{-1}$, $[\lambda_{k,t}] = \text{dimensionless}$, $[T_{c,t}] = \text{nat}$.

*Remark (Temporal Processing).* The Governor processes a window of $H$ diagnostic snapshots. This enables detection of first and second differences $\Delta s_t$, $\Delta^2 s_t$, which are required for PID-like control (Proposition {prf:ref}`prop-subsumption-of-section`).

:::
:::{prf:proposition} Subsumption of Section 3.5
:label: prop-subsumption-of-section

The methods of Section 3.5 are recovered as special cases of $\pi_{\mathfrak{G}}$:

| Method                     | Governor Instantiation                                                       |
|----------------------------|------------------------------------------------------------------------------|
| Primal-Dual (3.5.A)        | $\pi_{\mathfrak{G}}(s_t) = \lambda_{t-1} + \eta_\lambda s_t$ (affine, $H=1$) |
| PID (3.5.B)                | Linear filter with fixed $(K_p, K_i, K_d)$, $H \geq 2$                       |
| Learned Precisions (3.5.C) | Diagonal, no temporal dependence, $H=0$                                      |

*Proof.* Direct verification. The Primal-Dual update is a memoryless affine map. The PID controller is a linear filter over error history. Learned precisions ignore temporal structure entirely. $\square$

:::
(sec-bilevel-optimization-objective)=
### 26.4 Bilevel Optimization Objective

The Governor is trained to solve a bilevel optimization problem {cite}`franceschi2018bilevel`.

:::{prf:definition} Inner Problem: Agent Optimization
:label: def-inner-problem-agent-optimization

Given fixed control $\Lambda$, the agent minimizes the regularized objective:

$$
\theta^*(\Lambda) = \arg\min_{\theta} \left[ \mathcal{L}_{\text{task}}(\theta) + \sum_{k=1}^K \lambda_k C_k(\theta) \right].
$$
:::
:::{prf:definition} Outer Problem: Governor Optimization
:label: def-outer-problem-governor-optimization

The Governor minimizes the **Training Regret** over the distribution of tasks $\mathcal{T}$:

$$
J(\phi) = \mathbb{E}_{\mathcal{T} \sim P(\mathcal{T})} \left[ \sum_{t=0}^T \left( \mathcal{L}_{\text{task}}(\theta_t) + \gamma_{\text{viol}} \sum_{k=1}^K \text{ReLU}(C_k(\theta_t))^2 \right) \right],
$$
subject to: $\theta_{t+1} = \Phi(\theta_t, \pi_{\mathfrak{G}}(\Psi(\theta_t); \phi))$.

Units: $[J] = \text{nat}$, $[\gamma_{\text{viol}}] = \text{dimensionless}$.

The outer objective penalizes cumulative task loss (convergence speed) and squared constraint violations (feasibility). The weight $\gamma_{\text{viol}}$ trades off these two objectives.

:::
:::{prf:theorem} Bilevel Structure
:label: thm-bilevel-structure

The training of the Universal Governor has bilevel structure:

$$
\min_\phi \; J(\phi) \quad \text{s.t.} \quad \theta_t = \theta_t(\Lambda_{0:t-1}), \quad \Lambda_t = \pi_{\mathfrak{G}}(s_{t:t-H}; \phi).
$$
The inner problem (agent learning) depends on the outer variables (Governor parameters) through the control sequence $\{\Lambda_t\}$.

*Remark (Gradient Computation).* Computing $\nabla_\phi J$ requires differentiating through the entire training trajectory. In practice, we use truncated backpropagation through time or evolutionary strategies.

**Cross-references:** Section 3.4 (Joint Optimization).

:::
(sec-stability-analysis-via-lyapunov-functions)=
### 26.5 Stability Analysis via Lyapunov Functions

We establish convergence guarantees using Lyapunov stability theory {cite}`khalil2002nonlinear,lasalle1960invariance`.

:::{prf:definition} Training Lyapunov Function
:label: def-training-lyapunov-function

Define the candidate Lyapunov function for the training dynamics:

$$
V_{\mathfrak{L}}(\theta) = \mathcal{L}_{\text{task}}(\theta) + \sum_{k=1}^K \frac{\mu_k}{2} \max(0, C_k(\theta))^2,
$$
where $\mu_k > 0$ are penalty weights for constraint violations.

Units: $[V_{\mathfrak{L}}] = \text{nat}$, $[\mu_k] = \text{dimensionless}$.

$V_{\mathfrak{L}}$ is the augmented Lagrangian with quadratic penalty. If $\Delta V_{\mathfrak{L}} < 0$ along the training trajectory, training converges (Theorem {prf:ref}`thm-stable-training-trajectory`).

:::
:::{prf:theorem} Stable Training Trajectory
:label: thm-stable-training-trajectory

If the Governor $\pi_{\mathfrak{G}}$ selects $\Lambda_t$ such that:

$$
\Delta V_{\mathfrak{L}} := V_{\mathfrak{L}}(\theta_{t+1}) - V_{\mathfrak{L}}(\theta_t) < 0 \quad \forall t \text{ where } \theta_t \notin \Omega,
$$
then the training process converges to the largest invariant set $\Omega$ where $\Delta V_{\mathfrak{L}} = 0$. Under standard regularity (twice-differentiable $\mathcal{L}$, LICQ), $\Omega$ consists of KKT points.

*Proof.* $V_{\mathfrak{L}}$ is bounded below by $\inf \mathcal{L}_{\text{task}}$. By hypothesis, $V_{\mathfrak{L}}(\theta_t)$ is strictly decreasing. Since $V_{\mathfrak{L}}$ is bounded below and strictly decreasing, $\lim_{t \to \infty} V_{\mathfrak{L}}(\theta_t)$ exists. By LaSalle's invariance principle {cite}`lasalle1960invariance`, trajectories converge to the largest invariant set $\Omega$ where $\Delta V_{\mathfrak{L}} = 0$. At points in $\Omega$, either (i) $\nabla \mathcal{L}_{\text{task}} = 0$ and all constraints are satisfied, or (ii) the trajectory is at a boundary where the gradient is balanced by constraint forces. $\square$

:::
:::{prf:corollary} Existence of Descent Direction
:label: cor-existence-of-descent-direction

At any non-stationary point $\theta$ where LICQ holds (the gradients $\{\nabla C_k : C_k(\theta) = 0\}$ for active constraints are linearly independent), there exist multipliers $\lambda_k \geq 0$ and step size $\eta > 0$ such that $\Delta V_{\mathfrak{L}} < 0$.

*Proof.* At a non-KKT point, either (i) the unconstrained gradient $-\nabla \mathcal{L}_{\text{task}}$ points into the feasible region, giving descent, or (ii) some constraint is active with $\nabla C_k \neq 0$. Under LICQ, we can solve for $\lambda_k$ such that the projected gradient onto the feasible tangent cone is non-zero {cite}`nocedal2006numerical`. Taking $\eta$ sufficiently small ensures descent. $\square$

**Cross-references:** Section 2.3 (Lyapunov-Constrained Control).

:::
:::{prf:corollary} The Varentropy Brake (Annealing Safety Margin)
:label: cor-varentropy-brake

The training process involves lowering $T_c$ (annealing) to converge on a Nash equilibrium. The stability of this process is governed by the Varentropy (Corollary {prf:ref}`cor-varentropy-stability`).

For the optimization trajectory to remain in the basin of attraction of the global minimum, the cooling schedule must be modulated by the Varentropy:

$$
\frac{d T_c}{dt} = - \eta \cdot \frac{T_c}{1 + \gamma V_H(\theta_t)},
$$
where $\eta, \gamma > 0$ are constants.

*Units:* $[\dot{T}_c] = \mathrm{nat}/[\text{time}]$.

**Mechanism:**
- When $V_H(\theta_t)$ is high (system is near a critical decision point/ridge), the effective cooling rate $\dot{T}_c \to 0$. The Governor "freezes" the temperature to allow the agent to resolve the bifurcation via exploration rather than collapsing into a random mode.
- This prevents **Spontaneous Symmetry Breaking** errors where rapid cooling locks the agent into a suboptimal local minimum.

*Proof:* See Appendix {ref}`E.10 <sec-appendix-e-proof-of-corollary-varentropy-brake>`.

:::

::::{admonition} Physics Isomorphism: Lyapunov Stability
:class: note
:name: pi-lyapunov

**In Physics:** A Lyapunov function $V(x)$ certifies stability if $V > 0$ away from equilibrium and $\dot{V} \leq 0$ along trajectories. For $\dot{V} \leq -\lambda V$, convergence is exponential {cite}`khalil2002nonlinear,lasalle1961stability`.

**In Implementation:** The training Lyapunov function (Definition {prf:ref}`def-training-lyapunov-function`):

$$
\mathcal{L}_{\text{Lyap}}(\theta) = \mathcal{L}_{\text{task}}(\theta) + \sum_k \frac{\mu_k}{2} \max(0, C_k(\theta))^2
$$
with $\Delta\mathcal{L}_{\text{Lyap}} < 0$ along gradient flow.

**Correspondence Table:**
| Dynamical Systems | Agent (Training) |
|:------------------|:-----------------|
| State $x$ | Parameters $\theta$ |
| Lyapunov function $V$ | Training loss $\mathcal{L}_{\text{Lyap}}$ |
| Equilibrium $x^*$ | Trained parameters $\theta^*$ |
| $\dot{V} \leq 0$ | Loss decrease |
| Exponential stability | Convergence rate $\eta$ |
| LaSalle invariance | Convergence to KKT manifold |

**Diagnostic:** StabilityCheck monitors $\Delta\mathcal{L}_{\text{Lyap}}/\mathcal{L}_{\text{Lyap}}$.
::::

::::{note} Connection to RL #23: MAML as Degenerate Meta-Stability
**The General Law (Fragile Agent):**
The **Universal Governor** solves bilevel optimization over Sieve diagnostics:

$$
V_{\mathfrak{L}} = \mathcal{L}_{\text{task}} + \sum_k \frac{\mu_k}{2} \max(0, C_k)^2
$$
where $C_k(\theta)$ are constraint residuals from the Sieve Diagnostic Nodes.

**The Degenerate Limit:**
Replace Sieve residuals with task loss only. Set history window $H=1$. Ignore constraint structure.

**The Special Case (Standard RL):**

$$
\theta^* = \theta - \alpha \nabla_\theta \mathcal{L}_{\text{inner}}, \quad \phi^* = \arg\min_\phi \mathcal{L}_{\text{outer}}(\theta^*(\phi))
$$
This recovers **MAML** {cite}`finn2017maml` and **Meta-RL** {cite}`hospedales2021metalearning`.

**What the generalization offers:**
- **Constraint-aware adaptation:** Governor modulates multipliers based on Sieve violations
- **Temporal processing:** $H$-step history enables PID-like control dynamics
- **Training Lyapunov:** Convergence guaranteed by $\Delta V_{\mathfrak{L}} < 0$ descent
- **Diagnostic abstraction:** Transfer via geometric invariants, not task-specific features
::::

(sec-transfer-via-geometric-invariance)=
### 26.6 Transfer via Geometric Invariance

:::{prf:proposition} Structure of Diagnostic Inputs
:label: prop-structure-of-diagnostic-inputs

The input to the Governor, $s_t = \Psi(\theta_t)$, consists of quantities that depend only on the learned representations, not on the raw data $\mathcal{D}$:
- Entropies: $H(K)$, $H(Y|K)$, $I(K;X)$
- Spectral norms: $\|\nabla V\|$, $\lambda_{\max}(G)$
- Curvatures: $\|\nabla^2 V\|$, $R_{\text{Ric}}$

These are computed from the model's internal state $\theta_t$ and its outputs on training batches.

*Example:* Codebook collapse is diagnosed by $H(K) \to 0$. The correction (increase VQ commitment loss $\beta$) depends only on the diagnostic value, not on whether the data is images, audio, or tabular.

:::

:::{prf:proposition} Transfer via Meta-Generalization
:label: prop-transfer-via-meta-generalization

Under the conditions of the Meta-Generalization Metatheorem (**MT: Meta-Generalization** in `metalearning.md`), the Governor $\pi_{\mathfrak{G}}$ trained on a distribution of optimization landscapes $\mathcal{S}$ generalizes to new systems drawn from $\mathcal{S}$.

Specifically, if:
1. **Compact structural manifold:** The optimal diagnostic-to-correction mappings $\{\phi^*(S) : S \in \text{supp}(\mathcal{S})\}$ lie on a compact $C^1$ submanifold of the policy space
2. **Uniform local strong convexity:** The training regret $J(\phi)$ satisfies $c\,\text{dist}(\phi, \mathcal{M})^2 \leq J(\phi) \leq C\,\text{dist}(\phi, \mathcal{M})^2$ near the optimal manifold
3. **Lipschitz continuity:** The regret is Lipschitz in both the policy parameters and the training landscape

Then, with probability at least $1 - \delta$, a Governor trained on $N$ sampled landscapes satisfies:

$$
\mathbb{E}_{S \sim \mathcal{S}}[J_S(\hat{\phi}_N)] \leq C_1\left(\varepsilon_N + \sqrt{\frac{\log(1/\delta)}{N}}\right)
$$
where $\varepsilon_N$ is the optimization accuracy.

*Proof sketch (from **MT: Meta-Generalization**):*
1. The optimal corrections form a compact manifold $\mathcal{M}$ in policy space
2. Lipschitz continuity ensures uniform convergence of empirical risk to population risk
3. Approximate minimization on training landscapes implies bounded population risk
4. Local strong convexity implies the learned policy is close to the optimal manifold

In plain terms: if different training landscapes require similar corrections for similar diagnostic signatures, and the training distribution is diverse enough, the learned mapping transfers to new landscapes in the same structural class.

::::{warning} Caveat

The Meta-Generalization Metatheorem is proven in the unpublished document `metalearning.md`. While the proof follows standard statistical learning arguments (uniform convergence, Rademacher complexity bounds), the document has not undergone peer review. The assumptions (compactness, Lipschitz, strong convexity) must be verified for specific applications.
::::

:::

:::{prf:proposition} Dimensional Analysis
:label: prop-dimensional-analysis

All inputs to $\pi_{\mathfrak{G}}$ are either:
1. **Dimensionless ratios:** $\nu_{\text{cap}} = I_{\text{bulk}}/C_\partial$
2. **Entropies:** measured in nats
3. **Normalized defects:** $(C_k - \epsilon_k)/\epsilon_k$

All outputs are either dimensionless (multipliers $\lambda_k$) or have standard units ($\eta$ in step$^{-1}$, $T_c$ in nat). This ensures the Governor's function approximator operates in a well-conditioned, scale-invariant regime.

:::
(sec-meta-training-protocol-canonical-obstruction-suite)=
### 26.7 Meta-Training Protocol: Canonical Obstruction Suite

To train the Governor $\phi$, we do not use real task data. We use a set of **Canonical Topological Obstructions**.

:::{prf:definition} Canonical Obstruction Suite
:label: def-canonical-obstruction-suite

A distribution of synthetic optimization landscapes $\{\mathcal{L}_{\text{syn}}^{(i)}\}$ constructed to elicit specific failure modes:

| Obstruction            | Hessian Property                          | Failure Mode            | Diagnostic Signal                              | Required Correction                        |
|------------------------|-------------------------------------------|-------------------------|------------------------------------------------|--------------------------------------------|
| **Rosenbrock Valley**  | $\kappa(\nabla^2\mathcal{L}) \gg 1$       | Oscillation             | High $\lVert\nabla\mathcal{L}\rVert$ variance  | Reduce $\eta$ (gain scheduling)            |
| **Saddle Point**       | $\lambda_{\min}(\nabla^2\mathcal{L}) < 0$ | Stagnation              | Low $\lVert\nabla\mathcal{L}\rVert$, flat loss | Increase $T_c$ (entropy injection)         |
| **Disconnected Modes** | Multimodal landscape                      | Mode collapse           | $H(K) \to 0$                                   | Increase jump rate $\lambda_{\text{jump}}$ |
| **Noise Floor**        | High aleatoric uncertainty                | Overfitting             | $I(K; Z_{\text{tex}}) > 0$                     | Texture firewalling                        |
| **Constraint Cliff**   | Sharp constraint boundary                 | Oscillation at boundary | $C_k$ sign changes                             | Increase $\mu_k$ (barrier strength)        |

*Remark (Training Protocol).* The Governor is trained via reinforcement learning on this suite, with reward $r_t = -\Delta V_{\mathfrak{L}}$. Episodes terminate when $V_{\mathfrak{L}}$ plateaus or diverges.

:::
(sec-implementation-the-neural-governor-module)=
### 26.8 Implementation: The Neural Governor Module

We provide the implementation of the meta-controller. Note the use of bounded activations to ensure control signals remain in the admissible set $\Lambda_{\text{adm}}$.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class UniversalGovernor(nn.Module):
    """
    Implements the meta-policy π_𝔊: s_{t:t-H} → Λ_t.

    Subsumes Section 3.5 methods as special cases:
    - Primal-Dual: H=1, linear layers, no hidden state
    - PID: H≥2, linear layers with fixed weights
    - Learned Precisions: H=0, diagonal output

    References: Definition 26.3.2, Proposition 26.3.3
    """
    def __init__(
        self,
        num_constraints: int,  # K = number of Sieve nodes
        history_len: int = 100,  # H = temporal horizon
        hidden_dim: int = 128,
        num_layers: int = 2
    ):
        super().__init__()
        self.num_constraints = num_constraints
        self.history_len = history_len

        # Temporal processing of diagnostic stream (Definition 26.3.2)
        self.rnn = nn.GRU(
            input_size=num_constraints,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        # Policy Heads (output Λ_t components)
        # Constraint multipliers: λ_k ≥ 0 (Definition 26.2.3)
        self.lambda_head = nn.Linear(hidden_dim, num_constraints)

        # Learning rate scaling: η_t / η_base ∈ (0, 2) (gain scheduling)
        self.lr_scale_head = nn.Linear(hidden_dim, 1)

        # Cognitive temperature: T_c ≥ 0 (entropy injection)
        self.temp_head = nn.Linear(hidden_dim, 1)

        # Initialize with reasonable defaults
        nn.init.constant_(self.lambda_head.bias, 1.0)  # softplus(1.0) ≈ 1.31
        nn.init.constant_(self.lr_scale_head.bias, 0.0)  # sigmoid(0) = 0.5, so scale ≈ 1.0
        nn.init.constant_(self.temp_head.bias, 0.0)  # softplus(0) ≈ 0.69

    def forward(self, sieve_residuals: torch.Tensor) -> dict:
        """
        Args:
            sieve_residuals: [B, T, K] normalized constraint violations.
                             Positive values indicate violation (C_k > 0).

        Returns:
            control_dict: Dictionary containing:
                - lambda_multipliers: [B, K] constraint weights
                - lr_scale: [B, 1] learning rate multiplier
                - temp_scale: [B, 1] temperature multiplier
        """
        # 1. Process history to detect trends (Definition 26.3.1)
        # Maps s_{t:t-H} → hidden state h_t
        out, h_n = self.rnn(sieve_residuals)
        state = out[:, -1, :]  # [B, hidden_dim]

        # 2. Compute Constraint Multipliers (Dual Variables)
        # Softplus ensures λ_k ≥ 0 (Definition 26.2.3)
        log_lambdas = self.lambda_head(state)
        lambdas = F.softplus(log_lambdas)  # [B, K]

        # 3. Compute Learning Rate Scaling (Gain Scheduling)
        # Sigmoid × 2.0 gives range (0, 2) for η_t / η_base
        lr_scale = 2.0 * torch.sigmoid(self.lr_scale_head(state))  # [B, 1]

        # 4. Compute Cognitive Temperature (Entropy Control)
        # Softplus ensures T_c ≥ 0
        temp_scale = F.softplus(self.temp_head(state))  # [B, 1]

        return {
            "lambda_multipliers": lambdas,
            "lr_scale": lr_scale,
            "temp_scale": temp_scale
        }

    def compute_lyapunov_descent(
        self,
        task_loss: torch.Tensor,
        constraints: torch.Tensor,
        mu: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute V_𝔏 for monitoring (Definition 26.5.1).

        Args:
            task_loss: [B] current task loss
            constraints: [B, K] constraint values C_k(θ)
            mu: [K] penalty weights

        Returns:
            V_L: [B] Lyapunov function value
        """
        violations = F.relu(constraints)  # max(0, C_k)
        penalty = 0.5 * (mu * violations.pow(2)).sum(dim=-1)
        return task_loss + penalty
```

*Remark (Gradient Clipping).* In practice, we apply gradient clipping to the Governor's outputs to prevent extreme control signals during early training.

(sec-summary-and-diagnostic-node-a)=
### 26.9 Summary and Diagnostic Node

**Table 26.9.1 (Summary of Meta-Stability Theory).**

| Aspect            | Formula                                                                         | Units               | Reference                                              |
|-------------------|---------------------------------------------------------------------------------|---------------------|--------------------------------------------------------|
| Diagnostic State  | $s_t = \Psi(\theta_t) = [C_1, \ldots, C_K]^\top$                                | nat / dimensionless | Def {prf:ref}`def-diagnostic-state-space`              |
| Governor Policy   | $\Lambda_t = \pi_{\mathfrak{G}}(s_{t:t-H}; \phi)$                               | mixed               | Def {prf:ref}`def-the-universal-governor`              |
| Training Lyapunov | $V_{\mathfrak{L}} = \mathcal{L} + \sum_k \frac{\mu_k}{2}\max(0,C_k)^2$          | nat                 | Def {prf:ref}`def-training-lyapunov-function`          |
| Training Regret   | $J(\phi) = \mathbb{E}[\sum_t \mathcal{L}_t + \gamma_{\text{viol}}\sum_k C_k^2]$ | nat                 | Def {prf:ref}`def-outer-problem-governor-optimization` |
| Subsumption       | Primal-Dual, PID, Learned Precisions                                            | —                   | Prop {prf:ref}`prop-subsumption-of-section`            |

(node-42)=
**Node 42: GovernorStabilityCheck**

Following the diagnostic node convention (Section 3.1), we define:

| **#**  | **Name**                   | **Component**       | **Type**              | **Interpretation**                   | **Proxy**                                                                               | **Cost** |
|--------|----------------------------|---------------------|-----------------------|--------------------------------------|-----------------------------------------------------------------------------------------|----------|
| **42** | **GovernorStabilityCheck** | **Meta-Controller** | **Learning Dynamics** | Is the Governor maintaining descent? | $\Delta V_{\mathfrak{L}} = V_{\mathfrak{L}}(\theta_{t+1}) - V_{\mathfrak{L}}(\theta_t)$ | $O(K)$   |

**Trigger conditions:**
- Positive GovernorStabilityCheck ($\Delta V_{\mathfrak{L}} > 0$): Training is ascending the Lyapunov potential; instability detected.
- Remedy: Reduce learning rate; increase constraint penalties $\mu_k$; check for conflicting gradients.
- Persistent positive: Governor policy $\phi$ may need retraining on expanded Obstruction Suite.

**Cross-references:** Section 3 (Sieve Diagnostic Nodes), Section 3.5 (Adaptive Multipliers), Section 2.3 (Lyapunov-Constrained Control).



(sec-section-non-local-memory-as-self-interaction-functional)=
