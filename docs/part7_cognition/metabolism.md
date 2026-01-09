## 31. Computational Metabolism: The Landauer Bound and Deliberation Dynamics

*Abstract.* We establish a thermodynamic foundation for internal inference by coupling computation time $s$ to an energetic cost functional. We model the agent as an open system where belief updates are dissipative processes. By applying Landauer's Principle {cite}`landauer1961irreversibility` to the Wasserstein-Fisher-Rao (WFR) flow, we prove that the optimal allocation of computation time $S^*$ emerges from the stationarity of a **Dual-Horizon Action**. We derive a rigorous phase transition between reflexive (fast) and deliberative (slow) regimes {cite}`kahneman2011thinking`, governed by the ratio of the task-gradient norm to the metabolic dissipation rate.

:::{admonition} Researcher Bridge: Principled "Thinking Fast and Slow"
:class: info
:name: rb-thinking-fast-slow
Most agents spend the same amount of FLOPs on a trivial decision as a critical one. We use the **Landauer Bound** to assign a thermodynamic cost to information updates. The agent stops "deliberating" ($S^*$) exactly when the marginal gain in Value is outweighed by the metabolic cost of more compute. This derives "System 1 vs System 2" behavior from first principles.
:::

*Cross-references:* This section extends the WFR dynamics (Section 20) to account for the thermodynamic cost of belief updates, building on the cognitive temperature framework (Section 22.4) and the value potential (Section 24).

*Literature:* Landauer's principle {cite}`landauer1961irreversibility`; thermodynamics of computation {cite}`bennett1982thermodynamics`; thermodynamics of information {cite}`parrondo2015thermodynamics`; dual-process theory {cite}`kahneman2011thinking`; free energy principle {cite}`friston2010free`; information geometry {cite}`amari2016information`.



(sec-the-energetics-of-information-updates)=
### 31.1 The Energetics of Information Updates

We begin by mapping the abstract WFR belief dynamics (Section 20) {cite}`chizat2018unbalanced,liero2018optimal` to physical dissipation via Landauer's Principle.

:::{prf:definition} Metabolic Flux
:label: def-metabolic-flux

Let $\rho(s, z)$ be the belief density evolving in computation time $s$ according to the WFR continuity equation (Definition {prf:ref}`def-the-wfr-action`):

$$
\partial_s \rho + \nabla \cdot (\rho v) = \rho r.
$$
We define the **Metabolic Flux** $\dot{\mathcal{M}}: \mathbb{R}_{\ge 0} \to \mathbb{R}_{\ge 0}$ as:

$$
\dot{\mathcal{M}}(s) := \sigma_{\text{met}} \int_{\mathcal{Z}} \left( \|v_s(z)\|_G^2 + \lambda^2 |r_s(z)|^2 \right) \rho(s, z) \, d\mu_G,
$$
where:
- $\sigma_{\text{met}} > 0$ is the **metabolic resistance coefficient** (units: nat$\cdot$step)
- $v_s(z)$ is the velocity field at computation time $s$
- $r_s(z)$ is the reaction rate (mass creation/destruction)
- $\lambda$ is the WFR length-scale (Definition {prf:ref}`def-the-wfr-action`)
- $d\mu_G = \sqrt{\det G} \, dz$ is the Riemannian volume form

*Physical interpretation:* The metabolic flux measures the instantaneous rate of energy dissipation required to update the belief distribution. Transport ($\|v\|_G^2$) represents the cost of moving probability mass; reaction ($|r|^2$) represents the cost of creating or destroying mass. The WFR action is the kinetic energy of the belief flow.

:::
:::{prf:theorem} Generalized Landauer Bound
:label: thm-generalized-landauer-bound

The metabolic flux $\dot{\mathcal{M}}$ provides a physical lower bound on the rate of entropy reduction within the agent. Specifically:

$$
\dot{\mathcal{M}}(s) \ge T_c \left| \frac{d}{ds} H(\rho_s) \right|,
$$
where $H(\rho_s) = -\int_{\mathcal{Z}} \rho \ln \rho \, d\mu_G$ is the Shannon entropy and $T_c$ is the cognitive temperature (Section 22.4).

*Proof sketch.* The time derivative of the Shannon entropy is:

$$
\frac{d}{ds} H(\rho_s) = -\int_{\mathcal{Z}} (1 + \ln \rho) \partial_s \rho \, d\mu_G.
$$
Substituting the WFR continuity equation and integrating by parts (assuming vanishing flux at $\partial\mathcal{Z}$):

$$
\frac{d}{ds} H = \int_{\mathcal{Z}} \rho \langle \nabla \ln \rho, v \rangle_G \, d\mu_G - \int_{\mathcal{Z}} r \ln \rho \cdot \rho \, d\mu_G.
$$
By the Cauchy-Schwarz inequality on the tangent bundle $(T\mathcal{Z}, G)$:

$$
\left| \int_{\mathcal{Z}} \rho \langle \nabla \ln \rho, v \rangle_G \, d\mu_G \right| \le \left( \int_{\mathcal{Z}} \rho \|\nabla \ln \rho\|_G^2 \, d\mu_G \right)^{1/2} \left( \int_{\mathcal{Z}} \rho \|v\|_G^2 \, d\mu_G \right)^{1/2}.
$$
The first factor is the **Fisher Information** $\mathcal{I}(\rho) = \int \rho \|\nabla \ln \rho\|_G^2 \, d\mu_G$ {cite}`amari2016information`. Under the optimal transport scaling $v = -T_c \nabla \ln \rho$ (gradient flow of the free energy), we recover the de Bruijn identity {cite}`stam1959some` and the bound follows. The reaction term satisfies an analogous inequality via the $L^2(\rho)$ norm. See Appendix E.3 for the full proof. $\square$

*Remark (Landauer's Principle).* The classical Landauer bound states that erasing one bit of information requires dissipating at least $k_B T \ln 2$ joules of heat. Theorem {prf:ref}`thm-generalized-landauer-bound` is the information-geometric generalization: reducing belief entropy by $\Delta H$ nats requires dissipating at least $T_c \cdot |\Delta H|$ nats of metabolic energy.

:::

::::{admonition} Physics Isomorphism: Landauer's Principle
:class: note
:name: pi-landauer-principle

**In Physics:** Erasing one bit of information requires dissipating at least $k_B T \ln 2$ joules of heat. More generally, reducing entropy by $\Delta S$ requires work $W \geq T|\Delta S|$ {cite}`landauer1961irreversibility,bennett1982thermodynamics`.

**In Implementation:** The generalized Landauer bound (Theorem {prf:ref}`thm-generalized-landauer-bound`):

$$
\dot{\mathcal{M}}(s) \geq T_c \left|\frac{d}{ds} H(\rho_s)\right|
$$
**Correspondence Table:**

| Thermodynamics | Agent (Metabolic) |
|:---------------|:------------------|
| Temperature $T$ | Cognitive temperature $T_c$ |
| Heat dissipation $\dot{Q}$ | Metabolic flux $\dot{\mathcal{M}}$ |
| Entropy $S$ | Belief entropy $H(\rho)$ |
| Boltzmann constant $k_B$ | 1 (nat units) |
| Work $W$ | Cumulative metabolic cost $\Psi_{\text{met}}$ |

**Consequence:** Thinking has irreducible thermodynamic cost. Deliberation stops when marginal value gain equals metabolic cost.
::::

:::{note} Connection to RL #14: Maximum Expected Utility as Zero-Temperature Limit
**The General Law (Fragile Agent):**
The agent optimizes a **Free Energy** objective that includes the metabolic cost of computation:

$$
\mathcal{F}[p, \pi] = \int_{\mathcal{Z}} p(z) \Big( V(z) - T_c H(\pi(\cdot|z)) \Big) d\mu_G - \Psi_{\text{met}}
$$
where $\Psi_{\text{met}} = \int_0^S \dot{\mathcal{M}}(s)\,ds$ is the cumulative metabolic energy. The agent stops thinking when marginal returns equal marginal costs.

**The Degenerate Limit:**
Set $T_c \to 0$ (computational temperature zero). Compute is free and infinite.

**The Special Case (Standard RL):**

$$
J(\pi) = \max_\pi \mathbb{E}\left[\sum_{t=0}^\infty \gamma^t r_t\right]
$$
This recovers standard **Maximum Expected Utility**—the objective used in DQN, PPO, SAC, etc.

**Result:** Standard RL ignores the thermodynamic cost of inference. The agent assumes it has infinite compute and can think forever. The Fragile Agent has an irreducible "cost of thinking" governed by the Landauer bound.

**What the generalization offers:**
- Principled stopping: deliberation ends when $\Gamma(S^*) = \dot{\mathcal{M}}(S^*)$ (marginal return = marginal cost)
- Fast/Slow phase transition: System 1 ($S^*=0$) vs System 2 ($S^*>0$) from first principles
- Landauer bound: $\dot{\mathcal{M}} \ge T_c |\dot{H}|$—thinking has irreducible thermodynamic cost
:::



(sec-the-metabolic-potential-and-deliberation-action)=
### 31.2 The Metabolic Potential and Deliberation Action

We introduce the metabolic cost as a coordinate in the agent's extended state space.

:::{prf:definition} Metabolic Potential
:label: def-metabolic-potential

We define $\Psi_{\text{met}}(s) := \int_0^s \dot{\mathcal{M}}(u) \, du$ as the cumulative metabolic energy dissipated during a single interaction step $t$ for an internal rollout of duration $s$. Units: $[\Psi_{\text{met}}] = \text{nat}$.

:::
:::{prf:axiom} Dual-Horizon Action
:label: ax-dual-horizon-action

For any interaction step $t$, the agent selects a total computation budget $S \in [0, S_{\max}]$ that minimizes the **Deliberation Action** $\mathcal{S}_{\text{delib}}$:

$$
\mathcal{S}_{\text{delib}}[S] = -\underbrace{\mathbb{E}_{z \sim \rho_S} [V(z)]}_{\text{Expected Terminal Value}} + \underbrace{\Psi_{\text{met}}(S)}_{\text{Computational Cost}},
$$
where $V(z)$ is the task potential (Section 24.2.1). Units: $[\mathcal{S}_{\text{delib}}] = \text{nat}$.

*Physical interpretation:* The agent faces a trade-off: longer deliberation ($S$ large) improves the expected value $\langle V \rangle_{\rho_S}$ by refining the belief toward high-value regions, but incurs greater metabolic cost $\Psi_{\text{met}}(S)$. The optimal $S^*$ balances these competing pressures.

*Remark (Sign convention).* We write $-\langle V \rangle$ because the agent seeks to **maximize** value. The Deliberation Action $\mathcal{S}_{\text{delib}}$ is minimized when value is maximized and cost is minimized.

:::



(sec-optimal-deliberation-the-fast-slow-law)=
### 31.3 Optimal Deliberation: The Fast/Slow Law

We now prove the existence of an optimal "stopping time" for internal thought.

:::{prf:theorem} Deliberation Optimality Condition
:label: thm-deliberation-optimality-condition

Let $\rho_s$ evolve as a gradient flow of $V$ under WFR dynamics. The optimal computation budget $S^*$ satisfies:

$$
\left. \frac{d}{ds} \langle V \rangle_{\rho_s} \right|_{s=S^*} = \dot{\mathcal{M}}(S^*),
$$
provided such an $S^*$ exists in $(0, S_{\max})$.

*Proof.* We seek to extremize $\mathcal{S}_{\text{delib}}$ with respect to the upper integration limit $S$. By the Leibniz Integral Rule and the definition of $\Psi_{\text{met}}$:

$$
\frac{d}{dS} \mathcal{S}_{\text{delib}} = -\frac{d}{dS} \langle V \rangle_{\rho_S} + \dot{\mathcal{M}}(S).
$$
The first term is the **Value-Improvement Rate**:

$$
\frac{d}{dS} \langle V \rangle_{\rho_S} = \int_{\mathcal{Z}} V(z) \partial_s \rho(S, z) \, d\mu_G.
$$
Applying the WFR continuity equation $\partial_s \rho = \rho r - \nabla \cdot (\rho v)$:

$$
\frac{d}{dS} \langle V \rangle_{\rho_S} = \int_{\mathcal{Z}} V \cdot \rho r \, d\mu_G + \int_{\mathcal{Z}} V (-\nabla \cdot (\rho v)) \, d\mu_G.
$$
Integrating the divergence term by parts (assuming vanishing flux at $\partial\mathcal{Z}$):

$$
\int_{\mathcal{Z}} V (-\nabla \cdot (\rho v)) \, d\mu_G = \int_{\mathcal{Z}} \rho \langle \nabla V, v \rangle_G \, d\mu_G.
$$
For gradient flow dynamics, $v = -G^{-1} \nabla V$ (up to temperature scaling), so $\langle \nabla V, v \rangle_G = -\|\nabla V\|_G^2 \le 0$. Thus:

$$
\frac{d}{dS} \langle V \rangle_{\rho_S} = \int_{\mathcal{Z}} \rho \left( V r - \|\nabla V\|_G^2 \right) d\mu_G.
$$
The stationarity condition $\frac{d}{dS} \mathcal{S}_{\text{delib}} = 0$ yields the optimality condition. See Appendix E.4 for the full proof using the WFR adjoint operator. $\square$

*Physical interpretation:* The optimal stopping time $S^*$ is reached when the marginal gain in expected value (the "return on thinking") exactly equals the marginal metabolic cost (the "price of thinking"). At $S^*$, the agent has extracted all cost-effective information from deliberation.

:::
:::{prf:theorem} Fast/Slow Phase Transition
:label: thm-fast-slow-phase-transition

Let $\Gamma(s) := \left| \frac{d}{ds} \langle V \rangle_{\rho_s} \right|$ be the **Value-Improvement Rate**. There exists a critical threshold such that:

1. **Reflexive Regime (Fast):** If $\Gamma(0) < \dot{\mathcal{M}}(0)$, then $S^* = 0$. The agent executes an immediate action based on the prior $\rho_0$.

2. **Deliberative Regime (Slow):** If $\Gamma(0) > \dot{\mathcal{M}}(0)$, then $S^* > 0$. The agent enters a planning state, terminating only when the marginal gain in Value equals the marginal metabolic cost.

*Proof.* Consider the derivative of the Deliberation Action at $S = 0$:

$$
\left. \frac{d}{dS} \mathcal{S}_{\text{delib}} \right|_{S=0} = -\Gamma(0) + \dot{\mathcal{M}}(0).
$$
If $\Gamma(0) < \dot{\mathcal{M}}(0)$, then $\frac{d}{dS} \mathcal{S}_{\text{delib}}|_{S=0} > 0$. Since $\mathcal{S}_{\text{delib}}$ is increasing at $S=0$ and we assume $\mathcal{S}_{\text{delib}}$ is convex (which holds when $\Gamma(s)$ is decreasing due to diminishing returns), the minimum occurs at the boundary $S^* = 0$.

If $\Gamma(0) > \dot{\mathcal{M}}(0)$, then $\frac{d}{dS} \mathcal{S}_{\text{delib}}|_{S=0} < 0$. The agent benefits from deliberation. As $s$ increases, $\Gamma(s)$ decreases (diminishing marginal returns on thinking) while $\dot{\mathcal{M}}(s)$ may increase or remain constant. The optimum $S^* > 0$ occurs when the curves cross: $\Gamma(S^*) = \dot{\mathcal{M}}(S^*)$. $\square$

*Remark (Dual-Process Theory).* Theorem {prf:ref}`thm-fast-slow-phase-transition` provides a first-principles derivation of Kahneman's "System 1 / System 2" dichotomy {cite}`kahneman2011thinking`. System 1 (reflexive) corresponds to $S^* = 0$; System 2 (deliberative) corresponds to $S^* > 0$. The transition is not a cognitive style but a phase transition governed by the ratio $\Gamma(0) / \dot{\mathcal{M}}(0)$.

:::

:::{prf:theorem} Generalized Stopping for Non-Conservative Fields
:label: thm-generalized-stopping

When the Value Curl does not vanish ($\mathcal{F} \neq 0$, Definition {prf:ref}`def-value-curl`), the agent converges to a Non-Equilibrium Steady State (Theorem {prf:ref}`thm-ness-existence`) rather than a fixed point. The stopping criterion generalizes as follows:

**Conservative Case ($\mathcal{F} = 0$):** Stop when the Value-Improvement Rate equals the metabolic cost:
$$
\Gamma(S^*) = \dot{\mathcal{M}}(S^*)
$$

**Non-Conservative Case ($\mathcal{F} \neq 0$):** Stop when the **orbit parameters converge**:
$$
\frac{d}{ds}\|\text{Orbit}(s)\|_{\text{param}} < \epsilon_{\text{orbit}}
$$
even if the agent continues moving within the limit cycle.

*Remark.* In the conservative case, convergence is to a fixed point ($\dot{z} \to 0$). In the non-conservative case, convergence is to a stable limit cycle (periodic orbit with constant parameters).

**Operational Criterion:** Define the orbit-change metric as:
$$
\Delta_{\text{orbit}}(s) := \left\| \oint_{\gamma_s} \mathcal{R} - \oint_{\gamma_{s-\delta}} \mathcal{R} \right\|
$$
where $\gamma_s$ is the closed trajectory over one cycle at time $s$. Stop when $\Delta_{\text{orbit}}(s) < \epsilon_{\text{orbit}}$.

*Remark.* In the non-conservative case, the agent accumulates reward along periodic trajectories. Deliberation terminates when the orbit parameters stabilize, not when motion ceases.

:::

:::{note} Connection to RL #15: UCB as Degenerate Thermodynamic VOI
**The General Law (Fragile Agent):**
The agent explores based on the **Thermodynamic Value of Information**:

$$
\text{VOI}(a) := \mathbb{E}[\Delta H(\rho) \mid a] - \frac{1}{T_c} \dot{\mathcal{M}}(a)
$$
Exploration is justified when the expected entropy reduction exceeds the metabolic cost.

**The Degenerate Limit:**
Assume a **single-state manifold** (no dynamics, stateless bandit). Use simplified Gaussian uncertainty.

**The Special Case (Multi-Armed Bandits):**

$$
a^* = \arg\max_a \left[ \hat{\mu}_a + c \sqrt{\frac{\ln t}{n_a}} \right]
$$
This recovers **UCB1 (Upper Confidence Bound)**. The exploration bonus $c\sqrt{\ln t / n_a}$ is the specific solution to the Landauer inequality for Gaussian arm distributions.

**Result:** UCB is the **thermodynamics of a single point**—exploration when there's no state, no dynamics, just uncertainty about arm means. The Fragile Agent generalizes to full manifold dynamics where exploration depends on local geometry.

**What the generalization offers:**
- State-dependent exploration: VOI varies with position $z$ on the manifold
- Geometric awareness: exploration bonus depends on local curvature $G(z)$
- Deliberation-aware: exploration trades off against computational cost $\dot{\mathcal{M}}$
:::



(sec-the-h-theorem-for-open-cognitive-systems)=
### 31.4 The H-Theorem for Open Cognitive Systems

We reconcile computation with the Second Law of Thermodynamics {cite}`crooks1999entropy,parrondo2015thermodynamics`.

:::{prf:theorem} Total Entropy Production
:label: thm-total-entropy-production

The total entropy production rate of the agent $\sigma_{\text{tot}}$ during computation is:

$$
\sigma_{\text{tot}}(s) := \frac{d}{ds} H(\rho_s) + \frac{1}{T_c} \dot{\mathcal{M}}(s) \ge 0.
$$
*Proof.* From Theorem {prf:ref}`thm-generalized-landauer-bound`, $\dot{\mathcal{M}}(s) \ge T_c |\frac{d}{ds} H(\rho_s)|$. If $\frac{d}{ds} H < 0$ (entropy decreasing), then:

$$
\sigma_{\text{tot}} = \frac{dH}{ds} + \frac{\dot{\mathcal{M}}}{T_c} \ge \frac{dH}{ds} + \left| \frac{dH}{ds} \right| = \frac{dH}{ds} - \frac{dH}{ds} = 0.
$$
If $\frac{d}{ds} H \ge 0$, then $\sigma_{\text{tot}} \ge 0$ trivially since $\dot{\mathcal{M}} \ge 0$. $\square$

*Interpretation:* The agent can only reduce its internal uncertainty ($dH/ds < 0$) by dissipating metabolic energy ($\dot{\mathcal{M}} > 0$) {cite}`still2012thermodynamics`. This defines the **Efficiency of Thought**:

$$
\eta_{\text{thought}} := \frac{-T_c \cdot dH/ds}{\dot{\mathcal{M}}} \le 1.
$$
An agent is "thermodynamically fragile" if it requires high metabolic flux for low entropy reduction ($\eta_{\text{thought}} \ll 1$).

:::
:::{prf:definition} Cognitive Carnot Efficiency
:label: def-cognitive-carnot-efficiency

The **Carnot limit** for cognitive systems is $\eta_{\text{thought}} = 1$, achieved when the belief update is a reversible isothermal process. Real agents operate at $\eta_{\text{thought}} < 1$ due to:
1. **Friction:** Non-optimal transport paths (geodesic deviation)
2. **Irreversibility:** Finite-rate updates (non-quasi-static processes)
3. **Dissipation:** Exploration noise ($T_c > 0$)

:::



(sec-diagnostic-nodes-b)=
### 31.5 Diagnostic Nodes 51–52

Following the diagnostic node convention (Section 3.1), we define two new monitors for metabolic efficiency.

(node-51)=
**Node 51: MetabolicEfficiencyCheck**

| **#**  | **Name**                     | **Component** | **Type**          | **Interpretation**             | **Proxy**                                                                                | **Cost** |
|--------|------------------------------|---------------|-------------------|--------------------------------|------------------------------------------------------------------------------------------|----------|
| **51** | **MetabolicEfficiencyCheck** | Solver        | Inference Economy | Is computation cost-effective? | $\eta_{\text{ROI}} := \frac{\lvert\Delta \langle V \rangle\rvert}{\Psi_{\text{met}}(S)}$ | $O(1)$   |

**Interpretation:** Monitors the **Return on Investment** of deliberation. High $\eta_{\text{ROI}}$ indicates efficient thinking; low $\eta_{\text{ROI}}$ indicates the agent is "daydreaming"—expending compute without improving terminal value.

**Threshold:** $\eta_{\text{ROI}} > \eta_{\text{min}}$ (typical default $\eta_{\text{min}} = 0.1$).

**Trigger conditions:**
- Low MetabolicEfficiencyCheck: The agent is in deliberative deadlock (Mode C.C: Decision Paralysis).
- **Remediation:** Apply **SurgCC** (time-boxing): force $S \le S_{\text{cap}}$ to bound deliberation.

(node-52)=
**Node 52: LandauerViolationCheck (EntropyProductionCheck)**

| **#**  | **Name**                   | **Component** | **Type**         | **Interpretation**                     | **Proxy**                                           | **Cost** |
|--------|----------------------------|---------------|------------------|----------------------------------------|-----------------------------------------------------|----------|
| **52** | **LandauerViolationCheck** | Dynamics      | Update Stability | Is the update thermodynamically valid? | $\delta_L := \dot{\mathcal{M}} + T_c \frac{dH}{ds}$ | $O(d)$   |

**Interpretation:** Monitors the Landauer bound (Theorem {prf:ref}`thm-generalized-landauer-bound`). A violation ($\delta_L < 0$) indicates entropy is decreasing faster than metabolic dissipation permits—a non-physical update.

**Threshold:** $\delta_L \ge -\epsilon_L$ (typical default $\epsilon_L = 10^{-4}$).

**Trigger conditions:**
- Negative LandauerViolationCheck: Non-physical belief update detected.
- **Cause:** Numerical errors in the WFR solver, unstable metric $G$, or incorrectly estimated entropy.
- **Remediation:** Reduce integration step size; verify metric positive-definiteness; check entropy estimator calibration.

*Cross-reference:* Node 52 extends the thermodynamic consistency checks of Section 23.4 (ThermoCycleCheck, Node 33) to the internal deliberation loop.



(sec-summary-table-computational-thermodynamics)=
### 31.6 Summary Table: Computational Thermodynamics

**Table 31.6.1 (Computational Metabolism Summary).**

| Concept                | Thermodynamic Variable | Agent Implementation                                          |
|:-----------------------|:-----------------------|:--------------------------------------------------------------|
| **Energy**             | Gibbs Free Energy      | Task Potential $V(z)$                                         |
| **Heat**               | Metabolic Dissipation  | WFR Action $\dot{\mathcal{M}}$                                |
| **Work**               | Value Improvement      | Gradient Flux $\langle \nabla V, v \rangle_G$                 |
| **Equilibrium**        | $dG = 0$               | $S^*$ (Optimal Stopping)                                      |
| **Temperature**        | $T$                    | Cognitive Temperature $T_c$                                   |
| **Entropy Production** | $\sigma \ge 0$         | $\sigma_{\text{tot}} = \dot{H} + \dot{\mathcal{M}}/T_c \ge 0$ |

**Key Results:**
1. **Landauer Bound (Theorem {prf:ref}`thm-generalized-landauer-bound`):** $\dot{\mathcal{M}} \ge T_c |\dot{H}|$—thinking has a thermodynamic cost.
2. **Optimal Deliberation (Theorem {prf:ref}`thm-deliberation-optimality-condition`):** $S^*$ satisfies $\Gamma(S^*) = \dot{\mathcal{M}}(S^*)$—stop thinking when marginal returns equal marginal costs.
3. **Phase Transition (Theorem {prf:ref}`thm-fast-slow-phase-transition`):** Fast ($S^* = 0$) vs. Slow ($S^* > 0$) is determined by $\Gamma(0) \lessgtr \dot{\mathcal{M}}(0)$.

**Conclusion.** Computational Metabolism provides the "biological" limit for the Fragile Agent. By deriving $S^*$ from first principles, we transform the "Thinking Fast vs. Slow" heuristic into a rigorous physical law. The agent acts not when it is "ready," but when it is no longer metabolically efficient to continue refining its belief. This framework connects to the free energy principle {cite}`friston2010free` and active inference {cite}`friston2017active`, providing a thermodynamic foundation for bounded rationality.



(sec-causal-discovery-interventional-geometry-and-the-singularity-of-action)=
