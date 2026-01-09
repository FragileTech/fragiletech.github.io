## 15. Implementation Note: Entropy-Regularized Optimal Transport Bridge

:::{admonition} Researcher Bridge: KL Control as a Schrödinger Bridge
:class: tip
:name: rb-kl-control-bridge
Entropy-regularized control can be read as an optimal transport problem on trajectories. This is the same math behind soft policy iteration, just framed as a path-measure bridge.
:::

This is optional machinery, but it provides a clean path-space view of KL-regularized control and filtering.

**Entropic bridge (Schrödinger bridge) viewpoint.** Given a reference dynamics (e.g. the macro kernel $\bar{P}$, or a continuous diffusion on $\mathcal{Z}_\mu$) and two marginals (a prior belief and a boundary-conditioned posterior), the bridge problem finds the path measure closest in KL to the reference subject to matching the marginals. This is the rigorous “most likely flow under entropy regularization” principle (entropic optimal transport) {cite}`cuturi2013sinkhorn,leonard2014schrodinger`.

In the Fragile Agent:
- the reference process is the world model’s internal rollout,
- the boundary observations induce marginal constraints (via the shutter),
- the Sieve imposes feasibility constraints (via projections),
so each training update can be read as an entropic optimal transport step on belief trajectories.



(sec-theorem-the-information-stability-threshold)=
## 16. Theorem: The Information–Stability Threshold (Coupling Window)

:::{admonition} Researcher Bridge: The Stable Learning Window
:class: warning
:name: rb-stable-learning-window
The coupling window is the stability region where representation and dynamics stay grounded. For RL readers, it plays the role of a learning-rate and discount range where updates contract rather than diverge.
:::

The coupling-window view implies a necessary **window condition**: coupling must be strong enough to remain grounded (BoundaryCheck) but not so strong that the macro register loses coherence (dispersion/mixing).

We state this as a rate balance rather than an ill-typed scalar comparison.

:::{prf:definition} Grounding rate
:label: def-grounding-rate

Let $G_t:=I(X_t;K_t)$ be the symbolic mutual information injected through the boundary (Node 13). The *grounding rate* is the average information inflow per step:

$$
\lambda_{\text{in}} := \mathbb{E}[G_t].
$$
Units: $[\lambda_{\text{in}}]=\mathrm{nat/step}$.

:::
:::{prf:definition} Mixing rate
:label: def-mixing-rate

Let $S_t:=H(K_t)$ be the macro entropy. The *mixing rate* is the expected entropy growth not attributable to purposeful exploration:

$$
\lambda_{\text{mix}} := \mathbb{E}[(S_{t+1}-S_t)_+].
$$
Units: $[\lambda_{\text{mix}}]=\mathrm{nat/step}$.

:::
:::{prf:theorem} Information–stability window; operational
:label: thm-information-stability-window-operational

A necessary condition for stable, grounded macrostates is the existence of constants $0<\epsilon<\log|\mathcal{K}|$ such that, along typical trajectories,

$$
\epsilon \le I(X_t;K_t) \quad\text{and}\quad H(K_t)\le \log|\mathcal{K}|-\epsilon,
$$
and the net entropy balance satisfies

$$
\lambda_{\text{in}} \gtrsim \lambda_{\text{mix}}.
$$
Violations correspond to identifiable barrier modes:
- If $I(X;K)\approx 0$: under-coupling → ungrounded inference / decoupling (Mode D.C).
- If $H(K)\approx \log|\mathcal{K}|$: over-coupling or dispersion → symbol dispersion (BarrierScat).

*Remark.* This theorem is intentionally stated at the level of measurable information quantities (Gate Nodes) so it can be audited online; strengthening it to a sufficient condition requires specifying the macro kernel class and a contraction inequality (e.g. log-Sobolev / Doeblin-type conditions).

:::

::::{note} Connection to RL #9: Conservative Q-Learning as Soft Coupling Window
**The General Law (Fragile Agent):**
The **Coupling Window** (Theorem {prf:ref}`thm-information-stability-window-operational`) imposes a **hard constraint** on information flow:

$$
\epsilon \le I(X_t; K_t) \quad \text{and} \quad H(K_t) \le \log|\mathcal{K}| - \epsilon.
$$
If violated, the Sieve halts execution (BoundaryCheck failure). This ensures that offline data cannot drive the agent into ungrounded regions of state space.

**The Degenerate Limit:**
Replace the hard constraint with a soft Q-value penalty. Allow violations if reward is high enough.

**The Special Case (Standard RL - CQL):**
Conservative Q-Learning {cite}`kumar2020conservative` adds a penalty for out-of-distribution actions:

$$
\min_Q \mathbb{E}_{s \sim \mathcal{D}, a \sim \mu}\left[\log \sum_a \exp Q(s,a)\right] - \mathbb{E}_{s,a \sim \mathcal{D}}[Q(s,a)] + \text{Bellman}.
$$
This softly penalizes overestimation on unseen actions but **does not prevent** the agent from taking them.

**Result:** CQL is the $\epsilon \to 0$ limit where coupling constraints become soft penalties rather than hard firewalls.

**What the generalization offers:**
- **Hard guarantees**: BoundaryCheck halts execution when grounding fails—the agent cannot "pay the fine" and proceed
- **Auditable thresholds**: $I(X_t; K_t)$ is computed at runtime; failures are logged with specific diagnostic codes
- **Information-theoretic grounding**: The constraint is derived from the Data Processing Inequality, not a heuristic penalty term
- **Bidirectional protection**: Both under-coupling (ungrounded) and over-coupling (dispersion) are detected and blocked
::::



(sec-summary-unified-information-theoretic-control-view)=
## 17. Summary: Unified Information-Theoretic Control View

| Level | Formalism | Law |
| :--- | :--- | :--- |
| Geometry | Riemannian $(\mathcal{Z},G)$ | Distance measured by a sensitivity metric (Fisher/Hessian) |
| Boundary | Markov blanket $B_t$ | Environment = boundary law $P_{\partial}$ (Section 1.1) |
| Exploration | Causal entropy / MaxEnt RL | Reachability pressure via path entropy on $\mathcal{K}$ (Section 10) |
| Belief Dynamics | Filtering + projection | Predict → update → project (Section 11) |
| Optimality | Soft Bellman / log-normalizer | Soft value = log-normalizer; exploration gradient from path entropy (Section 13) |

**Fragile conclusion.** The agent is a controller with explicit information and stability constraints. Macro symbols remain meaningful only inside the coupling window (Theorem {prf:ref}`thm-information-stability-window-operational`); outside it, the system either exhibits ungrounded inference (under-coupling) or loses macro structure through excessive mixing/dispersion (over-coupling).



(sec-capacity-constrained-metric-law-geometry-from-interface-limits)=
