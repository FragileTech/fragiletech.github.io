## 11. Intrinsic Motivation: Maximum-Entropy Exploration

:::{admonition} Researcher Bridge: Max-Entropy Exploration in Macro Space
:class: info
:name: rb-maxent-exploration
This is the MaxEnt RL idea applied to discrete macro trajectories. Instead of adding a scalar bonus, we maximize the entropy of reachable macro futures, which is the discrete version of "keep options open."
:::

The previous layers define representation ($K,z_n,z_{\mathrm{tex}}$), predictive dynamics ($\bar{P}$), and stability/value constraints ($V,G$, Sieve checks). This layer formalizes an **intrinsic exploration pressure** on the discrete macro register: prefer policies that keep the set of reachable future macrostates diverse, which supports reachability/controllability and reduces brittle overcommitment to narrow paths.

(sec-path-entropy-and-exploration-gradients)=
### 11.1 Path Entropy and Exploration Gradients

We work on the **macro model** (the discrete register). Assume a macro Markov kernel

$$
\bar{P}(k'\mid k,a),\qquad k,k'\in\mathcal{K},\ a\in\mathcal{A},
$$
which is the learned effective dynamics demanded by Causal Enclosure (Section 2.8).

:::{prf:definition} Macro Path Distribution
:label: def-macro-path-distribution

Fix a horizon $H\in\mathbb{N}$ and a (possibly stochastic) policy $\pi(a\mid k)$. The induced distribution over length-$H$ macro trajectories

$$
\xi := (K_{t+1},\dots,K_{t+H}) \in \mathcal{K}^H
$$
conditioned on $K_t=k$ is

$$
P_\pi(\xi\mid k)
:=
\sum_{a_{t:t+H-1}}
\prod_{h=0}^{H-1}\pi(a_{t+h}\mid K_{t+h})\ \bar{P}(K_{t+h+1}\mid K_{t+h},a_{t+h}).
$$
(For continuous $\mathcal{A}$, replace the sum by an integral with respect to the action reference measure.)

:::
:::{prf:definition} Causal Path Entropy
:label: def-causal-path-entropy

The causal path entropy at $(k,H)$ under $\pi$ is the Shannon entropy of the path distribution:

$$
S_c(k,H;\pi) := H\!\left(P_\pi(\cdot\mid k)\right)
= -\sum_{\xi\in\mathcal{K}^H} P_\pi(\xi\mid k)\log P_\pi(\xi\mid k).
$$
This quantity is well-typed precisely because the macro register is discrete: there is no differential-entropy ambiguity.

:::
:::{prf:definition} Exploration Gradient, metric form
:label: def-exploration-gradient-metric-form

Let $z_{\text{macro}}=e_k\in\mathbb{R}^{d_m}$ denote the code embedding of $k$ (Section 2.2b), and let $G$ be the relevant metric on the macro chart (Section 2.5). Define the exploration gradient as the metric gradient of path entropy:

$$
\mathbf{g}_{\text{expl}}(e_k) := T_c\ \nabla_G S_c(k,H;\pi),
$$
where $T_c>0$ is an entropy-regularization coefficient. Operationally, gradients are taken through the continuous pre-quantization coordinates (straight-through VQ estimator); in the strictly symbolic limit, the gradient becomes a discrete preference ordering induced by $S_c(k,H;\pi)$.

**Interpretation (Exploration / Reachability).** $S_c(k,H;\pi)$ measures how many future macro-trajectories remain plausible from $k$ under $\pi$ and $\bar{P}$. Increasing $S_c$ preserves **future reachability**: the agent stays inside regions with many reachable, non-absorbing macrostates.

:::
(sec-maxent-duality-utility-entropy-regularization)=
### 11.2 MaxEnt Duality: Utility + Entropy Regularization

The same object appears from a variational principle.

:::{prf:definition} MaxEnt RL objective on macrostates
:label: def-maxent-rl-objective-on-macrostates

Let $\mathcal{R}(k,a)$ be an instantaneous reward/cost-rate term (Section 1.1.2, Section 2.7) and let $\gamma\in(0,1)$ be the discount factor (dimensionless). The maximum-entropy objective is

$$
J_{T_c}(\pi)
:=
\mathbb{E}_\pi\left[\sum_{t\ge 0}\gamma^t\left(\mathcal{R}(K_t,A_t) + T_c\,\mathcal{H}(\pi(\cdot\mid K_t))\right)\right],
$$
where $\mathcal{H}$ is Shannon entropy. This is the standard “utility + entropy regularization” objective.

**Regimes.**
- $T_c\to 0$: $\pi$ collapses toward determinism; behavior can be brittle under distribution shift.
- $T_c\to\infty$: $\pi$ approaches maximal entropy; behavior becomes overly random and may degrade grounding (BarrierScat).
- The useful regime is intermediate: enough entropy to remain robust, enough utility to remain directed.

:::
:::{prf:proposition} Soft Bellman form, discrete actions
:label: prop-soft-bellman-form-discrete-actions

Assume finite $\mathcal{A}$. Define the soft state value

$$
V^*(k) := \max_{\pi} \ \mathbb{E}\Big[\sum_{t\ge 0}\gamma^t(\mathcal{R}+T_c\mathcal{H})\ \Big|\ K_0=k\Big].
$$
Then $V^*$ satisfies the entropic Bellman fixed point

$$
V^*(k)
=
T_c \log \sum_{a\in\mathcal{A}}
\exp\!\left(\frac{1}{T_c}\left(\mathcal{R}(k,a)+\gamma\,\mathbb{E}_{k'\sim\bar{P}(\cdot\mid k,a)}[V^*(k')]\right)\right),
$$
and the corresponding optimal policy is the softmax policy

$$
\pi^*(a\mid k)\propto
\exp\!\left(\frac{1}{T_c}\left(\mathcal{R}(k,a)+\gamma\,\mathbb{E}[V^*(k')]\right)\right).
$$
*Proof sketch.* Standard convex duality / log-sum-exp variational identity: maximizing expected reward plus entropy yields a softmax (exponential-family) distribution; substituting back produces the log-partition recursion. (This is the “soft”/MaxEnt Bellman equation used in SAC-like methods.)

**Consequence.** The same mathematics can be read as:
1) maximize reward while retaining policy entropy (MaxEnt RL), or
2) maximize reachability/diversity of future macro trajectories (intrinsic motivation).

:::



(sec-belief-dynamics-prediction-update-projection)=

## 14. Duality of Exploration and Soft Optimality

:::{admonition} Researcher Bridge: Soft RL Equals Exploration Duality
:class: info
:name: rb-soft-rl-duality
If you know SAC or KL control, this section formalizes why maximizing entropy and optimizing soft value are the same problem. The exploration gradient is just the covariant form of that duality.
:::

This section makes the exploration layer precise: the exploration gradient (Section 10.1.3) is dual to entropy-regularized (soft) optimal control once the macro channel is discrete.

(sec-formal-definitions)=
### 14.1 Formal Definitions (Path Space, Causal Entropy, Exploration Gradient)

:::{prf:definition} Causal Path Space
:label: def-causal-path-space

For a macrostate $k\in\mathcal{K}$ and horizon $H$, define the future macro path space

$$
\Gamma_H(k) := \mathcal{K}^H.
$$
:::
:::{prf:definition} Path Probability
:label: def-path-probability

$P_\pi(\xi\mid k)$ is the induced path probability from Definition 10.1.1.

:::
:::{prf:definition} Causal Entropy
:label: def-causal-entropy

$S_c(k,H;\pi)$ is the Shannon entropy of $P_\pi(\cdot\mid k)$ (Definition 10.1.2).

:::
:::{prf:definition} Exploration gradient, covariant form
:label: def-exploration-gradient-covariant-form

On a macro chart with metric $G$ (Section 2.5),

$$
\mathbf{g}_{\text{expl}}(e_k) := T_c\,\nabla_G S_c(k,H;\pi).
$$
:::
(sec-the-equivalence-theorem)=
### 14.2 The Equivalence Theorem (Duality of Causal Regulation)

We state the equivalence in the setting where it is unambiguous.

:::{prf:theorem} Equivalence of Entropy-Regularized Control Forms; discrete macro
:label: thm-equivalence-of-entropy-regularized-control-forms-discrete-macro

Assume:
1. finite macro alphabet $\mathcal{K}$ and (for simplicity) finite action set $\mathcal{A}$,
2. an enclosure-consistent macro kernel $\bar{P}(k'\mid k,a)$,
3. bounded reward flux $\mathcal{R}(k,a)$.

Then the following are equivalent characterizations of the same optimal control law:

1. **MaxEnt control (utility + freedom):** $\pi^*$ maximizes $J_{T_c}(\pi)$ from Definition 10.2.1.
2. **Exponentially tilted trajectory measure (KL-regularization).** Fix a reference (prior) policy $\pi_0(a\mid k)$ with full support (uniform when $\mathcal{A}$ is finite). For the finite-horizon trajectory

   $$
   \omega := (A_t,\dots,A_{t+H-1},K_{t+1},\dots,K_{t+H}),
   $$
   the optimal controlled path law admits an exponential-family form relative to the reference measure induced by $\pi_0$ and $\bar{P}$:

   $$
   P^*(\omega\mid K_t=k)\ \propto\
   \Big[\prod_{h=0}^{H-1}\pi_0(A_{t+h}\mid K_{t+h})\,\bar{P}(K_{t+h+1}\mid K_{t+h},A_{t+h})\Big]\,
   \exp\!\left(\frac{1}{T_c}\sum_{h=0}^{H-1}\gamma^h\,\mathcal{R}(K_{t+h},A_{t+h})\right),
   $$
   where the normalizer is the (state-dependent) path-space normalizing constant.
3. **Soft Bellman optimality:** the optimal value function $V^*$ satisfies the soft Bellman recursion of Proposition 10.2.2, and $\pi^*$ is the corresponding softmax policy.

Moreover, the path-space log-normalizer is (up to scaling) the soft value. Gradients of the log-normalizer therefore induce a well-defined exploration direction in any differentiable macro coordinate system. The link between soft optimality and path entropy is cleanest when stated as a KL-regularized variational identity: if $P_0(\omega\mid k)$ denotes the reference trajectory measure induced by $\pi_0$ and $\bar{P}$, then

$$
\log Z(k)
=
\sup_{P(\cdot\mid k)}
\left\{
\frac{1}{T_c}\,\mathbb{E}_{P}\!\left[\sum_{h=0}^{H-1}\gamma^h\,\mathcal{R}\right]
-D_{\mathrm{KL}}(P(\cdot\mid k)\Vert P_0(\cdot\mid k))
\right\},
$$
and the optimizer is exactly the exponentially tilted law {math}`P^*`. In the special case where {math}`P_0` is uniform (or treated as constant), the KL term differs from Shannon path entropy by an additive constant, recovering the standard "maximize entropy subject to expected reward" view.

*Proof sketch.* Set up the constrained variational problem "maximize path entropy subject to an expected reward constraint." The Euler–Lagrange condition yields an exponential-family distribution on paths. The normalizer obeys dynamic programming and equals the soft value. Differentiating the log-normalizer yields the corresponding exploration-gradient direction.

:::

::::{note} Connection to RL #22: KL-Regularized Policies as Degenerate Exploration Duality
**The General Law (Fragile Agent):**
MaxEnt control is equivalent to an **Exponentially Tilted Trajectory Measure**:

$$
P^*(\omega|K_t=k) \propto P_0(\omega|k) \exp\!\left(\frac{1}{T_c}\sum_{h=0}^{H-1} \gamma^h \mathcal{R}(K_{t+h}, A_{t+h})\right)
$$
The path-space log-normalizer equals the soft value (Theorem {prf:ref}`thm-equivalence-of-entropy-regularized-control-forms-discrete-macro`). This is a **Schrödinger bridge** formulation.

**The Degenerate Limit:**
Use single-step KL penalty instead of path-space tilting. Ignore the trajectory structure.

**The Special Case (Standard RL):**

$$
J(\pi) = \mathbb{E}[R] - \lambda D_{\mathrm{KL}}(\pi \| \pi_0)
$$
This recovers **KL-Regularized Policy Gradient** and exponential family policies.

**What the generalization offers:**
- **Path-space view**: The optimal policy is a Schrödinger bridge between prior and reward-weighted measures
- **Trajectory entropy**: Explores future *macro-trajectories* $\omega = (A_t, \ldots, K_{t+H})$, not just single actions
- **Variational principle**: Soft value = log-partition function of trajectory measure (eq. above)
- **Causal entropy**: $S_c(k, H; \pi)$ measures future reachability under causal interventions
::::



(sec-implementation-note-entropy-regularized-optimal-transport-bridge)=
