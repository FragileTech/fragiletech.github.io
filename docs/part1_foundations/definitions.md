## 1. Introduction: The Agent as a Bounded-Rationality Controller

:::{admonition} Researcher Bridge: Bounded Rationality as a POMDP with Costs
:class: info
:name: rb-bounded-rationality
Standard RL frames the agent as a policy that maximizes return in a POMDP. Here we make the usual hidden constraints explicit: limited bandwidth, memory, and compute. Think of it as a POMDP with an information bottleneck and hard safety contracts that shape the feasible policy class.
:::

This document presents the **Fragile** interpretation of the Hypostructure: a deployed agent is a persistent **controller under partial observability** whose competence is bounded by (i) finite sensing/communication bandwidth, (ii) finite internal memory/representation capacity, and (iii) finite compute for inference and planning.

The framework is stated strictly in **information theory, optimization, and control**: discrete/continuous latent state construction (representation), stability constraints (Lyapunov-style), and capacity/sufficiency conditions (information bottlenecks).

This is the native language of **Safe RL**, **Robust Control**, and **Embodied AI**.

(sec-definitions-interaction-under-partial-observability)=
### 1.1 Definitions: Interaction Under Partial Observability

:::{admonition} Researcher Bridge: Markov Blanket = Observation/Action Interface
:class: tip
:name: rb-markov-blanket
If you are used to POMDP notation, the "boundary" here is just the observation, action, reward, and termination channels treated as a single interface. The environment is an input-output law, not a latent object the agent can access directly. This re-typing lets us attach geometric and information constraints to the interface itself.
:::

In the Fragile Agent framework, we do **not** treat the environment as a passive data provider. We treat the agent as a **partially observed control problem** whose only coupling to the external world is through a well-defined **interface / Markov blanket**. All RL primitives are re-typed as **signals and constraints at this interface**.

(sec-the-environment-is-an-input-output-law)=
#### 1.1.1 The Environment is an Input–Output Law (Not an Internal Object)

:::{prf:definition} Bounded-Rationality Controller
:label: def-bounded-rationality-controller

The agent is a controller with internal state

$$
Z_t := (K_t, Z_{n,t}, Z_{\mathrm{tex},t}) \in \mathcal{Z}=\mathcal{K}\times\mathcal{Z}_n\times\mathcal{Z}_{\mathrm{tex}},
$$
and internal components (Encoder/Shutter, World Model, Critic, Policy). Its evolution is driven only by the observable interaction stream at the interface (observations/feedback) and by its own outgoing control signals (actions).

:::
:::{prf:definition} Boundary / Markov Blanket
:label: def-boundary-markov-blanket

The boundary variables at time $t$ are the interface tuple

$$
B_t := (x_t,\ r_t,\ d_t,\ \iota_t,\ a_t),
$$
where:
- $x_t\in\mathcal{X}$ is the observation (input sample),
- $r_t\in\mathbb{R}$ is reward/utility (scalar feedback; equivalently negative instantaneous cost),
- $d_t\in\{0,1\}$ is termination (absorbing event / task boundary),
- $\iota_t$ denotes any additional side channels (costs, constraints, termination reasons, privileged signals),
- $a_t\in\mathcal{A}$ is action (control signal sent outward).

:::
:::{prf:definition} Environment as Generative Process
:label: def-environment-as-generative-process

The "environment" is the conditional law of future interface signals given past interface history. Concretely it is a (possibly history-dependent) kernel on incoming boundary signals conditional on outgoing control:

$$
P_{\partial}(x_{t+1}, r_t, d_t, \iota_{t+1}\mid x_{\le t}, a_{\le t}).
$$
In the Markov case this reduces to the familiar RL kernel

$$
P_{\partial}(x_{t+1}, r_t, d_t, \iota_{t+1}\mid x_t, a_t),
$$
but the **interpretation changes**: $P_{\partial}$ is not “a dataset generator”; it is the **input–output law** that the controller must cope with under partial observability and model mismatch.

This is the categorical move: we do not assume access to the environment’s latent variables; we work only with the **law over observable interface variables**.

:::
(sec-re-typing-standard-rl-primitives-as-interface-signals)=
#### 1.1.2 Re-typing Standard RL Primitives as Interface Signals

1. **Environment (Stochastic Process / Unmodeled Disturbance).**
   - *Standard:* a black box providing states and rewards.
   - *Fragile:* a high-dimensional, partially observed stochastic process; only its induced interface law $P_{\partial}$ is accessible to the agent.
   - *Role:* supplies observations and feedback signals; may be non-stationary, adversarial, or only approximately Markov.

2. **Observation $x_t$ (Observation Stream).**
   - *Standard:* an input tensor.
   - *Fragile:* the only exogenous input available to the controller. The encoder/shutter transduces it into internal coordinates:

     $$
     x_t \mapsto (K_t, Z_{n,t}, Z_{\mathrm{tex},t}),
     $$
     where $K_t$ is the **discrete predictive signal** (bounded-rate latent statistic), $Z_{n,t}$ is a **structured nuisance / gauge residual** (pose/basis/disturbance coordinates), and $Z_{\mathrm{tex},t}$ is a **texture residual** (high-rate reconstruction detail).
   - *Boundary gate nodes (Section 3):*
     - **Node 14 (InputSaturationCheck):** input saturation (sensor dynamic range exceeded).
     - **Node 15 (SNRCheck):** low signal-to-noise (SNR too low to support stable inference).
     - **Node 13 (BoundaryCheck):** the channel is open in the only well-typed sense: $I(X;K)>0$ (symbolic mutual information).

3. **Action $a_t$ (Control / Actuation).**
   - *Standard:* a vector sent to the environment.
   - *Fragile:* a control signal chosen to minimize expected future cost under uncertainty and constraints. Like observations, actions decompose into structured components: $a_t = (A_t, z_{n,\text{motor}}, z_{\text{tex,motor}})$ where $A_t$ is the discrete motor macro, $z_{n,\text{motor}}$ is motor nuisance (compliance), and $z_{\text{tex,motor}}$ is motor texture (tremor). See Section 23.3 for details.
   - *Cybernetic constraints:*
     - **Node 2 (ZenoCheck):** limits chattering (bounded variation in control outputs).
     - **BarrierSat:** actuator saturation (finite control authority).
   - *Boundary interpretation (Section 23.1):* Actions impose **Neumann boundary conditions** (clamping flux/momentum) on the agent's internal manifold, dual to the Dirichlet conditions imposed by sensors.

4. **Reward $r_t$ (Utility / Negative Cost Signal).**
   - *Standard:* a scalar to maximize.
   - *Fragile:* a scalar feedback signal used to define the control objective. In continuous-time derivations it appears as an instantaneous **cost rate**; in discrete time it appears as an incremental term in the Bellman/HJB consistency relation (Section 2.7).
   - *Mechanism:* the critic's $V$ is the internal value/cost-to-go; reward provides the task-aligned signal shaping $V$.
   - *Boundary interpretation (Section 24.1):* Reward is a **Scalar Charge Density** $\sigma_r$ on the boundary. The Critic solves the **Screened Poisson Equation** (Theorem {prf:ref}`thm-the-hjb-helmholtz-correspondence`) to propagate this boundary condition into the bulk, generating the potential field $V(z)$.

5. **Termination $d_t$ (Absorbing Boundary Event).**
   - *Standard:* end-of-episode flag.
   - *Fragile:* an absorbing event: the trajectory has entered a terminal region (failure/success) or exited the modeled domain. It is part of the task specification, not a training artifact.

6. **Episode / Rollout (Finite-Horizon Segment).**
   - *Standard:* a finite trajectory segment.
   - *Fragile:* a finite window used to estimate a cumulative objective under uncertainty. “Success” is satisfying task constraints while maintaining stability; “failure” is crossing a monitored limit (Section 4).

(sec-symmetries-and-gauge-freedoms)=
#### 1.1.4 Symmetries and Gauge Freedoms (Operational)

Many of the largest stability and sample-efficiency failures in practice come from **wasting capacity on nuisance degrees of freedom**: the agent learns separate internal states for observations that differ only by pose, basis choice, or arbitrary internal labeling. We formalize these nuisance directions as **symmetries** (group actions) and treat “quotienting them out” as an explicit design constraint.

We use “gauge” in the minimal, operational sense: a **gauge transformation** is a change of coordinates or representation that should not change the agent’s control-relevant decisions, except through explicitly modeled nuisance variables.

:::{prf:definition} Agent symmetry group; operational
:label: def-agent-symmetry-group-operational

Let:
- $G_{\text{obj}}$ be an **objective/feedback gauge** acting on scalar feedback signals (e.g., change of units or baseline shift). A common choice is the positive affine group

  $$
  G_{\text{obj}} := \{(a,b): a>0,\ r\mapsto ar+b\}.
  $$
  (If representing value as a unit-norm phase variable, one may instead use $U(1)$; Section 3.3.C treats the real-valued case via projective heads.)
- $G_{\text{spatial}}$ be an **observation gauge** acting on raw observations $x$ (e.g., pose/translation/rotation; choose $SE(3)$, $SE(2)$, $\mathrm{Sim}(2)$, or a task-specific subgroup depending on sensors).
- $S_{|\mathcal{K}|}$ be the **symbol-permutation symmetry** of the discrete macro register: relabeling code indices is unobservable if downstream components depend only on embeddings $\{e_k\}$.
- $\mathrm{Symp}(2n,\mathbb{R})$ be an optional **phase-space symmetry** acting on canonical latent coordinates $z=(q,p)\in\mathbb{R}^{2n}$ when the world model is parameterized as a symplectic/Hamiltonian system (Section 3.3.B).

The (candidate) total symmetry group is the direct product

$$
\mathcal{G}_{\mathbb{A}}
:=
G_{\text{obj}}
\times
G_{\text{spatial}}
\times
S_{|\mathcal{K}|}
\times
\mathrm{Symp}(2n,\mathbb{R}).
$$
**Internal vs. external symmetries.**
- **Internal (objective) gauge:** transformations of the scalar feedback scale/offset (and any potentials) that should not qualitatively change the policy update direction.
- **External (observation) gauge:** transformations of the input stream that change *pose* but not *identity*.

**Principle of covariance (engineering requirement).** The internal maps of the agent should be invariant/equivariant under $\mathcal{G}_{\mathbb{A}}$ in the following typed sense:
- **Shutter $E$**: canonicalize or quotient $G_{\text{spatial}}$ before discretization, so the macro register is approximately invariant:

  $$
  K(x)\approx K(g\cdot x)\quad (g\in G_{\text{spatial}}),
  $$
  while $z_n$ carries structured nuisance parameters (pose/basis/disturbance coordinates) and $z_{\mathrm{tex}}$ carries reconstruction-only texture (Section 2.2b, Section 3.3.A).
- **World model $S$ and policy $\pi$:** be covariant to symbol permutations $S_{|\mathcal{K}|}$ by treating $K$ only through its embedding $e_K$ (not the integer label) and by using permutation-invariant diagnostics.
- **Critic/value and dual variables:** enforce stability and constraint satisfaction in a way that is robust to re-scaling/offset of the scalar feedback (Section 3.3.C, Section 3.5).

These are *requirements on representations and interfaces*, not philosophical claims: if an invariance is not enforced, the corresponding failure modes (symmetry blindness, brittle scaling, uncontrolled drift) become more likely and harder to debug.

:::
(sec-units-and-dimensional-conventions)=
#### 1.2 Units and Dimensional Conventions (Explicit)

This document expresses objectives in **information units** so that likelihoods, code lengths, KL terms, and entropy regularizers share a common scale.

**Base units.**
- Interaction time is measured in **environment steps** ($t \in \mathbb{Z}_{\ge 0}$). If a physical clock is needed, introduce $\Delta t$ with $[\\Delta t]=\mathrm{s}$.
- Information: entropies/information/KL are in **nats** (dimensionless but tracked): $[H]=[S]=[I]=[D_{\mathrm{KL}}]=\mathrm{nat}$.
- Costs / values / losses (including $V$ and negative rewards) are measured in **nats**: $[V]=\mathrm{nat}$.
- Cost rates are measured in $\mathrm{nat/step}$ (or $\mathrm{nat\,s^{-1}}$ after dividing by $\Delta t$).

**Discrete vs continuous reward.**
- Per-step reward $r_t$ (or cost $c_t=-r_t$) has units $\mathrm{nat}$.
- A continuous-time cost rate $\mathcal{R}$ has units $\mathrm{nat\,s^{-1}}$ and links to discrete time by $r_t \approx \int_{t}^{t+\Delta t}\mathcal{R}(u)\,du$.

**Regularization / precision coefficients.**
- MaxEnt / entropy-regularized control introduces a trade-off coefficient (often written $T_c$ or $\alpha_{\text{ent}}$) multiplying an entropy term. Because entropy is in nats, this coefficient is dimensionless and simply sets relative weight in the objective.
- Exponential-family (softmax/logit) policies use a precision parameter $\beta$ so that $\exp(\beta\,\cdot)$ is dimensionless. Here $\beta$ is dimensionless and interpretable as an inverse-variance / “sharpness” control knob.

**Conventions for generic coefficients.**
- Numerical stabilizers like $\epsilon$ always inherit the units of the quantity they are added to.
- Composite-loss weights (e.g. $\lambda_{\text{*}}$ used to sum training losses) are taken dimensionless unless explicitly stated otherwise.

(sec-the-chronology-temporal-distinctions)=
#### 1.3 The Chronology: Temporal Distinctions

We distinguish four temporal dimensions. They are orthogonal (or nested) and must not be conflated. Using one symbol for all of them is a chronological category error (e.g., confusing "thinking longer" with "getting older").

| Symbol     | Name                 | Domain                           | Role                                              | Physics Analogy                  |
|:-----------|:---------------------|:---------------------------------|:--------------------------------------------------|:---------------------------------|
| **$t$**    | **Interaction Time** | $\mathbb{Z}_{\ge 0}$             | External environment clock ($x_t, a_t$).          | Coordinate time (observer clock) |
| **$s$**    | **Computation Time** | $\mathbb{R}_{\ge 0}$             | Internal solver time for belief/planning updates. | Proper time (agent thinking)     |
| **$\tau$** | **Scale Time**       | $\mathbb{R}_{\ge 0}$             | Resolution depth (root to leaf).                  | Renormalization scale            |
| **$t'$**   | **Memory Time**      | $\{t' \in \mathbb{Z} : t' < t\}$ | Index of stored past states on the screen.        | Retarded time                    |

(sec-interaction-time-the-discrete-clock)=
##### 1.3.1 Interaction Time ($t$): The Discrete Clock
This is the Markov Decision Process index imposed by the environment.
- **Update:** $z_t \to z_{t+1}$.
- **Constraint:** the agent must emit $a_t$ before $t$ increments (real-time constraint).

(sec-computation-time-the-continuous-thought)=
##### 1.3.2 Computation Time ($s$): The Continuous Thought
This is the integration variable of the internal solver and the Equation of Motion (Section 22). It represents the agent's "thinking" process:

$$
\frac{dz}{ds} = -G^{-1}\nabla \Phi_{\text{eff}} + \dots
$$
- **Relationship to $t$:** to transition from $t$ to $t+1$, the agent integrates its internal dynamics from $s=0$ to $s=S_{\text{budget}}$.
- **Thinking fast vs. slow:** small $S_{\text{budget}}$ yields reflexive action; large $S_{\text{budget}}$ yields deliberate planning.
- **Thermodynamics:** this is the time variable in which Fokker-Planck dynamics evolve internal belief toward equilibrium (Section 22.5).

(sec-scale-time-the-holographic-depth)=
##### 1.3.3 Scale Time ($\tau$): The Holographic Depth
This is the radial coordinate in the Poincare disk (Sections 21, 7.12). It corresponds to resolution depth.
- **Dynamics:** $dr/d\tau = \operatorname{sech}^2(\tau/2)$ (the holographic law).
- **Discretization:** in stacked TopoEncoders, layer $\ell$ corresponds to scale time $\tau_\ell$.
- **Direction:** $\tau \to \infty$ (UV) is high energy, fine detail; $\tau \to 0$ (IR) is low energy, coarse structure.
- **Process:** generation flows in $+\tau$ (root to boundary); inference flows in $-\tau$.

(sec-memory-time-the-historical-coordinate)=
##### 1.3.4 Memory Time ($t'$): The Historical Coordinate
This is the time coordinate of the Holographic Screen.
- **Structure:** the screen stores tuples $(z_{t'}, a_{t'}, r_{t'})$ at past indices.
- **Access:** attention computes distances between the current state $z_t$ and stored states $z_{t'}$.
- **Causality:** we enforce $t' < t$ (no access to the future).



(sec-the-control-loop-representation-and-control)=
