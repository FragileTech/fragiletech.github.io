## 23. The Boundary Interface: Symplectic Structure

{cite}`arnold1989mathematical`

:::{admonition} Researcher Bridge: Observations and Actions as Boundary Conditions
:class: info
:name: rb-boundary-conditions
In standard RL, observations and actions are inputs and outputs. Here they are boundary conditions on the latent dynamics, which is why sensor and motor channels appear as Dirichlet and Neumann conditions.
:::

We have defined the internal dynamics of the agent (the interior) as a Jump-Diffusion process on a Riemannian fiber bundle (Sections 20-22). We now rigorously define its coupling to the external world.

The interface exchanges information with the environment via two asymmetric boundary conditions: **Dirichlet** (position-clamping for sensors) and **Neumann** (flux-clamping for motors).

(sec-the-symplectic-interface-position-momentum-duality)=
### 23.1 The Symplectic Interface: Position-Momentum Duality

The boundary $\partial\mathcal{Z}$ between agent and environment is not merely a surface—it is a **symplectic manifold** where observations and actions live as conjugate variables.

:::{prf:definition} Symplectic Boundary Manifold
:label: def-symplectic-boundary-manifold

The agent's interface is a symplectic manifold $(\partial\mathcal{Z}, \omega)$ with canonical coordinates $(q, p) \in T^*\mathcal{M}$ where:
- $q \in \mathcal{Q}$ is the **position bundle** (sensory configuration)
- $p \in T^*_q\mathcal{Q}$ is the **momentum bundle** (motor flux)

The symplectic form is:

$$
\omega = \sum_{i=1}^n dq^i \wedge dp_i.
$$
Units: $[\omega] = [q][p] = \mathrm{nat}$.

*Remark (Causal Structure).* The symplectic structure encodes causality: observations fix "where" the belief state is (position), while actions fix "how" it flows outward (momentum/flux). These cannot be treated symmetrically as static fields.

:::
:::{prf:definition} Dirichlet Boundary Condition — Sensors
:label: def-dirichlet-boundary-condition-sensors

The sensory input stream $\phi(x)$ imposes a **Dirichlet** (position-clamping) condition on the belief density:

$$
\rho_{\partial}^{\text{sense}}(q, t) = \delta(q - q_{\text{obs}}(t)),
$$
where $q_{\text{obs}}(t) = E_\phi(x_t)$ is the encoded observation. This clamps the *configuration* of the belief state.

*Interpretation:* Information flow from environment to agent (observation).

:::
:::{prf:definition} Neumann Boundary Condition — Motors
:label: def-neumann-boundary-condition-motors

The motor output stream $A(x)$ imposes a **Neumann** (flux-clamping) condition:

$$
\nabla_n \rho \cdot \mathbf{n} \big|_{\partial\mathcal{Z}_{\text{motor}}} = j_{\text{motor}}(p, t),
$$
where $j_{\text{motor}}$ is the motor current density determined by the policy:

$$
j_{\text{motor}} = D_A(u_\pi) = \text{Decoder}(z, u_\pi, z_{\text{tex,motor}}).
$$
*Interpretation:* Information flow from agent to environment (action).

Units: $[j_{\text{motor}}] = \mathrm{nat}/\text{step}$.

:::

::::{admonition} Physics Isomorphism: Hamiltonian Boundary Conditions
:class: note
:name: pi-hamiltonian-bc

**In Physics:** In Hamiltonian mechanics, canonical coordinates $(q, p)$ satisfy $\dot{q} = \partial H/\partial p$ (position from momentum) and $\dot{p} = -\partial H/\partial q$ (momentum from position). Boundary conditions fix either $q$ (Dirichlet) or $\partial_n q \propto p$ (Neumann) {cite}`arnold1989mathematical`.

**In Implementation:** The agent's interface imposes dual boundary conditions:
- **Perception (Dirichlet):** Observations fix position $z|_{\partial\mathcal{Z}} = z_{\text{obs}}$
- **Action (Neumann):** Motors fix momentum flux $\partial_n z|_{\partial\mathcal{Z}} = u_\pi$
- **Reward (Source):** Scalar charge injection $\sigma_r|_{\partial\mathcal{Z}}$

**Correspondence Table:**

| Hamiltonian Mechanics | Agent (Symplectic Interface) |
|:----------------------|:-----------------------------|
| Position $q$ | Latent state $z$ |
| Momentum $p$ | Policy gradient $\nabla_z V$ |
| Dirichlet BC $q\vert_{\Gamma} = q_0$ | Observation constraint |
| Neumann BC $\partial_n q\vert_{\Gamma} = f$ | Action constraint |
| Hamiltonian $H(q,p)$ | Effective potential $\Phi_{\text{eff}}$ |
::::

:::{prf:proposition} Symplectic Duality Principle
:label: prop-symplectic-duality-principle

Under the canonical transformation $(q, p) \mapsto (p, -q)$:
- Dirichlet conditions become Neumann conditions
- Sensors become motors
- Perception becomes action

This duality is the mathematical foundation for the symmetric treatment of sensing and actuation.

*Proof sketch.* The symplectic form $\omega$ is invariant under canonical transformations. The Legendre transform $\mathcal{L}: T\mathcal{Q} \to T^*\mathcal{Q}$ maps velocity to momentum, exchanging position-fixing (Dirichlet) for flux-fixing (Neumann). $\square$

**Cross-references:** Section 2.11.4 (Observation inflow), Definition {prf:ref}`def-dirichlet-boundary-condition-sensors`.

:::
(sec-the-dual-atlas-architecture)=
### 23.2 The Dual Atlas Architecture

To implement the symplectic interface, we require two symmetric topological structures: a **Visual Atlas** for perception and an **Action Atlas** for actuation. This symmetrizes the architecture from Section 2.2b.

:::{prf:definition} Visual Atlas — Perception
:label: def-visual-atlas-perception

The Visual Atlas $\mathcal{A}_{\text{vis}} = \{(U_\alpha, \phi_\alpha, e_\alpha^{\text{vis}})\}_{\alpha \in \mathcal{K}_{\text{vis}}}$ is a chart atlas on the sensory manifold $\mathcal{Q}$ with:
- **Charts** $U_\alpha \subset \mathcal{Q}$: Objects, Scenes, Viewpoints
- **Chart maps** $\phi_\alpha: U_\alpha \to \mathbb{R}^{d_{\text{vis}}}$: Local coordinates
- **Codebook embeddings** $e_\alpha^{\text{vis}} \in \mathbb{R}^{d_m}$: Discrete macro codes

*Input:* Raw observations $\phi_{\text{raw}}$ (pixels, sensors).
*Output:* Latent state $z \in \mathcal{Z}$ (configuration).

:::
:::{prf:definition} Action Atlas — Actuation
:label: def-action-atlas-actuation

The Action Atlas $\mathcal{A}_{\text{act}} = \{(V_\beta, \psi_\beta, e_\beta^{\text{act}})\}_{\beta \in \mathcal{K}_{\text{act}}}$ is a chart atlas on the motor manifold $T^*\mathcal{Q}$ with:
- **Charts** $V_\beta \subset T^*\mathcal{Q}$: Gaits, Grasps, Tool Affordances (topologically distinct control regimes)
- **Chart maps** $\psi_\beta: V_\beta \to \mathbb{R}^{d_{\text{act}}}$: Local motor coordinates
- **Codebook embeddings** $e_\beta^{\text{act}} \in \mathbb{R}^{d_m}$: Action primitive codes

*Input:* Intention $u_{\text{intent}} \in T_z\mathcal{Z}$ (from Policy, Section 21.2).
*Output:* Actuation $a_{\text{raw}}$ (torques, voltages).

*Remark (Jump Operator in Action Atlas).* The **Jump Operator** $L_{\beta \to \beta'}$ in the Action Atlas represents **Task Switching**: transitioning from one control primitive to another (e.g., "Walk" $\to$ "Jump", "Grasp" $\to$ "Release"). This mirrors the chart transition operator in the Visual Atlas (Section 20.6).

:::
:::{prf:theorem} Atlas Duality via Legendre Transform
:label: thm-atlas-duality-via-legendre-transform

The Visual and Action Atlases are related by the Legendre transform $\mathcal{L}: T\mathcal{Q} \to T^*\mathcal{Q}$:

$$
\mathcal{A}_{\text{act}} = \mathcal{L}(\mathcal{A}_{\text{vis}}),
$$
where the chart transition functions satisfy:

$$
\psi_\beta \circ \mathcal{L} \circ \phi_\alpha^{-1} = \nabla_{\dot{q}} L(q, \dot{q})
$$
for Lagrangian $L(q, \dot{q}) = \frac{1}{2}\|\dot{q}\|_G^2 - V(q)$.

*Proof.* **Step 1 (Legendre transform definition).** The Legendre transform of a convex Lagrangian $L(q,\dot{q})$ is defined by:

$$
\mathcal{L}: T\mathcal{Q} \to T^*\mathcal{Q}, \qquad (q, \dot{q}) \mapsto \left(q, \frac{\partial L}{\partial \dot{q}}\right).
$$
For $L = \frac{1}{2}\|\dot{q}\|_G^2 - V(q)$, this gives $p = G(q)\dot{q}$, which is invertible when $G > 0$.

**Step 2 (Symplectic preservation).** The Legendre transform is a diffeomorphism that pulls back the canonical symplectic form $\omega_{T^*\mathcal{Q}} = dp \wedge dq$ to the Poincare-Cartan form $\omega_{T\mathcal{Q}} = d\theta_L$ where $\theta_L = \frac{\partial L}{\partial \dot{q}^i}dq^i$. This ensures that Hamiltonian flow on $T^*\mathcal{Q}$ corresponds to Lagrangian flow on $T\mathcal{Q}$.

**Step 3 (Chart compatibility).** Let $(U_\alpha, \phi_\alpha)$ be a chart in $\mathcal{A}_{\text{vis}}$ with coordinates $(q^\alpha, \dot{q}^\alpha)$. Define the induced action chart $(V_\beta, \psi_\beta)$ by $V_\beta = \mathcal{L}(U_\alpha \times T_{U_\alpha}\mathcal{Q})$ with coordinates $(q^\alpha, p^\alpha)$. The transition function is:

$$
\psi_\beta \circ \mathcal{L} \circ \phi_\alpha^{-1}: (q^\alpha, \dot{q}^\alpha) \mapsto (q^\alpha, G_{\alpha\beta}(q)\dot{q}^\beta),
$$
which is smooth and invertible by positive-definiteness of $G$. $\square$

*Remark (Why Legendre?).* The Legendre transform is the unique smooth map relating configuration-velocity (perception) to configuration-momentum (action) that:
1. Preserves the symplectic structure (Proposition {prf:ref}`prop-symplectic-duality-principle`)
2. Interchanges Dirichlet and Neumann boundary conditions (Section 23.1)
3. Maps kinetic energy to Hamiltonian dynamics

*Cross-reference:* The metric $G$ appearing here is the capacity-constrained metric from Theorem {prf:ref}`thm-capacity-constrained-metric-law`, ensuring that the "mass" in the Legendre relation $p = G\dot{q}$ is the same "mass" that determines geodesic inertia (Definition {prf:ref}`def-mass-tensor`).

:::

::::{admonition} Physics Isomorphism: Legendre Transform
:class: note
:name: pi-legendre-transform

**In Physics:** The Legendre transform maps between Lagrangian and Hamiltonian formulations: $H(q,p) = p\dot{q} - L(q,\dot{q})$ where $p = \partial L/\partial \dot{q}$. It exchanges velocity for momentum as the independent variable {cite}`arnold1989mathematical`.

**In Implementation:** The Visual and Action Atlases are related by Legendre duality (Theorem {prf:ref}`thm-atlas-duality-via-legendre-transform`):

$$
\mathcal{L}: T\mathcal{Q} \to T^*\mathcal{Q}, \quad (z, \dot{z}) \mapsto (z, p = G\dot{z})
$$
**Correspondence Table:**
| Analytical Mechanics | Agent (Symplectic Interface) |
|:---------------------|:-----------------------------|
| Configuration space $\mathcal{Q}$ | Latent state space $\mathcal{Z}$ |
| Tangent bundle $T\mathcal{Q}$ | Velocity representation |
| Cotangent bundle $T^*\mathcal{Q}$ | Momentum representation |
| Lagrangian $L(q,\dot{q})$ | Kinetic action |
| Hamiltonian $H(q,p)$ | Effective potential $\Phi_{\text{eff}}$ |
| Velocity $\dot{q}$ | Policy output $u_\pi$ |
| Momentum $p$ | Value gradient $\nabla V$ |

**Duality:** Perception (Dirichlet) fixes position; Action (Neumann) fixes momentum flux.
::::

:::{prf:definition} The Holographic Shutter — Unified Interface
:label: def-the-holographic-shutter-unified-interface

The Shutter is extended from Section 2.2b to a symmetric tuple:

$$
\mathbb{S} = (\mathcal{A}_{\text{vis}}, \mathcal{A}_{\text{act}}),
$$
where:
- **Ingress (Perception):** $E_\phi: \mathcal{Q} \to \mathcal{Z}$ via Visual Atlas
- **Egress (Actuation):** $D_A: T_z\mathcal{Z} \times \mathcal{Z} \to T^*\mathcal{Q}$ via Action Atlas
- **Proprioception (Inverse Model):** $E_A: T^*\mathcal{Q} \to T_z\mathcal{Z}$ maps realized actions back to intentions

**Cross-references:** Section 2.2b (VQ-VAE Shutter), Section 7.8 (AttentiveAtlasEncoder), Section 7.10 (TopologicalDecoder).

:::
(sec-motor-texture-the-action-residual)=
### 23.3 Motor Texture: The Action Residual

Just as visual texture captures reconstruction-only detail (Section 21.3), **motor texture** captures actuation-only detail that is excluded from planning.

:::{prf:definition} Motor Texture Decomposition
:label: def-motor-texture-decomposition

The motor output decomposes as:

$$
a_t = (A_t, z_{n,\text{motor}}, z_{\text{tex,motor}}),
$$
where:
- $A_t \in \mathcal{K}_{\text{act}}$ is the **discrete motor macro** (action primitive/chart index)
- $z_{n,\text{motor}} \in \mathbb{R}^{d_{\text{motor},n}}$ is **motor nuisance** (impedance, compliance, force distribution)
- $z_{\text{tex,motor}} \in \mathbb{R}^{d_{\text{motor,tex}}}$ is **motor texture** (tremor, fine-grained noise, micro-corrections)

*Remark (Parallel to Visual Decomposition).* This mirrors the visual decomposition $(K_t, z_{n,t}, z_{\text{tex},t})$ from Section 2.2b:

| Component                 | Visual Domain                 | Motor Domain                              |
|---------------------------|-------------------------------|-------------------------------------------|
| **Macro (discrete)**      | Object/Scene chart $K$        | Action primitive $A$                      |
| **Nuisance (continuous)** | Pose/viewpoint $z_n$          | Compliance/impedance $z_{n,\text{motor}}$ |
| **Texture (residual)**    | Pixel detail $z_{\text{tex}}$ | Tremor/noise $z_{\text{tex,motor}}$       |

:::
:::{prf:definition} Compliance Tensor
:label: def-compliance-tensor

The motor nuisance encodes the **compliance tensor**:

$$
C_{ij}(z_{n,\text{motor}}) = \frac{\partial a^i}{\partial f^j},
$$
where $f$ is the external force/feedback. This determines how the motor output responds to perturbations:
- **High compliance** ($C$ large): Soft, yielding response (safe interaction)
- **Low compliance** ($C$ small): Stiff, precise response (accurate positioning)

Units: $[C_{ij}] = [a]/[f]$.

:::
:::{prf:definition} Motor Texture Distribution
:label: def-motor-texture-distribution

At the motor boundary, texture is sampled from a geometry-dependent Gaussian:

$$
z_{\text{tex,motor}} \sim \mathcal{N}(0, \Sigma_{\text{motor}}(z)),
$$
where:

$$
\Sigma_{\text{motor}}(z) = \sigma_{\text{motor}}^2 \cdot G_{\text{motor}}^{-1}(z) = \sigma_{\text{motor}}^2 \cdot \frac{(1-|z|^2)^2}{4} I_{d_{\text{motor,tex}}}.
$$
This follows the same conformal scaling as visual texture (Definition {prf:ref}`def-boundary-texture-distribution`), ensuring consistent thermodynamic behavior.

:::
:::{prf:axiom} Motor Texture Firewall
:label: ax-motor-texture-firewall

Motor texture is decoupled from the Bulk dynamics:

$$
\partial_{z_{\text{tex,motor}}} \dot{z} = 0, \qquad \partial_{z_{\text{tex,motor}}} u_\pi = 0.
$$
The policy $\pi_\theta$ operates on $(K, z_n, A, z_{n,\text{motor}})$ but **never** on $(z_{\text{tex}}, z_{\text{tex,motor}})$.

*Remark (Sim-to-Real Gap).* The **motor texture variance** $\sigma_{\text{motor}}^2$ is the mathematical definition of the "Sim-to-Real gap":
- **Simulation:** $\sigma_{\text{motor}} \approx 0$ (deterministic, no tremor)
- **Reality:** $\sigma_{\text{motor}} > 0$ (friction, sensor noise, motor tremor)
- **Robustness:** The Bulk policy $u_\pi$ is invariant; only the Action Decoder learns to manage domain-specific noise.

**Cross-references:** Section 21.3 (Texture Firewall), Axiom {prf:ref}`ax-bulk-boundary-decoupling`.

:::
(sec-the-belief-evolution-cycle-perception-dreaming-action)=
### 23.4 The Belief Evolution Cycle: Perception–Dreaming–Action

The agent's interaction loop is a **belief density evolution cycle** on the information manifold.

:::{prf:definition} Cycle Phases
:label: def-cycle-phases


| Phase             | Process            | Information Flow                      | Entropy Change               |
|-------------------|--------------------|---------------------------------------|------------------------------|
| **I. Perception** | Compression        | Mutual information $I(X;K)$ extracted | $\Delta S_{\text{bulk}} < 0$ |
| **II. Dreaming**  | Internal evolution | No external exchange                  | $\Delta S = 0$ (isentropic)  |
| **III. Action**   | Expansion          | Mutual information $I(A;K)$ injected  | $\Delta S_{\text{bulk}} > 0$ |

*Remark (Statistical mechanics analogy).* This cycle is structurally analogous to a Stirling cycle in thermodynamics.

:::
:::{prf:theorem} Perception as Compression
:label: thm-perception-as-compression

During perception, the agent compresses external entropy into internal free energy:

$$
W_{\text{compress}} = T_c \cdot I(X_t; K_t) \geq 0,
$$
where $I(X_t; K_t)$ is the mutual information extracted from the observation $X_t$ into the macro-state $K_t$.

*Mechanism:* The Visual Encoder $E_\phi$ compresses high-entropy raw data $\phi_{\text{raw}}$ into a low-entropy macro-state $z$. The "heat" absorbed is the raw sensory stream.

*Information-theoretic interpretation:* Entropy decreases ($\Delta S < 0$). The Information Bottleneck cost bounds the compression.

:::
:::{prf:theorem} Action as Expansion
:label: thm-action-as-expansion

During action, the agent expands internal free energy into external control:

$$
W_{\text{expand}} = T_c \cdot I(A_t; K_t) \geq 0,
$$
where $I(A_t; K_t)$ is the mutual information injected from the intention into the motor output.

*Mechanism:* The Action Decoder $D_A$ "expands" the low-entropy Intention $u_\pi$ into high-dimensional motor commands $a_{\text{raw}}$, injecting motor texture.

*Information-theoretic interpretation:* Entropy increases ($\Delta S > 0$). The agent injects stochastic texture into motor outputs.

:::
:::{prf:definition} Dreaming as Unitary Evolution
:label: def-dreaming-as-unitary-evolution

In the dreaming phase, the internal dynamics are approximately unitary (energy-conserving):

$$
\partial_s \rho + [H_{\text{internal}}, \rho]_{\text{Poisson}} = 0,
$$
where $H_{\text{internal}}$ is the effective Hamiltonian:

$$
H_{\text{internal}}(z, p) = \frac{1}{2}\|p\|_{G^{-1}}^2 + V_{\text{critic}}(z).
$$
*Mechanism:* The agent is decoupled from the boundary (adiabatic/isolated). The Bulk evolves under Hamiltonian dynamics (BAOAB integrator with $\gamma \to 0$).

*Information-theoretic interpretation:* Isentropic ($\Delta S = 0$). Internal planning proceeds without information exchange with the environment.

:::
:::{prf:proposition} Carnot Efficiency Bound
:label: prop-carnot-efficiency-bound

The agent's efficiency in converting sensory information to control information is bounded:

$$
\eta = \frac{I(A_t; K_t)}{I(X_t; K_t)} \leq 1 - \frac{T_{\text{motor}}}{T_{\text{sensor}}},
$$
where $T_{\text{sensor}}$ and $T_{\text{motor}}$ are the effective temperatures at the sensory and motor boundaries.

*Interpretation:* Perfect efficiency ($\eta = 1$) requires $T_{\text{motor}} = 0$ (deterministic motors) or $T_{\text{sensor}} \to \infty$ (infinite sensory entropy). Real systems operate at $\eta < 1$.

**Cross-references:** Section 22.7 (Adaptive Thermodynamics), Section 14.2 (MaxEnt Control).

*Forward reference (Reward as Heat).* Section 24.3 establishes that Reward is the thermodynamic **heat input** that drives the cycle: the Boltzmann-Value Law (Axiom {prf:ref}`ax-the-boltzmann-value-law`) identifies $V(z) = E(z) - T_c S(z)$ as Gibbs Free Energy, and Theorem {prf:ref}`thm-wfr-consistency-value-creates-mass` proves that WFR dynamics materialize the agent in high-value regions ("Value Creates Mass").

:::
(sec-wfr-boundary-conditions-waking-vs-dreaming)=
### 23.5 WFR Boundary Conditions: Waking vs Dreaming

The **Wasserstein-Fisher-Rao** (WFR) equation from Section 20 governs the belief density $\rho$. The distinction between Waking and Dreaming is rigorously defined by the **boundary condition** on $\rho$. Boundary conditions update at interaction time $t$, while internal flow evolves in computation time $s$ (Section 1.3).

:::{prf:definition} Waking: Boundary Clamping
:label: def-waking-boundary-clamping

During waking ($u_\pi \neq 0$), the sensory stream creates a high-mass source at the encoded location:

$$
\rho_{\partial}^{\text{sense}}(z, t) = \delta(z - z_{\text{obs}}(t)) \quad \text{(Dirichlet)},
$$
and the motor stream creates a flux sink:

$$
\nabla_n \rho \cdot \mathbf{n} = j_{\text{motor}}(u_\pi) \quad \text{(Neumann)}.
$$
The internal belief $\rho_{\text{bulk}}$ evolves to minimize the **WFR Geodesic Distance** to $\rho_{\partial}$:
- **Small Error** ($d_{\text{WFR}} < \lambda$): Transport dominates ($v$ term). The agent smoothly tracks the observation.
- **Large Error** ($d_{\text{WFR}} > \lambda$): Reaction dominates ($r$ term). The agent "teleports" (Surprise/Saccade) to the new reality via chart jump.

:::
:::{prf:definition} Dreaming: Reflective Boundary
:label: def-dreaming-reflective-boundary

During dreaming ($u_\pi = 0$), the sensory stream is cut. The boundary condition becomes **Reflective**:

$$
\nabla_n \rho \cdot \mathbf{n} = 0 \quad \text{(Reflective/Neumann-zero)}.
$$
The system is closed:
- Total mass is conserved: $\int_{\mathcal{Z}} \rho\, r\, d\mu_G = 0$
- Dynamics are driven purely by the internal potential $V_{\text{critic}}(z)$
- No information enters or leaves the boundary

:::
:::{prf:theorem} WFR Mode Switching
:label: thm-wfr-mode-switching

The transition from waking to dreaming corresponds to a **boundary condition phase transition**:

| Mode         | Sensory BC                             | Motor BC             | Internal Flow | Information Balance       |
|--------------|----------------------------------------|----------------------|---------------|---------------------------|
| **Waking**   | Dirichlet ($\delta$-clamp)             | Neumann (flux-clamp) | Source-driven | $\oint j_{\text{in}} > 0$ |
| **Dreaming** | Reflective ($\nabla \rho \cdot n = 0$) | Reflective           | Recirculating | $\oint j = 0$             |

:::
:::{prf:proposition} Grounding Rate via Boundary Flux
:label: prop-grounding-rate-via-boundary-flux

The grounding rate (cf. Definition 16.1.1) is:

$$
G_t = \oint_{\partial\mathcal{Z}_{\text{sense}}} j_{\text{obs}} \cdot dA - \oint_{\partial\mathcal{Z}_{\text{motor}}} j_{\text{motor}} \cdot dA,
$$
which is:
- **Positive** during waking (net information inflow from sensors)
- **Zero** during dreaming (closed system)
- **Negative** during pure actuation (net information outflow to motors)

**Cross-references:** Section 20.2 (WFR Action), Section 20.6 (WFR World Model), Section 2.11.4 (Observation Inflow).

:::
(sec-the-context-space-unified-definition)=
### 23.6 The Context Space: Unified Definition

The Action Atlas admits a deeper structure: the **Context Space** $\mathcal{C}$ is the abstract space of boundary conditions that unifies RL actions, classification labels, and LLM prompts.

:::{prf:definition} Context Space
:label: def-context-space

The **Context Space** $\mathcal{C}$ is a manifold parameterizing the control/conditioning signal for the agent:

$$
\mathcal{C} := \{c : c \text{ specifies a boundary condition on } \partial\mathcal{Z}\}.
$$
The context determines the target distribution at the motor boundary via the effective potential:

$$
\pi(a | z, c) \propto \exp\left(-\frac{1}{T_c} \Phi_{\text{eff}}(z, K, c)\right).
$$
Units: $[\mathcal{C}]$ inherits from the task domain.

:::
:::{prf:definition} Context Instantiation Functor
:label: def-context-instantiation-functor

The Context Space admits a functor $\mathcal{I}: \mathbf{Task} \to \mathcal{C}$ with three canonical instantiations:

| Task Domain        | Context $c \in \mathcal{C}$ | Motor Output $a$           | Effective Potential $\Phi_{\text{eff}}$      |
|--------------------|-----------------------------|----------------------------|----------------------------------------------|
| **RL**             | Action space $\mathcal{A}$  | Motor command (torques)    | $V_{\text{critic}}(z, K)$                    |
| **Classification** | Label space $\mathcal{Y}$   | Class prediction $\hat{y}$ | $-\log p(y\mid z)$ (cross-entropy)           |
| **LLM**            | Prompt space $\mathcal{P}$  | Token sequence             | $-\log p(\text{token}\mid z, \text{prompt})$ |

*Key Insight:* In all cases, the context $c$ functions as the **symmetry-breaking boundary condition** that determines which direction the holographic expansion takes at the origin.

:::
:::{prf:theorem} Universal Context Structure
:label: thm-universal-context-structure

All context instantiations share the same geometric structure:

1. **Embedding:** $c \mapsto e_c \in \mathbb{R}^{d_c}$ maps the context to a latent vector
2. **Symmetry-Breaking Kick:** $e_c$ determines the initial control field:

   $$
   u_\pi(0) = G^{-1}(0) \cdot e_c = \frac{1}{4} e_c
   $$
   (at the Poincare disk origin where $G(0) = 4I$)
3. **Motor Distribution:** The output distribution is:

   $$
   \pi(a | z, c) = \text{softmax}\left(-\frac{\Phi_{\text{eff}}(z, K, c)}{T_c}\right)
   $$
*Proof.* The holographic expansion (Section 21) is invariant to the interpretation of the control field $u_\pi$. Whether $u_\pi$ encodes "go left" (RL), "class = cat" (classification), or "continue with tone = formal" (LLM), the bulk dynamics follow the same geodesic SDE (Section 22). The interpretation is purely a boundary condition. $\square$

:::
:::{prf:definition} Context-Conditioned WFR
:label: def-context-conditioned-wfr

The WFR dynamics (Section 20.2) generalize to context-conditioned form:

$$
\partial_s \rho + \nabla \cdot (\rho\, v_c) = \rho\, r_c,
$$
where:
- $v_c(z) = -G^{-1}(z) \nabla_z \Phi_{\text{eff}}(z, K, c) + u_\pi(z, c)$ is the context-conditioned velocity
- $r_c(z)$ is the context-conditioned reaction rate (chart jumps influenced by context)

:::
:::{prf:corollary} Prompt = Action = Label
:label: cor-prompt-action-label

The following are isomorphic as boundary conditions on $\partial\mathcal{Z}$:

$$
\text{RL Action} \;\cong\; \text{Classification Label} \;\cong\; \text{LLM Prompt}.
$$
Each specifies:
1. **Which chart** to route to (discrete macro $K$ or $A$)
2. **Where in the chart** to aim (continuous nuisance $z_n$ or $z_{n,\text{motor}}$)
3. **What texture** to inject (visual or motor texture)

*Remark (Unified Training Objective).* This isomorphism enables transfer learning across task domains: an agent trained on RL can be fine-tuned for classification by reinterpreting the action space as label space, with the same holographic dynamics.

**Cross-references:** Section 21.2 (Control Field), Theorem {prf:ref}`thm-unified-control-interpretation`, Definition {prf:ref}`def-effective-potential`.

*Forward reference (Effective Potential Resolution).* Section 24.2 resolves the meaning of $\Phi_{\text{eff}} = V_{\text{critic}}$: the Critic solves the **Screened Poisson Equation** to compute the potential from boundary reward charges. The discount factor $\gamma$ determines the screening length $\ell = -1/\ln\gamma$ (Corollary {prf:ref}`cor-discount-as-screening-length`), explaining why distant rewards are exponentially suppressed in policy.

:::
(sec-implementation-the-holographicinterface-module)=
### 23.7 Implementation: The HolographicInterface Module

We provide the Python implementation of the Holographic Interface, combining the Dual Atlas, Motor Texture, and Context Space.

**Algorithm 23.7.1 (HolographicInterface Module).**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Literal, Optional, Dict, Tuple
from enum import Enum


class BoundaryConditionType(Enum):
    """Definition 23.1.2-23.1.3: Boundary condition types."""
    DIRICHLET = "dirichlet"    # Position clamping (sensors)
    NEUMANN = "neumann"        # Flux clamping (motors)
    REFLECTIVE = "reflective"  # Dreaming mode (zero flux)


class ContextType(Enum):
    """Definition 23.6.2: Context instantiation types."""
    RL = "rl"                        # Action space
    CLASSIFICATION = "classification"  # Label space
    LLM = "llm"                      # Prompt space


@dataclass
class InterfaceConfig:
    """Configuration for HolographicInterface."""
    obs_dim: int = 64
    action_dim: int = 8
    latent_dim: int = 32
    hidden_dim: int = 256
    num_visual_charts: int = 8
    num_action_charts: int = 4
    codes_per_chart: int = 64
    context_dim: int = 64
    sigma_motor: float = 0.1
    T_c: float = 1.0


class DualAtlasEncoder(nn.Module):
    """
    Definitions 23.2.1-23.2.2: Symmetric encoder for Visual/Action atlases.

    Extends AttentiveAtlasEncoder (Section 7.8) with unified interface.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        num_charts: int,
        codes_per_chart: int,
        atlas_type: Literal["visual", "action"],
    ):
        super().__init__()
        self.atlas_type = atlas_type
        self.num_charts = num_charts
        self.latent_dim = latent_dim

        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # Cross-attention routing (Section 7.8)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.chart_queries = nn.Parameter(torch.randn(num_charts, hidden_dim) * 0.02)
        self.scale = hidden_dim ** 0.5

        # Per-chart codebooks
        self.codebooks = nn.ModuleList([
            nn.Embedding(codes_per_chart, latent_dim)
            for _ in range(num_charts)
        ])

        # Residual decomposition
        self.nuisance_head = nn.Linear(hidden_dim, latent_dim)
        self.texture_head = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Encode input to (macro, nuisance, texture) triple."""
        B = x.shape[0]

        # Feature extraction
        h = self.feature_extractor(x)

        # Cross-attention routing
        k = self.key_proj(h)  # [B, H]
        attn = torch.einsum('bh,ch->bc', k, self.chart_queries) / self.scale
        chart_probs = F.softmax(attn, dim=-1)  # [B, C]
        chart_idx = chart_probs.argmax(dim=-1)  # [B]

        # VQ from selected chart
        z_macro = torch.stack([
            self.codebooks[c.item()](torch.zeros(1, dtype=torch.long, device=x.device)).squeeze()
            for c in chart_idx
        ])  # [B, D]

        # Residual decomposition
        z_nuisance = self.nuisance_head(h)
        z_texture = self.texture_head(h)

        return {
            'chart_idx': chart_idx,
            'chart_probs': chart_probs,
            'z_macro': z_macro,
            'z_nuisance': z_nuisance,
            'z_texture': z_texture,
        }


def sample_motor_texture(
    z: torch.Tensor,
    d_motor_tex: int,
    sigma_motor: float,
) -> torch.Tensor:
    """
    Definition 23.3.3: Sample motor texture with conformal scaling.

    Sigma_motor(z) = sigma^2 * G^{-1}(z) = sigma^2 * (1-|z|^2)^2 / 4
    """
    B = z.shape[0]
    device = z.device

    # Conformal factor at z (Poincare disk)
    r_sq = (z ** 2).sum(dim=-1, keepdim=True)
    G_inv_scale = (1.0 - r_sq.clamp(max=0.99)) ** 2 / 4.0

    # Sample with geometry-dependent variance
    xi = torch.randn(B, d_motor_tex, device=device)
    z_tex_motor = sigma_motor * torch.sqrt(G_inv_scale) * xi

    return z_tex_motor


class ContextConditionedPolicy(nn.Module):
    """
    Definition 23.6.4: Context-conditioned policy for unified task handling.

    Unifies RL actions, classification labels, and LLM tokens.
    """

    def __init__(
        self,
        latent_dim: int,
        context_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__()

        # Context embedding
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(
        self,
        z: torch.Tensor,
        context: torch.Tensor,
        T_c: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute context-conditioned action distribution.

        pi(a|z, c) = softmax(-Phi_eff(z, K, c) / T_c)
        """
        # Embed context
        c_embed = self.context_encoder(context)

        # Concatenate state and context
        z_c = torch.cat([z, c_embed], dim=-1)

        # Compute logits (negative effective potential)
        logits = self.policy_net(z_c)

        # Softmax with temperature
        probs = F.softmax(logits / T_c, dim=-1)

        return {
            'logits': logits,
            'probs': probs,
            'context_embedding': c_embed,
        }


class HolographicInterface(nn.Module):
    """
    Section 23: The Holographic Interface.

    Implements the symplectic boundary between Agent and Environment.
    Combines:
    - Dual Atlas (Visual + Action)
    - Motor Texture sampling
    - Context-conditioned policy
    - Thermodynamic cycle tracking

    Cross-references:
    - Section 20 (WFR Geometry)
    - Section 21 (Holographic Generation {cite}`thooft1993holographic,susskind1995world`)
    - Section 22 (Geodesic SDE)
    """

    def __init__(self, config: InterfaceConfig):
        super().__init__()
        self.config = config

        # Visual Atlas (Definition 23.2.1)
        self.visual_atlas = DualAtlasEncoder(
            config.obs_dim, config.hidden_dim, config.latent_dim,
            config.num_visual_charts, config.codes_per_chart, "visual"
        )

        # Action Atlas (Definition 23.2.2)
        self.action_atlas = DualAtlasEncoder(
            config.action_dim, config.hidden_dim, config.latent_dim,
            config.num_action_charts, config.codes_per_chart, "action"
        )

        # Context-conditioned policy (Definition 23.6.4)
        self.policy = ContextConditionedPolicy(
            config.latent_dim, config.context_dim,
            config.action_dim, config.hidden_dim
        )

        # Action decoder (tangent bundle decoder)
        self.action_decoder = nn.Sequential(
            nn.Linear(config.latent_dim * 2, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.action_dim),
        )

    def forward_perception(
        self,
        x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Phase I: Compression (Environment -> Bulk).
        Thermodynamics: Entropy reduction, heat release.
        Implements Definition 23.1.2 (Dirichlet BC).
        """
        return self.visual_atlas(x)

    def forward_actuation(
        self,
        z: torch.Tensor,
        u_intent: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Phase III: Expansion (Bulk -> Environment).
        Thermodynamics: Entropy increase, work done.
        Implements Definition 23.1.3 (Neumann BC).
        """
        B = z.shape[0]

        # Sample motor texture (Definition 23.3.3)
        z_tex_motor = sample_motor_texture(
            z, self.config.latent_dim, self.config.sigma_motor
        )

        # Decode intention to action
        z_u = torch.cat([z, u_intent], dim=-1)
        a_base = self.action_decoder(z_u)

        # Add motor texture
        a_raw = a_base + z_tex_motor[:, :self.config.action_dim]

        return {
            'action': a_raw,
            'action_base': a_base,
            'motor_texture': z_tex_motor,
        }

    def forward_proprioception(
        self,
        a_realized: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Inverse model: Map realized actions back to latent intentions.
        Used to calculate execution error.
        """
        return self.action_atlas(a_realized)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        mode: Literal["waking", "dreaming"] = "waking",
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass through holographic interface.

        Args:
            x: Observation [B, obs_dim]
            context: Context conditioning [B, context_dim]
            mode: "waking" (boundary clamped) or "dreaming" (reflective)
        """
        # Phase I: Perception (compression)
        vis_out = self.forward_perception(x)
        z = vis_out['z_nuisance']  # Use nuisance as state

        # Get context-conditioned policy
        policy_out = self.policy(z, context, self.config.T_c)

        if mode == "dreaming":
            # Reflective boundary: no actuation
            return {
                'visual': vis_out,
                'policy': policy_out,
                'mode': mode,
            }

        # Phase III: Action (expansion)
        u_intent = policy_out['context_embedding']
        act_out = self.forward_actuation(z, u_intent, context)

        return {
            'visual': vis_out,
            'policy': policy_out,
            'action': act_out,
            'mode': mode,
        }
```

**Cross-references:** Section 7.8 (AttentiveAtlasEncoder), Section 7.10 (TopologicalDecoder), Algorithm 22.4.2 (BAOAB).

::::{note} Connection to RL #7: Dreamer/World Models as Generic RNN Dynamics
**The General Law (Fragile Agent):**
The HolographicInterface implements latent dynamics via **symplectic integrators** on the state-space manifold $(\mathcal{Z}, G, \omega)$. The BAOAB algorithm (Section 22.4) preserves the symplectic structure:

$$
\hat{z}_{t+1} = \Phi_{\text{BAOAB}}(z_t) \quad \text{with} \quad \omega(\Phi_* X, \Phi_* Y) = \omega(X, Y).
$$
The dual atlas structure (Definition 23.2) decomposes latent space into Visual and Action atlases with matched boundary conditions.

**The Degenerate Limit:**
Remove symplectic structure: replace $\Phi_{\text{BAOAB}}$ with generic RNN/GRU. Ignore boundary condition matching.

**The Special Case (Standard RL - Dreamer, MuZero):**
World-model RL uses generic neural network dynamics:

$$
z_{t+1} = f_\theta(z_t, a_t), \quad \hat{r}_t = r_\theta(z_t), \quad \hat{\gamma}_t = \gamma_\theta(z_t).
$$
The RNN/GRU/Transformer architecture has no geometric constraints—it's a universal function approximator.

**Result:** Dreamer/MuZero/PlaNet are the $\omega \to 0$ limit where symplectic structure is ignored. The agent learns arbitrary dynamics rather than energy-conserving flows.

**What the generalization offers:**
- **Conservation guarantees**: Symplectic integrators preserve phase-space volume (Liouville's theorem)
- **Long-horizon stability**: Energy drift bounded by $O((\Delta t)^k)$ for $k$th-order integrators
- **Interpretable rollouts**: Latent trajectories follow geodesics modified by potential forces
- **Boundary semantics**: Dirichlet/Neumann conditions distinguish observation vs action interfaces (Definition 23.1)
::::

(sec-summary-tables-and-diagnostic-nodes-a)=
### 23.8 Summary Tables and Diagnostic Nodes

**Summary of Holographic Interface:**

| Component              | Visual (Perception)           | Motor (Action)                  |
|------------------------|-------------------------------|---------------------------------|
| **Boundary Condition** | Dirichlet (position clamp)    | Neumann (flux clamp)            |
| **Atlas**              | $\mathcal{A}_{\text{vis}}$    | $\mathcal{A}_{\text{act}}$      |
| **Macro**              | Chart index $K$               | Action primitive $A$            |
| **Nuisance**           | Pose/viewpoint $z_n$          | Compliance $z_{n,\text{motor}}$ |
| **Texture**            | Pixel detail $z_{\text{tex}}$ | Tremor $z_{\text{tex,motor}}$   |
| **Thermodynamics**     | Compression ($\Delta S < 0$)  | Expansion ($\Delta S > 0$)      |

**Context Space Instantiation:**

| Task           | Context $c$  | Output          | Potential $\Phi_{\text{eff}}$              |
|----------------|--------------|-----------------|--------------------------------------------|
| RL             | Action space | Motor command   | $V_{\text{critic}}$                        |
| Classification | Label space  | Class $\hat{y}$ | $-\log p(y\mid z)$                         |
| LLM            | Prompt space | Token           | $-\log p(\text{tok}\mid z, \text{prompt})$ |

(node-30)=
**Node 30: SymplecticBoundaryCheck**

| **#**  | **Name**                    | **Component** | **Type**           | **Interpretation**               | **Proxy**                                                | **Cost** |
|--------|-----------------------------|---------------|--------------------|----------------------------------|----------------------------------------------------------|----------|
| **30** | **SymplecticBoundaryCheck** | **Interface** | **BC Consistency** | Are sensor/motor BCs compatible? | $\lVert\omega(j_{\text{sense}}, j_{\text{motor}})\rVert$ | $O(Bd)$  |

**Trigger conditions:**
- High SymplecticBoundaryCheck: Sensor and motor boundary conditions violate symplectic structure.
- Remedy: Recalibrate boundary coupling; verify Legendre transform consistency; check phase space constraints.

(node-31)=
**Node 31: DualAtlasConsistencyCheck**

| **#**  | **Name**                      | **Component** | **Type**          | **Interpretation**                     | **Proxy**                                                                  | **Cost**  |
|--------|-------------------------------|---------------|-------------------|----------------------------------------|----------------------------------------------------------------------------|-----------|
| **31** | **DualAtlasConsistencyCheck** | **Encoder**   | **Atlas Duality** | Are Visual and Action atlases aligned? | $\lVert e_\alpha^{\text{vis}} - \mathcal{L}(e_\beta^{\text{act}})\rVert^2$ | $O(BK^2)$ |

**Trigger conditions:**
- High DualAtlasConsistencyCheck: Visual and Action atlases have drifted apart.
- Remedy: Increase Legendre alignment loss; verify codebook coupling; check chart transition consistency.

(node-32)=
**Node 32: MotorTextureCheck**

| **#**  | **Name**              | **Component** | **Type**           | **Interpretation**                       | **Proxy**                                                  | **Cost**               |
|--------|-----------------------|---------------|--------------------|------------------------------------------|------------------------------------------------------------|------------------------|
| **32** | **MotorTextureCheck** | **Policy**    | **Motor Firewall** | Is motor texture decoupled from control? | $\lVert\partial_{z_{\text{tex,motor}}} \pi(a\mid z)\rVert$ | $O(Bd_{\text{motor}})$ |

**Trigger conditions:**
- High MotorTextureCheck: Motor texture is leaking into control decisions (firewall violated).
- Remedy: Increase motor texture firewall penalty; verify motor residual decomposition; check Axiom {prf:ref}`ax-motor-texture-firewall`.

(node-33)=
**Node 33: ThermoCycleCheck**

| **#**  | **Name**             | **Component**   | **Type**           | **Interpretation**                               | **Proxy**                                              | **Cost** |
|--------|----------------------|-----------------|--------------------|--------------------------------------------------|--------------------------------------------------------|----------|
| **33** | **ThermoCycleCheck** | **World Model** | **Energy Balance** | Is perception/action thermodynamically balanced? | $\lvert W_{\text{compress}} - W_{\text{expand}}\rvert$ | $O(B)$   |

**Trigger conditions:**
- High ThermoCycleCheck: Thermodynamic imbalance between perception and action phases.
- Remedy: Recalibrate information flow rates; verify boundary coupling; check Carnot efficiency bound (Proposition {prf:ref}`prop-carnot-efficiency-bound`).

(node-34)=
**Node 34: ContextGroundingCheck**

| **#**  | **Name**                  | **Component** | **Type**             | **Interpretation**                          | **Proxy**                 | **Cost** |
|--------|---------------------------|---------------|----------------------|---------------------------------------------|---------------------------|----------|
| **34** | **ContextGroundingCheck** | **Policy**    | **Context Validity** | Is context properly grounding motor output? | $I(A_t; c) / I(X_t; K_t)$ | $O(B)$   |

**Trigger conditions:**
- Low ContextGroundingCheck: Context is not influencing motor output (ungrounded generation).
- Remedy: Increase context embedding strength; verify context-conditioned potential; check symmetry-breaking kick.



(sec-the-reward-field-value-forms-and-hodge-geometry)=
