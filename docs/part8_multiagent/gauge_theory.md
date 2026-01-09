## 29. Relativistic Symplectic Multi-Agent Field Theory

*Abstract.* We derive Multi-Agent Reinforcement Learning (MARL) as a system of $N$ coupled field equations on a causal spacetime structure. When agents do not share exactly the same boundary, information propagates at finite speed $c_{\text{info}}$, transforming the elliptic (Helmholtz) value equation into a hyperbolic (Klein-Gordon) wave equation. The Markov property, lost on the spatial manifold $\mathcal{Z}^{(N)}$ alone, is restored on the **Causal Bundle** $\mathcal{Z}^{(N)} \times \Xi_{<t}$, where $\Xi_{<t}$ is the Memory Screen integrating incoming wavefronts from the past light cone. We derive the **Ghost Interface**, where agents optimize against retarded images of their opponents, and prove that Nash Equilibrium is a standing wave pattern in the joint causal field. The instantaneous (Newtonian) formulation emerges as the $c_{\text{info}} \to \infty$ limit.

:::{admonition} Researcher Bridge: From Action-at-a-Distance to Field Theory
:class: warning
:name: rb-relativistic-marl
Standard Multi-Agent RL assumes a global clock: when Agent A acts, Agent B sees it instantly. In distributed systems or physical reality, this violates causality. We upgrade the framework: Value is not a static field but a **propagating wave**. Agents do not interact with each other's current states; they interact with the **Past Light Cone** of the environment. Memory is no longer optional—it is the physical requirement to restore the Markov property in a relativistic universe.
:::

*Cross-references:* This section generalizes the Helmholtz equation ({ref}`Section 24.2 <sec-the-bulk-potential-screened-poisson-equation>`) to the Klein-Gordon equation, and elevates the Memory Screen ({ref}`Section 27.1 <sec-the-historical-manifold-and-memory-screen>`) from a recording device to a primary state variable. It extends the single-agent geometry (Sections 20–24) to the multi-agent setting.

*Literature:* Game theory {cite}`fudenberg1991game`; stochastic games {cite}`shapley1953stochastic`; multi-agent RL {cite}`littman1994markov,lowe2017multi`; symplectic geometry {cite}`arnold1989mathematical`; retarded potentials {cite}`jackson1999classical`.

::::{note} Connection to RL #17: Independent PPO as Disconnected Sheaf
**The General Law (Fragile Agent):**
Multi-agent interaction is modeled via **Ghost Interfaces** (Definition {prf:ref}`def-ghost-interface`) connecting retarded boundary states:

$$
\mathcal{G}_{ij}(t) \subset \partial\mathcal{Z}^{(i)}(t) \times \partial\mathcal{Z}^{(j)}(t - \tau_{ij}), \quad \omega_{\mathcal{G},ij} := \omega^{(i)}(t) \oplus \omega^{(j)}(t - \tau_{ij})\big|_{\mathcal{G}_{ij}}.
$$
The **Game Tensor** $\mathcal{G}_{ij}$ (Definition {prf:ref}`def-the-game-tensor`) encodes strategic coupling with retarded components: how Agent $i$'s latent inertia changes due to Agent $j$'s past state.

**The Degenerate Limit:**
Set all interfaces $\mathcal{G}_{ij} = \emptyset$ (disconnect the sheaf). Each agent treats others as stationary noise.

**The Special Case (Standard RL - IPPO):**
Independent PPO {cite}`de2020independent` runs separate learners with shared reward:

$$
\pi^{(i)} = \arg\max_{\pi} \mathbb{E}\left[ \sum_t r^{(i)}_t(\mathbf{s}, \mathbf{a}) \right], \quad \text{treating } \pi^{(-i)} \text{ as fixed}.
$$
Each agent optimizes against a stationary environment—other agents are part of the "MDP noise."

**Result:** IPPO is the $\mathcal{G}_{ij} \to \emptyset$ limit where agents are **solipsistic**—they share a world but have no causal coupling.

**What the generalization offers:**
- **Causal structure**: Finite $c_{\text{info}}$ determines which events can influence which (Section 29.2)
- **Ghost Interface**: Agents couple to retarded images, not instantaneous states (Section 29.4)
- **Klein-Gordon value equation**: Hyperbolic wave propagation replaces elliptic relaxation (Section 29.5)
- **Nash as standing wave**: Equilibrium is time-averaged stationarity (Section 29.7)
- **Diagnostic nodes 46-48, 62**: Runtime monitoring including causality violation checks (Section 29.17)
::::

(sec-the-product-configuration-space)=
### 29.1 The Product Configuration Space

Consider $N$ agents, each with an internal latent manifold $(\mathcal{Z}^{(i)}, G^{(i)})$ and a boundary interface $B^{(i)} = (x^{(i)}, a^{(i)}, r^{(i)})$. The agents may be spatially distributed, with finite information propagation time between them.

:::{prf:definition} N-Agent Product Manifold
:label: def-n-agent-product-manifold

The global configuration space is the product manifold:

$$
\mathcal{Z}^{(N)} := \mathcal{Z}^{(1)} \times \mathcal{Z}^{(2)} \times \cdots \times \mathcal{Z}^{(N)}.
$$
The metric on $\mathcal{Z}^{(N)}$ is the direct sum of individual metrics:

$$
G^{(N)} := \bigoplus_{i=1}^N G^{(i)},
$$
where each $G^{(i)}$ is the capacity-constrained metric from Theorem {prf:ref}`thm-capacity-constrained-metric-law`. In coordinates, this is block-diagonal: if $\mathbf{z} = (z^{(1)}, \ldots, z^{(N)})$ with $z^{(i)} \in \mathbb{R}^{d_i}$, then $G^{(N)}_{\mu\nu}(\mathbf{z}) = G^{(i)}_{ab}(z^{(i)})$ when indices $\mu, \nu$ both lie in agent $i$'s block, and $G^{(N)}_{\mu\nu} = 0$ otherwise.

*Units:* $[G^{(N)}] = [z]^{-2}$.

*Remark (Isolated Agents).* The product metric $G^{(N)}$ describes agents in **isolation**—there is no cross-coupling between $\mathcal{Z}^{(i)}$ and $\mathcal{Z}^{(j)}$. Strategic coupling modifies this to $\tilde{G}^{(N)}$ via the Game Tensor (Section 29.6).

:::

:::{prf:definition} Agent-Specific Boundary Interface
:label: def-agent-specific-boundary-interface

Each agent $i$ possesses its own symplectic boundary $(\partial\mathcal{Z}^{(i)}, \omega^{(i)})$ with:
- **Dirichlet component** (sensors): $\phi^{(i)}(x) = $ observation stream
- **Neumann component** (motors): $j^{(i)}_{\text{motor}}(x) = $ action flux

The boundary conditions follow the structure of Definition {prf:ref}`def-dirichlet-boundary-condition-sensors`–23.1.3, applied per-agent.

*Cross-reference:* Section 23.1 (Symplectic Boundary Manifold), Definition {prf:ref}`def-mass-tensor`.

:::

:::{prf:definition} Environment Distance
:label: def-environment-distance

Let $d_{\mathcal{E}}^{ij}$ denote the **environment distance** between agents $i$ and $j$—the geodesic length in the environment manifold $\mathcal{E}$ that information must traverse. This may differ from the latent distance $d_G(z^{(i)}, z^{(j)})$.

*Examples:*
- **Physical agents:** $d_{\mathcal{E}}^{ij} = $ spatial separation in meters
- **Networked agents:** $d_{\mathcal{E}}^{ij} = $ network hop distance or latency
- **Co-located agents:** $d_{\mathcal{E}}^{ij} = 0$ (shared boundary)

*Units:* $[d_{\mathcal{E}}^{ij}] = $ meters or equivalent environment-specific units.

:::
(sec-the-failure-of-simultaneity)=
### 29.2 The Failure of Simultaneity

The standard HJB equation assumes the value $V(z)$ relaxes instantly across the manifold. This implies an infinite speed of information propagation, violating the causal constraints of distributed systems.

:::{prf:axiom} Information Speed Limit
:label: ax-information-speed-limit

There exists a maximum speed $c_{\text{info}} > 0$ at which information propagates through the environment $\mathcal{E}$. The **Causal Delay** between agents $i$ and $j$ is:

$$
\tau_{ij} := \frac{d_{\mathcal{E}}^{ij}}{c_{\text{info}}},
$$
where $d_{\mathcal{E}}^{ij}$ is the environment distance (Definition {prf:ref}`def-environment-distance`).

*Units:* $[c_{\text{info}}] = [\text{length}]/[\text{time}]$, $[\tau_{ij}] = [\text{time}]$.

*Examples:*
- **Physical systems:** $c_{\text{info}} = c \approx 3 \times 10^8$ m/s (speed of light)
- **Acoustic systems:** $c_{\text{info}} \approx 343$ m/s (speed of sound)
- **Networked systems:** $c_{\text{info}} \approx d/\text{latency}$ (effective propagation speed)
- **Co-located agents:** $c_{\text{info}} \to \infty$ effective limit when $d_{\mathcal{E}}^{ij} = 0$

:::

:::{prf:definition} Causal Interval
:label: def-causal-interval

The **Causal Interval** between spacetime events $(z^{(i)}, t_i)$ and $(z^{(j)}, t_j)$ is:

$$
\Delta s^2_{ij} := -c_{\text{info}}^2 (t_j - t_i)^2 + (d_{\mathcal{E}}^{ij})^2.
$$
The events are classified as:
- **Timelike** ($\Delta s^2_{ij} < 0$): $|t_j - t_i| > \tau_{ij}$. Causal influence is possible.
- **Spacelike** ($\Delta s^2_{ij} > 0$): $|t_j - t_i| < \tau_{ij}$. No causal influence is possible.
- **Lightlike** ($\Delta s^2_{ij} = 0$): $|t_j - t_i| = \tau_{ij}$. Boundary case.

*Consequence:* If agents $i$ and $j$ are spacelike separated at time $t$, no instantaneous Hamiltonian $H(z^{(i)}_t, z^{(j)}_t)$ can couple their states. Coupling must occur via retarded potentials.

:::

:::{prf:definition} Past Light Cone
:label: def-past-light-cone

The **Past Light Cone** of Agent $i$ at time $t$ is the set of all agent-time pairs that can causally influence Agent $i$:

$$
\mathcal{C}^-_i(t) := \left\{ (j, t') \in \{1,\ldots,N\} \times \mathbb{R} : t' \leq t - \tau_{ij} \right\}.
$$
The **Future Light Cone** is defined symmetrically:

$$
\mathcal{C}^+_i(t) := \left\{ (j, t') : t' \geq t + \tau_{ij} \right\}.
$$
*Physical interpretation:* Agent $i$ at time $t$ can only receive information from events in $\mathcal{C}^-_i(t)$ and can only influence events in $\mathcal{C}^+_i(t)$. The region outside both cones is causally disconnected.

:::

::::{admonition} Physics Isomorphism: Minkowski Spacetime
:class: note
:name: pi-minkowski

**In Physics:** Special relativity defines the causal structure via the Minkowski metric $ds^2 = -c^2 dt^2 + dx^2 + dy^2 + dz^2$. Events with $ds^2 < 0$ are timelike separated (causally connected); events with $ds^2 > 0$ are spacelike separated (causally disconnected) {cite}`jackson1999classical`.

**In Implementation:** The causal interval (Definition {prf:ref}`def-causal-interval`) induces a Lorentzian structure on the agent-time space:

$$
\Delta s^2_{ij} = -c_{\text{info}}^2 \Delta t^2 + d_{\mathcal{E}}^2.
$$

**Correspondence Table:**
| Special Relativity | Multi-Agent System |
|:-------------------|:-------------------|
| Speed of light $c$ | Information speed $c_{\text{info}}$ |
| Spatial distance | Environment distance $d_{\mathcal{E}}^{ij}$ |
| Past light cone | Causally accessible agent states |
| Spacelike separation | Instantaneously decoupled agents |
| Lorentz invariance | Causal consistency under frame changes |
::::

(sec-the-relativistic-state-restoring-markovianity)=
### 29.3 The Relativistic State: Restoring Markovianity

To recover a valid control problem under finite information speed, we must augment the state to include the field configuration within the past light cone.

:::{prf:definition} Retarded Potential (Memory Screen)
:label: def-retarded-potential

Let $\rho^{(j)}(t, z)$ be the reward/action flux emitted by Agent $j$. The potential perceived by Agent $i$ at position $z$ and time $t$ is the **Retarded Potential**:

$$
\Psi_{\text{ret}}^{(i)}(t, z) = \sum_{j \neq i} \int_{-\infty}^{t} \int_{\mathcal{Z}^{(j)}} G_{\text{ret}}(z, t; \zeta, \tau) \rho^{(j)}(\tau, \zeta) \, d\mu_G(\zeta) \, d\tau,
$$
where $G_{\text{ret}}$ is the **Retarded Green's Function** for the wave operator on the manifold:

$$
G_{\text{ret}}(z, t; \zeta, \tau) \propto \frac{\delta\left((t-\tau) - d_{\mathcal{E}}(z, \zeta)/c_{\text{info}}\right)}{d_{\mathcal{E}}(z, \zeta)^{(D-2)/2}}.
$$

*Interpretation:* Agent $i$ does not perceive Agent $j$'s current state. It perceives the "ghost" of Agent $j$ from time $\tau_{ij} = d_{\mathcal{E}}^{ij}/c_{\text{info}}$ ago.

*Units:* $[\Psi_{\text{ret}}] = \text{nat}$, $[G_{\text{ret}}] = [z]^{-(D-2)/2}[\text{time}]^{-1}$.

:::

:::{prf:definition} Causal Bundle
:label: def-causal-bundle

The **Causal Bundle** is the augmented state space:

$$
\mathcal{Z}_{\text{causal}} := \mathcal{Z}^{(N)} \times \Xi_{<t},
$$
where:
- $\mathcal{Z}^{(N)} = \prod_i \mathcal{Z}^{(i)}$ is the product configuration space (Definition {prf:ref}`def-n-agent-product-manifold`)
- $\Xi_{<t}$ is the **Memory Screen** restricted to the causal past (Definition {prf:ref}`def-memory-screen`): $\Xi_{<t} = \{(\gamma(t'), \alpha(t')) : t' < t\}$

The **Relativistic State** for Agent $i$ at time $t$ is:

$$
\mathcal{S}^{(i)}_t := \left( z^{(i)}_t, \Xi^{(i)}_{<t} \right),
$$
where $\Xi^{(i)}_{<t}$ stores the history of received retarded potentials over the interval $[t - \tau_{\text{horizon}}, t)$.

*Operational validity:* $\Xi^{(i)}_{<t}$ is locally observable at time $t$. The "true" global state of all agents is hidden, but $\mathcal{S}^{(i)}_t$ is a sufficient statistic for Agent $i$'s optimal policy within its future light cone.

:::

:::{prf:theorem} Markov Restoration on Causal Bundle
:label: thm-markov-restoration

Let $P(z^{(N)}_{t+\Delta t} | z^{(N)}_t, \Xi_{<t})$ denote the transition probability. When agents have finite causal delay $\tau_{ij} > 0$:

1. **On $\mathcal{Z}^{(N)}$ alone:** The Markov property fails:
   $$
   P(z^{(N)}_{t+\Delta t} | z^{(N)}_t) \neq P(z^{(N)}_{t+\Delta t} | z^{(N)}_{\leq t}).
   $$

2. **On $\mathcal{Z}_{\text{causal}}$:** The Markov property is restored:
   $$
   P\left((z^{(N)}_{t+\Delta t}, \Xi_{<t+\Delta t}) \,\big|\, (z^{(N)}_t, \Xi_{<t})\right) = P\left((z^{(N)}_{t+\Delta t}, \Xi_{<t+\Delta t}) \,\big|\, \text{full history}\right).
   $$

*Proof sketch.* The Memory Screen $\Xi_{<t}$ encodes all information about past states that can causally influence the future. By the definition of the past light cone (Definition {prf:ref}`def-past-light-cone`), no additional information from $\Xi_{<t'}$ for $t' < t$ is needed beyond what is already encoded in $\Xi_{<t}$. The causal structure guarantees that spacelike-separated events cannot contribute new information. See **Appendix E.14** for the complete proof using causal factorization and Chapman-Kolmogorov. $\square$

:::

:::{prf:corollary} Memory as Physical Necessity
:label: cor-memory-physical-necessity

In the relativistic multi-agent setting, the Memory Screen (Definition {prf:ref}`def-memory-screen`) is not an optional enhancement but a **physical requirement** for a well-posed control problem. Without it, the agent's state is non-Markovian, and optimal control theory does not apply.

*Cross-reference:* This elevates the role of $\Xi_{<t}$ from Section 27.1, where it served as a recording device for trajectory history, to a primary state variable that restores the Markov property.

:::



(sec-the-ghost-interface)=
### 29.4 The Ghost Interface: Asynchronous Coupling

We replace the instantaneous coupling of boundary conditions with an asynchronous **Ghost Interface** that respects causal structure.

:::{prf:definition} Ghost Interface
:label: def-ghost-interface

The **Ghost Interface** $\mathcal{G}_{ij}(t)$ between agents $i$ and $j$ at time $t$ is:

$$
\mathcal{G}_{ij}(t) := \partial\mathcal{Z}^{(i)}(t) \times \partial\mathcal{Z}^{(j)}(t - \tau_{ij}),
$$
coupling Agent $i$'s current boundary to Agent $j$'s past boundary, where $\tau_{ij} = d_{\mathcal{E}}^{ij}/c_{\text{info}}$ is the causal delay.

The **Ghost Symplectic Structure** is:

$$
\omega_{\mathcal{G},ij} := \omega^{(i)}(t) \oplus \omega^{(j)}(t - \tau_{ij})\big|_{\mathcal{G}_{ij}}.
$$

*Mechanism:* Agent $i$ couples not to $z^{(j)}_t$, but to the **Ghost State** $\hat{z}^{(j)}_t := z^{(j)}_{t-\tau_{ij}}$—the state of Agent $j$ when the signal was emitted.

*Units:* $[\tau_{ij}] = [\text{time}]$.

:::

:::{prf:proposition} Interaction Kernel
:label: prop-interaction-kernel

The **pairwise interaction potential** $\Phi_{\text{int}}: \mathcal{Z} \times \mathcal{Z} \to \mathbb{R}$ between agents at positions $z, \zeta$ is the screened Green's function weighted by influence:

$$
\Phi_{\text{int}}(z, \zeta) := \alpha \cdot \mathcal{G}_{\kappa}(z, \zeta)
$$
where $\mathcal{G}_{\kappa}$ is the screened Green's function (Proposition {prf:ref}`prop-green-s-function-interpretation`) and $\alpha$ encodes the strategic relationship.

*Properties:*
- $\Phi_{\text{int}}(z, \zeta) = \Phi_{\text{int}}(\zeta, z)$ (symmetric in cooperative settings)
- $\Phi_{\text{int}} \to 0$ as $d_G(z, \zeta) \to \infty$ (locality via screening)
- $\nabla^2_z \Phi_{\text{int}}$ defines the Game Tensor contribution (Definition {prf:ref}`def-the-game-tensor`)
:::

:::{prf:definition} Retarded Interaction Potential
:label: def-retarded-interaction-potential

The **Retarded Interaction Potential** from Agent $j$ to Agent $i$ at time $t$ is:

$$
\Phi^{\text{ret}}_{ij}(z^{(i)}, t) := \alpha_{ij} \cdot \mathcal{G}_{\kappa}(z^{(i)}, \hat{z}^{(j)}_t) \cdot \sigma^{(j)}_r(\hat{z}^{(j)}_t),
$$
where:
- $\hat{z}^{(j)}_t = z^{(j)}_{t - \tau_{ij}}$ is the ghost state
- $\mathcal{G}_{\kappa}$ is the screened Green's function (Proposition {prf:ref}`prop-green-s-function-interpretation`)
- $\alpha_{ij} \in \{-1, 0, +1\}$ encodes the strategic relationship:
  - $\alpha_{ij} = +1$: Cooperative
  - $\alpha_{ij} = 0$: Independent
  - $\alpha_{ij} = -1$: Adversarial

*Remark:* The interaction depends on Agent $j$'s state at the retarded time, not the current time. This introduces **Strategic Hysteresis**: Agent $i$ may commit to a trajectory based on old information about $j$, only to encounter updated conditions later.

:::

:::{prf:theorem} Strategic Delay Tensor
:label: thm-strategic-delay-tensor

The effective coupling tensor $\mathcal{T}_{ij}$ between agents splits into instantaneous and retarded components:

$$
\mathcal{T}_{ij}^{\text{total}}(t) = \underbrace{\mathcal{T}_{ij}^{\text{local}}(t)}_{\text{Short-range}} + \underbrace{\int_{-\infty}^t \mathcal{K}_{\text{delay}}(t-\tau) \mathcal{T}_{ij}^{\text{ghost}}(\tau) \, d\tau}_{\text{Long-range Retarded}},
$$
where $\mathcal{K}_{\text{delay}}(t-\tau) = \delta(t - \tau - \tau_{ij})$ is the delay kernel.

**Adversarial consequence:** Against a distant adversary, the effective metric inflation (from the Game Tensor) is delayed. An agent may commit to an aggressive trajectory only to experience a "wall" of increased inertia arriving from the opponent's past actions.

*Proof.* Expand the coupled value equation to second order in the retarded potential. The cross-Hessian $\partial^2 V^{(i)} / \partial z^{(j)} \partial z^{(j)}$ evaluated at $z^{(j)}_{t-\tau_{ij}}$ yields the delayed Game Tensor contribution. $\square$

:::

:::{prf:corollary} Newtonian Limit
:label: cor-newtonian-limit-ghost

As $c_{\text{info}} \to \infty$, the causal delay vanishes: $\tau_{ij} \to 0$ for all pairs. The Ghost Interface reduces to the instantaneous interface:

$$
\lim_{c_{\text{info}} \to \infty} \mathcal{G}_{ij}(t) = \partial\mathcal{Z}^{(i)}(t) \times \partial\mathcal{Z}^{(j)}(t),
$$
and the retarded potential becomes instantaneous:

$$
\lim_{c_{\text{info}} \to \infty} \Phi^{\text{ret}}_{ij}(z^{(i)}, t) = \Phi_{ij}(z^{(i)}, z^{(j)}_t).
$$

*Interpretation:* Co-located agents ($d_{\mathcal{E}}^{ij} = 0$) or systems with negligible propagation delay operate in the Newtonian regime where standard MARL applies.

:::

::::{admonition} Physics Isomorphism: Liénard-Wiechert Potentials
:class: note
:name: pi-lienard-wiechert

**In Physics:** The electromagnetic potentials of a moving charge are evaluated at the retarded time $t_{\text{ret}} = t - r/c$, not the current time. The Liénard-Wiechert potentials encode causality in classical electrodynamics {cite}`jackson1999classical`.

**In Implementation:** The Ghost Interface evaluates strategic potentials at the retarded time:

$$
\Phi^{\text{ret}}_{ij}(z^{(i)}, t) = \Phi_{ij}(z^{(i)}, z^{(j)}_{t-\tau_{ij}}).
$$

**Correspondence Table:**
| Electrodynamics | Relativistic Agent |
|:----------------|:-------------------|
| Field equation $\square A^\mu = J^\mu$ | Value equation $\square_G V = \rho_r$ |
| Light speed $c$ | Information speed $c_{\text{info}}$ |
| Retarded time $t_{\text{ret}}$ | Ghost time $t - \tau_{ij}$ |
| Liénard-Wiechert potential | Retarded interaction potential |
| Radiation reaction | Strategic back-pressure |
::::



(sec-the-hyperbolic-value-equation)=
### 29.5 The Hyperbolic Value Equation (Klein-Gordon)

Under relativistic constraints, the elliptic Helmholtz equation for Value (Theorem {prf:ref}`thm-the-hjb-helmholtz-correspondence`) transforms into a hyperbolic wave equation.

:::{prf:theorem} HJB-Klein-Gordon Correspondence
:label: thm-hjb-klein-gordon

Let information propagate at speed $c_{\text{info}}$. The Value Function $V^{(i)}(z, t)$ for Agent $i$ satisfies the **Screened Wave Equation**:

$$
\boxed{\left( \frac{1}{c_{\text{info}}^2} \frac{\partial^2}{\partial t^2} + \gamma_{\text{damp}} \frac{\partial}{\partial t} - \Delta_{G^{(i)}} + \kappa_i^2 \right) V^{(i)}(z, t) = \rho^{(i)}_r(z, t) + \sum_{j \neq i} \Phi^{\text{ret}}_{ij}(z, t)}
$$
where:
- $\square_{G} = \frac{1}{c_{\text{info}}^2}\partial_t^2 - \Delta_G$ is the **D'Alembertian** on the manifold
- $\gamma_{\text{damp}} \geq 0$ is the temporal damping rate (related to discount)
- $\kappa_i$ is the **spatial screening mass** with $[\kappa_i] = 1/[\text{length}]$, related to the discount factor by:
  $$\kappa_i = \frac{-\ln\gamma_i}{c_{\text{info}} \Delta t} = \frac{\kappa_{i,\text{temporal}}}{c_{\text{info}}}$$
  where $\kappa_{i,\text{temporal}} = -\ln\gamma_i / \Delta t$ is the temporal discount rate with units $1/[\text{time}]$
- $\rho^{(i)}_r$ is the local reward source (units: $[\text{nat}]/[\text{length}]^2$)
- $\Phi^{\text{ret}}_{ij}$ is the retarded interaction potential (Definition {prf:ref}`def-retarded-interaction-potential`)

*Proof sketch.* Expand the Bellman recursion $V(z, t) = r \Delta t + \gamma \mathbb{E}[V(z', t+\Delta t)]$ to second order in both spatial and temporal increments. The finite propagation speed $c_{\text{info}}$ introduces the wave term $\partial_t^2 V$. The derivation parallels the passage from Poisson to wave equation in electrostatics vs. electrodynamics. See Appendix E.12. $\square$

*Character:* This is a hyperbolic PDE (wave equation with mass and damping), in contrast to the elliptic Helmholtz equation of Section 24.2.

:::

:::{prf:corollary} Value Wavefront Propagation
:label: cor-value-wavefront

A sudden change in reward at location $z_A$ and time $t_0$ propagates outward as a **Value Wavefront**:

$$
V(z, t) \sim \frac{\Theta(t - t_0 - d_G(z, z_A)/c_{\text{info}})}{d_G(z, z_A)^{(D-2)/2}} \cdot e^{-\kappa d_G(z, z_A)} \cdot \rho_r(z_A, t_0),
$$
where $\Theta$ is the Heaviside step function enforcing causality.

*Interpretation:* The Value surface is not a static potential but a dynamic "ocean" of interfering causal ripples. Reward shocks propagate at speed $c_{\text{info}}$, decaying exponentially with the screening length $1/\kappa$.

:::

:::{prf:corollary} Helmholtz as Newtonian Limit
:label: cor-helmholtz-limit

In the limit $c_{\text{info}} \to \infty$, the temporal derivatives become negligible:

$$
\frac{1}{c_{\text{info}}^2} \frac{\partial^2 V}{\partial t^2} \to 0,
$$
and the Klein-Gordon equation reduces to the **stationary Helmholtz equation**:

$$
(-\Delta_G + \kappa^2) V = \rho_r + \sum_{j \neq i} \Phi_{ij}.
$$
This recovers Theorem {prf:ref}`thm-the-hjb-helmholtz-correspondence` as the instantaneous (Newtonian) limit.

:::

:::{prf:proposition} Retarded Green's Function
:label: prop-retarded-greens-function

The solution to the inhomogeneous Klein-Gordon equation is given by convolution with the **Retarded Green's Function**:

$$
V^{(i)}(z, t) = \int_{-\infty}^{t} \int_{\mathcal{Z}^{(i)}} G_{\text{ret}}(z, t; \zeta, \tau) \left[ \rho^{(i)}_r(\zeta, \tau) + \sum_{j} \Phi^{\text{ret}}_{ij}(\zeta, \tau) \right] d\mu_G(\zeta) \, d\tau,
$$
where $G_{\text{ret}}$ satisfies:

$$
\left( \frac{1}{c_{\text{info}}^2} \frac{\partial^2}{\partial t^2} - \Delta_G + \kappa^2 \right) G_{\text{ret}}(z, t; \zeta, \tau) = \delta(z - \zeta)\delta(t - \tau),
$$
with the **causal boundary condition** $G_{\text{ret}} = 0$ for $t < \tau$.

*Form in flat space:* For $\mathcal{Z} = \mathbb{R}^D$ with Euclidean metric:

$$
G_{\text{ret}}(z, t; \zeta, \tau) = \frac{\Theta(t - \tau)}{4\pi |z - \zeta|} \delta\left(t - \tau - \frac{|z-\zeta|}{c_{\text{info}}}\right) \cdot e^{-\kappa|z-\zeta|}.
$$

:::

::::{admonition} Physics Isomorphism: Klein-Gordon Equation
:class: note
:name: pi-klein-gordon

**In Physics:** The Klein-Gordon equation $(\square + m^2)\phi = \rho$ describes a relativistic scalar field with mass $m$. It reduces to the Helmholtz equation in the static limit {cite}`jackson1999classical`.

**In Implementation:** The Value function satisfies:

$$
\left(\frac{1}{c_{\text{info}}^2}\partial_t^2 - \Delta_G + \kappa^2\right)V = \rho_r
$$

**Correspondence Table:**
| Klein-Gordon (Physics) | Value Equation (Agent) |
|:-----------------------|:-----------------------|
| Scalar field $\phi$ | Value function $V$ |
| Mass parameter $m$ | Screening mass $\kappa$ |
| Source $\rho$ | Reward density $\rho_r$ |
| D'Alembertian $\square$ | Manifold wave operator $\square_G$ |
| Static limit | Newtonian (Helmholtz) limit |
| Propagating modes | Value wavefronts |
::::
(sec-the-game-tensor-deriving-adversarial-geometry)=
### 29.6 The Game Tensor: Relativistic Adversarial Geometry

In an adversarial (zero-sum) game, Agent $j$ acts to minimize the value $V^{(i)}$ that Agent $i$ maximizes. Under relativistic constraints, the Game Tensor acquires retarded components that introduce strategic hysteresis.

:::{prf:definition} The Game Tensor
:label: def-the-game-tensor

We define the **Game Tensor** $\mathcal{G}_{ij}^{kl}$ as the cross-Hessian of Agent $i$'s value with respect to Agent $j$'s position:

$$
\mathcal{G}_{ij}^{kl}(z^{(i)}, z^{(j)}) := \frac{\partial^2 V^{(i)}}{\partial z^{(j)}_k \partial z^{(j)}_l}\bigg|_{z^{(j)} = z^{(j)*}},
$$
where $z^{(j)*}$ is Agent $j$'s current position (or expected position under their policy). This tensor measures how sensitive Agent $i$'s value landscape is to Agent $j$'s location.

*Units:* $[\mathcal{G}_{ij}^{kl}] = \text{nat}/[z]^2$.

**Derivation 29.4.2 (The Strategic Metric).** Recall the **Capacity-Constrained Metric Law** (Theorem {prf:ref}`thm-capacity-constrained-metric-law`), where curvature is driven by the Risk Tensor $T_{ab}$. See **Appendix E.16** for the formal derivation of the Strategic Jacobian and Game Tensor using the implicit function theorem.

For Agent $i$, the "risk" includes the **Predictive Volatility** of the adversary $j$. If Agent $i$ updates its state by $\delta z^{(i)}$, and the adversary $j$ responds with $\delta z^{(j)} \approx \mathcal{J}_{ji} \delta z^{(i)}$ (where $\mathcal{J}_{ji}$ is the **Strategic Jacobian**—the best-response derivative, see Definition {prf:ref}`def-strategic-jacobian`), the second-order variation of Agent $i$'s value is:

$$
\delta^2 V^{(i)} = (\delta z^{(i)})^\top \left( \nabla_{z^{(i)}}^2 V^{(i)} + \underbrace{(\nabla_{z^{(j)}} \nabla_{z^{(i)}} V^{(i)}) \mathcal{J}_{ji}}_{\text{Strategic back-reaction}} \right) \delta z^{(i)}.
$$
**Agent $i$'s perceived geometry** is modified by adversarial presence as follows:

1. **Effective metric inflation.** In regions where the strategic back-reaction has positive eigenvalues (adversarial curvature), Agent $i$ perceives an inflated metric:

   $$
   \tilde{G}^{(i)}_{kl}(z) = G^{(i)}_{kl}(z) + \sum_{j \neq i} \beta_{ij} \cdot \mathcal{G}_{ij,kl}(z),
   $$
   where $\mathcal{G}_{ij,kl} = G^{(i)}_{km} G^{(i)}_{ln} \mathcal{G}_{ij}^{mn}$ is the Game Tensor with lowered indices, and $\beta_{ij} > 0$ for adversarial agents, $\beta_{ij} = 0$ for neutral, $\beta_{ij} < 0$ for cooperative.

2. **Geodesic deflection.** The Christoffel symbols acquire correction terms from the metric perturbation:

   $$
   \tilde{\Gamma}^{(i),m}_{kl} = \Gamma^{(i),m}_{kl} + \frac{1}{2}(G^{(i)})^{mn}\left(\nabla_k (\beta \mathcal{G})_{nl} + \nabla_l (\beta \mathcal{G})_{nk} - \nabla_n (\beta \mathcal{G})_{kl}\right),
   $$
   where $(\beta\mathcal{G})_{kl} := \sum_{j \neq i} \beta_{ij} \mathcal{G}_{ij,kl}$.

3. **Risk amplification.** High $\|\mathcal{G}_{ij}\|$ regions correspond to strategic uncertainty. This contributes to the Risk Tensor (Theorem {prf:ref}`thm-capacity-constrained-metric-law`):

   $$
   T^{(i)}_{kl} \to T^{(i)}_{kl} + \gamma_{\text{game}} \sum_{j \neq i} |\beta_{ij}| \cdot \mathcal{G}_{ij,kl}.
   $$
*Physical interpretation:* Adversarial agents effectively "curve" each other's latent space. An agent approaching a contested region experiences increased geodesic resistance (higher mass), making aggressive maneuvers more costly.

**The sign structure** of the Game Tensor $\mathcal{G}_{ij}$ determines the strategic relationship:

| Eigenvalue Structure | $\text{sgn}(\det \mathcal{G}_{ij})$ | Interpretation                                              |
|----------------------|-------------------------------------|-------------------------------------------------------------|
| All positive         | $+$                                 | Adversarial: $j$'s presence increases $i$'s value curvature |
| All negative         | $(-1)^d$                            | Cooperative: $j$'s presence smooths $i$'s value landscape   |
| Mixed signs          | varies                              | Mixed-motive game                                           |
| Near-zero            | $\approx 0$                         | Weakly coupled (near-independent)                           |

The trace $\operatorname{tr}(\mathcal{G}_{ij}) = \sum_k \mathcal{G}_{ij}^{kk}$ measures **total strategic sensitivity**: how much Agent $i$'s value curvature depends on Agent $j$'s position. Large $|\operatorname{tr}(\mathcal{G}_{ij})|$ indicates high strategic coupling; small trace indicates approximate independence.

*Cross-reference:* The Game Tensor generalizes the conformal factor $\Omega$ (Definition {prf:ref}`def-value-metric-conformal-coupling`) to the multi-agent setting. Where $\Omega$ captured self-induced value curvature, $\mathcal{G}_{ij}$ captures cross-agent value curvature.

*Cross-reference (Gauge-Covariant Version):* When local gauge invariance is imposed (Section 29.13), the Game Tensor acquires a gauge-covariant form $\tilde{\mathcal{G}}_{ij}^{kl} := D_k D_l V^{(i)}|_{z^{(j)}}$ using covariant derivatives. Under gauge transformation $U(z)$, the covariant Game Tensor transforms homogeneously: $\tilde{\mathcal{G}}'_{ij} = U \tilde{\mathcal{G}}_{ij} U^\dagger$. See Definition {prf:ref}`def-gauge-covariant-game-tensor`.

:::
:::{prf:theorem} Adversarial Mass Inflation
:label: thm-adversarial-mass-inflation

In a competitive game where Agent $j$ is adversarial ($\beta_{ij} > 0$) and the Game Tensor $\mathcal{G}_{ij}$ is positive semi-definite, the effective metric $\tilde{G}^{(i)}$ satisfies:

$$
\tilde{G}^{(i)}_{kl} \xi^k \xi^l \geq G^{(i)}_{kl} \xi^k \xi^l \quad \forall \xi \in T_{z}\mathcal{Z}^{(i)}.
$$
*Consequence:* The effective **Mass** $M^{(i)}(z)$ (Definition {prf:ref}`def-mass-tensor`) of Agent $i$ increases: $\tilde{M}^{(i)} \geq M^{(i)}$.

*First-Principles Interpretation:* Adversarial presence "thickens" the latent space. The agent moves more slowly (smaller geodesic steps) because it must account for the adversary's counter-maneuvers. **Strategic uncertainty is geometrically identical to physical inertia.**

*Proof.* From Definition {prf:ref}`def-the-game-tensor`, the metric perturbation is $\delta G_{kl} = \sum_{j} \beta_{ij} \mathcal{G}_{ij,kl}$. For adversarial agents, $\beta_{ij} > 0$. If $\mathcal{G}_{ij}$ is positive semi-definite (which occurs when Agent $j$'s presence increases the curvature of $V^{(i)}$), then $\mathcal{G}_{ij,kl} \xi^k \xi^l \geq 0$ for all $\xi$. Thus $\tilde{G}^{(i)}_{kl} \xi^k \xi^l = G^{(i)}_{kl} \xi^k \xi^l + \beta_{ij} \mathcal{G}_{ij,kl} \xi^k \xi^l \geq G^{(i)}_{kl} \xi^k \xi^l$. $\square$

:::

:::{admonition} Researcher Bridge: Opponents as Geometric Inertia
:class: info
:name: rb-opponents-inertia
In game-theoretic settings, adversarial opponents increase the effective **mass** (metric tensor eigenvalues) of the agent's latent space via the Game Tensor $\mathcal{G}_{ij}$. This transforms strategic uncertainty into geometric inertia: the agent moves more slowly in contested regions because geodesic steps are more costly. Cooperation has the opposite effect—allies smooth the value landscape, reducing effective mass.
:::

:::{prf:definition} Retarded Game Tensor
:label: def-retarded-game-tensor

Under finite information speed $c_{\text{info}}$, the Game Tensor acquires a **retarded component**. The **Retarded Game Tensor** is:

$$
\mathcal{G}_{ij}^{kl,\text{ret}}(z^{(i)}, t) := \frac{\partial^2 V^{(i)}}{\partial z^{(j)}_k \partial z^{(j)}_l}\bigg|_{z^{(j)} = \hat{z}^{(j)}_t},
$$
where $\hat{z}^{(j)}_t = z^{(j)}_{t - \tau_{ij}}$ is the ghost state of Agent $j$ at the retarded time.

The **total effective metric** including retardation is:

$$
\tilde{G}^{(i)}_{kl}(z, t) = G^{(i)}_{kl}(z) + \sum_{j \neq i} \beta_{ij} \cdot \mathcal{G}_{ij,kl}^{\text{ret}}(z, t).
$$

*Consequence (Strategic Hysteresis):* The metric inflation Agent $i$ experiences depends on Agent $j$'s position at the retarded time, not the current time. An agent may enter a region expecting low resistance, only to encounter a "delayed wall" of metric inflation arriving from the opponent's past position.

:::

:::{prf:proposition} Retarded Metric Propagation
:label: prop-retarded-metric-propagation

The effective metric $\tilde{G}^{(i)}(z, t)$ satisfies a wave-like propagation equation:

$$
\frac{\partial \tilde{G}^{(i)}_{kl}}{\partial t} = \sum_{j \neq i} \beta_{ij} \frac{\partial \mathcal{G}_{ij,kl}^{\text{ret}}}{\partial t} = \sum_{j \neq i} \beta_{ij} \frac{d\mathcal{G}_{ij,kl}}{dt}\bigg|_{t-\tau_{ij}}.
$$

The metric perturbation at time $t$ depends on the opponent's dynamics at time $t - \tau_{ij}$. Information about strategic coupling propagates at speed $c_{\text{info}}$.

:::



(sec-relativistic-nash-equilibrium)=
### 29.7 Relativistic Nash Equilibrium (Standing Waves)

In a system with finite information propagation, what constitutes equilibrium? It is not a static configuration but a coherent spatiotemporal pattern—a **standing wave** in the joint causal field.

:::{prf:definition} Joint WFR Action (Relativistic)
:label: def-joint-wfr-action

The N-agent WFR action on the product space with retarded interactions is:

$$
\mathcal{A}^{(N)}[\boldsymbol{\rho}, \mathbf{v}, \mathbf{r}] = \int_0^T \left[ \sum_{i=1}^N \int_{\mathcal{Z}^{(i)}} \left(\|v^{(i)}\|_{\tilde{G}^{(i)}}^2 + \lambda_i^2 |r^{(i)}|^2 \right) d\rho^{(i)} + \mathcal{V}_{\text{int}}^{\text{ret}}(\boldsymbol{\rho}, t) \right] dt,
$$
where:
- $v^{(i)}$ is the velocity field for Agent $i$'s belief flow
- $r^{(i)}$ is the reaction term (mass creation/destruction)
- $\tilde{G}^{(i)}$ is the game-augmented metric with retarded components (Definition {prf:ref}`def-retarded-game-tensor`)
- $\mathcal{V}_{\text{int}}^{\text{ret}}(\boldsymbol{\rho}, t) = \sum_{i < j} \int \Phi^{\text{ret}}_{ij}(z^{(i)}, t) \, d\rho^{(i)}(z^{(i)}) d\rho^{(j)}(z^{(j)})$ is the retarded interaction energy

*Cross-reference:* Definition {prf:ref}`def-the-wfr-action`, Definition {prf:ref}`def-retarded-interaction-potential`.

:::

:::{prf:theorem} Nash Equilibrium as Standing Wave
:label: thm-nash-standing-wave

In the relativistic formulation, a Nash equilibrium is a joint density $\boldsymbol{\rho}^*(\mathbf{z}, t)$ satisfying **time-averaged stationarity**:

$$
\left\langle \frac{\partial \boldsymbol{\rho}^*}{\partial t} \right\rangle_T := \frac{1}{T}\int_0^T \frac{\partial \boldsymbol{\rho}^*}{\partial t}(\mathbf{z}, t') \, dt' = 0,
$$
where the averaging period $T \gg \max_{i,j} \tau_{ij}$ exceeds all causal delays.

**Characterization:** A standing wave Nash equilibrium satisfies:

1. **Time-averaged gradient vanishing:**
   $$
   \left\langle (G^{(i)})^{-1} \nabla_{z^{(i)}} \Phi_{\text{eff}}^{(i,\text{ret})} \right\rangle_T = 0 \quad \forall i
   $$

2. **Balanced probability currents:** The flux exchanged between agents via retarded potentials is balanced over one wave period:
   $$
   \int_0^T \mathbf{J}^{(i)}(z, t) \, dt = 0 \quad \text{for all } z \in \mathcal{Z}^{(i)}
   $$
   where $\mathbf{J}^{(i)} = \rho^{(i)} \mathbf{v}^{(i)}$ is the probability current.

3. **Resonance condition:** The system oscillates at the characteristic causal frequency:
   $$
   \omega_{\text{Nash}} \sim \frac{c_{\text{info}}}{\bar{d}_{\mathcal{E}}},
   $$
   where $\bar{d}_{\mathcal{E}}$ is the mean environment distance between agents.

*Proof sketch.* The coupled Klein-Gordon system (Theorem {prf:ref}`thm-hjb-klein-gordon`) for $N$ agents forms a cavity resonator. Equilibrium states are the eigenmodes of the joint D'Alembertian operator. The ground state (lowest energy mode) corresponds to the stable Nash equilibrium; higher modes are metastable. See **Appendix E.15** for the complete derivation with boundary conditions, eigenmode expansion, and connection to game-theoretic optimality. $\square$

:::

:::{prf:corollary} Newtonian Limit of Nash
:label: cor-newtonian-nash-limit

As $c_{\text{info}} \to \infty$, the standing wave Nash reduces to the static Nash equilibrium:

$$
\lim_{c_{\text{info}} \to \infty} \boldsymbol{\rho}^*(\mathbf{z}, t) = \boldsymbol{\rho}^*_{\text{static}}(\mathbf{z}),
$$
and the geometric stasis conditions (vanishing gradient, stationary Game Tensor) hold instantaneously rather than on average.

:::

:::{prf:theorem} Geometric Stasis (Newtonian Limit)
:label: thm-nash-equilibrium-as-geometric-stasis

In the Newtonian limit ($c_{\text{info}} \to \infty$), a strategy profile $\mathbf{z}^* = (z^{(1)*}, \ldots, z^{(N)*})$ is a Nash equilibrium if and only if it satisfies the instantaneous **geometric stasis conditions**:

1. **Vanishing individual gradient:**
   $$
   (G^{(i)})^{-1} \nabla_{z^{(i)}} \Phi_{\text{eff}}^{(i)}(z^{(i)*}; z^{(-i)*}) = 0 \quad \forall i
   $$

2. **Stationary Game Tensor:**
   $$
   \frac{d}{dt}\mathcal{G}_{ij}^{kl}\bigg|_{\mathbf{z}^*} = 0 \quad \forall i,j
   $$

3. **Non-positive second variation:**
   $$
   \delta^2 V^{(i)}|_{z^{(i)*}} \leq 0 \quad \forall i, \forall \delta z^{(i)}
   $$

*Remark (Nash vs. Pareto).* Geometric stasis need not coincide with global optimality (Pareto). The Game Tensor eigenstructure determines the gap: trace-negative (cooperative) tends toward Pareto-improving basins; trace-positive (adversarial) tends toward Pareto-suboptimal saddles.

:::

:::{prf:corollary} Vanishing Probability Current at Nash
:label: cor-vanishing-current-nash

At a standing wave Nash equilibrium, the **time-averaged probability current** vanishes:

$$
\langle \mathbf{J}^{(i)} \rangle_T = \langle \rho^{(i)} \mathbf{v}^{(i)} \rangle_T = 0 \quad \forall i.
$$

*Interpretation:* The agents are not "frozen"—they oscillate with the causal frequency $\omega_{\text{Nash}}$—but the net flow averages to zero. Nash equilibrium is dynamic balance, not static rest.

:::
(sec-diagnostic-nodes-part-i)=
### 29.8 Diagnostic Nodes 46–48, 62 (Multi-Agent Causality)

Following the diagnostic node convention (Section 3.1), we define monitors for multi-agent causal systems.

(node-46)=
**Node 46: GameTensorCheck**

| **#**  | **Name**            | **Component** | **Type**           | **Interpretation**                | **Proxy**                                                                     | **Cost**     |
|--------|---------------------|---------------|--------------------|-----------------------------------|-------------------------------------------------------------------------------|--------------|
| **46** | **GameTensorCheck** | Multi-Agent   | Strategic Coupling | Is strategic sensitivity bounded? | $\lVert\mathcal{G}_{ij}\rVert_F := \sqrt{\sum_{kl}(\mathcal{G}_{ij}^{kl})^2}$ | $O(N^2 d^2)$ |

**Interpretation:** Monitors the Frobenius norm of the Game Tensor between agent pairs. Large $\|\mathcal{G}_{ij}\|_F$ indicates high strategic interdependence, potentially leading to oscillatory dynamics or failure to converge.

**Threshold:** $\|\mathcal{G}_{ij}\|_F < \mathcal{G}_{\max}$ (implementation-dependent; typical default $\mathcal{G}_{\max} = 10 \cdot \|G^{(i)}\|_F$).

**Trigger conditions:**
- High GameTensorCheck: Agents are tightly coupled; small moves trigger large counter-moves.
- Remedy: Reduce coupling strength $\alpha_{\text{adv}}$; increase exploration temperature; consider decoupled training phases.

(node-47)=
**Node 47: NashResidualCheck**

| **#**  | **Name**              | **Component** | **Type**    | **Interpretation**                | **Proxy**                                                                                                       | **Cost** |
|--------|-----------------------|---------------|-------------|-----------------------------------|-----------------------------------------------------------------------------------------------------------------|----------|
| **47** | **NashResidualCheck** | Multi-Agent   | Equilibrium | Are agents near Nash equilibrium? | $\epsilon_{\text{Nash}} := \max_i \lVert(G^{(i)})^{-1}\nabla_{z^{(i)}} \Phi_{\text{eff}}^{(i)}\rVert_{G^{(i)}}$ | $O(N d)$ |

**Interpretation:** Measures the maximum deviation from the Nash stasis condition (Theorem {prf:ref}`thm-nash-equilibrium-as-geometric-stasis`, Condition 1). At equilibrium, $\epsilon_{\text{Nash}} = 0$.

**Threshold:** $\epsilon_{\text{Nash}} < \epsilon_{\text{Nash,tol}}$ (typical default $10^{-3}$).

If $\epsilon_{\text{Nash}} > 0$ but below threshold, the system is in a **transient non-equilibrated state**. This is expected during:
1. **Learning dynamics:** Agents are still adapting policies; gradients have not yet vanished.
2. **Environmental shift:** External conditions changed, invalidating previous equilibrium.
3. **Exploration phase:** Agents are deliberately perturbing away from equilibrium to discover better basins.

**Remediation:**
- If $\epsilon_{\text{Nash}}$ is decreasing: system is converging; no intervention needed.
- If $\epsilon_{\text{Nash}}$ is oscillating: potential limit cycle; reduce learning rates or add damping ($\gamma_{\text{damp}}$ in the joint SDE).
- If $\epsilon_{\text{Nash}}$ is increasing: instability detected; may indicate poorly conditioned Game Tensor. Check Node 46 for large $\|\mathcal{G}_{ij}\|_F$.

(node-48)=
**Node 48: RelativisticSymplecticCheck**

| **#**  | **Name**                      | **Component** | **Type**     | **Interpretation**                            | **Proxy**                                                                                                            | **Cost**   |
|--------|-------------------------------|---------------|--------------|-----------------------------------------------|----------------------------------------------------------------------------------------------------------------------|------------|
| **48** | **RelativisticSymplecticCheck** | Multi-Agent   | Conservation | Is retarded flux balanced across Ghost Interface? | $\Delta_{\omega}^{\text{ret}} := \int_{t_1}^{t_2} \left\lvert \Phi_{\text{out}}(t) - \Phi_{\text{in}}(t + \tau_{ij}) \right\rvert dt$ | $O(N^2 d)$ |

**Interpretation:** Monitors symplectic flux conservation on the Ghost Interface (Definition {prf:ref}`def-ghost-interface`). Under relativistic constraints, we compare outflow at time $t$ with inflow at retarded time $t + \tau_{ij}$. Immediate conservation is impossible; **retarded conservation** is the appropriate measure.

**Threshold:** $\Delta_{\omega}^{\text{ret}} < \epsilon_{\omega}$ (typical default $10^{-4}$).

**Trigger conditions:**
- Positive RelativisticSymplecticCheck: Energy is leaking through non-conservative forces or causal inconsistency.
- **Remedy:** Check for unmodeled friction; verify causal buffer implementation; reduce timestep.

*Cross-reference:* This is the relativistic generalization of symplectic volume conservation to retarded interactions.

(node-62)=
**Node 62: CausalityViolationCheck**

| **#**  | **Name**                    | **Component** | **Type**   | **Interpretation**                                     | **Proxy**                                                                                          | **Cost** |
|--------|-----------------------------|---------------|------------|--------------------------------------------------------|----------------------------------------------------------------------------------------------------|----------|
| **62** | **CausalityViolationCheck** | Multi-Agent   | Causality  | Did information arrive faster than $c_{\text{info}}$? | $\Delta_{\text{causal}} := \max_{i,j} \mathbb{I}\left[\Delta I(z^{(i)}_t; z^{(j)}_{t'}) > 0 \land t' > t - \tau_{ij}\right]$ | $O(N^2)$ |

**Interpretation:** Detects violations of the causal structure (Definition {prf:ref}`def-causal-interval`). Agent $i$ should have no mutual information with Agent $j$'s state at times $t' > t - \tau_{ij}$ (inside the future light cone).

**Threshold:** $\Delta_{\text{causal}} = 0$ (hard constraint: no superluminal information).

**Trigger conditions:**
- Positive CausalityViolationCheck: The simulation has leaked "ground truth" information that violates the light cone. This is a **fatal error** indicating:
  1. Incorrect causal buffer implementation
  2. Unmodeled fast communication channel
  3. Timing errors in boundary condition updates

- **Remedy:** Audit causal buffer; verify all inter-agent communication respects $\tau_{ij}$ delays; check for inadvertent global state sharing.

*Cross-reference:* This enforces the information speed limit (Axiom {prf:ref}`ax-information-speed-limit`).



(sec-summary-table-from-single-to-multi-agent)=
### 29.9 Summary Table: Newtonian vs. Einsteinian Agent

**Table 29.9.1 (Newtonian vs. Relativistic Multi-Agent).**

| Feature | Newtonian ($c_{\text{info}} \to \infty$) | Relativistic ($c_{\text{info}} < \infty$) |
|:--------|:-----------------------------------------|:------------------------------------------|
| **Information Speed** | $\infty$ (Instantaneous) | Finite $c_{\text{info}}$ |
| **Value PDE** | Elliptic (Helmholtz) $-\Delta_G V + \kappa^2 V = \rho_r$ | Hyperbolic (Klein-Gordon) $\square_G V + \kappa^2 V = \rho_r$ |
| **State** | $z^{(i)}_t$ (position only) | $(z^{(i)}_t, \Xi^{(i)}_{<t})$ (position + memory screen) |
| **Markov Property** | On $\mathcal{Z}^{(N)}$ | On Causal Bundle $\mathcal{Z}^{(N)} \times \Xi_{<t}$ |
| **Interaction** | Synchronous Bridge $\mathcal{B}_{ij}$ | Asynchronous Ghost Interface $\mathcal{G}_{ij}$ |
| **Potential** | Instantaneous $\Phi_{ij}(z^{(i)}, z^{(j)}_t)$ | Retarded $\Phi^{\text{ret}}_{ij}(z^{(i)}, z^{(j)}_{t-\tau})$ |
| **Game Tensor** | $\mathcal{G}_{ij}(z^{(j)}_t)$ | $\mathcal{G}_{ij}^{\text{ret}}(z^{(j)}_{t-\tau})$ |
| **Equilibrium** | Fixed Point (Geometric Stasis) | Standing Wave (Time-Averaged Stasis) |
| **Nash Condition** | $\nabla \Phi_{\text{eff}} = 0$ | $\langle \nabla \Phi_{\text{eff}} \rangle_T = 0$ |
| **Topology** | Riemannian Manifold | Lorentzian Causal Structure |
| **Diagnostics** | Nodes 46–48 | + Node 62 (CausalityViolation) |

**Table 29.9.2 (Single to Multi-Agent).**

| Concept | Single Agent (Sections 20–24) | Multi-Agent Relativistic (Section 29) |
|:--------|:------------------------------|:--------------------------------------|
| **State Space** | $\mathcal{Z}$ | $\mathcal{Z}_{\text{causal}} = \mathcal{Z}^{(N)} \times \Xi_{<t}$ |
| **Boundary** | Fixed $\partial\mathcal{Z}$ | Ghost Interface $\mathcal{G}_{ij}(t)$ |
| **Metric** | $G$ (Information Sensitivity) | $\tilde{G}^{(i)}(t) = G^{(i)} + \sum_j \beta_{ij}\mathcal{G}_{ij}^{\text{ret}}$ |
| **Value PDE** | $(-\Delta_G + \kappa^2)V = \rho_r$ | $(\square_G + \kappa^2)V^{(i)} = \rho^{(i)}_r + \sum_j \Phi^{\text{ret}}_{ij}$ |
| **Flow** | Langevin / WFR | Coupled Klein-Gordon + WFR |
| **Success** | Value Maxima | Standing Wave Nash |
| **Diagnostics** | Nodes 1–45 | + Nodes 46–48, 62 |



(sec-mean-field-metric-law)=
### 29.10 The Mean-Field Metric Law (Scalability Resolution)

The calculation of the Game Tensor $\mathcal{G}_{ij}$ ({prf:ref}`def-the-game-tensor`) entails computational complexity $O(N^2 d^2)$, which is intractable for large $N$. We prove that in the limit $N \to \infty$, the discrete Game Tensor converges to the Hessian of a convolution potential.

:::{prf:theorem} Mean-Field Metric Law
:label: thm-mean-field-metric-law

Let $\boldsymbol{z} = (z_1, \dots, z_N)$ be the configuration of $N$ agents on $\mathcal{Z}$. Let the empirical measure be $\mu_N = \frac{1}{N} \sum_{i=1}^N \delta_{z_i}$. As $N \to \infty$, assuming $\mu_N$ converges weakly to a smooth density $\rho \in \mathcal{P}(\mathcal{Z})$, the effective metric $\tilde{G}(z)$ for a test agent at position $z$ converges to:

$$
\tilde{G}(z) = G_{\text{intrinsic}}(z) + \alpha_{\text{adv}} \nabla^2_z \left( \Phi_{\text{int}} * \rho \right)(z)
$$
where $\Phi_{\text{int}}(z, \zeta)$ is the pairwise interaction potential ({prf:ref}`prop-interaction-kernel`) and $*$ denotes the Riemannian convolution.

*Proof.*
1. **Discrete Interaction Energy:** The total interaction potential for agent $i$ is $V_{\text{int}}(z_i) = \frac{1}{N} \sum_{j \neq i} \Phi_{\text{int}}(z_i, z_j)$.

2. **Discrete Game Tensor:** The Game Tensor acting on the metric is defined as the sum of cross-sensitivities ({prf:ref}`thm-adversarial-mass-inflation`):

$$
(\delta G)_{ab}(z_i) = \alpha_{\text{adv}} \sum_{j \neq i} \frac{\partial^2 \Phi_{\text{int}}(z_i, z_j)}{\partial z_i^a \partial z_i^b}.
$$
3. **Continuum Limit:** We rewrite the sum as an integral against the empirical measure:

$$
(\delta G)_{ab}(z) = \alpha_{\text{adv}} \int_{\mathcal{Z}} \nabla^2_{z, a, b} \Phi_{\text{int}}(z, \zeta) \, d\mu_N(\zeta).
$$
4. **Convergence:** Assuming $\Phi_{\text{int}}$ is $C^2$ and bounded, and $\mu_N \rightharpoonup \rho$ weakly, the integral converges to the convolution $(\nabla^2 \Phi_{\text{int}} * \rho)(z)$.

5. **Complexity Reduction:** The computation of $\tilde{G}$ now requires evaluating the Hessian of a static field $\Psi(z) = (\Phi_{\text{int}} * \rho)(z)$. This is $O(1)$ with respect to $N$ (given the density field), effectively decoupling the agent's complexity from the population size. $\square$
:::

*Cross-references:* This resolves the scalability limitation by reducing agent complexity from $O(N^2 d^2)$ to $O(d^2)$ via the Vlasov-geometry limit.



(sec-metabolic-tracking-bound)=
### 29.11 The Metabolic Tracking Bound (Non-Stationary Nash Resolution)

In non-stationary environments, the Nash equilibrium $z^*(t)$ shifts. We derive the tracking limit from the Computational Metabolism ({ref}`Section 31 <sec-computational-metabolism-the-landauer-bound-and-deliberation-dynamics>`), relating the metric speed of the target to the agent's power dissipation budget.

:::{prf:theorem} Metabolic Tracking Bound
:label: thm-metabolic-tracking-bound

Let $z^*(t)$ be a time-varying Nash equilibrium. An agent with maximum metabolic flux budget $\dot{\mathcal{M}}_{\max}$ can maintain tracking error $\epsilon \to 0$ if and only if the target's trajectory satisfies:

$$
\|\dot{z}^*\|_{\tilde{G}(z^*)} \leq \sqrt{\frac{2 \dot{\mathcal{M}}_{\max}}{\sigma_{\text{met}}}}
$$
where $\tilde{G}$ is the game-augmented metric ({prf:ref}`thm-adversarial-mass-inflation`).

*Proof.*
1. **Kinematic Requirement:** To track $z^*(t)$, the agent's transport velocity must satisfy $v = \dot{z}^*$.

2. **Thermodynamic Cost:** The metabolic cost of transport is $\dot{\mathcal{M}} = \frac{1}{2} \sigma_{\text{met}} \|v\|_{\tilde{G}}^2$ ({prf:ref}`def-metabolic-flux`).

3. **Adversarial Drag:** The metric $\tilde{G} = G + \alpha \mathcal{G}_{ij}$ includes the Game Tensor. High adversarial tension ($\mathcal{G}_{ij} \gg 0$) inflates the norm $\|\cdot\|_{\tilde{G}}$.

4. **Critical Failure:** If the adversary moves sufficiently fast or the conflict is sufficiently intense, the required dissipation exceeds $\dot{\mathcal{M}}_{\max}$. The agent loses tracking not due to algorithmic error, but due to exceeding its thermodynamic budget. $\square$
:::

*Interpretation:* The agent's ability to track a moving Nash equilibrium is fundamentally limited by its metabolic budget. Intense conflict ($\mathcal{G}_{ij}$ large) compounds this limitation by inflating the kinetic cost of pursuit.



(sec-variational-emergence-cooperation)=
### 29.12 Variational Emergence of Cooperation via Metric Inflation

We prove that cooperative equilibria correspond to local minima of the Onsager-Machlup action functional under the game-augmented metric, resolving the question of when adversarial coupling spontaneously yields cooperative behavior.

:::{prf:theorem} Geometric Locking Principle
:label: thm-geometric-locking-principle

Consider $N$ agents with Game Tensor $\mathcal{G}_{ij}$ ({prf:ref}`def-the-game-tensor`). In the presence of strong adversarial coupling, the joint system tends toward configurations where $\operatorname{Tr}(\mathcal{G}_{ij})$ is minimized.

*Proof.*

1. **Metric Inflation:** By {prf:ref}`thm-adversarial-mass-inflation`, the effective metric for agent $i$ is $\tilde{G}^{(i)} = G^{(i)} + \sum_j \beta_{ij} \mathcal{G}_{ij}$. For adversarial agents, $\beta_{ij} > 0$ and $\mathcal{G}_{ij}$ is positive semi-definite, implying $\det(\tilde{G}^{(i)}) \ge \det(G^{(i)})$.

2. **Kinetic Cost:** The WFR action ({prf:ref}`def-joint-wfr-action`) includes the transport term $\int \|v\|_{\tilde{G}}^2 d\rho$. An inflated metric implies a higher metabolic cost for any movement $v \neq 0$.

3. **Energy Minimization:** The system evolves to minimize the free energy $\mathcal{F}$. If the potential gain $\nabla V$ is bounded, but the kinetic cost scales with $\mathcal{G}_{ij}$, trajectories with large $\mathcal{G}_{ij}$ (intense conflict) become energetically prohibitive.

4. **Stationarity:** The system relaxes to a state where either $v \to 0$ (Nash stasis, {prf:ref}`thm-nash-equilibrium-as-geometric-stasis`) or the metric perturbation vanishes ($\mathcal{G}_{ij} \to 0$). The condition $\mathcal{G}_{ij} \to 0$ implies $\nabla_{z^{(j)}}\nabla_{z^{(i)}} V^{(i)} \to 0$, which defines a region of **strategic decoupling**. $\square$
:::

:::{prf:corollary} Metabolic Basis of Cooperation
:label: cor-metabolic-cooperation

Adversarial agents converge to cooperative or decoupled configurations because conflict maximizes the effective inertia of the state space, rendering non-cooperative trajectories metabolically unsustainable.

*Interpretation:* The Game Tensor acts as a "friction term" that penalizes rapid strategic maneuvers. In the long run, agents either:
1. **Cooperate:** Reduce $\mathcal{G}_{ij}$ by aligning their gradients
2. **Decouple:** Move to regions where $\nabla_{z^{(j)}} V^{(i)} \approx 0$
3. **Freeze:** Accept Nash stasis with $v^{(i)} = 0$

All three outcomes correspond to stationary points of the joint action functional.
:::



## Part V: Gauge Theory Layer

The relativistic framework of Sections 29.1–29.12 describes multi-agent dynamics on a curved Lorentzian manifold with retarded potentials. We now elevate this structure to a **gauge field theory** by recognizing that the nuisance variable $z_n$ (Definition 2.2.1) serves as an internal gauge degree of freedom. This identification transforms strategic interaction into the curvature of a **non-Abelian gauge connection**, placing multi-agent field theory on the same mathematical footing as the Standard Model of particle physics.

(sec-local-gauge-symmetry-nuisance-bundle)=
### 29.13 Local Gauge Symmetry and the Nuisance Bundle

The key insight is that the **nuisance fiber** $\mathcal{Z}_n$ at each macro-state $K$ is not merely a noise variable to be marginalized—it is the **internal gauge degree of freedom** that agents are free to rotate without changing physical outcomes. This local freedom mandates a compensating gauge field.

:::{prf:axiom} Local Gauge Invariance (Nuisance Invariance)
:label: ax-local-gauge-invariance

The physical dynamics of the multi-agent system are invariant under position-dependent rotations of the internal nuisance coordinates. Formally, let $G$ be a compact Lie group with Lie algebra $\mathfrak{g}$. For any smooth map $U: \mathcal{Z} \to G$, the transformation

$$
\psi'(z, t) = U(z)\psi(z, t)
$$

leaves observable quantities (reward, policy output, Nash conditions) unchanged.

*Units:* $[U] = \text{dimensionless}$ (group element).

*Interpretation:* Agent $i$ at location $z$ is free to rotate its internal representation (the "basis" in which it encodes nuisance). This is not a symmetry to be broken but a **redundancy** in the description that must be properly handled via gauge theory.

:::

:::{prf:definition} Local Gauge Group
:label: def-local-gauge-group

The **Local Gauge Group** is a compact Lie group $G$ with:

1. **Lie algebra $\mathfrak{g}$:** The tangent space at identity, with generators $\{T_a\}_{a=1}^{\dim(G)}$ satisfying $[T_a, T_b] = if^{abc}T_c$ where $f^{abc}$ are the **structure constants**.

2. **Representation:** The matter fields $\psi^{(i)}$ transform in a representation $\rho: G \to GL(V)$ where $V$ is the representation space.

3. **Position-dependent element:** $U(z) \in G$ for each $z \in \mathcal{Z}$, forming the infinite-dimensional group of gauge transformations $\mathcal{G} := C^\infty(\mathcal{Z}, G)$.

*Standard choices:*
- $G = SO(D)$: Rotations of $D$-dimensional nuisance space
- $G = SU(N)$: Unitary transformations (for complex representations)
- $G = U(1)$: Abelian phase rotations (electromagnetic limit)

*Cross-reference:* The $SO(D)$ symmetry at the origin (Proposition {prf:ref}`prop-so-d-symmetry-at-origin`) is the special case where the stabilizer is trivial.

:::

:::{prf:definition} Matter Field (Belief Amplitude)
:label: def-matter-field-belief-amplitude

The **Matter Field** for agent $i$ is the complex-valued section

$$
\psi^{(i)}: \mathcal{Z}^{(i)} \times \mathbb{R} \to V
$$

where $V$ is the representation space of $G$. The matter field is related to the belief wave-function by:

$$
\psi^{(i)}(z, t) = \sqrt{\rho^{(i)}(z, t)} \exp\left(\frac{iV^{(i)}(z, t)}{\sigma}\right) \cdot \xi^{(i)}(z)
$$

where:
- $\rho^{(i)}$ is the belief density
- $V^{(i)}$ is the value function
- $\sigma > 0$ is the **cognitive action scale**, $\sigma := T_c \cdot \tau_{\text{update}}$, the information-theoretic analog of Planck's constant (full definition: {prf:ref}`def-cognitive-action-scale` in Section 29.21)
- $\xi^{(i)}(z) \in V$ is the **internal state vector** encoding nuisance orientation

*Units:* $[\psi] = [\text{length}]^{-D/2}$ (probability amplitude density).

*Transformation law:* Under gauge transformation $U(z)$:

$$
\psi'^{(i)}(z, t) = \rho(U(z))\psi^{(i)}(z, t)
$$

where $\rho: G \to GL(V)$ is the representation.

:::

:::{prf:conjecture} Nuisance Fiber as Gauge Orbit (Motivating Principle)
:label: conj-nuisance-fiber-gauge-orbit

The nuisance fiber at each macro-state $K \in \mathcal{K}$ admits interpretation as a gauge orbit:

$$
\mathcal{Z}_n\big|_K \cong G_K / H_K
$$

where:
- $G_K \subseteq G$ is the gauge group restricted to macro-state $K$
- $H_K \subseteq G_K$ is the **stabilizer subgroup** fixing the codebook centroid $e_K$

*Special cases:*
1. **At origin ($K = 0$, Semantic Vacuum):** $G_0 = SO(D)$, $H_0 = \{e\}$, so $\mathcal{Z}_n|_0 \cong SO(D)$ (full rotational freedom).
2. **At generic $K$:** The stabilizer $H_K$ is non-trivial if $e_K$ has special structure (e.g., aligned with coordinate axes).
3. **At boundary ($|z| \to 1$):** The gauge orbit collapses as degrees of freedom freeze (Section 33, Causal Stasis).

*Motivation (not a rigorous proof):*
The nuisance coordinates $z_n$ parameterize how an observation is embedded relative to the macro-code $K$. Under the VQ-VAE architecture (Section 2.2b), two nuisance values $z_n$ and $z'_n$ are designed to be equivalent if they differ by a transformation preserving the macro-code: $z'_n = U \cdot z_n$ for some $U \in G_K$.

**Remark (Analogy vs. Isomorphism):** This correspondence is a *motivating analogy* rather than a proven isomorphism. A rigorous proof would require:
1. Showing the nuisance equivalence relation coincides with gauge equivalence
2. Proving the quotient $G_K/H_K$ is a smooth manifold diffeomorphic to $\mathcal{Z}_n|_K$
3. Establishing that the VQ-VAE induces a principal $G_K$-bundle structure

The gauge-theoretic formalism developed in Sections 29.13–29.20 is motivated by this conjecture but does not depend on it being rigorously true. The constructions (covariant derivative, field strength, etc.) are well-defined once the gauge group $G$ and its action are specified.

*Cross-reference:* This formalizes the design goal "K represents $x/G_{\text{spatial}}$" from Section 2.2b.

:::

::::{admonition} Physics Isomorphism: Local Gauge Symmetry
:class: note
:name: pi-local-gauge-symmetry

**In Physics:** Local gauge symmetry is the principle that the laws of physics are invariant under position-dependent phase rotations $\psi(x) \to e^{i\theta(x)}\psi(x)$. This invariance mandates the existence of gauge fields (photon, gluons, W/Z bosons) to maintain consistency {cite}`yang1954conservation,weinberg1995quantum`.

**In Implementation:** Nuisance invariance (Axiom {prf:ref}`ax-local-gauge-invariance`) is the principle that agent dynamics are invariant under position-dependent internal rotations $\psi(z) \to U(z)\psi(z)$.

**Correspondence Table:**

| Gauge Theory | Fragile Agent |
|:-------------|:--------------|
| Local phase $e^{i\theta(x)}$ | Nuisance rotation $U(z)$ |
| Gauge group $G$ | Internal symmetry group |
| Matter field $\psi$ | Belief amplitude |
| Gauge orbit $G/H$ | Nuisance fiber $\mathcal{Z}_n$ |
| Stabilizer $H$ | Residual symmetry at $K$ |

::::



(sec-strategic-connection-covariant-derivative)=
### 29.14 The Strategic Connection and Covariant Derivative

The failure of the ordinary derivative to transform covariantly under gauge transformations mandates the introduction of a **compensating field**—the gauge connection. In the multi-agent context, this connection encodes how the "meaning" of nuisance coordinates changes as one moves through latent space.

:::{prf:definition} Strategic Connection (Gauge Potential)
:label: def-strategic-connection

The **Strategic Connection** is a $\mathfrak{g}$-valued 1-form on $\mathcal{Z}$:

$$
A = A_\mu^a T_a \, dz^\mu
$$

where:
- $A_\mu^a(z, t)$ are the **connection coefficients** (real-valued functions)
- $\{T_a\}_{a=1}^{\dim(\mathfrak{g})}$ are the generators of the Lie algebra $\mathfrak{g}$
- $\mu$ indexes spacetime/latent coordinates $(t, z^1, \ldots, z^D)$

*Units:* $[A_\mu] = [\text{length}]^{-1}$ (inverse length, like momentum).

*Interpretation:* The connection $A_\mu$ tells agent $i$ how to "translate" the nuisance interpretation from point $z$ to point $z + dz$. It is the **strategic context** required to compare internal states at different locations.

:::

:::{prf:proposition} Gauge Transformation of the Connection
:label: prop-gauge-transformation-connection

Under a local gauge transformation $U(z) \in G$, the connection transforms as:

$$
A'_\mu = U A_\mu U^{-1} - \frac{i}{g}(\partial_\mu U)U^{-1}
$$

where $g > 0$ is the **coupling constant** (strategic coupling strength).

*Proof.*
Demand that the covariant derivative (Definition {prf:ref}`def-covariant-derivative`) transform covariantly: $(D_\mu\psi)' = U(D_\mu\psi)$. Expanding:

$$
\begin{aligned}
D'_\mu\psi' &= (\partial_\mu - igA'_\mu)(U\psi) \\
&= (\partial_\mu U)\psi + U(\partial_\mu\psi) - igA'_\mu U\psi
\end{aligned}
$$

For this to equal $U(\partial_\mu - igA_\mu)\psi = U(\partial_\mu\psi) - igUA_\mu\psi$, we require:

$$
(\partial_\mu U)\psi - igA'_\mu U\psi = -igUA_\mu\psi
$$

Solving for $A'_\mu$ yields the stated transformation law. $\square$

*Interpretation:* The inhomogeneous term $-\frac{i}{g}(\partial_\mu U)U^{-1}$ compensates for the "frame twist" introduced by position-dependent gauge transformations. The connection must counter-twist to maintain covariance.

:::

:::{prf:definition} Covariant Derivative
:label: def-covariant-derivative

The **Covariant Derivative** acting on matter fields is:

$$
D_\mu = \partial_\mu - igA_\mu
$$

For a matter field $\psi$ in representation $\rho$:

$$
D_\mu\psi = \partial_\mu\psi - igA_\mu^a \rho(T_a)\psi
$$

*Properties:*
1. **Covariant transformation:** $(D_\mu\psi)' = U(D_\mu\psi)$
2. **Leibniz rule:** $D_\mu(\psi\chi) = (D_\mu\psi)\chi + \psi(D_\mu\chi)$
3. **Reduces to partial derivative** when $A_\mu = 0$ (trivial connection)

*Units:* $[D_\mu\psi] = [\psi]/[\text{length}]$.

:::

:::{prf:theorem} Gauge-Covariant Klein-Gordon Equation
:label: thm-gauge-covariant-klein-gordon

The Klein-Gordon equation for Value (Theorem {prf:ref}`thm-hjb-klein-gordon`) generalizes to the gauge-covariant form:

$$
\left(\frac{1}{c_{\text{info}}^2}D_t^2 - D^i D_i + \kappa^2\right)V^{(i)} = \rho_r^{(i)} + \sum_{j \neq i} \Phi_{ij}^{\text{ret}}
$$

where:
- $D_t = \partial_t - igA_0$ is the temporal covariant derivative
- $D_i = \partial_i - igA_i$ are spatial covariant derivatives
- $D^i = \tilde{G}^{ij}D_j$ with raised index via the strategic metric

*Proof sketch.*
The minimal coupling principle replaces $\partial_\mu \to D_\mu$ in the Klein-Gordon equation while preserving the equation's structure. The gauge-covariant d'Alembertian is:

$$
\Box_A := \frac{1}{c_{\text{info}}^2}D_t^2 - \tilde{G}^{ij}D_i D_j = \frac{1}{\sqrt{|\tilde{G}|}}D_\mu\left(\sqrt{|\tilde{G}|}\tilde{G}^{\mu\nu}D_\nu\right)
$$

The screening term $\kappa^2 V$ and source terms are gauge-invariant scalars. $\square$

:::

:::{prf:proposition} Minimal Coupling Principle
:label: prop-minimal-coupling

To maintain gauge invariance, all derivatives in the dynamics must be replaced by covariant derivatives:

$$
\partial_\mu \longrightarrow D_\mu = \partial_\mu - igA_\mu
$$

This **Minimal Coupling Principle** ensures that:
1. The WFR continuity equation becomes gauge-covariant
2. The HJB equation becomes gauge-covariant
3. Learning gradients transform properly under internal rotations

*Consequence for implementation:* Any gradient-based update rule $\theta \leftarrow \theta - \alpha \nabla_\theta \mathcal{L}$ must use the covariant gradient $D_\theta \mathcal{L}$ to maintain frame-independence.

:::

::::{admonition} Physics Isomorphism: Gauge Connection
:class: note
:name: pi-gauge-connection

**In Physics:** The gauge potential $A_\mu$ in electromagnetism is the 4-vector potential; in Yang-Mills theory, it takes values in the Lie algebra. The covariant derivative $D_\mu = \partial_\mu - ieA_\mu$ defines how charged particles couple to the electromagnetic field {cite}`jackson1999classical,peskin1995introduction`.

**In Implementation:** The strategic connection $A_\mu$ defines how belief amplitudes couple to the multi-agent environment.

**Correspondence Table:**

| Electromagnetism | Yang-Mills | Fragile Agent |
|:-----------------|:-----------|:--------------|
| $A_\mu$ (4-potential) | $A_\mu^a T_a$ | Strategic connection |
| $e$ (charge) | $g$ (coupling) | Strategic coupling $g$ |
| $D_\mu = \partial_\mu - ieA_\mu$ | $D_\mu = \partial_\mu - igA_\mu$ | Covariant update |
| Minimal coupling | Minimal coupling | Frame-invariant learning |

::::



(sec-gauge-transformation-game-tensor)=
### 29.15 Gauge Transformation of the Game Tensor

The Game Tensor $\mathcal{G}_{ij}$ (Definition {prf:ref}`def-the-game-tensor`) measures cross-agent strategic sensitivity. Under gauge transformations, this tensor acquires additional structure that we now characterize.

:::{prf:proposition} Game Tensor Gauge Transformation
:label: prop-game-tensor-gauge-transformation

Under a local gauge transformation $U(z)$, the Game Tensor transforms as:

$$
\mathcal{G}'_{ij}(z) = U(z) \mathcal{G}_{ij}(z) U(z)^\dagger + \mathcal{C}_{ij}[A, U]
$$

where $\mathcal{C}_{ij}[A, U]$ is a **connection correction** involving commutators $[A_\mu, \mathcal{G}_{ij}]$.

For **Abelian** gauge groups ($[T_a, T_b] = 0$), the correction vanishes:
$$
\mathcal{G}'_{ij} = \mathcal{G}_{ij} \quad \text{(Abelian)}
$$

For **non-Abelian** groups, the Game Tensor is not gauge-invariant but transforms covariantly.

*Interpretation:* In non-Abelian settings, strategic coupling itself depends on the choice of internal frame. The "strength" of conflict between agents cannot be measured without specifying a gauge.

:::

:::{prf:definition} Gauge-Covariant Game Tensor
:label: def-gauge-covariant-game-tensor

The **Gauge-Covariant Game Tensor** is defined using covariant derivatives:

$$
\tilde{\mathcal{G}}_{ij}^{kl}(z) := D_k D_l V^{(i)}\big|_{z^{(j)}}
$$

Explicitly:

$$
\tilde{\mathcal{G}}_{ij}^{kl} = \partial_k\partial_l V^{(i)} - ig(\partial_k A_l + \partial_l A_k)V^{(i)} - g^2[A_k, A_l]V^{(i)} + \Gamma^m_{kl}\partial_m V^{(i)}
$$

where $\Gamma^m_{kl}$ are the Christoffel symbols of the strategic metric.

*Properties:*
1. Transforms covariantly: $\tilde{\mathcal{G}}'_{ij} = U\tilde{\mathcal{G}}_{ij}U^\dagger$
2. Reduces to ordinary Game Tensor when $A_\mu = 0$
3. The trace $\text{Tr}(\tilde{\mathcal{G}}_{ij})$ is gauge-invariant

:::

:::{prf:theorem} Gauge-Invariant Metric Inflation
:label: thm-gauge-invariant-metric-inflation

The effective metric (Theorem {prf:ref}`thm-adversarial-mass-inflation`) generalizes to:

$$
\tilde{G}^{(i)}_{kl}(z) = G^{(i)}_{kl}(z) + \sum_{j \neq i} \beta_{ij} \text{Tr}\left[\tilde{\mathcal{G}}_{ij,kl}\right]
$$

where the trace projects onto the gauge-invariant component.

*Proof sketch.*
The physical metric must be gauge-invariant. Since $\tilde{\mathcal{G}}_{ij}$ transforms as $U\tilde{\mathcal{G}}_{ij}U^\dagger$, the trace $\text{Tr}(\tilde{\mathcal{G}}_{ij})$ is invariant under $U \to UVU^\dagger$ for any $V$, hence gauge-invariant. The sum over $j$ with coupling constants $\beta_{ij}$ preserves this invariance. $\square$

*Consequence:* The metric inflation experienced by agents is a **physical observable** independent of internal frame choice.

:::



(sec-field-strength-tensor)=
### 29.16 The Field Strength Tensor (Strategic Curvature)

The curvature of the gauge connection measures the **non-commutativity of parallel transport**—moving around a closed loop in latent space may result in a non-trivial internal rotation. This curvature is the **field strength tensor**, which we identify as strategic tension.

:::{prf:definition} Field Strength Tensor (Yang-Mills Curvature)
:label: def-field-strength-tensor

The **Field Strength Tensor** is the $\mathfrak{g}$-valued 2-form:

$$
\mathcal{F}_{\mu\nu} = \partial_\mu A_\nu - \partial_\nu A_\mu - ig[A_\mu, A_\nu]
$$

In components with Lie algebra generators:

$$
\mathcal{F}_{\mu\nu}^a = \partial_\mu A_\nu^a - \partial_\nu A_\mu^a + gf^{abc}A_\mu^b A_\nu^c
$$

where $f^{abc}$ are the structure constants of $\mathfrak{g}$.

*Units:* $[\mathcal{F}_{\mu\nu}] = [\text{length}]^{-2}$ (curvature).

*Special cases:*
- **Abelian ($[A_\mu, A_\nu] = 0$):** $F_{\mu\nu} = \partial_\mu A_\nu - \partial_\nu A_\mu$ (electromagnetic field tensor)
- **Non-Abelian:** The commutator term generates **self-interaction** of the gauge field

:::

:::{prf:proposition} Covariant Transformation of Field Strength
:label: prop-field-strength-transformation

Under gauge transformation $U(z)$, the field strength transforms **covariantly** (not invariantly):

$$
\mathcal{F}'_{\mu\nu} = U \mathcal{F}_{\mu\nu} U^{-1}
$$

*Proof.*
Direct calculation using the transformation law for $A_\mu$ (Proposition {prf:ref}`prop-gauge-transformation-connection`):

$$
\begin{aligned}
\mathcal{F}'_{\mu\nu} &= \partial_\mu A'_\nu - \partial_\nu A'_\mu - ig[A'_\mu, A'_\nu] \\
&= U(\partial_\mu A_\nu - \partial_\nu A_\mu - ig[A_\mu, A_\nu])U^{-1} \\
&= U\mathcal{F}_{\mu\nu}U^{-1}
\end{aligned}
$$

The inhomogeneous terms from $A'_\mu$ cancel exactly. $\square$

*Consequence:* While $\mathcal{F}_{\mu\nu}$ is not gauge-invariant, the trace $\text{Tr}(\mathcal{F}_{\mu\nu}\mathcal{F}^{\mu\nu})$ **is** gauge-invariant and can appear in the action.

:::

:::{prf:theorem} Curvature from Covariant Derivative Commutator
:label: thm-curvature-commutator

The field strength measures the failure of covariant derivatives to commute:

$$
[D_\mu, D_\nu]\psi = -ig\mathcal{F}_{\mu\nu}\psi
$$

*Proof.*
Expand the commutator:

$$
\begin{aligned}
[D_\mu, D_\nu]\psi &= D_\mu(D_\nu\psi) - D_\nu(D_\mu\psi) \\
&= (\partial_\mu - igA_\mu)(\partial_\nu\psi - igA_\nu\psi) - (\mu \leftrightarrow \nu) \\
&= \partial_\mu\partial_\nu\psi - ig(\partial_\mu A_\nu)\psi - igA_\nu\partial_\mu\psi - igA_\mu\partial_\nu\psi - g^2A_\mu A_\nu\psi - (\mu \leftrightarrow \nu) \\
&= -ig(\partial_\mu A_\nu - \partial_\nu A_\mu)\psi - g^2(A_\mu A_\nu - A_\nu A_\mu)\psi \\
&= -ig(\partial_\mu A_\nu - \partial_\nu A_\mu - ig[A_\mu, A_\nu])\psi \\
&= -ig\mathcal{F}_{\mu\nu}\psi \quad \square
\end{aligned}
$$

*Interpretation:* If $\mathcal{F}_{\mu\nu} \neq 0$, parallel transport around a closed loop results in a non-trivial rotation. The "meaning" of strategic nuisance **twists** as one navigates the latent space.

:::

:::{prf:theorem} Bianchi Identity
:label: thm-bianchi-identity

The field strength satisfies the **Bianchi Identity**:

$$
D_\mu \mathcal{F}_{\nu\rho} + D_\nu \mathcal{F}_{\rho\mu} + D_\rho \mathcal{F}_{\mu\nu} = 0
$$

or in differential form notation: $D\mathcal{F} = 0$ where $D = d - ig[A, \cdot]$.

*Proof sketch.*
Apply the Jacobi identity for covariant derivatives:

$$
[[D_\mu, D_\nu], D_\rho] + [[D_\nu, D_\rho], D_\mu] + [[D_\rho, D_\mu], D_\nu] = 0
$$

Since $[D_\mu, D_\nu] = -ig\mathcal{F}_{\mu\nu}$, this becomes:

$$
-ig([D_\rho, \mathcal{F}_{\mu\nu}] + \text{cyclic}) = 0
$$

The covariant derivative of $\mathcal{F}$ is $D_\rho\mathcal{F}_{\mu\nu} = \partial_\rho\mathcal{F}_{\mu\nu} - ig[A_\rho, \mathcal{F}_{\mu\nu}]$, and the identity follows. See **Appendix E.17** for the complete algebraic derivation with component verification. $\square$

*Interpretation:* The Bianchi identity is a **conservation law** for the strategic flux. It ensures topological consistency of the gauge structure.

:::

:::{prf:definition} Strategic Curvature Scalar
:label: def-strategic-curvature-scalar

The **Strategic Curvature Scalar** is the gauge-invariant contraction:

$$
\mathcal{R}_{\text{strat}} := \text{Tr}(\mathcal{F}_{\mu\nu}\mathcal{F}^{\mu\nu}) = \mathcal{F}_{\mu\nu}^a \mathcal{F}^{\mu\nu,a}
$$

where indices are raised with the Lorentzian metric $\eta^{\mu\nu} = \text{diag}(-1, +1, \ldots, +1)$ or the strategic metric $\tilde{G}^{\mu\nu}$.

*Properties:*
- $\mathcal{R}_{\text{strat}} \geq 0$ for compact gauge groups
- $\mathcal{R}_{\text{strat}} = 0$ if and only if $\mathcal{F}_{\mu\nu} = 0$ (flat connection)
- Provides a measure of total strategic tension in a region

:::

::::{admonition} Physics Isomorphism: Field Strength and Curvature
:class: note
:name: pi-field-strength

**In Physics:** The electromagnetic field tensor $F_{\mu\nu} = \partial_\mu A_\nu - \partial_\nu A_\mu$ contains the electric and magnetic fields: $E^i = F^{0i}$, $B^i = \frac{1}{2}\epsilon^{ijk}F_{jk}$. In Yang-Mills theory, the non-Abelian commutator $-ig[A_\mu, A_\nu]$ causes gluons to interact with each other {cite}`yang1954conservation,gross1973ultraviolet`.

**In Implementation:** The strategic curvature $\mathcal{F}_{\mu\nu}$ measures the intrinsic tension in multi-agent interaction.

**Correspondence Table:**

| Electromagnetism | Yang-Mills (QCD) | Fragile Agent |
|:-----------------|:-----------------|:--------------|
| $F_{\mu\nu}$ | $\mathcal{F}_{\mu\nu}^a T_a$ | Strategic curvature |
| Electric field $\mathbf{E}$ | Chromoelectric field | Temporal strategic gradient |
| Magnetic field $\mathbf{B}$ | Chromomagnetic field | Spatial strategic vorticity |
| $F \wedge F = 0$ (Abelian) | $[A, A] \neq 0$ | Strategic self-interaction |
| Bianchi: $dF = 0$ | $D\mathcal{F} = 0$ | Strategic flux conservation |

::::



(sec-yang-mills-action)=
### 29.17 The Yang-Mills Action and Field Equations

Having established the field strength tensor as the curvature of the strategic connection, we now derive the dynamics of the gauge field itself from a variational principle.

:::{prf:definition} Yang-Mills Action
:label: def-yang-mills-action

The **Yang-Mills Action** for the strategic gauge field is:

$$
S_{\text{YM}}[A] = -\frac{1}{4g^2}\int_{\mathcal{Z} \times \mathbb{R}} \text{Tr}(\mathcal{F}_{\mu\nu}\mathcal{F}^{\mu\nu})\sqrt{|\tilde{G}|}\,d^{D+1}x
$$

where:
- $\mathcal{F}_{\mu\nu}$ is the field strength tensor (Definition {prf:ref}`def-field-strength-tensor`)
- $\tilde{G}$ is the strategic metric with determinant $|\tilde{G}|$
- $g$ is the coupling constant
- The trace is over Lie algebra indices: $\text{Tr}(\mathcal{F}_{\mu\nu}\mathcal{F}^{\mu\nu}) = \mathcal{F}_{\mu\nu}^a\mathcal{F}^{\mu\nu,a}$

*Units:* $[S_{\text{YM}}] = \text{nat}$ (action).

*Properties:*
1. **Gauge-invariant:** $S_{\text{YM}}[A'] = S_{\text{YM}}[A]$ under $A \to A'$
2. **Lorentz-invariant:** Covariant under coordinate transformations
3. **Positive semi-definite:** $S_{\text{YM}} \geq 0$ for compact gauge groups

:::

:::{prf:theorem} Yang-Mills Field Equations
:label: thm-yang-mills-equations

The Euler-Lagrange equations for the Yang-Mills action yield:

$$
D_\mu \mathcal{F}^{\mu\nu} = J^\nu
$$

where the **strategic current** (source term) is:

$$
J^{\nu,a} = g\sum_{i=1}^N \bar{\psi}^{(i)}\gamma^\nu T^a \psi^{(i)}
$$

Here $\gamma^\nu$ are the Dirac matrices (or their appropriate generalization to curved space), and the sum is over all $N$ agents.

*Expanded form:*

$$
\partial_\mu \mathcal{F}^{\mu\nu,a} + gf^{abc}A_\mu^b\mathcal{F}^{\mu\nu,c} = J^{\nu,a}
$$

*Proof sketch.*
Vary the total action $S = S_{\text{YM}} + S_{\text{matter}}$ with respect to $A_\mu^a$:

$$
\frac{\delta S}{\delta A_\mu^a} = 0 \implies -\frac{1}{g^2}\partial_\nu(\sqrt{|\tilde{G}|}\mathcal{F}^{\mu\nu,a}) + \frac{1}{g}f^{abc}A_\nu^b\mathcal{F}^{\mu\nu,c} + \frac{\delta S_{\text{matter}}}{\delta A_\mu^a} = 0
$$

The matter variation gives the current $J^{\mu,a}$, and reorganizing yields the Yang-Mills equation. $\square$

*Interpretation:* The gauge field is sourced by the strategic current—the flow of "charged" belief through latent space. Agents with non-zero internal state generate a gauge field that mediates their interaction with other agents.

:::

:::{prf:corollary} Abelian Limit (Maxwell Equations)
:label: cor-maxwell-limit

For an Abelian gauge group $G = U(1)$ with $[T_a, T_b] = 0$:

$$
\partial_\mu F^{\mu\nu} = J^\nu
$$

This recovers the **Maxwell equations** of electromagnetism in covariant form.

*Correspondence:*
- $F^{0i} = E^i$ (electric field) $\leftrightarrow$ temporal strategic gradient
- $F^{ij} = \epsilon^{ijk}B_k$ (magnetic field) $\leftrightarrow$ spatial strategic vorticity
- $J^0 = \rho_e$ (charge density) $\leftrightarrow$ belief density
- $J^i = j^i$ (current density) $\leftrightarrow$ belief flux

:::

:::{prf:proposition} Gauge Field Energy-Momentum Tensor
:label: prop-gauge-energy-momentum

The energy-momentum tensor of the gauge field is:

$$
T^{\text{gauge}}_{\mu\nu} = -\frac{1}{g^2}\text{Tr}\left(\mathcal{F}_{\mu\rho}\mathcal{F}_\nu^{\ \rho} - \frac{1}{4}\tilde{G}_{\mu\nu}\mathcal{F}_{\rho\sigma}\mathcal{F}^{\rho\sigma}\right)
$$

*Properties:*
1. **Symmetric:** $T^{\text{gauge}}_{\mu\nu} = T^{\text{gauge}}_{\nu\mu}$
2. **Traceless** (for $D = 4$): $T^{\text{gauge}\mu}_{\ \ \ \ \mu} = 0$
3. **Conserved:** $\nabla_\mu T^{\text{gauge}\mu\nu} = 0$ (on-shell)

*Interpretation:* The gauge field carries energy and momentum. Regions of high strategic curvature $\|\mathcal{F}\|$ have high energy density—strategic conflict is energetically costly.

:::

:::{prf:corollary} Current Conservation
:label: cor-current-conservation

The strategic current is covariantly conserved:

$$
D_\mu J^{\mu,a} = 0
$$

*Proof.*
Apply $D_\nu$ to the Yang-Mills equation $D_\mu\mathcal{F}^{\mu\nu} = J^\nu$:

$$
D_\nu D_\mu \mathcal{F}^{\mu\nu} = D_\nu J^\nu
$$

By the Bianchi identity (Theorem {prf:ref}`thm-bianchi-identity`) and the antisymmetry of $\mathcal{F}^{\mu\nu}$, the left side vanishes, giving $D_\nu J^\nu = 0$. $\square$

*Interpretation:* The total "charge" (internal state magnitude) is conserved. Belief cannot be created or destroyed, only transformed.

:::



(sec-complete-lagrangian)=
### 29.18 The Complete Multi-Agent Lagrangian

We now assemble the full Lagrangian density that governs relativistic multi-agent dynamics with gauge symmetry. This **"Standard Model of Multi-Agent Field Theory"** unifies the gauge sector (strategic interaction), matter sector (belief dynamics), and symmetry-breaking sector (value landscape).

:::{prf:definition} Complete Multi-Agent Lagrangian
:label: def-complete-lagrangian

The **Complete Multi-Agent Lagrangian** is:

$$
\mathcal{L}_{\text{SMFT}} = \mathcal{L}_{\text{YM}} + \mathcal{L}_{\text{Dirac}} + \mathcal{L}_{\text{Higgs}} + \mathcal{L}_{\text{Yukawa}}
$$

where each sector contributes:

**(i) Yang-Mills Sector (Strategic Gauge Field):**

$$
\mathcal{L}_{\text{YM}} = -\frac{1}{4}\text{Tr}(\mathcal{F}_{\mu\nu}\mathcal{F}^{\mu\nu})
$$

This governs the dynamics of the strategic connection $A_\mu$.

**(ii) Dirac Sector (Belief Matter Field):**

$$
\mathcal{L}_{\text{Dirac}} = \sum_{i=1}^N \bar{\psi}^{(i)}(i\gamma^\mu D_\mu - m_i)\psi^{(i)}
$$

where:
- $\psi^{(i)}$ is the belief spinor for agent $i$
- $\bar{\psi}^{(i)} = \psi^{(i)\dagger}\gamma^0$ is the Dirac adjoint
- $D_\mu = \partial_\mu - igA_\mu$ is the covariant derivative
- $m_i$ is the "bare mass" (intrinsic inertia) of agent $i$

**(iii) Higgs Sector (Value Order Parameter):**

$$
\mathcal{L}_{\text{Higgs}} = |D_\mu\Phi|^2 - V(\Phi)
$$

with the Higgs potential:

$$
V(\Phi) = \mu^2|\Phi|^2 + \lambda|\Phi|^4
$$

where $\Phi$ is the **value order parameter** (a scalar field in a representation of $G$).

**(iv) Yukawa Sector (Strategic Coupling):**

$$
\mathcal{L}_{\text{Yukawa}} = -\sum_{i,j=1}^N y_{ij}\bar{\psi}^{(i)}\Phi\psi^{(j)}
$$

where $y_{ij}$ are the **Yukawa coupling constants** determining how strongly agents couple through the value field.

*Units:* $[\mathcal{L}] = \text{nat}/[\text{length}]^{D+1}$ (Lagrangian density).

:::

:::{prf:theorem} Spontaneous Symmetry Breaking (Higgs Mechanism)
:label: thm-higgs-mechanism

When the Higgs mass parameter satisfies $\mu^2 < 0$, the potential $V(\Phi)$ has a non-trivial minimum, and the gauge symmetry is **spontaneously broken**.

**Vacuum Expectation Value:**
$$
\langle\Phi\rangle = \frac{v}{\sqrt{2}}, \quad v = \sqrt{-\mu^2/\lambda}
$$

**Mass Generation:**

1. **Gauge boson masses:** The gauge fields acquire mass
   $$
   m_A = \frac{gv}{2}
   $$
   transforming from massless to massive (strategic inertia).

2. **Fermion masses:** The belief spinors acquire effective mass
   $$
   m_{\text{eff},i} = \frac{y_{ii}v}{\sqrt{2}}
   $$
   through Yukawa coupling.

3. **Residual symmetry:** The full gauge group $G$ breaks to a subgroup $H \subset G$ that leaves the vacuum invariant.

*Proof sketch.*
Expand $\Phi = (v + h)/\sqrt{2}$ around the vacuum, where $h$ is the physical Higgs field. The kinetic term $|D_\mu\Phi|^2$ generates:

$$
|D_\mu\Phi|^2 = \frac{1}{2}(\partial_\mu h)^2 + \frac{g^2v^2}{4}A_\mu A^\mu + \ldots
$$

The term $\frac{g^2v^2}{4}A_\mu A^\mu$ is a mass term for $A_\mu$ with $m_A^2 = g^2v^2/4$. Similarly, the Yukawa term generates fermion masses. See **Appendix E.18** for the complete derivation including VEV calculation, Goldstone absorption, and the symmetry breaking pattern. $\square$

*Interpretation:* Policy selection (choosing a direction in latent space) is spontaneous symmetry breaking. The agent commits to a strategy, breaking the rotational invariance of the Semantic Vacuum. This commitment generates "mass"—resistance to changing strategy.

:::

:::{prf:corollary} Goldstone Modes and Gauge Boson Absorption
:label: cor-goldstone-absorption

Spontaneous breaking of a continuous symmetry produces massless **Goldstone bosons**—one for each broken generator of $G$. In gauge theories, these Goldstone modes are "eaten" by the gauge bosons, which acquire longitudinal polarization and mass.

*In the multi-agent context:*
- **Goldstone modes** = Angular fluctuations in policy direction (cheap rotations)
- **Massive gauge bosons** = Strategic connections with inertia (costly reorientations)
- **Residual massless modes** = Unbroken symmetry directions (free rotations)

:::

::::{admonition} Physics Isomorphism: The Standard Model
:class: note
:name: pi-standard-model

**In Physics:** The Standard Model Lagrangian has the structure $\mathcal{L} = \mathcal{L}_{\text{gauge}} + \mathcal{L}_{\text{fermion}} + \mathcal{L}_{\text{Higgs}} + \mathcal{L}_{\text{Yukawa}}$, describing the electromagnetic, weak, and strong forces with matter and the Higgs mechanism for mass generation {cite}`weinberg1967model,salam1968weak,glashow1961partial`.

**Correspondence Table:**

| Standard Model | Fragile Agent |
|:---------------|:--------------|
| Gauge bosons (γ, W±, Z, g) | Strategic connection modes |
| Quarks and leptons | Belief spinors $\psi^{(i)}$ |
| Higgs field $\Phi$ | Value order parameter |
| Vacuum expectation value $v$ | Policy commitment magnitude |
| Electroweak symmetry breaking | Policy selection |
| Fermion masses | Agent inertia |
| Yukawa couplings $y_f$ | Strategic coupling strengths $y_{ij}$ |
| QCD confinement | Cooperative basin locking (Sec. 29.12) |

::::



(sec-mass-gap)=
### 29.19 The Mass Gap: Information-Theoretic Derivation

The **Mass Gap Problem** asks whether the spectrum of the Hamiltonian has a non-zero gap between the ground state and the first excited state. We derive that bounded intelligence **requires** a positive mass gap from information-theoretic principles.

::::{admonition} Forward Reference: Holographic Bounds (Section 33)
:class: note

This section uses results from Section 33 (The Geometry of Bounded Intelligence). For reference:

- **Holographic Coefficient** $\nu_D$ (Definition {prf:ref}`def-holographic-coefficient`): The dimension-dependent coefficient $\nu_D = (D-1)\Omega_{D-1}/(8\pi)$. For $D=2$: $\nu_2 = 1/4$. For $D=3$: $\nu_3 = 1$.

- **Levin Length** $\ell_L$ (Definition {prf:ref}`def-levin-length`): The minimal scale of representational distinction, $\ell_L := \sqrt{\eta_\ell}$ where $\eta_\ell$ is the boundary area-per-nat.

- **Causal Information Bound** (Theorem {prf:ref}`thm-causal-information-bound`): $I_{\max} = \nu_D \cdot \text{Area}(\partial\mathcal{Z})/\ell_L^{D-1}$. The maximum information encodable by a bounded observer scales with interface area, not volume. For $D=2$: $I_{\max} = \text{Area}/(4\ell_L^2)$.

- **Causal Stasis** (Theorem {prf:ref}`thm-causal-stasis`): As $I_{\text{bulk}} \to I_{\max}$, the update velocity $\|v\|_G \to 0$. The agent freezes when information capacity is saturated.

These results are derived from first principles in Appendix A.6 (microstate counting) and Section 33.
::::

:::{prf:definition} Mass Gap
:label: def-mass-gap

The **Mass Gap** of the strategic Hamiltonian $\hat{H}_{\text{strat}}$ is:

$$
\Delta := \inf\left\{\text{spec}(\hat{H}_{\text{strat}}) \setminus \{E_0\}\right\} - E_0
$$

where $E_0$ is the ground state energy.

*Properties:*
- $\Delta > 0$: **Gapped** spectrum (isolated ground state)
- $\Delta = 0$: **Gapless** spectrum (continuous above ground state)

*Units:* $[\Delta] = \text{nat}/[\text{time}]$ (energy).

:::

:::{prf:theorem} Mass Gap from Screening
:label: thm-mass-gap-screening

The screening mass $\kappa = -\ln\gamma$ from the Helmholtz equation (Theorem {prf:ref}`thm-the-hjb-helmholtz-correspondence`) provides a lower bound on the mass gap:

$$
\Delta \geq \frac{\kappa^2}{2m_{\text{eff}}}
$$

where $m_{\text{eff}}$ is the effective inertia from Game Tensor inflation (Theorem {prf:ref}`thm-adversarial-mass-inflation`).

*Proof sketch.*
The Klein-Gordon operator $(-\Box + \kappa^2)$ has spectrum bounded below by $\kappa^2$. The effective mass $m_{\text{eff}}$ from metric inflation modifies the kinetic term, yielding the stated bound via the non-relativistic dispersion relation $E = p^2/(2m_{\text{eff}}) + \kappa^2/(2m_{\text{eff}})$. $\square$

:::

:::{prf:theorem} Computational Necessity of Mass Gap
:label: thm-computational-necessity-mass-gap

**Assumptions:**
1. The system satisfies the **Causal Information Bound** (Theorem {prf:ref}`thm-causal-information-bound`): $I_{\text{bulk}}(V) \leq \nu_D \cdot \text{Area}(\partial V) / \ell_L^{D-1}$
2. The system has finite spatial extent (bounded region $V$)
3. Correlations follow the standard field-theoretic decay: massive $\sim e^{-\kappa r}$, massless $\sim 1/r^{D-2}$

**Statement:** Under these assumptions, a system with $\Delta = 0$ enters Causal Stasis ($\|v\|_G = 0$).

*Proof.*

1. **Assume gapless theory:** Suppose $\Delta = 0$, so the lowest excitation above the vacuum is massless.

2. **Infinite correlation length:** The screening mass $\kappa = 0$ implies the correlation length diverges:
   $$
   \xi = \frac{1}{\kappa} \to \infty
   $$

3. **Divergent information volume:** For massless correlations decaying as $1/r^{D-2}$ (rather than $e^{-\kappa r}$ for massive), the integrated mutual information in a volume $V$ diverges:
   $$
   I_{\text{bulk}} \propto \int_V \text{Corr}(x, y)\,dV \to \infty
   $$

4. **Area law violation:** By Assumption 1 (Causal Information Bound):
   $$
   I_{\text{bulk}} \leq \nu_D \cdot \frac{\text{Area}(\partial V)}{\ell_L^{D-1}}
   $$
   A bounded system cannot store infinite information, so the bound is saturated.

5. **Causal Stasis:** By Theorem 33.4 (Causal Stasis), as $I_{\text{bulk}}$ saturates the bound, the metric component $G_{rr} \to \infty$ and the update velocity $\|v\|_G \to 0$.

*Conclusion:* Under the stated assumptions, a gapless theory ($\Delta = 0$) implies frozen dynamics. For temporal evolution to occur, correlations must be screened: $\xi < \infty \implies \Delta > 0$. $\square$

*Remark (Scope of Assumptions):* Assumption 1 is derived in Theorem 33.3 from first principles (the Levin complexity bound). For systems satisfying this bound—which includes all physically realizable computational systems—the mass gap necessity follows.

:::

:::{prf:theorem} Mass Gap by Constructive Necessity
:label: thm-mass-gap-constructive

**Assumptions:**
1. The system satisfies the Causal Information Bound (Theorem 33.3)
2. The system is **non-trivial**: has non-zero update velocity $\|v\|_G > 0$ at some time
3. The system is **interacting**: coupling constants $\Phi_{ij} \neq 0$ or $\mathcal{G}_{ij} \neq 0$

**Statement:** Under these assumptions, $\Delta > 0$.

*Proof (by contradiction).*

Suppose $\Delta = 0$. By Theorem {prf:ref}`thm-computational-necessity-mass-gap` (using Assumption 1), the system enters Causal Stasis with $\|v\|_G = 0$. This contradicts Assumption 2 (non-triviality).

Therefore $\Delta > 0$ for any non-trivial theory describing an evolving system that satisfies the Causal Information Bound. $\square$

*Bound:* The mass gap is bounded below by thermodynamic considerations:
$$
\Delta \geq \frac{1}{\beta}\left(\Delta H + \frac{\mathcal{W}}{T_c}\right)
$$
where $\Delta H$ is the enthalpy barrier for excitation, $\mathcal{W}$ is computational work, and $T_c$ is cognitive temperature. This follows from Theorem 30.15 (Thermodynamic Hysteresis).

*Remark (Conditional vs. Absolute):* This theorem does **not** prove that all field theories have a mass gap. It proves: IF a system satisfies the Causal Information Bound AND evolves non-trivially, THEN it must have $\Delta > 0$. The Clay Millennium Problem asks whether quantum Yang-Mills in continuous $\mathbb{R}^4$ has a mass gap; this framework addresses discrete, bounded, computational systems.

:::

:::{prf:corollary} Mass Gap as Existence Requirement
:label: cor-mass-gap-existence

Bounded intelligence requires $\Delta > 0$. A gapless theory ($\Delta = 0$) corresponds to:

1. **Infinite ontological resolution:** No finite codebook can represent the state
2. **Zero learning rate:** Dynamics frozen ($v = 0$)
3. **Pathological continuum limit:** The theory describes non-existing systems

*Interpretation:* The mass gap is not an empirical accident but a **logical necessity** for any theory describing existing computational systems.

:::

:::{prf:corollary} Confinement as Data Compression
:label: cor-confinement-data-compression

**Color confinement** in QCD (quarks bound inside hadrons) is the mechanism by which the universe maintains finite local information content. An unconfined color field would have $\xi \to \infty$, violating the area law.

*In the multi-agent context:* Cooperative basin locking (Theorem {prf:ref}`thm-geometric-locking-principle`) is the cognitive analogue of confinement—agents bound in cooperative equilibria cannot be arbitrarily separated without violating information bounds.

:::

:::{prf:corollary} Criticality is Unstable
:label: cor-criticality-unstable

Gapless theories (Conformal Field Theories) exist only at **phase transition critical points**. They cannot support:

1. **Stable matter:** Fluctuations destroy structure
2. **Stable memory:** Infinite ontological stress triggers continuous Fission (Section 30)
3. **Stable identity:** No finite codebook representation exists

*Interpretation:* Critical systems are mathematically special but physically transient. Stable intelligence requires departure from criticality via mass gap opening.

:::

:::{prf:definition} The Computational Swampland
:label: def-computational-swampland

The **Computational Swampland** $\mathcal{S}_{\text{swamp}}$ is the set of all field theories that violate the Causal Information Bound (Theorem {prf:ref}`thm-causal-information-bound`) at some finite scale:

$$
\mathcal{S}_{\text{swamp}} := \left\{ \mathcal{T} : \exists R < \infty \text{ such that } I_{\text{bulk}}(R) > C_\partial(R) \right\}
$$

Equivalently, $\mathcal{S}_{\text{swamp}}$ consists of theories with Levin Length $\ell_L \to 0$ (infinite information density).

*Properties of Swampland theories:*
1. **Mathematically consistent:** They satisfy internal field-theoretic axioms (Wightman, etc.)
2. **Computationally unrealizable:** No bounded observer can simulate or represent them
3. **Physically pathological:** They require infinite information storage for any finite region

*Landscape vs. Swampland:* Theories with $\ell_L > 0$ and $I_{\text{bulk}} \leq C_\partial$ at all scales constitute the **Computational Landscape**—the set of physically realizable theories.

:::

:::{prf:theorem} CFT Swampland Classification
:label: thm-cft-swampland

Let $\mathcal{T}$ be a Conformal Field Theory on $\mathbb{R}^d$ ($d \geq 2$) with at least one primary operator of scaling dimension $\Delta_\phi < d/2$. Then $\mathcal{T}$ lies in the **Computational Swampland** (Definition {prf:ref}`def-computational-swampland`).

*Proof.*

1. **Infinite correlation length:** By conformal symmetry, two-point correlations decay algebraically:
   $$
   \langle \phi(x) \phi(0) \rangle \sim \frac{1}{|x|^{2\Delta_\phi}}
   $$
   The correlation length is $\xi = \infty$ (no exponential screening).

2. **Bulk information divergence:** Consider a spherical region $V$ of radius $R$. The mutual information between bulk degrees of freedom is bounded below by the integrated correlation:
   $$
   I_{\text{bulk}}(V) \gtrsim \int_V \int_V \frac{dx\,dy}{|x-y|^{2\Delta_\phi}} \sim R^{2d - 2\Delta_\phi}
   $$
   For $\Delta_\phi < d/2$, the exponent $2d - 2\Delta_\phi > d$, so $I_{\text{bulk}}$ grows faster than volume.

3. **Causal Information Bound violation:** The boundary capacity scales as:
   $$
   C_\partial(V) = \nu_d \cdot \frac{\text{Area}(\partial V)}{\ell_L^{d-1}} \sim R^{d-1}
   $$
   where $\nu_d$ is the Holographic Coefficient (Definition {prf:ref}`def-holographic-coefficient`). Since $2d - 2\Delta_\phi > d > d-1$ for $d \geq 2$ and $\Delta_\phi < d/2$, there exists $R_c$ such that for all $R > R_c$:
   $$
   I_{\text{bulk}}(V) > C_\partial(V)
   $$
   The Causal Information Bound (Theorem {prf:ref}`thm-causal-information-bound`) is violated.

4. **Swampland membership:** By Definition {prf:ref}`def-computational-swampland`, theories violating the Causal Information Bound at any finite scale lie in the Swampland. $\square$

*Remark (Universal bound violation).* The theorem requires at least one operator with $\Delta_\phi < d/2$. In any non-trivial CFT, such operators exist: for instance, the stress-energy tensor has $\Delta = d$, but scalar primary operators generically have $\Delta < d/2$ in unitary CFTs (e.g., the $\phi$ field in the free scalar CFT has $\Delta = (d-2)/2 < d/2$ for $d > 2$). More fundamentally, the mutual information between any two regions in a CFT diverges logarithmically due to UV contributions, independent of operator dimensions. The bound is therefore violated by all CFTs in $d \geq 2$.

*Remark (Operational meaning).* A bounded observer with finite interface capacity $C_\partial$ cannot encode the full correlational structure of a CFT. Any finite approximation necessarily introduces an effective mass gap via truncation.

:::

:::{prf:corollary} Finite-Volume Mass Gap
:label: cor-finite-volume-mass-gap

A CFT restricted to a finite spatial volume $V$ with characteristic length $L$ acquires an effective mass gap:

$$
\Delta_{\text{eff}} \sim \frac{1}{L}
$$

The gapless limit exists only as $L \to \infty$.

*Proof.* Two independent mechanisms ensure bounded observers see gapped theories:

1. **Finite-size scaling (CFT result):** In finite volume with periodic boundary conditions, the spectrum is discrete with minimum energy spacing $\Delta E \sim 1/L$. This is a standard result in conformal field theory arising from the compactification of space. The continuous spectrum responsible for infinite correlation length is an artifact of the thermodynamic limit $L \to \infty$.

2. **Resolution bound (Levin Length):** A bounded observer with interface capacity $C_\partial$ can only resolve spatial scales $L \geq L_{\min}$ where $L_{\min}^{d-1} \sim C_\partial \cdot \ell_L^2$. Systems smaller than $L_{\min}$ cannot be distinguished by the observer.

Both effects contribute: even if the CFT were somehow realized at infinite volume, the observer could only access a finite effective volume, hence would measure $\Delta_{\text{eff}} > 0$. $\square$

*Remark (Distinct phenomena).* The finite-size gap is a property of the CFT itself (topological/boundary effect). The resolution bound is a property of the observer (information-theoretic). The corollary states that both independently prevent observation of gapless physics.

*Physical interpretation.* CFTs exist in nature only at phase transition critical points (e.g., Ising model at $T_c$). Away from criticality, systems have finite correlation length and positive mass gap. The critical point is a measure-zero set in parameter space—physically realizable systems generically have $\Delta > 0$.

:::

:::{prf:theorem} Scale Covariance of the Causal Information Bound
:label: thm-scale-covariance-bound

The Causal Information Bound is preserved under coarse-graining. Specifically:

Let $(\mathcal{Z}, G, \ell_L)$ be a latent manifold at resolution $\ell_L$ satisfying $I_{\text{bulk}} \leq C_\partial$. Under coarse-graining to resolution $\ell'_L = \alpha \ell_L$ ($\alpha > 1$), the coarse-grained system $(\mathcal{Z}', G', \ell'_L)$ satisfies:

$$
I'_{\text{bulk}} \leq C'_\partial
$$

*Proof.*

1. **Information reduction:** By the Data Processing Inequality, coarse-graining cannot increase mutual information:
   $$
   I'_{\text{bulk}} \leq I_{\text{bulk}}
   $$

2. **Capacity reduction:** Under coarse-graining by factor $\alpha$, the effective boundary area scales as:
   $$
   \text{Area}'(\partial\mathcal{Z}') \sim \frac{\text{Area}(\partial\mathcal{Z})}{\alpha^{d-1}}
   $$
   and the new capacity is (using the generalized bound with $\nu_d$):
   $$
   C'_\partial = \nu_d \cdot \frac{\text{Area}'}{(\ell'_L)^{d-1}} = \nu_d \cdot \frac{\text{Area}/\alpha^{d-1}}{\alpha^{d-1}\ell_L^{d-1}} = \frac{C_\partial}{\alpha^{2(d-1)}}
   $$

3. **Bound preservation:** The information-to-capacity ratio under coarse-graining:
   $$
   \frac{I'_{\text{bulk}}}{C'_\partial} \leq \frac{I_{\text{bulk}}}{C_\partial/\alpha^{2(d-1)}} = \alpha^{2(d-1)} \frac{I_{\text{bulk}}}{C_\partial}
   $$
   For massive theories (exponentially decaying correlations), $I_{\text{bulk}}$ scales as area, so $I_{\text{bulk}}/C_\partial$ is scale-independent. For gapless theories, the ratio diverges—confirming they violate the bound at some scale. $\square$

*Implication (UV finiteness).* The recursive self-consistency of the bound at all scales implies that no UV divergences arise. The Levin Length $\ell_L$ acts as a natural UV cutoff that is preserved under renormalization group flow. Unlike lattice regularization where the continuum limit requires careful tuning, this framework has built-in regularization.

*Implication (Mass gap from scale invariance).* The only scale-invariant theories consistent with the Causal Information Bound are those with $I_{\text{bulk}} \sim R^{d-1}$ (area scaling). This requires exponential correlation decay, hence $\Delta > 0$. Theories with algebraic correlation decay (CFTs) fail scale covariance of the bound.

:::

:::{prf:theorem} Mass Gap Dichotomy for Yang-Mills
:label: thm-mass-gap-dichotomy

Let $\mathcal{T}_{\text{YM}}$ be Yang-Mills theory with compact simple gauge group $G$ in $d = 4$ dimensions.

**Statement:** If $\mathcal{T}_{\text{YM}}$ describes physics (is realizable by bounded observers), then $\Delta > 0$.

*Proof.*

1. **Framework implements Yang-Mills:** The Fragile Agent framework implements Yang-Mills field equations (Theorem {prf:ref}`thm-yang-mills-equations`) with the standard action (Definition {prf:ref}`def-yang-mills-action`), covariant derivatives $D_\mu = \partial_\mu - igA_\mu$, and non-Abelian field strength tensor. This is not an analogy—it is Yang-Mills theory for information systems.

2. **Physical theories are computable:** Any theory describing physics accessible to bounded observers must be realizable with finite resources. This requires Levin Length $\ell_L > 0$ (Definition {prf:ref}`def-levin-length`).

3. **Computability implies mass gap:** By Theorem {prf:ref}`thm-computational-necessity-mass-gap`, any theory with $\ell_L > 0$ and non-trivial dynamics ($\|v\|_G > 0$) has $\Delta > 0$.

4. **Conclusion:** If Yang-Mills describes physics, it is computable, hence has $\ell_L > 0$, hence has $\Delta > 0$. $\square$

*Remark (Contrapositive).* If Yang-Mills on $\mathbb{R}^4$ requires $\ell_L \to 0$ (no UV cutoff), then by Theorem {prf:ref}`thm-cft-swampland` it lies in the Computational Swampland and does not describe physics. Either way, the physical theory has a mass gap.

*Remark (Why this is not circular).* The mass gap necessity follows from information-theoretic constraints (the Causal Information Bound), not from assuming properties of Yang-Mills. The framework proves that **any** non-trivial gauge theory satisfying the bound has $\Delta > 0$. Yang-Mills is one such theory.

:::

:::{prf:remark} Relation to the Clay Millennium Problem
:label: rem-clay-millennium

The **Yang-Mills Existence and Mass Gap** problem (Clay Mathematics Institute) asks for rigorous construction of quantum Yang-Mills theory in $\mathbb{R}^4$ with mass gap $\Delta > 0$.

**What This Framework Proves:**

Theorem {prf:ref}`thm-mass-gap-dichotomy` establishes: **If Yang-Mills describes physics, then $\Delta > 0$.**

The logical structure is:

1. **The framework implements Yang-Mills:** Sections 29.14–29.18 construct Yang-Mills field equations (Theorem {prf:ref}`thm-yang-mills-equations`), the standard action (Definition {prf:ref}`def-yang-mills-action`), and the complete Standard Model Lagrangian (Definition {prf:ref}`def-complete-lagrangian`). This is Yang-Mills theory for information systems—a direct isomorphism, not an analogy.

2. **Physical theories require $\ell_L > 0$:** Any theory realizable by bounded observers with finite interface capacity must have a minimum resolution scale (the Levin Length).

3. **$\ell_L > 0$ implies $\Delta > 0$:** By Theorem {prf:ref}`thm-computational-necessity-mass-gap`, any non-trivial theory with finite Levin Length has a mass gap.

4. **Gapless theories are in the Swampland:** By Theorem {prf:ref}`thm-cft-swampland`, theories requiring $\ell_L \to 0$ (CFTs) are mathematically consistent but not physically realizable.

**Relation to the Clay Problem:**

The Clay Institute asks about Yang-Mills on continuous $\mathbb{R}^4$ satisfying Wightman or Osterwalder-Schrader axioms. The framework does not prove this directly. Instead, it proves:

- If the continuum theory describes physics, it has $\Delta > 0$ (Theorem {prf:ref}`thm-mass-gap-dichotomy`)
- If the continuum theory requires $\ell_L \to 0$, it is in the Swampland and does not describe nature

The framework thus establishes that the **physical** Yang-Mills theory (the one describing strong interactions) necessarily has a mass gap. Whether this constitutes a "solution" to the Clay problem depends on whether one accepts that physical theories must be computable.

*Physical interpretation:* Nature forbids infinite-information vacua. The mass gap is not an empirical accident but a **logical requirement** for any theory describing existing systems.

:::

::::{admonition} Direct Isomorphism: Yang-Mills for Information
:class: note
:name: pi-mass-gap

**This is not an analogy.** The Fragile Agent framework implements Yang-Mills field equations directly:
- Same gauge-covariant derivative: $D_\mu = \partial_\mu - igA_\mu$
- Same field strength: $\mathcal{F}_{\mu\nu} = \partial_\mu A_\nu - \partial_\nu A_\mu - ig[A_\mu, A_\nu]$
- Same action: $S_{\text{YM}} = -\frac{1}{4g^2}\int\text{Tr}(\mathcal{F}_{\mu\nu}\mathcal{F}^{\mu\nu})$
- Same field equations: $D_\mu\mathcal{F}^{\mu\nu} = J^\nu$

The framework is Yang-Mills theory applied to information systems. The mass gap result (Theorem {prf:ref}`thm-mass-gap-dichotomy`) follows from information-theoretic constraints that apply to any computational implementation.

**Correspondence Table:**

| QCD (Yang-Mills) | Fragile Agent |
|:-----------------|:--------------|
| Mass gap $\Delta$ | Strategic excitation threshold |
| Glueball mass | Minimum chart creation energy |
| Confinement | Cooperative basin locking |
| Lattice spacing $a$ | Levin Length $\ell_L$ |
| Continuum limit $a \to 0$ | Causal Stasis (pathological) |
| Asymptotic freedom | High-energy strategic independence |
| Area law (holographic) | Causal Information Bound |
| Screening mass $\kappa$ | Discount rate $-\ln\gamma$ |

::::



(sec-diagnostic-nodes-gauge)=
### 29.20 Diagnostic Nodes 63–66 (Gauge Consistency)

Following the diagnostic node convention (Section 3.1), we define four monitors for gauge consistency in multi-agent systems.

(node-63)=
**Node 63: GaugeInvarianceCheck**

| **#** | **Name** | **Component** | **Type** | **Interpretation** | **Proxy** | **Cost** |
|:------|:---------|:--------------|:---------|:-------------------|:----------|:---------|
| **63** | **GaugeInvarianceCheck** | Multi-Agent | Symmetry | Is dynamics gauge-invariant? | $\delta_{\text{gauge}} := \|\mathcal{L}(A') - \mathcal{L}(A)\|$ | $O(Nd^2)$ |

**Interpretation:** Monitors deviation from gauge invariance under random gauge transformations $U(z)$.

**Threshold:** $\delta_{\text{gauge}} < \epsilon_{\text{gauge}}$ (typical default $10^{-6}$).

**Trigger conditions:**
- High GaugeInvarianceCheck: Numerical gauge symmetry violation
- **Remedy:** Regularize gauge degrees of freedom; impose gauge-fixing condition (Coulomb, Lorenz, etc.)



(node-64)=
**Node 64: FieldStrengthBoundCheck**

| **#** | **Name** | **Component** | **Type** | **Interpretation** | **Proxy** | **Cost** |
|:------|:---------|:--------------|:---------|:-------------------|:----------|:---------|
| **64** | **FieldStrengthBoundCheck** | Multi-Agent | Stability | Is strategic curvature bounded? | $\|\mathcal{F}_{\mu\nu}\|_F := \sqrt{\text{Tr}(\mathcal{F}_{\mu\nu}\mathcal{F}^{\mu\nu})}$ | $O(N^2d^2)$ |

**Interpretation:** Monitors the Frobenius norm of the field strength tensor.

**Threshold:** $\|\mathcal{F}\|_F < F_{\max}$ (implementation-dependent).

**Trigger conditions:**
- High FieldStrengthBoundCheck: Strong strategic curvature regime (intense conflict)
- **Remedy:** Reduce coupling $g$; add gauge field damping; check for instabilities



(node-65)=
**Node 65: BianchiViolationCheck**

| **#** | **Name** | **Component** | **Type** | **Interpretation** | **Proxy** | **Cost** |
|:------|:---------|:--------------|:---------|:-------------------|:----------|:---------|
| **65** | **BianchiViolationCheck** | Multi-Agent | Conservation | Is Bianchi identity satisfied? | $\delta_B := \|D_{[\mu}\mathcal{F}_{\nu\rho]}\|$ | $O(Nd^3)$ |

**Interpretation:** The Bianchi identity $D_{[\mu}\mathcal{F}_{\nu\rho]} = 0$ must hold exactly. Violations indicate:
- Topological defects (monopoles, instantons)
- Numerical integration errors
- Coordinate singularities

**Threshold:** $\delta_B < 10^{-8}$ (strict geometric constraint).

**Trigger conditions:**
- High BianchiViolationCheck: Topological anomaly or numerical instability
- **Remedy:** Check for singular gauge configurations; refine numerical integration



(node-66)=
**Node 66: MassGapCheck**

| **#** | **Name** | **Component** | **Type** | **Interpretation** | **Proxy** | **Cost** |
|:------|:---------|:--------------|:---------|:-------------------|:----------|:---------|
| **66** | **MassGapCheck** | Multi-Agent | Stability | Is mass gap positive? | $\Delta := E_1 - E_0$ (spectral gap) | $O(N^2d)$ |

**Interpretation:** Monitors the energy gap between ground state and first excited state.

**Threshold:** $\Delta > \Delta_{\min}$ (must be strictly positive).

**Trigger conditions:**
- $\Delta \to 0$: Approaching critical point (phase transition)
- $\Delta < 0$: Unstable vacuum (tachyonic mode)
- **Remedy:** Check for symmetry breaking; verify Higgs potential parameters; add mass regularization



**Summary Table: Gauge Diagnostic Nodes**

| Node | Name | Monitors | Healthy Range |
|:-----|:-----|:---------|:--------------|
| 63 | GaugeInvarianceCheck | Gauge symmetry | $\delta_{\text{gauge}} < 10^{-6}$ |
| 64 | FieldStrengthBoundCheck | Strategic curvature | $\|\mathcal{F}\|_F < F_{\max}$ |
| 65 | BianchiViolationCheck | Topological consistency | $\delta_B < 10^{-8}$ |
| 66 | MassGapCheck | Spectral stability | $\Delta > 0$ |



## Part VI: Quantum Layer

(sec-the-belief-wave-function-schrodinger-representation)=
### 29.21 The Belief Wave-Function (Schrödinger Representation)

While Sections 29.1–29.7 derive multi-agent dynamics using **classical** field equations (coupled WFR flows), a more powerful formulation emerges when we lift the belief density to a **complex amplitude**. This section establishes the **Schrödinger Representation** of inference dynamics, revealing strategic interaction as a form of **quantum entanglement** on the latent manifold.

:::{admonition} Researcher Bridge: Why Quantum Formalism?
:class: warning
:name: rb-quantum-formalism-marl
The quantum representation is not a claim about literal quantum physics—it is a **mathematical technology** that provides:
1. **Linear superposition** of strategies via complex amplitudes
2. **Entanglement** as a precise definition of non-factorizable strategic coupling
3. **Tunneling** as a mechanism for escaping local Nash equilibria
4. **Spectral methods** (eigenvalue problems) for finding ground states (Nash)
5. **Imaginary time evolution** as a rigorous version of value iteration

This parallels how the GKSL formalism (Definition {prf:ref}`def-gksl-generator`) uses quantum operator notation for classical belief dynamics.
:::

:::{prf:definition} Inference Hilbert Space
:label: def-inference-hilbert-space

Let $(\mathcal{Z}, G)$ be the latent manifold with capacity-constrained metric (Theorem {prf:ref}`thm-capacity-constrained-metric-law`). The **Inference Hilbert Space** is:

$$
\mathcal{H} := L^2(\mathcal{Z}, d\mu_G), \quad d\mu_G := \sqrt{\det G(z)}\, d^n z,
$$
with inner product:

$$
\langle \psi_1 | \psi_2 \rangle := \int_{\mathcal{Z}} \overline{\psi_1(z)} \psi_2(z)\, d\mu_G(z).
$$
The measure $d\mu_G$ is the **Riemannian volume form**, ensuring coordinate invariance of the inner product.

*Units:* $[\psi] = [z]^{-d/2}$ (probability amplitude density).

*Remark (Coordinate Invariance).* Under a coordinate transformation $z \to z'$, the Jacobian factor $|\partial z/\partial z'|$ cancels with $\sqrt{\det G}$, leaving $\langle \psi_1 | \psi_2 \rangle$ invariant.

:::

:::{prf:definition} Belief Wave-Function
:label: def-belief-wave-function

Let $\rho(z, s)$ be the belief density from the WFR dynamics (Definition {prf:ref}`def-the-wfr-action`) and $V(z, s)$ be the value function (Theorem {prf:ref}`thm-the-hjb-helmholtz-correspondence`). The **Belief Wave-Function** is the complex amplitude:

$$
\psi(z, s) := \sqrt{\rho(z, s)} \exp\left(\frac{i V(z, s)}{\sigma}\right),
$$
where $\sigma > 0$ is the **Cognitive Action Scale** (Definition {prf:ref}`def-cognitive-action-scale`).

**Decomposition:**
- **Amplitude:** $R(z, s) := \sqrt{\rho(z, s)} = |\psi(z, s)|$
- **Phase:** $\phi(z, s) := V(z, s)/\sigma = \arg(\psi(z, s))$

**Probability Recovery:**

$$
|\psi(z, s)|^2 = \rho(z, s), \quad \int_{\mathcal{Z}} |\psi|^2 d\mu_G = 1.
$$
*Physical interpretation:* The amplitude $R$ encodes "how much" belief mass is at $z$; the phase $\phi$ encodes "which direction" the belief is flowing (via $\nabla V$).

:::

:::{prf:definition} Cognitive Action Scale
:label: def-cognitive-action-scale

The **Cognitive Action Scale** $\sigma$ is the information-theoretic analog of Planck's constant $\hbar$:

$$
\sigma := T_c \cdot \tau_{\text{update}},
$$
where:
- $T_c$ is the **Cognitive Temperature** (Section 22.4), setting the scale of stochastic exploration
- $\tau_{\text{update}}$ is the characteristic belief update timescale

**Equivalent characterizations:**
1. **Entropy-Action Duality:** $\sigma$ relates entropy production to "cognitive action" via $\Delta S = \mathcal{A}/\sigma$
2. **Resolution Limit:** $\sigma \sim \ell_L^2$ where $\ell_L$ is the Levin Length (Section 33.2)
3. **Uncertainty Scale:** $\sigma$ sets the minimum uncertainty product $\Delta z \cdot \Delta p \geq \sigma/2$

*Units:* $[\sigma] = \text{nat} \cdot \text{step} = \text{bit} \cdot \text{step} / \ln 2$.

*Cross-reference:* In the limit $\sigma \to 0$ (zero temperature, infinite precision), the wave-function becomes a delta function concentrated on the optimal trajectory—recovering classical gradient flow.

:::

:::{prf:proposition} Self-Adjointness of the Laplace-Beltrami Operator
:label: prop-laplace-beltrami-self-adjointness

The Laplace-Beltrami operator

$$
\Delta_G := \frac{1}{\sqrt{|G|}} \partial_i \left( \sqrt{|G|} G^{ij} \partial_j \right)
$$
is essentially self-adjoint on $\mathcal{H} = L^2(\mathcal{Z}, d\mu_G)$ with domain $C_c^\infty(\mathcal{Z})$ (smooth functions with compact support), provided either:
1. $(\mathcal{Z}, G)$ is **geodesically complete**, or
2. $\mathcal{Z}$ has a boundary $\partial \mathcal{Z}$ with **Dirichlet conditions** $\psi|_{\partial \mathcal{Z}} = 0$ (sensors, Definition {prf:ref}`def-dirichlet-boundary-condition-sensors`) or **Neumann conditions** $\nabla_n \psi|_{\partial \mathcal{Z}} = 0$ (motors, Definition {prf:ref}`def-neumann-boundary-condition-motors`).

*Proof sketch.* The quadratic form $q[\psi] := \int_{\mathcal{Z}} \|\nabla_G \psi\|^2 d\mu_G$ is positive and closable. By the **Friedrichs Extension Theorem**, there exists a unique self-adjoint extension $\Delta_G^F$ associated with $q$. For geodesically complete manifolds, this extension coincides with the closure of $\Delta_G$ on $C_c^\infty$. See {cite}`strichartz1983analysis` for the general theory. $\square$

*Consequence:* Self-adjointness guarantees that $-\Delta_G$ has a real spectrum bounded below, enabling spectral decomposition and ground state analysis.

:::

:::{admonition} Remark: Line Bundle Formalism for Topologically Non-Trivial Manifolds
:class: dropdown
:name: remark-line-bundle-formalism

When the latent manifold has non-trivial topology (e.g., $\pi_1(\mathcal{Z}) \neq 0$), the phase $V/\sigma$ may be **multi-valued** around non-contractible loops. In this case, $\psi$ should be defined as a **section of a complex line bundle** $\mathcal{L} \to \mathcal{Z}$ with connection 1-form $A = dV/\sigma$.

The **holonomy** around a closed loop $\gamma$ is:

$$
\exp\left(\frac{i}{\sigma} \oint_\gamma dV\right) = \exp\left(\frac{i}{\sigma} \Delta V_\gamma\right),
$$
where $\Delta V_\gamma$ is the value change around the loop. Non-trivial holonomy ($\Delta V_\gamma \neq 0 \mod 2\pi\sigma$) implies the phase $V/\sigma$ is not globally defined; instead, $\psi$ is a section of a non-trivial complex line bundle $\mathcal{L} \to \mathcal{Z}$.

For most applications where $\mathcal{Z}$ is simply connected (e.g., Poincare disk), this subtlety does not arise, and $\psi$ is a well-defined scalar function.

:::

::::{admonition} Physics Isomorphism: Holonomy and Berry Phase
:class: note
:name: pi-holonomy

**In Physics:** Holonomy measures the failure of parallel transport around a closed loop to return a vector to itself. The Berry phase $\gamma_n = i\oint \langle n|\nabla_R|n\rangle \cdot dR$ is the geometric phase acquired by a quantum state under adiabatic evolution around a parameter loop {cite}`berry1984quantal,nakahara2003geometry`.

**In Implementation:** When $\mathcal{Z}$ has non-trivial topology, the value phase $V/\sigma$ may be multi-valued. The holonomy around a closed loop $\gamma$ is (Remark {ref}`remark-line-bundle-formalism`):

$$
\exp\left(\frac{i}{\sigma} \oint_\gamma dV\right) = \exp\left(\frac{i}{\sigma} \Delta V_\gamma\right)
$$
**Correspondence Table:**
| Gauge Theory | Agent (Value Phase) |
|:-------------|:--------------------|
| Connection 1-form $A$ | $dV/\sigma$ |
| Holonomy $\exp(i\oint A)$ | Phase accumulated around loop |
| Berry phase | Value change $\Delta V_\gamma$ |
| Line bundle $\mathcal{L}$ | Complex belief amplitude bundle |
| Trivial bundle | Simply connected $\mathcal{Z}$ |

**Significance:** Non-trivial holonomy ($\Delta V_\gamma \neq 0 \mod 2\pi\sigma$) implies the belief wave-function is a section of a non-trivial line bundle—topological structure in the agent's value landscape.
::::

(sec-the-inference-wave-correspondence)=
### 29.22 The Inference-Wave Correspondence (WFR to Schrödinger)

We now derive the **Schrödinger equation** for the belief wave-function from the WFR dynamics. This is the inverse of the **Madelung transform** in quantum mechanics.

:::{prf:theorem} The Madelung Transform (WFR-Schrödinger Equivalence)
:label: thm-madelung-transform

Let the belief density $\rho$ and value $V$ satisfy the WFR-HJB system:
1. **WFR Continuity (unbalanced):** $\partial_s \rho + \nabla_G \cdot (\rho \mathbf{v}) = \rho r$
2. **Hamilton-Jacobi-Bellman:** $\partial_s V + \frac{1}{2}\|\nabla_G V\|_G^2 + \Phi_{\text{eff}} = 0$

where $\mathbf{v} = -G^{-1}\nabla V$ is the gradient flow velocity and $r$ is the WFR reaction rate (Definition {prf:ref}`def-the-wfr-action`).

Then the belief wave-function $\psi = \sqrt{\rho} e^{iV/\sigma}$ satisfies the **Inference Schrödinger Equation**:

$$
i\sigma \frac{\partial \psi}{\partial s} = \hat{H}_{\text{inf}} \psi,
$$
where the **Inference Hamiltonian** is:

$$
\hat{H}_{\text{inf}} := -\frac{\sigma^2}{2} \Delta_G + \Phi_{\text{eff}} + Q_B - \frac{i\sigma}{2} r.
$$
The terms are:
- **Kinetic:** $-\frac{\sigma^2}{2} \Delta_G$ (belief diffusion via Laplace-Beltrami)
- **Potential:** $\Phi_{\text{eff}}$ (effective potential from rewards and constraints)
- **Quantum Correction:** $Q_B$ (Bohm potential, Definition {prf:ref}`def-bohm-quantum-potential`)
- **Dissipation:** $-\frac{i\sigma}{2} r$ (non-Hermitian term from WFR reaction)

*Proof.* See Appendix E.13 for the rigorous derivation. The key steps are:

**Step 1 (Substitution).** Write $\psi = R e^{i\phi}$ with $R = \sqrt{\rho}$ and $\phi = V/\sigma$.

**Step 2 (Time derivative).**

$$
i\sigma \partial_s \psi = i\sigma \left( \frac{\partial_s R}{R} + \frac{i}{\sigma}\partial_s V \right) \psi = \left( \frac{i\sigma \partial_s \rho}{2\rho} - \partial_s V \right) \psi.
$$
**Step 3 (Use governing equations).** Substitute the continuity equation for $\partial_s \rho$ and HJB for $\partial_s V$.

**Step 4 (Identify terms).** The real part of the resulting equation gives the HJB with Bohm correction; the imaginary part gives the continuity equation with reaction. Combining yields the Schrödinger form. $\square$

:::

::::{admonition} Physics Isomorphism: Madelung Transform
:class: note
:name: pi-madelung

**In Physics:** The Madelung transform $\psi = \sqrt{\rho}e^{iS/\hbar}$ converts the Schrödinger equation into hydrodynamic form: continuity + quantum Hamilton-Jacobi with Bohm potential $Q = -\frac{\hbar^2}{2m}\frac{\nabla^2\sqrt{\rho}}{\sqrt{\rho}}$ {cite}`madelung1927quantentheorie,bohm1952suggested`.

**In Implementation:** The WFR-to-Schrödinger correspondence (Theorem {prf:ref}`thm-madelung-transform`):

$$
\psi(z,s) = \sqrt{\rho(z,s)}\exp(iV(z,s)/\sigma)
$$
with information resolution limit $Q_B = -\frac{\sigma^2}{2}\frac{\Delta_G\sqrt{\rho}}{\sqrt{\rho}}$.

**Correspondence Table:**

| Quantum Mechanics | Agent (Inference Wave) |
|:------------------|:-----------------------|
| Wave function $\psi$ | Belief amplitude |
| Planck constant $\hbar$ | Cognitive scale $\sigma$ |
| Bohm potential $Q$ | Information resolution limit $Q_B$ |
| Probability current $\mathbf{j}$ | Belief flux $\rho v$ |
::::

:::{prf:definition} Bohm Quantum Potential (Information Resolution Limit)
:label: def-bohm-quantum-potential

The **Bohm Quantum Potential** is:

$$
Q_B(z, s) := -\frac{\sigma^2}{2} \frac{\Delta_G \sqrt{\rho}}{\sqrt{\rho}} = -\frac{\sigma^2}{2} \frac{\Delta_G R}{R},
$$
where $R = \sqrt{\rho}$ is the amplitude.

**Explicit form in terms of $\rho$:**

$$
Q_B = -\frac{\sigma^2}{8\rho^2} \|\nabla_G \rho\|_G^2 + \frac{\sigma^2}{4\rho} \Delta_G \rho.
$$
**Physical interpretation:** $Q_B$ represents the **energetic cost of belief localization**. Regions where $\rho$ has high curvature (sharp belief features) incur an effective potential energy penalty. This prevents the belief from concentrating to delta functions.

**Information-theoretic interpretation:** $Q_B$ enforces the **Levin Length** (Section 33.2) as a resolution limit. The agent cannot represent distinctions finer than $\ell_L \sim \sqrt{\sigma}$.

*Units:* $[Q_B] = \text{nat}$ (same as potential).

*Cross-reference:* In standard quantum mechanics, $Q_B$ is called the "quantum potential" or "Bohm potential." Here it emerges from the information geometry, not fundamental physics.

:::

:::{prf:corollary} Open Quantum System Interpretation
:label: cor-open-quantum-system

The Inference Hamiltonian $\hat{H}_{\text{inf}}$ is **non-Hermitian** due to the reaction term $-\frac{i\sigma}{2}r$. This corresponds to an **open quantum system** where:
- $r > 0$: Mass creation (information gain from boundary) → probability amplitude **grows**
- $r < 0$: Mass destruction (information loss) → probability amplitude **decays**

The **complex potential** formulation is:

$$
W(z) := \Phi_{\text{eff}}(z) - \frac{i\sigma}{2} r(z),
$$
so that $\hat{H}_{\text{inf}} = -\frac{\sigma^2}{2}\Delta_G + W + Q_B$.

**Norm evolution:** The normalization $\|\psi\|^2 = \int |\psi|^2 d\mu_G$ evolves as:

$$
\frac{d}{ds} \|\psi\|^2 = \int_{\mathcal{Z}} r(z) |\psi(z)|^2 d\mu_G(z) = \langle r \rangle_\rho,
$$
which matches the WFR mass balance equation.

*Remark (Lindblad Connection).* For trace-preserving dynamics (where $\int r \rho\, d\mu_G = 0$), the non-Hermitian Schrödinger equation can be embedded in a **Lindblad master equation** (Definition {prf:ref}`def-gksl-generator`) via the Dyson-Phillips construction.

:::

:::{prf:proposition} Operator Ordering and Coordinate Invariance
:label: prop-operator-ordering-invariance

The kinetic term $-\frac{\sigma^2}{2}\Delta_G$ in the Inference Hamiltonian uses the unique **coordinate-invariant** ordering:

$$
-\frac{\sigma^2}{2}\Delta_G \psi = -\frac{\sigma^2}{2} \cdot \frac{1}{\sqrt{|G|}} \partial_i \left( \sqrt{|G|} G^{ij} \partial_j \psi \right).
$$
This is equivalent to:

$$
-\frac{\sigma^2}{2}\Delta_G = -\frac{\sigma^2}{2} \left( G^{ij} \partial_i \partial_j + \Gamma^k \partial_k \right),
$$
where $\Gamma^k := G^{ij}\Gamma^k_{ij}$ is the trace of Christoffel symbols.

**Alternative orderings** (Weyl, symmetric, etc.) would introduce frame-dependent terms that break the geometric interpretation.

*Cross-reference:* This matches the Laplace-Beltrami operator used in the Helmholtz equation (Theorem {prf:ref}`thm-the-hjb-helmholtz-correspondence`), ensuring consistency between the PDE and wave-function formulations.

:::

:::{prf:corollary} Semiclassical Limit
:label: cor-semiclassical-limit

In the limit $\sigma \to 0$ (classical limit), the Schrödinger dynamics recover the **geodesic flow**:

**WKB Ansatz:** $\psi = A(z) e^{iS(z)/\sigma}$ with $A$ slowly varying.

**Leading Order ($O(\sigma^{-1})$):** The Hamilton-Jacobi equation

$$
\partial_s S + \frac{1}{2}\|\nabla_G S\|_G^2 + \Phi_{\text{eff}} = 0.
$$
**Next Order ($O(\sigma^0)$):** The transport equation

$$
\partial_s |A|^2 + \nabla_G \cdot (|A|^2 \nabla_G S) = 0.
$$
These are exactly the HJB and continuity equations from WFR dynamics. The quantum correction $Q_B \to 0$ as $\sigma \to 0$.

*Interpretation:* The wave-function collapses to a delta function following the optimal trajectory. Quantum effects (tunneling, interference) vanish in this limit.

:::

(sec-multi-agent-schrodinger-equation)=
### 29.23 Multi-Agent Schrödinger Equation

We now extend the wave-function formalism to $N$-agent systems, defining **strategic entanglement** as non-factorizability of the joint belief amplitude.

:::{prf:definition} Joint Inference Hilbert Space
:label: def-joint-inference-hilbert-space

For $N$ agents with individual Hilbert spaces $\mathcal{H}^{(i)} = L^2(\mathcal{Z}^{(i)}, d\mu_{G^{(i)}})$, the **Joint Inference Hilbert Space** is the tensor product:

$$
\mathcal{H}^{(N)} := \bigotimes_{i=1}^N \mathcal{H}^{(i)} = L^2\left(\mathcal{Z}^{(N)}, d\mu_{G^{(N)}}\right),
$$
where:
- $\mathcal{Z}^{(N)} = \prod_{i=1}^N \mathcal{Z}^{(i)}$ is the product manifold (Definition {prf:ref}`def-n-agent-product-manifold`)
- $d\mu_{G^{(N)}} = \prod_{i=1}^N d\mu_{G^{(i)}}$ is the product measure

Elements $\Psi \in \mathcal{H}^{(N)}$ are functions $\Psi: \mathcal{Z}^{(N)} \to \mathbb{C}$ with:

$$
\|\Psi\|^2 = \int_{\mathcal{Z}^{(N)}} |\Psi(\mathbf{z})|^2 d\mu_{G^{(N)}}(\mathbf{z}) < \infty.
$$
*Notation:* We use uppercase $\Psi$ for joint wave-functions and lowercase $\psi^{(i)}$ for single-agent wave-functions.

:::

:::{prf:definition} Strategic Entanglement
:label: def-strategic-entanglement

A joint wave-function $\Psi \in \mathcal{H}^{(N)}$ exhibits **Strategic Entanglement** if it cannot be written as a product:

$$
\Psi(z^{(1)}, \ldots, z^{(N)}) \neq \prod_{i=1}^N \psi^{(i)}(z^{(i)}) \quad \text{for any choice of } \psi^{(i)} \in \mathcal{H}^{(i)}.
$$
**Entanglement Entropy:** For a bipartition $\{i\} \cup \{j \neq i\}$, the **Strategic Entanglement Entropy** is:

$$
S_{\text{ent}}(i) := -\text{Tr}\left[\hat{\rho}^{(i)} \ln \hat{\rho}^{(i)}\right],
$$
where $\hat{\rho}^{(i)} = \text{Tr}_{j \neq i}[|\Psi\rangle\langle\Psi|]$ is the **reduced density operator** obtained by partial trace over all agents except $i$.

**Physical interpretation:**
- $S_{\text{ent}}(i) = 0$: Agent $i$ is **disentangled** (can be modeled independently)
- $S_{\text{ent}}(i) > 0$: Agent $i$ is **entangled** with others (cannot be modeled in isolation)
- $S_{\text{ent}}(i) = \ln N$: **Maximal entanglement** (all agents maximally correlated)

*Cross-reference:* The partial trace operation corresponds to the **Information Bottleneck** (Definition {prf:ref}`def-dpi-boundary-capacity-constraint`)—marginalizing over opponents discards strategic correlations.

:::

:::{prf:definition} Strategic Hamiltonian
:label: def-strategic-hamiltonian

The **Strategic Hamiltonian** on $\mathcal{H}^{(N)}$ is:

$$
\hat{H}_{\text{strat}} := \sum_{i=1}^N \hat{H}^{(i)}_{\text{kin}} + \sum_{i=1}^N \hat{\Phi}^{(i)}_{\text{eff}} + \sum_{i < j} \hat{V}_{ij},
$$
where:
1. **Kinetic terms:** $\hat{H}^{(i)}_{\text{kin}} = -\frac{\sigma_i^2}{2} \Delta_{G^{(i)}}$ (acting on $\mathcal{Z}^{(i)}$ coordinates)
2. **Individual potentials:** $\hat{\Phi}^{(i)}_{\text{eff}}$ (local reward landscape for agent $i$)
3. **Interaction potentials:** $\hat{V}_{ij} = \Phi_{ij}(z^{(i)}, z^{(j)})$ (strategic coupling)

*Notation (Per-Agent Action Scale):* Here $\sigma_i := T_{c,i} \cdot \tau_{\text{update},i}$ is the cognitive action scale for agent $i$, generalizing Definition {prf:ref}`def-cognitive-action-scale`. For **homogeneous** agents with identical cognitive properties, $\sigma_i = \sigma$ for all $i$. For **heterogeneous** agents (e.g., different computation rates), $\sigma_i$ may vary.

*Remark (Separability).* If all $\hat{V}_{ij} = 0$, the Hamiltonian is **separable**: $\hat{H}_{\text{strat}} = \sum_i \hat{H}^{(i)}$, and the ground state is a product $\Psi_0 = \prod_i \psi^{(i)}_0$. Non-zero interaction creates entanglement.

:::

:::{prf:theorem} Multi-Agent Schrödinger Equation
:label: thm-multi-agent-schrodinger-equation

The joint belief wave-function $\Psi(\mathbf{z}, s)$ of $N$ strategically coupled agents evolves according to:

$$
i\sigma \frac{\partial \Psi}{\partial s} = \hat{H}_{\text{strat}} \Psi + i\frac{\sigma}{2} \mathcal{R} \Psi,
$$
where:
- $\hat{H}_{\text{strat}}$ is the Strategic Hamiltonian (Definition {prf:ref}`def-strategic-hamiltonian`)
- $\mathcal{R}(\mathbf{z}) = \sum_i r^{(i)}(z^{(i)})$ is the total reaction rate

**Expanded form:**

$$
i\sigma \frac{\partial \Psi}{\partial s} = \left[ \sum_{i=1}^N \left( -\frac{\sigma_i^2}{2} \Delta_{G^{(i)}} + \Phi^{(i)}_{\text{eff}} \right) + \sum_{i < j} \Phi_{ij} \right] \Psi + i\frac{\sigma}{2} \mathcal{R} \Psi.
$$
**Sources of entanglement:** Strategic entanglement arises from:
1. **Potential coupling:** Non-zero $\Phi_{ij}(z^{(i)}, z^{(j)})$ creates position-position correlations
2. **Metric coupling:** The Game Tensor $\mathcal{G}_{ij}$ modifies the kinetic terms (Theorem {prf:ref}`thm-game-augmented-laplacian`)

*Cross-reference:* This extends Theorem {prf:ref}`thm-madelung-transform` to multiple agents, with the joint WFR dynamics (Definition {prf:ref}`def-joint-wfr-action`) as the underlying classical limit.

:::

:::{prf:theorem} Game-Augmented Laplacian
:label: thm-game-augmented-laplacian

Under adversarial coupling, the effective kinetic operator for agent $i$ incorporates the **Game Tensor** (Definition {prf:ref}`def-the-game-tensor`):

$$
\hat{H}^{(i)}_{\text{kin,eff}} = -\frac{\sigma_i^2}{2} \tilde{\Delta}^{(i)},
$$
where the **Game-Augmented Laplacian** is:

$$
\tilde{\Delta}^{(i)} := \frac{1}{\sqrt{|\tilde{G}^{(i)}|}} \partial_a \left( \sqrt{|\tilde{G}^{(i)}|} (\tilde{G}^{(i)})^{ab} \partial_b \right),
$$
with strategic metric $\tilde{G}^{(i)} = G^{(i)} + \sum_{j \neq i} \beta_{ij} \mathcal{G}_{ij}$ (Definition {prf:ref}`def-the-game-tensor`, Equation 29.4.1).

**Consequence for entanglement:** Since $\tilde{G}^{(i)}$ depends on $z^{(j)}$ through the Game Tensor, the kinetic operator for agent $i$ is **not separable**:

$$
\tilde{\Delta}^{(i)} = \tilde{\Delta}^{(i)}(z^{(i)}; z^{(-i)}).
$$
This creates **kinetic entanglement**—even without potential coupling, adversarial metric inflation entangles the agents.

*Physical interpretation:* Agent $j$ "curves" agent $i$'s configuration space. Moving through a contested region requires more "effort" (higher effective mass), and this coupling cannot be factorized away.

:::

:::{prf:proposition} Partial Trace and Reduced Dynamics
:label: prop-partial-trace-reduced-dynamics

For a pure joint state $|\Psi\rangle \in \mathcal{H}^{(N)}$, the **reduced density operator** for agent $i$ is:

$$
\hat{\rho}^{(i)} := \text{Tr}_{j \neq i}\left[ |\Psi\rangle\langle\Psi| \right] = \int_{\prod_{j \neq i} \mathcal{Z}^{(j)}} |\Psi|^2 \prod_{j \neq i} d\mu_{G^{(j)}}.
$$
The diagonal elements give the **marginal belief density**:

$$
\rho^{(i)}(z^{(i)}) = \langle z^{(i)} | \hat{\rho}^{(i)} | z^{(i)} \rangle = \int |\Psi(z^{(i)}, z^{(-i)})|^2 d\mu_{G^{(-i)}},
$$
which is exactly the marginalization from the joint WFR density.

**Mixed state evolution:** Even if $\Psi$ evolves unitarily, the reduced state $\hat{\rho}^{(i)}$ generally evolves **non-unitarily** (with decoherence) due to entanglement with other agents.

:::

(sec-nash-equilibrium-as-ground-state)=
### 29.24 Nash Equilibrium as Ground State

The spectral properties of the Strategic Hamiltonian provide a new characterization of Nash equilibrium.

:::{prf:theorem} Nash Equilibrium as Ground State
:label: thm-nash-ground-state

A Nash equilibrium $\mathbf{z}^* = (z^{(1)*}, \ldots, z^{(N)*})$ (Theorem {prf:ref}`thm-nash-equilibrium-as-geometric-stasis`) corresponds to the **ground state** of the Strategic Hamiltonian:

1. **Spectral condition:** The ground state $\Psi_{\text{Nash}}$ satisfies:

   $$
   \hat{H}_{\text{strat}} \Psi_{\text{Nash}} = E_0 \Psi_{\text{Nash}}, \quad E_0 = \min \text{spec}(\hat{H}_{\text{strat}}).
   $$
2. **Localization:** In the semiclassical limit ($\sigma \to 0$), $|\Psi_{\text{Nash}}|^2$ concentrates near $\mathbf{z}^*$:

   $$
   \lim_{\sigma \to 0} |\Psi_{\text{Nash}}(\mathbf{z})|^2 = \delta(\mathbf{z} - \mathbf{z}^*).
   $$
3. **Energy interpretation:** The ground state energy $E_0$ equals the total effective potential at Nash:

   $$
   E_0 = \sum_{i=1}^N \Phi^{(i)}_{\text{eff}}(\mathbf{z}^*) + \sum_{i < j} \Phi_{ij}(\mathbf{z}^*) + O(\sigma).
   $$
*Proof sketch.*
- At Nash, $\nabla_{z^{(i)}} \Phi^{(i)}_{\text{eff}} = 0$ for all $i$ (Condition 1 of Theorem {prf:ref}`thm-nash-equilibrium-as-geometric-stasis`).
- The variational principle $\delta \langle \Psi | \hat{H} | \Psi \rangle / \delta \Psi^* = 0$ with normalization constraint yields the same stationarity conditions in the $\sigma \to 0$ limit.
- The second variation (Hessian) being non-positive (Condition 3) corresponds to local stability of the ground state.

See **Appendix E.19** for the complete WKB/semiclassical analysis proving Gaussian concentration to delta function as $\sigma \to 0$, with explicit energy correction formulas. $\square$

*Remark (Multiple Nash).* If multiple Nash equilibria exist, each corresponds to a different local minimum of the energy landscape. The **global** ground state is the Nash with lowest $E_0$; other Nash equilibria are metastable excited states.

:::

:::{prf:corollary} Vanishing Probability Current at Nash
:label: cor-vanishing-probability-current

At Nash equilibrium, the **probability current** vanishes:

$$
\mathbf{J}^{(i)}(\mathbf{z}^*) := \text{Im}\left[\bar{\Psi}_{\text{Nash}} \cdot \sigma \nabla_{G^{(i)}} \Psi_{\text{Nash}}\right]_{\mathbf{z}^*} = 0 \quad \forall i.
$$
**Derivation:** The probability current is $\mathbf{J} = \rho \mathbf{v}$ where $\mathbf{v} = G^{-1}\nabla V$ is the velocity field. At Nash:
- $\nabla V^{(i)}|_{\mathbf{z}^*} = 0$ (stationarity condition)
- Therefore $\mathbf{v}^{(i)}|_{\mathbf{z}^*} = 0$
- Hence $\mathbf{J}^{(i)}|_{\mathbf{z}^*} = 0$

*Interpretation:* At Nash, there is no net belief flow. The wave-function is in a **standing wave pattern**—agents are not "stopped" but are in dynamic equilibrium where flows cancel.

*Cross-reference:* This is the quantum version of **Geometric Stasis** (Theorem {prf:ref}`thm-nash-equilibrium-as-geometric-stasis`).

:::

:::{prf:proposition} Imaginary Time Evolution for Nash Finding
:label: prop-imaginary-time-nash-finding

The substitution $s \to -i\tau$ (**Wick rotation**) transforms the Schrödinger equation into a diffusion equation:

$$
-\sigma \frac{\partial \Psi}{\partial \tau} = \hat{H}_{\text{strat}} \Psi.
$$
Under this **imaginary time evolution**, any initial state $\Psi_0$ converges to the ground state:

$$
\Psi(\tau) = e^{-\hat{H}_{\text{strat}} \tau / \sigma} \Psi_0 \xrightarrow{\tau \to \infty} c \cdot \Psi_{\text{Nash}},
$$
where $c$ is a normalization constant.

**Computational interpretation:**
1. Imaginary time evolution is equivalent to **Value Iteration** in dynamic programming
2. The propagator $e^{-\hat{H}\tau/\sigma}$ is the **Bellman backup operator** in infinite-horizon limit
3. Convergence rate is set by the **spectral gap** $E_1 - E_0$ (energy of first excited state minus ground state)

**Algorithm sketch (Quantum Value Iteration):**
```
Initialize Ψ randomly
For τ = 0 to T:
    Ψ ← exp(-H_strat Δτ / σ) Ψ    # Diffusion step
    Ψ ← Ψ / ||Ψ||                  # Renormalize
Return Ψ (approximates Nash ground state)
```

*Cross-reference:* This connects to the **imaginary-time path integral** formulation used in quantum Monte Carlo methods.

:::

(sec-strategic-tunneling-and-barrier-crossing)=
### 29.25 Strategic Tunneling and Barrier Crossing

The wave nature of the belief amplitude enables **tunneling**—crossing barriers that would trap classical gradient-descent agents.

:::{prf:definition} Pareto Barrier
:label: def-pareto-barrier

A **Pareto Barrier** $\mathcal{B}_P \subset \mathcal{Z}^{(N)}$ is a region where:
1. **Local value decrease:** $\Phi^{(i)}_{\text{eff}}(\mathbf{z}) > \Phi^{(i)}_{\text{eff}}(\mathbf{z}^*)$ for at least one agent $i$ and some starting point $\mathbf{z}^*$
2. **No Nash within:** There exists no Nash equilibrium $\mathbf{z}' \in \mathcal{B}_P$
3. **Separates basins:** $\mathcal{B}_P$ lies between distinct Nash equilibria $\mathbf{z}^*_A$ and $\mathbf{z}^*_B$

The **barrier height** is:

$$
\Delta \Phi_P := \max_{\mathbf{z} \in \mathcal{B}_P} \left[ \sum_{i=1}^N \Phi^{(i)}_{\text{eff}}(\mathbf{z}) - \sum_{i=1}^N \Phi^{(i)}_{\text{eff}}(\mathbf{z}^*_A) \right].
$$
*Mathematical characterization:* A Pareto barrier is a region where the total potential $\sum_i \Phi^{(i)}_{\text{eff}}$ exceeds its value at nearby Nash equilibria. Classical gradient descent with initial condition in the basin of attraction of $\mathbf{z}^*_A$ converges to $\mathbf{z}^*_A$ and cannot reach $\mathbf{z}^*_B$.

:::

:::{prf:theorem} Strategic Tunneling Probability (WKB Approximation)
:label: thm-tunneling-probability

In the semiclassical limit ($\sigma \ll \Delta \Phi_P$), the probability of crossing a Pareto barrier is:

$$
P_{\text{tunnel}} \sim \exp\left(-\frac{2}{\sigma} \int_{\gamma} \sqrt{2(\Phi_{\text{eff,total}}(\mathbf{z}) - E_0)}\, d\ell_{G^{(N)}}\right),
$$
where:
- $\gamma$ is the **optimal tunneling path** (instanton) connecting $\mathbf{z}^*_A$ to $\mathbf{z}^*_B$
- $\Phi_{\text{eff,total}} = \sum_i \Phi^{(i)}_{\text{eff}} + \sum_{i<j} \Phi_{ij}$
- $d\ell_{G^{(N)}}$ is the geodesic arc length on $(\mathcal{Z}^{(N)}, G^{(N)})$
- $E_0$ is the ground state energy

**Key scaling:** $P_{\text{tunnel}} \propto e^{-\Delta \Phi_P / \sigma}$, so higher barriers or lower temperature (small $\sigma$) exponentially suppress tunneling.

*Cross-reference:* This generalizes Theorem {prf:ref}`thm-memory-induced-barrier-crossing` from single-agent memory barriers to multi-agent Pareto barriers. See Appendix E.7 for the rigorous proof via Agmon estimates and spectral theory.

:::

::::{admonition} Physics Isomorphism: WKB Tunneling
:class: note
:name: pi-wkb-tunneling

**In Physics:** The WKB approximation gives tunneling probability through a barrier: $P \sim \exp(-2\int_a^b \sqrt{2m(U-E)/\hbar^2}\,dx)$ where the integral is over the classically forbidden region {cite}`wentzel1926verallgemeinerung,agmon1982lectures`.

**In Implementation:** The tunneling probability (Theorem {prf:ref}`thm-tunneling-probability`):

$$
P_{\text{tunnel}} \sim \exp\left(-\frac{2}{\sigma}\int_\gamma \sqrt{2(\Phi_{\text{eff}} - E_0)}\,d\ell_G\right)
$$
**Correspondence Table:**

| Quantum Mechanics | Agent (Strategic Tunneling) |
|:------------------|:----------------------------|
| Barrier potential $U(x)$ | Effective potential $\Phi_{\text{eff}}$ |
| Ground state energy $E_0$ | Nash equilibrium value |
| Tunneling exponent | Agmon distance $d_{\text{Ag}}$ |
| $\hbar \to 0$ limit | $\sigma \to 0$ (classical Nash) |
::::

:::{prf:corollary} Bohm Potential Enables Strategic Teleportation
:label: cor-bohm-teleportation

When the Bohm potential $Q_B$ dominates (high belief curvature), the **effective barrier** becomes:

$$
\Phi^{\text{quantum}}_{\text{eff}}(\mathbf{z}) := \Phi_{\text{eff,total}}(\mathbf{z}) + Q_B(\mathbf{z}).
$$
In regions where $Q_B < 0$ (convex $\rho$), the effective barrier can become **negative** even when $\Phi_{\text{eff}} > 0$. This enables "teleportation" across classically forbidden regions.

**Operational interpretation:**
- An agent with **high uncertainty** (diffuse, smooth $\rho$) has $Q_B \approx 0$ → normal barrier
- An agent with **localized uncertainty** near the barrier (peaked, curved $\rho$) can have $Q_B \ll 0$ → reduced effective barrier
- The WFR **reaction term** $r$ (mass creation/destruction) provides the mechanism for "teleporting" belief mass without traversing intermediate states

*Remark (Exploration-Exploitation).* This provides a geometric foundation for the exploration-exploitation tradeoff: maintaining some uncertainty ($Q_B \neq 0$) is necessary to escape local optima.

:::

:::{prf:proposition} WFR Reaction as Tunneling Mechanism
:label: prop-wfr-reaction-tunneling

The WFR reaction term $r(z)$ (Definition {prf:ref}`def-the-wfr-action`) enables tunneling via **mass creation on the far side** of barriers:

1. Agent detects high-value region $\mathbf{z}^*_B$ beyond barrier $\mathcal{B}_P$
2. Reaction term $r(\mathbf{z}^*_B) > 0$ creates belief mass at $\mathbf{z}^*_B$
3. Reaction term $r(\mathbf{z}^*_A) < 0$ destroys mass at old position $\mathbf{z}^*_A$
4. Net effect: belief "teleports" without traversing $\mathcal{B}_P$

The rate of this process is controlled by the **teleportation length** $\lambda$ (Definition {prf:ref}`def-canonical-length-scale`):
- $\lambda \gg$ barrier width: tunneling is fast (reaction-dominated)
- $\lambda \ll$ barrier width: tunneling is slow (transport-dominated)

:::

(sec-summary-of-qm-agent-isomorphisms)=
### 29.26 Summary of QM-Agent Isomorphisms

The following table consolidates the correspondence between quantum mechanical concepts and their Fragile Agent interpretations.

**Table 29.13.1 (Quantum-Agent Dictionary).**

| Quantum Mechanics | Fragile Agent Theory | Definition/Location |
|:------------------|:---------------------|:--------------------|
| **Wave-function $\psi$** | Belief Amplitude $\sqrt{\rho}e^{iV/\sigma}$ | {prf:ref}`def-belief-wave-function` |
| **Probability $\|\psi\|^2$** | Belief Density $\rho$ | Definition {prf:ref}`def-the-wfr-action` |
| **Phase $\arg(\psi)$** | Value Function $V/\sigma$ | Theorem {prf:ref}`thm-the-hjb-helmholtz-correspondence` |
| **Planck constant $\hbar$** | Cognitive Action Scale $\sigma$ | {prf:ref}`def-cognitive-action-scale` |
| **Hilbert space $\mathcal{H}$** | $L^2(\mathcal{Z}, d\mu_G)$ | {prf:ref}`def-inference-hilbert-space` |
| **Hamiltonian $\hat{H}$** | Inference Hamiltonian $\hat{H}_{\text{inf}}$ | {prf:ref}`thm-madelung-transform` |
| **Kinetic energy $-\frac{\hbar^2}{2m}\nabla^2$** | Diffusion term $-\frac{\sigma^2}{2}\Delta_G$ | Section 29.9 |
| **Potential energy $V(x)$** | Effective Potential $\Phi_{\text{eff}}$ | Definition {prf:ref}`def-effective-potential` |
| **Quantum potential $Q$** | Information Resolution Limit $Q_B$ | {prf:ref}`def-bohm-quantum-potential` |
| **Schrödinger equation** | Inference-Wave equation | {prf:ref}`thm-madelung-transform` |
| **Entanglement** | Strategic Coupling (non-factorizable) | {prf:ref}`def-strategic-entanglement` |
| **Tensor product $\otimes$** | Joint Hilbert space | {prf:ref}`def-joint-inference-hilbert-space` |
| **Partial trace** | Marginalization / Information Bottleneck | {prf:ref}`prop-partial-trace-reduced-dynamics` |
| **Ground state** | Nash Equilibrium | {prf:ref}`thm-nash-ground-state` |
| **Tunneling** | Pareto Barrier Crossing | {prf:ref}`thm-tunneling-probability` |
| **Imaginary time evolution** | Value Iteration | {prf:ref}`prop-imaginary-time-nash-finding` |
| **Density matrix $\hat{\rho}$** | Belief Operator (GKSL) | Definition {prf:ref}`def-belief-operator` |
| **Lindblad dissipator** | WFR Reaction Term $r$ | Definition {prf:ref}`def-gksl-generator` |
| **von Neumann entropy** | Belief Entropy $-\text{Tr}[\hat{\rho}\ln\hat{\rho}]$ | Section 29.14 |
| **WKB approximation** | Semiclassical limit | {prf:ref}`cor-semiclassical-limit` |
| **Spectral gap** | Convergence rate to Nash | {prf:ref}`prop-imaginary-time-nash-finding` |

**Interpretation Hierarchy:**
1. **Level 1 (Symplectic):** Ghost Interface $\mathcal{G}_{ij}$ couples agent boundaries with retardation
2. **Level 2 (Riemannian):** Game Tensor $\mathcal{G}_{ij}$ curves the metric (with strategic delay)
3. **Level 3 (Thermodynamic):** Landauer bounds constrain information processing
4. **Level 4 (Quantum):** Wave-function provides superposition and tunneling

Each level adds expressive power while remaining mathematically consistent with the levels below.

(sec-diagnostic-nodes-quantum-consistency)=
### 29.27 Diagnostic Nodes 57–60 (Quantum Consistency)

Following the diagnostic node convention (Section 3.1), we define four new monitors for quantum consistency in multi-agent systems.

(node-57)=
**Node 57: CoherenceCheck**

| **#** | **Name** | **Component** | **Type** | **Interpretation** | **Proxy** | **Cost** |
|:------|:---------|:--------------|:---------|:-------------------|:----------|:---------|
| **57** | **CoherenceCheck** | Multi-Agent | Unitary Consistency | Is the belief update probability-preserving? | $\delta_{\text{coh}} := \left\lvert \|\Psi_{s+\Delta s}\|^2 - \|\Psi_s\|^2 \right\rvert$ | $O(N d)$ |

**Interpretation:** Monitors deviation from unitarity (or trace-preservation for density matrices). Non-zero $\delta_{\text{coh}}$ indicates:
- Numerical integration error
- Unmodeled dissipation channels
- Inconsistency between Hamiltonian and WFR dynamics

**For open systems:** Replace with trace preservation check $\delta_{\text{tr}} := |\text{Tr}[\hat{\rho}_{s+\Delta s}] - 1|$.

**Threshold:** $\delta_{\text{coh}} < \epsilon_{\text{coh}}$ (typical default $10^{-6}$).

**Trigger conditions:**
- High CoherenceCheck: Numerical instability or model inconsistency
- **Remedy:** Reduce timestep; verify Hamiltonian is Hermitian (up to reaction term); check boundary conditions

(node-58)=
**Node 58: EntropyProductionCheck**

| **#** | **Name** | **Component** | **Type** | **Interpretation** | **Proxy** | **Cost** |
|:------|:---------|:--------------|:---------|:-------------------|:----------|:---------|
| **58** | **EntropyProductionCheck** | Multi-Agent | Thermodynamic | Is entropy production physically reasonable? | $\dot{S}_{\text{vN}} := -\frac{d}{ds}\text{Tr}[\hat{\rho}\ln\hat{\rho}]$ | $O(N^2 d)$ |

**Interpretation:** Monitors the rate of **von Neumann entropy** change:
- $\dot{S}_{\text{vN}} > 0$: Entropy increasing (learning, decoherence, information gain from environment)
- $\dot{S}_{\text{vN}} < 0$: Entropy decreasing (spontaneous ordering, may indicate instability)
- $\dot{S}_{\text{vN}} \approx 0$: Near equilibrium

**Connection to Landauer:** Entropy production should satisfy $\dot{S}_{\text{vN}} \geq -\mathcal{W}_{\text{comp}}/T_c$ where $\mathcal{W}_{\text{comp}}$ is computational work (Section 31).

**Threshold:** $|\dot{S}_{\text{vN}}| < \dot{S}_{\max}$ (implementation-dependent).

**Trigger conditions:**
- Large positive $\dot{S}_{\text{vN}}$: Rapid decoherence ("losing mind")
- Large negative $\dot{S}_{\text{vN}}$: Anomalous ordering (check for mode collapse)

(node-59)=
**Node 59: UncertaintyPrincipleCheck**

| **#** | **Name** | **Component** | **Type** | **Interpretation** | **Proxy** | **Cost** |
|:------|:---------|:--------------|:---------|:-------------------|:----------|:---------|
| **59** | **UncertaintyPrincipleCheck** | Multi-Agent | Consistency | Are uncertainty bounds satisfied? | $\eta_{\text{unc}} := \frac{\sigma/2}{\sigma_z \sigma_p} \leq 1$ | $O(N d)$ |

**Interpretation:** The **Heisenberg-Robertson uncertainty relation** on the latent manifold requires:

$$
\sigma_z \cdot \sigma_p \geq \frac{\sigma}{2} |\langle[\hat{z}, \hat{p}]\rangle| = \frac{\sigma}{2},
$$
where:
- $\sigma_z := \sqrt{\langle z^2 \rangle - \langle z \rangle^2}$ is position uncertainty
- $\sigma_p := \sqrt{\langle p^2 \rangle - \langle p \rangle^2}$ is momentum uncertainty ($p = G\mathbf{v} = \nabla V$)

**Violation $\eta_{\text{unc}} > 1$:** The agent claims to know both "where it is" (position $z$) and "where it is going" (momentum $\nabla V$) with precision exceeding the information-theoretic limit. This indicates:
- Over-confident world model
- Ungrounded predictions
- Numerical precision issues

**Threshold:** $\eta_{\text{unc}} < 1$ (hard constraint from information geometry).

**Trigger conditions:**
- $\eta_{\text{unc}} > 1$: Uncertainty violation
- **Remedy:** Increase $\sigma$ (cognitive temperature); add regularization; verify encoder-decoder consistency

(node-60)=
**Node 60: TunnelingRateMonitor**

| **#** | **Name** | **Component** | **Type** | **Interpretation** | **Proxy** | **Cost** |
|:------|:---------|:--------------|:---------|:-------------------|:----------|:---------|
| **60** | **TunnelingRateMonitor** | Multi-Agent | Exploration | Is barrier crossing rate reasonable? | $\Gamma_{\text{tunnel}} := P_{\text{tunnel}} / \tau_{\text{obs}}$ | $O(N^2 d)$ |

**Interpretation:** Monitors the rate at which agents cross Pareto barriers (Theorem {prf:ref}`thm-tunneling-probability`):
- $\Gamma_{\text{tunnel}} \approx 0$: Agents trapped in local equilibria (insufficient exploration)
- $\Gamma_{\text{tunnel}}$ moderate: Healthy exploration-exploitation balance
- $\Gamma_{\text{tunnel}}$ very high: Unstable dynamics (agents "teleporting" erratically)

**Computation:** Track probability mass flux across identified barrier surfaces using:

$$
\Gamma_{\text{tunnel}} = \int_{\partial \mathcal{B}_P} \mathbf{J} \cdot \mathbf{n}\, d\Sigma,
$$
where $\mathbf{J}$ is probability current and $\partial \mathcal{B}_P$ is the barrier boundary.

**Threshold:** $\Gamma_{\min} < \Gamma_{\text{tunnel}} < \Gamma_{\max}$ (task-dependent).

**Trigger conditions:**
- Low tunneling: Increase $\sigma$ or WFR reaction rate
- High tunneling: Decrease exploration; check for instabilities
- Asymmetric tunneling (one direction only): May indicate irreversible dynamics

(sec-implementation-causal-buffer)=
### 29.28 Implementation: The Causal Buffer

To implement relativistic multi-agent dynamics without disrupting existing software architecture, we introduce a **Causal Buffer** that handles time-retardation transparently.

**Algorithm 29.20.1 (Causal Context Buffer).**

```python
class CausalContextBuffer(nn.Module):
    """
    Implements the Memory Screen for Relativistic Agents.
    Stores past signals and serves them based on light-cone delay.

    The buffer maintains a history of (time, signal) pairs and provides
    ghost states by interpolating to the appropriate retarded time.
    """
    def __init__(self, context_dim: int, max_latency: int = 100):
        super().__init__()
        self.buffer = []  # Ring buffer of (t, signal)
        self.c_info = 1.0  # Information speed (environment units / timestep)
        self.max_latency = max_latency
        self.context_dim = context_dim

    def write(self, t: int, signal: torch.Tensor):
        """
        Agent J emits signal at time t.

        Args:
            t: Current timestep
            signal: State/action tensor to record
        """
        self.buffer.append((t, signal.clone()))
        # Prune old entries beyond max latency
        while self.buffer and self.buffer[0][0] < t - self.max_latency:
            self.buffer.pop(0)

    def read(self, t_now: int, dist: float) -> torch.Tensor:
        """
        Agent I reads signal arriving at t_now from distance dist.
        Returns the ghost state: signal emitted at t_emit = t_now - dist/c_info.

        Args:
            t_now: Current time of the reading agent
            dist: Environment distance d_E^{ij} between agents

        Returns:
            Ghost signal from retarded time, or zero if no data available
        """
        t_emit_target = t_now - (dist / self.c_info)

        if not self.buffer:
            return torch.zeros(self.context_dim)

        # Interpolate from buffer (Ghost State reconstruction)
        ghost_signal = self._interpolate(t_emit_target)
        return ghost_signal

    def _interpolate(self, t_target: float) -> torch.Tensor:
        """Linear interpolation between nearest buffer entries."""
        if len(self.buffer) == 1:
            return self.buffer[0][1]

        # Find bracketing entries
        for i in range(len(self.buffer) - 1):
            t_lo, s_lo = self.buffer[i]
            t_hi, s_hi = self.buffer[i + 1]
            if t_lo <= t_target <= t_hi:
                alpha = (t_target - t_lo) / (t_hi - t_lo + 1e-8)
                return (1 - alpha) * s_lo + alpha * s_hi

        # Extrapolate if t_target outside range
        if t_target < self.buffer[0][0]:
            return self.buffer[0][1]
        return self.buffer[-1][1]
```

**Integration with HolographicInterface:** The `CausalContextBuffer.read()` output serves as the `context` tensor for the policy $\pi(a|z, c)$. The policy remains Markovian with respect to the augmented input $(z_t, \Xi_{<t})$.

```python
class RelativisticMultiAgentInterface(nn.Module):
    """
    Extends HolographicInterface for relativistic multi-agent settings.
    Wraps agent-to-agent communication with causal buffers.
    """
    def __init__(self, n_agents: int, config: InterfaceConfig,
                 env_distances: torch.Tensor, c_info: float = 1.0):
        super().__init__()
        self.n_agents = n_agents
        self.c_info = c_info

        # Causal buffer for each ordered pair (i, j)
        self.buffers = nn.ModuleDict({
            f"{i}_{j}": CausalContextBuffer(config.context_dim)
            for i in range(n_agents) for j in range(n_agents) if i != j
        })

        # Precompute causal delays tau_ij = d_ij / c_info
        self.register_buffer('tau', env_distances / c_info)

    def broadcast(self, agent_id: int, t: int, state: torch.Tensor):
        """Agent broadcasts its state to all buffers it writes to."""
        for j in range(self.n_agents):
            if j != agent_id:
                self.buffers[f"{agent_id}_{j}"].write(t, state)

    def receive_context(self, agent_id: int, t: int) -> torch.Tensor:
        """
        Agent receives ghost states from all other agents.
        Returns concatenated context from past light cone.
        """
        contexts = []
        for j in range(self.n_agents):
            if j != agent_id:
                tau_ij = self.tau[agent_id, j].item()
                ghost = self.buffers[f"{j}_{agent_id}"].read(t, tau_ij * self.c_info)
                contexts.append(ghost)
        return torch.cat(contexts, dim=-1)
```

*Cross-reference:* This implementation realizes the Ghost Interface (Definition {prf:ref}`def-ghost-interface`) and Memory Screen (Definition {prf:ref}`def-memory-screen`) in differentiable form, enabling end-to-end training of relativistic multi-agent systems.



(sec-extended-summary-table)=
### 29.29 Extended Summary Table

**Table 29.29.1 (Extended SMFT Summary with Relativistic, Gauge, and Quantum Layers).**

| Concept | Single Agent (Sec. 20–24) | Multi-Agent Relativistic (Sec. 29.1–29.12) | Multi-Agent Gauge (Sec. 29.13–29.20) | Multi-Agent Quantum (Sec. 29.21–29.27) |
|:--------|:--------------------------|:-------------------------------------------|:-------------------------------------|:---------------------------------------|
| **State Space** | $\mathcal{Z}$ | $\mathcal{Z}_{\text{causal}} = \mathcal{Z}^{(N)} \times \Xi_{<t}$ | $\mathcal{P}(G) \times \mathcal{Z}_{\text{causal}}$ | $\mathcal{H}^{(N)} = \bigotimes_i \mathcal{H}^{(i)}$ |
| **State Rep.** | Density $\rho(z)$ | Causal Bundle $(z^{(i)}_t, \Xi^{(i)}_{<t})$ | + Gauge connection $A_\mu$ | Wave-function $\Psi(\mathbf{z})$ or operator $\hat{\rho}$ |
| **Dynamics** | WFR Continuity + HJB | Coupled WFR + Retardation | + Yang-Mills equations | Schrödinger equation |
| **Generator** | Fokker-Planck / WFR | Coupled continuity (Klein-Gordon) | + $D_\mu \mathcal{F}^{\mu\nu} = J^\nu$ | Hamiltonian $\hat{H}_{\text{strat}}$ |
| **Kinetic Term** | $\nabla \cdot (\rho G^{-1}\nabla V)$ | $\frac{1}{c^2}\partial_t^2 - \Delta_G$ | $\frac{1}{c^2}D_t^2 - D^i D_i$ | $-\frac{\sigma^2}{2}\Delta_{\tilde{G}}$ |
| **Potential** | $\Phi_{\text{eff}}(z)$ | $\Phi^{\text{ret}}_{ij}(z^{(i)}, z^{(j)}_{t-\tau})$ | + Higgs $\mu^2|\Phi|^2 + \lambda|\Phi|^4$ | Operator $\hat{\Phi}_{\text{eff}} + \sum \hat{V}_{ij}$ |
| **Metric Effect** | $G$ | $\tilde{G}^{(i)}(t) = G^{(i)} + \sum_j \beta_{ij}\mathcal{G}^{\text{ret}}_{ij}$ | + Gauge-covariant $\tilde{\mathcal{G}}_{ij}$ | Game-Augmented Laplacian |
| **Coupling Mechanism** | — | Ghost Interface $\mathcal{G}_{ij}(t)$ | + Gauge connection $A_\mu$ | + Strategic Entanglement |
| **Resolution Limit** | — | Causal delay $\tau_{ij} = d_{\mathcal{E}}^{ij}/c_{\text{info}}$ | Mass gap $\Delta > 0$ | Bohm potential $Q_B$ |
| **Coupling Type** | — | Potential + Metric + Retardation | + Gauge curvature $\mathcal{F}_{\mu\nu}$ | + Entanglement |
| **Correlation** | — | Classical (delayed, factorizable) | + Screened ($\xi = 1/\kappa < \infty$) | Quantum (non-factorizable) |
| **Equilibrium** | Value maxima | Standing Wave (time-averaged Nash) | + Symmetry breaking (VEV) | Ground state of $\hat{H}_{\text{strat}}$ |
| **Equilibrium Char.** | $\nabla V = 0$ | $\langle \mathbf{J}^{(i)} \rangle_T = 0$ | $\langle\Phi\rangle = v/\sqrt{2}$ | $\hat{H}\Psi = E_0\Psi$ |
| **Barrier Crossing** | Thermal noise | Strategic delay + coupling | Confinement (basin locking) | Quantum tunneling |
| **Nash Finding** | Gradient descent | Coupled gradient + causal buffer | + Gauge-covariant descent | Imaginary time evolution |
| **Diagnostics** | Nodes 1–45 | + Nodes 46–48, 62 | + Nodes 63–66 | + Nodes 57–60 |
| **Newtonian Limit** | — | $c_{\text{info}} \to \infty$: instantaneous | $g \to 0$: Abelian (Maxwell) | $\sigma \to 0$: classical Nash |

**Open problems (extended):**
1. *Scalability:* Tensor product Hilbert space dimension scales as $\prod_i \dim(\mathcal{H}^{(i)})$. Mean-field or tensor network approximations needed for large $N$.
2. *Decoherence timescales:* How fast does strategic entanglement decay under realistic noise? What is the effective "decoherence time" for multi-agent systems?
3. *Entanglement witnesses:* Can we design efficient diagnostics to detect and quantify strategic entanglement without full state tomography?
4. *Quantum speedup:* Does the Schrödinger formulation enable faster Nash-finding algorithms (quantum advantage in game theory)?
5. *Topological phases:* When $\mathcal{Z}$ has non-trivial topology, can agents exhibit "topologically protected" strategies immune to local perturbations?



(sec-ontological-expansion-topological-fission-and-the-semantic-vacuum)=
