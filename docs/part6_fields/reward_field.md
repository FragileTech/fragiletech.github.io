## 24. The Reward Field: Value Forms and Hodge Geometry

{cite}`evans2010pde,sutton2018rl`

We have defined Observations as **Configuration Constraints** (manifold position, Section 23.1) and Actions as **Momentum Constraints** (tangent vectors, Section 23.1). We now define the third component of the interface: **Reward**.

We rigorously frame Reward not as a scalar signal, but as a **Differential 1-Form** on the latent manifold. This generalization is fundamental: the agent harvests reward by moving through the field, and the reward it collects depends on both position and direction of motion. The standard scalar value function $V(z)$ emerges as the special case where the reward field is **conservative** (curl-free).

:::{admonition} Researcher Bridge: Beyond Conservative Value Functions
:class: tip
:name: rb-non-conservative-value
Standard RL assumes a scalar Value function $V(z)$ exists such that reward gradients are conservative: $\mathcal{R} = \nabla V$. This implies that the total reward around any closed loop is zero—no cyclic preference structures exist. But many real-world scenarios violate this:
- **Rock-Paper-Scissors**: Cyclic dominance creates non-zero reward loops
- **Exploration-Exploitation Orbits**: Optimal behavior may involve sustained cycling
- **Paradoxical Preferences**: Humans exhibit intransitive preferences

We generalize by treating reward as a **1-form field** $\mathcal{R}$, with scalar value as the special case where $d\mathcal{R} = 0$ (curl vanishes). The **Hodge Decomposition** separates the optimizable (gradient) component from the cyclic (solenoidal) component.
:::

(sec-the-reward-1-form)=
### 24.1 The Reward 1-Form

We begin with the most general formulation: reward is a **differential 1-form** on the latent manifold.

:::{prf:definition} The Reward 1-Form
:label: def-reward-1-form

Let $\mathcal{R}$ be a differential 1-form on the latent manifold $(\mathcal{Z}, G)$. The **instantaneous reward rate** received by the agent moving with velocity $v \in T_z\mathcal{Z}$ is:

$$
r_t = \langle \mathcal{R}(z), v \rangle_G = \mathcal{R}_i(z) \dot{z}^i.
$$

*Units:* $[\mathcal{R}] = \mathrm{nat}/[\text{length}]$.

The cumulative reward along a trajectory $\gamma: [0,T] \to \mathcal{Z}$ is the **line integral**:

$$
R_{\text{cumulative}} = \int_\gamma \mathcal{R} = \int_0^T \mathcal{R}_i(\gamma(t)) \dot{\gamma}^i(t) \, dt.
$$

*Remark.* Instantaneous reward depends on both position $z$ and velocity $\dot{z}$. A stationary agent ($\dot{z} = 0$) receives zero instantaneous reward.

:::

:::{prf:definition} The Reward Flux (Boundary Form)
:label: def-the-reward-flux

The environment provides reward via a flux form $J_r$ on the boundary $\partial\Omega$:

$$
\int_{\partial\Omega} J_r = \text{Cumulative Boundary Reward}.
$$
In the discrete limit, this manifests as point charges $r_t$ deposited at the boundary coordinates $(t, z_{\text{boundary}})$.

*Units:* $[J_r] = \mathrm{nat}/\mathrm{area}$, $[r_t] = \mathrm{nat}$.

*Relation to 1-form:* The boundary reward flux $J_r$ and the bulk 1-form $\mathcal{R}$ are related by Stokes' theorem: the boundary integral of $J_r$ equals the bulk integral of $d\mathcal{R}$ plus boundary terms.

:::

::::{admonition} Physics Isomorphism: Electromagnetism
:class: note
:name: pi-electromagnetism-reward

**In Physics:** A charged particle moving through an electromagnetic field experiences a force that depends on both position and velocity. The electric field $\mathbf{E}$ creates conservative (gradient) forces, while the magnetic field $\mathbf{B}$ creates velocity-dependent (curl) forces via the Lorentz force law: $\mathbf{F} = q(\mathbf{E} + \mathbf{v} \times \mathbf{B})$.

**In Implementation:** An agent moving through a reward field experiences:
- **Gradient force:** $-\nabla\Phi$ (climb toward value peaks)
- **Lorentz force:** $\mathcal{F} \cdot \dot{z}$ (orbit around value cycles)

**Correspondence Table:**

| Electromagnetism | Agent (Reward Field) |
|:-----------------|:---------------------|
| 4-potential $A_\mu$ | Reward 1-form $\mathcal{R}$ |
| Electric potential $\phi$ | Scalar potential $\Phi$ |
| Magnetic field $\mathbf{B} = \nabla \times \mathbf{A}$ | Value Curl $\mathcal{F} = d\mathcal{R}$ |
| Lorentz force $\mathbf{v} \times \mathbf{B}$ | Orbiting strategy |
| Cyclotron orbit | Value harvesting cycle |

::::
(sec-hodge-decomposition-of-value)=
### 24.2 The Hodge Decomposition of Value

The central theorem of this section decomposes any reward 1-form into three orthogonal components: gradient, solenoidal, and harmonic. This decomposition separates the optimizable component from the inherently cyclic components.

:::{prf:theorem} Hodge Decomposition of the Reward Field
:label: thm-hodge-decomposition

On the compact latent Riemannian manifold $(\mathcal{Z}, G)$, the Reward 1-form $\mathcal{R}$ uniquely decomposes into:

$$
\mathcal{R} = \underbrace{d\Phi}_{\text{Gradient}} + \underbrace{\delta \Psi}_{\text{Solenoidal}} + \underbrace{\eta}_{\text{Harmonic}}
$$
where:
1. **$\Phi \in \Omega^0(\mathcal{Z})$** (Scalar Potential): The conservative/optimizable component. $d\Phi$ is an exact form.
2. **$\Psi \in \Omega^2(\mathcal{Z})$** (Vector Potential): The rotational/cyclic component. $\delta\Psi$ is a coexact form (divergence-free).
3. **$\eta \in \mathcal{H}^1(\mathcal{Z})$** (Harmonic Flux): Topological cycles from manifold holes. Satisfies $d\eta = 0$ and $\delta\eta = 0$.

*Units:* $[\Phi] = \mathrm{nat}$, $[\Psi] = \mathrm{nat} \cdot [\text{length}]^2$, $[\eta] = \mathrm{nat}/[\text{length}]$.

*Proof sketch.* The Hodge decomposition follows from the orthogonal decomposition of $L^2(\Omega^1)$ into exact, coexact, and harmonic forms. The Hodge Laplacian $\Delta_H = d\delta + \delta d$ has kernel equal to the harmonic forms. The explicit solution uses the Green's operator $G = (\Delta_H)^{-1}$ on the orthogonal complement of harmonic forms: $\Phi = \delta G \mathcal{R}$, $\Psi = d G \mathcal{R}$, $\eta = \mathcal{R} - d\Phi - \delta\Psi$. $\square$

:::

:::{prf:definition} The Value Curl (Vorticity Tensor)
:label: def-value-curl

The **Value Curl** is the exterior derivative of the reward form:

$$
\mathcal{F} := d\mathcal{R} = d\delta\Psi.
$$
In coordinates: $\mathcal{F}_{ij} = \partial_i \mathcal{R}_j - \partial_j \mathcal{R}_i$.

*Units:* $[\mathcal{F}] = \mathrm{nat}/[\text{length}]^2$ (curvature of value).

**Properties:**
1. $\mathcal{F}$ is antisymmetric: $\mathcal{F}_{ij} = -\mathcal{F}_{ji}$
2. $\mathcal{F}$ satisfies the Bianchi identity: $d\mathcal{F} = 0$
3. $\mathcal{F}$ is gauge-invariant: if $\mathcal{R} \to \mathcal{R} + d\chi$, then $\mathcal{F} \to \mathcal{F}$

:::

:::{prf:definition} Conservative Reward Field
:label: def-conservative-reward-field

The reward field $\mathcal{R}$ is **conservative** if and only if:

$$
\mathcal{F} = d\mathcal{R} = 0 \quad \text{(curl-free)}.
$$
Equivalently, $\mathcal{R} = d\Phi$ for some scalar potential $\Phi$ (the solenoidal and harmonic components vanish).

**Conservative Special Case:** When $\mathcal{F} = 0$ everywhere, we recover standard scalar value functions with $V(z) = \Phi(z)$. The cumulative reward around any closed loop vanishes:

$$
\oint_\gamma \mathcal{R} = \int_\Sigma d\mathcal{R} = \int_\Sigma \mathcal{F} = 0.
$$

*Remark.* Standard RL assumes conservative reward fields. The scalar value function $V(s)$ exists precisely because path-independence holds.

:::

:::{prf:proposition} Value Cycle Detection
:label: prop-value-cycle-detection

The Value Curl $\mathcal{F}$ can be estimated from trajectory data. For a closed loop $\gamma$ in latent space:

$$
\oint_\gamma \mathcal{R} = \int_\Sigma \mathcal{F} \, d\Sigma \neq 0 \implies \text{Non-conservative rewards.}
$$
**Diagnostic:** If the TD-error accumulated around closed loops in latent space has non-zero mean, the value field is non-conservative.

:::

:::{note} Connection to RL #31: Path-Independence as Degenerate Value Curl
**The General Law (Fragile Agent):**
The Reward 1-form $\mathcal{R}$ decomposes via Hodge theory into gradient, solenoidal, and harmonic components. The **Value Curl** $\mathcal{F} = d\mathcal{R}$ measures non-conservative structure:

$$
\oint_\gamma \mathcal{R} = \int_\Sigma \mathcal{F} \, d\Sigma
$$
Non-zero Value Curl implies optimal strategies may involve sustained orbiting rather than converging to fixed points.

**The Degenerate Limit:**
Assume $\mathcal{F} = 0$ everywhere (curl-free reward field).

**The Special Case (Standard RL):**

$$
V(s) = \mathbb{E}\left[\sum_{t=0}^\infty \gamma^t r_t \mid s_0 = s\right]
$$
This recovers the **scalar Value function**, which exists precisely because rewards are path-independent (conservative). The Value at state $s$ is well-defined regardless of how the agent arrived there.

**What the generalization offers:**
- **Non-transitive games:** Rock-Paper-Scissors has $\mathcal{F} \neq 0$; no scalar $V$ exists
- **Cyclic exploration:** Optimal agents may orbit through value cycles indefinitely
- **Richer equilibria:** NESS (Non-Equilibrium Steady States) with persistent probability currents
:::

(sec-the-bulk-potential-screened-poisson-equation)=
### 24.3 The Conservative Case: Scalar Potential and Screened Poisson Equation

When the Value Curl vanishes ($\mathcal{F} = 0$), the reward field is conservative and we recover the standard scalar value function framework. In this regime, the Value function $V(z) = \Phi(z)$ obeys the Bellman Equation, which in the continuum limit becomes the **Screened Poisson (Helmholtz) Equation**.

:::{prf:theorem} The HJB-Helmholtz Correspondence {cite}`bellman1957dynamic,evans2010pde`
:label: thm-the-hjb-helmholtz-correspondence

Let the discount factor be $\gamma = e^{-\kappa \Delta t}$ where $\kappa > 0$ is the **screening mass**. The Bellman condition

$$
V(z) = \mathbb{E}[r + \gamma V(z')]
$$
approaches the following PDE in the limit $\Delta t \to 0$:

$$
\boxed{-\Delta_G V(z) + \kappa^2 V(z) = \rho_r(z)}
$$
where:
- $\Delta_G = \frac{1}{\sqrt{|G|}} \partial_i \left( \sqrt{|G|} G^{ij} \partial_j \right)$ is the **Laplace-Beltrami operator** on the manifold $(\mathcal{Z}, G)$
- $\kappa^2$ is the "mass" of the scalar field, causing the influence of distant rewards to decay exponentially
- $\rho_r(z)$ is the internal reward density plus propagated boundary conditions

*Proof sketch.* Consider the continuous-time limit of the Bellman equation for a diffusion process $dz = b(z) dt + \sigma(z) dW$ with $\sigma\sigma^T = 2T_c G^{-1}$. Expanding $V(z') = V(z + dz)$ to second order and taking expectations:

$$
V(z) = r \Delta t + \gamma \mathbb{E}[V(z')] \approx r \Delta t + (1 - \kappa \Delta t)\left(V + \nabla V \cdot b \Delta t + T_c \Delta_G V \Delta t\right).
$$
Rearranging and dividing by $\Delta t$, then taking $\Delta t \to 0$:

$$
\kappa V = r + \nabla V \cdot b + T_c \Delta_G V.
$$
For the stationary case ($b = 0$) and absorbing the temperature into the source term, this yields the Helmholtz equation $-\Delta_G V + \kappa^2 V = \rho_r$. Details in Appendix A.5. $\square$

Units: $[\kappa] = 1/\text{length}$, $[\Delta_G V] = \mathrm{nat}/\text{length}^2$, $[\rho_r] = \mathrm{nat}/\text{length}^2$.

*Cross-reference (Relativistic Extension):* This **elliptic** Helmholtz equation assumes instantaneous value propagation. When agents interact across spatial or computational separation with finite information speed $c_{\text{info}}$, the equation generalizes to the **hyperbolic Klein-Gordon equation**: $(\frac{1}{c^2}\partial_t^2 - \Delta_G + \kappa^2)V = \rho_r$. See Theorem {prf:ref}`thm-hjb-klein-gordon` in Section 29.5.

*Cross-reference (Gauge-Covariant Generalization):* When the dynamics must be invariant under local nuisance transformations (Section 29.13), all partial derivatives $\partial_\mu$ are promoted to covariant derivatives $D_\mu = \partial_\mu - igA_\mu$, where $A_\mu$ is the Strategic Connection (Definition {prf:ref}`def-strategic-connection`). The Helmholtz operator becomes $-D_\mu D^\mu + \kappa^2$.

:::

::::{admonition} Physics Isomorphism: Yukawa Potential
:class: note
:name: pi-yukawa-potential

**In Physics:** The Yukawa (screened Coulomb) potential satisfies $(-\nabla^2 + m^2)\phi = \rho$ where $m$ is the mediating boson mass. The screening length $\ell = 1/m$ determines the range of the force {cite}`yukawa1935interaction`.

**In Implementation:** The value function satisfies $(-\Delta_G + \kappa^2)V = \rho_r$ where:

$$
\kappa = -\ln\gamma, \quad \ell_\gamma = 1/\kappa
$$
**Correspondence Table:**

| Physics (Yukawa) | Agent (Bellman-Helmholtz) |
|:-----------------|:--------------------------|
| Scalar field $\phi$ | Value function $V(z)$ |
| Mass $m$ | Discount rate $\kappa = -\ln\gamma$ |
| Screening length $1/m$ | Reward horizon $\ell_\gamma = 1/\kappa$ |
| Charge density $\rho$ | Reward density $\rho_r$ |
| Laplacian $\nabla^2$ | Laplace-Beltrami $\Delta_G$ |

**Loss Function:** PINN regularizer enforcing $\|(-\Delta_G + \kappa^2)V - \rho_r\|^2$.
::::

:::{note} Connection to RL #4: Bellman Equation as Degenerate Helmholtz PDE
**The General Law (Fragile Agent):**
The Value Function $V(z)$ satisfies the **Screened Poisson (Helmholtz) Equation** on $(\mathcal{Z}, G)$:

$$
(-\Delta_G + \kappa^2) V(z) = \rho_r(z)
$$
where $\Delta_G$ is the Laplace-Beltrami operator on the Riemannian manifold and $\kappa = -\ln\gamma$ is the screening mass.

**The Degenerate Limit:**
Discretize space on a lattice. Replace $\Delta_G$ with the graph Laplacian $\mathcal{L}_{\text{graph}}$.

**The Special Case (Standard RL):**
The Green's function solution on a discrete graph is the **Neumann series** expansion:

$$
V(s) = \sum_{t=0}^\infty \gamma^t \mathbb{E}[r_t | s_0 = s] = (I - \gamma P)^{-1} r
$$
This recovers the **Bellman equation** $V = r + \gamma P V$.

**Result:** The "screening mass" $\kappa$ is the physicist's name for $-\ln\gamma$. Standard RL is Field Theory on a discrete lattice with flat metric. The Fragile Agent solves the PDE on a learned Riemannian manifold.

**What the generalization offers:**
- Geometric propagation: rewards propagate as sources in a scalar field, respecting manifold curvature
- Conformal coupling: high-value-curvature regions modulate the metric (Section 24.4)
- Continuous limit: natural extension to continuous state spaces without discretization artifacts
- Physical interpretation: $\gamma$ has a spatial meaning (screening length), not just temporal (horizon)
:::

:::{prf:proposition} Green's Function Interpretation
:label: prop-green-s-function-interpretation

The Critic computes the **Green's function** of the screened Laplacian on the latent geometry:

$$
V(z) = \int_{\partial\Omega} G_\kappa(z, z') \sigma_r(z') \, d\Sigma(z'),
$$
where $G_\kappa(z, z')$ is the Green's function satisfying $(-\Delta_G + \kappa^2) G_\kappa(z, \cdot) = \delta_z$.

*Remark.* The value at $z$ is a weighted integral of boundary rewards, with weights given by the Green's function. This is a superposition principle: the Helmholtz equation is linear.

:::

::::{admonition} Physics Isomorphism: Green's Function
:class: note
:name: pi-green-function

**In Physics:** The Green's function $G(x, x')$ is the fundamental solution satisfying $\mathcal{L}G(x, \cdot) = \delta(x - \cdot)$ for a linear operator $\mathcal{L}$. In electrostatics, $G$ is the potential at $x$ due to a unit charge at $x'$. For the screened Laplacian, $G_\kappa \sim e^{-\kappa r}/r^{(d-2)/2}$ {cite}`jackson1999classical`.

**In Implementation:** The Critic computes the Green's function of the screened Laplacian (Proposition {prf:ref}`prop-green-s-function-interpretation`):

$$
V(z) = \int_{\partial\Omega} G_\kappa(z, z') \sigma_r(z') \, d\Sigma(z')
$$
where $(-\Delta_G + \kappa^2) G_\kappa(z, \cdot) = \delta_z$.

**Correspondence Table:**
| Electrostatics | Agent (Critic) |
|:---------------|:---------------|
| Green's function $G(x, x')$ | Value kernel $G_\kappa(z, z')$ |
| Charge density $\rho$ | Reward density $\sigma_r$ |
| Electrostatic potential $\phi$ | Value function $V$ |
| Screening length $1/m$ | Reward horizon $\ell_\gamma = 1/\kappa$ |
| Superposition principle | Linearity of Helmholtz equation |

**Loss Function:** TD-error $\|(-\Delta_G + \kappa^2)V - \rho_r\|^2$ trains the critic as an implicit Green's function solver.
::::

:::{prf:proposition} Green's Function Decay
:label: prop-green-s-function-decay

On a manifold with bounded curvature, the Green's function decays exponentially:

$$
G_\kappa(z, z') \sim \frac{1}{d_G(z, z')^{(d-2)/2}} \exp\left(-\kappa \cdot d_G(z, z')\right),
$$
where $d_G$ is the geodesic distance and $d$ is the dimension.

:::
:::{prf:corollary} Discount as Screening Length
:label: cor-discount-as-screening-length

The discount factor $\gamma$ determines a characteristic **screening length**:

$$
\ell_{\text{screen}} = \frac{1}{\kappa} = \frac{\Delta t}{-\ln\gamma}.
$$
For $\gamma = 0.99$ and $\Delta t = 1$: $\ell_{\text{screen}} \approx 100$ steps.

*Interpretation:* Rewards at geodesic distance $> \ell_{\text{screen}}$ from state $z$ are exponentially suppressed in their contribution to $V(z)$. This is the **temporal horizon** recast as a **spatial horizon** in latent space.

**Table 24.2.5 (Discount-Screening Correspondence).**

| Discount $\gamma$ | Screening Mass $\kappa$ | Screening Length $\ell$ | Interpretation                    |
|-------------------|-------------------------|-------------------------|-----------------------------------|
| $\gamma \to 1$    | $\kappa \to 0$          | $\ell \to \infty$       | Infinite horizon (massless field) |
| $\gamma = 0.99$   | $\kappa \approx 0.01$   | $\ell \approx 100$      | Standard RL                       |
| $\gamma = 0.9$    | $\kappa \approx 0.1$    | $\ell \approx 10$       | Short horizon                     |
| $\gamma \to 0$    | $\kappa \to \infty$     | $\ell \to 0$            | Myopic (infinitely massive)       |

**Cross-references:** Section 2.7 (HJB Equation), Theorem {prf:ref}`thm-capacity-constrained-metric-law`.

:::

::::{note} Connection to RL #30: Temporal Horizon as Degenerate Screening Length
**The General Law (Fragile Agent):**
The discount factor $\gamma$ defines a **Screening Length** with geometric meaning:

$$
\kappa = -\ln\gamma, \quad \ell_{\text{screen}} = \frac{1}{\kappa}
$$
Value correlations decay exponentially with **geodesic distance**:

$$
G_\kappa(z, z') \sim \frac{1}{d_G(z,z')^{(d-2)/2}} \exp\!\left(-\kappa \cdot d_G(z, z')\right)
$$
The Critic computes the Green's function of the screened Laplacian $(-\Delta_G + \kappa^2)^{-1}$.

**The Degenerate Limit:**
Interpret $\gamma$ purely temporally. Ignore spatial/geometric structure of the latent space.

**The Special Case (Standard RL):**

$$
V(s) = \sum_{t=0}^\infty \gamma^t r_t
$$
This recovers the standard **temporal horizon interpretation** where $\gamma$ controls how far into the future rewards are considered.

**What the generalization offers:**
- **Spatial credit assignment:** Rewards propagate via geodesics in latent space, not just time steps
- **Physical units:** $\kappa$ has units of inverse length; $\ell_{\text{screen}}$ is a correlation length
- **Green's function decay:** The Critic is a PDE solver; value correlations decay geometrically
- **Unified view:** Temporal and spatial horizons are the same phenomenon in different coordinates
::::

(sec-thermodynamic-interpretation-energy-vs-probability)=
### 24.4 Thermodynamic Interpretation: Energy vs Probability

We explicitly resolve the ambiguity between "Energy" and "Probability" in the value function interpretation. The scalar potential $\Phi(z)$ from the Hodge decomposition plays the role of Gibbs Free Energy in the conservative case.

:::{prf:axiom} The Generalized Boltzmann-Value Law
:label: ax-the-boltzmann-value-law

The scalar potential $\Phi(z)$ from the Hodge decomposition (Theorem {prf:ref}`thm-hodge-decomposition`) represents the **Gibbs Free Energy** of the state $z$:

$$
\Phi(z) = E(z) - T_c S(z),
$$
where:
- $E(z)$ is the **task risk/cost** at state $z$
- $S(z)$ is the **exploration entropy** (measure of uncertainty/optionality)
- $T_c$ is the **cognitive temperature** (Section 21.1)

*Units:* $[\Phi] = [E] = [T_c S] = \mathrm{nat}$.

**Conservative Case ($\mathcal{F} = 0$):** When the Value Curl vanishes, $\mathcal{R} = d\Phi$ and the scalar potential $\Phi$ is the complete value function $V(z)$.

**Non-Conservative Case ($\mathcal{F} \neq 0$):** The scalar potential $\Phi$ captures only the optimizable component of the reward field. The solenoidal component $\delta\Psi$ creates additional cyclic dynamics.

:::
:::{prf:definition} Canonical Ensemble {cite}`sutton2018rl`
:label: def-canonical-ensemble

This potential induces a probability measure on the manifold via the **Canonical Ensemble**:

$$
P_{\text{stationary}}(z) = \frac{1}{Z} \exp\left(\frac{V(z)}{T_c}\right),
$$
where $Z = \int_{\mathcal{Z}} \exp(V(z)/T_c) \, d\mu_G(z)$ is the partition function.

*Sign Convention:* If $V$ is "Reward" (higher is better), use $+V/T_c$. If $V$ is "Cost" (lower is better), use $-V/T_c$. Throughout this document we use the **Reward convention** unless otherwise noted.

:::

::::{admonition} Physics Isomorphism: Canonical Ensemble
:class: note
:name: pi-canonical-ensemble

**In Physics:** The canonical ensemble describes a system in thermal equilibrium with a heat bath at temperature $T$. The probability of microstate $i$ is $P_i = Z^{-1}\exp(-E_i/k_B T)$ where $Z = \sum_i \exp(-E_i/k_B T)$ is the partition function {cite}`landau1980statistical`.

**In Implementation:** The stationary policy distribution (Definition {prf:ref}`def-canonical-ensemble`):

$$
P_{\text{stationary}}(z) = \frac{1}{Z} \exp\left(\frac{V(z)}{T_c}\right)
$$
where $Z = \int_{\mathcal{Z}} \exp(V(z)/T_c) \, d\mu_G(z)$ is the partition function.

**Correspondence Table:**
| Statistical Mechanics | Agent (MaxEnt RL) |
|:----------------------|:------------------|
| Energy $E$ | Negative reward $-r$ |
| Temperature $k_B T$ | Cognitive temperature $T_c$ |
| Partition function $Z$ | Soft value normalization |
| Free energy $F = -k_BT\log Z$ | Soft value function |
| Boltzmann distribution | MaxEnt optimal policy $\pi^* \propto \exp(Q/T_c)$ |
| Entropy $S = -k_B\sum p\log p$ | Policy entropy $H(\pi)$ |

**Consequence:** MaxEnt RL is not an approximation—it is the exact solution to entropy-regularized control, recovering the Boltzmann distribution as the unique maximizer.
::::

:::{prf:theorem} WFR Consistency: Value Creates Mass
:label: thm-wfr-consistency-value-creates-mass

In the WFR dynamics (Section 20), the reaction rate $r(z)$ in the unbalanced continuity equation is determined by the value function:

$$
r(z) = \frac{1}{s_r} \left( V(z) - \bar{V} \right),
$$
where $\bar{V} = \mathbb{E}_\rho[V]$ is the mean value and $s_r$ is the reaction time scale (computation time).

*Consequence:* The mass evolution satisfies:

$$
\dot{m}(s) = m(s) \cdot r(z(s)) \propto m(s) \cdot (V(z(s)) - \bar{V}).
$$

Probability density increases in regions where $V > \bar{V}$ and decreases where $V < \bar{V}$.

*Proof.* The WFR optimal reaction rate minimizes $\int \lambda^2 r^2 \, d\rho$ subject to the constraint that the endpoint marginals match. The solution is $r \propto (V - \bar{V})$, where $V$ appears because it determines the target stationary distribution. $\square$

:::
:::{prf:corollary} Conservative Equilibrium Distribution
:label: cor-equilibrium-distribution

**Conservative Case ($\mathcal{F} = 0$):** At equilibrium ($\partial_s \rho = 0$), the WFR dynamics with reaction rate $r(z) \propto (\Phi(z) - \bar{\Phi})$ converge to the Boltzmann distribution:

$$
\rho_\infty(z) \propto \exp\left(\frac{\Phi(z)}{T_c}\right),
$$
which is exactly the canonical ensemble (Definition {prf:ref}`def-canonical-ensemble`).

*Remark.* In the conservative case, the stationary distribution has zero probability current ($J = 0$). The distribution concentrates in high-$\Phi$ regions with concentration controlled by $T_c$.

:::

:::{prf:theorem} Non-Equilibrium Steady State (NESS)
:label: thm-ness-existence

**Non-Conservative Case ($\mathcal{F} \neq 0$):** If the Value Curl does not vanish, the stationary distribution $\rho_\infty$ is a **Non-Equilibrium Steady State** satisfying:

1. **Stationarity:** $\partial_s \rho_\infty = 0$
2. **Persistent Current:** The probability current $J = \rho v - D\nabla\rho$ is non-zero and divergence-free: $\nabla \cdot J = 0$ but $J \neq 0$
3. **Entropy Production:** The system continually produces entropy at rate:

$$
\dot{S}_i = \int_{\mathcal{Z}} \frac{\|J\|_G^2}{\rho D} \, d\mu_G > 0
$$

*Remark.* The probability density $\rho_\infty$ is time-independent, but individual trajectories circulate indefinitely. This distinguishes NESS from true equilibrium (where $J = 0$).

:::

:::{prf:proposition} NESS Decomposition
:label: prop-ness-decomposition

The probability current in a NESS decomposes into:

$$
J = J_{\text{gradient}} + J_{\text{cyclic}}
$$
where:
- $J_{\text{gradient}} = -D\rho\nabla\ln\rho + \rho\nabla\Phi$ derives from the scalar potential
- $J_{\text{cyclic}} = \rho \cdot v_{\text{curl}}$ derives from the solenoidal component

At stationarity, $\nabla \cdot J = 0$, but only $J_{\text{gradient}} = 0$ at true equilibrium. NESS has $J_{\text{cyclic}} \neq 0$.

:::

**Table 24.4.5 (Thermodynamic-RL Dictionary).**

| Thermodynamics         | RL / Control                               | Mathematical Object |
|------------------------|--------------------------------------------|---------------------|
| Energy $E$             | Negative reward $-r$                       | Instantaneous cost  |
| Free Energy $F$        | Scalar potential $\Phi$                    | Gibbs free energy   |
| Temperature $T$        | Cognitive temperature $T_c$                | Entropy weighting   |
| Entropy $S$            | Policy entropy $H(\pi)$                    | Exploration measure |
| Partition function $Z$ | Soft value $\log \sum_a \exp(Q/T_c)$       | Normalization       |
| Boltzmann distribution | MaxEnt policy $\pi^* \propto \exp(Q/T_c)$ | Conservative solution |
| **Probability current $J$** | **Value harvesting flow** | **NESS circulation** |
| **Entropy production $\dot{S}_i$** | **Cyclic reward rate** | **Perpetual motion** |

**Cross-references:** Section 20.2 (WFR dynamics), Section 23.4 (Thermodynamic Cycle), Section 14.2 (MaxEnt control), Theorem {prf:ref}`thm-hodge-decomposition` (Hodge Decomposition).

:::

:::{prf:corollary} The Varentropy-Stability Relation (Cognitive Heat Capacity)
:label: cor-varentropy-stability

Let $\mathcal{I}(a|z) = -\ln \pi(a|z)$ be the surprisal of an action. Define the **Policy Varentropy** $V_H(z)$ as the variance of the surprisal under the Boltzmann policy:

$$
V_H(z) := \mathrm{Var}_{a \sim \pi}[\mathcal{I}(a|z)] = \mathbb{E}_{\pi}\left[ \left( \ln \pi(a|z) + H(\pi) \right)^2 \right].
$$
*Units:* $\mathrm{nat}^2$.

Under the Boltzmann-Value Law (Axiom {prf:ref}`ax-the-boltzmann-value-law`), the Varentropy equals the **Heat Capacity** $C_v$ of the decision state:

$$
V_H(z) = \beta^2 \mathrm{Var}_\pi[Q] = C_v,
$$
where $\beta = 1/T_c$ is the inverse cognitive temperature. Equivalently:

$$
V_H(z) = T_c \frac{\partial H(\pi)}{\partial T_c}.
$$
**Operational Consequence:**
1. **Thermal Stability:** $V_H$ measures the sensitivity of the agent's exploration strategy to changes in the cognitive temperature $T_c$.
2. **Phase Transitions:** A divergence or spike in $V_H$ signals a second-order phase transition (critical point) where the policy is bifurcating from a single mode to multiple modes (or collapsing).
3. **Governor Constraint:** To ensure quasi-static evolution (reversible learning), the annealing rate $\dot{T}_c$ must satisfy the adiabatic condition:

$$
|\dot{T}_c| \ll \frac{T_c}{\sqrt{V_H(z)}}.
$$
*Proof:* See Appendix {ref}`E.8 <sec-appendix-e-proof-of-corollary-varentropy-stability>`.

:::
(sec-geometric-back-reaction-the-conformal-coupling)=
### 24.5 Geometric Back-Reaction: The Conformal Coupling

Does the Reward field change the Geometry? **Yes.** From Theorem {prf:ref}`thm-capacity-constrained-metric-law`, the curvature is driven by the Risk Tensor. Both the scalar potential $\Phi$ and the Value Curl $\mathcal{F}$ contribute to risk, and therefore modify the metric.

:::{prf:definition} Value-Metric Conformal Coupling
:label: def-value-metric-conformal-coupling

We model the effect of Value on the Metric $G$ as a **Conformal Transformation**:

$$
\tilde{G}_{ij}(z) = \Omega^2(z) \cdot G_{ij}(z),
$$
where the conformal factor $\Omega(z)$ depends on the **Hessian of the Value**:

$$
\Omega(z) = 1 + \alpha_{\text{conf}} \cdot \|\nabla^2_G V(z)\|_{\text{op}},
$$
with $\alpha_{\text{conf}} \ge 0$ the conformal coupling strength and $\|\cdot\|_{\text{op}}$ the operator norm.

Units: $[\Omega] = 1$ (dimensionless), $[\alpha_{\text{conf}}] = \text{length}^2/\mathrm{nat}$.

:::

::::{admonition} Physics Isomorphism: Conformal Transformation
:class: note
:name: pi-conformal-coupling

**In Physics:** A conformal transformation rescales the metric by a position-dependent factor: $\tilde{g}_{\mu\nu} = \Omega^2(x) g_{\mu\nu}$. In scalar field theory, conformal coupling $\xi R\phi^2$ couples the field to spacetime curvature. Weyl transformations preserve angles but not distances {cite}`wald1984general`.

**In Implementation:** The value-metric conformal coupling (Definition {prf:ref}`def-value-metric-conformal-coupling`):

$$
\tilde{G}_{ij}(z) = \Omega(z)^2 \cdot G_{ij}(z), \quad \Omega(z) = 1 + \alpha_{\text{conf}} \|\nabla^2_G V(z)\|_{\text{op}}
$$
**Correspondence Table:**
| Conformal Field Theory | Agent (Value-Metric Coupling) |
|:-----------------------|:------------------------------|
| Conformal factor $\Omega^2$ | $\left(1 + \alpha\|\nabla^2 V\|\right)^2$ |
| Weyl rescaling | Value-dependent metric inflation |
| Conformal anomaly | Curvature-dependent deliberation cost |
| Preserved angles | Preserved local policy directions |
| Dilated distances | Increased caution in high-curvature regions |

**Effect:** High-curvature value regions acquire increased effective mass, automatically slowing the agent near critical decisions.
::::

:::{prf:proposition} Risk-Curvature Mechanism
:label: prop-risk-curvature-mechanism

The conformal factor encodes the local "importance" of the value landscape:

| Value Landscape           | $\lVert\nabla^2 V\rVert$ | $\Omega$    | Effect                           |
|---------------------------|--------------------------|-------------|----------------------------------|
| **Flat** (low importance) | $\approx 0$              | $\approx 1$ | Default hyperbolic bulk geometry |
| **Curved** (ridge/valley) | $\gg 0$                  | $\gg 1$     | Distances expand, mass increases |
| **Saddle** (transition)   | moderate                 | $> 1$       | Intermediate slowdown            |

:::
:::{prf:corollary} Inertia at Critical Regions
:label: cor-inertia-at-critical-regions

Near sharp ridges or valleys of $V$ (where $\|\nabla^2 V\|$ is large), the conformal factor causes:

1. **Inertia Increase:** The effective mass $\tilde{G}(z) = \Omega^2(z) G(z)$ increases, so the agent slows down near critical decision boundaries (Section 22.2 mass scaling).

2. **Resolution Increase:** The capacity-constrained metric allocates more volume to high-curvature regions (Theorem {prf:ref}`thm-capacity-constrained-metric-law`), allowing higher-fidelity representation of value gradients.

3. **Stability:** The agent cannot "rush through" regions of high value curvature—it is forced to carefully navigate decision boundaries.

*Remark (Physical analogy).* The conformal scaling of effective velocity is mathematically analogous to gravitational time dilation in general relativity, where proper time dilates in regions of high gravitational potential.

:::
:::{prf:proposition} Conformal Laplacian Transformation
:label: prop-conformal-laplacian-transformation

Under the conformal transformation $G \to \tilde{G} = \Omega^2 G$, the Laplace-Beltrami operator acting on a scalar function $f$ transforms as:

$$
\Delta_{\tilde{G}} f = \Omega^{-2} \left( \Delta_G f + (d-2) \frac{G^{ij} \partial_i \Omega}{\Omega} \partial_j f \right),
$$
where $d$ is the dimension. For the Value function $V$ itself (which determines $\Omega$), this creates a **nonlinear coupling**. The screened Poisson equation (Theorem {prf:ref}`thm-the-hjb-helmholtz-correspondence`) in the conformally modified metric becomes:

$$
-\Delta_{\tilde{G}} V + \tilde{\kappa}^2 V = \tilde{\rho}_r,
$$
with effective screening mass $\tilde{\kappa}^2 = \Omega^{-2} \kappa^2$.

*Remark (Self-Consistency).* Since $\Omega$ depends on $\nabla^2 V$, the equation becomes nonlinear: the geometry adapts to the value landscape which in turn affects the geometry. In practice, we solve this iteratively or treat $\Omega$ as slowly-varying.

*Interpretation:* In high-curvature regions ($\Omega$ large), the effective screening mass decreases, making the field more "massless" and allowing longer-range correlations. This is the **self-focusing** effect: important regions become more interconnected.

**Cross-references:** Theorem {prf:ref}`thm-capacity-constrained-metric-law`, Section 22.1 (Mass=Metric), Proposition {prf:ref}`prop-mass-scaling-near-boundary`.

:::

::::{note} Connection to RL #27: Auxiliary Tasks as Degenerate Conformal Back-Reaction
**The General Law (Fragile Agent):**
The value function modulates the metric via **Conformal Coupling**:

$$
\tilde{G}_{ij} = \Omega^2(z)\, G_{ij}, \quad \Omega(z) = 1 + \alpha_{\text{conf}} \|\nabla^2_G V(z)\|_{\text{op}}
$$
High-curvature value regions acquire inertia, slowing agent dynamics near critical decision boundaries.

**The Degenerate Limit:**
Remove metric coupling ($\alpha_{\text{conf}} \to 0$). Treat auxiliary losses as independent add-ons.

**The Special Case (Standard RL):**

$$
\mathcal{L} = \mathcal{L}_{\text{RL}} + \sum_k \lambda_k \mathcal{L}_{\text{aux},k}
$$
This recovers **Auxiliary Tasks** (reward prediction, inverse dynamics, world models) {cite}`jaderberg2017unreal`.

**What the generalization offers:**
- **Explicit geometry feedback:** Value landscape modifies the metric, not just the loss
- **Inertia in risky regions:** High value curvature physically slows exploration
- **Self-consistency:** $\Omega$ depends on $V$, creating a nonlinear feedback loop
- **Resolution allocation:** Capacity-constrained metric assigns more volume to high-curvature regions
::::

(sec-implementation-the-holographiccritic-module)=
### 24.6 Implementation: The HolographicCritic Module

We update the architecture to include the Critic as the third pillar of the Holographic Interface. The Critic is not merely a value predictor—it is the **Field Solver** that computes the potential landscape from boundary charges.

```python
import torch
import torch.nn as nn
from torch import Tensor
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class CriticConfig:
    """Configuration for the HolographicCritic (Section 24)."""
    latent_dim: int = 32          # Dimension of latent space Z
    hidden_dim: int = 256         # Hidden layer dimension
    gamma: float = 0.99           # Discount factor
    alpha_conf: float = 0.1       # Conformal coupling strength (Definition 24.4.1)
    grad_reg_weight: float = 0.01 # Geometric gradient regularization

    @property
    def screening_mass(self) -> float:
        """Corollary 24.2.4: kappa = -ln(gamma)/Delta_t. Assumes Delta_t = 1."""
        return -torch.log(torch.tensor(self.gamma)).item()


class HolographicCritic(nn.Module):
    """
    Section 24: The Reward Encoder / Field Solver.

    Maps Boundary Charges (rewards r) to Bulk Potential (value V).
    Solves the Screened Poisson Equation on the latent manifold (Theorem {prf:ref}`thm-the-hjb-helmholtz-correspondence`).

    The Critic does not "predict" reward—it PROPAGATES boundary conditions
    into the bulk to compute the resulting potential field.
    """

    def __init__(self, config: CriticConfig):
        super().__init__()
        self.config = config
        self.kappa = config.screening_mass  # Screening mass (Corollary 24.2.4)
        self.alpha_conf = config.alpha_conf

        # Geometry-aware network for V(z)
        # Note: SiLU activation preserves smoothness needed for Hessian computation
        self.net = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, 1)  # Scalar potential V(z)
        )

        # Initialize to near-zero to start with flat potential
        self._init_weights()

    def _init_weights(self):
        """Initialize to small weights for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z: Tensor) -> Tensor:
        """
        Compute V(z): the scalar potential at bulk location z.

        Args:
            z: Latent positions [B, D]

        Returns:
            V: Scalar potential [B, 1]
        """
        return self.net(z)

    def compute_helmholtz_loss(
        self,
        z: Tensor,
        z_next: Tensor,
        r: Tensor,
        metric: 'PoincareDiskMetric'
    ) -> Tuple[Tensor, dict]:
        """
        Enforce the Screened Poisson/Bellman equation (Theorem {prf:ref}`thm-the-hjb-helmholtz-correspondence`).

        The loss has two components:
        1. TD Error: Enforces Bellman consistency (the PDE source term)
        2. Geometric Regularization: Ensures V respects manifold structure

        Args:
            z: Current latent positions [B, D]
            z_next: Next latent positions [B, D]
            r: Rewards [B, 1]
            metric: The Riemannian metric on Z

        Returns:
            loss: Total loss scalar
            info: Dictionary with diagnostic values
        """
        gamma = self.config.gamma

        # Compute V(z) and V(z')
        V = self(z)
        with torch.no_grad():
            V_next = self(z_next)

        # 1. TD Error (Bellman/Helmholtz source term)
        # V(z) = r + gamma * V(z') corresponds to the PDE source
        td_error = V - (r + gamma * V_next)
        loss_pde = td_error.pow(2).mean()

        # 2. Geometric Regularization
        # Penalize large metric-weighted gradient norm ||grad V||_G^2
        # This is a smoothness prior that respects the manifold geometry
        z.requires_grad_(True)
        V_for_grad = self(z)
        grad_V = torch.autograd.grad(
            V_for_grad.sum(), z, create_graph=True
        )[0]  # [B, D]

        # Compute ||grad V||_G^2 = G^{ij} (d_i V)(d_j V)
        G_inv = metric.inverse(z)  # [B, D, D]
        grad_norm_sq = torch.einsum(
            'bi,bij,bj->b', grad_V, G_inv, grad_V
        )  # [B]
        loss_smoothness = grad_norm_sq.mean()

        # Total loss
        loss = loss_pde + self.config.grad_reg_weight * loss_smoothness

        info = {
            'td_error': td_error.abs().mean().item(),
            'grad_norm': grad_norm_sq.sqrt().mean().item(),
            'V_mean': V.mean().item(),
            'V_std': V.std().item(),
        }

        return loss, info

    def compute_hessian_norm(self, z: Tensor) -> Tensor:
        """
        Compute ||nabla^2 V(z)||_op for conformal coupling (Definition 24.4.1).

        Args:
            z: Latent positions [B, D]

        Returns:
            hess_norm: Operator norm of Hessian [B]
        """
        B, D = z.shape
        z = z.requires_grad_(True)

        # Compute gradient
        V = self(z)  # [B, 1]
        grad_V = torch.autograd.grad(
            V.sum(), z, create_graph=True
        )[0]  # [B, D]

        # Compute Hessian row by row
        hessian = []
        for i in range(D):
            grad_i = torch.autograd.grad(
                grad_V[:, i].sum(), z, retain_graph=True
            )[0]  # [B, D]
            hessian.append(grad_i)

        H = torch.stack(hessian, dim=1)  # [B, D, D]

        # Operator norm = largest singular value
        # For efficiency, use Frobenius norm as upper bound
        hess_norm = torch.linalg.matrix_norm(H, ord='fro')  # [B]

        return hess_norm

    def conformal_factor(self, z: Tensor) -> Tensor:
        """
        Definition 24.4.1: Compute Omega(z) from Value Hessian.

        Omega(z) = 1 + alpha_conf * ||nabla^2 V(z)||

        Args:
            z: Latent positions [B, D]

        Returns:
            Omega: Conformal factor [B]
        """
        hess_norm = self.compute_hessian_norm(z)
        Omega = 1.0 + self.alpha_conf * hess_norm
        return Omega

    def conformally_scaled_metric(
        self,
        z: Tensor,
        base_metric: 'PoincareDiskMetric'
    ) -> Tensor:
        """
        Compute the conformally scaled metric G_tilde = Omega^2 * G.

        Args:
            z: Latent positions [B, D]
            base_metric: The base Poincare disk metric

        Returns:
            G_tilde: Conformally scaled metric [B, D, D]
        """
        Omega = self.conformal_factor(z)  # [B]
        G = base_metric(z)  # [B, D, D]
        G_tilde = Omega.unsqueeze(-1).unsqueeze(-1).pow(2) * G
        return G_tilde


def compute_wfr_reaction_rate(
    V: Tensor,
    s_r: float = 1.0
) -> Tensor:
    """
    Theorem {prf:ref}`thm-wfr-consistency-value-creates-mass`: Compute WFR reaction rate from Value function.

    r(z) = (V(z) - mean(V)) / s_r

    High value creates mass; low value depletes mass.

    Args:
        V: Value function evaluations [B, 1]
        s_r: Reaction time scale

    Returns:
        r: Reaction rates [B, 1]
    """
    V_mean = V.mean()
    r = (V - V_mean) / s_r
    return r
```

**Algorithm 24.5.1 (Critic Training Loop).**

```python
def train_critic_step(
    critic: HolographicCritic,
    batch: dict,
    metric: 'PoincareDiskMetric',
    optimizer: torch.optim.Optimizer
) -> dict:
    """
    Single training step for the HolographicCritic.
    Enforces the Screened Poisson equation (Theorem {prf:ref}`thm-the-hjb-helmholtz-correspondence`).
    """
    z = batch['z']           # Current latent [B, D]
    z_next = batch['z_next'] # Next latent [B, D]
    r = batch['reward']      # Rewards [B, 1]

    optimizer.zero_grad()
    loss, info = critic.compute_helmholtz_loss(z, z_next, r, metric)
    loss.backward()
    optimizer.step()

    return info
```

**Cross-references:** Section 22.4 (BAOAB integrator uses $\nabla\Phi_{\text{eff}}$), Section 23.7 (HolographicInterface).

(sec-the-unified-holographic-dictionary)=
### 24.7 The Unified Holographic Dictionary

This completes the **Holographic Dictionary** for the Fragile Agent. We now have a complete mapping between boundary data (observations, actions, rewards) and bulk objects (position, momentum, potential).

**Table 24.6.1 (Complete Holographic Dictionary).**

| Phenomenon     | Boundary (Data)  | Bulk (Latent)                           | Mathematical Object | Neural Component | Boundary Condition         | Section |
|----------------|------------------|-----------------------------------------|---------------------|------------------|----------------------------|---------|
| **Perception** | Pixels $\phi(x)$ | Position $q \in \mathcal{Z}$            | Manifold Point      | Visual Encoder   | Dirichlet (clamp position) | [23.1](#sec-the-symplectic-interface-position-momentum-duality) |
| **Action**     | Torques $A(x)$   | Momentum $p \in T\mathcal{Z}$           | Tangent Vector      | Action Encoder   | Neumann (clamp flux)       | [23.1](#sec-the-symplectic-interface-position-momentum-duality) |
| **Reward**     | Charge $r(x)$    | Potential $V \in C^\infty(\mathcal{Z})$ | Scalar Field        | Critic           | Source (Poisson)           | [24.1](#sec-the-reward-1-form) |
| **State**      | —                | $(q, p)$                                | Phase space point   | Full state       | Combined BCs               | [23.1](#sec-the-symplectic-interface-position-momentum-duality) |
| **Dynamics**   | —                | Geodesic flow                           | Hamiltonian flow    | BAOAB integrator | —                          | [22.4](#sec-the-geodesic-baoab-integrator) |

:::{prf:theorem} RL as Electrodynamics on a Curved Manifold
:label: thm-rl-as-electrodynamics-on-a-curved-manifold

The complete agent dynamics can be summarized as follows:

The agent is a **particle** with:
- **Position** $q \in \mathcal{Z}$ (from Perception / Dirichlet BC)
- **Momentum** $p \in T_q\mathcal{Z}$ (from Action / Neumann BC)
- **Mass** $G(q)$ (the Riemannian metric = information geometry)
- **Potential Energy** $V(q)$ (from Reward / Poisson source)
- **External Forces** $u_\pi(q)$ (from Policy / symmetry-breaking kick)

moving according to the **geodesic SDE** (Definition {prf:ref}`def-bulk-drift-continuous-flow`):

$$
dq^k = G^{kj}(q) p_j \, ds, \qquad
dp_k = -\frac{\partial V}{\partial q^k} ds - \frac{1}{2}\frac{\partial G^{ij}}{\partial q^k} p_i p_j \, ds + u_{\pi,k} \, ds + \sqrt{2T_c} \, (G^{1/2})_{kj} dW^j_s,
$$
on a **curved manifold** with metric $G$ satisfying the **capacity constraint** (Theorem {prf:ref}`thm-capacity-constrained-metric-law`), in a **screened potential** $V$ satisfying the **Helmholtz equation** (Theorem {prf:ref}`thm-the-hjb-helmholtz-correspondence`). The term $\frac{1}{2}\frac{\partial G^{ij}}{\partial q^k} p_i p_j$ encodes the geodesic correction (equivalent to Christoffel symbols in the position formulation).

*Conclusion:* **Reinforcement Learning is Electrodynamics on a Curved Manifold.** The standard RL components (encoder, critic, policy) are revealed to be the components of a field theory:

| RL Component      | Field Theory Role                                         |
|-------------------|-----------------------------------------------------------|
| Encoder           | **Coordinate Chart** (embedding from boundary to bulk)    |
| Critic            | **Field Solver** (Green's function of screened Laplacian) |
| Policy            | **External Force** (symmetry-breaking current)            |
| Discount $\gamma$ | **Screening Mass** (controls correlation length)          |
| Temperature $T_c$ | **Thermal Bath** (fluctuation-dissipation source)         |

:::
:::{prf:corollary} The Three Boundary Conditions
:label: cor-the-three-boundary-conditions

The agent-environment interface decomposes into exactly three types of boundary conditions:

1. **Dirichlet** (Sensors): Clamp position $q = q_{\text{obs}}$. Information flows **in**.
2. **Neumann** (Motors): Clamp flux $\nabla_n \cdot p = j_{\text{motor}}$. Information flows **out**.
3. **Source** (Rewards): Inject charge $\sigma_r$ at boundary. Creates **potential field**.

These three conditions fully specify the agent's interaction with its environment.

**Cross-references:** Section 23 (Holographic Interface), Section 22 (Equations of Motion), Section 18 (Capacity-Constrained Geometry).

:::
(sec-diagnostic-nodes-for-the-scalar-field)=
### 24.8 Diagnostic Nodes for the Reward Field

We define six diagnostic nodes (35-39, 61) to monitor the health of the Critic/Value system, including the new ValueCurlCheck for non-conservative reward fields.

(node-35)=
**Node 35: HelmholtzResidualCheck**

| **#** | **Name** | **Component** | **Type** | **Interpretation** | **Proxy** | **Cost** |
|-------|----------|---------------|----------|-------------------|-----------|----------|
| **35** | **HelmholtzResidualCheck** | **Critic** | **PDE Consistency** | Is the Helmholtz equation satisfied? | $\lVert-\Delta_G V + \kappa^2 V - \rho_r\rVert$ | $O(B \cdot D)$ |

**Trigger conditions:**
- High HelmholtzResidualCheck: Bellman equation not converged; Critic training unstable.
- Remedy: Reduce learning rate; increase batch size; check reward normalization.

(node-36)=
**Node 36: GreensFunctionDecayCheck**

| **#** | **Name** | **Component** | **Type** | **Interpretation** | **Proxy** | **Cost** |
|-------|----------|---------------|----------|-------------------|-----------|----------|
| **36** | **GreensFunctionDecayCheck** | **Critic** | **Screening Length** | Is value correlation decaying correctly? | $\mathbb{E}[\lVert V(z) - V(z')\rVert \cdot e^{\kappa d_G(z,z')}]$ | $O(B^2)$ |

**Trigger conditions:**
- High GreensFunctionDecayCheck: Value correlations extending beyond screening length; potential "leaking".
- Remedy: Check discount factor; verify metric computation; inspect reward structure.

(node-37)=
**Node 37: BoltzmannConsistencyCheck**

| **#** | **Name** | **Component** | **Type** | **Interpretation** | **Proxy** | **Cost** |
|-------|----------|---------------|----------|-------------------|-----------|----------|
| **37** | **BoltzmannConsistencyCheck** | **Critic + Policy** | **Equilibrium** | Does empirical distribution match Boltzmann? | $D_{\mathrm{KL}}(P_{\text{empirical}} \lVert P_{\text{Boltzmann}})$ | $O(B \cdot D)$ |

**Trigger conditions:**
- High BoltzmannConsistencyCheck: Agent not sampling from equilibrium distribution; exploration-exploitation imbalance.
- Remedy: Adjust cognitive temperature $T_c$; check policy entropy; verify WFR reaction rate.

(node-38)=
**Node 38: ConformalBackReactionCheck**

| **#** | **Name** | **Component** | **Type** | **Interpretation** | **Proxy** | **Cost** |
|-------|----------|---------------|----------|-------------------|-----------|----------|
| **38** | **ConformalBackReactionCheck** | **Critic** | **Geometry Coupling** | Is value curvature affecting metric appropriately? | $\text{Var}(\Omega(z))$ in high-$\lVert\nabla^2 V\rVert$ regions | $O(B \cdot D^2)$ |

**Trigger conditions:**
- Low ConformalBackReactionCheck: Value landscape is flat; agent not distinguishing important regions.
- High ConformalBackReactionCheck: Excessive metric distortion; agent "stuck" at decision boundaries.
- Remedy: Adjust conformal coupling $\alpha_{\text{conf}}$; verify Hessian computation.

(node-39)=
**Node 39: ValueMassCorrelationCheck**

| **#** | **Name** | **Component** | **Type** | **Interpretation** | **Proxy** | **Cost** |
|-------|----------|---------------|----------|-------------------|-----------|----------|
| **39** | **ValueMassCorrelationCheck** | **WFR + Critic** | **Mass Creation** | Is high value creating mass (WFR consistency)? | $\text{corr}(m_t, V(z_t))$ | $O(B)$ |

**Trigger conditions:**
- Low ValueMassCorrelationCheck: WFR reaction not aligned with value; agent not "materializing" in high-value regions.
- Remedy: Check reaction rate computation; verify WFR dynamics; inspect value function gradients.

(node-61)=
**Node 61: ValueCurlCheck**

| **#** | **Name** | **Component** | **Type** | **Interpretation** | **Proxy** | **Cost** |
|-------|----------|---------------|----------|-------------------|-----------|----------|
| **61** | **ValueCurlCheck** | **Critic** | **Topology** | Is the value field conservative? | $\oint_\gamma \delta_{\text{TD}} \approx \int \|\nabla \times \mathcal{R}\|$ | $O(T)$ |

**Trigger conditions:**
- Near-zero ValueCurlCheck ($\mathcal{F} \approx 0$): Reward field is conservative; standard RL applies. Equilibrium is a fixed point distribution.
- Non-zero ValueCurlCheck ($\mathcal{F} \neq 0$): Reward field is non-conservative; expect NESS with persistent probability currents. Consider:
  - **Productive curl:** Value cycles that harvest reward continuously (e.g., exploration-exploitation orbits)
  - **Pathological curl:** Indicates preference intransitivity or reward misspecification
- Remedy: If unexpected non-conservative structure, verify reward function consistency; check for cyclic dependencies in multi-objective rewards.

**Diagnostic Implementation:**
```python
def value_curl_check(
    z_trajectory: Tensor,  # [T, D] closed loop trajectory
    rewards: Tensor,       # [T] rewards along trajectory
) -> float:
    """
    Estimate Value Curl via loop integral of TD-errors.
    Non-zero return indicates non-conservative reward field.
    """
    # Cumulative reward around loop should be zero for conservative field
    loop_integral = rewards.sum().item()
    return abs(loop_integral)
```

**Table 24.8.1 (Section 24 Diagnostic Summary).**

| # | Name | Monitors | Healthy Range |
|---|------|----------|---------------|
| 35 | HelmholtzResidualCheck | PDE consistency | $< 0.1$ |
| 36 | GreensFunctionDecayCheck | Screening behavior | $\approx 1.0$ (constant after scaling) |
| 37 | BoltzmannConsistencyCheck | Equilibrium sampling | $< 0.5$ nats |
| 38 | ConformalBackReactionCheck | Geometry coupling | $0.1 < \text{Var}(\Omega) < 2.0$ |
| 39 | ValueMassCorrelationCheck | WFR-Value alignment | $> 0.5$ |
| 61 | ValueCurlCheck | Non-conservative structure | Context-dependent (see above) |

**Cross-references:** Section 3 (Sieve Diagnostic Nodes), Section 23.8 (Interface Diagnostics Nodes 30-34), Theorem {prf:ref}`thm-hodge-decomposition` (Hodge Decomposition), Definition {prf:ref}`def-value-curl` (Value Curl).



(sec-supervised-topology-semantic-potentials-and-metric-segmentation)=
