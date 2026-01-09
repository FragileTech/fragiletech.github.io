## 20. Wasserstein-Fisher-Rao Geometry: Unified Transport on Hybrid State Spaces

The latent bundle $\mathcal{Z} = \mathcal{K} \times \mathcal{Z}_n \times \mathcal{Z}_{\mathrm{tex}}$ (Section 2.2a) combines a discrete macro-state $K$ with continuous nuisance coordinates $z_n$. The product metric $d_{\mathcal{K}} \oplus G_n$ (Definition 2.2.1) and the Sasaki-like warped metric (Section 7.11.3) were heuristic constructions that treat the discrete and continuous components separately.

:::{admonition} Researcher Bridge: Handling Distribution Shift
:class: info
:name: rb-distribution-shift
Standard Bayesian filters fail during "surprises" because they can't handle mass appearing or disappearing (Unbalanced Transport). The **Wasserstein-Fisher-Rao (WFR)** metric allows the agent's belief to both **flow** (smooth tracking) and **jump** (teleporting probability mass). This provides a unified variational principle for both continuous state-tracking and discrete hypothesis-switching.
:::

This section introduces the **Wasserstein-Fisher-Rao (WFR)** metric—also known as **Hellinger-Kantorovich** {cite}`chizat2018unbalanced,liero2018optimal`—which provides a rigorous, unified variational principle. The key insight is to treat the agent's internal state not as a *point* in $\mathcal{Z}$, but as a *measure* (belief state) $\rho_s \in \mathcal{M}^+(\mathcal{Z})$ evolving on the bundle.

(sec-motivation-the-failure-of-product-metrics)=
### 20.1 Motivation: The Failure of Product Metrics

**The Problem with Sasaki-like Constructions.**

The metric tensor from Section 7.11.3 (where $\rho_{\text{depth}}$ denotes resolution depth, not density):

$$
ds^2 = d\rho_{\text{depth}}^2 + d\sigma_{\mathcal{K}}^2 + e^{-2\rho_{\text{depth}}}\|dz_n\|^2
$$
assumes a fixed point moving through the bundle. This creates two problems:

1. **Discontinuous Jumps:** When the agent transitions from chart $K_i$ to chart $K_j$, the metric provides no principled way to measure the "cost" of the jump versus continuous motion along an overlap.

2. **No Mass Conservation:** A point either is or isn't at a location. But the agent's *belief* can be partially in multiple charts simultaneously (soft routing, Section 7.8).

**The WFR Solution.**

The Wasserstein-Fisher-Rao metric resolves both issues by lifting dynamics to the space of measures $\mathcal{M}^+(\mathcal{Z})$. In this space:
- **Transport (Wasserstein):** Probability mass moves along continuous coordinates via the continuity equation.
- **Reaction (Fisher-Rao):** Probability mass is created/annihilated locally, enabling discrete chart transitions.

The metric determines the optimal path by minimizing the total cost: transport cost $\int\|v\|_G^2\,d\rho$ plus reaction cost $\int\lambda^2|r|^2\,d\rho$.

(sec-the-wfr-metric)=
### 20.2 The WFR Metric (Benamou-Brenier Formulation)

Let $\rho(s, z)$ be a time-varying density on the latent bundle $\mathcal{Z}$, evolving in computation time $s$. The WFR distance is defined by the minimal action of a generalized continuity equation.

:::{prf:definition} The Generalized WFR Action
:label: def-the-wfr-action

The squared WFR distance $d^2_{\mathrm{WFR}}(\rho_0, \rho_1)$ is the infimum of the generalized energy functional:

$$
\mathcal{E}[\rho, v, r] = \int_0^1 \int_{\mathcal{Z}} \left( \underbrace{\|v_s(z)\|_G^2}_{\text{Transport Cost}} + \underbrace{\lambda^2 |r_s(z)|^2}_{\text{Reaction Cost}} - \underbrace{2\langle \mathbf{A}(z), v_s(z) \rangle}_{\text{Vector Potential}} \right) d\rho_s(z) \, ds
$$
subject to the **Unbalanced Continuity Equation**:

$$
\partial_s \rho + \nabla \cdot (\rho v) = \rho r
$$
where:
- $v_s(z) \in T_z\mathcal{Z}$ is the **velocity field** (transport/flow)
- $r_s(z) \in \mathbb{R}$ is the **reaction rate** (growth/decay of mass)
- $\lambda > 0$ is the **length-scale parameter** balancing transport and reaction
- $G$ is the Riemannian metric on the continuous fibres (Section 2.5)
- $\mathbf{A}(z)$ is the **vector potential** satisfying $d\mathbf{A} = \mathcal{F}$ (the Value Curl)

*Units:* $[\mathbf{A}] = \mathrm{nat}/[\text{length}]$.

**Conservative Limit:** When $\mathcal{F} = 0$ (Definition {prf:ref}`def-conservative-reward-field`), we can choose the gauge $\mathbf{A} = 0$ and recover the standard WFR action without the vector potential term.

**Non-Conservative Case:** When $\mathcal{F} \neq 0$, the vector potential term couples the transport velocity to the solenoidal component of the reward field. The Euler-Lagrange equations of this action yield the Lorentz-Langevin equation (Definition {prf:ref}`def-bulk-drift-continuous-flow`).

*Remark (Gauge Invariance).* The action is invariant under gauge transformations $\mathbf{A} \to \mathbf{A} + d\chi$ for any scalar $\chi$, since $d(d\chi) = 0$. We fix the gauge via the Coulomb condition $\delta\mathbf{A} = 0$ (divergence-free).

*Forward reference (Boundary Conditions).* Section 23.5 specifies how boundary conditions on $\partial\mathcal{Z}$ (sensory and motor boundaries) constrain the WFR dynamics: **Waking** imposes Dirichlet (sensors) + Neumann (motors) BCs; **Dreaming** imposes reflective BCs on both, enabling recirculating flow without external input.

:::

::::{admonition} Physics Isomorphism: Wasserstein-Fisher-Rao Geometry
:class: note
:name: pi-wfr-metric

**In Physics:** The Wasserstein-Fisher-Rao (WFR) metric on probability measures combines optimal transport (Wasserstein) with information geometry (Fisher-Rao). It is the unique metric allowing both mass transport and creation/annihilation {cite}`liero2018optimal,chizat2018interpolating`.

**In Implementation:** The belief density $\rho$ evolves under the WFR metric on $\mathcal{P}(\mathcal{Z})$:

$$
d_{\text{WFR}}^2(\rho_0, \rho_1) = \inf_{\rho, v, r} \int_0^1 \int_{\mathcal{Z}} \left( \|v\|_G^2 + \lambda^2 r^2 \right) \rho \, d\mu_G \, dt
$$
**Correspondence Table:**
| Optimal Transport | Agent (Belief Dynamics) |
|:------------------|:------------------------|
| Wasserstein distance $W_2$ | Transport cost for belief |
| Fisher-Rao distance | Information cost for reweighting |
| Transport velocity $v$ | Belief flow in $\mathcal{Z}$ |
| Reaction rate $r$ | Mass creation/annihilation |
| Benamou-Brenier formula | Dynamic formulation |
| Geodesic interpolation | Optimal belief transition |

**Significance:** WFR unifies transport (Wasserstein) and reweighting (Fisher-Rao) in a single Riemannian geometry.
::::

:::{prf:remark} Units
:label: rem-units

$[v] = \text{length}/\text{time}$, $[r] = 1/\text{time}$, and $[\lambda] = \text{length}$. The ratio $\|v\|/(\lambda |r|)$ determines whether transport or reaction dominates.

:::
(sec-transport-vs-reaction-components)=
### 20.3 Transport vs. Reaction Components

The belief state $\rho_s$ evolves on the bundle $\mathcal{Z}$ via two mechanisms.

**1. Transport (Wasserstein Component):**
The density evolves via the continuity equation $\partial_s\rho + \nabla\cdot(\rho v) = 0$ along the continuous coordinates $z_n$. The transport cost is $\int \|v\|_G^2\, d\rho$. In the limit $r \to 0$, the dynamics reduce to the standard Wasserstein-2 ($W_2$) optimal transport on the Riemannian manifold.

**2. Reaction (Fisher-Rao Component):**
The density undergoes local mass creation/annihilation via the source term $\rho r$. This corresponds to discrete chart transitions: mass decreases on Chart A ($r < 0$) and increases on Chart B ($r > 0$). The reaction cost is $\int \lambda^2|r|^2\, d\rho$. In the limit $v \to 0$, the dynamics reduce to the Fisher-Rao metric on the probability simplex $\Delta^{|\mathcal{K}|}$.

**3. The Coupling Constant $\lambda$ (Reaction-Transport Crossover Scale):**

This parameter defines the characteristic length scale at which transport cost exceeds reaction cost:
- If $\|z_A - z_B\|_G < \lambda$: Transport is preferred (continuous regime)
- If $\|z_A - z_B\|_G > \lambda$: Reaction is preferred (discrete chart transition)

**Operational interpretation:** $\lambda$ is exactly the **radius of the chart overlap region** (Section 7.13). Within overlaps, transport is efficient; across non-overlapping regions, reaction dominates.

:::{prf:definition} Canonical length-scale
:label: def-canonical-length-scale

Let $G$ be the latent metric on $\mathcal{Z}$. The canonical choice for $\lambda$ is the **geodesic injectivity radius**:

$$
\lambda := \min_{z \in \mathcal{Z}} \text{inj}_G(z),
$$
where $\text{inj}_G(z)$ is the injectivity radius at $z$ -- the largest $r$ such that the exponential map $\exp_z: T_z\mathcal{Z} \to \mathcal{Z}$ is a diffeomorphism on $B_r(0)$.

*Default value.* If the injectivity radius is unknown or the metric is learned, a practical default is:

$$
\lambda_{\text{default}} = \sqrt{\frac{\text{tr}(G^{-1})}{n}} \approx \text{mean characteristic length of } \mathcal{Z}.
$$
This corresponds to the RMS geodesic step size in an isotropic metric.

*Cross-reference:* The screening length $\ell_{\text{screen}} = 1/\kappa$ from Section 24.2 plays an analogous role for temporal horizons; $\lambda$ plays the corresponding role for spatial horizons in the WFR geometry.

:::
(sec-reconciling-discrete-and-continuous)=
### 20.4 Reconciling Discrete and Continuous

:::{prf:proposition} Limiting Regimes
:label: prop-limiting-regimes

The WFR metric seamlessly unifies discrete and continuous dynamics:

1. **Continuous Movement (Flow):** When moving within a chart, $r \approx 0$. The dynamics are dominated by $\nabla \cdot (\rho v)$, and the metric reduces to $W_2$ (Wasserstein-2). This recovers the Riemannian manifold structure of the nuisance fibres.

2. **Discrete Movement (Jump):** When the flow reaches a topological obstruction (chart boundary without overlap), transport becomes infinitely expensive. It becomes cheaper to use the source term $r$:
   - $r < 0$ on the old chart (mass destruction)
   - $r > 0$ on the new chart (mass creation)
   This recovers the **Fisher-Rao metric** on the discrete simplex $\Delta^{|\mathcal{K}|}$.

3. **Mixed Regime (Overlap):** In chart overlaps, both $v$ and $r$ are active. The optimal path smoothly interpolates between transport and reaction.

*Proof sketch.* The cone-space representation of WFR (lifting $\rho$ to $(\sqrt{\rho}, \sqrt{\rho} \cdot z)$) shows that the WFR geodesic projects to a $W_2$ geodesic when $r = 0$, and to a Fisher-Rao geodesic when $v = 0$. $\square$

:::

::::{note} Connection to RL #26: Distributional RL as Degenerate WFR Geometry
**The General Law (Fragile Agent):**
Belief states evolve on $\mathcal{M}^+(\mathcal{Z})$ via **Wasserstein-Fisher-Rao dynamics**:

$$
d^2_{\text{WFR}}(\rho_0, \rho_1) = \inf \int_0^1 \int_{\mathcal{Z}} \left( \|v_s\|_G^2 + \lambda^2 |r_s|^2 \right) d\rho_s\, ds
$$
subject to the unbalanced continuity equation $\partial_s \rho + \nabla \cdot (\rho v) = \rho r$.

**The Degenerate Limit:**
Restrict to value distributions at single states (no spatial transport). Use Euclidean metric ($G \to I$).

**The Special Case (Standard RL):**

$$
Z(s, a) \stackrel{D}{=} R + \gamma Z(S', A'), \quad Q(s,a) = \mathbb{E}[Z(s,a)]
$$
This recovers **Distributional RL**: C51, QR-DQN, IQN {cite}`bellemare2017c51,dabney2018qrdqn`.

**What the generalization offers:**
- **Unified transport-reaction**: WFR handles continuous flow (within charts) and discrete jumps (between charts) in one framework
- **Belief geometry**: The metric on $\mathcal{M}^+(\mathcal{Z})$ respects both $W_2$ (spatial) and Fisher-Rao (probabilistic)
- **Teleportation length**: $\lambda$ determines when transport beats reaction (Proposition {prf:ref}`prop-limiting-regimes`)
- **GKSL embedding**: Quantum-like master equations embed naturally (Section 20.5)
::::

(sec-connection-to-gksl-master-equation)=
### 20.5 Connection to GKSL / Master Equation (Section 12.5)

The WFR framework provides a natural interpretation of the GKSL (Lindblad) master equation from Section 12.5.

**Correspondence Table:**

| GKSL Component                                                                                    | WFR Interpretation                                     |
|---------------------------------------------------------------------------------------------------|--------------------------------------------------------|
| $-i[H, \varrho]$ (Commutator)                                                                     | Transport velocity $v$ (Hamiltonian drift)             |
| $\sum_j \gamma_j(L_j \varrho L_j^\dagger - \frac{1}{2}\{L_j^\dagger L_j, \varrho\})$ (Dissipator) | Reaction rate $r$ (jump operators)                     |
| Unitary evolution                                                                                 | Mass-preserving transport ($\int \rho = \text{const}$) |
| Lindblad jumps                                                                                    | Mass redistribution ($r < 0$ source, $r > 0$ sink)     |

:::{prf:proposition} GKSL Embedding
:label: prop-gksl-embedding

The GKSL generator from Definition {prf:ref}`def-gksl-generator` embeds into the WFR framework as follows:
- The Hamiltonian $H$ generates the velocity field via $v \propto G^{-1}\nabla_z \langle H \rangle_\varrho$ (gradient of expected energy)
- Each Lindblad operator $L_j$ contributes to the reaction rate via $r \propto \sum_j \gamma_j(\mathrm{Tr}(L_j^\dagger L_j \varrho) - 1)$

This provides a **geometric foundation** for the otherwise algebraic GKSL construction. The correspondence is heuristic; see Carlen & Maas (2014) {cite}`carlen2014wasserstein` for rigorous connections between quantum Markov semigroups and gradient flows on Wasserstein space.

:::
(sec-the-unified-world-model)=
### 20.6 The Unified World Model

The WFR formulation enables a **single World Model** that predicts both transport and reaction, eliminating the need for separate "macro predictor" and "micro dynamics" modules.

:::{prf:definition} WFR World Model
:label: def-wfr-world-model

The policy outputs a generalized velocity field $(v, r)$ to minimize the WFR path length to the target distribution (goal).

```python
import torch
import torch.nn as nn
from typing import Tuple

class WFRWorldModel(nn.Module):
    """
    Unified World Model using Unbalanced Optimal Transport dynamics.

    Predicts the 'Generalized Velocity' (v, r) for belief particles.
    No separate 'discrete' and 'continuous' modules.
    """

    def __init__(
        self,
        macro_embed_dim: int,
        nuisance_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        # Input: particle state + action
        # State includes: macro embedding, nuisance coords, mass (weight)
        input_dim = macro_embed_dim + nuisance_dim + 1 + action_dim

        # Single MLP backbone for unified dynamics
        self.dynamics_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # Head 1: Transport velocity (Riemannian motion on fibre)
        self.head_v = nn.Linear(hidden_dim, nuisance_dim)

        # Head 2: Reaction rate (Fisher-Rao mass creation/destruction)
        self.head_r = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        z_t: torch.Tensor,           # [B, D] latent state (macro_embed + nuisance)
        mass_t: torch.Tensor,        # [B, 1] particle weight (belief mass)
        action_t: torch.Tensor,      # [B, A] action
        dt: float = 0.1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict next state via WFR dynamics.

        Returns:
            z_next: [B, D] next latent state
            mass_next: [B, 1] next particle mass
            v_t: [B, nuisance_dim] transport velocity
            r_t: [B, 1] reaction rate
        """
        # Unified prediction
        inp = torch.cat([z_t, mass_t, action_t], dim=-1)
        feat = self.dynamics_net(inp)

        v_t = self.head_v(feat)  # Transport velocity
        r_t = self.head_r(feat)  # Reaction rate (log-growth)

        # Integrate dynamics (Euler step)
        # Position update (Transport): z' = z + v * dt
        z_next = z_t.clone()
        z_next[..., -self.head_v.out_features:] += v_t * dt

        # Mass update (Reaction): m' = m * exp(r * dt)
        # If r > 0: hypothesis gaining probability (jumping in)
        # If r < 0: hypothesis losing probability (jumping out)
        mass_next = mass_t * torch.exp(r_t * dt)

        return z_next, mass_next, v_t, r_t
```

**How this handles the "Jump" seamlessly:**

- **Deep inside a Chart:** Model predicts $r \approx 0$ and $v \neq 0$. Particle moves normally.
- **Approaching a Boundary:** Model sees invalid description (high prediction error). Predicts $r < 0$ for current chart, $r > 0$ for neighboring chart particles.
- **Result:** Probability mass smoothly "tunnels" between charts without hard discrete switching.

:::
(sec-scale-renormalization)=
### 20.7 Scale Renormalization (Connection to Section 7.12)

For stacked TopoEncoders (Section 7.12), the WFR metric applies recursively with **scale-dependent coupling**.

Recall the WFR action:

$$
\mathcal{E} = \int \left( \|v\|_G^2 + \lambda^2 |r|^2 \right) d\rho
$$
For a hierarchy of layers $\ell = 0, \ldots, L$:

:::{prf:definition} Scale-Dependent Teleportation Cost
:label: def-scale-dependent-teleportation-cost

$$
\lambda^{(\ell)} \propto \sigma^{(\ell)} \quad \text{(jump cost scales with residual variance)}
$$
where $\sigma^{(\ell)}$ is the scale factor from Definition {prf:ref}`def-the-rescaling-operator-renormalization`.

**Interpretation:**
- **Layer 0 (Bulk / IR):** High $\lambda^{(0)}$. Jumping is expensive; macro-structure is rigid. Transport dominates.
- **Layer $L$ (Texture / UV):** Low $\lambda^{(L)}$. "Mass" (texture details) can appear/disappear cheaply. Reaction dominates.

**Correspondence with Cosmological Constant:**
In the capacity-constrained metric law (Section 18), the term $\Lambda G_{ij}$ plays the role of a baseline curvature. The correspondence is:

$$
\Lambda^{(\ell)} \sim \frac{1}{(\lambda^{(\ell)})^2}
$$
- Bulk (low $\Lambda$): Flat, rigid, transport-dominated
- Boundary (high $\Lambda$): Curved, fluid, reaction-dominated

:::
(sec-connection-to-einstein-equations)=
### 20.8 Connection to Einstein Equations (Section 18)

The WFR dynamics provide the **stress-energy tensor** $T_{ij}$ that drives curvature in Theorem {prf:ref}`thm-capacity-constrained-metric-law`.

:::{prf:theorem} WFR Stress-Energy Tensor; variational form
:label: thm-wfr-stress-energy-tensor-variational-form

Let the WFR action be

$$
\mathcal{S}_{\mathrm{WFR}}
=
\frac12\int_0^T\int_{\mathcal{Z}}
\rho\left(\|v\|_G^2+\lambda^2 r^2\right)\,d\mu_G\,ds,
$$
with continuity equation

$$
\partial_s\rho+\nabla\!\cdot(\rho v)=\rho r.
$$
Define

$$
T_{ij}:=
-\frac{2}{\sqrt{|G|}}\frac{\delta(\sqrt{|G|}\,\mathcal{L}_{\mathrm{WFR}})}{\delta G^{ij}}
\quad\text{(holding }\rho,v,r\text{ fixed).}
$$
Then

$$
T_{ij}=\rho\,v_i v_j + P\,G_{ij},
\qquad
P=\frac12\,\rho\left(\|v\|_G^2+\lambda^2 r^2\right),
$$
which is the perfect-fluid form with reaction contributing an additive pressure term
{math}`P_{\mathrm{react}}=\tfrac12\lambda^2\rho r^2`.

*Proof sketch.* Vary $\mathcal{S}_{\mathrm{WFR}}$ with respect to $G^{ij}$ while holding
$(\rho,v,r)$ fixed. Use $\delta\|v\|_G^2=-v_i v_j\,\delta G^{ij}$ and
$\delta d\mu_G=-\tfrac12 G_{ij}\delta G^{ij}d\mu_G$, then collect terms to match
$\delta\mathcal{S}_{\mathrm{WFR}}=-\tfrac12\int T_{ij}\delta G^{ij}d\mu_G\,ds$.
See Appendix C for the full derivation. $\square$

:::

::::{admonition} Physics Isomorphism: Stress-Energy Tensor
:class: note
:name: pi-stress-energy

**In Physics:** The stress-energy tensor $T_{\mu\nu}$ is derived from the variation of the matter action with respect to the metric: $T_{\mu\nu} = -\frac{2}{\sqrt{-g}}\frac{\delta S_M}{\delta g^{\mu\nu}}$ {cite}`wald1984general`.

**In Implementation:** The WFR stress-energy tensor (Theorem {prf:ref}`thm-wfr-stress-energy-tensor-variational-form`) is:

$$
T_{ij} = \rho\left(v_i v_j - \frac{1}{2}\|v\|_G^2 G_{ij}\right) + \frac{\lambda^2}{2}\rho r^2 G_{ij}
$$
derived from $\delta \mathcal{S}_{\text{WFR}}/\delta G^{ij}$.

**Correspondence Table:**

| Field Theory | Agent (WFR) |
|:-------------|:------------|
| Matter density $\rho_m$ | Belief density $\rho$ |
| 4-velocity $u^\mu$ | Transport velocity $v^i$ |
| Pressure $p$ | Reaction pressure $\frac{\lambda^2}{2}\rho r^2$ |
| Rest mass density | WFR kinetic energy $\frac{1}{2}\rho\|v\|_G^2$ |
::::

**Implications:**
1. **High velocity ($v$):** Agent moves fast through a region → $T_{ij}$ large → curvature $R_{ij}$ increases → latent space contracts. This is the **Natural Gradient** effect derived from first principles.

2. **High reaction ($r$):** Agent jumps frequently → $P_{\mathrm{react}}$ increases → capacity stress increases. This triggers the boundary-capacity constraint (Definition {prf:ref}`def-dpi-boundary-capacity-constraint`).

**Consistency with existing losses:**

| Existing Loss                                    | WFR Interpretation                             | Status     |
|--------------------------------------------------|------------------------------------------------|------------|
| $\mathcal{L}_{\mathrm{pred}}$ (Prediction)       | Minimizing transport cost $\lVert v\rVert_G^2$ | Compatible |
| $\mathcal{L}_{\mathrm{closure}}$ (Macro closure) | Penalizing reaction $r$ in macro channel       | Compatible |
| Dissipation (Axiom D)                            | $r < 0$ (entropy production)                   | Compatible |
| Capacity ($I < C$)                               | Metric curves to keep WFR path within budget   | Compatible |

:::

(sec-comparison-sasaki-vs-wfr)=
### 20.9 Comparison: Sasaki vs. WFR

| Feature                     | Sasaki (Product Metric)          | WFR (Unbalanced Transport)             |
|-----------------------------|----------------------------------|----------------------------------------|
| **State representation**    | Fixed point                      | Probability mass / belief              |
| **Topology changes**        | Manual patching required         | Handled natively via $r$               |
| **Path type**               | "Walk then Jump" (discontinuous) | Smooth interpolation                   |
| **Optimization**            | Combinatorial + Gradient descent | Convex (generalized geodesics)         |
| **Theoretical consistency** | Ad-hoc construction              | Gradient flow of entropy (rigorous)    |
| **Multi-scale**             | Separate metrics per scale       | Unified with scale-dependent $\lambda$ |

(sec-implementation-wfr-consistency-loss)=
### 20.10 Implementation: WFR Consistency Loss

:::{prf:definition} WFR Consistency Loss / WFRCheck
:label: def-wfr-consistency-loss-wfrcheck

The cone-space representation linearizes WFR locally. From $\partial_s \rho = \rho r - \nabla \cdot (\rho v)$ and $u = \sqrt{\rho}$, we have $\partial_s u = \frac{\rho r - \nabla \cdot (\rho v)}{2\sqrt{\rho}}$. Define the consistency loss:

$$
\mathcal{L}_{\mathrm{WFR}} = \left\| \sqrt{\rho_{t+1}} - \sqrt{\rho_t} - \frac{\Delta t}{2\sqrt{\rho_t}}\left(\rho_t r_t - \nabla \cdot (\rho_t v_t)\right) \right\|_{L^2}^2
$$
This penalizes deviations from the unbalanced continuity equation.

**Practical implementation:**

```python
def compute_wfr_consistency_loss(
    rho_t: torch.Tensor,       # [B, K] belief over charts at time t
    rho_t1: torch.Tensor,      # [B, K] belief over charts at time t+1
    v_t: torch.Tensor,         # [B, K, d_n] transport velocity per chart
    r_t: torch.Tensor,         # [B, K] reaction rate per chart
    dt: float = 0.1,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Compute WFR consistency loss (cone-space formulation).

    Penalizes violation of unbalanced continuity equation.
    """
    sqrt_rho_t = torch.sqrt(rho_t + eps)
    sqrt_rho_t1 = torch.sqrt(rho_t1 + eps)

    # Approximate divergence term (finite difference)
    # In practice, use automatic differentiation if v is differentiable
    div_rho_v = torch.zeros_like(rho_t)  # Placeholder for nabla . (rho v)

    # Predicted change in sqrt(rho) from: d/ds sqrt(rho) = (rho*r - div(rho*v)) / (2*sqrt(rho))
    predicted_delta = (dt / (2 * sqrt_rho_t + eps)) * (rho_t * r_t - div_rho_v)

    # Actual change
    actual_delta = sqrt_rho_t1 - sqrt_rho_t

    # L2 loss
    loss = ((actual_delta - predicted_delta) ** 2).mean()

    return loss
```

:::
(sec-node-wfrcheck)=
### 20.11 Node 23: WFRCheck

Following the diagnostic node convention (Section 3.1), we define:

| **#**  | **Name**     | **Component**   | **Type**                 | **Interpretation**          | **Proxy**                    | **Cost** |
|--------|--------------|-----------------|--------------------------|-----------------------------|------------------------------|----------|
| **23** | **WFRCheck** | **World Model** | **Dynamics Consistency** | Transport-Reaction balance? | $\mathcal{L}_{\mathrm{WFR}}$ | $O(BK)$  |

**Trigger conditions:**
- High $\mathcal{L}_{\mathrm{WFR}}$: World model's $(v, r)$ predictions violate continuity
- Remedy: Increase training on transitions; check for distribution shift



(sec-radial-generation-entropic-drift-and-policy-control)=
