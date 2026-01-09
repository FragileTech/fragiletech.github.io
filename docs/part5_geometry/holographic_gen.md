## 21. Radial Generation: Entropic Drift and Policy Control

{cite}`ho2020ddpm,sohldickstein2015deep,nickel2017poincare`

:::{admonition} Researcher Bridge: Diffusion-Style Generation with Policy Drift
:class: info
:name: rb-diffusion-generation
If you know diffusion or score-based models, the radial expansion here is the generative flow. The policy is the controllable drift term that steers generation toward high-value regions.
:::

Data generation is defined as **radial expansion** of the latent state from the low-entropy origin ($z=0$) toward the high-entropy boundary ($|z| \to 1$). The expansion is driven by the **entropic drift** (the natural tendency of hyperbolic volume to grow) and steered by the **policy control field** $u_\pi$.

This section establishes the following unification: by identifying the **policy** as the source of initial direction selection, we merge Generative Modeling and Reinforcement Learning into a single variational operation:
- **RL:** The policy chooses a direction to maximize value $V(z)$.
- **Generation:** The policy (or context) chooses a direction to maximize semantic alignment with conditioning.
- Both contribute to the drift term in the latent SDE.

(sec-hyperbolic-volume-and-entropic-drift)=
### 21.1 Hyperbolic Volume and Entropic Drift

Consider the latent agent as a particle in the Poincare disk $\mathbb{D} = \{z \in \mathbb{C} : |z| < 1\}$. The number of distinguishable microstates (volume) grows exponentially with radius $r$.

:::{prf:definition} Manifold Boundary and Interior
:label: def-manifold-boundary-and-interior

Let $\mathcal{Z}$ be the latent manifold with Poincare disk model. The **boundary** is the $(n-1)$-dimensional limit set:

$$
\partial\mathcal{Z} := \{z \in \mathbb{C}^n : |z| = 1\}.
$$
The **interior** (or bulk) is the open disk:

$$
\text{int}(\mathcal{Z}) := \{z \in \mathbb{C}^n : |z| < 1\}.
$$
These are standard differential geometry terms; the boundary is the ideal boundary at infinity in the hyperbolic metric.

:::
:::{prf:definition} Hyperbolic Volume Growth
:label: def-hyperbolic-volume-growth

With metric $G_{ij} = \frac{4\delta_{ij}}{(1-|z|^2)^2}$, the volume of a hyperbolic ball $B_r(0)$ grows exponentially:

$$
\mathrm{Vol}(B_r(0)) = 4\pi \sinh^2\!\left(\frac{r}{2}\right) \;\approx\; \pi e^r \quad \text{as } r \to \infty.
$$
Units: $[\mathrm{Vol}] = [z]^2$.

:::
:::{prf:definition} The Entropic Force
:label: def-the-entropic-force

The "Free Energy" of a state at radius $r$ is dominated by the entropic volume term $S(r) \sim 2 \tanh^{-1}(r)$. To maximize entropy (fill the capacity), the agent experiences a radial force:

$$
F_{\text{entropy}}(z) = \nabla_G S(z) = \frac{z}{\|z\|}
$$
In normalized hyperbolic coordinates, this yields a **constant radial drift**.

Units: $[F_{\text{entropy}}] = [z]/\tau$.

:::
:::{prf:proposition} Isotropic Radial Expansion
:label: prop-isotropic-radial-expansion

If acting alone (no policy steering), the entropic drift produces the isotropic expansion:

$$
r(\tau) = \tanh(\tau/2)
$$
This represents isotropic diffusion—expanding uniformly in all directions.

*Proof.* The overdamped equation $\dot{r} = (1-r^2)/2$ (from the Riemannian gradient of $U(z) = -2\operatorname{artanh}(|z|)$) integrates to $r(\tau) = \tanh(\tau/2 + \operatorname{artanh}(r_0))$. For $r_0 = 0$, we get $r(\tau) = \tanh(\tau/2)$. $\square$

:::
:::{prf:definition} Hyperbolic Information Potential
:label: def-hyperbolic-information-potential

The **information potential** $U: \mathbb{D} \to \mathbb{R}$ is the negative hyperbolic distance from the origin:

$$
U(z) := -d_{\mathbb{D}}(0, z) = -2 \operatorname{artanh}(|z|) = -\log\!\left(\frac{1+|z|}{1-|z|}\right).
$$
Units: $[U] = \mathrm{nat}$.

*Remark (Thermodynamic Interpretation).* At origin ($z=0$): $U = 0$ (maximum potential, maximum entropy). At boundary ($|z| \to 1$): $U \to -\infty$ (minimum potential, fully specified). The depth $-U(z)$ measures the **information content** of the state.

:::
:::{prf:proposition} Riemannian Gradient of $U$
:label: prop-riemannian-gradient-of

The gradient in the Poincare metric is:

$$
\nabla_G U(z) = G^{-1} \nabla U = -\frac{(1-|z|^2)}{2} z.
$$
The **entropic drift** (negative gradient) pushes radially outward:

$$
-\nabla_G U(z) = \frac{(1-|z|^2)}{2} z.
$$
*Remark (Connection to Section 7.11).* The Poincare coordinate $z$ relates to depth via $\rho = d_{\mathbb{D}}(0, z) = 2\operatorname{artanh}(|z|)$. Chart transitions are handled by the WFR jump process (Section 22.2).

**Cross-references:** Definition {prf:ref}`def-information-density-and-bulk-information-volume`, Theorem {prf:ref}`thm-capacity-constrained-metric-law`.

:::
(sec-policy-control-field)=
### 21.2 Policy Control Field

At the origin ($z=0$), the system has full rotational symmetry $SO(D)$. To generate specific content (or solve a task), this symmetry must be broken. The **policy** provides the initial direction via the control field $u_\pi$.

:::{prf:proposition} SO(D) Symmetry at Origin
:label: prop-so-d-symmetry-at-origin

At $z = 0$:
1. The metric is isotropic: $G(0) = 4I$
2. The entropic force vanishes: $F_{\text{entropy}}(0) = 0$
3. The system has full rotational symmetry $SO(D)$

*Cross-reference (Gauge Breaking):* This $SO(D)$ symmetry is the special case where the stabilizer subgroup $H_0 = \{e\}$ is trivial. In multi-agent settings, this symmetry is spontaneously broken via the Higgs mechanism (Theorem {prf:ref}`thm-higgs-mechanism`), yielding massive gauge bosons and effective agent masses.

:::
:::{prf:definition} The Control Field
:label: def-the-control-field

The Policy $\pi_\theta(a|z)$ outputs a **control field** $u_\pi(z)$ on the tangent bundle $T\mathbb{D}$:

$$
u_\pi(z) = G^{-1}(z) \cdot \mathbb{E}_{a \sim \pi_\theta}[a]
$$
This vector field represents the **Information Preference** of the agent (or the User).

Units: $[u_\pi] = [z]/\tau$.

*Remark (Context-Conditioning).* Section 23.6 generalizes this to **context-conditioned policies** $\pi(a|z,c)$ where the context $c \in \mathcal{C}$ unifies: RL action spaces, classification label spaces, and LLM prompt spaces. The control field becomes $u_\pi(z,c) = G^{-1}(z) \cdot \nabla_z \Phi_{\text{eff}}(z,K,c)$ where the effective potential depends on task context.

:::
:::{prf:definition} Control Field at Origin
:label: def-control-field-at-origin

At $\tau=0$, the total drift is:

$$
F_{\text{total}} = F_{\text{entropy}} + u_\pi(0)
$$
Since $F_{\text{entropy}}(0) = 0$ (isotropic), the initial trajectory is determined **entirely** by $u_\pi(0)$.

:::
:::{prf:theorem} Unified Control Interpretation
:label: thm-unified-control-interpretation

The control field $u_\pi$ admits three equivalent interpretations:

| **Mode**                     | **Control Field $u_\pi$**                          | **Interpretation**                         |
|------------------------------|----------------------------------------------------|--------------------------------------------|
| **RL**                       | $u_\pi = G^{-1} \nabla_z V_{\text{critic}}$        | Points toward high-value regions           |
| **Conditioned Generation**   | $u_\pi = G^{-1} \cdot \text{embed}(\text{prompt})$ | Clamped to user's prompt embedding         |
| **Unconditional (Dreaming)** | $u_\pi = 0$                                        | Pure thermal fluctuation selects direction |

*Proof.* In all cases, $u_\pi$ is a tangent vector at $z$. The RL case follows from the policy gradient theorem {cite}`sutton1999policy`; the generation case follows from treating the prompt as a target direction; the unconditional case reduces to pure Langevin dynamics where noise breaks symmetry. $\square$

:::
:::{prf:theorem} Pitchfork Bifurcation Structure {cite}`strogatz2015nonlinear`
:label: thm-pitchfork-bifurcation-structure

Near the origin, the combined dynamics exhibit a **supercritical pitchfork bifurcation**:

$$
\dot{r} = \mu r - r^3 + \sigma \xi
$$
where $r = |z|$, $\mu = 1$ (unstable fixed point), and $\sigma = \sqrt{2T_c}$ is the noise amplitude.

**Phase Transition:**
- **Symmetric phase** ($T_c$ large): Random walk near origin, symmetry preserved
- **Broken phase** ($T_c$ small): Deterministic flow to boundary along selected direction

*Proof sketch (Bifurcation derivation).* Near the origin, the Langevin dynamics (from Section 22.2) in radial coordinate $r = |z|$ becomes:

$$
dr = \left(\frac{1-r^2}{2} + u_\pi^r\right) d\tau + \sqrt{T_c(1-r^2)} dW_\tau
$$
where $u_\pi^r = u_\pi \cdot \hat{r}$ is the radial component of the control field. Taylor expanding near $r = 0$:

$$
dr \approx \left(\frac{1}{2} + u_\pi^r - \frac{r^2}{2}\right) d\tau + \sqrt{T_c}\, dW_\tau.
$$
For small control $u_\pi^r \ll 1$ and setting $\mu = 1/2 + u_\pi^r$, this matches the normal form $\dot{r} = \mu r - r^3/2 + \sigma\xi$.

**Critical temperature:** The effective potential $U_{\text{eff}}(r) = -\mu r^2/2 + r^4/8$ has minima at $r^* = \pm\sqrt{2\mu}$ for $\mu > 0$. The barrier height is $\Delta U = \mu^2/4$. Symmetry is preserved when thermal fluctuations overcome the barrier:

$$
T_c^* = \frac{\mu^2}{4} = \frac{1}{16}(1 + 2u_\pi^r)^2 \approx \frac{1}{16}.
$$
For $T_c > T_c^*$: symmetric phase; for $T_c < T_c^*$: broken phase with directional flow. $\square$

:::

::::{admonition} Physics Isomorphism: Spontaneous Symmetry Breaking
:class: note
:name: pi-symmetry-breaking

**In Physics:** Spontaneous symmetry breaking occurs when a system's ground state has lower symmetry than its Hamiltonian. The classic example is the Mexican hat potential $V(\phi) = -\mu^2|\phi|^2 + \lambda|\phi|^4$: for $\mu^2 > 0$, the $U(1)$-symmetric origin becomes unstable and the system selects a direction {cite}`goldstone1961field,weinberg1996qft`.

**In Implementation:** The radial dynamics exhibit supercritical pitchfork bifurcation (Theorem {prf:ref}`thm-pitchfork-bifurcation-structure`):

$$
\dot{r} = \mu r - r^3 + \sigma\xi
$$
where $\mu = 1/2$ and the $SO(D)$ symmetry at the origin is broken when the policy selects a direction.

**Correspondence Table:**
| Phase Transition Theory | Agent (Policy Emergence) |
|:------------------------|:-------------------------|
| Order parameter $\phi$ | Radial coordinate $r = \|z\|$ |
| Control parameter $\mu$ | $\mu = 1/2$ (supercritical) |
| Critical temperature $T_c^*$ | $T_c^* = 1/16$ (barrier height) |
| Symmetric phase ($\phi = 0$) | Semantic vacuum (origin) |
| Broken phase ($\phi \neq 0$) | Policy-selected direction |
| Goldstone modes | Angular fluctuations in $\theta$ |

**Significance:** Policy selection is not arbitrary—it is a geometric phase transition where the agent spontaneously breaks $SO(D)$ symmetry to select a generation direction.
::::

**Algorithm 21.2.6 (Control Field Computation).**

```python
import torch
from typing import Literal, Optional


def poincare_metric_inv(z: torch.Tensor) -> torch.Tensor:
    """Compute inverse Poincare metric G^{-1}(z) = (1 - |z|^2)^2 / 4."""
    r_sq = (z ** 2).sum(dim=-1, keepdim=True)
    one_minus_r_sq = torch.clamp(1.0 - r_sq, min=1e-8)
    return (one_minus_r_sq ** 2) / 4.0


def compute_control_field(
    z: torch.Tensor,                    # [B, D] current position (near origin)
    mode: Literal["rl", "generation", "dreaming"],
    prompt_embed: Optional[torch.Tensor] = None,  # [B, D] for generation mode
    grad_V: Optional[torch.Tensor] = None,        # [B, D] critic gradient for RL
    T_c: float = 1.0,                   # Temperature (for dreaming mode)
) -> torch.Tensor:
    """
    Compute the control field u_pi that selects initial direction.

    Breaks SO(D) symmetry at the origin, unifying RL, generation,
    and dreaming into a single operation.

    Cross-ref: Theorem {prf:ref}`thm-unified-control-interpretation`
    """
    B, D = z.shape
    G_inv = poincare_metric_inv(z)  # [B, 1]

    if mode == "rl":
        # RL: u_pi = G^{-1} * grad V (points toward high value)
        if grad_V is None:
            raise ValueError("grad_V required for RL mode")
        u_pi = G_inv * grad_V

    elif mode == "generation":
        # Generation: u_pi = G^{-1} * prompt_embed
        if prompt_embed is None:
            raise ValueError("prompt_embed required for generation mode")
        # Normalize prompt to unit vector, then scale by metric
        prompt_norm = prompt_embed / (torch.norm(prompt_embed, dim=-1, keepdim=True) + 1e-8)
        u_pi = G_inv * prompt_norm

    elif mode == "dreaming":
        # Dreaming: pure thermal fluctuation (no deterministic kick)
        # The noise in the Langevin dynamics will break symmetry
        u_pi = torch.zeros(B, D, device=z.device, dtype=z.dtype)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return u_pi
```

**Cross-references:** Section 2.7 (HJB Correspondence), Section 14.2 (MaxEnt control equivalence).

:::

::::{note} Connection to RL #24: Diffusion Policies as Degenerate Radial Generation
**The General Law (Fragile Agent):**
Data generation is **radial expansion** from the vacuum (origin) to the boundary:

$$
F_{\text{total}} = \underbrace{F_{\text{entropy}}}_{\text{Hyperbolic drift}} + \underbrace{u_\pi}_{\text{Policy kick}}
$$
where $F_{\text{entropy}} = \nabla_G S(z)$ is entropic drift from hyperbolic volume growth and $u_\pi = G^{-1} \nabla_z V$ is the policy control field that breaks $SO(D)$ symmetry.

**The Degenerate Limit:**
Replace hyperbolic geometry with Euclidean ($G \to I$). Reverse the direction (boundary to origin). Remove value-based steering.

**The Special Case (Standard RL):**

$$
dz_t = s_\theta(z_t, t)\, dt + \sigma(t)\, dW_t, \quad z_T \sim \mathcal{N}(0, I)
$$
This recovers **Diffusion Models** {cite}`ho2020ddpm` and **Diffusion Policies** for robotic control.

**What the generalization offers:**
- **Hyperbolic structure**: Exponential volume growth provides natural hierarchy (Theorem **Thm: Hyperbolic Volume Growth**)
- **Forward generation**: Origin to boundary matches RL's forward dynamics semantics
- **Policy unification**: RL control and conditional generation share the same drift term $u_\pi$
- **Symmetry breaking**: Policy kicks at origin select generation mode (Theorem {prf:ref}`thm-pitchfork-bifurcation-structure`)
::::

(sec-bulk-boundary-independence)=
### 21.3 Bulk-Boundary Independence

We strictly enforce the separation of **Planning** (interior $\text{int}(\mathcal{Z})$) and **Observation** (boundary $\partial\mathcal{Z}$). This is formalized as a partition condition.

*Remark (Motor Extension).* The independence constraint applies equally to the **motor/action boundary**: motor texture $z_{\text{tex,motor}}$ (tremor, fine motor noise) is sampled at the output interface and does not participate in planning. Section 23.3 formalizes the motor texture distribution $z_{\text{tex,motor}} \sim \mathcal{N}(0, \sigma_{\text{motor}}^2 G^{-1}(z))$ with the same conformal scaling as visual texture. Theorem {prf:ref}`ax-motor-texture-firewall` establishes the duality: $\Sigma_{\text{motor}} = \omega \cdot \Sigma_{\text{visual}} \cdot \omega^{-1}$.

:::{prf:axiom} Bulk-Boundary Decoupling
:label: ax-bulk-boundary-decoupling

The state decomposition $Z = (K, z_n, z_{\text{tex}})$ satisfies a **partition condition**:

1. **Interior (Planning Domain):** The trajectory $z(\tau)$ evolves strictly on the manifold $\mathcal{Z} = \mathcal{K} \times \mathcal{Z}_n$. It contains no texture component. Planning depends only on geometry and topology:

$$
\dot{z} = f(z, u_\pi) \quad (\text{No } z_{\text{tex}} \text{ dependence})
$$
2. **Boundary Interface:** Texture $z_{\text{tex}}$ is a stochastic component that exists **only** at the interface where the internal state meets the external observation:

$$
z_{\text{tex}} \sim \mathcal{N}(0, \Sigma(z_{\text{final}}))
$$
Formally, the partition condition is:

$$
\frac{\partial}{\partial z_{\text{tex}}} \left[ \dot{z}^k, \lambda_{\text{jump}}, u_\pi \right] = 0 \quad \forall \tau \in [0, \tau_{\text{stop}})
$$
:::
:::{prf:definition} Boundary Texture Distribution
:label: def-boundary-texture-distribution

At the terminal position $z_{\text{final}}$, texture is sampled from a **geometry-dependent** Gaussian:

$$
z_{\text{tex}} \sim \mathcal{N}\big(0,\, \Sigma(z_{\text{final}})\big),
$$
where the covariance matrix is:

$$
\Sigma(z) = \sigma_{\text{tex}}^2 \cdot G^{-1}(z) = \sigma_{\text{tex}}^2 \cdot \frac{(1-|z|^2)^2}{4} I.
$$
Units: $[\Sigma] = [z_{\text{tex}}]^2$.

:::
:::{prf:proposition} Conformal Texture Scaling
:label: prop-conformal-texture-scaling

The texture variance scales with the inverse metric:

| **Region** | **$\lvert z\rvert$** | **$\Sigma(z)$**                            | **Interpretation**        |
|------------|----------------------|--------------------------------------------|---------------------------|
| Origin     | $\approx 0$          | $\sigma_{\text{tex}}^2/4 \cdot I$          | Moderate texture (coarse) |
| Mid-disk   | $\approx 0.5$        | $\sigma_{\text{tex}}^2 \cdot 9/64 \cdot I$ | Reduced texture           |
| Boundary   | $\to 1$              | $\to 0$                                    | Deterministic texture     |

*Remark (Conformal suppression).* Near the boundary (high resolution/specificity), the metric $G$ diverges, so $G^{-1} \to 0$ and texture fluctuations are suppressed.

:::
:::{prf:definition} Boundary Decoder
:label: def-boundary-decoder

The Decoder $\mathcal{D}$ is the **only** component that sees texture. It performs the **boundary synthesis**:

$$
x = \mathcal{D}(z_{\text{final}}, z_{\text{tex}})
$$
where:
- $z_{\text{final}} = (e_K, z_n)$: Determines the shape, physics, and causal structure
- $z_{\text{tex}}$: "Paints" the high-frequency details onto that structure

:::
:::{prf:proposition} Epistemic Barrier
:label: prop-epistemic-barrier

The partition condition enforces **BarrierEpi** (Epistemic Limit): The agent does not waste capacity predicting the noise—it only predicts the *statistics* of the noise ($\Sigma$).

:::
:::{prf:definition} Stopping Criterion
:label: def-stopping-criterion

The flow terminates when the radial coordinate exceeds a cutoff:

$$
\tau_{\text{stop}} := \inf\{\tau \ge 0 : |z(\tau)| \ge R_{\text{cutoff}}\}
$$
This is equivalent to the information stopping criterion $I_{\text{bulk}}(z) \ge C_\partial$ (Theorem {prf:ref}`thm-capacity-constrained-metric-law`).

**Algorithm 21.3.7 (Boundary Texture Sampling).**

```python
import torch

def sample_boundary_texture(
    z_final: torch.Tensor,        # [B, D] final semantic position
    texture_dim: int,             # Dimension of texture space
    sigma_tex: float = 1.0,       # Base texture std dev
) -> torch.Tensor:
    """
    Sample texture with geometry-dependent variance.

    Implements Definition 21.3.2:
        z_tex ~ N(0, Sigma(z_final))
        Sigma(z) = sigma_tex^2 * G^{-1}(z)

    The partition condition (Axiom 21.3.1) ensures this is called
    ONLY at terminal time, not during interior dynamics.

    Cross-ref: Proposition 21.3.3 (Conformal Scaling)
    """
    B = z_final.shape[0]

    # Compute G^{-1}(z) = (1 - |z|^2)^2 / 4
    r_sq = (z_final ** 2).sum(dim=-1, keepdim=True)  # [B, 1]
    one_minus_r_sq = torch.clamp(1.0 - r_sq, min=1e-6)
    G_inv_scale = (one_minus_r_sq ** 2) / 4.0  # [B, 1]

    # Texture std = sigma_tex * sqrt(G^{-1})
    texture_std = sigma_tex * torch.sqrt(G_inv_scale)  # [B, 1]

    # Sample isotropic Gaussian, then scale
    z_tex = torch.randn(B, texture_dim, device=z_final.device)
    z_tex = z_tex * texture_std  # broadcast scaling

    return z_tex
```

:::
(sec-summary-and-diagnostic-node)=
### 21.4 Summary and Diagnostic Node

**Summary of Radial Generation:**

| **Aspect**          | **Formula**                                   | **Units**            | **Reference**                                    |
|---------------------|-----------------------------------------------|----------------------|--------------------------------------------------|
| Entropic Drift      | $F_{\text{entropy}} = z/\lVert z\rVert$       | $[z]/\tau$           | Def {prf:ref}`def-hyperbolic-volume-growth`      |
| Radial Expansion    | $r(\tau) = \tanh(\tau/2)$                     | dimensionless        | Prop {prf:ref}`def-the-entropic-force`           |
| Control Field       | $u_\pi = G^{-1} \mathbb{E}[a]$                | $[z]/\tau$           | Def {prf:ref}`def-the-control-field`             |
| Partition Condition | $\partial_{z_{\text{tex}}} \dot{z} = 0$       | -                    | Axiom {prf:ref}`ax-bulk-boundary-decoupling`     |
| Texture Covariance  | $\Sigma(z) = \sigma_{\text{tex}}^2 G^{-1}(z)$ | $[z_{\text{tex}}]^2$ | Def {prf:ref}`def-boundary-texture-distribution` |
| Stopping            | $\lvert z\rvert \ge R_{\text{cutoff}}$        | dimensionless        | Def {prf:ref}`def-stopping-criterion`            |

(node-25)=
**Node 25: RadialGenCheck (HoloGenCheck)**

| **#**  | **Name**           | **Component** | **Type**                | **Interpretation**       | **Proxy**                                                         | **Cost** |
|--------|--------------------|---------------|-------------------------|--------------------------|-------------------------------------------------------------------|----------|
| **25** | **RadialGenCheck** | **Generator** | **Generation Validity** | Did flow reach boundary? | $\mathbb{I}(\lvert z_{\text{final}}\rvert \ge R_{\text{cutoff}})$ | $O(B)$   |

**Trigger conditions:**
- Low RadialGenCheck: Generation terminated too early (insufficient specificity).
- Remedy: Increase $\tau_{\text{max}}$ or decrease $R_{\text{cutoff}}$.

**Cross-references:** Section 2.2b (VQ-VAE texture channel), Section 7.10 (TopologicalDecoder), Section 18 (Capacity constraints).



(sec-the-equations-of-motion-geodesic-jump-diffusion)=
