## 22. The Equations of Motion: Geodesic Jump-Diffusion

{cite}`oksendal2003sde,risken1996fokkerplanck`

:::{admonition} Researcher Bridge: Continuous-Time Actor-Critic
:class: info
:name: rb-continuous-actor-critic
The equations of motion are the continuous-time limit of policy updates with stochastic exploration noise. Think of it as a Langevinized actor-critic where the metric defines the preconditioner.
:::

We derive the rigorous equation of motion (EoM) for the agent. This equation unifies the WFR geometry (Section 20), the Metric Law (Section 18), and the Policy-driven Expansion (Section 21).

(sec-the-stochastic-action-principle)=
### 22.1 The Stochastic Action Principle (Mass = Metric)

The classical Lagrangian approach extends to stochastic systems via the **Onsager-Machlup functional**, which assigns a probability to paths based on their "action."

:::{prf:definition} Mass Tensor
:label: def-mass-tensor

We define the **inertial mass tensor** $\mathbf{M}(z)$ as the capacity-constrained metric:

$$
\mathbf{M}(z) := G(z).
$$
This definition has the following operational consequences:
- **High curvature regions** (large $G$) have larger effective mass, yielding smaller velocity updates per unit force
- **Low curvature regions** (small $G$) have smaller effective mass, yielding larger velocity updates per unit force

Units: $[\mathbf{M}_{ij}] = [z]^{-2}$ (same as metric).

*Remark (Risk-Metric Coupling).* Combined with the Metric Law (Theorem {prf:ref}`thm-capacity-constrained-metric-law`), this yields a causal chain:

$$
\text{High risk } T_{ij} \;\Rightarrow\; \text{Large } G_{ij} \;\Rightarrow\; \text{Large } \mathbf{M}_{ij} \;\Rightarrow\; \text{Reduced step size}
$$
The metric-weighted step size decreases in high-curvature (high-risk) regions without explicit penalty terms.

:::
:::{prf:definition} Extended Onsager-Machlup Action
:label: def-extended-onsager-machlup-action

Let $(\mathcal{Z}, G)$ be the latent Riemannian manifold with the capacity-constrained metric (Section 18). For a path $z: [0, T] \to \mathcal{Z}$, the extended Onsager-Machlup action is:

$$
S_{\mathrm{OM}}[z] = \int_0^T \left( \frac{1}{2}\mathbf{M}(z)\|\dot{z}\|^2 + \Phi_{\text{eff}}(z) + \frac{T_c}{12}\,R(z) + T_c \cdot H_{\pi}(z) \right) ds,
$$
where:
- $\mathbf{M}(z)\|\dot{z}\|^2 = G_{ij}(z)\,\dot{z}^i\,\dot{z}^j$ is the kinetic energy (mass = metric)
- $\Phi_{\text{eff}}(z)$ is the effective potential (Definition {prf:ref}`def-effective-potential`)
- $R(z)$ is the scalar curvature of the metric $G$
- $H_{\pi}(z) = -\mathbb{E}_{a \sim \pi}[\log \pi(a|z)]$ is the policy entropy
- $T_c > 0$ is the cognitive temperature (cf. Section 21.2)

Units: $[S_{\mathrm{OM}}] = \mathrm{nat}$.

*Remark (Curvature Correction).* The term $\frac{T_c}{12}R(z)$ is a stochastic correction that accounts for the path-measure distortion on curved spaces. In flat space ($R = 0$), this term vanishes. The entropy term $T_c H_{\pi}$ ensures the agent prefers stochastic policies in uncertain regions.

:::

::::{admonition} Physics Isomorphism: Onsager-Machlup Action
:class: note
:name: pi-onsager-machlup

**In Physics:** The Onsager-Machlup functional assigns probability to paths in stochastic thermodynamics: $P[\gamma] \propto \exp(-S_{OM}[\gamma]/k_B T)$ where $S_{OM}$ includes kinetic and potential terms plus a curvature correction {cite}`onsager1953fluctuations`.

**In Implementation:** The extended Onsager-Machlup action (Definition {prf:ref}`def-extended-onsager-machlup-action`):

$$
S_{\text{OM}}[z] = \int_0^T \left(\frac{1}{2}G_{ij}\dot{z}^i\dot{z}^j + \Phi_{\text{eff}} + \frac{T_c}{12}R + T_c H_\pi\right)ds
$$
**Correspondence Table:**

| Statistical Mechanics | Agent (Path Integral) |
|:----------------------|:----------------------|
| Temperature $k_B T$ | Cognitive temperature $T_c$ |
| Kinetic energy $\frac{1}{2}m\lvert\dot{x}\rvert^2$ | $\frac{1}{2}G_{ij}\dot{z}^i\dot{z}^j$ (mass = metric) |
| Potential $U(x)$ | Effective potential $\Phi_{\text{eff}}$ |
| Curvature correction $\frac{k_BT}{12}R$ | $\frac{T_c}{12}R$ |
| Boltzmann weight $e^{-S/k_BT}$ | Path probability $e^{-S_{\text{OM}}/T_c}$ |
::::

:::{prf:proposition} Mass Scaling Near Boundary
:label: prop-mass-scaling-near-boundary

For the Poincare disk, the mass tensor scales as:

$$
\mathbf{M}(z) = \frac{4}{(1-|z|^2)^2} I_d \quad \xrightarrow{|z| \to 1} \quad +\infty.
$$
The metric diverges as $|z| \to 1$, which bounds all finite-action trajectories to the interior of the disk.

*Proof.* Direct evaluation of the Poincare metric. The factor $(1-|z|^2)^{-2}$ diverges as $|z| \to 1$. $\square$

:::
:::{prf:proposition} Most Probable Path
:label: prop-most-probable-path

For the controlled diffusion

$$
dz^k = b^k(z)\,ds + \sqrt{2T_c}\,\sigma^{kj}(z)\,dW^j_s,
$$
where $\sigma \sigma^T = G^{-1}$, the most probable path connecting $z(0) = z_0$ and $z(T) = z_1$ minimizes the Onsager-Machlup action $S_{\mathrm{OM}}[z]$ subject to the boundary conditions.

*Proof sketch.* This follows from the Girsanov theorem and the Cameron-Martin formula adapted to Riemannian manifolds. See {cite}`ikeda1989stochastic` Chapter V or Appendix A.4 for details. $\square$

:::
(sec-the-coupled-jump-diffusion-sde)=
### 22.2 The Coupled Jump-Diffusion SDE

The agent's state is not merely a point $z$ but a **particle with mass** $(z, m)$, where $m$ is the importance weight (belief probability). The dynamics couple continuous transport with discrete topological jumps.

*Cross-reference (WFR Boundary Conditions).* The SDE below assumes **Waking mode** boundary conditions (Definition {prf:ref}`def-waking-boundary-clamping`): Dirichlet on sensors (clamping observed position), Neumann on motors (clamping output flux). In **Dreaming mode** (Definition {prf:ref}`def-dreaming-reflective-boundary`), both boundaries become reflective and the flow recirculates internally without external grounding. See Section 23.5 for the mode-switching table and the thermodynamic interpretation (Section 23.4).

:::{prf:definition} Bulk Drift - Continuous Flow (Lorentz-Langevin Equation)
:label: def-bulk-drift-continuous-flow

The position coordinates $z^k$ evolve according to the **Lorentz-Langevin SDE**:

$$
dz^k = \underbrace{\left( -G^{kj}\partial_j \Phi + u_\pi^k \right)}_{\text{gradient + control}} ds \;+\; \underbrace{\beta_{\text{curl}}\, G^{km} \mathcal{F}_{mj} \dot{z}^j\,ds}_{\text{Lorentz force}} \;-\; \underbrace{\Gamma^k_{ij}\dot{z}^i \dot{z}^j\,ds}_{\text{geodesic correction}} \;+\; \underbrace{\sqrt{2T_c}\,(G^{-1/2})^{kj}\,dW^j_s}_{\text{thermal noise}},
$$
where:
- $\Phi$ is the **scalar potential** from the Hodge decomposition (Theorem {prf:ref}`thm-hodge-decomposition`)
- $\mathcal{F}_{ij} = \partial_i \mathcal{R}_j - \partial_j \mathcal{R}_i$ is the **Value Curl** tensor (Definition {prf:ref}`def-value-curl`)
- $\beta_{\text{curl}} \ge 0$ is the **curl coupling strength** (dimensionless)
- $u_\pi^k$ is the **control field** from the policy (Definition {prf:ref}`prop-so-d-symmetry-at-origin`)
- $\Gamma^k_{ij}$ are the **Christoffel symbols** of the Levi-Civita connection (Section 2.5.1, Theorem {prf:ref}`thm-capacity-constrained-metric-law`)
- $G^{-1/2}$ is the matrix square root of the inverse metric
- $W_s$ is a standard Wiener process

*Units:* $[dz] = [z]$, $[\Phi] = \mathrm{nat}$, $[\mathcal{F}_{ij}] = \mathrm{nat}/[z]^2$, $[\Gamma^k_{ij}] = [z]^{-1}$.

*Remark (Four-Force Decomposition).* The drift decomposes into:
1. **Gradient force**: $-G^{-1}\nabla\Phi$ — force proportional to scalar potential gradient
2. **Lorentz force**: $\beta_{\text{curl}} G^{-1}\mathcal{F}\dot{z}$ — velocity-dependent force from Value Curl
3. **Control field**: $u_\pi$ — policy-induced drift (Section 21.2)
4. **Geodesic correction**: $-\Gamma(\dot{z},\dot{z})$ — parallel transport on curved space

**Conservative Limit:** When $\mathcal{F} = 0$ (Definition {prf:ref}`def-conservative-reward-field`), the Lorentz term vanishes and we recover the standard geodesic SDE.

**Non-Conservative Dynamics:** When $\mathcal{F} \neq 0$, the Lorentz force induces rotational dynamics. Trajectories may converge to limit cycles rather than fixed points (Theorem {prf:ref}`thm-ness-existence`).

*Remark (Connection Specification).* The Christoffel symbols $\Gamma^k_{ij}$ are explicitly those of the **Levi-Civita connection** induced by the capacity-constrained metric $G$ from Theorem {prf:ref}`thm-capacity-constrained-metric-law`. This ensures metric compatibility ($\nabla G = 0$) and torsion-freeness.

:::
:::{prf:proposition} a (Explicit Christoffel Symbols for Poincare Disk)
:label: prop-a-explicit-christoffel-symbols-for-poincar-disk

For the Poincare disk model with metric $G_{ij} = \frac{4\delta_{ij}}{(1-|z|^2)^2}$, the Christoffel symbols in Cartesian coordinates are:

$$
\Gamma^k_{ij}(z) = \frac{2}{1-|z|^2}\left(\delta^k_i z_j + \delta^k_j z_i - \delta_{ij} z^k\right).
$$
The geodesic correction term $\Gamma^k_{ij}\dot{z}^i\dot{z}^j$ contracts to:

$$
\Gamma^k_{ij}\dot{z}^i\dot{z}^j = \frac{4(z \cdot \dot{z})}{1-|z|^2}\dot{z}^k - \frac{2|\dot{z}|^2}{1-|z|^2}z^k.
$$
*Proof.* Direct computation from $\Gamma^k_{ij} = \frac{1}{2}G^{k\ell}(\partial_i G_{j\ell} + \partial_j G_{i\ell} - \partial_\ell G_{ij})$ using $\partial_m[(1-|z|^2)^{-2}] = 4z_m(1-|z|^2)^{-3}$. $\square$

*Geometric interpretation:* The first term $(z \cdot \dot{z})\dot{z}$ accelerates motion radially when moving outward; the second term $|\dot{z}|^2 z$ provides centripetal correction. Together they ensure geodesics are circular arcs perpendicular to the boundary.

:::
:::{prf:definition} Mass Evolution - Jump Process
:label: def-mass-evolution-jump-process

The importance weight $m(s)$ evolves according to a coupled jump-diffusion:

$$
dm = m \cdot r(z, a)\,ds + m \cdot (\eta - 1)\,dN_s,
$$
where:
- $r(z, a)$ is the **reaction rate** from the WFR dynamics (Section 20.2)
- $N_s$ is a Poisson process with intensity $\lambda_{\text{jump}}(z)$
- $\eta$ is the multiplicative jump factor (typically $\eta > 1$ for jumps to higher-value charts)

*Interpretation:* Between jumps, mass evolves smoothly via the reaction term $r$. At jump times, the mass is rescaled by factor $\eta$, and the position is teleported via the chart transition operator $L_{i \to j}$.

:::
:::{prf:proposition} Jump Intensity from Value Discontinuity
:label: prop-jump-intensity-from-value-discontinuity

The jump intensity $\lambda_{\text{jump}}(z)$ is determined by the value difference across chart boundaries:

$$
\lambda_{\text{jump}}(z) = \lambda_0 \cdot \exp\left(\beta \cdot \left( V_{\text{target}}(L(z)) - V_{\text{source}}(z) - c_{\text{transport}} \right) \right),
$$
where:
- $\lambda_0 > 0$ is a base jump rate
- $\beta > 0$ is the inverse temperature (sharpness)
- $V_{\text{target}}$ and $V_{\text{source}}$ are the value functions on the target and source charts
- $L: \mathcal{Z}_{\text{source}} \to \mathcal{Z}_{\text{target}}$ is the chart transition operator
- $c_{\text{transport}} \ge 0$ is the transport cost (WFR term)

*Remark (SMC Interpretation).* The mass $m(s)$ is precisely the **importance weight** in Sequential Monte Carlo (SMC) / particle filtering. The agent is a single-particle realization of the WFR flow from Section 20. Multiple particles can be used for ensemble-based generation.

**Cross-references:** Section 20.2 (WFR action), Section 20.6 (WFR world model), Section 11 (Filtering and projection).

:::
(sec-the-unified-effective-potential)=
### 22.3 The Unified Effective Potential

The effective potential unifies three terms: the hyperbolic information potential $U$ from holographic generation (Section 21.1), the learned value function $V$ from control (Section 2.7), and the risk-stress contribution $\Psi_{risk}$ from the stress-energy tensor (Section 18).

:::{prf:definition} Effective Potential
:label: def-effective-potential

The unified effective potential is:

$$
\Phi_{\text{eff}}(z, K) = \alpha\, U(z) + (1 - \alpha)\, V_{\text{critic}}(z, K) + \gamma_{risk}\, \Psi_{\text{risk}}(z),
$$
where:
- $U(z) = -d_{\mathbb{D}}(0, z) = -2\operatorname{artanh}(|z|)$ is the **hyperbolic information potential** (Definition {prf:ref}`def-hyperbolic-volume-growth`)
- $V_{\text{critic}}(z, K)$ is the **learned value/critic function** on chart $K$ (Section 2.7)
- $\Psi_{\text{risk}}(z) = \frac{1}{2}\operatorname{tr}(T_{ij} G^{ij})$ is the **risk-stress contribution** (Theorem {prf:ref}`thm-capacity-constrained-metric-law`)
- $\alpha \in [0, 1]$ is the generation-vs-control hyperparameter
- $\gamma_{risk} \ge 0$ is the risk aversion coefficient

Units: $[\Phi_{\text{eff}}] = \mathrm{nat}$.

:::
:::{prf:proposition} Mode Interpretation
:label: prop-mode-interpretation

The parameter $\alpha$ interpolates between pure generation and pure control:

| Regime              | $\alpha$ Value      | Behavior                                                       |
|---------------------|---------------------|----------------------------------------------------------------|
| **Pure Generation** | $\alpha = 1$        | Flow follows $-\nabla_G U$ (holographic expansion, Section 21) |
| **Pure Control**    | $\alpha = 0$        | Flow follows $-\nabla_G V_{\text{critic}}$ (policy gradient)   |
| **Hybrid**          | $\alpha \in (0, 1)$ | Balanced generation and control                                |

*Remark (Risk Modulation).* The $\gamma_{risk}$ term provides an additional penalty in high-stress regions (large $T_{ij}$), which further discourages risky trajectories beyond the geometric slowdown from Mass=Metric.

:::
:::{prf:corollary} Gradient Decomposition
:label: cor-gradient-decomposition

The gradient of the effective potential decomposes as:

$$
\nabla_G \Phi_{\text{eff}} = \alpha\, \nabla_G U + (1 - \alpha)\, \nabla_G V_{\text{critic}} + \gamma_{risk}\, \nabla_G \Psi_{\text{risk}}.
$$
For the Poincare disk model, the first term simplifies to:

$$
\nabla_G U = -\frac{(1-|z|^2)}{2}\, \hat{z}, \qquad \hat{z} = \frac{z}{|z|}.
$$
**Cross-references:** Definition {prf:ref}`def-hyperbolic-volume-growth`, Section 2.7 (Critic $V$), Section 14.2 (MaxEnt control), Theorem {prf:ref}`thm-capacity-constrained-metric-law`.

*Forward reference (Scalar Field Interpretation).* Section 24 provides the complete field-theoretic interpretation of $V_{\text{critic}}$: the Critic solves the **Screened Poisson Equation** (Theorem {prf:ref}`thm-the-hjb-helmholtz-correspondence`) with rewards as boundary charges (Definition {prf:ref}`def-the-reward-flux`), the Value represents **Gibbs Free Energy** (Axiom {prf:ref}`ax-the-boltzmann-value-law`), and the Value Hessian induces a **Conformal Coupling** to the metric (Definition {prf:ref}`def-value-metric-conformal-coupling`).

:::

:::{prf:definition} Cognitive Temperature
:label: def-cognitive-temperature

The **cognitive temperature** $T_c > 0$ is the exploration-exploitation tradeoff parameter that controls:

1. **Diffusion magnitude:** The thermal noise term in the geodesic SDE scales as $\sqrt{2T_c}\,dW$
2. **Boltzmann policy:** The softmax temperature in $\pi(a|z) \propto \exp(Q(z,a)/T_c)$
3. **Free energy tradeoff:** The entropy-energy balance $\Phi = E - T_c S$

*Units:* nat (dimensionless in natural units where $k_B = 1$).

*Correspondence:* $T_c$ is the agent-theoretic analogue of thermodynamic temperature $k_B T$ in statistical mechanics.
:::

(sec-the-geodesic-baoab-integrator)=
### 22.4 The Geodesic Boris-BAOAB Integrator

We provide the numerical integrator for the controlled geodesic SDE (Definition {prf:ref}`def-bulk-drift-continuous-flow`). The **Boris-BAOAB** scheme extends the standard BAOAB {cite}`leimkuhler2016computation` to handle the velocity-dependent Lorentz force from non-conservative reward fields.

:::{prf:definition} Boris-BAOAB Splitting
:label: def-baoab-splitting

The Boris-BAOAB integrator splits the Lorentz-Langevin dynamics into five substeps per time step $h$:

1. **B** (half kick + Boris rotation):
   - Half-kick from gradient: $p^- \leftarrow p - \frac{h}{2}\nabla\Phi(z)$
   - Boris rotation (if $\mathcal{F} \neq 0$):
     - $t \leftarrow \frac{h}{2}\beta_{\text{curl}} G^{-1}\mathcal{F}$ (rotation vector)
     - $p' \leftarrow p^- + p^- \times t$
     - $s \leftarrow \frac{2t}{1 + |t|^2}$
     - $p^+ \leftarrow p^- + p' \times s$
   - Half-kick from gradient: $p \leftarrow p^+ - \frac{h}{2}\nabla\Phi(z)$

2. **A** (half drift): $z \leftarrow \operatorname{Exp}_z\left(\frac{h}{2} G^{-1}(z)\, p\right)$

3. **O** (thermostat): $p \leftarrow c_1 p + c_2\, G^{1/2}(z)\, \xi$, where $\xi \sim \mathcal{N}(0, I)$

4. **A** (half drift): $z \leftarrow \operatorname{Exp}_z\left(\frac{h}{2} G^{-1}(z)\, p\right)$

5. **B** (half kick + Boris rotation): Same as step 1

where $c_1 = e^{-\gamma h}$ and $c_2 = \sqrt{(1 - c_1^2) T_c}$.

**Conservative Limit:** When $\mathcal{F} = 0$, the Boris rotation is identity and we recover standard BAOAB.

*Remark (Boris Rotation).* The Boris algorithm is a volume-preserving integrator for magnetic-like forces. It rotates the momentum around the local Value Curl axis, preserving the norm $|p|$ while changing direction. This ensures the Lorentz force does no net work, consistent with physics.

*Remark (O-step).* The O-step implements the **Ornstein-Uhlenbeck thermostat**, which exactly preserves the Maxwell-Boltzmann momentum distribution $p \sim \mathcal{N}(0, T_c G)$.

**Algorithm 22.4.2 (Full Geodesic BAOAB with Jump Step).**

```python
import torch
import math
from dataclasses import dataclass
from typing import Tuple, Optional, Callable


@dataclass
class GeodesicState:
    """State of the geodesic integrator."""
    z: torch.Tensor          # [B, d] latent position
    p: torch.Tensor          # [B, d] momentum (covariant)
    K: torch.Tensor          # [B] chart index (integer)
    m: torch.Tensor          # [B] importance weight (mass)
    s: float                 # computation time


def poincare_metric(z: torch.Tensor) -> torch.Tensor:
    """
    Poincare disk metric tensor G_{ij}(z).

    G_{ij} = 4 delta_{ij} / (1 - |z|^2)^2

    Returns: [B, d, d] metric tensor
    """
    B, d = z.shape
    r_sq = (z ** 2).sum(dim=-1, keepdim=True)  # [B, 1]
    conformal_factor = 4.0 / (1.0 - r_sq + 1e-8) ** 2  # [B, 1]
    return conformal_factor.unsqueeze(-1) * torch.eye(d, device=z.device).expand(B, d, d)


def poincare_metric_inv(z: torch.Tensor) -> torch.Tensor:
    """
    Inverse Poincare metric G^{ij}(z).

    G^{ij} = (1 - |z|^2)^2 / 4 * delta^{ij}
    """
    B, d = z.shape
    r_sq = (z ** 2).sum(dim=-1, keepdim=True)
    inv_conformal = (1.0 - r_sq + 1e-8) ** 2 / 4.0
    return inv_conformal.unsqueeze(-1) * torch.eye(d, device=z.device).expand(B, d, d)


def mobius_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Möbius addition in the Poincare disk: a ⊕ b.

    (a + b) / (1 + <a, b>)  [simplified for small b]

    Full formula:
    a ⊕ b = ((1 + 2<a,b> + |b|^2) a + (1 - |a|^2) b) / (1 + 2<a,b> + |a|^2|b|^2)
    """
    a_sq = (a ** 2).sum(dim=-1, keepdim=True)
    b_sq = (b ** 2).sum(dim=-1, keepdim=True)
    a_dot_b = (a * b).sum(dim=-1, keepdim=True)

    num = (1 + 2*a_dot_b + b_sq) * a + (1 - a_sq) * b
    denom = 1 + 2*a_dot_b + a_sq * b_sq + 1e-8

    return num / denom


def poincare_exp_map(z: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Exponential map on Poincare disk: Exp_z(v).

    Exp_z(v) = φ_{-z}(tanh(||v||_z / 2) * v / ||v||)

    where φ_{-z} is Möbius translation and ||v||_z is the hyperbolic norm.
    """
    # Compute hyperbolic norm of v at z
    r_sq = (z ** 2).sum(dim=-1, keepdim=True)
    lambda_z = 2.0 / (1.0 - r_sq + 1e-8)  # conformal factor
    v_norm = torch.sqrt((v ** 2).sum(dim=-1, keepdim=True) + 1e-8) * lambda_z

    # Direction (normalized in Euclidean sense)
    v_dir = v / (torch.sqrt((v ** 2).sum(dim=-1, keepdim=True)) + 1e-8)

    # Magnitude in disk: tanh(||v||_z / 2)
    magnitude = torch.tanh(v_norm / 2.0)

    # Point at origin in direction v_dir with magnitude
    w = magnitude * v_dir

    # Translate by -z (Möbius addition)
    return mobius_add(z, w)


def christoffel_contraction(z: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Compute Γ^k_{ij} v^i v^j for Poincare disk.

    For G = 4/(1-|z|²)² I, the Christoffel symbols contract to:
    Γ(v,v)^k = 2/(1-|z|²) * (z^k |v|² + 2(z·v)v^k) - 4(z·v)²z^k/(1-|z|²)²
    """
    r_sq = (z ** 2).sum(dim=-1, keepdim=True)
    v_sq = (v ** 2).sum(dim=-1, keepdim=True)
    z_dot_v = (z * v).sum(dim=-1, keepdim=True)
    one_minus_r_sq = 1.0 - r_sq + 1e-8

    term1 = (2.0 / one_minus_r_sq) * z * v_sq
    term2 = (4.0 / one_minus_r_sq) * z_dot_v * v
    term3 = -(4.0 / one_minus_r_sq**2) * z_dot_v**2 * z

    return term1 + term2 + term3


def geodesic_baoab_step(
    state: GeodesicState,
    grad_Phi: torch.Tensor,           # [B, d] gradient of effective potential
    u_pi: torch.Tensor,               # [B, d] control field from policy
    T_c: float,                       # cognitive temperature
    gamma: float,                     # friction coefficient
    h: float,                         # time step
    jump_rate_fn: Optional[Callable] = None,  # λ(z, K) -> [B]
    chart_transition_fn: Optional[Callable] = None,  # L(z, K_src, K_tgt) -> z'
    value_fn: Optional[Callable] = None,  # V(z, K) -> [B]
) -> GeodesicState:
    """
    Full Geodesic BAOAB integrator with Poisson jump process.

    Implements Algorithm 22.4.2:
    1. B-step: half kick from potential + control
    2. A-step: half drift via exponential map
    3. O-step: Ornstein-Uhlenbeck thermostat
    4. A-step: half drift
    5. B-step: half kick
    6. Jump-step: Poisson process for chart transitions

    Cross-references:
        - Definition 22.2.1 (Bulk Drift SDE)
        - Definition 22.2.2 (Jump Process)
        - Section 2.5.1 (Christoffel symbols)
    """
    z, p, K, m = state.z, state.p, state.K, state.m
    B, d = z.shape
    device = z.device

    # BAOAB coefficients
    c1 = math.exp(-gamma * h)
    c2 = math.sqrt((1 - c1**2) * T_c) if T_c > 0 else 0.0

    # ===== B-step: half kick =====
    # p ← p - (h/2) * (∇Φ_eff - u_π)
    total_force = grad_Phi - u_pi  # Note: gradient is positive, so subtract
    p = p - (h / 2) * total_force

    # ===== A-step: half drift =====
    # z ← Exp_z((h/2) G^{-1} p)
    G_inv = poincare_metric_inv(z)
    velocity = torch.einsum('bij,bj->bi', G_inv, p)  # contravariant velocity

    # Apply geodesic correction to velocity
    geodesic_corr = christoffel_contraction(z, velocity)
    velocity_corrected = velocity - (h / 4) * geodesic_corr  # half of half-step

    z = poincare_exp_map(z, (h / 2) * velocity_corrected)

    # ===== O-step: thermostat =====
    # p ← c₁ p + c₂ G^{1/2} ξ
    G = poincare_metric(z)
    # G^{1/2} via Cholesky (for diagonal, just sqrt of diagonal)
    r_sq = (z ** 2).sum(dim=-1, keepdim=True)
    G_sqrt_factor = 2.0 / (1.0 - r_sq + 1e-8)  # sqrt of conformal factor

    xi = torch.randn_like(p)
    p = c1 * p + c2 * G_sqrt_factor * xi

    # ===== A-step: half drift =====
    G_inv = poincare_metric_inv(z)
    velocity = torch.einsum('bij,bj->bi', G_inv, p)
    geodesic_corr = christoffel_contraction(z, velocity)
    velocity_corrected = velocity - (h / 4) * geodesic_corr

    z = poincare_exp_map(z, (h / 2) * velocity_corrected)

    # ===== B-step: half kick =====
    # Recompute gradient at new position (for accuracy)
    # In practice, often reuse grad_Phi for efficiency
    p = p - (h / 2) * total_force

    # ===== Jump-step: Poisson process =====
    if jump_rate_fn is not None and chart_transition_fn is not None:
        # Compute jump probability
        lambda_jump = jump_rate_fn(z, K)  # [B]
        prob_jump = 1 - torch.exp(-lambda_jump * h)

        # Sample jumps
        u = torch.rand(B, device=device)
        jumps = u < prob_jump  # [B] boolean

        if jumps.any() and value_fn is not None:
            # Determine target chart (simplified: assume single target)
            K_target = (K + 1) % 4  # Example: cycle through 4 charts

            # Apply chart transition for jumping particles
            z_new = chart_transition_fn(z, K, K_target)
            z = torch.where(jumps.unsqueeze(-1), z_new, z)
            K = torch.where(jumps, K_target, K)

            # Update mass (importance weight)
            eta = 1.1  # Jump mass factor
            m = torch.where(jumps, m * eta, m)

    # Project to ensure we stay in disk
    z_norm = torch.sqrt((z ** 2).sum(dim=-1, keepdim=True))
    z = torch.where(z_norm > 0.999, z * 0.999 / z_norm, z)

    return GeodesicState(z=z, p=p, K=K, m=m, s=state.s + h)
```

:::
:::{prf:proposition} BAOAB Preserves Boltzmann
:label: prop-baoab-preserves-boltzmann

The BAOAB integrator preserves the Boltzmann distribution $\rho(z, p) \propto \exp(-\Phi_{\text{eff}}(z)/T_c - \|p\|_G^2 / (2T_c))$ to second order in $h$.

*Proof sketch.* The symmetric splitting B-A-O-A-B ensures time-reversibility of the deterministic steps. The O-step exactly samples the Maxwell-Boltzmann momentum distribution. Together, these guarantee that $\rho$ is a fixed point of the numerical flow up to $O(h^3)$ errors. See {cite}`leimkuhler2016computation`. $\square$

*Remark (Comparison to Euler-Maruyama).* Euler-Maruyama has $O(h)$ bias in the stationary distribution, whereas BAOAB achieves $O(h^2)$. For long trajectories, this difference is critical.

:::

::::{admonition} Physics Isomorphism: Langevin Thermostat
:class: note
:name: pi-langevin-thermostat

**In Physics:** The Langevin equation $m\ddot{x} = -\nabla U - \gamma\dot{x} + \sqrt{2\gamma k_B T}\,\xi(t)$ describes Brownian motion in a potential with friction $\gamma$ and thermal noise. The Ornstein-Uhlenbeck thermostat samples the Maxwell-Boltzmann distribution $p \propto \exp(-mv^2/2k_BT)$ {cite}`leimkuhler2016computation`.

**In Implementation:** The BAOAB integrator (Definition {prf:ref}`def-baoab-splitting`) splits the dynamics:
- **B:** $p \gets p - \frac{h}{2}\nabla_z\Phi_{\text{eff}}$ (kick)
- **A:** $z \gets z + \frac{h}{2}G^{-1}p$ (drift)
- **O:** $p \gets c_1 p + c_2 G^{1/2}\xi$ (thermostat)
- **A, B:** repeat

**Correspondence Table:**
| Molecular Dynamics | Agent (BAOAB) |
|:-------------------|:--------------|
| Position $x$ | Latent state $z$ |
| Momentum $p$ | Auxiliary variable $p$ |
| Potential $U(x)$ | Effective potential $\Phi_{\text{eff}}(z)$ |
| Friction $\gamma$ | Damping coefficient |
| Temperature $k_B T$ | Cognitive temperature $T_c$ |
| Maxwell-Boltzmann | Stationary policy distribution |

**Advantage:** BAOAB preserves the Boltzmann distribution to $O(h^2)$ (Proposition {prf:ref}`prop-baoab-preserves-boltzmann`), avoiding the $O(h)$ bias of Euler-Maruyama.
::::

::::{admonition} Physics Isomorphism: Detailed Balance
:class: note
:name: pi-detailed-balance

**In Physics:** A stochastic process satisfies detailed balance if transition rates satisfy $\pi(x)W(x \to y) = \pi(y)W(y \to x)$ for all states $x, y$. This implies the stationary distribution $\pi$ and time-reversibility {cite}`vanKampen1992stochastic`.

**In Implementation:** The WFR dynamics satisfy detailed balance at equilibrium:

$$
\rho_*(z) \cdot J(z \to z') = \rho_*(z') \cdot J(z' \to z)
$$
where $\rho_* \propto \exp(-\Phi_{\text{eff}}/T_c)\sqrt{|G|}$ is the Boltzmann distribution.

**Correspondence Table:**
| Statistical Mechanics | Agent (Equilibrium) |
|:----------------------|:--------------------|
| Transition rate $W(x \to y)$ | Jump rate $\lambda_{KK'}$ |
| Stationary distribution $\pi$ | Equilibrium belief $\rho_*$ |
| Detailed balance | Reversibility at Nash |
| Entropy production $\dot{S}$ | Zero at equilibrium |
| Fluctuation-dissipation | Einstein relation for $T_c$ |

**Consequence:** Detailed balance ensures the BAOAB thermostat samples the correct distribution.
::::

(sec-the-overdamped-limit)=
### 22.5 The Overdamped Limit

In many applications (diffusion models, biological control), the system operates in the **overdamped regime** where friction dominates inertia. We derive this limit rigorously.

:::{prf:theorem} Overdamped Limit
:label: thm-overdamped-limit

Consider the second-order SDE from Definition {prf:ref}`def-bulk-drift-continuous-flow` with friction coefficient $\gamma$:

$$
m\,\ddot{z}^k + \gamma\,\dot{z}^k + G^{kj}\partial_j\Phi + \Gamma^k_{ij}\dot{z}^i\dot{z}^j = \sqrt{2T_c}\,\left(G^{-1/2}\right)^{kj}\,\xi^j,
$$
where $m$ is the "inertial mass" and $\xi$ is white noise. In the limit $\gamma \to \infty$ with $m$ fixed (or equivalently, $m \to 0$ with $\gamma$ fixed), the dynamics reduce to the first-order Langevin equation:

$$
dz^k = -G^{kj}(z)\,\partial_j\Phi_{\text{gen}}(z)\,ds + \sqrt{2T_c}\,\left(G^{-1/2}(z)\right)^{kj}\,dW^j_s.
$$
*Proof sketch.* In the high-friction limit, velocity equilibrates instantaneously to the force: $\gamma\,\dot{z} \approx -G^{-1}\nabla\Phi$. The geodesic term $\Gamma(\dot{z},\dot{z}) \sim O(|\dot{z}|^2) = O(\gamma^{-2})$ is negligible. What remains is the gradient flow with diffusion. See Appendix A.4 for the full singular perturbation analysis. $\square$

:::
:::{prf:corollary} Recovery of Holographic Flow
:label: cor-recovery-of-holographic-flow

Setting $\alpha = 1$ (pure generation) and $T_c \to 0$ (deterministic limit) in the overdamped equation recovers the holographic gradient flow from Section 21.2:

$$
\dot{z} = -G^{-1}(z)\,\nabla U(z).
$$
For the Poincare disk, this gives $\dot{z} = \frac{(1-|z|^2)}{2}\,z$, which integrates to $|z(\tau)| = \tanh(\tau/2)$.

*Proof.* Direct substitution of $\Phi_{\text{gen}} = U$ into the overdamped equation. The explicit solution for the radial coordinate $r(\tau) = |z(\tau)|$ satisfies $\dot{r} = \frac{1-r^2}{2}$, which integrates to $r(\tau) = \tanh(\tau/2 + \operatorname{artanh}(r_0))$. For $r_0 = 0$, we get $r(\tau) = \tanh(\tau/2)$. $\square$

*Remark.* This proves that the "ad-hoc" holographic law from Section 21 is actually the **optimal control trajectory** for the geometry defined in Section 18, vindicating the intuition.

:::
:::{prf:corollary} Fokker-Planck Duality {cite}`risken1996fokkerplanck`
:label: cor-fokker-planck-duality

The stationary distribution of the overdamped SDE is:

$$
p_*(z) \propto \exp\left(-\frac{\Phi_{\text{gen}}(z)}{T_c}\right)\,\sqrt{|G(z)|},
$$
where $|G| = \det(G)$ is the metric determinant. This is the Boltzmann distribution on the curved manifold.

*Proof.* The Fokker-Planck equation for the overdamped dynamics is:

$$
\partial_s p = \nabla_i\left( G^{ij}\left( p\,\partial_j\Phi + T_c\,\partial_j p \right) \right).
$$
Setting $\partial_s p = 0$ and using detailed balance gives $p \propto e^{-\Phi/T_c} \sqrt{|G|}$. The $\sqrt{|G|}$ factor accounts for the Riemannian volume form. $\square$

**Cross-references:** Section 21.2 (Langevin dynamics), Theorem {prf:ref}`thm-equivalence-of-entropy-regularized-control-forms-discrete-macro`, Section 2.11 (Belief density evolution).

:::

::::{admonition} Physics Isomorphism: Fokker-Planck Equation
:class: note
:name: pi-fokker-planck

**In Physics:** The Fokker-Planck equation describes the time evolution of probability density under drift and diffusion: $\partial_t p = -\nabla \cdot (p\,\mathbf{F}) + D\nabla^2 p$. On a Riemannian manifold with metric $g$, the diffusion term becomes the Laplace-Beltrami operator {cite}`risken1996fokkerplanck`.

**In Implementation:** The belief density $\rho(z,s)$ evolves via (Corollary {prf:ref}`cor-fokker-planck-duality`):

$$
\partial_s p = \nabla_i\left( G^{ij}\left( p\,\partial_j\Phi_{\text{eff}} + T_c\,\partial_j p \right) \right)
$$
with stationary distribution $p_*(z) \propto \exp(-\Phi_{\text{eff}}(z)/T_c)\sqrt{|G(z)|}$.

**Correspondence Table:**
| Statistical Physics | Agent (Belief Dynamics) |
|:--------------------|:------------------------|
| Probability density $p(x,t)$ | Belief density $\rho(z,s)$ |
| Drift force $\mathbf{F}$ | Effective potential gradient $-\nabla\Phi_{\text{eff}}$ |
| Diffusion constant $D$ | Cognitive temperature $T_c$ |
| Laplacian $\nabla^2$ | Laplace-Beltrami $\Delta_G$ |
| Boltzmann equilibrium | WFR stationary distribution |

**Loss Function:** Stein discrepancy $\mathbb{E}[\|\nabla \log p - \nabla \log p_*\|^2_G]$.
::::

(sec-agent-lifecycle-summary)=
### 22.6 Agent Lifecycle Summary

The complete agent lifecycle integrates the components from Sections 21-22 into a coherent execution flow.

:::{prf:definition} Agent Lifecycle Phases
:label: def-agent-lifecycle-phases


| Phase           | Time Interval                | Dynamics                         | Texture      | Key Operations                                                                         |
|-----------------|------------------------------|----------------------------------|--------------|----------------------------------------------------------------------------------------|
| **1. Init**     | $\tau = 0$                   | $z(0) = 0$                       | None         | Initialize at origin; $p(0) \sim \mathcal{N}(0, T_c G(0))$                             |
| **2. Kick**     | $[0, \tau_{kick}]$           | Langevin at origin               | None         | Apply symmetry-breaking control $u_\pi$ (Def. {prf:ref}`prop-so-d-symmetry-at-origin`) |
| **3. Bulk**     | $[\tau_{kick}, \tau_{stop}]$ | BAOAB + Jumps                    | **Firewall** | Geodesic flow with chart transitions                                                   |
| **4. Boundary** | $\tau = \tau_{stop}$         | $\lVert z\rVert \geq R_{cutoff}$ | Sampled      | Sample texture $z_{tex} \sim \mathcal{N}(0, \Sigma(z))$                                |
| **5. Decode**   | Post-$\tau_{stop}$           | —                                | Used         | $x = \text{Decoder}(e_K, z_n, z_{tex})$                                                |

*Remark.* The **Texture Firewall** (Axiom {prf:ref}`ax-bulk-boundary-decoupling`) ensures that $\partial_{z_{tex}} \dot{z} = 0$ throughout the bulk phase—texture is completely invisible to the dynamics.

**Algorithm 22.6.2 (Full Agent Loop).**

```python
def run_agent_loop(
    policy: Policy,
    decoder: Decoder,
    T_c: float,
    gamma: float,
    h: float,
    R_cutoff: float = 0.95,
    max_steps: int = 1000,
) -> torch.Tensor:
    """
    Execute the full agent lifecycle from init to decode.

    Returns: Generated output x
    """
    B, d = 1, policy.latent_dim
    device = policy.device

    # ===== Phase 1: Init =====
    z = torch.zeros(B, d, device=device)
    p = torch.randn(B, d, device=device) * math.sqrt(T_c * 4.0)  # G(0) = 4I
    K = torch.zeros(B, dtype=torch.long, device=device)
    m = torch.ones(B, device=device)
    state = GeodesicState(z=z, p=p, K=K, m=m, s=0.0)

    # ===== Phase 2: Kick =====
    # Apply symmetry-breaking control at origin
    u_pi = policy.symmetry_breaking_kick(z, mode='generation')

    # ===== Phase 3: Bulk (with Texture Firewall) =====
    for step in range(max_steps):
        # Compute effective potential gradient
        grad_Phi = compute_effective_potential_gradient(
            state.z, state.K, policy.value_fn, alpha=0.5
        )

        # Update control field
        u_pi = policy.control_field(state.z, state.K)

        # BAOAB step (texture is invisible here)
        state = geodesic_baoab_step(
            state, grad_Phi, u_pi, T_c, gamma, h,
            jump_rate_fn=policy.jump_rate,
            chart_transition_fn=policy.chart_transition
        )

        # Check boundary condition
        z_norm = torch.sqrt((state.z ** 2).sum(dim=-1))
        if (z_norm >= R_cutoff).all():
            break

    # ===== Phase 4: Boundary - Sample texture =====
    z_tex = sample_holographic_texture(state.z, sigma_tex=0.1)

    # ===== Phase 5: Decode =====
    embedding = policy.chart_embedding(state.K)  # e_K
    x = decoder(embedding, state.z, z_tex)

    return x
```

:::
:::{prf:proposition} Phase Transition Interpretation
:label: prop-phase-transition-interpretation

The agent lifecycle corresponds to a thermodynamic phase transition:

| Phase | Thermodynamic Analogy | Order Parameter |
|-------|----------------------|-----------------|
| Init (gas) | High entropy, symmetric | $\lVert z\rVert = 0$ |
| Kick (nucleation) | Symmetry breaking | $u_\pi \neq 0$ |
| Bulk (liquid) | Directed flow | $0 < \lVert z\rVert < R_{cutoff}$ |
| Boundary (solid) | Crystallization | $\lVert z\rVert \geq R_{cutoff}$ |

:::
(sec-adaptive-thermodynamics)=
### 22.7 Adaptive Thermodynamics (Fluctuation-Dissipation)

The temperature $T_c$ and friction $\gamma$ need not be constant—they can adapt to the local geometry to maintain the Einstein relation.

:::{prf:definition} Einstein Relation on Manifolds
:label: def-einstein-relation-on-manifolds

The fluctuation-dissipation relation requires:

$$
\sigma^2(z) = \frac{2\gamma(z)\, T_c}{G(z)},
$$
where $\sigma^2$ is the noise variance. This ensures the correct equilibrium distribution.

:::
:::{prf:proposition} Automatic Phase Transitions
:label: prop-automatic-phase-transitions

With adaptive temperature $T_c(z)$ satisfying the Einstein relation:

| Regime                      | Metric $G(z)$ | Effective Noise | Phase Behavior                |
|-----------------------------|---------------|-----------------|-------------------------------|
| **Uncertain** (near origin) | Small         | Large           | Gas phase (exploration)       |
| **Certain** (near boundary) | Large         | Small           | Solid phase (crystallization) |

*Remark.* This automatic phase transition emerges from the geometry alone—no explicit temperature schedule is needed.

:::
:::{prf:definition} Fisher-Covariance Duality
:label: def-fisher-covariance-duality

The inverse relationship between uncertainty and metric:

$$
G(z) \approx \Sigma^{-1}(z),
$$
where $\Sigma(z)$ is the posterior covariance of the belief at $z$. This duality underlies the Mass=Metric principle (Definition {prf:ref}`def-mass-tensor`).

**Algorithm 22.7.4 (Adaptive Temperature).**

```python
def adaptive_temperature(
    z: torch.Tensor,
    base_T: float,
    certainty_scale: float = 1.0,
) -> torch.Tensor:
    """
    Compute adaptive temperature based on local geometry.

    T_c(z) = base_T * (1 - |z|^2)^2 / 4

    This maintains constant effective noise: sigma^2 * G = 2 * gamma * T_c
    """
    r_sq = (z ** 2).sum(dim=-1, keepdim=True)
    # Conformal factor inverse: G^{-1} = (1-|z|^2)^2 / 4
    inv_conformal = (1.0 - r_sq + 1e-8) ** 2 / 4.0
    return base_T * inv_conformal * certainty_scale
```

:::
:::{prf:corollary} Deterministic Boundary
:label: cor-deterministic-boundary

As $|z| \to 1$:

$$
T_c(z) \to 0, \qquad \text{noise} \to 0.
$$
The agent becomes deterministic at the boundary, ensuring reproducible outputs.

:::
(sec-summary-tables-and-diagnostic-nodes)=
### 22.8 Summary Tables and Diagnostic Nodes

**Summary of Equations of Motion:**

| Equation                 | Expression                                                                                                                  | Regime       | Units                |
|--------------------------|-----------------------------------------------------------------------------------------------------------------------------|--------------|----------------------|
| Extended Onsager-Machlup | $S_{\mathrm{OM}} = \int (\frac{1}{2}\mathbf{M}\lVert\dot{z}\rVert^2 + \Phi_{\text{eff}} + \frac{T_c}{12}R + T_c H_\pi)\,ds$ | Path-space   | nat                  |
| Full Geodesic SDE        | $dz = (-G^{-1}\nabla\Phi_{\text{eff}} + u_\pi - \Gamma(\dot{z},\dot{z}))\,ds + \sqrt{2T_c}\,G^{-1/2}\,dW_s$                 | Second-order | $[z]$                |
| Overdamped               | $dz = (-G^{-1}\nabla\Phi_{\text{eff}} + u_\pi)\,ds + \sqrt{2T_c}\,G^{-1/2}\,dW_s$                                           | First-order  | $[z]$                |
| Jump Intensity           | $\lambda_{K\to j} = \lambda_0 \exp(\beta\,\Delta V)$                                                                        | Discrete     | step$^{-1}$          |
| Mass = Metric            | $\mathbf{M}(z) \equiv G(z)$                                                                                                 | Kinematic    | $[z]^{-2}$           |
| Texture Covariance       | $\Sigma_{\text{tex}}(z) = \sigma_{\text{tex}}^2\, G^{-1}(z)$                                                                | Boundary     | $[z_{\text{tex}}]^2$ |

**Effective Potential Decomposition:**

$$
\Phi_{\text{eff}}(z, K) = \alpha\,U(z) + (1-\alpha)\,V_{\text{critic}}(z, K) + \gamma_{\text{risk}}\,\Psi_{\text{risk}}(z)
$$
where $\alpha \in [0,1]$ interpolates generation/control and $\gamma_{\text{risk}} \ge 0$ is risk aversion.

**BAOAB Coefficients:**

$$
c_1 = e^{-\gamma h}, \qquad c_2 = \sqrt{(1 - c_1^2)\,T_c}
$$
*Cross-reference:* The boundary-reached condition is monitored by **[Node 25 (HoloGenCheck)](#node-25)** defined in Section 21.4.

(node-26)=
**Node 26: GeodesicCheck**

| **#**  | **Name**          | **Component**            | **Type**                   | **Interpretation**                    | **Proxy**                                                                                  | **Cost**  |
|--------|-------------------|--------------------------|----------------------------|---------------------------------------|--------------------------------------------------------------------------------------------|-----------|
| **26** | **GeodesicCheck** | **World Model / Policy** | **Trajectory Consistency** | Is trajectory approximately geodesic? | $\lVert\ddot{z} + \Gamma(\dot{z},\dot{z}) + G^{-1}\nabla\Phi_{\text{eff}} - u_\pi\rVert_G$ | $O(BZ^2)$ |

**Trigger conditions:**
- High GeodesicCheck: Trajectory deviates from controlled geodesic (unexpected forces or integration errors).
- Remedy: Reduce time step $h$; verify Christoffel computation; check metric consistency.

(node-27)=
**Node 27: OverdampedCheck**

| **#**  | **Name**            | **Component** | **Type**            | **Interpretation**              | **Proxy**                                             | **Cost** |
|--------|---------------------|---------------|---------------------|---------------------------------|-------------------------------------------------------|----------|
| **27** | **OverdampedCheck** | **Policy**    | **Regime Validity** | Is friction dominating inertia? | $\gamma / \lVert G^{-1}\nabla\Phi_{\text{eff}}\rVert$ | $O(BZ)$  |

**Trigger conditions:**
- Low OverdampedCheck: Operating in inertial regime; use full BAOAB integrator.
- Remedy: Increase friction $\gamma$ if overdamped limit desired; otherwise switch to second-order integrator.

(node-28)=
**Node 28: JumpConsistencyCheck**

| **#**  | **Name**                 | **Component**   | **Type**        | **Interpretation**                  | **Proxy**                                                       | **Cost**  |
|--------|--------------------------|-----------------|-----------------|-------------------------------------|-----------------------------------------------------------------|-----------|
| **28** | **JumpConsistencyCheck** | **World Model** | **WFR Balance** | Are jump rates consistent with WFR? | $\lvert\sum_j \lambda_{K\to j} - \sum_i \lambda_{i\to K}\rvert$ | $O(BK^2)$ |

**Trigger conditions:**
- High JumpConsistencyCheck: Jump rates violate detailed balance; may cause mass accumulation/depletion.
- Remedy: Recalibrate jump rates; verify value function consistency across charts.

(node-29)=
**Node 29: TextureFirewallCheck**

| **#**  | **Name**                 | **Component**  | **Type**                     | **Interpretation**              | **Proxy**                                       | **Cost**             |
|--------|--------------------------|----------------|------------------------------|---------------------------------|-------------------------------------------------|----------------------|
| **29** | **TextureFirewallCheck** | **Generation** | **Bulk-Boundary Separation** | Is texture decoupled from bulk? | $\lVert\partial_{z_{\text{tex}}} \dot{z}\rVert$ | $O(BZ_{\text{tex}})$ |

**Trigger conditions:**
- High TextureFirewallCheck: Texture is leaking into dynamics (firewall violated).
- Remedy: Review implementation; ensure texture sampled only at boundary; verify Axiom {prf:ref}`ax-bulk-boundary-decoupling`.



(sec-the-boundary-interface-symplectic-structure)=
