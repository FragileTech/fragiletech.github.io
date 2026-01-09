## 18. Capacity-Constrained Metric Law: Geometry from Interface Limits

:::{admonition} Researcher Bridge: Information Bottleneck Becomes Geometry
:class: info
:name: rb-info-bottleneck-geometry
When you push a model to the edge of representational capacity, the geometry must adapt. This is the rigorous version of information bottleneck regularization: capacity limits induce curvature that slows updates in overloaded regions.
:::

Section 9.10 used a “gravity” analogy to motivate curvature as a regulator. This section removes the analogy: the curvature law is derived as a structural response to **information-theoretic constraints** induced by the agent’s finite-bandwidth boundary (Markov blanket).

The key idea is operational: **the representational complexity of the internal state is bounded by the capacity of the interface channel.** When the agent operates near this bound, curvature appears as the geometric mechanism that prevents internal information volume from exceeding what can be grounded at the interface.

(sec-the-boundary-bulk-information-inequality)=
### 18.1 The Boundary–Bulk Information Inequality

:::{prf:definition} DPI / boundary-capacity constraint
:label: def-dpi-boundary-capacity-constraint

Consider the boundary stream $(X_t)_{t\ge 0}$ and the induced internal state process $(Z_t)_{t\ge 0}$ produced by the shutter (Definition {prf:ref}`def-bounded-rationality-controller`). Because all internal state is computed from boundary influx and internal memory, any information in the bulk must be mediated by a finite-capacity channel. Operationally, the data-processing constraint is:

$$
I_{\text{bulk}} \;\le\; C_{\partial},
$$
where $C_{\partial}$ is the effective information capacity of the boundary channel and $I_{\text{bulk}}$ is the amount of information the agent can stably maintain in $\mathcal{Z}$ without violating Causal Enclosure (no internal source term $\sigma$; Definition {prf:ref}`def-source-residual`).
Units: $[I_{\text{bulk}}]=[C_{\partial}]=\mathrm{nat}$.

:::
:::{prf:definition} Information density and bulk information volume
:label: def-information-density-and-bulk-information-volume

Let $\rho(z,s)$ denote the probability density of the agent's belief state at position $z \in \mathcal{Z}$ and computation time $s$. The **information density** $\rho_I(z,s)\ge 0$ is defined as:

$$
\rho_I(z,s) := -\rho(z,s) \log \rho(z,s) + \frac{1}{2}\rho(z,s) \log\det G(z),
$$
with units of nats per unit Riemannian volume $d\mu_G=\sqrt{|G|}\,dz^n$ ($n=\dim\mathcal{Z}$). The first term is the local entropy contribution (Shannon density); the second term is the geometric correction accounting for the metric-induced volume distortion.

*Remark.* Integrating $\rho_I$ over $\mathcal{Z}$ yields the differential entropy $h[\rho] = -\int \rho \log \rho \, d\mu_G$ plus the expected log-volume $\frac{1}{2}\mathbb{E}_\rho[\log\det G]$. The latter term ensures that the information measure respects the intrinsic geometry: regions with curved (high-$|G|$) geometry contribute more information capacity.

:::
:::{prf:definition} a (Bulk information volume)
:label: def-a-bulk-information-volume

Define the bulk information volume over a region $\Omega\subseteq\mathcal{Z}$ by

$$
I_{\text{bulk}}(\Omega) := \int_{\Omega} \rho_I(z,s)\, d\mu_G.
$$
When $\Omega=\mathcal{Z}$ we write $I_{\text{bulk}}:=I_{\text{bulk}}(\mathcal{Z})$. This is conceptually distinct from the probability-mass balance in Section 2.11; here the integral measures grounded structure in nats.

:::
:::{prf:definition} Boundary capacity: area law at finite resolution
:label: def-boundary-capacity-area-law-at-finite-resolution

Let $dA_G$ be the induced $(n-1)$-dimensional area form on $\partial\mathcal{Z}$. If the boundary interface has a minimal resolvable scale $\ell>0$ (pixel/token floor), then an operational capacity bound is an area law:

$$
C_{\partial}(\partial\mathcal{Z})
:=
\frac{1}{\eta_\ell}\oint_{\partial\mathcal{Z}} dA_G,
$$
where $\eta_\ell$ is the effective boundary area-per-nat at resolution $\ell$ (a resolution-dependent constant set by the interface).
Units: $[\eta_\ell]=[dA_G]/\mathrm{nat}$ and $[\ell]$ is the chosen boundary resolution length scale.

*Remark (discrete macro specialization).* For the split shutter, the most conservative computable proxy is

$$
C_{\partial}\ \approx\ \mathbb{E}[I(X_t;K_t)]\ \le\ \log|\mathcal{K}|,
$$
which is exactly Node 13 (BoundaryCheck) and Theorem {prf:ref}`thm-information-stability-window-operational`’s grounding condition.

:::
(sec-main-result)=
### 18.2 Main Result (Capacity-Saturated Metric Law)

The detailed variational construction is recorded in Appendix A. The main consequence is an Euler–Lagrange identity that ties curvature of the latent geometry to a risk-induced tensor under a finite-capacity boundary.

:::{prf:theorem} Capacity-constrained metric law
:label: thm-capacity-constrained-metric-law

Under the regularity and boundary-clamping hypotheses stated in Appendix A, and under the soundness condition that bulk structure is boundary-grounded (no internal source term $\sigma$ on $\operatorname{int}(\mathcal{Z})$; Definition {prf:ref}`def-source-residual`), stationarity of a capacity-constrained curvature functional implies

$$
R_{ij} - \frac{1}{2}R\,G_{ij} + \Lambda G_{ij} = \kappa\, T_{ij},
$$
where $\Lambda$ and $\kappa$ are constants and $T_{ij}$ is the **total Risk Tensor** induced by the reward field. *Units:* $\Lambda$ has the same units as curvature ($[R]\sim [z]^{-2}$), and $\kappa$ is chosen so that $\kappa\,T_{ij}$ matches those curvature units.

*Operational reading.* Curvature is the geometric mechanism that prevents the internal information volume (Definition 18.1.2a) from exceeding the boundary's information bandwidth (Definition {prf:ref}`def-a-bulk-information-volume`) while remaining grounded.

**Implementation hook.** The squared residual of this identity defines a capacity-consistency regularizer $\mathcal{L}_{\text{cap-metric}}$; see Appendix B for the consolidated list of loss definitions and naming conventions.

:::

:::{prf:definition} Extended Risk Tensor with Maxwell Stress
:label: def-extended-risk-tensor

The total Risk Tensor $T_{ij}$ decomposes into gradient and curl contributions:

$$
T_{ij} = T_{ij}^{\text{gradient}} + T_{ij}^{\text{Maxwell}},
$$
where:

1. **Gradient Stress** (from scalar potential $\Phi$):

$$
T_{ij}^{\text{gradient}} = \partial_i \Phi \, \partial_j \Phi - \frac{1}{2}G_{ij} \|\nabla\Phi\|_G^2
$$
2. **Maxwell Stress** (from Value Curl $\mathcal{F}$):

$$
T_{ij}^{\text{Maxwell}} = \mathcal{F}_{ik}\mathcal{F}_j^{\;k} - \frac{1}{4}G_{ij}\mathcal{F}^{kl}\mathcal{F}_{kl}
$$
*Units:* $[T_{ij}] = \mathrm{nat}^2/[z]^2$.

**Conservative Limit:** When $\mathcal{F} = 0$ (Definition {prf:ref}`def-conservative-reward-field`), the Maxwell term vanishes and we recover the standard gradient-only risk tensor.

**Non-Conservative Case:** When $\mathcal{F} \neq 0$, the Maxwell stress contributes additional terms to the curvature equation.

:::

::::{admonition} Physics Isomorphism: Einstein Field Equations
:class: note
:name: pi-einstein-equations

**In Physics:** Einstein's field equations relate spacetime curvature to stress-energy: $R_{\mu\nu} - \frac{1}{2}Rg_{\mu\nu} + \Lambda g_{\mu\nu} = 8\pi G T_{\mu\nu}$ {cite}`einstein1915field,wald1984general`.

**In Implementation:** The capacity-constrained metric law (Theorem {prf:ref}`thm-capacity-constrained-metric-law`) relates latent geometry to risk:

$$
R_{ij} - \frac{1}{2}R\,G_{ij} + \Lambda G_{ij} = \kappa T_{ij}
$$
**Correspondence Table:**

| General Relativity | Agent (Metric Law) |
|:-------------------|:-------------------|
| Spacetime metric $g_{\mu\nu}$ | Latent metric $G_{ij}$ |
| Ricci tensor $R_{\mu\nu}$ | Ricci tensor $R_{ij}$ (of $G$) |
| Cosmological constant $\Lambda$ | Baseline curvature $\Lambda$ |
| Stress-energy $T_{\mu\nu}$ | Risk tensor $T_{ij}$ |
| Gravitational coupling $8\pi G$ | Capacity coupling $\kappa$ |
| Schwarzschild horizon | Saturation horizon (Lemma {prf:ref}`lem-metric-divergence-at-saturation`) |

**Loss Function:** $\mathcal{L}_{\text{EFE}} := \|R_{ij} - \frac{1}{2}R\,G_{ij} + \Lambda G_{ij} - \kappa T_{ij}\|_F^2$.
::::

(sec-diagnostic-node-capacity-saturation)=
### 18.3 Diagnostic Node: Capacity Saturation

| #  | Name                    | Measures                        | Trigger                                         |
|----|-------------------------|---------------------------------|-------------------------------------------------|
| 40 | CapacitySaturationCheck | Bulk-boundary information ratio | $I_{\text{bulk}} / C_{\partial} > 1 - \epsilon$ |

:::{prf:definition} Capacity saturation diagnostic
:label: def-capacity-saturation-diagnostic

Compute the capacity saturation ratio:

$$
\nu_{\text{cap}}(s) := \frac{I_{\text{bulk}}(s)}{C_{\partial}},
$$
where $I_{\text{bulk}}(s) = \int_{\mathcal{Z}} \rho_I(z,s)\, d\mu_G$ per Definition 18.1.2a.

*Interpretation:*
- $\nu_{\text{cap}} \ll 1$: Under-utilized capacity; the agent may be compressing excessively (lossy representation).
- $\nu_{\text{cap}} \approx 1$: Operating at capacity limit; geometry must regulate to prevent overflow.
- $\nu_{\text{cap}} > 1$: **Violation** of the DPI constraint (Definition {prf:ref}`def-dpi-boundary-capacity-constraint`); indicates ungrounded structure.

*Cross-reference:* When $\nu_{\text{cap}} > 1$, the curvature correction (Theorem {prf:ref}`thm-capacity-constrained-metric-law`) is insufficient. This triggers geometric reflow—the metric $G$ must increase $|G|$ (expand volume) to bring $I_{\text{bulk}}$ back within bounds.

:::

::::{note} Connection to RL #25: Information Bottleneck as Degenerate Capacity-Constrained Metric
**The General Law (Fragile Agent):**
The latent metric obeys a **Capacity-Constrained Consistency Law** (Theorem {prf:ref}`thm-capacity-constrained-metric-law`):

$$
R_{ij} - \frac{1}{2}R\, G_{ij} + \Lambda G_{ij} = \kappa\, T_{ij}
$$
where $R_{ij}$ is Ricci curvature and $T_{ij}$ is the Risk Tensor. The constraint is the **DPI inequality**: $I_{\text{bulk}} \le C_\partial \le \log|\mathcal{K}|$.

**The Degenerate Limit:**
Remove geometric structure ($G \to I$, $R_{ij} \to 0$). Replace the area law with a scalar rate constraint $\beta$.

**The Special Case (Standard RL):**

$$
\max_\theta I(Z; Y) - \beta I(Z; X)
$$
This recovers the **Information Bottleneck** {cite}`tishby2015ib` and **Variational Information Bottleneck (VIB)** {cite}`alemi2016vib`.

**What the generalization offers:**
- **Geometric response**: Curvature *emerges* from capacity constraints—it's not imposed by hand
- **Area law**: Boundary capacity scales with interface area $C_\partial \sim \text{Area}(\partial\mathcal{Z})$, not arbitrary $\beta$
- **Grounded structure**: Bulk information must be mediated by finite-bandwidth boundary (DPI)
- **Diagnostic saturation**: CapacitySaturationCheck (Node 40) monitors $\nu_{\text{cap}} = I_{\text{bulk}}/C_\partial$ at runtime
::::



(sec-conclusion)=
