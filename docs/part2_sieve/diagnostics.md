## 3. Diagnostics: Stability Checks (Monitors)

Stability and data-quality are monitored via 29 distinct checks (Gate Nodes). Each corresponds to a specific, testable condition on the interaction between the agent and its environment.

**Relation to prior work.** Many safe-RL formulations express safety as one (or a few) expected-cost constraints in a constrained MDP {cite}`altman1999constrained,achiam2017constrained`. The Fragile Agent keeps that spirit but broadens the constraint surface to include **representation and interface diagnostics** (grounding, mixing, saturation, switching, stiffness) that can be audited online, alongside Lyapunov-style stability constraints {cite}`chow2018lyapunov`.

:::{admonition} Researcher Bridge: Safety as a Unit Test
:class: warning
:name: rb-safety-unit-test
Standard RL safety relies on reward shaping, which provides no formal guarantee that the agent avoids bad states. The Sieve replaces probabilistic incentives with **Hard Runtime Assertions**. Each of the 60 nodes is a mathematical contract. If a check fails (e.g., the agent exhibits chattering or its belief decouples from the sensors), the system does not receive a penalty; it **halts or reverts**. This architecture enforces safety constraints in the same manner that a type system enforces invariants at compile time.
:::

:::{note} Connection to RL #8: Constrained MDPs as Soft Sieve
**The General Law (Fragile Agent):**
Safety is a **topological constraint** enforced by the Sieve—a hard binary filter:

$$
\text{Sieve}(a \mid z) = \begin{cases} \text{PASS} & \text{if all 60 diagnostics pass} \\ \text{BLOCK} & \text{otherwise} \end{cases}
$$
Actions failing any diagnostic are **blocked**, not penalized. The Sieve is not subject to reward-cost trade-offs.

**The Degenerate Limit:**
Replace the hard binary filter with a soft Lagrange multiplier $\lambda > 0$.

**The Special Case (Standard RL):**

$$
J_{\text{CMDP}}(\pi) = \mathbb{E}\left[\sum_t \gamma^t r_t\right] - \lambda \sum_i c_i(\pi)
$$
This recovers **Constrained MDPs** (CMDPs) with penalty-based constraint satisfaction.

**Result:** CMDPs will violate safety if $\lambda$ is too small or reward magnitude is large enough to outweigh the penalty. The Fragile Sieve makes violations *impossible*, not just expensive.

**What the generalization offers:**
- Hard guarantees: topological constraints cannot be circumvented by high rewards
- Typed diagnostics: 59 constraints with semantic identity (what/where/why), not just scalar cost
- Fail-fast semantics: violations halt execution and trigger remediation, not gradual penalty accumulation
- Auditable: each check has known compute cost and clear interpretation (Sections 3–6)
:::

:::{figure} ../svg_images/sieve_diagnostic_system.svg
:name: fig-sieve-diagnostics
:width: 100%

**The Sieve: 60 Runtime Contracts.** The diagnostic monitoring architecture organized into six categories: Stability, Capacity, Grounding, Safety, Multi-Agent, and Ontology. Each diagnostic feeds into three intervention levels (WARN, HALT, KILL). The design principle: failure modes must be observable and trigger explicit remediation.
:::

(sec-the-stability-checks)=
### The 29 Stability Checks

| Node    | Check                                             | Component                | Interpretation                  | Meaning                                     | Regularization Factor ($\mathcal{L}_{\text{check}}$)                                                                                       | Compute                         |
|---------|---------------------------------------------------|--------------------------|---------------------------------|---------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------|
| **1**   | **CostBoundCheck ($D_C$)**                        | **Critic**               | **Cost Budget Check**           | Is current cost ($V(z)$) within budget?     | $\max(0, V(z) - V_{\text{max}})^2$ (Cost Bound)                                                                                            | $O(B)$ ✓                        |
| **2**   | **ZenoCheck ($\mathrm{Rec}_N$)**                  | **Policy**               | **Action Frequency Limit**      | Switching policies too fast?                | $D_{\mathrm{KL}}(\pi_t \Vert \pi_{t-1})$ (Smoothness)                                                                                      | $O(BA)$ ✓                       |
| **3**   | **CompactCheck ($C_\mu$)**                        | **VQ-VAE**               | **Belief Concentration**        | Macro assignment sharp?                     | $H(q(K \mid x))$ (Symbol Entropy)                                                                                                          | $O(BZ)$ ✓                       |
| **4**   | **ScaleCheck ($\mathrm{SC}_\lambda$)**            | **All**                  | **Adaptation Scaling**          | Adaptation speed > Disturbance speed?       | $\Vert \nabla \theta \Vert / \Vert \Delta S \Vert$ (Relative Rate)                                                                         | $O(P)$ ⚡                        |
| **5**   | **ParamCheck ($\mathrm{SC}_{\partial c}$)**       | **World Model**          | **Stationarity Check**          | Dynamics stable?                            | $\Vert \nabla_t S_t \Vert^2$ (Time Derivative Penalty)                                                                                     | $O(P_{WM})$ ⚡                   |
| **6**   | **GeomCheck ($\mathrm{Cap}_H$)**                  | **VQ-VAE / WM**          | **Blind Spot Check**            | Unobservable states negligible?             | $\mathcal{L}_{\text{contrastive}}$ (InfoNCE)                                                                                               | $O(B^2Z)$ ⚡                     |
| **7**   | **StiffnessCheck ($\mathrm{LS}_\sigma$)**         | **Critic**               | **Responsiveness / Gain**       | Gradient signal strong enough?              | $\max(0, \epsilon - \Vert \nabla V \Vert)$ (Gain > $\epsilon$)                                                                             | $O(BZ)$ ✓                       |
| **7a**  | **BifurcateCheck ($\mathrm{LS}_{\partial^2 V}$)** | **World Model**          | **Instability Check**           | Bifurcation point?                          | $\det(J_{S_t})$ (Jacobian Determinant)                                                                                                     | $O(Z^3)$ ✗                      |
| **7b**  | **SymCheck ($G_{\mathrm{act}}$)**                 | **Policy**               | **Alternative Strategy Search** | Symmetric strategies available?             | $-\sum \pi(a_i) \log \pi(a_i)$ (Policy Entropy)                                                                                            | $O(BA)$ ✓                       |
| **7c**  | **CheckSC ($\mathrm{SC}_{\partial c}$)**          | **Critic**               | **New Mode Viability**          | New mode stable?                            | $\text{Var}(V(z'))$ (Variance Check)                                                                                                       | $O(B)$ ✓                        |
| **7d**  | **CheckTB ($\mathrm{TB}_S$)**                     | **Policy**               | **Transition Feasibility**      | Switching cost affordable?                  | $\Vert V(\pi') - V(\pi) \Vert - B_{\text{switch}}$                                                                                         | $O(B)$ ⚡                        |
| **8**   | **TopoCheck ($\mathrm{TB}_\pi$)**                 | **Policy**               | **Sector Reachability**         | Goal reachable?                             | $T_{\text{reach}}(z_{\text{goal}})$ (Reachability Map)                                                                                     | $O(HBZ)$ ✗                      |
| **9**   | **TameCheck ($\mathrm{TB}_O$)**                   | **World Model**          | **Interpretability Check**      | Dynamics Lipschitz-bounded?                               | $\Vert \nabla^2 S_t \Vert$ (Hessian Norm / Smoothness)                                                                                     | $O(Z^2 P_{WM})$ ✗               |
| **10**  | **ErgoCheck ($\mathrm{TB}_\rho$)**                | **Policy**               | **Exploration/Mixing**          | Sufficient exploration?                     | $-H(\pi)$ (Max Entropy)                                                                                                                    | $O(BA)$ ✓                       |
| **11**  | **ComplexCheck ($\mathrm{Rep}_K$)**               | **VQ-VAE**               | **Model Capacity Check**        | Symbolic rate within budget?                | $\mathrm{Rep}_K := H(K)/\log\lvert\mathcal{K}\rvert$ (Rate Utilization)                                                                    | $O(B)$ ✓                        |
| **12**  | **OscillateCheck ($\mathrm{GC}_\nabla$)**         | **WM / Policy**          | **Oscillation / Chattering**    | Limit cycles?                               | $\Vert z_t - z_{t-2} \Vert$ (Period-2 Penalty)                                                                                             | $O(BZ)$ ✓                       |
| **12a** | **HolonomyCheck ($\mathrm{GC}_{\mathrm{holo}}$)** | **WM / Policy**          | **Loop Drift**                  | Near-closed loop changes policy/value?      | $\mathbb{I}[d_G(z_t,z_{t-L})<\epsilon_z]\cdot \mathrm{ReLU}(D_{\mathrm{KL}}(\pi(\cdot\mid z_t)\Vert \pi(\cdot\mid z_{t-L}))-\epsilon_h)^2$ | $O(BA)$ ✓                       |
| **13**  | **BoundaryCheck ($\mathrm{Bound}_\partial$)**     | **VQ-VAE**               | **Input Informativeness**       | External signal present?                    | $I(X;K)$ (Symbolic MI $>0$)                                                                                                                | $O(B)$ ✓                        |
| **14**  | **InputSaturationCheck ($\mathrm{Bound}_B$)**     | **Boundary**             | **Input Saturation**            | Inputs clipping?                            | $\mathbb{I}(\lvert x \rvert > x_{\text{max}})$ (Saturation Flag)                                                                           | $O(BD)$ ✓                       |
| **15**  | **SNRCheck ($\mathrm{Bound}_{\Sigma}$)**          | **Boundary**             | **Signal-to-Noise**             | Signal strength sufficient?                 | $\text{SNR} < \epsilon$ (Noise Floor Check)                                                                                                | $O(BD)$ ✓                       |
| **16**  | **AlignCheck ($\mathrm{GC}_T$)**                  | **Critic**               | **Objective Alignment**         | Proxy matches objective?                    | $\lvert V_{\text{proxy}} - V_{\text{true}} \rvert$ (Alignment Error)                                                                       | $O(B)$ ✗                        |
| **17**  | **Lock ($\mathrm{Cat}_{\mathrm{Hom}}$)**          | **WM**                   | **Structural Constraint**       | Hard safe-guards active?                    | $\mathbb{I}(\text{Unsafe}) \cdot \infty$ (Hard Constraint)                                                                                 | $O(B)$ ✓                        |
| **18**  | **SymmetryCheck ($\mathrm{Sym}_G$)**              | **Shutter**              | **Orbit Invariance**            | Macro invariant to nuisance group?          | $\mathbb{E}_{g\sim G_{\text{spatial}}}\!\left[D_{\mathrm{KL}}(q(K\!\mid x)\Vert q(K\!\mid g\!\cdot\! x))\right]$                           | $O(B)$ ✓                        |
| **19**  | **DisentanglementCheck ($\mathrm{Decorr}_{Kn}$)** | **Shutter / WM**         | **Macro–Nuisance Leakage**      | Macro correlated with nuisance residual?    | $\left\lVert\mathrm{Cov}(z_{\text{macro}},z_n)\right\rVert_F^2$                                                                            | $O(Bd_md_n)$ ✓                  |
| **20**  | **LipschitzCheck ($\mathrm{Lip}_\Theta$)**        | **WM / Critic**          | **Gain Control**                | Operator norms bounded?                     | $\max_\ell \sigma(W_\ell)$ (spectral norm monitor)                                                                                         | $O(P)$ ⚡                        |
| **21**  | **SymplecticCheck ($\mathrm{Symp}$)**             | **World Model**          | **Volume Preservation**         | Transition approximately symplectic?        | $\left\lVert J_S^\top J J_S - J\right\rVert_F^2$                                                                                           | $O(BZ^2)$ ✗                     |
| **22**  | **MECCheck ($\mathrm{MEC}$)**                     | **Belief / WM**          | **CPTP Consistency**            | Operator update matches GKSL form?          | $\left\lVert\frac{\varrho_{t+1}-\varrho_t}{\Delta t}-\mathcal{L}_{\text{GKSL}}(\varrho_t)\right\rVert_F^2$                                 | $O(BZ^3)$ ✗                     |
| **23**  | **NEPCheck ($\mathrm{NEP}$)**                     | **Belief / Boundary**    | **Update vs Evidence**          | Internal update supported by boundary info? | $\mathrm{ReLU}(D_{\mathrm{KL}}(p_{t+1}\Vert p_t)-I(X_t;K_t))^2$                                                                            | $O(B\lvert\mathcal{K}\rvert)$ ✓ |
| **24**  | **QSLCheck ($\mathrm{QSL}$)**                     | **All**                  | **Update Speed Limit**          | Step too large in $d_G$?                    | $\mathrm{ReLU}(d_G(z_{t+1},z_t)-v_{\max})^2$                                                                                               | $O(BZ)$ ✓                       |
| **25**  | **HoloGenCheck**                                  | **Generator**            | **Generation Validity**         | Did flow reach boundary?                    | $\mathbb{I}(\lvert z_{\text{final}}\rvert \ge R_{\text{cutoff}})$                                                                          | $O(B)$ ✓                        |
| **26**  | **GeodesicCheck**                                 | **World Model / Policy** | **Trajectory Consistency**      | Is trajectory approximately geodesic?       | $\lVert\ddot{z} + \Gamma(\dot{z},\dot{z}) + G^{-1}\nabla\Phi\rVert_G$                                                                      | $O(BZ^2)$ ✗                     |
| **27**  | **OverdampedCheck**                               | **Policy**               | **Regime Validity**             | Is friction >> 1 satisfied?                 | $\gamma / \lVert G\,\nabla\Phi\rVert$                                                                                                      | $O(BZ)$ ✓                       |

**Compute Legend:** ✓ Low (typically online) | ⚡ Moderate (often amortized/approximated) | ✗ High (often offline or coarse approximations)
**Variables:** $B$ = batch, $Z$ = latent dim, $A$ = actions, $P$ = params, $H$ = horizon, $D$ = observation dim
**Threshold units:** whenever a node uses a threshold $\epsilon$, it inherits the units of the compared quantity (e.g., $\epsilon$ is dimensionless for SNR checks; $\epsilon$ has the same units as $\|\nabla V\|$ for stiffness checks; $\epsilon$ is in nats when compared to $I(X;K)$ or $H(K)$). Budgets like $V_{\text{max}},V_{\text{limit}}$, and $B_{\text{switch}}$ share units with $V$ (nats in the convention of Section 1.2).

**Geometric Properties of Key Nodes:**

| Node | Space | Formal Property | Verification Criterion |
|------|-------|-----------------|------------------------|
| **1 (CostBound)** | $V \in \mathcal{F}(\mathcal{Z})$ | Sublevel Set Compactness | Is $\{z \mid V(z) \leq c\}$ compact? |
| **7 (Stiffness)** | $G \in T^*_2(\mathcal{Z})$ | Spectral Gap | Is $\lambda_{\min}(G) > \epsilon$? (No flat directions) |
| **9 (Tameness)** | $f: \mathcal{Z} \to T\mathcal{Z}$ | Lipschitz Continuity | Is $\lVert\nabla_z f\rVert_G < K$? (Bounded sensitivity) |
| **17 (Lock)** | $H_n(\mathcal{Z})$ | Homological Obstruction | Does the prohibited configuration induce a non-trivial cycle? |

Each node corresponds to a verifiable geometric property. The Sieve acts as a **topological filter**: problems that fail these checks are rejected before gradient updates can corrupt the agent.

(sec-theory-thin-interfaces)=
### 3.1 Theory: Thin Interfaces

In the Hypostructure framework, **Thin Interfaces** are defined as minimal couplings between components. Instead of monolithic end-to-end training, we enforce structural contracts (the checks) via **Defect Functionals** ($\mathcal{L}_{\text{check}}$).

*   **Principle:** Components (VQ-VAE, WM, Critic, Policy) should be **autonomous** but **aligned**.
*   **Mechanism:** Each component minimizes its own objective *subject to* the cybernetic constraints imposed by the others.

(sec-scaling-exponents-characterizing-the-agent)=
### 3.2 Scaling Exponents: Characterizing the Agent

We characterize the training dynamics of the Fragile Agent using four **scaling coefficients**. These are *diagnostic* summaries of state-space behavior, not optimizer statistics.

The geometric metric $G$ is the **State-Space Fisher Information** (see Section 2.6), ensuring coordinate invariance. Common diagonal approximations include:
- `policy_fisher`: $G_{ii} = \mathbb{E}[(\partial \log \pi / \partial z_i)^2]$
- `state_fisher`: $G_{ii} = \mathbb{E}[(\partial \log \pi / \partial z_i)^2] + \text{Hess}_z(V)_{ii}$ (Hessian + Fisher sensitivity)
- `grad_rms`: $G_{ii} = \mathbb{E}[(\partial V / \partial z_i)^2]^{1/2}$
- `obs_var`: $G_{ii} = \text{Var}(z_i)$

| Component       | Exponent              | Symbol   | Units         | Interpretation                                               | Diagnostics                                                                                 |
|:----------------|:----------------------|:---------|:--------------|:-------------------------------------------------------------|:--------------------------------------------------------------------------------------------|
| **Critic**      | **Curvature scale**   | $\alpha$ | dimensionless | **Value curvature:** magnitude of value gradients/curvature. | High $\alpha$: strong supervision.<br>Low $\alpha$: flat value surface (BarrierGap).            |
| **Policy**      | **Exploration scale** | $\beta$  | dimensionless | **Policy variance / update scale.**                          | High $\beta$: high noise/plasticity.<br>Low $\beta$: near-deterministic/frozen.             |
| **World Model** | **Volatility scale**  | $\gamma$ | dimensionless | **Dynamics non-stationarity / rollout volatility.**          | High $\gamma$: unstable/chaotic predictions.<br>Low $\gamma$: stable dynamics.              |
| **VQ-VAE**      | **Drift scale**       | $\delta$ | dimensionless | **Representation drift:** codebook/encoder stability.        | High $\delta$: symbol churn (representation drift).<br>Low $\delta$: stable representation. |

**The Stability Hierarchy (BarrierTypeII):**
Stable training requires separation of timescales: the representation should change slowest, the world model should not drift faster than the critic can track, and the policy should not update faster than the critic’s usable signal. A practical regime is:

$$
\delta \ll \gamma \ll \alpha,\qquad \beta \le \alpha
$$
1.  **$\delta \ll \gamma$ (Representation Stability):** the representation (encoder/codebook) drifts slower than the learned dynamics model.
2.  **$\gamma \ll \alpha$ (Predictability / Trackability):** the learned dynamics do not drift faster than the value function can track.
3.  **$\beta \le \alpha$ (Two-Time-Scale Actor–Critic):** policy updates stay within the critic’s validity region. If $\beta>\alpha$, skip or shrink the policy update (BarrierTypeII; see Section 4.1.2).

(sec-defect-functionals-implementing-regulation)=
### 3.3 Defect Functionals: Implementing Regulation

We regulate the Fragile Agent by augmenting the loss function with specific terms for each component. These terms are non-negotiable cybernetic contracts.

(sec-gauge-invariant-regulation)=
#### 3.3.0 Gauge-Invariant Regulation (Symmetry Quotienting)

The “Fragile” design is compatible with (and benefits from) an explicit **symmetry layer** (Section 1.1.4): identify nuisance degrees of freedom as group actions and enforce invariance/equivariance so that capacity is spent on control-relevant structure.

The table below summarizes a minimal, implementable set of **gauge-invariant regulation** mechanisms. Each item is expressed as a concrete loss/monitor and mapped to an existing Fragile failure mode (Section 5). The intent is not to import physics metaphors, but to use the standard mathematical language of symmetry and invariance (group actions, quotienting, equivariance).

| Method                                              | Gauge / nuisance variable                     | Implementation (loss / constraint)                                                                                                                  | Failure mode mitigated                   | Notes                                                                                                                                                                                                          |
|:----------------------------------------------------|:----------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Projective (bounded) value head**                 | reward scale / value magnitude drift          | $u(z)=\phi(z)/\lVert\phi(z)\rVert,\ \ \omega=\tilde\omega/\lVert\tilde\omega\rVert,\ \ V(z)=V_{\mathrm{scale}}\,(1-u(z)\cdot\omega)$                | Mode C.E (divergence / blow-up)          | Bounded state-dependent part; $V_{\mathrm{scale}}$ (units: nat) can be calibrated or learned as an adaptive multiplier; does not eliminate the need for consistent units elsewhere.                            |
| **Orbit-invariance loss**                           | pose/basis nuisance $g\in G_{\text{spatial}}$ | $\mathcal{L}_{\text{orbit}}=\mathbb{E}_{g}\big[D_{\mathrm{KL}}(q(K\mid x)\Vert q(K\mid g\cdot x))\big]$                                             | Mode S.D (symmetry blindness)            | Implements “$K$ approximates $x/G$” by encouraging macro assignments to be invariant under nuisance transforms.                                                                                                |
| **Macro–(nuisance+texture) cross-covariance**       | leakage between $K$ and residual channels     | $\mathcal{L}_{K\perp \bullet}=\lVert\mathrm{Cov}(z_{\text{macro}},z_n)\rVert_F^2 + \lVert\mathrm{Cov}(z_{\text{macro}},z_{\mathrm{tex}})\rVert_F^2$ | Mode T.C (overfitting to residuals)      | Practical surrogate for reducing residual leakage into the macro register. Texture leakage is always a defect; nuisance leakage is a defect when it changes macro identity. Monitored by DisentanglementCheck. |
| **Spectral (Lipschitz) barrier**                    | gain / sensitivity drift                      | spectral norm constraints (per-layer) {cite}`miyato2018spectral`                                                                                    | Mode B.E (fragility)                     | Bounds local gain; supports stable rollouts and well-conditioned metrics.                                                                                                                                      |
| **Symplectic / Hamiltonian world model (optional)** | phase-space distortion                        | parameterize $\dot{z}=J\nabla H(z,a)$ or penalize symplectic defect                                                                                 | Mode D.E (oscillation) / numeric blow-up | Appropriate when the latent dynamics are well-modeled as near-Hamiltonian; otherwise treat as optional structure.                                                                                              |
| **Hodge-style alignment (optional)**                | solenoidal loop component in induced flow     | $\mathcal{L}_{\text{Hodge}}=1-\cos(\Delta z,\ -G^{-1}\nabla V)$                                                                                     | Mode D.E (oscillatory)                   | Encourages the policy-induced state velocity to align with value descent, suppressing circular components that cause chattering.                                                                               |
| **Canonicalization shutter (STN) (optional)**       | input frame / pose gauge                      | $x\mapsto \tilde x=C_\psi(x)$, then VQ on $\tilde x$ {cite}`jaderberg2015stn`                                                                       | Mode S.D / Node 11 (capacity)            | Reduces the effective entropy of $K$ by canonicalizing nuisance transforms before discretization.                                                                                                              |
| **Diagonal metric law**                             | coordinate basis choice                       | natural-gradient / trust region with state metric $G$                                                                                               | Mode B.C (control deficit)               | Enforces coordinate-invariant update geometry in latent state space (Sections 2.5–2.6).                                                                                                                        |

(sec-a-vq-vae-regulation)=
#### A. VQ-VAE Regulation (The Shutter)
*   **Symbolic Bottleneck (Node 3 / 11):** the shutter is a split latent $(K,z_n,z_{\mathrm{tex}})$ with $K\in\mathcal{K}$ discrete (Section 2.2b). A canonical objective is:

    $$
    \mathcal{L}_{\text{shutter}}
    =
    \mathcal{L}_{\text{recon}}
    + \underbrace{\lVert \operatorname{sg}[z_e]-e_{K}\rVert_2^2 + \beta\lVert z_e-\operatorname{sg}[e_K]\rVert_2^2}_{\text{VQ codebook + commitment}}
    + \underbrace{\beta_n D_{\mathrm{KL}}(q(z_n \mid x) \Vert p(z_n))}_{\text{nuisance prior (regularize)}}
    + \underbrace{\beta_{\mathrm{tex}} D_{\mathrm{KL}}(q(z_{\mathrm{tex}} \mid x) \Vert p(z_{\mathrm{tex}}))}_{\text{texture-as-residual}}
    + \underbrace{\lambda_{\text{use}} D_{\mathrm{KL}}(\hat{p}(K)\ \Vert\ \mathrm{Unif}(\mathcal{K}))}_{\text{anti-collapse (optional)}}.
    $$
    Units: {math}`\beta`, {math}`\beta_n`, {math}`\beta_{\mathrm{tex}}`, and {math}`\lambda_{\text{use}}` are dimensionless weights; each {math}`D_{\mathrm{KL}}` is measured in nats.
    *   *Effect:* The macro channel is a bounded-rate symbolic register. The nuisance channel is regularized but typed (it may be used to explain structured deviations or support actuation). The texture channel is reconstruction-only: it is forced toward a high-entropy prior and must not be required for macro closure or control.
*   **Orbit invariance (Node 18: SymmetryCheck; optional but recommended when $G_{\text{spatial}}$ is known).**
    Sample nuisance transforms $g\sim G_{\text{spatial}}$ (data augmentation, known pose perturbations, or learned warps) and penalize changes in macro assignment:

    $$
    \mathcal{L}_{\text{orbit}}
    :=
    \mathbb{E}_{g}\!\left[D_{\mathrm{KL}}\!\left(q(K\mid x)\ \Vert\ q(K\mid g\cdot x)\right)\right].
    $$
    This is a direct operationalization of the quotient intent “$K$ approximates $x/G_{\text{spatial}}$” (Section 2.2b) and prevents symmetry-blind representations.
*   **Macro–residual disentanglement (Node 19: DisentanglementCheck).**
    Enforce that the control-relevant macro embedding $z_{\text{macro}}:=e_K$ does not carry the same variation as either residual channel by discouraging cross-covariance:

    $$
    \mathcal{L}_{K\perp \bullet}
    :=
    \left\|\mathrm{Cov}(z_{\text{macro}}, z_n)\right\|_F^2
    +
    \left\|\mathrm{Cov}(z_{\text{macro}}, z_{\mathrm{tex}})\right\|_F^2.
    $$
    This complements enclosure/closure constraints (Section 2.8). Texture leakage is treated as strictly disallowed; nuisance leakage is allowed only insofar as it does not alter macro identity.
*   **Canonicalization shutter (optional).**
    If $G_{\text{spatial}}$ corresponds to a known input-frame nuisance (pose/basis), insert $x\mapsto \tilde x=C_\psi(x)$ (e.g., an STN) before the VQ encoder and train $C_\psi$ jointly using $\mathcal{L}_{\text{orbit}}$ and reconstruction/closure losses (Section 2.2b) {cite}`jaderberg2015stn`.
*   **Contrastive Anchoring (Node 6):**

    $$
    \mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp(\text{sim}(z_t, z_{t+k}))}{\sum \exp(\text{sim}(z_t, z_{neg}))}
    $$
    *   *Effect:* Ensures the latent space captures long-term structural dependencies (slow features), not just pixel reconstruction.

*   **VICReg: Variance-Invariance-Covariance Regularization (Alternative to InfoNCE):**

    VICReg {cite}`bardes2022vicreg` provides an alternative approach to preventing representation collapse **without requiring negative samples**. While InfoNCE contrasts positive pairs against negatives, VICReg uses geometric constraints.

    **The Collapse Problem:**
    Self-supervised learning can produce trivial solutions where the encoder maps all inputs to a constant. VICReg prevents this through three orthogonal constraints:

    **1. Invariance Loss (Metric Stability):**

    $$
    \mathcal{L}_{\text{inv}} = \lVert z - z'\rVert^2
    $$
    - $z, z'$ are embeddings of two augmented views of the same input
    - *Effect:* Forces representations to be stable under perturbations

    **2. Variance Loss (Non-Collapse):**

    $$
    \mathcal{L}_{\text{var}} = \frac{1}{d} \sum_{j=1}^{d} \max(0, \gamma - \sqrt{\text{Var}(z_j) + \epsilon})
    $$
- $\gamma$ is the target standard deviation (typically 1)
- Units: $[\gamma]=[z_j]$ and $[\epsilon]=[z_j]^2$ in this expression.
- *Effect:* Forces each dimension to have non-trivial variance (prevents collapse to a point)

    **3. Covariance Loss (Decorrelation):**

    $$
    \mathcal{L}_{\text{cov}} = \frac{1}{d} \sum_{i \neq j} [\text{Cov}(z)]_{ij}^2
    $$
    - *Effect:* Forces off-diagonal covariance to zero (decorrelates dimensions)

    **Combined VICReg Loss:**

$$
\mathcal{L}_{\text{VICReg}} = \lambda \mathcal{L}_{\text{inv}} + \mu \mathcal{L}_{\text{var}} + \nu \mathcal{L}_{\text{cov}}
$$
Units: $\lambda,\mu,\nu$ are dimensionless weights; each component loss is taken dimensionless (nats after normalization).

**Comparison: InfoNCE vs VICReg vs Barlow Twins:**

| Method           | Negative Samples       | Collapse Prevention        | Computation | Citation                  |
|------------------|------------------------|----------------------------|-------------|---------------------------|
| **InfoNCE**      | Required ($B^2$ pairs) | Contrastive pushing        | $O(B^2 Z)$  | {cite}`oord2018cpc`       |
| **VICReg**       | None                   | Variance constraint        | $O(B Z^2)$  | {cite}`bardes2022vicreg`  |
| **Barlow Twins** | None                   | Cross-correlation identity | $O(B Z^2)$  | {cite}`zbontar2021barlow` |

**When to Use Which:**
- **InfoNCE:** When you have large batches and care about discriminative features
- **VICReg:** When you want geometric constraints without mining hard negatives
- **Barlow Twins:** When you want redundancy reduction (information-theoretic)

::::{note} Connection to RL #29: Contrastive RL as Degenerate InfoNCE Anchoring
**The General Law (Fragile Agent):**
**InfoNCE** anchors the latent space to capture long-term structural dependencies:

$$
\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp(\text{sim}(z_t, z_{t+k})/\tau)}{\sum_j \exp(\text{sim}(z_t, z^-_j)/\tau)}
$$
This is one of *multiple* anchoring signals in the Fragile Agent, applied specifically to the **macro channel** $K$ to ensure slow features dominate over fast texture.

**The Degenerate Limit:**
Use InfoNCE as the *primary* representation objective rather than auxiliary anchoring. No macro-micro split—all features treated uniformly.

**The Special Case (Standard RL):**

$$
I(z_t; z_{t+k}) \ge \log N - \mathcal{L}_{\text{CPC}}
$$
This recovers **Contrastive Predictive Coding (CPC)** {cite}`oord2018cpc` and **Contrastive RL** methods.

**What the generalization offers:**
- **Macro-micro split**: InfoNCE anchors the macro channel $K$; texture $z_{\text{tex}}$ is separate (Section 2.2b)
- **Multiple anchoring signals**: InfoNCE + VICReg + disentanglement losses work together (Table above)
- **Structural filtering**: Slow features → $K$; fast features → $z_n$, $z_{\text{tex}}$
- **Audit-friendly**: Node 6 (CollapseCheck) monitors whether contrastive loss is preventing collapse
::::

*   **Whitening / Orthogonality (Node 6 — Identifiability):**

    $$
    \mathcal{L}_{\text{orth}} = \lVert \operatorname{Cov}(z) - I \rVert_F^2 \quad \text{or} \quad \lVert J_S^T J_S - I \rVert^2
    $$
    *   *Effect:* Removes redundant/degenerate directions in the representation. Approximate whitening makes the latent coordinates identifiable up to permutation/sign, improves conditioning, and reduces collapse/exploding-gradient pathologies.
    Units: $\mathcal{L}_{\text{orth}}$ is dimensionless; the weight multiplying it is dimensionless.

(sec-b-world-model-regulation)=
#### B. World Model Regulation (Dynamics Model)
*   **Lipschitz Constraint (BarrierOmin / Node 9):**

    $$
    \mathcal{L}_{\text{Lip}} = \mathbb{E}_{z, z'}[(\lVert S(z) - S(z')\rVert / \lVert z - z'\rVert - K)^+]^2
    $$
    Or via Spectral Normalization on weights.
    *   *Effect:* Enforces **tameness**: bounds sensitivity of the learned dynamics and reduces non-smooth / ill-conditioned rollouts that destabilize planning and control.
*   **Forward Consistency (Node 5):**

    $$
    \mathcal{L}_{\text{pred}} = \lVert S(z_t, a_t) - z_{t+1} \rVert^2
    $$
    *   *Effect:* Standard dynamics learning, but constrained by the Lyapunov potential (see below).
*   **Symplectic / Hamiltonian parameterization (Node 21; optional).**
    If the latent state is organized as canonical coordinates $z=(q,p)\in\mathbb{R}^{2n}$, a structured world model can be parameterized by a learned Hamiltonian $H_\psi(q,p,a)$. Hamiltonian dynamics take the form

    $$
    \dot q = \nabla_p H_\psi(q,p,a),
    \qquad
    \dot p = -\nabla_q H_\psi(q,p,a).
    $$
    Equivalently, $\dot z = J\nabla_z H_\psi(z,a)$ for the canonical symplectic matrix $J$. This induces a divergence-free flow in $z$ and supports stable long-horizon rollouts when the environment is approximately conservative in the chosen coordinates {cite}`greydanus2019hamiltonian`. For a discrete-time transition $S$, one can monitor (or penalize) departures from symplecticity via

    $$
    \mathcal{L}_{\text{symp}}
    :=
    \left\|J_S^\top J J_S - J\right\|_F^2,
    \qquad
    J_S := \frac{\partial S(z,a)}{\partial z}.
    $$
    This is optional: if the environment is strongly dissipative or control-dominated, forcing symplectic structure can be counterproductive.
*   **Residual-event (jump) codebook (optional).**
    To separate "modeled dynamics" from "unmodeled disturbance", maintain a discrete codebook over one-step residuals

    $$
    \Delta z_{n,t} := z_{n,t+1}-S_n(z_{n,t},K_t,a_t),
    \qquad
    J_t := \mathrm{VQ}(\Delta z_{n,t})\in\{1,\dots,|\mathcal{J}|\}.
    $$
    Here $z_n$ is the **structured nuisance** coordinate (Section 2.2b). Texture $z_{\mathrm{tex}}$ is explicitly not used to form jump types: it is treated as an emission residual for reconstruction/likelihood, not as a disturbance class for dynamics. The resulting index $J_t$ provides an online-codable label for recurring disturbance types and supports conditional noise modeling (e.g., a mixture model for nuisance residuals). Section 11.5 shows how the same idea can be lifted to operator-valued belief updates, where discrete residual types parameterize jump operators.

(sec-c-critic-regulation)=
#### C. Critic Regulation (Value / Lyapunov Function)

The Critic does not just predict reward; it defines a **stability-oriented potential** over latent state. We impose Lyapunov-style constraints as *sufficient conditions* for local stability, enforced approximately via sampled penalties {cite}`chang2019neural,chow2018lyapunov,kolter2019safe`.

*Forward reference (Field Solver Interpretation).* Section 24 provides a deeper interpretation: the Critic is a **Field Solver** that propagates boundary reward charges into the bulk via the **Screened Poisson Equation** (Theorem {prf:ref}`thm-the-hjb-helmholtz-correspondence`). The Value function $V(z)$ is the Green's function of the screened Laplacian (Proposition {prf:ref}`prop-green-s-function-interpretation`), with the discount factor determining the screening length. This Helmholtz PDE perspective unifies the Lyapunov constraints below with the geometric regularization in Section 24.5.

**Euclidean vs Riemannian Critic Losses:**

| Loss Type        | Euclidean (Standard)                                               | Riemannian (Lyapunov)                                                         |
|------------------|--------------------------------------------------------------------|-------------------------------------------------------------------------------|
| **Primary**      | $\mathcal{L} = \lVert V_{\text{pred}} - V_{\text{target}}\rVert^2$ | $\mathcal{L}_{\text{Lyap}} = \mathbb{E}[\max(0, \dot{V}(z) + \alpha V(z))^2]$ |
| **Goal**         | Accuracy                                                           | Stability-oriented constraint                                                 |
| **Failure Mode** | Flat plateaus, irregular value surfaces                                   | Mitigated                                                                     |
| **Geometry**     | Ignores curvature                                                  | Encourages a well-conditioned potential                                       |

*   **Projective (bounded) value head (optional; objective gauge robustness).**
    If the dominant instability comes from value-scale drift (objective gauge $G_{\text{obj}}$; Section 1.1.4), parameterize the critic so its *state-dependent* output is bounded and scale-free. One implementable pattern is:

    $$
    u(z):=\frac{\phi(z)}{\|\phi(z)\|+\epsilon},
    \qquad
    \omega:=\frac{\tilde \omega}{\|\tilde \omega\|+\epsilon},
    \qquad
    V(z):=V_{\mathrm{scale}}\,(1-u(z)\cdot \omega),
    $$
    where $\phi$ is a learned embedding and $\tilde\omega$ is a learned goal direction. The dot product is dimensionless; $V_{\mathrm{scale}}$ carries units of nats (Section 1.2) and can be calibrated or learned via adaptive multipliers (Section 3.5).

    This does **not** make the entire RL pipeline invariant to arbitrary reward rescaling by itself (targets still change under $r\mapsto ar+b$), but it bounds critic outputs and makes the *directional part* of the value function less sensitive to magnitude drift.

*   **Lyapunov Decay (Node 7 - Stiffness):**
    Enforce a sampled Lyapunov decrease condition (a sufficient stability surrogate):

    $$
    \mathcal{L}_{\text{Lyapunov}} = \mathbb{E}_{z} [\max(0, \dot{V}(z) + \alpha V(z))^2]
    $$
    * *Mechanism:* Penalize states where the estimated decrease $\dot{V}$ is not sufficiently negative (relative to rate $\alpha$). This encourages $V$ to decrease along trajectories in regions the agent visits.

*   **Eikonal-style Gradient Regularization (BarrierGap - Geometric Constraint):**

    $$
    \mathcal{L}_{\text{Eikonal}} = (\lVert\nabla_z V\rVert - 1)^2
    $$
    * *Effect:* Encourages distance-like scaling of $V$ and mitigates exploding/vanishing gradients. It does not, by itself, guarantee that $V$ is an exact geodesic distance without additional conditions (e.g. boundary conditions and regularity).

*   **Lyapunov Stiffness (Node 7):**

    $$
    \mathcal{L}_{\text{Stiff}} = \max(0, \epsilon - \lVert\nabla V(z)\rVert)^2 + \lVert\nabla V(z)\rVert^2_{\text{reg}}
    $$
    *   *Effect:* The gradient $\nabla V$ must be non-zero (to drive the policy) but bounded (to prevent explosion).
*   **Safety Budget (Node 1):**

    $$
    \mathcal{L}_{\text{Risk}} = \lambda_{\text{safety}} \cdot \mathbb{E}[\max(0, V(z) - V_{\text{limit}})]
    $$
    *   *Effect:* Hard Lagrangian enforcement of the risk budget.

(sec-d-policy-regulation)=
#### D. Policy Regulation (Controller / Geometry-Aware Updates)

The Policy is the controller. Its objective is to choose actions that reduce expected cost while respecting stability and information constraints. We replace purely Euclidean policy-gradient updates with a **natural-gradient / information-geometric** update that respects the local sensitivity metric $G$ (Section 2.5).

**Euclidean vs Riemannian Policy Losses:**

| Loss Type                   | Euclidean (Standard)                            | Geometry-aware (Natural)                                                                                  |
|-----------------------------|-------------------------------------------------|-----------------------------------------------------------------------------------------------------------|
| **Primary**                 | $\mathcal{L} = -\log \pi(a\mid z) \cdot A(z,a)$ | $\mathcal{L}_{\text{nat}} = -\mathbb{E}\left[\frac{\nabla_z V(z) \cdot f(z, a)}{\sqrt{G_{ii}(z)}}\right]$ |
| **What it maximizes**       | Advantage (scalar)                              | Value-decrease rate normalized by $G$                                                                     |
| **Geometry**                | Ignores local conditioning                      | Uses Fisher/Hessian sensitivity metric $G$                                                                |
| **Ill-conditioned regions** | Aggressive steps can destabilize                | Geometry-scaled steps are conservative                                                                    |
| **Mechanism**               | Push toward high reward                         | Push along manifold                                                                                       |

*   **Value-Decrease Maximization (Node 10 — Natural Gradient):**

    $$
    \mathcal{L}_{\text{nat}} = -\mathbb{E}_{z, a \sim \pi} \left[ \frac{\nabla_z V(z) \cdot f(z, a)}{\sqrt{G_{ii}(z)}} \right]
    $$
    * *Mechanism:* Maximize the alignment between the value gradient $\nabla V$ and the realized dynamics $f(z,a)$, normalized by the local sensitivity scale ($G_{ii}$).
    * *Effect:* Where the metric indicates high sensitivity or ill-conditioning (large $G$), effective steps shrink; where it is well-conditioned (small $G$), steps can be larger.

*   **Hodge-style alignment (optional; complements Node 10 and BarrierBode).**
    View the policy-induced state change as a vector field on latent space (either the true environment dynamics $f$ or the world-model prediction $S(z,a)-z$). A simple alignment surrogate encourages the task-relevant component of the flow to be gradient-like:

    $$
    g(z):= -G^{-1}(z)\nabla_z V(z),
    \qquad
    \Delta z := S(z_t,a_t)-z_t,
    \qquad
    \mathcal{L}_{\text{Hodge}} := 1-\cos(\Delta z,\ g(z_t)).
    $$
    This does not remove exploration; rather it penalizes large *solenoidal* (looping) components when they produce oscillatory instability (Mode D.E). It is most appropriate when $V$ is well-shaped and the world model is reliable on-policy.

*   **Geodesic Stiffness (Node 2 - Zeno Constraint):**

    $$
    \mathcal{L}_{\text{Zeno}} = \lVert\pi_t - \pi_{t-1}\rVert^2_{G}
    $$
    * *Effect:* Penalizes high-frequency switching, weighted by geometry. Switching is penalized more strongly in regions where the metric indicates high sensitivity (large $G$).

*   **Standard Zeno Constraint (Euclidean fallback):**

    $$
    \mathcal{L}_{\text{Zeno}}^{\text{Euc}} = D_{\mathrm{KL}}(\pi(\cdot \mid z_t) \Vert \pi(\cdot \mid z_{t-1}))
    $$
    *   *Effect:* Penalizes high-frequency action switching (chattering).
*   **Entropy Regularization (Node 10):**

    $$
    \mathcal{L}_{\text{Ent}} = -\mathcal{H}(\pi(\cdot \mid z))
    $$
    *   *Effect:* Prevents premature collapse to deterministic policies (BarrierMix).

(sec-e-cross-network-synchronization)=
#### E. Cross-Network Synchronization (Alignment Terms)
A key design choice in the Fragile Agent is to make inter-component alignment explicit via **synchronization losses**:

1.  **Shutter $\leftrightarrow$ WM (Macro Closure / Predictability):**
    *   The shutter is not merely compressing $x_t$; it is defining the **macro-effective ontology** $K_t\in\mathcal{K}$ on which the World Model claims to be Markov (Causal Enclosure; Section 2.8).

        $$
        \mathcal{L}_{\text{Sync}_{K-W}} = \mathrm{CE}\!\left(K_{t+1},\ \hat{p}_\phi(K_{t+1}\mid K_t, A_t)\right)
        $$
    *   *Meaning:* If the shutter emits macrostates that the WM cannot predict (large closure cross-entropy), then the ontology is inconsistent: either the symbol inventory is unstable (codebook churn) or the WM class is misspecified (Mode D.C / T.E).

2.  **Critic $\leftrightarrow$ Policy (Audit / Advantage Gap):**
    *   The Critic is the risk auditor. If the Policy acts in a way the Critic didn't anticipate, there is a control gap.

        $$
        \mathcal{L}_{\text{Sync}_{V-\pi}} = \lVert V(z) - (r + \gamma V(z')) \rVert^2 \quad (\text{TD-Error})
        $$
    *   *Critically:* We track the **Advantage Gap** $\Delta A = |A^{\pi}(s, a) - A^{\text{Buffer}}(s, a)|$. If $\Delta A$ grows, the policy has drifted off-manifold (BarrierTypeII).

3.  **WM $\leftrightarrow$ Policy (Control-Awareness):**
    *   The WM should allocate capacity where the Policy visits (On-Policy dynamics).

        $$
        \mathcal{L}_{\text{Sync}_{W-\pi}} = \mathbb{E}_{z \sim \pi} [\mathcal{L}_{\text{pred}}(z)]
        $$
    *   *Meaning:* Accuracy on the *optimal path* matters more than global accuracy.

(sec-f-exploration-and-coupling-regularizers)=
#### F. Exploration and Coupling Regularizers (Path Entropy, KL-Control, Window)
The synchronization and component losses above enforce internal consistency. The following regularizers make information/coupling constraints explicit in online-auditable form.

*   **KL-Control (Relative-Entropy Control; Theorem {prf:ref}`thm-equivalence-of-entropy-regularized-control-forms-discrete-macro`).** Fix a reference actuator prior $\pi_0(a\mid k)$ with full support. Define control effort as KL deviation from this prior:

    $$
    \mathcal{L}_{\text{KL-ctrl}}
    :=
    T_c\,\mathbb{E}_{K_t}\!\left[D_{\mathrm{KL}}\!\left(\pi(\cdot\mid K_t)\ \Vert\ \pi_0(\cdot\mid K_t)\right)\right].
    $$
    When $\pi_0$ is uniform, this reduces (up to an additive constant) to standard entropy regularization; when $\pi_0$ encodes actuator limits, it becomes a calibrated control-effort penalty.

*   **Path-Entropy Exploration (Future Flexibility; Definition 10.1.2).** Encourage non-degenerate reachable macro futures by maximizing causal path entropy (equivalently minimizing its negative):

    $$
    \mathcal{L}_{\text{expl}}
    :=
    -\sum_{h=1}^{H} w_h\,S_c(K_t,h;\pi),
    $$
    with weights $w_h\ge 0$. In practice, a computable proxy is the entropy of the WM-predicted horizon marginals $\hat{P}_\phi(K_{t+h}\mid K_t)$ obtained by rollout or dynamic programming.

*   **Information–Stability Window (Theorem {prf:ref}`thm-information-stability-window-operational`).** Penalize both under-coupling (loss of grounding) and over-coupling (symbol dispersion / saturation). With thresholds $0<\epsilon<\log|\mathcal{K}|$,

    $$
    \mathcal{L}_{\text{window}}
    :=
    \mathrm{ReLU}\!\big(\epsilon - I(X_t;K_t)\big)^2
    +
    \mathrm{ReLU}\!\big(H(K_t)-(\log|\mathcal{K}|-\epsilon)\big)^2.
    $$
    This is an explicit online enforcement of the coupling window: $I(X;K)$ must not collapse, and $H(K)$ must not saturate.

*   **Regularized Objective Descent (Sections 9.11 and 11–14).** Define an instantaneous (per-step) regularized objective

    $$
    F_t
    :=
    V(Z_t)
    + \beta_K\big(-\log p_\psi(K_t)\big)
    + \beta_n D_{\mathrm{KL}}\!\left(q(z_{n,t}\mid x_t)\ \Vert\ p(z_n)\right)
    + \beta_{\mathrm{tex}} D_{\mathrm{KL}}\!\left(q(z_{\mathrm{tex},t}\mid x_t)\ \Vert\ p(z_{\mathrm{tex}})\right)
    + T_c D_{\mathrm{KL}}\!\left(\pi(\cdot\mid K_t)\ \Vert\ \pi_0(\cdot\mid K_t)\right),
    \qquad \text{where } Z_t=(K_t,z_{n,t},z_{\mathrm{tex},t}).
    $$
    A monotonicity surrogate is then enforced by

    $$
    \mathcal{L}_{\downarrow F}
    :=
    \mathbb{E}\!\left[\mathrm{ReLU}\!\left(F_{t+1}-F_t\right)^2\right],
    $$
    optionally applied only inside the safety budget (Node 1 / CostBoundCheck) to avoid suppressing necessary exploration.

(sec-joint-optimization)=
### 3.4 Joint Optimization
The total Fragile Agent training objective is the weighted sum of component and synchronization tasks:

$$
\mathcal{L}_{\text{Fragile}} = \mathcal{L}_{\text{Task}} + \sum \lambda_i \mathcal{L}_{\text{Self-Reg}_i} + \sum \lambda_{ij} \mathcal{L}_{\text{Sync}_{ij}}
$$
This defines the coupled-system "stiffness". In practice, the coefficients $\lambda$ should be treated as **adaptive multipliers** (not fixed constants): different constraints become active at different times, and gradient scales drift as representation and policy change (Section 3.5). If $\lambda_{\text{Sync}}$ is too low, components drift out of alignment; if it is too high, optimization becomes over-regularized and can stall (BarrierBode).

(sec-adaptive-multipliers-learned-penalties-setpoints-and-calibration)=
### 3.5 Adaptive Multipliers: Learned Penalties, Setpoints, and Calibration

In the Fragile Agent, **static loss weights are a failure mode**: hard-coding numbers like $\lambda=0.1$ implicitly assumes a constant exchange rate between heterogeneous terms even though their typical magnitudes and gradients change across training, operating regimes, and distribution shift.

We distinguish three classes of coefficients:
- **Dual multipliers (constraints):** enforce nonnegotiable inequalities (Gate Nodes / Barriers).
- **Setpoint controllers (regulators):** maintain a metric near a target (entropy, KL-per-update, code usage), rather than driving it to zero.
- **Learned precisions (multi-task scaling):** balance likelihood-style losses with unknown noise scales (reconstruction vs prediction vs auxiliary SSL).

(sec-method-a-primal-dual-updates)=
#### 3.5.1 Method A: Primal–Dual Updates (Projected Dual Ascent)

Choose online-computable nonnegative constraint metrics $\mathcal{C}_i(\theta)$ and tolerances $\epsilon_i$ defining a feasible set

$$
\mathcal{C}_i(\theta)\le \epsilon_i,
\qquad i=1,\dots,m.
$$
Examples include enclosure defects (Section 2.8), Zeno/step-size limits (Node 2), saturation measures (BarrierSat), and Lyapunov defects (Node 7).

Define the Lagrangian (units consistent with Section 1.2):

$$
\mathcal{L}(\theta,\lambda)
=
\mathcal{L}_{\text{Task}}(\theta)
\;+\;
\sum_{i=1}^{m}\lambda_i\big(\mathcal{C}_i(\theta)-\epsilon_i\big),
\qquad
\lambda_i\ge 0.
$$
**Online algorithm (two-step loop).**
1. **Primal step (agent):** update $\theta$ to reduce $\mathcal{L}(\theta,\lambda)$ using your standard optimizer.
2. **Dual step (multipliers):** increase pressure on violated constraints:

   $$
   \lambda_i
   \leftarrow
   \Pi_{[0,\lambda_{\max}]}\!\left(\lambda_i + \eta_{\lambda}\,(\mathcal{C}_i(\theta)-\epsilon_i)\right),
   $$
   where $\Pi$ is projection/clipping and $\eta_\lambda$ is a small dual step size.

**Implementation notes.**
- Compute $\mathcal{C}_i$ on the same batch as the primal loss; use `detach()` for the dual update so gradients do not flow into $\theta$ through $\lambda$.
- Use a separate optimizer (often much slower than the primal optimizer) and cap $\lambda$; if $\lambda_i$ repeatedly hits $\lambda_{\max}$, treat it as an **unsatisfied hard constraint** and halt or change architecture rather than silently continue.

```python
# Sketch: projected dual ascent with detached violations
violations = {name: (C - eps[name]).detach() for name, C in C_values.items()}
for name, v in violations.items():
    lambda_val[name] = torch.clamp(lambda_val[name] + eta_lambda[name] * v, 0.0, lambda_max[name])
```

This is the same mathematical pattern used to tune entropy coefficients or KL constraints in modern RL (e.g. SAC-style automatic entropy tuning) {cite}`haarnoja2018soft`.

(sec-method-b-setpoint-controllers)=
#### 3.5.2 Method B: Setpoint Controllers (PI/PID Regulation)

Some metrics should be regulated around a target value or rate. Typical examples:
- **Policy KL per update** (trust region): keep $D_{\mathrm{KL}}(\pi_t\Vert\pi_{t-1})$ in a target band.
- **Entropy / mixing:** keep $H(\pi(\cdot\mid K))$ within a target range.
- **Code usage:** keep $H(K)$ away from collapse and away from saturation.

Let $m_t$ be a measured scalar metric and $m^\star$ its target. Define the error $e_t := m^\star - m_t$. A discrete PID update for a positive coefficient $\lambda$ is:

$$
\lambda_{t+1}
=
\Pi_{[\lambda_{\min},\lambda_{\max}]}\!\Big(
\lambda_t
+
K_p e_t
+
K_i \sum_{t' \le t} e_{t'}
+
K_d(e_t-e_{t-1})
\Big).
$$
**Sign discipline.** Choose the loss term so that increasing $\lambda$ pushes the metric in the desired direction. For example, if you want higher entropy when it is too low, include $\lambda_{\text{ent}}\,(-H(\pi))$ in the minimized loss: increasing $\lambda_{\text{ent}}$ increases the incentive to raise $H$.

In practice, PI control (no derivative term) is often sufficient; add derivative damping only if oscillations are observed.

This “multiplier as controller” viewpoint is used directly in PID Lagrangian methods for constrained RL {cite}`stooke2020responsive`.

(sec-method-c-learned-precisions)=
#### 3.5.3 Method C: Learned Precisions (Homoscedastic Uncertainty Weighting)

When combining multiple likelihood-style losses with different natural scales (e.g., reconstruction vs dynamics prediction vs auxiliary self-supervision), it is often better to learn their relative weights as **inverse variances** (precisions) rather than choose them by hand.

Assume each loss $\mathcal{L}_i$ is (or is proportional to) a negative log-likelihood with unknown homoscedastic noise variance $\sigma_i^2$. Learning $s_i:=\log\sigma_i^2$ yields the objective {cite}`kendall2018multi`:

$$
\mathcal{L}_{\text{total}}
=
\sum_i \frac12\Big(\exp(-s_i)\,\mathcal{L}_i + s_i\Big).
$$
The effective weight is {math}`\exp(-s_i)`, and the {math}`s_i` term prevents degenerate solutions where all weights collapse to zero.

This method is appropriate for **multi-task scaling**, not for nonnegotiable safety constraints (use Method A for those).

(sec-recommended-mixing)=
#### 3.5.4 Recommended Mixing (Practical Policy)

| Term type                           | Examples in this document                                          | Mechanism                     |
|-------------------------------------|--------------------------------------------------------------------|-------------------------------|
| **Objective anchor**                | task loss / return surrogate                                       | fixed scale (e.g., 1.0)       |
| **Hard constraints**                | enclosure/closure, Lyapunov defects, saturation, budget exceedance | Method A (primal–dual)        |
| **Setpoints / regulators**          | entropy targets, KL-per-update target, code usage target           | Method B (PI/PID)             |
| **Multi-task likelihood balancing** | recon vs prediction vs auxiliary SSL losses                        | Method C (learned precisions) |

(sec-calibrating-tolerances)=
#### 3.5.5 Calibrating Tolerances $\epsilon_i$ (Feasibility and Units)

Dual methods only work if constraints are **feasible**: setting $\epsilon_i$ below the system’s achievable resolution forces multipliers to diverge and produces brittle training.

A practical, implementable calibration procedure is:
1. **Collect a calibration buffer** $\mathcal{D}_{\text{cal}}$ by running a baseline policy for $N$ steps (random, scripted, or a known-safe controller), logging the metrics used in your Gate Nodes / Barriers.
2. **Estimate achievable baselines** by computing empirical summaries for each metric (median, quantiles, MAD).
3. **Set tolerances** using quantiles plus a margin:

   $$
   \epsilon_i := Q_p\!\big(\mathcal{C}_i(\mathcal{D}_{\text{cal}})\big) + \Delta_i,
   $$
   with $p$ chosen by strictness (e.g. $p=0.9$ for "usually satisfied", $p=0.99$ for "almost always satisfied").

**Calibration phases (practical).**
1. **Empirical floors (data stream):** for losses tied to prediction/reconstruction, estimate what is achievable on $\mathcal{D}_{\text{cal}}$ with a low-capacity baseline or an ensemble (a proxy for irreducible/aleatoric error).
2. **Architecture bounds:** for discrete/finite-capacity objects, compute tolerances analytically (e.g., code usage, sampling noise floors).
3. **Requirements:** for safety budgets (risk, decay rates), set $\epsilon$ from task-level specifications.

| Constraint metric $\mathcal{C}_i$                                |         Units | Example tolerance choice $\epsilon_i$                           | Notes                                                                          |
|------------------------------------------------------------------|--------------:|-----------------------------------------------------------------|--------------------------------------------------------------------------------|
| Reconstruction NLL / distortion                                  |           nat | $Q_{0.9}(\mathcal{L}_{\text{recon}}(\mathcal{D}_{\text{cal}}))$ | Prefer likelihood losses (nats); if using MSE, fix/learn the scale (Method C). |
| One-step prediction NLL                                          |      nat/step | $Q_{0.9}(\mathcal{L}_{\text{pred}}(\mathcal{D}_{\text{cal}}))$  | Use an ensemble baseline to separate reducible vs irreducible error.           |
| Macro closure defect (e.g. $H(K_{t+1}\!\mid K_t,a_t)$ surrogate) |           nat | baseline Markov predictor + margin                              | Prevents “macro depends on micro” failure (Section 2.8).                       |
| KL-per-update (trust region)                                     |           nat | $\epsilon_{\text{KL}}\approx c/B$                               | Sampling noise scales as $O(1/B)$ for batch size $B$.                          |
| Code usage gap $\log\lvert\mathcal{K}\rvert-H(K)$                |           nat | $-\log(1-\rho_{\text{dead}})$                                   | Purely architectural.                                                          |
| Numerical residuals (orthogonality, symmetry)                    | dimensionless | $\approx 10^{-6}$ (float32)                                     | Treat as a numeric floor, not a learnable target.                              |
| Lyapunov decay margin                                            |   step$^{-1}$ | $1/T_{\text{stab}}$                                             | “Stabilize in $T_{\text{stab}}$ steps” requirement.                            |
| Risk/cost budget                                                 |           nat | $V_{\max}$ from spec                                            | If interpreted as log-risk, map probabilities via $-\log p$.                   |

**Architecture-derived tolerances (often better than environment guesses).**
- **Codebook usage.** If you allow a dead-code fraction $\rho_{\text{dead}}$, then “not too collapsed” can be stated as

  $$
  H(K)\ \ge\ \log\!\big((1-\rho_{\text{dead}})\,|\mathcal{K}|\big)
  \quad\Longleftrightarrow\quad
  \log|\mathcal{K}|-H(K)\ \le\ -\log(1-\rho_{\text{dead}}).
  $$
  With $\rho_{\text{dead}}=0.05$, the right-hand side is $\approx 0.051$ nats.
- **Sampling noise floors.** For per-update KL constraints estimated from batches, a typical noise scale is $O(1/B)$, so a conservative starting tolerance is $\epsilon_{\text{KL}}\approx c/B$ with $c\in[0.5,5]$ depending on variance.

**Safety budgets.** Budgets like $V_{\max}$ are part of the task specification; if $V$ is interpreted as a log-risk surrogate, then requirements on survival probability can be mapped to nats via $-\log p$ (Section 1.2).

(sec-using-scaling-exponents-to-gate-updates-and-tune-step-sizes)=
#### 3.5.6 Using Scaling Exponents to Gate Updates and Tune Step Sizes

The scaling exponents $(\alpha,\beta,\gamma,\delta)$ (Section 3.2) become actionable when treated as **online diagnostics** driving a simple update scheduler {cite}`konda2000actor`:
- If representation drift $\delta$ is high, freeze downstream learning (policy/critic/world) until the shutter stabilizes.
- If world-model volatility $\gamma$ is high, avoid policy learning on shifting dynamics (freeze or reduce policy step size).
- If the policy update scale $\beta$ exceeds critic signal strength $\alpha$ (BarrierTypeII), skip policy updates until the critic recovers.

One implementable pattern is a “gate + ratio” rule with EMA-smoothed exponents:
```python
# Sketch: gate policy updates if actor outruns critic
alpha = ema(alpha)          # critic signal / curvature proxy
beta  = ema(beta_kl)        # mean KL(π_t || π_{t-1}) per update
gamma = ema(world_drift)    # WM parameter drift proxy
delta = ema(code_drift)     # codebook/encoder drift proxy

if delta > delta_max:
    lr_policy = 0.0
    lr_world *= 0.5
    lr_critic *= 0.5
elif gamma > gamma_max:
    lr_policy = 0.0
elif beta > min(beta_max, alpha):
    lr_policy *= 0.98
    lr_critic *= 1.02
```

This is not ad-hoc tuning; it is a direct operationalization of the two-time-scale requirement already encoded as BarrierTypeII (Section 4.1.2).

:::{note} Neural Unification (Section 26)
:class: seealso

The three adaptive multiplier methods above (Primal–Dual, PID, Learned Precisions) are **special cases** of a more general neural meta-controller. Section 26 introduces the **Universal Governor** $\pi_{\mathfrak{G}}$, which learns a temporal policy over the diagnostic stream $s_t = [C_1(\theta_t), \ldots, C_K(\theta_t)]$ and outputs all hyperparameters $\Lambda_t = (\eta_t, \vec{\lambda}_t, T_{c,t})$ jointly:

- **Primal–Dual (Method A)** = affine policy, memoryless ($H=0$)
- **PID (Method B)** = linear temporal filter with hand-tuned $(K_p, K_i, K_d)$
- **Learned Precisions (Method C)** = diagonal covariance, no temporal processing

The Governor subsumes these by learning the appropriate response to each diagnostic signature via bilevel optimization. See Section 26 for stability guarantees via Lyapunov analysis.
:::



(sec-limits-barriers)=
