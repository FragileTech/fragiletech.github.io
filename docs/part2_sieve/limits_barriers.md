## 4. Limits: Barriers (The Limits of Control)

:::{admonition} Researcher Bridge: Barriers vs. Trust Regions
:class: warning
:name: rb-barriers-trust-regions
Standard RL uses trust regions, clipping, or penalty terms to avoid instability. Barriers are the formal limit surfaces those heuristics approximate. When a barrier activates, the correct response is to halt, project, or reshape updates rather than incur a soft penalty.
:::

Barriers represent the fundamental limits of the control loop.

| Barrier ID         | Name                    | Bottleneck        | Limit                             | Mechanism                                                                                     | Regularization Factor ($\mathcal{L}_{\text{barrier}}$)                                                             | Compute        |
|--------------------|-------------------------|-------------------|-----------------------------------|-----------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|----------------|
| **BarrierSat**     | Saturation              | **Policy**        | **Actuator Saturation**           | Policy cannot output enough control authority to counter disturbance.                         | $\Vert \pi(s) \Vert < F_{\text{max}}$ (Soft Clipping)                                                              | $O(BA)$ ✓      |
| **BarrierCausal**  | Causal Censor           | **World Model**   | **Computational Horizon**         | Failure happens faster than WM can predict/compute.                                           | $T_{\text{horizon}}$ (Discount Factor $\gamma < 1$)                                                                | $O(1)$ ✓       |
| **BarrierScat**    | Representation Collapse | **VQ-VAE**        | **Grounding Loss**                | Symbol channel loses grounding; macrostates become noise-like.                                | $\mathrm{ReLU}(\epsilon-I(X;K))^2 + \mathrm{ReLU}(H(K)-(\log\lvert\mathcal{K}\rvert-\epsilon))^2$ (Window Penalty) | $O(B)$ ✓       |
| **BarrierTypeII**  | Type II Exclusion       | **Critic/Policy** | **Scaling Mismatch**              | $\beta>\alpha$ (Policy update scale outruns critic signal).                                   | $\max(0, \beta - \alpha)$ (Scaling Penalty)                                                                        | $O(P)$ ⚡       |
| **BarrierVac**     | Model Stability Limit   | **World Model**   | **Regime Stability**              | Operational mode is metastable; WM predicts collapse.                                         | $\Vert \nabla^2 V(z) \Vert$ (Hessian Regularization)                                                               | $O(BZ^2)$ ✗    |
| **BarrierCap**     | Capacity                | **Policy**        | **Fundamental Uncontrollability** | Unsafe region is too large for Policy to steer around.                                         | $V(z) \to \infty$ for $z \in \text{Bad}$ (Safe RL)                                                                 | $O(B)$ ⚡       |
| **BarrierGap**     | Spectral Gap            | **Critic**        | **Convergence Stagnation**        | Error surface is too flat ($\nabla V \approx 0$).                                             | $\max(0, \epsilon - \Vert \nabla V \Vert)$ (Stiffness)                                                             | $O(BZ)$ ✓      |
| **BarrierAction**  | Action Gap              | **Critic**        | **Cost Prohibitive**              | Correct move requires more cost budget ($V$) than affordable.                                 | $\Vert \nabla_\pi V(s, \pi) \Vert$ (Action Gradient)                                                               | $O(BAZ)$ ⚡     |
| **BarrierOmin**    | O-Minimal               | **World Model**   | **Model Mismatch**                | World exhibits non-smooth or non-stationary structure outside the WM class.                   | $\Vert \nabla S_t \Vert$ for O-Minimality (Lipschitz)                                                              | $O(ZP_{WM})$ ⚡ |
| **BarrierMix**     | Mixing                  | **Policy**        | **Exploration Trap**              | Policy converges to a local minimum with insufficient state coverage.                                                            | $-H(\pi)$ (Entropy Bonus)                                                                                          | $O(BA)$ ✓      |
| **BarrierEpi**     | Epistemic               | **VQ-VAE/WM**     | **Information Overload**          | Environment complexity exceeds $\log\lvert\mathcal{K}\rvert$ and/or WM class; closure breaks. | $\mathcal{L}_{\text{recon}} + \mathcal{L}_{\text{Sync}_{K-W}}$ (Distortion + Closure)                              | $O(BD)$ ✓      |
| **BarrierFreq**    | Frequency               | **World Model**   | **Loop Instability**              | Positive feedback causes oscillation amplification.                                           | $\Vert J_{WM} \Vert < 1$ (Jacobian Spectral Norm)                                                                  | $O(Z^2)$ ✗     |
| **BarrierBode**    | Bode Sensitivity        | **Policy**        | **Waterbed Effect**               | Suppressing error in one domain increases it in another.                                      | $\int_{0}^{\infty} \log \lvert S(j\omega) \rvert d\omega = \text{const.}$ (Bode sensitivity integral)              | FFT ✗          |
| **BarrierInput**   | Input Stability         | **All**           | **Resource Exhaustion**           | Agent runs out of battery/compute/tokens.                                                     | $\text{Cost}(s) > \text{Budget}$ (Resource Penalty)                                                                | $O(B)$ ✓       |
| **BarrierVariety** | Requisite Variety       | **Policy**        | **Ashby's Deficit**               | Policy states < Disturbance states.                                                           | $\dim(Z) \ge \dim(\mathcal{X})$ (Width Penalty)                                                                    | $O(1)$ ✓       |
| **BarrierLock**    | Exclusion               | **World Model**   | **Hard-Coded Safety**             | Safety interlock successfully prevents illegal state.                                         | $\mathbb{I}(s \in \text{Forbidden}) \cdot \infty$                                                                  | $O(B)$ ✓       |

**Compute Legend:** ✓ Low (typically online) | ⚡ Moderate (often amortized/approximated) | ✗ High (often offline or coarse approximations)

(sec-barrier-implementation-details)=
### 4.1 Barrier Implementation Details

Implementing these barriers requires rigorous cybernetic engineering. We divide them into **Single-Barrier Limits** and **Cross-Barrier Dilemmas**.

(sec-a-single-barrier-enforcement)=
#### A. Single-Barrier Enforcement (Hard Constraints)

1.  **BarrierSat (Actuator Limit):**
    *   *Constraint:* $\lVert\pi(s)\rVert \le F_{\max}$.
    *   *Implementation:* **Squashing Function**. Use `tanh` on the policy mean: $\mu(z) = F_{max} \cdot \tanh(f_\theta(z))$. Do not rely on clipping losses alone; the architecture must be incapable of exceeding limits.

2.  **BarrierTypeII (Scaling Mismatch):**
    *   *Constraint:* $\alpha > \beta$ (Critic is steeper than Policy).
    *   *Implementation:* **Two-Time-Scale Updating**.
        *   If $\text{Scale}(\text{Critic}) \le \text{Scale}(\text{Policy})$, **skip** the Policy update step ($k_\pi = 0$).
        *   Resume policy updates only when the Critic has re-established a valid gradient (restored a usable value landscape).

3.  **BarrierOmin (Tameness):**
    *   *Constraint:* $\lVert S\rVert_{\mathrm{Lip}} \le K$.
    *   *Implementation:* **Spectral normalization** bounds the operator norm of each linear layer. With 1-Lipschitz activations, this upper-bounds the network Lipschitz constant by the product of per-layer spectral norms; choose per-layer caps so the implied global bound is $\le K$ {cite}`miyato2018spectral`.

4.  **BarrierGap (Spectral Gap):**
    *   *Constraint:* $\lVert\nabla V\rVert \ge \epsilon$ (No flat plateaus).
    *   *Implementation:* **Gradient Penalty**.

        $$
        \mathcal{L}_{GP} = \mathbb{E}_{\hat{s}} [(\lVert\nabla_{\hat{s}} V(\hat{s})\rVert - K)^2]
        $$
        Gradient-norm penalties discourage vanishing gradients on sampled points and help avoid large flat regions; they do not provide a global guarantee without additional assumptions {cite}`gulrajani2017improved`.

(sec-b-cross-barrier-regularization)=
#### B. Cross-Barrier Regularization (Cybernetic Dilemmas)

The most dangerous failures occur when barriers conflict. We model these as **Trade-off Functionals**:

1.  **The Information-Control Tradeoff (BarrierScat vs BarrierCap):**
    *   *Classes:* **Rate-Distortion Optimization.**
    *   *Conflict:* High compression (anti-collapse) removes details needed for fine control (capacity/controllability).
    *   *Regularization:*

        $$
        \mathcal{L}_{\text{InfoControl}}
        =
        \underbrace{\beta_K\,\mathbb{E}[-\log p_\psi(K)] + \beta_n D_{\mathrm{KL}}(q(z_n \mid x)\Vert p(z_n)) + \beta_{\mathrm{tex}} D_{\mathrm{KL}}(q(z_{\mathrm{tex}} \mid x)\Vert p(z_{\mathrm{tex}}))}_{\text{Compression (Rate)}}
        +
        \underbrace{\gamma\,\mathbb{E}[\mathfrak{D}(Z,A)]}_{\text{Control Effort}}
        $$
        where {math}`\mathfrak{D}` is an actuation cost (e.g. KL-control to a prior {math}`\pi_0`, or a calibrated norm/penalty on actions).
    *   *Mechanism:* Use Lagrange multipliers to find the Pareto frontier. If control performance drops, decrease $\beta_K,\beta_n,\beta_{\mathrm{tex}}$ (allocate more bits to the shutter).

2.  **The Stability-Plasticity Dilemma (BarrierVac vs BarrierPZ):**
    *   *Conflict:* A stable World Model (model stability limit) resists updating to new dynamics (plasticity / Zeno).
    *   *Regularization:* **Elastic Weight Consolidation (EWC)**.

        $$
        \mathcal{L}_{\text{EWC}} = \sum_i F_i (\theta_i - \theta^*_{i,old})^2
        $$
    *   *Mechanism:* The Fisher Information Matrix $F_i$ quantifies parameter sensitivity. Updates are permitted for low-sensitivity weights while high-sensitivity (structurally important) weights are constrained.

3.  **The Sensitivity Integral (BarrierBode):**
    *   *Conflict:* Suppressing error in one frequency band amplifies it in another (Bode sensitivity integral constraint: $\int_{0}^{\infty} \log |S(j\omega)| d\omega = \text{const.}$; equal to $0$ under standard stable/minimum-phase assumptions).
    *   *Regularization:* **Frequency-Weighted Cost**.

        $$
        \mathcal{L}_{\text{Bode}} = \lVert \mathcal{F}(e_t) \cdot W(\omega) \rVert^2
        $$
    *   *Mechanism:* Explicitly decide *where* to be blind. We penalize high-frequency errors heavily (instability) while accepting low-frequency drift (steady-state error), or vice versa.



(sec-failure-modes)=
