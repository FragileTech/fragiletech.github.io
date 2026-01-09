## 5. Failure Modes (Observed Pathologies)

:::{admonition} Researcher Bridge: RL Pathologies, Named and Localized
:class: info
:name: rb-rl-pathologies
If you have seen mode collapse, oscillation, overfitting, or deadlock in RL, this table is the same landscape but made explicit. Each failure is tied to a component and a diagnostic signature, so it can be detected and corrected rather than discovered post hoc.
:::

When Limits are breached or Interfaces fail, the agent exhibits specific pathologies.

| Mode    | Standard Name       | Failed Component     | Fragile (Pathology) Name      | Description                                                                     |
|---------|---------------------|----------------------|-------------------------------|---------------------------------------------------------------------------------|
| **D.D** | Dispersion-Decay    | **All (Optimal)**    | **Success (Convergence)**     | Agent solves task; error drops to a stable floor.                               |
| **S.E** | Subcritical-Equilib | **Policy**           | **Curriculum Stumble**        | Task difficulty increases faster than adaptation rate.                          |
| **C.D** | Conc-Dispersion     | **Policy/Shutter**   | **Mode Collapse / Obsession** | Policy concentrates on a single mode, neglecting remaining state space.         |
| **C.E** | Conc-Escape         | **Policy/Critic**    | **Divergence / Blow-up**      | Gradients/activations diverge; optimization becomes unstable.                   |
| **T.E** | Topo-Extension      | **Shutter/WM**       | **Wrong Paradigm**            | Architecture is topologically insufficient.                                     |
| **S.D** | Struct-Dispersion   | **Shutter**          | **Symmetry Blindness**        | Fails to exploit available symmetries.                                          |
| **C.C** | Event Accumulation  | **Policy/WM**        | **Decision Paralysis**        | Input happens faster than decision loop (Zeno).                                 |
| **T.D** | Glassy Freeze       | **Policy**           | **Learned Helplessness**      | Policy converges to suboptimal fixed point with zero gradient.                  |
| **D.E** | Oscillatory         | **Policy**           | **Pilot-Induced Oscillation** | Overcorrection causes increasing instability.                                   |
| **T.C** | Labyrinthine        | **World Model**      | **Overfitting to Noise**      | WM models noise instead of signal.                                              |
| **D.C** | Semantic Horizon    | **Shutter/WM**       | **Ungrounded inference**      | Distribution shift causes internal rollouts to decouple from boundary evidence. |
| **B.E** | Sensitivity Expl.   | **Critic**           | **Fragility**                 | Optimization for a single condition induces high sensitivity to perturbations.  |
| **B.D** | Resource Depletion  | **Boundary/Shutter** | **Starvation**                | Input or power resources depleted.                                              |
| **B.C** | Control Deficit     | **Policy**           | **Overwhelmed**               | Disturbance more complex than controller (Ashby).                               |



(sec-interventions)=
## 6. Interventions (Mitigations)

:::{admonition} Researcher Bridge: Heuristic Fixes as Typed Surgeries
:class: tip
:name: rb-heuristic-fixes
These interventions correspond mathematically to common RL stabilizers: target networks, clipping, entropy tuning, replay, and resets. Each intervention is triggered by a specific diagnostic condition rather than manual hyperparameter tuning.
:::

Interventions are external mitigations to restore stability, re-ground the representation, or reduce unsafe update rates.

| Surgery ID     | Target Mode        | Target Component     | Fragile (Upgrade) Translation | Mechanism                                                                                                                                                                                                                                                                                                                               |
|----------------|--------------------|----------------------|-------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **SurgCE**     | C.E (Divergence)   | **Policy/Critic**    | **Limiter / trust region**    | **Gradient Clipping / Trust Region:** Clamp outputs; enforce $\Vert \pi_{new} - \pi_{old} \Vert < \delta$.                                                                                                                                                                                                                              |
| **SurgCC**     | C.C (Zeno)         | **WM/Policy**        | **Time-boxing / Rate Limit**  | **Skip-Frame / Latency:** Force fixed $\Delta t$; ignore inputs during cool-down.                                                                                                                                                                                                                                                       |
| **SurgCD_Alt** | C.D (Obsession)    | **Policy**           | **Reset / Reshuffling**       | **Re-initialization:** Reset parameters of the obsession-locked sub-module to random.                                                                                                                                                                                                                                                   |
| **SurgSE**     | S.E (Stumble)      | **World Model**      | **Curriculum Ease-off**       | **Curriculum Learning:** Reduce Task Difficulty or Rewind to earlier level.                                                                                                                                                                                                                                                             |
| **SurgSC**     | S.C (Instability)  | **Critic**           | **Parameter Freezing**        | **Target Network Freeze:** Stop updating Target V; switch to slower exponential moving average.                                                                                                                                                                                                                                         |
| **SurgCD**     | C.D (Collapse)     | **Shutter**          | **Feature Pruning**           | **Dead Code Pruning:** Identify and excise unused macro symbols / dead fibres.                                                                                                                                                                                                                                                          |
| **SurgSD**     | S.D (Blindness)    | **Shutter**          | **Augmentation / Ghost Vars** | **Domain Randomization:** Inject noise into $x$ to force the shutter to learn robust macrostates.                                                                                                                                                                                                                                       |
| **SurgTE**     | T.E (Paradigm)     | **Shutter/WM**       | **Architecture Search**       | **Neural Architecture Search (NAS):** Modify shutter+WM class to match topology (e.g., add hierarchy / memory).                                                                                                                                                                                                                         |
| **SurgTC**     | T.C (Overfit)      | **WM**               | **Regularization**            | **Weight Decay / Dropout:** Increase $\lambda \lVert\theta\rVert^2$ penalty.                                                                                                                                                                                                                                                            |
| **SurgTD**     | T.D (Helplessness) | **Policy**           | **Noise Injection**           | **Parameter Space Noise:** Add $\xi \sim \mathcal{N}(0, \Sigma)$ to Policy weights.                                                                                                                                                                                                                                                     |
| **SurgDC**     | D.C (Ungrounded)   | **Shutter/WM**       | **Smoothing / fallback**      | **OOD rejection:** if nuisance surprisal spikes (e.g. $D_{\mathrm{KL}}(q(z_n\mid x)\Vert p(z_n))>\tau_n$) and/or texture surprisal spikes (e.g. $D_{\mathrm{KL}}(q(z_{\mathrm{tex}}\mid x)\Vert p(z_{\mathrm{tex}}))>\tau_{\mathrm{tex}}$) and/or macro surprisal spikes (e.g. $-\log p_\psi(K)>\tau_K$), trigger fallback (safe stop). |
| **SurgDE**     | D.E (Oscillate)    | **Policy**           | **Damping**                   | **Triggered by OscillateCheck / HolonomyCheck:** reduce policy step size (lower LR), decrease Adam $\beta_1$, increase batch size, or temporarily freeze policy updates until the critic signal is stable.                                                                                                                              |
| **SurgBE**     | B.E (Fragile)      | **Critic**           | **Saturation / Anti-Windup**  | **Spectral Normalization:** Constrain Lipschitz constant of $V(z)$.                                                                                                                                                                                                                                                                     |
| **SurgBD**     | B.D (Starve)       | **Boundary/Shutter** | **Replay Buffer / Reservoir** | **Experience Replay:** Train on historical buffers to prevent catastrophic forgetting.                                                                                                                                                                                                                                                  |
| **SurgBC**     | B.C (Deficit)      | **Policy**           | **Controller Expansion**      | **Width Expansion:** Dynamically add neurons to the Policy network (Net2Net).                                                                                                                                                                                                                                                           |



(sec-computational-considerations)=
