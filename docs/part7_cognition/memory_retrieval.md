## 27: Non-Local Memory as Self-Interaction Functional

:::{admonition} Researcher Bridge: Experience Replay as a Potential Field
:class: info
:name: rb-experience-replay
Standard Experience Replay buffers are just "bags of transitions" sampled at random. We reframe Memory as a **Self-Interaction Functional**. Past successes and failures act like physical magnets (attractive or repulsive charges) that generate a non-local force $\Psi_{\text{mem}}$. The agent does not just "sample" the past; it is literally **pulled** toward high-reward trajectories by the gradient of its own history.
:::

(sec-the-historical-manifold-and-memory-screen)=
### 27.1 The Historical Manifold and Memory Screen

**Motivation.** Sections 20–24 developed local dynamics: the geodesic SDE (Definition {prf:ref}`def-bulk-drift-continuous-flow`) evolves $z_t$ based on $\Phi_{\text{eff}}(z_t)$ and its gradient at the current position. This is a Markovian formulation—future evolution depends only on present state. However, intelligent agents demonstrably use *memory*: past experience influences current decisions through mechanisms beyond local gradients {cite}`lin1992experiencereplay,mnih2015dqn`. This section extends the geometric framework to include *non-local* contributions arising from the agent's trajectory history.

:::{prf:definition} Historical Record
:label: def-historical-record

Let $\gamma: [0, T] \to \mathcal{Z}$ be the agent's trajectory on the latent manifold $(\mathcal{Z}, G)$ over time interval $[0, T]$. The *historical record* is the pair $(\gamma, \alpha)$ where $\alpha: [0, T] \to \mathbb{R}$ is the reward flux along the trajectory (Definition {prf:ref}`def-the-reward-flux`).

*Units:* $[\gamma(t)] = [z]$, $[\alpha(t)] = \text{nat}/[s]$.

*Cross-reference:* This connects to Memory Time $t' < t$ (Definition 1.3.4).

:::
:::{prf:definition} Memory Screen
:label: def-memory-screen

The *memory screen* is the signed measure on $\mathcal{Z}$ defined by

$$
\Xi_T := \int_0^T \alpha(t') \, \delta_{\gamma(t')} \, dt',
$$
where:
- $\delta_{\gamma(t')}$ is the Dirac measure concentrated at $\gamma(t') \in \mathcal{Z}$,
- $\alpha(t') = J_r(t')$ is the (signed) reward flux at time $t'$ (Definition {prf:ref}`def-the-reward-flux`).

*Units:* $[\Xi_T] = \text{nat}$ (total signed measure), $[\alpha] = \text{nat}/[s]$ (reward flux rate).

*Interpretation:* $\Xi_T$ encodes where the agent has been, weighted by the sign and magnitude of reward received. Positive rewards contribute positive measure (attractive memory); negative rewards contribute negative measure (repulsive memory).

*Cross-reference (Relativistic Multi-Agent):* In Chapter 29, the Memory Screen is elevated from an auxiliary construct to a **primary state variable**. The Causal Bundle $\mathcal{Z}_{\text{causal}} := \mathcal{Z}^{(N)} \times \Xi_{<t}$ restores the Markov property in relativistic multi-agent settings where finite information speed creates non-Markovian dynamics. See Definition {prf:ref}`def-causal-bundle`.

:::
:::{prf:remark} Connection to Holographic Persistence
:label: rem-connection-to-holographic-persistence

The memory screen $\Xi_T$ provides the mathematical realization of holographic persistence (FAQ D.5.3). The measure $\Xi_T$ on $\mathcal{Z}$ acts as a "hologram" of the agent's history projected onto the latent space, from which non-local forces can be computed.

:::



(sec-the-non-local-interaction-functional)=
### 27.2 The Non-Local Interaction Functional

**Motivation.** Given the memory screen $\Xi_T$, we construct a *potential* $\Psi_{\text{mem}}(z)$ that exerts influence at the current position $z$ based on the entire historical distribution. The key mathematical object is an integral kernel that smooths and propagates the memory measure.

:::{prf:definition} Memory Kernel via Heat Equation {cite}`grigoryan2009heat,rosenberg1997laplacian`
:label: def-memory-kernel-via-heat-equation

The canonical memory kernel is the *Heat Kernel* $H_\tau(z, z')$ on $(\mathcal{Z}, G)$, defined as the fundamental solution to the heat equation:

$$
(\partial_\tau - \Delta_G) H_\tau(z, z') = 0, \quad H_0(z, z') = \delta(z - z'),
$$
where:
- $\tau > 0$ is the *diffusion time* (memory smoothing scale),
- $\Delta_G = G^{ij}\nabla_i\nabla_j$ is the Laplace-Beltrami operator on $(\mathcal{Z}, G)$ (Definition 2.5.3).

*Units:* $[H_\tau] = [z]^{-d}$ (probability density), $[\tau] = [z]^2$ (diffusion time in geometric units).

*Interpretation:* $H_\tau(z, z')$ measures how much influence a memory at $z'$ has on the current position $z$ after diffusion time $\tau$. Larger $\tau$ yields smoother, more diffuse memory influence. For compact manifolds, $H_\tau$ admits an eigenfunction expansion; for non-compact manifolds with bounded geometry, Gaussian upper bounds hold {cite}`grigoryan2009heat`.

:::
:::{prf:definition} Memory Potential
:label: def-memory-potential

The *memory potential* is defined by

$$
\Psi_{\text{mem}}(z) := -\int_{\mathcal{Z}} H_\tau(z, z') \, d\Xi_T(z').
$$
Expanding using Definition {prf:ref}`def-memory-screen`:

$$
\Psi_{\text{mem}}(z) = -\int_0^T \alpha(t') H_\tau(z, \gamma(t')) \, dt'.
$$
*Units:* $[\Psi_{\text{mem}}] = \text{nat}$.

*Interpretation:* The memory potential is the convolution of the heat kernel with the signed reward-weighted trajectory measure. Since $\Xi_T$ is a signed measure:
- Near high-reward past positions ($\alpha > 0$): $\Psi_{\text{mem}} < 0$, creating a potential well. The force $-\nabla_G \Psi_{\text{mem}}$ points toward the memory (attractive).
- Near high-penalty past positions ($\alpha < 0$): $\Psi_{\text{mem}} > 0$, creating a potential barrier. The force $-\nabla_G \Psi_{\text{mem}}$ points away from the memory (repulsive).

The sign convention ensures that the drift $-G^{-1}\nabla \Psi_{\text{mem}}$ moves toward rewarding experiences and away from penalizing ones.

:::

::::{admonition} Physics Isomorphism: Heat Kernel
:class: note
:name: pi-heat-kernel

**In Physics:** The heat kernel $H_t(x, y)$ is the fundamental solution of the heat equation $\partial_t u = \Delta u$, satisfying $H_t(x, \cdot) \to \delta_x$ as $t \to 0$. On a Riemannian manifold, it encodes diffusion and satisfies $H_t(x, y) \sim (4\pi t)^{-d/2}\exp(-d^2(x,y)/4t)$ for small $t$ {cite}`berline1992heat,grigoryan2009heat`.

**In Implementation:** The memory potential uses heat kernel convolution (Definition {prf:ref}`def-memory-potential`):

$$
\Psi_{\text{mem}}(z) = -\int_0^T \alpha(t') H_\tau(z, \gamma(t'))\, dt'
$$
where $H_\tau$ is the heat kernel on $(\mathcal{Z}, G)$.

**Correspondence Table:**
| Heat Equation Theory | Agent (Non-Local Memory) |
|:---------------------|:-------------------------|
| Heat kernel $H_t(x, y)$ | Memory diffusion kernel |
| Diffusion time $t$ | Memory timescale $\tau$ |
| Heat source | Past reward-weighted positions |
| Temperature evolution | Belief spread over time |
| Short-time asymptotics | Geodesic distance dominance |

**Connection:** The Matérn kernel $K_\nu \propto (-\Delta_G + \kappa^2)^{-\nu}$ generalizes the heat kernel; for $\nu = 1$, it recovers the screened Poisson Green's function (Section 24.2).
::::

:::{prf:proposition} Kernel Alternatives {cite}`rasmussen2006gp`
:label: prop-kernel-alternatives

Alternative kernels may be used depending on application requirements:

1. **Gaussian (RBF) Kernel:**

   $$
   K_{\text{Gauss}}(z, z') := \exp\left(-\frac{d_G(z, z')^2}{2\ell^2}\right),
   $$
   where $d_G$ is the geodesic distance and $\ell > 0$ is the length scale. This provides fast (exponential) decay, suitable for short-range memory effects.

2. **Matérn Kernel:**

   $$
   K_{\nu}(z, z') \propto (-\Delta_G + \kappa^2)^{-\nu}\delta(z - z'),
   $$
   where $\nu > 0$ is the smoothness parameter and $\kappa > 0$ is the inverse correlation length. For $\nu = 1$, this recovers the Green's function $G_\kappa$ from Section 24.2. The Matérn kernel has polynomial (rather than exponential) tails, providing longer-range correlations. See {cite}`rasmussen2006gp` Chapter 4 for the Euclidean case.

*Cross-reference:* The Matérn kernel with $\nu = 1$ coincides with the screened Poisson Green's function (Definition {prf:ref}`prop-green-s-function-decay`), establishing a direct connection between memory effects and value propagation.

:::
:::{prf:theorem} Non-Markovian Nature of Memory
:label: thm-non-markovian-nature-of-memory

The force field $-\nabla_G \Psi_{\text{mem}}$ violates the Markov property.

*Proof.* By Definition {prf:ref}`def-memory-potential`, $\Psi_{\text{mem}}(z_t)$ depends on $\Xi_T$, which contains $\gamma(t')$ for all $t' < t$. Therefore, $\nabla_G \Psi_{\text{mem}}(z_t)$ depends on the entire trajectory history $\{\gamma(t')\}_{t' \in [0,t)}$, not merely on $z_t$. This violates the Markov property $P(z_{t+\delta} | z_t, \{z_s\}_{s<t}) = P(z_{t+\delta} | z_t)$. $\square$

*Remark (State Augmentation):* The non-Markovian character is essential for capturing genuine memory effects. The system state must be *augmented* to include $\Xi_T$ (or a sufficient statistic thereof) to recover a Markovian description in an extended state space.

*Remark (Computational Complexity):* Naively, evaluating $\Psi_{\text{mem}}(z)$ requires $O(T)$ kernel evaluations where $T$ is the trajectory length. For long histories, approximations are necessary: (i) truncate to recent history, (ii) subsample the trajectory, (iii) use inducing points {cite}`rasmussen2006gp`, or (iv) maintain a running kernel density estimate.

:::



(sec-memory-augmented-equations-of-motion)=
### 27.3 Memory-Augmented Equations of Motion

**Motivation.** We now extend the geodesic SDE (Definition {prf:ref}`def-bulk-drift-continuous-flow`) to include the memory-induced force $-\nabla_G \Psi_{\text{mem}}$.

:::{prf:definition} Memory-Augmented Geodesic SDE
:label: def-memory-augmented-geodesic-sde

The memory-augmented dynamics on $(\mathcal{Z}, G)$ are:

$$
dz^k = \left[ -G^{kj}\partial_j\bigl(\Phi_{\text{eff}} + \Psi_{\text{mem}}\bigr) + u_\pi^k \right] ds - \Gamma^k_{ij}\dot{z}^i\dot{z}^j\,ds + \sqrt{2T_c}\,(G^{-1/2})^{kj}\,dW^j_s,
$$
where:
- $\Phi_{\text{eff}}$ is the effective potential (Definition {prf:ref}`def-effective-potential`),
- $\Psi_{\text{mem}}$ is the memory potential (Definition {prf:ref}`def-memory-potential`),
- $\Gamma^k_{ij}$ are the Christoffel symbols of $G$ (Definition 2.5.1),
- $u_\pi^k$ is the policy control field (Definition {prf:ref}`def-the-control-field`),
- $T_c$ is the cognitive temperature (Section 22.4),
- $W^j_s$ is a standard Wiener process.

*Cross-reference:* Definition {prf:ref}`def-bulk-drift-continuous-flow`.

*Units:* All terms have units $[z]/[s]$.

:::
:::{prf:lemma} Virtual Work of Recall
:label: lem-virtual-work-of-recall

The infinitesimal work performed by the memory force during displacement $dz$ is:

$$
dW_{\text{mem}} := \langle -\nabla_G \Psi_{\text{mem}}, dz \rangle_G = -G_{kj}\,G^{k\ell}\partial_\ell \Psi_{\text{mem}}\, dz^j = -\partial_j \Psi_{\text{mem}}\, dz^j.
$$
*Units:* $[dW_{\text{mem}}] = \text{nat}$.

*Interpretation:* When the agent moves toward regions of low $\Psi_{\text{mem}}$ (attractive memory, i.e., $d\Psi_{\text{mem}} < 0$), positive work $dW_{\text{mem}} > 0$ is extracted from the memory field. This corresponds to "reward from recall"—revisiting previously successful states.

:::
:::{prf:theorem} Memory-Induced Barrier Crossing
:label: thm-memory-induced-barrier-crossing

Let $z_t$ be the current position and suppose there exists a past time $t^* < t$ with $z^* := \gamma(t^*)$ such that:
1. $d_G(z_t, z^*) < \ell_{\text{mem}}$ for some memory influence radius $\ell_{\text{mem}}$,
2. $|\alpha(t^*)|$ is large (strong reward signal at time $t^*$).

Then the memory gradient $\|\nabla_G \Psi_{\text{mem}}\|_G$ can exceed the local barrier gradient $\|\nabla_G \Phi_{\text{eff}}\|_G$, enabling transitions that would be forbidden under purely local dynamics.

*Proof sketch.* By Definition {prf:ref}`def-memory-potential` and the concentration of $H_\tau$ near the diagonal for small $\tau$:

$$
\|\nabla_G \Psi_{\text{mem}}(z_t)\|_G \approx |\alpha(t^*)| \cdot \|\nabla_G H_\tau(z_t, z^*)\|_G.
$$
For $d_G(z_t, z^*) \sim O(\sqrt{\tau})$, the gradient $\|\nabla_G H_\tau\|_G \sim O(\tau^{-(d+1)/2})$ can be made arbitrarily large by choosing small $\tau$. If $|\alpha(t^*)|$ is sufficiently large, this dominates $\|\nabla_G \Phi_{\text{eff}}\|_G$. $\square$

*Cross-reference:* BarrierGap diagnostic (Section 4).

*Interpretation:* Strong memories can "pull" the agent across local energy barriers, providing a mechanism for experience-guided exploration that transcends gradient-based planning.

:::

::::{note} Connection to RL #20: Experience Replay as Degenerate Non-Local Memory
**The General Law (Fragile Agent):**
Trajectory history induces a **Memory Potential** via heat-kernel convolution:

$$
\Psi_{\text{mem}}(z) = -\int_0^T \alpha(t') H_\tau(z, \gamma(t'))\, dt'
$$
where $H_\tau$ is the heat kernel on $(\mathcal{Z}, G)$ and $\alpha(t')$ is the reward flux at past times.

**The Degenerate Limit:**
Replace geometric kernel with uniform sampling. Ignore metric structure ($G \to I$).

**The Special Case (Standard RL):**

$$
\mathcal{D} = \{(s_t, a_t, r_t, s_{t+1})\}, \quad \text{sample } \sim \text{Uniform}(\mathcal{D})
$$
This recovers **Experience Replay** {cite}`lin1992experience,mnih2015humanlevel`.

**What the generalization offers:**
- **Geometric memory:** Distances measured in $d_G$, not Euclidean; nearby trajectories interact more strongly
- **Reward-signed forces:** Positive rewards attract (revisit success); negative repel (avoid failure)
- **Heat-kernel smoothing:** Memory influence decays with diffusion time $\tau$
- **Barrier crossing:** Strong memories can pull the agent across local energy barriers
::::



(sec-wfr-dynamics-with-memory-sources)=
### 27.4 WFR Dynamics with Memory Sources

**Motivation.** Lifting to measure space via the Wasserstein-Fisher-Rao framework (Section 20) {cite}`chizat2018wfr`, we obtain a reaction-diffusion PDE incorporating memory.

:::{prf:definition} Memory-Augmented Reaction-Diffusion
:label: def-memory-augmented-reaction-diffusion

The WFR dynamics with memory are:

$$
\partial_s \rho + \nabla \cdot (\rho \mathbf{v}) = \rho \left(\frac{\Phi_{\text{eff}} + \Psi_{\text{mem}} - \bar{\Phi}_{\text{aug}}}{T_c}\right),
$$
where:
- $\rho(z, s)$ is the belief density,
- $\mathbf{v} = -G^{-1}\nabla(\Phi_{\text{eff}} + \Psi_{\text{mem}}) + u_\pi$ is the augmented drift,
- $\bar{\Phi}_{\text{aug}} = \int_{\mathcal{Z}} (\Phi_{\text{eff}} + \Psi_{\text{mem}}) \rho \, d\mu_G$ is the mean augmented potential.

*Cross-reference:* Definition {prf:ref}`def-the-wfr-action`, Theorem {prf:ref}`thm-wfr-consistency-value-creates-mass`.

*Units:* $[\partial_s \rho] = [z]^{-d}/[s]$, all terms balance.

:::
:::{prf:proposition} Mass Creation from Experience
:label: prop-mass-creation-from-experience

The memory contribution to the reaction term is:

$$
r_{\text{mem}}(z) := \frac{\rho(z)(\Psi_{\text{mem}}(z) - \bar{\Psi}_{\text{mem}})}{T_c},
$$
where $\bar{\Psi}_{\text{mem}} = \int_{\mathcal{Z}} \Psi_{\text{mem}} \rho \, d\mu_G$.

*Interpretation:* Belief mass is created where $\Psi_{\text{mem}} < \bar{\Psi}_{\text{mem}}$ (attractive memory) and destroyed where $\Psi_{\text{mem}} > \bar{\Psi}_{\text{mem}}$ (repulsive memory). This acts as a *virtual source* that redistributes probability toward remembered high-reward regions, even when local dynamics (via $\Phi_{\text{eff}}$) do not support such transitions.

:::



(sec-stability-analysis-and-diagnostic)=
### 27.5 Stability Analysis and Diagnostic

**Motivation.** Non-local memory introduces a potential source of instability: if memory forces dominate local dynamics, the agent may overfit to history and fail to adapt to environmental changes. Conversely, if memory is too weak, the agent exhibits catastrophic forgetting.

:::{prf:definition} Non-Locality Ratio
:label: def-non-locality-ratio

The *non-locality ratio* at position $z$ is:

$$
\Omega_{\text{mem}}(z) := \frac{\|\nabla_G \Psi_{\text{mem}}(z)\|_G}{\|\nabla_G \Phi_{\text{eff}}(z)\|_G + \epsilon},
$$
where $\epsilon > 0$ is a regularization constant preventing division by zero.

*Units:* $[\Omega_{\text{mem}}] = \text{dimensionless}$.

**Heuristic 27.5.2 (Homeostatic Bound on Memory).** For stable operation, the non-locality ratio should satisfy:

$$
\Omega_{\text{mem}} \in [\Omega_{\min}, \Omega_{\max}],
$$
with empirically recommended bounds $\Omega_{\min} \approx 0.01$, $\Omega_{\max} \approx 10$. These bounds are task-dependent and should be tuned based on the environment's stationarity.

*Boundary cases:*
- $\Omega_{\text{mem}} \to 0$: Pure Markovian dynamics; agent exhibits catastrophic forgetting.
- $\Omega_{\text{mem}} \to \infty$: Pure memory-driven dynamics; agent overfits to historical experience and fails to respond to current environmental gradients.

*Cross-reference:* The Governor (Section 26) can regulate $\Omega_{\text{mem}}$ by adjusting the memory smoothing scale $\tau$ or the reward flux weighting in $\alpha(t')$.

(node-43)=
**Node 43: MemoryBalanceCheck**

| **#**  | **Name**               | **Component**     | **Type**              | **Interpretation**              | **Proxy**                                                                                                 | **Cost**               |
|--------|------------------------|-------------------|-----------------------|---------------------------------|-----------------------------------------------------------------------------------------------------------|------------------------|
| **43** | **MemoryBalanceCheck** | **Memory Screen** | **Non-Local Balance** | Is memory contribution bounded? | $\Omega_{\text{mem}} = \lVert\nabla_G\Psi_{\text{mem}}\rVert_G / \lVert\nabla_G\Phi_{\text{eff}}\rVert_G$ | $O(\lvert\Xi_T\rvert)$ |

**Trigger conditions:**
- $\Omega_{\text{mem}} < \Omega_{\min}$: Memory underutilized; increase $\alpha$ weighting or decrease $\tau$.
- $\Omega_{\text{mem}} > \Omega_{\max}$: Memory dominates; increase $\tau$ to smooth memory influence or decay old experiences.
- Persistent imbalance: Re-examine memory kernel choice or trajectory sampling strategy.

**Cross-references:** Section 3 (Sieve Diagnostic Nodes), Section 26 (Governor regulation), Node 42 (GovernorStabilityCheck).

:::



(sec-summary-memory-as-non-local-interface)=
### 27.6 Summary: Memory as Non-Local Interface

**Table 27.6.1 (Pillar Locality Comparison).**

| Pillar     | Operator            | Geometric Role                   | Locality      |
|------------|---------------------|----------------------------------|---------------|
| Perception | $E_\phi$            | Dirichlet BC (position clamping) | Local         |
| Action     | $D_A$               | Neumann BC (flux clamping)       | Local         |
| Value      | $\Phi_{\text{eff}}$ | Source BC (Helmholtz solution)   | Local         |
| **Memory** | $\Psi_{\text{mem}}$ | **Fredholm integral operator**   | **Non-local** |

*Key insight:* Memory introduces the first genuinely non-local contribution to the agent dynamics. While perception, action, and value all depend on local data (position, flux, source at $z$), memory integrates information over the entire trajectory history via the kernel $H_\tau$.

**Table 27.6.2 (Memory Kernel Comparison).**

| Kernel         | Asymptotic Form                         | Decay Rate              | Use Case                                  |
|----------------|-----------------------------------------|-------------------------|-------------------------------------------|
| Heat $H_\tau$  | $(4\pi\tau)^{-d/2}\exp(-d_G^2/4\tau)$   | Gaussian                | Default; smooth diffusive influence       |
| Gaussian/RBF   | $\exp(-d_G^2/2\ell^2)$                  | Exponential             | Short-range memory; fast computation      |
| Matérn $K_\nu$ | $d_G^{\nu-d/2} K_{\nu-d/2}(\kappa d_G)$ | Polynomial $\times$ exp | Long-range; connects to value propagation |

*Note:* $K_{\nu}$ denotes the modified Bessel function of the second kind. For the Matérn kernel on curved manifolds, the formula is approximate; exact expressions require spectral methods.

**Summary.** This section introduced non-local memory as a self-interaction functional, extending the Markovian dynamics of Sections 20–24. The memory screen $\Xi_T$ (Definition {prf:ref}`def-memory-screen`) encodes reward-weighted trajectory history; the memory potential $\Psi_{\text{mem}}$ (Definition {prf:ref}`def-memory-potential`) converts this into a force field via heat kernel convolution; and the Non-Locality Ratio $\Omega_{\text{mem}}$ (Definition {prf:ref}`def-non-locality-ratio`) provides a diagnostic for balancing memory against local gradients. Node 43 (MemoryBalanceCheck) monitors this ratio during training.



(sec-section-hyperbolic-active-retrieval-geodesic-search-and-semantic-pull-back)=
## 28 · Hyperbolic Active Retrieval: Geodesic Search and Semantic Pull-Back

:::{admonition} Researcher Bridge: Retrieval-Augmented Control
:class: info
:name: rb-retrieval-augmented
If you know episodic control or retrieval-augmented generation, this is the geometric version: retrieval is a geodesic search in a shared embedding space. The firewall ensures retrieved texture does not leak into policy decisions.
:::

*Cross-references:* {ref}`Section 21 <sec-radial-generation-entropic-drift-and-policy-control>` (Poincare metric), {ref}`Section 22 <sec-the-equations-of-motion-geodesic-jump-diffusion>` (Equations of Motion), {ref}`Section 27 <sec-section-non-local-memory-as-self-interaction-functional>` (Memory Potential), {ref}`Section 7.8 <sec-tier-the-attentive-atlas>` (Atlas architecture), {ref}`Section 2.8 <sec-conditional-independence-and-sufficiency>` (macro closure).

**Motivation.** While Section 27 treated memory as self-interaction—retrieval from the agent's own trajectory—this section addresses *external* retrieval from knowledge bases, embedding indices, and document stores. The central observation is that the Poincare disk geometry introduced in Section 21 applies equally to both internal latent representations and external knowledge embeddings. This isomorphism enables principled Retrieval-Augmented Generation (RAG) as geodesic search on a shared hyperbolic manifold.

The key challenge is the **texture firewall problem**: external documents contain high-frequency texture ($z_{\text{tex}}$) that must be delivered to the decoder but excluded from the control loop to prevent Mode T.C (Labyrinthine Overfitting). We solve this by extending the existing TextureFirewallCheck (Node 29) to external retrieval, ensuring that only bulk coordinates $(K, z_n)$ influence policy gradients.

(sec-the-isomorphism-of-semantic-manifolds)=
### 28.1 The Isomorphism of Semantic Manifolds

:::{prf:definition} External Knowledge Manifold
:label: def-external-knowledge-manifold

Let $\mathcal{Z}_{\text{ext}}$ denote the external knowledge manifold equipped with metric $G_{\text{ext}}$, structured as a fiber bundle:

$$
\mathcal{Z}_{\text{ext}} = \mathcal{K} \times \mathcal{Z}_n \times \mathcal{Z}_{\text{tex}},
$$
where $\mathcal{K}$ is the macro-concept space, $\mathcal{Z}_n$ the nuisance coordinates, and $\mathcal{Z}_{\text{tex}}$ the texture fiber.

*Units:* $[G_{\text{ext},ij}] = [z]^{-2}$ (matching the internal metric).

*Cross-reference:* This decomposition mirrors Section 2.8's latent structure $(K, z_n, z_{\text{tex}})$ and Section 7.8's Atlas architecture.

:::
:::{prf:axiom} Metric Isometry
:label: ax-metric-isometry

There exists a canonical isometry $\Phi: \mathcal{Z}_{\text{int}} \to \mathcal{Z}_{\text{ext}}$ such that for all $z, z' \in \mathcal{Z}_{\text{int}}$:

$$
d_{G_{\text{int}}}(z, z') = d_{G_{\text{ext}}}(\Phi(z), \Phi(z')),
$$
where both manifolds carry the Poincare metric (Definition {prf:ref}`def-hyperbolic-volume-growth`):

$$
G_{ij}(z) = \frac{4\delta_{ij}}{(1 - \|z\|^2)^2}.
$$
*Interpretation:* The isometry axiom asserts that embedding models trained on shared semantic corpora induce compatible distance structures. This is the mathematical foundation for cross-modal retrieval.

:::
:::{prf:definition} Knowledge Atom
:label: def-knowledge-atom

A *knowledge atom* is a triple $\xi = (K, z_n, z_{\text{tex}}) \in \mathcal{Z}_{\text{ext}}$ where:
- $K \in \mathcal{K}$: macro-concept (topic, entity class, logical category)
- $z_n \in \mathcal{Z}_n$: nuisance coordinates (style, formatting, source metadata)
- $z_{\text{tex}} \in \mathcal{Z}_{\text{tex}}$: high-frequency texture (specific wording, surface form)

*Cross-reference:* Compare Section 2.8's decomposition. The macro closure mechanism (Definition 2.8.1) applies equally to external atoms.

:::
(sec-geodesic-search-in-hyperbolic-space)=
### 28.2 Geodesic Search in Hyperbolic Space

:::{prf:definition} Hyperbolic Geodesic Distance
:label: def-hyperbolic-geodesic-distance

For points $z, \xi \in \mathbb{D}^d$ (the Poincare disk), the geodesic distance is:

$$
d_{\mathbb{D}}(z, \xi) = \operatorname{acosh}\left(1 + \frac{2\|z - \xi\|^2}{(1 - \|z\|^2)(1 - \|\xi\|^2)}\right).
$$
*Units:* $[d_{\mathbb{D}}] = [z]$ (dimensionless in Poincare coordinates).

*Cross-reference:* This is the distance function induced by the Poincare metric $G_{ij}$ (Definition {prf:ref}`def-hyperbolic-volume-growth`). See also Definition {prf:ref}`prop-isotropic-radial-expansion` for the hyperbolic potential $U(z) = -2\operatorname{artanh}(\|z\|)$.

:::
:::{prf:definition} Retrieval Measure via Geodesic Functional
:label: def-retrieval-measure-via-geodesic-functional

Given a query position $z \in \mathcal{Z}_{\text{int}}$ and archive prior $\mu_{\mathcal{E}} \in \mathcal{P}(\mathcal{Z}_{\text{ext}})$, the *retrieval measure* is:

$$
\nu_\omega = \arg\min_{\nu \in \mathcal{P}(\mathcal{Z}_{\text{ext}})} \left\{ \int d_{\mathbb{D}}(z, \xi) \, d\nu(\xi) + T_{\text{ret}} D_{\text{KL}}(\nu \| \mu_{\mathcal{E}}) \right\},
$$
where $T_{\text{ret}} > 0$ is the *retrieval temperature*.

*Units:* $[T_{\text{ret}}] = \text{nat}$.

*Interpretation:* This variational problem balances semantic proximity (first term) against prior plausibility (KL term). At $T_{\text{ret}} \to 0$, retrieval concentrates on the nearest neighbor; at $T_{\text{ret}} \to \infty$, it reverts to the archive prior.

:::
:::{prf:proposition} Exponential Complexity of Specificity
:label: prop-exponential-complexity-of-specificity

The volume of a geodesic ball in the Poincare disk grows exponentially with radius:

$$
\text{Vol}(B_r(z)) \sim \sinh^{d-1}(r) \sim \frac{1}{2^{d-1}} e^{(d-1)r} \quad \text{as } r \to \infty.
$$
*Proof sketch:* The hyperbolic metric has constant negative curvature $\kappa = -1$. Standard volume comparison (Bishop-Gromov) yields exponential growth. $\square$

*Interpretation:* As the agent descends toward the boundary (increasing semantic specificity), the number of accessible knowledge atoms grows exponentially. This captures the combinatorial explosion of specific facts relative to abstract concepts—compare TopoEncoder hierarchy (Section 25).

:::
(sec-the-retrieval-texture-firewall)=
### 28.3 The Retrieval Texture Firewall

:::{prf:definition} Bulk Projection Operator
:label: def-bulk-projection-operator

The *bulk projection* $\Pi_{\text{bulk}}: \mathcal{Z}_{\text{ext}} \to \mathcal{K} \times \mathcal{Z}_n$ is defined by:

$$
\Pi_{\text{bulk}}(\xi) = \Pi_{\text{bulk}}(K, z_n, z_{\text{tex}}) := (K, z_n).
$$
*Interpretation:* This projection discards texture, retaining only control-relevant coordinates.

*Cross-reference:* This extends the internal texture exclusion of Section 2.8 to external retrieval.

:::
:::{prf:definition} Bulk-Filtered Retrieval Potential
:label: def-bulk-filtered-retrieval-potential

The *retrieval potential* is:

$$
\Psi_{\text{ret}}(z) = -\Lambda_{\text{ret}} \int_{\mathcal{Z}_{\text{ext}}} \exp\left(-\lambda \, d_{\mathbb{D}}(z, \Pi_{\text{bulk}}(\xi))\right) d\nu_\omega(\xi),
$$
with the firewall constraint:

$$
\frac{\partial \Psi_{\text{ret}}}{\partial z_{\text{tex,ext}}} \equiv 0.
$$
*Units:* $[\Psi_{\text{ret}}] = \text{nat}$, $[\Lambda_{\text{ret}}] = \text{nat}$, $[\lambda] = [z]^{-1}$.

*Cross-reference:* Compare the memory potential $\Psi_{\text{mem}}$ (Definition {prf:ref}`def-memory-potential`), which uses heat kernel rather than geodesic exponential. Both generate conservative forces.

:::
:::{prf:theorem} Stability of Retrieval Loop
:label: thm-stability-of-retrieval-loop

Under the firewall constraint (Definition {prf:ref}`def-bulk-filtered-retrieval-potential`), the retrieval force field:

$$
\mathbf{f}_{\text{ret}} = -G^{-1}\nabla_G \Psi_{\text{ret}}
$$
is smooth (Lipschitz in $z$) and independent of external texture coordinates $z_{\text{tex,ext}}$.

*Consequence:* The control loop remains stable; external texture cannot inject high-frequency gradients that would trigger Mode T.C (Labyrinthine Overfitting).

*Proof sketch:* The bulk projection $\Pi_{\text{bulk}}$ is a smooth submersion. Composition with the smooth geodesic exponential preserves smoothness. The firewall constraint ensures $\nabla_{z_{\text{tex,ext}}} \Psi_{\text{ret}} = 0$ by construction. $\square$

*Cross-reference:* This theorem extends TextureFirewallCheck (Node 29) to external retrieval. See Section 8.2.5 for Mode T.C classification.

**Heuristic 28.3.4 (Side-Channel Texture Delivery).**
External texture $z_{\text{tex,ext}}$ is delivered to the decoder via a side channel:
1. At stopping radius $R_{\text{cutoff}}$ (Section 21.3), retrieve the full atom $\xi = (K, z_n, z_{\text{tex}})$
2. Inject $z_{\text{tex}}$ directly to decoder attention, bypassing the EoM
3. The control loop only sees $(K, z_n)$

*Interpretation:* This is the retrieval analog of "reading a document without letting its style affect your reasoning."

:::
(sec-retrieval-augmented-equations-of-motion)=
### 28.4 Retrieval-Augmented Equations of Motion

:::{prf:definition} Retrieval-Augmented Geodesic SDE
:label: def-retrieval-augmented-geodesic-sde

The equations of motion with retrieval are:

$$
dz^k = \left[ -G^{kj}\partial_j(\Phi_{\text{eff}} + \Psi_{\text{mem}} + \Psi_{\text{ret}}) + u_\pi^k \right] ds - \Gamma^k_{ij}\dot{z}^i\dot{z}^j\,ds + \sqrt{2T_c}(G^{-1/2})^{kj}dW^j_s,
$$
where:
- $\Phi_{\text{eff}}$: effective potential (Definition {prf:ref}`def-effective-potential`)
- $\Psi_{\text{mem}}$: memory potential (Definition {prf:ref}`def-memory-potential`)
- $\Psi_{\text{ret}}$: retrieval potential (Definition {prf:ref}`def-bulk-filtered-retrieval-potential`)
- $\Gamma^k_{ij}$: Christoffel symbols (Definition 2.5.1, Definition 22.2.1a)
- $u_\pi^k$: policy control field (Definition {prf:ref}`def-the-control-field`)
- $T_c$: cognitive temperature (Section 22.4)

*Cross-reference:* This extends the memory-augmented SDE (Definition {prf:ref}`def-memory-augmented-geodesic-sde`) with the retrieval term $\Psi_{\text{ret}}$.

:::
:::{prf:proposition} Superposition of Non-Local Forces
:label: prop-superposition-of-non-local-forces

The total non-local force is:

$$
\mathbf{f}_{\text{non-local}} = -G^{-1}\nabla_G(\Psi_{\text{mem}} + \Psi_{\text{ret}}),
$$
where:
- Memory force $\mathbf{f}_{\text{mem}}$ integrates over the agent's past trajectory
- Retrieval force $\mathbf{f}_{\text{ret}}$ integrates over the external archive

*Interpretation:* The agent simultaneously experiences attraction to its own memory (Section 27) and to relevant external knowledge (this section).

:::
(sec-wfr-dynamics-retrieval-induced-mass-injection)=
### 28.5 WFR Dynamics: Retrieval-Induced Mass Injection

:::{prf:definition} Retrieval Source Term
:label: def-retrieval-source-term

The Wasserstein–Fisher–Rao continuity equation with retrieval is:

$$
\partial_s \rho + \nabla \cdot (\rho \mathbf{v}) = \rho \, r_{\text{local}}(z) + \sigma_{\text{ret}}(z),
$$
where:
- $r_{\text{local}}(z)$: local mass creation rate (reward-driven, Definition {prf:ref}`def-the-wfr-action`)
- $\sigma_{\text{ret}}(z)$: retrieval source term

The retrieval source is:

$$
\sigma_{\text{ret}}(z) = \eta_{\text{ret}} \cdot \Psi_{\text{ret}}(z) \cdot \mathbf{1}[\Psi_{\text{ret}}(z) > \Psi_{\text{threshold}}],
$$
with $[\sigma_{\text{ret}}] = \text{nat}/[z]^d/\text{step}$.

*Cross-reference:* Compare Section 27.4's memory mass creation. Both mechanisms inject mass at non-local locations.

:::
:::{prf:proposition} Non-Causal Transition via Retrieval
:label: prop-non-causal-transition-via-retrieval

Mass injection at retrieved locations enables transitions without continuous geodesic paths:

$$
\rho(z', s + \Delta s) > 0 \quad \text{even if} \quad d_G(z, z') > \sup_{0 \leq \tau \leq \Delta s} \|\mathbf{v}(z, s+\tau)\| \cdot \Delta s.
$$
*Interpretation:* Retrieval teleports probability mass to semantically relevant regions, bypassing the diffusion constraint. This is the WFR-level description of "jumping to a retrieved fact."

:::
(sec-diagnostic-nodes-for-retrieval-integrity)=
### 28.6 Diagnostic Nodes for Retrieval Integrity

We introduce two diagnostic nodes for monitoring retrieval health.

(node-44)=
**Node 44: HyperbolicAlignmentCheck**

| **#**  | **Name**                     | **Component** | **Type**           | **Interpretation**                       | **Proxy**                                                                                                                               | **Cost**                    |
|--------|------------------------------|---------------|--------------------|------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------|-----------------------------|
| **44** | **HyperbolicAlignmentCheck** | Interface     | Metric Consistency | Are internal/external manifolds aligned? | $\Delta_{\text{align}} := \mathbb{E}[\lVert d_{\mathbb{D}}^{\text{int}}(z, z') - d_{\mathbb{D}}^{\text{ext}}(\Phi(z), \Phi(z'))\rVert]$ | $O(\lVert\nu_\omega\rVert)$ |

**Interpretation:** Tests whether the isometry axiom (Axiom {prf:ref}`ax-metric-isometry`) holds empirically. Large $\Delta_{\text{align}}$ indicates embedding drift or domain shift between internal representations and external knowledge base.

**Threshold:** $\Delta_{\text{align}} < 0.1 \cdot \bar{d}_{\mathbb{D}}$ (alignment error below 10% of mean geodesic distance).

(node-45)=
**Node 45: RetrievalFirewallCheck**

| **#**  | **Name**                   | **Component** | **Type**         | **Interpretation**                         | **Proxy**                                                                                                  | **Cost**            |
|--------|----------------------------|---------------|------------------|--------------------------------------------|------------------------------------------------------------------------------------------------------------|---------------------|
| **45** | **RetrievalFirewallCheck** | Policy        | Causal Isolation | Is external texture isolated from control? | $\Gamma_{\text{leak}} := \lVert\nabla_{z_{\text{int}}} (\partial \pi / \partial z_{\text{tex,ext}})\rVert$ | $O(d_{\text{tex}})$ |

**Interpretation:** Measures texture leakage into policy gradients. Should be near-zero under Theorem {prf:ref}`thm-stability-of-retrieval-loop`.

**Threshold:** $\Gamma_{\text{leak}} < \epsilon_{\text{firewall}}$ (implementation-dependent; typically $10^{-6}$).

*Cross-reference:* Node 45 extends the internal TextureFirewallCheck (Node 29) to external retrieval.

(sec-summary)=
### 28.7 Summary

**Table 28.7.1 (Memory vs Retrieval Comparison).**

| Aspect         | Memory (Section 27)                                        | Retrieval (Section 28)                                                      |
|----------------|------------------------------------------------------------|-----------------------------------------------------------------------------|
| **Source**     | Internal trajectory $\gamma_{0:T}$                         | External archive $\mathcal{Z}_{\text{ext}}$                                 |
| **Kernel**     | Heat kernel $H_\tau(z, z')$                                | Geodesic exponential $\exp(-\lambda d_{\mathbb{D}})$                        |
| **Potential**  | $\Psi_{\text{mem}}$ (Def. {prf:ref}`def-memory-potential`) | $\Psi_{\text{ret}}$ (Def. {prf:ref}`def-bulk-filtered-retrieval-potential`) |
| **Firewall**   | Temporal (past vs present)                                 | Spatial (bulk vs texture)                                                   |
| **WFR source** | $r_{\text{mem}}(z)$                                        | $\sigma_{\text{ret}}(z)$                                                    |
| **Diagnostic** | Node 43 (MemoryBalanceCheck)                               | Nodes 44–45                                                                 |

*Key insight:* Memory and retrieval are dual non-local mechanisms. Memory integrates over temporal history; retrieval integrates over spatial archive. Both contribute conservative forces to the equations of motion (Definition {prf:ref}`def-retrieval-augmented-geodesic-sde`) and mass sources to WFR dynamics (Definition {prf:ref}`def-retrieval-source-term`).



(sec-bilevel-nonlocal-regulation)=
### 28.8 Bilevel Regulation of Non-Local Potentials (Joint Optimization Resolution)

The joint optimization of memory strength $\Lambda_{\text{mem}}$ and retrieval strength $\Lambda_{\text{ret}}$ is solved by the Universal Governor ({ref}`Section 26 <sec-theory-of-meta-stability-the-universal-governor-as-homeostatic-controller>`) acting on the diagnostic residuals of Nodes 43 and 53.

:::{prf:proposition} Optimal Non-Local Coupling
:label: prop-optimal-nonlocal-coupling

Let the control vector be $\Lambda = (\Lambda_{\text{mem}}, \Lambda_{\text{ret}})$. The optimal coupling is the fixed point of the Governor's policy $\pi_{\mathfrak{G}}$ ({prf:ref}`def-the-universal-governor`) given the diagnostic state $s_t = (\Delta_{\text{causal}}, \Omega_{\text{mem}})$.

**Control Law Derivation:**

1. **Surprise Signal:** Let $\Delta_{\text{causal}} = D_{\text{KL}}(P_{\text{int}} \| P_{\text{obs}})$ be the Interventional Gap (Node 53).

2. **Overfitting Signal:** Let $\Omega_{\text{mem}}$ be the Non-Locality Ratio ({prf:ref}`def-non-locality-ratio`, Node 43).

3. **Governor Update:** The Lyapunov descent condition $\Delta V_{\mathfrak{L}} < 0$ ({prf:ref}`def-training-lyapunov-function`) implies the following qualitative update dynamics:

$$
\begin{aligned}
\dot{\Lambda}_{\text{ret}} &\propto \alpha_1 \cdot \Delta_{\text{causal}} \\
\dot{\Lambda}_{\text{mem}} &\propto \alpha_2 \cdot (\Delta_{\text{causal}}^{\text{target}} - \Delta_{\text{causal}}) - \alpha_3 \cdot \operatorname{ReLU}(\Omega_{\text{mem}} - \Omega_{\max})
\end{aligned}
$$
where $\alpha_1, \alpha_2, \alpha_3 > 0$ are learning rates and $\Omega_{\max}$ is the maximum tolerable non-locality ratio.

*Proof sketch.* The Governor's outer objective ({prf:ref}`def-outer-problem-governor-optimization`) includes terms penalizing both prediction error (Interventional Gap) and overfitting (Non-Locality Ratio). The gradient of this objective with respect to $\Lambda$ yields the stated control law. At equilibrium, $\dot{\Lambda} = 0$, which implies a balance between reliance on memory and retrieval calibrated to the agent's surprise level. $\square$
:::

:::{prf:remark} Operational Interpretation
:label: rem-memory-retrieval-interpretation

- **If the agent is surprised by reality** ($\Delta_{\text{causal}}$ high): It must increase reliance on external truth ($\Lambda_{\text{ret}} \uparrow$).
- **If the agent is not surprised** ($\Delta_{\text{causal}}$ low): It can conserve bandwidth by relying on internal memory ($\Lambda_{\text{mem}} \uparrow$), subject to the constraint that it must not overfit ($\Omega_{\text{mem}} < \Omega_{\max}$).

This closes the joint optimization problem by reducing it to a specific instantiation of the Governor's Lyapunov stability framework ({prf:ref}`def-training-lyapunov-function`).
:::



(sec-safe-retrieval-bandwidth)=
### 28.9 The Safe Retrieval Bandwidth Corollary (Instability Resolution)

Retrieval-induced instability is identified as the violation of the **Causal Information Bound** ({ref}`Section 33 <sec-causal-information-bound>`). Retrieval functions as a mass-injection source term; stability is preserved only if the total bulk information respects the interface area law.

:::{prf:theorem} Safe Retrieval Bandwidth
:label: thm-safe-retrieval-bandwidth

Let $\sigma_{\text{ret}}(z)$ be the retrieval source term in the WFR continuity equation ({prf:ref}`def-retrieval-source-term`). The latent geometry remains non-singular if and only if the total information flux satisfies:

$$
\int_{\mathcal{Z}} \left( \rho_I(z) + \sigma_{\text{ret}}(z) \right) \, d\mu_G \leq \kappa \, C_{\partial}
$$
where $C_{\partial} = \nu_D \cdot \text{Area}(\partial\mathcal{Z})/\ell_L^{D-1}$ is the boundary capacity (Definition {prf:ref}`def-holographic-coefficient`, {prf:ref}`def-levin-length`).

*Proof.*
1. **Mass Augmentation:** Retrieval modifies the bulk information density: $\tilde{\rho}_I = \rho_I + \sigma_{\text{ret}}$.

2. **Metric Response:** By the Capacity-Constrained Metric Law ({prf:ref}`thm-capacity-constrained-metric-law`), the radial metric component scales as $G_{rr} \propto (1 - \tilde{I}_{\text{bulk}}/C_{\partial})^{-1}$.

3. **Singularity:** If $\int \sigma_{\text{ret}} > C_{\partial} - I_{\text{bulk}}$, then $G_{rr} \to \infty$ at a radius $r < 1$ (the horizon moves inward).

4. **Dynamical Consequence:** The update velocity $\|v\|_G \to 0$ (Causal Stasis, {ref}`Section 33 <sec-causal-information-bound>`). The instability manifests as the freezing of the agent's inference dynamics due to saturation of the holographic bound. $\square$
:::

*Interpretation:* External retrieval becomes destabilizing when it pushes the total information content beyond the holographic capacity of the interface. The remedy is to increase interface bandwidth (more sensors) or reduce retrieval intensity.



(sec-causal-isometry-theorem)=
### 28.10 The Causal Isometry Theorem (Cross-Modal Retrieval Resolution)

We prove that if two modalities allow for the solution of the same causal control task, their capacity-constrained geometries must be isometric in the bulk.

:::{prf:theorem} Causal Isometry Theorem
:label: thm-causal-isometry

Let $\mathcal{M}_A$ and $\mathcal{M}_B$ be latent manifolds encoding modalities $A$ and $B$ of a common environment $\mathcal{E}$. Let $\Phi_{\text{causal}}$ be the Causal Information Potential ({ref}`Section 32 <sec-causal-discovery-interventional-geometry-and-the-singularity-of-action>`). If both representations are **Interventionally Closed** ({prf:ref}`thm-interventional-closure`), then the induced metrics $G_A$ and $G_B$ are isometric.

*Proof.*
1. **Metric Genesis:** According to the Capacity-Constrained Metric Law ({prf:ref}`thm-capacity-constrained-metric-law`), the metric $G$ is determined by the solution to the Einstein-like equation $R_{ij} - \frac{1}{2}R G_{ij} + \Lambda G_{ij} = \kappa T_{ij}$, where the stress-energy tensor $T_{ij}$ is derived from the risk Lagrangian $\mathcal{L}_{\text{risk}}$.

2. **Risk Invariance:** The risk Lagrangian $\mathcal{L}_{\text{risk}}(V) = \frac{1}{2}\|\nabla V\|^2 + U(V)$ depends only on the Value function $V$ and the Causal Potential $\Psi_{\text{causal}}$.

3. **Task Invariance:** The potentials $V$ and $\Psi_{\text{causal}}$ are functions of the *causal graph* of the environment $\mathcal{E}$, which is an invariant independent of the sensory modality (pixels vs. tokens).

4. **Uniqueness:** Assuming the solution to the metric field equation is unique (guaranteed for the Poincare disk ansatz in the saturation limit), the geometries $G_A$ and $G_B$ are identical up to a diffeomorphism determined by the encoder parameterization. $\square$
:::

*Interpretation:* Latent representations of the same concept in different modalities (e.g., visual vs. textual) are geometrically isometric because the risk functional governing the metric depends only on the causal structure of the environment, not the sensory channel. This justifies cross-modal retrieval: information retrieved from one modality can inform reasoning in another if both are grounded in the same causal graph.



(sec-symplectic-multi-agent-field-theory)=
