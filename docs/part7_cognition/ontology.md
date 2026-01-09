## 30. Ontological Expansion: Topological Fission and the Semantic Vacuum

This section formalizes the mechanism by which agents expand their ontology—creating new conceptual distinctions—when the existing chart structure proves insufficient. The central object is the **Semantic Vacuum** at the origin $z=0$, where the agent's representation is maximally uncertain. Under **Ontological Stress**, this vacuum becomes unstable and undergoes **Topological Fission**: a pitchfork bifurcation (Theorem {prf:ref}`thm-pitchfork-bifurcation-structure`) that spawns new chart queries.

:::{admonition} Researcher Bridge: Dynamic Architecture vs. Fixed Capacity
:class: tip
:name: rb-dynamic-architecture
Standard models have fixed tensor shapes chosen at initialization. If the environment's complexity exceeds the model's capacity, it fails. **Ontological Fission** is our version of "Dynamic Architecture Growth." When the agent detects "Ontological Stress" (unaccounted-for structure in the noise floor), it triggers a **pitchfork bifurcation** to spawn new latent charts (experts). The model grows to match the data, rather than trying to cram the world into a fixed bottleneck.
:::

**Abstract.** We formalize the expansion of the latent manifold $(\mathcal{Z}, G)$ under representational stress. The **Semantic Vacuum** $\emptyset$ is defined as the fiber over the origin ($z=0$), characterized by maximal $SO(D)$ symmetry. When the residual texture $z_{\mathrm{tex}}$ exhibits temporal predictability—violating **Bulk-Boundary Decoupling** (Axiom {prf:ref}`ax-bulk-boundary-decoupling`)—the manifold undergoes **Topological Fission**: a supercritical pitchfork bifurcation that instantiates new chart queries, expanding the agent's categorical structure.

*Cross-references:* {ref}`Section 21 <sec-radial-generation-entropic-drift-and-policy-control>` (Pitchfork bifurcation, $SO(D)$ symmetry), {ref}`Section 7.8 <sec-tier-the-attentive-atlas>` (Attentive Atlas, chart queries), {ref}`Section 18.2 <sec-main-result>` (Capacity-constrained metric), {ref}`Section 2.11 <sec-variance-value-duality-and-information-conservation>` (Entropy-regularized objective).

*Literature:* Symmetry breaking in dynamical systems {cite}`strogatz2015nonlinear`; Ricci flow {cite}`hamilton1982ricci,perelman2002entropy`; ontology learning {cite}`wong2012`.



(sec-the-semantic-vacuum-as-a-reference-measure)=
### 30.1 The Semantic Vacuum as a Reference Measure

At the origin of the Poincare disk, the agent's belief state is maximally uncertain—all directions are equally probable. This is the **Semantic Vacuum**: the unique fiber over $z=0$ in the latent bundle.

:::{prf:definition} Semantic Vacuum
:label: def-semantic-vacuum

Let $(\mathbb{D}, G)$ be the Poincare disk with metric $G_{ij}(z) = 4\delta_{ij}/(1-|z|^2)^2$ (Definition {prf:ref}`def-hyperbolic-volume-growth`). The **Semantic Vacuum** is the fiber

$$
\emptyset := \{z \in \mathcal{Z} : |z| = 0\} = \{0\} \times \mathcal{Z}_{\text{tex}},
$$
equipped with the following properties:

1. **$SO(D)$ Symmetry:** At $z=0$, the metric is isotropic $G(0) = 4I$ (Proposition {prf:ref}`prop-so-d-symmetry-at-origin`), and the entropic force vanishes: $F_{\text{entropy}}(0) = 0$. The system has full rotational symmetry $SO(D)$.

2. **Infrared Limit:** For any TopoEncoder scale $\tau$ ({ref}`Section 7.12.3 <sec-rigorous-interpretation-renormalization-group-flow>`), $\lim_{\tau \to 0} z(\tau) = \emptyset$. The vacuum is the coarsest resolution.

3. **Reference Measure:** The vacuum carries the Dirac reference measure $\delta_0$ on the bulk coordinates $(K, z_n)$:

   $$
   \mu_{\emptyset} := \delta_0 \otimes \mathcal{N}(0, \sigma_{\text{tex}}^2 I),
   $$
   where the texture component is drawn from the isotropic prior (Definition {prf:ref}`def-boundary-texture-distribution` with $G^{-1}(0) = I/4$).

4. **Information Content:** At the vacuum, $U(0) = 0$ (Definition {prf:ref}`prop-isotropic-radial-expansion`), corresponding to zero information content (maximum entropy).

*Units:* $[\mu_{\emptyset}]$ is a probability measure; $[U] = \mathrm{nat}$.

*Remark (Unstable Equilibrium).* The vacuum is an **unstable fixed point** of the radial dynamics. From Theorem {prf:ref}`thm-pitchfork-bifurcation-structure`, the parameter $\mu = 1/2 > 0$ implies the origin is unstable: any perturbation grows exponentially until noise or policy breaks the symmetry.

:::
:::{prf:lemma} Default Mapping to Vacuum
:label: lem-default-mapping-to-vacuum

Let $\{q_i\}_{i=1}^{N_c}$ be the chart query bank (Definition {prf:ref}`def-attentive-routing-law`) and assume the queries are **centered**: $\sum_{i=1}^{N_c} q_i = 0$. Then for any key $k(x)$ such that all inner products are equal—$\langle q_i, k(x) \rangle = c$ for all $i$—the router weights are uniform:

$$
w_i(x) = \frac{1}{N_c} \quad \forall i \in \{1, \ldots, N_c\}.
$$
The resulting soft codebook embedding is the **barycenter**:

$$
z_q(x) = \sum_{i=1}^{N_c} w_i(x) e_{i, K_{\text{code},i}(x)} = \frac{1}{N_c} \sum_{i=1}^{N_c} e_{i,*},
$$
which equals $0$ if the per-chart codebooks are also centered ($\sum_c e_{i,c} = 0$ for each chart $i$).

*Proof.* From Definition {prf:ref}`def-attentive-routing-law`, $w_i(x) = \exp(\langle q_i, k(x)\rangle/\sqrt{d}) / \sum_j \exp(\langle q_j, k(x)\rangle/\sqrt{d})$. If $\langle q_i, k(x)\rangle = c$ for all $i$, then $w_i = e^{c/\sqrt{d}} / (N_c \cdot e^{c/\sqrt{d}}) = 1/N_c$. The soft code $z_q$ is the weighted sum; under centering, this is the barycenter at $0$. $\square$

*Interpretation.* When the observation $x$ is equally compatible with all charts (or incompatible with all), the router outputs uniform weights. Under centering, this maps to the vacuum—the maximum-entropy state in latent space.

**Architectural Requirement 30.1.3 (Codebook Centering).** To ensure the vacuum is reachable, initialize and regularize codebooks to satisfy $\sum_i q_i = 0$ and $\sum_c e_{i,c} = 0$. This can be enforced via:

$$
\mathcal{L}_{\text{center}} := \left\|\sum_{i=1}^{N_c} q_i\right\|^2 + \sum_{i=1}^{N_c} \left\|\sum_{c=1}^{N_v} e_{i,c}\right\|^2.
$$
:::



(sec-ontological-stress)=
### 30.2 Ontological Stress

The existing chart structure may be insufficient to discriminate observations that differ in task-relevant ways. We quantify this **Ontological Stress** via the conditional mutual information between consecutive texture components.

:::{prf:definition} Ontological Stress
:label: def-ontological-stress

Let $(K_t, z_{n,t}, z_{\text{tex},t})$ be the agent's state at time $t$ (Definition {prf:ref}`def-bounded-rationality-controller`). The **Ontological Stress** is the conditional mutual information:

$$
\Xi := I(z_{\text{tex},t}; z_{\text{tex},t+1} \mid K_t, z_{n,t}, A_t),
$$
where $I(\cdot;\cdot|\cdot)$ denotes conditional mutual information in nats.

*Units:* $[\Xi] = \mathrm{nat}$ (dimensionless information).

*Interpretation.* By Axiom {prf:ref}`ax-bulk-boundary-decoupling` (Bulk-Boundary Decoupling), texture should be unpredictable -- a white-noise residual. If $\Xi > 0$, then texture at time $t$ predicts texture at time $t+1$, conditional on the macro-state and action. This violates the partition condition: the texture channel contains structure that should have been captured by $(K, z_n)$ but was not. The agent's ontology is **too coarse**.

*Cross-reference.* Compare with the closure defect $I(K_{t+1}; Z_t \mid K_t, A_t)$ ({ref}`Section 2.8 <sec-conditional-independence-and-sufficiency>`). Ontological Stress is the dual: predictability *within* texture rather than *from* texture to macro.

:::
:::{prf:theorem} Vacuum Concentration Under Unknown Unknowns
:label: thm-vacuum-concentration-under-unknown-unknowns

Let $\mathcal{F}[p, \pi]$ be the entropy-regularized objective (Definition {prf:ref}`def-entropy-regularized-objective-functional`):

$$
\mathcal{F}[p, \pi] = \int_{\mathcal{Z}} p(z) \Big( V(z) - \tau H(\pi(\cdot|z)) \Big) d\mu_G.
$$
If the value function $V$ is **uninformative** in a region $\Omega \subset \mathcal{Z}$ -- i.e., $\nabla V|_\Omega \approx 0$ and $\nabla^2 V|_\Omega \approx 0$ -- then the entropy term dominates and the optimal belief concentrates toward maximum-entropy configurations:

$$
p^*(z) \propto \exp\left(-\frac{V(z)}{\tau}\right) \xrightarrow{\nabla V \to 0} \text{uniform on } \Omega.
$$
In the Poincare disk geometry, the maximum-entropy state is the vacuum $z = 0$.

*Proof sketch.* The stationary distribution of the Langevin dynamics (Definition {prf:ref}`def-bulk-drift-continuous-flow`) is $p(z) \propto \exp(-\Phi_{\text{eff}}(z)/T_c)$ where $\Phi_{\text{eff}}$ includes the hyperbolic potential $U(z)$. When $V$ is flat, $\Phi_{\text{eff}} \approx U(z) = -2\operatorname{artanh}(|z|)$, which is maximized at $z = 0$. The entropic drift $-\nabla_G U$ vanishes at the origin (Proposition {prf:ref}`def-hyperbolic-information-potential`), making it the unique stationary point. $\square$

*Interpretation.* When encountering observations outside the learned structure, the MaxEnt policy concentrates at the vacuum, correctly representing maximum uncertainty.

*Remark (Capacity Tension).* If belief mass accumulates at the vacuum such that bulk information $I_{\mathrm{bulk}}$ approaches the boundary capacity $C_\partial$ (Theorem {prf:ref}`thm-capacity-constrained-metric-law`), the current chart structure is insufficient. This tension -- high information density at a single point -- indicates fission is required to distribute the representational load.

:::

:::{note} Connection to RL #11: RND as Degenerate Ontological Stress
**The General Law (Fragile Agent):**
**Ontological Stress** measures predictability in the texture channel:

$$
\Xi := I(z_{\text{tex},t}; z_{\text{tex},t+1} \mid K_t, z_{n,t}, A_t)
$$
When $\Xi > \Xi_{\text{crit}}$, the system triggers **topological fission**: a pitchfork bifurcation that expands the chart structure (Section 30.4).

**The Degenerate Limit:**
Set $\Xi_{\text{crit}} \to \infty$ (never fission). Instead, feed $\Xi$ directly into the reward function as an exploration bonus.

**The Special Case (Standard RL):**

$$
r_{\text{RND}} = r + \beta \cdot \|f(s) - \hat{f}(s)\|^2
$$
This recovers **Random Network Distillation (RND)**—prediction error as "curiosity" reward.

**Result:** RND agents get "high" on Ontological Stress but never fix the underlying problem. They explore novel states but don't expand their representational capacity to *understand* them. The Fragile Agent uses $\Xi$ as a diagnostic trigger, not a reward signal.

**What the generalization offers:**
- Structural response: high $\Xi$ triggers chart fission, expanding ontology
- Principled threshold: $\Xi_{\text{crit}}$ balances exploration cost vs. complexity cost
- No reward hacking: exploration is architectural, not incentive-based
:::



(sec-the-fission-criterion)=
### 30.3 The Fission Criterion

Not all ontological stress justifies expansion. Creating new charts incurs **complexity costs** (additional parameters, increased inference time). We formalize when expansion is warranted.

:::{prf:axiom} Ontological Expansion Principle
:label: ax-ontological-expansion-principle

The agent should expand its chart structure (increase $N_c$) if and only if the expected value improvement exceeds the complexity cost:

$$
\mathbb{E}\left[\Delta V \mid \text{fission}\right] > \mathcal{C}_{\text{complexity}}(N_c \to N_c + 1),
$$
where $\Delta V$ is the value gain from finer discrimination and $\mathcal{C}_{\text{complexity}}$ is measured in nats (to match units with value).

*Remark.* This is the MDL/rate-distortion principle ({ref}`Section 2.2b <sec-the-shutter-as-a-vq-vae>`) applied to ontology: expand only if the distortion reduction exceeds the rate increase.

:::
:::{prf:theorem} Fission Criterion
:label: thm-fission-criterion

Let $\Xi$ be the Ontological Stress (Definition {prf:ref}`def-ontological-stress`) and let $\Xi_{\text{crit}} > 0$ be a threshold. Let $\Delta V_{\text{proj}}$ be the projected value improvement from splitting the highest-stress chart. The fission criterion is:

$$
\text{Fission} \iff \Xi > \Xi_{\text{crit}} \quad \text{AND} \quad \Delta V_{\text{proj}} > \mathcal{C}_{\text{complexity}}.
$$
*Units:* All quantities are in nats. The complexity cost $\mathcal{C}_{\text{complexity}}(N_c \to N_c + 1)$ includes the entropy increase $\log((N_c+1)/N_c)$ from the expanded codebook plus any regularization penalty on parameter count.

:::

:::{note} Connection to RL #12: Fixed Architecture as Degenerate Fission
**The General Law (Fragile Agent):**
When Ontological Stress $\Xi > \Xi_{\text{crit}}$, the system triggers a **pitchfork bifurcation** that spawns new chart queries:

$$
\frac{dr}{ds} = (\Xi - \Xi_{\text{crit}}) r - \alpha r^3 + \sigma\xi
$$
The network topology **expands** to accommodate new conceptual distinctions.

**The Degenerate Limit:**
Set $\Xi_{\text{crit}} \to \infty$ (infinite fission threshold). The network never expands.

**The Special Case (Standard Deep Learning):**

$$
|\theta| = \text{const} \quad \text{(parameter count fixed at initialization)}
$$
This recovers **standard deep learning**—the agent can never learn a concept that doesn't fit in its initial tensor shapes.

**Result:** Standard networks face a capacity crisis: they must either (1) compress concepts beyond recognition, or (2) fail silently when encountering structure beyond their representational budget. The Fragile Agent grows its ontology to match task complexity.

**What the generalization offers:**
- Dynamic capacity: network topology adapts to task requirements
- Principled growth: fission occurs when complexity cost is justified by value gain
- No catastrophic forgetting: new charts are topologically isolated (Section 30.7)
:::



(sec-symmetry-breaking-and-chart-birth)=
### 30.4 Symmetry Breaking and Chart Birth

When the Fission Criterion is satisfied, the agent creates a new chart by splitting an existing query vector. This process is a **pitchfork bifurcation** in the space of chart queries, extending the structure established in Theorem {prf:ref}`thm-pitchfork-bifurcation-structure`.

:::{prf:definition} Query Fission
:label: def-query-fission

Let $q_i \in \mathbb{R}^d$ be a chart query vector ({ref}`Section 7.8 <sec-tier-the-attentive-atlas>`) with associated codebook $\{e_{i,c}\}_{c=1}^{N_v}$. A **query fission** replaces $q_i$ with two daughter queries:

$$
q_i \mapsto \{q_i^+, q_i^-\} := \{q_i + \epsilon u, q_i - \epsilon u\},
$$
where $u \in \mathbb{R}^d$ is the **fission direction** (unit vector) and $\epsilon > 0$ is the **fission amplitude**.

The daughter codebooks are initialized as copies:

$$
e_{i^\pm, c} := e_{i, c} \quad \forall c \in \{1, \ldots, N_v\}.
$$
*Selection of fission direction.* The optimal $u$ maximizes the variance of router assignments under the new queries:

$$
u^* = \arg\max_{\|u\|=1} \text{Var}_{x \sim \mathcal{D}}\left[\langle k(x), u \rangle \mid w_i(x) > 1/N_c\right],
$$
i.e., the principal component of keys within the chart's Voronoi cell.

:::
:::{prf:theorem} Supercritical Pitchfork Bifurcation for Charts
:label: thm-supercritical-pitchfork-bifurcation-for-charts

The query fission dynamics exhibit the **supercritical pitchfork bifurcation** structure of Theorem {prf:ref}`thm-pitchfork-bifurcation-structure`. Let $r := \|q_i^+ - q_i^-\|/2 = \epsilon$ be the half-separation of daughter queries. The radial evolution satisfies:

$$
\frac{dr}{ds} = (\Xi - \Xi_{\text{crit}}) r - \alpha r^3 + \sigma\xi,
$$
where:
- $\Xi - \Xi_{\text{crit}}$ plays the role of the bifurcation parameter $\mu$ in Theorem {prf:ref}`thm-pitchfork-bifurcation-structure`
- $\alpha > 0$ is a stabilizing cubic coefficient (from competition for training data)
- $\sigma\xi$ is noise from stochastic gradient updates
- $s$ is the training step (flow time)

**Phase Transition:**
1. **Sub-critical ($\Xi < \Xi_{\text{crit}}$):** $r=0$ is the unique stable fixed point. The daughters collapse back to the parent ($r \to 0$).
2. **Super-critical ($\Xi > \Xi_{\text{crit}}$):** $r=0$ becomes unstable. The daughters separate toward a new equilibrium:

   $$
   r^* = \sqrt{\frac{\Xi - \Xi_{\text{crit}}}{\alpha}}.
   $$
*Proof.* The dynamics derive from the effective potential:

$$
\Phi_{\text{fission}}(r) = -\frac{(\Xi - \Xi_{\text{crit}})}{2} r^2 + \frac{\alpha}{4} r^4,
$$
which has the standard pitchfork form. For $\Xi > \Xi_{\text{crit}}$, the origin has $\Phi_{\text{fission}}''(0) = -(\Xi - \Xi_{\text{crit}}) < 0$, becoming unstable. Stable minima appear at $r = \pm r^*$. The cubic term arises from router saturation: as daughters separate, they compete for data, and the loss landscape penalizes excessive separation. This matches the normal form of Theorem {prf:ref}`thm-pitchfork-bifurcation-structure` with $\mu = \Xi - \Xi_{\text{crit}}$. $\square$

*Critical Temperature Constraint.* From Theorem {prf:ref}`thm-pitchfork-bifurcation-structure`, the critical temperature $T_c^* = 1/16$ implies that thermal fluctuations can restore symmetry (collapse daughters) if cognitive temperature exceeds the barrier height. For stable fission, require:

$$
T_c < \frac{(\Xi - \Xi_{\text{crit}})^2}{4\alpha}.
$$
:::



(sec-metric-relaxation-ontological-ricci-flow)=
### 30.5 Metric Relaxation: Ontological Ricci Flow

Following fission, the metric tensor $G$ must adapt to the new chart structure. We introduce a geometric flow that relaxes the metric toward consistency with the expanded ontology.

:::{prf:definition} Ontological Ricci Flow
:label: def-ontological-ricci-flow

Let $G_{ij}(z, s)$ be the capacity-constrained metric (Theorem {prf:ref}`thm-capacity-constrained-metric-law`) parameterized by flow time $s$. Define the **local stress field** $\Xi(z) := \mathbb{E}[\Xi \mid K = k(z)]$, where $k(z)$ is the chart containing $z$. The **Ontological Ricci Flow** is:

$$
\frac{\partial G_{ij}}{\partial s} = -2\left(R_{ij} - \frac{1}{2}R\, G_{ij} + \Lambda G_{ij} - \kappa T_{ij}\right) + \nu \nabla_i \nabla_j \Xi(z),
$$
where:
- $R_{ij}$ is the Ricci curvature tensor, $R = G^{ij}R_{ij}$ the scalar curvature
- $\Lambda, \kappa$ are constants from Theorem {prf:ref}`thm-capacity-constrained-metric-law`
- $T_{ij}$ is the risk tensor
- $\nu > 0$ is the stress-curvature coupling constant

*Units:* $[\partial G / \partial s] = [z]^{-2}$; $[\Xi] = \text{nat}$; $[\nabla_i \nabla_j \Xi] = \text{nat}/[z]^2$.

*Interpretation.* The first term drives the metric toward the capacity-constrained fixed point. The second term $\nu \nabla_i \nabla_j \Xi$ introduces curvature in regions of high stress gradient, expanding the metric where new distinctions are needed.

:::

::::{admonition} Physics Isomorphism: Ricci Flow
:class: note
:name: pi-ricci-flow

**In Physics:** Hamilton's Ricci flow evolves a Riemannian metric toward constant curvature: $\partial_t g_{ij} = -2R_{ij}$. It was used by Perelman to prove the Poincare conjecture {cite}`hamilton1982ricci,perelman2002entropy`.

**In Implementation:** The Ontological Ricci Flow (Definition {prf:ref}`def-ontological-ricci-flow`) evolves the latent metric:

$$
\frac{\partial G_{ij}}{\partial s} = -2\left(R_{ij} - \frac{1}{2}R\,G_{ij} + \Lambda G_{ij} - \kappa T_{ij}\right) + \nu\nabla_i\nabla_j\Xi
$$
**Correspondence Table:**

| Differential Geometry | Agent (Ontological Flow) |
|:----------------------|:-------------------------|
| Metric evolution $\partial_t g$ | Metric adaptation $\partial_s G$ |
| Ricci curvature $R_{ij}$ | Ricci curvature of $G$ |
| Flow singularities | Chart fission events |
| Entropy monotonicity | Ontological stress reduction |

**Fixed Point:** The capacity-constrained metric law + vanishing stress Hessian.
::::

:::{prf:proposition} Fixed Points of Ontological Ricci Flow
:label: prop-fixed-points-of-ontological-ricci-flow

The flow has fixed points when:
1. The capacity-constrained metric law is satisfied: $R_{ij} - \frac{1}{2}R\,G_{ij} + \Lambda G_{ij} = \kappa T_{ij}$
2. The Ontological Stress has vanishing Hessian: $\nabla_i \nabla_j \Xi = 0$

Condition (2) is satisfied when either $\Xi$ is constant (uniform stress) or $\Xi = 0$ (no stress).

*Computational Proxy.* In practice, we do not solve the Ricci flow PDE. The squared residual of the fixed-point condition can be used as a regularization loss:

$$
\mathcal{L}_{\text{Ricci}} := \left\|R_{ij} - \frac{1}{2}R\,G_{ij} + \Lambda G_{ij} - \kappa T_{ij}\right\|_F^2 + \nu^2 \|\nabla_i \nabla_j \Xi\|_F^2,
$$
encouraging the learned metric to satisfy the capacity constraint while penalizing stress gradients.

:::



(sec-diagnostic-nodes-a)=
### 30.6 Diagnostic Nodes 49–50

Following the diagnostic node convention ({ref}`Section 3.1 <sec-theory-thin-interfaces>`), we define two new monitors for ontological expansion.

(node-49)=
**Node 49: OntologicalStressCheck**

| **#**  | **Name**                   | **Component** | **Type**                     | **Interpretation**        | **Proxy**                                                               | **Cost**                      |
|--------|----------------------------|---------------|------------------------------|---------------------------|-------------------------------------------------------------------------|-------------------------------|
| **49** | **OntologicalStressCheck** | Atlas         | Representational Sufficiency | Is texture unpredictable? | $\Xi := I(z_{\text{tex},t}; z_{\text{tex},t+1} \mid K_t, z_{n,t}, A_t)$ | $O(B \cdot d_{\text{tex}}^2)$ |

**Interpretation:** Monitors the conditional mutual information between consecutive texture components. High $\Xi$ indicates the texture channel contains predictable structure that should be in the macro-state.

**Threshold:** $\Xi < \Xi_{\text{tol}}$ (typical default $\Xi_{\text{tol}} = 0.1$ nat).

**Trigger conditions:**
- High OntologicalStressCheck ($\Xi > \Xi_{\text{tol}}$): The current chart structure is insufficient. Consider query fission (Definition {prf:ref}`def-query-fission`).
- Persistent high stress after fission: The fission direction $u$ may be suboptimal; recompute via PCA on high-stress keys.

**Computational Proxy:** Estimate $\Xi$ via a variational bound using a learned texture predictor:

$$
\hat{\Xi} = \mathbb{E}\left[\log p_\phi(z_{\text{tex},t+1} \mid z_{\text{tex},t}, K_t, z_{n,t}, A_t) - \log p_\phi(z_{\text{tex},t+1} \mid K_t, z_{n,t}, A_t)\right],
$$
where $p_\phi$ is a small MLP. If $\hat{\Xi} \approx 0$, texture is unpredictable and the firewall holds.

*Cross-reference:* Extends TextureFirewallCheck (Node 29) from measuring $\|\partial_{z_{\text{tex}}} \dot{z}\|$ (static leak) to measuring $I(z_{\text{tex},t}; z_{\text{tex},t+1})$ (temporal structure).



(node-50)=
**Node 50: FissionReadinessCheck**

| **#**  | **Name**                  | **Component** | **Type**            | **Interpretation**      | **Proxy**                                                                                                        | **Cost**         |
|--------|---------------------------|---------------|---------------------|-------------------------|------------------------------------------------------------------------------------------------------------------|------------------|
| **50** | **FissionReadinessCheck** | Atlas         | Expansion Criterion | Should ontology expand? | $\mathbb{I}(\Xi > \Xi_{\text{crit}}) \cdot \mathbb{I}(\Delta V_{\text{proj}} > \mathcal{C}_{\text{complexity}})$ | $O(N_c \cdot B)$ |

**Interpretation:** Monitors both conditions of the fission criterion (Theorem {prf:ref}`thm-fission-criterion`). Returns 1 if fission is warranted, 0 otherwise.

**Threshold:** Binary—if FissionReadinessCheck = 1, initiate query fission.

**Trigger conditions:**
- FissionReadinessCheck = 1: Execute query fission procedure.
- FissionReadinessCheck = 0 but $\Xi$ increasing: Pre-emptively compute fission direction for warm-start.

**Remediation:**
- If repeatedly triggering fission: The base architecture may be too constrained. Increase $N_v$ (codes per chart) before increasing $N_c$ (chart count).
- If fission fails to reduce $\Xi$: The fission direction missed the relevant structure. Use supervised signal (if available) to guide $u$.



(sec-summary-the-lifecycle-of-an-ontology)=
### 30.7 Summary: The Lifecycle of an Ontology

**Table 30.7.1 (Ontological Expansion Summary).**

| Concept                   | Definition/Reference                                                                                              | Units      | Diagnostic |
|:--------------------------|:------------------------------------------------------------------------------------------------------------------|:-----------|:-----------|
| **Semantic Vacuum**       | $\emptyset = \{z : \lVert z\rVert = 0\}$ (Def {prf:ref}`def-semantic-vacuum`)                                     | —          | —          |
| **Ontological Stress**    | $\Xi = I(z_{\text{tex},t}; z_{\text{tex},t+1} \mid K_t, z_{n,t}, A_t)$ (Def {prf:ref}`def-ontological-stress`)    | nat        | Node 49    |
| **Fission Criterion**     | $\Xi > \Xi_{\text{crit}}$ AND $\Delta V > \mathcal{C}_{\text{complexity}}$ (Thm {prf:ref}`thm-fission-criterion`) | —          | Node 50    |
| **Query Fission**         | $q_i \mapsto \{q_i + \epsilon u, q_i - \epsilon u\}$ (Def {prf:ref}`def-query-fission`)                           | —          | —          |
| **Bifurcation Parameter** | $\mu = \Xi - \Xi_{\text{crit}}$ (Thm {prf:ref}`thm-supercritical-pitchfork-bifurcation-for-charts`)               | nat        | —          |
| **Ricci Flow**            | $\partial_s G = -2(\text{Einstein tensor}) + \nu \nabla^2 \Xi$ (Def {prf:ref}`def-ontological-ricci-flow`)        | $[z]^{-2}$ | —          |

**The Ontological Lifecycle:**

1. **Equilibrium:** The agent separates signal ($K, z_n$) from residual ($z_{\text{tex}}$). $\Xi \approx 0$.
2. **Stress Accumulation:** New data types appear; $z_{\text{tex}}$ becomes predictable. $\Xi$ rises.
3. **Saturation:** Unclassified observations accumulate at $z=0$.
4. **Bifurcation:** Fission criterion met; new query $q_*$ instantiated.
5. **Separation:** Daughter queries separate toward equilibrium $r^*$.
6. **Stabilization:** Metric relaxes to accommodate new chart structure.

**Conclusion.** Ontological expansion is a geometric response to representational insufficiency. The framework provides a principled criterion for when to expand chart structure (Theorem {prf:ref}`thm-fission-criterion`) and predicts the dynamics of chart separation via pitchfork bifurcation (Theorem {prf:ref}`thm-supercritical-pitchfork-bifurcation-for-charts`).

:::{note} Connection to RL #13: EWC as Degenerate Atlas
**The General Law (Fragile Agent):**
The agent maintains an **Atlas of Charts** $\{(\mathcal{U}_i, \phi_i)\}_{i=1}^{N_c}$:
- New tasks trigger **Fission**: create new chart $\mathcal{U}_{N_c+1}$ via pitchfork bifurcation
- Old charts are **topologically isolated**: transition maps prevent gradient flow between charts
- Parameters in different charts don't interfere

**The Degenerate Limit:**
Force the agent to use a **single chart** (one neural network). Add a quadratic penalty to prevent weights from moving: $\mathcal{L}_{\text{EWC}} = \frac{\lambda}{2} \sum_i F_i (\theta_i - \theta_i^*)^2$.

**The Special Case (Continual Learning):**

$$
\mathcal{L}_{\text{EWC}} = \mathcal{L}_{\text{task}} + \frac{\lambda}{2} \sum_i F_i (\theta_i - \theta_i^*)^2
$$
This recovers **Elastic Weight Consolidation (EWC)** -- the Fisher information $F_i$ acts as an "importance weight" preventing catastrophic forgetting.

**Result:** EWC tries to cram a complex manifold into a single flat coordinate system. Catastrophic forgetting occurs because there's no **topological glue** isolating old from new -- just a soft penalty that eventually breaks under task pressure.

**What the generalization offers:**
- True isolation: chart transitions prevent gradient interference, not just penalties
- Principled expansion: new charts created when fission criterion met
- No forgetting: old charts are frozen, not elastically constrained
:::



(sec-ontological-fusion-concept-consolidation)=
### 30.8 Ontological Fusion: Concept Consolidation

*Abstract.* If Fission ({ref}`Section 30.4 <sec-symmetry-breaking-and-chart-birth>`) is the birth of a concept driven by ontological stress, **Fusion** is the death or merging of concepts driven by **metabolic efficiency**. Without Fusion, the agent suffers from **topological heat death**: unbounded chart fragmentation where every observation eventually gets its own private chart, destroying generalization. Fusion is triggered when the **Discrimination Gain** of keeping two charts separate falls below the **Metabolic Cost** of maintaining them.

:::{admonition} Researcher Bridge: Pruning via Metabolic Efficiency
:class: important
:name: rb-pruning-efficiency
Most MoE (Mixture of Experts) or multi-chart models suffer from "Expert Explosion," where they create a new index for every minor variation. **Ontological Fusion** provides a principled way to forget. It merges latent charts when the **Discrimination Gain** (the information provided by keeping them separate) falls below the **Metabolic Cost** of maintaining them. It is the geometric derivation of Occam's Razor.
:::

*Cross-references:* This section addresses Open Problem 1 from Section 30.7. It is the dual of Section 30.4 (Fission) and connects to the Universal Governor's metabolic monitoring ({ref}`Section 26 <sec-theory-of-meta-stability-the-universal-governor-as-homeostatic-controller>`) and the complexity cost functional ({ref}`Section 30.3 <sec-the-fission-criterion>`).



(sec-ontological-redundancy)=
#### 30.8.1 Ontological Redundancy

We define a measure of functional similarity between charts that captures whether two charts are semantically interchangeable.

:::{prf:definition} Ontological Redundancy
:label: def-ontological-redundancy

Let $K_i$ and $K_j$ be two charts with associated belief distributions $\mu_i, \mu_j$, transition models $\bar{P}_i, \bar{P}_j$, and value functions $V_i, V_j$. Their **ontological redundancy** is:

$$
\Upsilon_{ij} := \exp\left(-\left[ d_{\text{WFR}}(\mu_i, \mu_j) + D_{\mathrm{KL}}(\bar{P}_i \| \bar{P}_j) + \|V_i - V_j\|_G^2 \right]\right)
$$
where:
- $d_{\text{WFR}}(\mu_i, \mu_j)$ is the Wasserstein-Fisher-Rao distance between belief distributions (Definition {prf:ref}`def-the-wfr-action`),
- $D_{\mathrm{KL}}(\bar{P}_i \| \bar{P}_j) := \mathbb{E}_{k \sim \mu_i}\left[ D_{\mathrm{KL}}(\bar{P}(\cdot|k, a) \| \bar{P}_j(\cdot|k, a)) \right]$ is the mean predictive divergence,
- $\|V_i - V_j\|_G^2 := \mathbb{E}_{z \sim \mu_i}\left[ (V_i(z) - V_j(z))^2 \right]$ is the mean squared value divergence.

*Units:* Dimensionless; $\Upsilon_{ij} \in [0, 1]$.

*Interpretation:* $\Upsilon_{ij} \to 1$ implies the charts are functionally redundant: they occupy similar regions of belief space, predict similar futures, and assign similar values. $\Upsilon_{ij} \to 0$ implies they are functionally distinct.
:::



(sec-discrimination-gain)=
#### 30.8.2 Discrimination Gain

Before destroying a chart, the agent must estimate the information loss.

:::{prf:definition} Discrimination Gain
:label: def-discrimination-gain

The **Discrimination Gain** $G_\Delta(i, j)$ is the mutual information the agent loses about observations by merging charts $i$ and $j$:

$$
G_\Delta(i, j) := I(X; \{K_i, K_j\}) - I(X; K_{i \cup j})
$$
where $K_{i \cup j}$ is the merged chart that routes observations previously assigned to $K_i$ or $K_j$ to a single index.

*Units:* nat.

*MDL interpretation:* $G_\Delta$ is the increase in **distortion** (description length) resulting from the merge. If $G_\Delta \approx 0$, the distinction between $K_i$ and $K_j$ carries negligible information about the observation stream.
:::

:::{prf:lemma} Redundancy-Gain Relationship
:label: lem-redundancy-gain

Under the assumption that charts partition the observation space and the encoder is deterministic given observation $x$:

$$
G_\Delta(i, j) \leq H(K_i, K_j) - H(K_{i \cup j}) = \log 2 - H(K_i | K_j) \cdot \mathbb{I}[\Upsilon_{ij} < 1]
$$
When $\Upsilon_{ij} \to 1$, the bound tightens: $G_\Delta \to 0$.

*Proof sketch.* The discrimination gain is upper-bounded by the entropy reduction from merging. When charts are redundant ($\Upsilon_{ij} \to 1$), they route to the same observations with high probability, so the conditional entropy $H(K_i | K_j) \to 0$. $\square$
:::



(sec-the-fusion-criterion)=
### 30.9 The Fusion Criterion

Fusion is the dual of Fission. Where Fission is triggered by high ontological stress ({prf:ref}`thm-fission-criterion`), Fusion is triggered by high redundancy and low discrimination gain.

:::{prf:axiom} Ontological Simplification Principle
:label: ax-ontological-simplification

The agent shall reduce ontological complexity when the expected value of maintaining a distinction is negative:

$$
\mathcal{C}_{\text{saved}}(N_c \to N_c - 1) > G_\Delta(i, j) + \mathbb{E}[\Delta V \mid \text{no fusion}]
$$
where $\mathcal{C}_{\text{saved}}$ is the metabolic savings from eliminating a chart.

*Remark.* This is the dual of {prf:ref}`ax-ontological-expansion-principle` (Ontological Expansion Principle). Both derive from the same MDL objective: minimize description length plus expected regret.
:::

:::{prf:theorem} Fusion Criterion
:label: thm-fusion-criterion

Charts $i$ and $j$ shall be merged if and only if:

$$
G_\Delta(i, j) < \mathcal{C}_{\text{complexity}}(N_c) - \mathcal{C}_{\text{complexity}}(N_c - 1) + \epsilon_{\text{hysteresis}}
$$
where:
- $\mathcal{C}_{\text{complexity}}(N_c) = \log N_c + \lambda_{\text{param}} |\theta_{\text{chart}}|$ is the metabolic cost of maintaining $N_c$ charts ({ref}`Section 30.3 <sec-the-fission-criterion>`),
- $\epsilon_{\text{hysteresis}} > 0$ is a hysteresis constant preventing oscillatory fission-fusion ("ontological churn").

*Proof sketch.* By {prf:ref}`ax-ontological-simplification`, fusion is justified when saved complexity exceeds lost discrimination. The complexity difference is:

$$
\mathcal{C}_{\text{complexity}}(N_c) - \mathcal{C}_{\text{complexity}}(N_c - 1) = \log\frac{N_c}{N_c - 1} + \lambda_{\text{param}} |\theta_{\text{chart}}|
$$
The hysteresis term $\epsilon_{\text{hysteresis}}$ breaks the symmetry with Fission, ensuring that a chart is not immediately re-created after being destroyed. $\square$

*Remark (Units):* All terms are in nats. The criterion is dimensionally consistent.
:::



(sec-topological-collapse-the-mechanism-of-fusion)=
### 30.10 Topological Collapse: The Mechanism of Fusion

Once the Fusion Criterion is met, the agent must physically merge the charts. This is not simple deletion—it is **topological surgery**.



(sec-query-coalescence)=
#### 30.10.1 Query Coalescence

:::{prf:definition} Query Coalescence
:label: def-query-coalescence

Given charts $i, j$ satisfying the Fusion Criterion ({prf:ref}`thm-fusion-criterion`), the merged query is the **usage-weighted barycenter**:

$$
q_{\text{merged}} := \frac{\bar{w}_i q_i + \bar{w}_j q_j}{\bar{w}_i + \bar{w}_j}
$$
where $\bar{w}_k := \mathbb{E}[w_k(x)]$ is the historical routing weight from the Attentive Atlas ({prf:ref}`def-attentive-routing-law`).

*Interpretation:* The more frequently used chart contributes more to the merged query position. This preserves the routing behavior for the majority of observations.
:::



(sec-fiber-reconciliation-via-jump-operators)=
#### 30.10.2 Fiber Reconciliation via Jump Operators

When merging chart $j$ into chart $i$, observations previously routed to $j$ must be re-embedded in $i$'s coordinate system.

:::{prf:definition} Fiber Reconciliation
:label: def-fiber-reconciliation

Let $L_{j \to i}: \mathcal{F}_j \to \mathcal{F}_i$ be the factorized jump operator ({prf:ref}`def-factorized-jump-operator`). For an observation $x$ previously assigned to chart $j$ with nuisance coordinates $z_n^{(j)}$, the reconciled coordinates in chart $i$ are:

$$
z_n^{(i, \text{reconciled})} := L_{j \to i}(z_n^{(j)}) = A_i(B_j z_n^{(j)} + c_j) + d_i
$$
where $B_j$ is the chart-to-global encoder and $A_i$ is the global-to-chart decoder.

*Codebook reconciliation:* The codebook entries of chart $j$ are projected into chart $i$'s Voronoi structure. Entries that fall within existing Voronoi cells of chart $i$ are absorbed; entries that create new structure may be retained if codebook capacity permits.
:::



(sec-subcritical-bifurcation-dynamics)=
#### 30.10.3 Subcritical Bifurcation Dynamics

Fusion is modeled as a **subcritical pitchfork bifurcation**—the dual of the supercritical bifurcation governing Fission.

:::{prf:theorem} Subcritical Pitchfork for Fusion
:label: thm-subcritical-pitchfork-fusion

Let $r(s) := \|q_i(s) - q_j(s)\|$ be the query separation at computation time $s$. During fusion, the dynamics become:

$$
\frac{dr}{ds} = -(\Upsilon_{ij} - \Upsilon_{\text{crit}}) r - \alpha r^3 + \sigma\xi(s)
$$
where:
- $\Upsilon_{\text{crit}} \in (0, 1)$ is the critical redundancy threshold,
- $\alpha > 0$ is the cubic stabilization coefficient,
- $\sigma\xi(s)$ is white noise with intensity $\sigma$.

When $\Upsilon_{ij} > \Upsilon_{\text{crit}}$:
1. The linear term is **negative** (attractive toward $r = 0$).
2. $r = 0$ becomes the **unique stable attractor**.
3. The queries "fall into each other" until they merge.

*Contrast with Fission ({prf:ref}`thm-supercritical-pitchfork-bifurcation-for-charts`):*

| Property                | Fission (Supercritical)          | Fusion (Subcritical)                |
|:------------------------|:---------------------------------|:------------------------------------|
| Linear term sign        | $+\mu r$ (repulsive from origin) | $-\mu r$ (attractive to origin)     |
| Trigger                 | $\Xi > \Xi_{\text{crit}}$        | $\Upsilon > \Upsilon_{\text{crit}}$ |
| Stable fixed points     | $r^* = \pm\sqrt{\mu/\alpha}$     | $r^* = 0$                           |
| Physical interpretation | Charts repel and separate        | Charts attract and merge            |

*Proof sketch.* The bifurcation structure follows from standard dynamical systems theory {cite}`strogatz2018nonlinear`. The key insight is that Fission and Fusion are **dual bifurcations**: Fission breaks $\mathbb{Z}_2$ symmetry (one chart → two); Fusion restores it (two charts → one). The sign flip in the linear term corresponds to the duality between expansion ($\Xi$) and contraction ($\Upsilon$) forces. $\square$
:::



(sec-diagnostic-nodes-fusion-and-codebook-liveness)=
### 30.11 Diagnostic Nodes 54–55: Fusion and Codebook Liveness

We introduce two new diagnostic nodes for the Sieve ({ref}`Section 3 <sec-diagnostics-stability-checks>`).



:::{prf:definition} Node 54 — FusionReadinessCheck
:label: node-fusion-readiness-check

**Component:** Atlas (Chart Router)

**Type:** Metabolic Efficiency

**Interpretation:** Are any two charts functionally redundant?

**Proxy:**

$$
\text{FusionReady} := \mathbb{I}\left[ \max_{i \neq j} \Upsilon_{ij} > \Upsilon_{\text{crit}} \right]
$$
**Computational cost:** $O(N_c^2)$ pairwise comparisons.

**Trigger condition:** Two or more charts have redundancy exceeding threshold.

**Remediation:**
1. Identify most redundant pair $(i^*, j^*) = \arg\max_{i \neq j} \Upsilon_{ij}$.
2. Verify Fusion Criterion ({prf:ref}`thm-fusion-criterion`).
3. If satisfied, initiate subcritical bifurcation dynamics.
4. Execute Query Coalescence and Fiber Reconciliation.
5. Decrement chart count: $N_c \to N_c - 1$.
:::



:::{prf:definition} Node 55 — CodebookLivenessCheck
:label: node-codebook-liveness-check

**Component:** Codebook (VQ Layer)

**Type:** Dead Code Detection

**Interpretation:** Are any code indices unused?

**Proxy:**

$$
\text{DeadCodeDetected} := \mathbb{I}\left[ \min_k P(K = k) < \epsilon_{\text{dead}} \right]
$$
where $P(K = k)$ is the empirical usage frequency of code $k$ over a trailing window.

**Computational cost:** $O(|\mathcal{K}|)$.

**Trigger condition:** Code usage falls below minimum threshold (default $\epsilon_{\text{dead}} = 10^{-4}$).

**Remediation:** Execute Lazarus Protocol ({prf:ref}`alg-lazarus`).

*Connection to existing diagnostics:* This node operationalizes the dead-code tolerance constraint from {ref}`Section 3.5.5 <sec-calibrating-tolerances>`: $H(K) \geq \log((1 - \rho_{\text{dead}})|\mathcal{K}|)$.
:::

**Summary Table:**

| #      | Name                  | Component | Type                 | Proxy                                                    | Cost       |
|--------|-----------------------|-----------|----------------------|----------------------------------------------------------|------------|
| **54** | FusionReadinessCheck  | Atlas     | Metabolic Efficiency | $\max_{i \neq j} \Upsilon_{ij} > \Upsilon_{\text{crit}}$ | $O(N_c^2)$ |
| **55** | CodebookLivenessCheck | Codebook  | Dead Code Detection  | $\min_k P(K=k) < \epsilon_{\text{dead}}$                 | $O(\lvert\mathcal{K}\rvert)$ |



(sec-symbolic-metabolism-intra-chart-fission-and-fusion)=
### 30.12 Symbolic Metabolism: Intra-Chart Fission and Fusion

While Sections 30.1–30.11 address **chart-level** (macro) topology, the codebook symbols **within** each chart also require lifecycle management. This creates a two-level metabolic hierarchy.



(sec-symbol-fission-cluster-splitting)=
#### 30.12.1 Symbol Fission: Cluster Splitting

Symbol fission occurs when a single code index $k$ is **overloaded**—representing two or more geometrically distinct clusters.

:::{prf:definition} Intra-Symbol Variance (Geometric Tension)
:label: def-intra-symbol-variance

For code $e_k$ in chart $i$, the **geometric tension** is:

$$
\sigma_k^2 := \mathbb{E}\left[ \|z_e - e_k\|^2 \;\Big|\; \text{VQ}(z_e) = k \right]
$$
where $z_e$ is the pre-quantized encoder output.

*Units:* $[z]^2$ (squared latent units).

*Interpretation:* High $\sigma_k^2$ indicates the symbol is overloaded—its Voronoi cell contains multiple distinct clusters that should be separated.
:::

**Symbol Fission Mechanism:**

1. **Detect tension:** If $\sigma_k^2 > \sigma_{\text{crit}}^2$, mark code $k$ for fission.
2. **Compute split direction:** Find the principal eigenvector $v_1$ of the conditional covariance:

   $$
   \Sigma_k := \mathbb{E}\left[ (z_e - e_k)(z_e - e_k)^\top \;\Big|\; \text{VQ}(z_e) = k \right]
   $$
3. **Instantiate daughter codes:**

   $$
   e_{k,+} := e_k + \epsilon v_1, \qquad e_{k,-} := e_k - \epsilon v_1
   $$
   where $\epsilon = \sqrt{\lambda_1 / 2}$ and $\lambda_1$ is the principal eigenvalue.
4. **Capacity check:** If the codebook is full, trigger Symbol Fusion elsewhere to free a slot.



(sec-symbol-fusion-synonym-merging)=
#### 30.12.2 Symbol Fusion: Synonym Merging

Symbol fusion is the **generalization** step—merging symbols that are functionally indistinguishable.

:::{prf:definition} Functional Indistinguishability
:label: def-functional-indistinguishability

Two symbols $k_1, k_2$ within the same chart are fusion candidates if the **policy divergence** and **value gap** are negligible:

$$
\mathcal{D}_f(k_1, k_2) := D_{\mathrm{KL}}\left( \pi(\cdot | k_1) \| \pi(\cdot | k_2) \right) + |V(k_1) - V(k_2)|
$$
If $\mathcal{D}_f(k_1, k_2) < \epsilon_{\text{indist}}$, the distinction provides no **control authority**.

*Units:* nat.

*Interpretation:* Symbols are functionally indistinguishable when the policy and value function treat them identically.
:::

**Symbol Fusion Mechanism:**

1. **Coalesce embeddings:**

   $$
   e_{\text{merged}} := \frac{1}{2}(e_{k_1} + e_{k_2})
   $$
2. **Remap transitions:** Update all entries in the world model $\bar{P}$ that reference $k_1$ or $k_2$ to point to the merged index.
3. **Free slot:** Return one index to the available pool for future Symbol Fission.



(sec-the-lazarus-protocol-dead-code-reallocation)=
#### 30.12.3 The Lazarus Protocol: Dead Code Reallocation

In standard VQ-VAEs, **codebook collapse** is a major failure mode where most codes are never used. The Lazarus Protocol recycles dead codes to high-information-density regions.

:::{prf:algorithm} Lazarus Reallocation
:label: alg-lazarus

**Input:** Dead code $k_{\text{dead}}$ with $P(K = k_{\text{dead}}) < \epsilon_{\text{dead}}$.

**Procedure:**
1. Find the most stressed symbol:

   $$
   k_{\text{stressed}} := \arg\max_k \sigma_k^2
   $$
2. Perform Symbol Fission on $k_{\text{stressed}}$, reusing index $k_{\text{dead}}$:
   - Compute split direction $v_1$ from $\Sigma_{k_{\text{stressed}}}$.
   - Set $e_{k_{\text{dead}}} := e_{k_{\text{stressed}}} + \epsilon v_1$.
   - Update $e_{k_{\text{stressed}}} := e_{k_{\text{stressed}}} - \epsilon v_1$.
3. Update Voronoi cells: The new code inherits half of $k_{\text{stressed}}$'s cell.

**Effect:** Vocabulary migrates to high-information-density regions. Dead codes are "resurrected" where they are needed.

*Connection to existing constraints:* This implements the anti-collapse regularizer from {ref}`Section 3.5.5 <sec-calibrating-tolerances>`: $\lambda_{\text{use}} D_{\mathrm{KL}}(\hat{p}(K) \| \text{Unif}(\mathcal{K}))$.
:::



(sec-measure-theoretic-formalization)=
#### 30.12.4 Measure-Theoretic Formalization

For maximum rigor, we treat the codebook not as a static list of vectors but as a **discrete measure** on the fiber. This enables a variational characterization of code allocation.

:::{prf:definition} Symbolic Voronoi Partition
:label: def-voronoi-partition

Let $\mathcal{Z}_i$ be the continuous fiber associated with chart $i$. The codebook $\mathcal{C}_i = \{e_{i,k}\}_{k=1}^{N_v}$ induces a partition $\{\mathcal{V}_k\}$ of $\mathcal{Z}_i$ via:

$$
\mathcal{V}_k := \left\{ z \in \mathcal{Z}_i : d_G(z, e_k) \leq d_G(z, e_j) \;\forall j \neq k \right\}
$$
The probability mass of symbol $k$ is the measure of its Voronoi cell:

$$
P(k) := \int_{\mathcal{V}_k} p(z)\, d\mu_G(z)
$$
where $d\mu_G = \sqrt{\det G}\, dz$ is the Riemannian volume form.
:::

:::{prf:definition} Local Distortion Functional
:label: def-local-distortion

The **local distortion** of symbol $k$ quantifies the representational error within its Voronoi cell:

$$
\mathcal{D}_k := \int_{\mathcal{V}_k} d_G(z, e_k)^2\, p(z)\, d\mu_G(z)
$$
*Units:* $[z]^2$ (weighted squared geodesic distance).

*Relation to geometric tension:* $\mathcal{D}_k = P(k) \cdot \sigma_k^2$, where $\sigma_k^2$ is the intra-symbol variance ({prf:ref}`def-intra-symbol-variance`).
:::

:::{prf:definition} Symbol Utility Functional
:label: def-symbol-utility

The **utility** $U_k$ of symbol $k$ measures its contribution to control authority and predictive accuracy:

$$
U_k := P(k) \cdot I(K=k; A) + P(k) \cdot I(K=k; K_{t+1})
$$
where:
- $I(K=k; A)$ is the mutual information between symbol activation and action selection,
- $I(K=k; K_{t+1})$ is the mutual information between symbol activation and next-state prediction.

*Units:* nat.

*Interpretation:* A symbol with $U_k \approx 0$ neither influences actions nor aids prediction—it is **semantically dead** regardless of its usage frequency.
:::

:::{prf:theorem} Optimal Reallocation Gradient
:label: thm-reallocation-gradient

Let $k_{\text{dead}}$ satisfy $U_{k_{\text{dead}}} < \epsilon_U$ and let $k_{\text{stressed}}$ satisfy $\mathcal{D}_{k_{\text{stressed}}} = \max_k \mathcal{D}_k$. The expected reduction in global distortion per reallocated code is:

$$
\frac{\delta \mathcal{D}}{\delta N_{\text{codes}}} \approx \frac{\mathcal{D}_{k_{\text{stressed}}}}{H(K = k_{\text{stressed}})}
$$
*Proof sketch.* In the high-resolution limit of vector quantization (Zador's theorem {cite}`zador1982asymptotic`), distortion scales as $\mathcal{D} \propto N_v^{-2/d}$ where $d$ is the latent dimension. Reallocating a code from a zero-utility region to a high-distortion region maximizes the gradient of the distortion functional. The denominator $H(K = k_{\text{stressed}})$ normalizes by the information content of the target symbol. $\square$
:::
:::{prf:corollary} The Bimodal Instability Theorem (Fission Trigger)
:label: cor-bimodal-instability

Let $K$ be a macro-symbol with associated policy $\pi(\cdot|K)$. The **Structural Stability** of $K$ is inversely proportional to its Varentropy.

If the policy $\pi(\cdot|K)$ is a mixture of two disjoint, equally weighted strategies (a "Buridan's Ass" scenario on a value ridge), the Varentropy satisfies:

$$
V_H(K) = \frac{1}{4}\left(\frac{\Delta Q}{T_c}\right)^2,
$$
where $\Delta Q = |Q_1 - Q_2|$ is the value gap between the modes. In the limit of distinct modes ($\Delta Q \gg T_c$), $V_H$ is maximized, whereas for a uniform (maximum entropy) distribution, $V_H = 0$.

*Units:* $\mathrm{nat}^2$.

**Refined Fission Criterion:**
The **Geometric Tension** $\sigma_k^2$ (Definition {prf:ref}`def-intra-symbol-variance`) is rigorously generalized by the **Varentropy Excess**:

$$
\text{Fission}(K) \iff V_H(K) > \mathcal{V}_{\text{crit}} \quad \text{AND} \quad H(K) > H_{\text{noise}}.
$$
**Interpretation:**
- **High $H$, Low $V_H$:** Aleatoric Uncertainty (Noise/Fog). The distribution is flat. *Action:* Smoothing/Integration.
- **High $H$, High $V_H$:** Epistemic Conflict (Bifurcation). The distribution is multimodal. *Action:* Topological Fission (Node 50).

*Proof:* See Appendix {ref}`E.9 <sec-appendix-e-proof-of-corollary-bimodal-instability>`.

:::



(sec-comparison-chart-vs-symbol-metabolism)=
### 30.13 Comparison: Chart vs. Symbol Metabolism

The fission/fusion dynamics operate at two hierarchical levels with analogous but distinct forces.

**Table 30.13.1 (Two-Level Metabolic Hierarchy).**

| Level             | Object          | Expansion Force                | Contraction Force                    | Geometry                   | Diagnostic       |
|:------------------|:----------------|:-------------------------------|:-------------------------------------|:---------------------------|:-----------------|
| **Chart** (Macro) | Query $q_i$     | Ontological Stress $\Xi$       | Redundancy $\Upsilon_{ij}$           | Hyperbolic (Poincare disk) | Nodes 49, 50, 54 |
| **Symbol** (Meso) | Embedding $e_k$ | Geometric Tension $\sigma_k^2$ | Indistinguishability $\mathcal{D}_f$ | Euclidean (Voronoi cell)   | Node 55          |

**Key distinctions:**

1. **Chart metabolism** governs the **global manifold partition**—how many semantic categories exist.
2. **Symbol metabolism** governs the **local tessellation within each chart**—how finely each category is discretized.
3. The **Universal Governor** ({ref}`Section 26 <sec-theory-of-meta-stability-the-universal-governor-as-homeostatic-controller>`) monitors both levels via the total entropy budget:

   $$
   H(K_{\text{chart}}) + \mathbb{E}_{i}[H(K_{\text{code}} | K_{\text{chart}} = i)] \leq B_{\text{metabolic}}
   $$


(sec-summary-the-topological-heartbeat)=
### 30.14 Summary: The Topological Heartbeat

The complete ontological lifecycle forms a **homeostatic cycle**:

**Table 30.14.1 (The Ontological Heartbeat).**

| Phase                 | Trigger                             | Mechanism                                                                                 | Effect                                |
|:----------------------|:------------------------------------|:------------------------------------------------------------------------------------------|:--------------------------------------|
| **Systole (Fission)** | $\Xi > \Xi_{\text{crit}}$           | Supercritical bifurcation ({prf:ref}`thm-supercritical-pitchfork-bifurcation-for-charts`) | $N_c \to N_c + 1$; manifold expands   |
| **Diastole (Fusion)** | $\Upsilon > \Upsilon_{\text{crit}}$ | Subcritical bifurcation ({prf:ref}`thm-subcritical-pitchfork-fusion`)                     | $N_c \to N_c - 1$; manifold contracts |

The {ref}`Universal Governor (Section 26) <sec-theory-of-meta-stability-the-universal-governor-as-homeostatic-controller>` maintains homeostasis by monitoring:

1. **Complexity budget:** $H(K_{\text{chart}}) + H(K_{\text{code}}) \leq B_{\text{metabolic}}$.
2. **Discrimination floor:** $G_\Delta(i, j) > G_{\min}$ for all retained chart pairs.
3. **Liveness constraint:** $P(K = k) > \epsilon_{\text{dead}}$ for all active codes.

**Conclusion.** By adding Fusion to Fission, the agent possesses a complete **topological metabolism**. Fission creates structure when the world demands finer distinctions; Fusion destroys structure when distinctions become redundant. The balance is governed by the same MDL principle that drives the entire framework: minimize description length plus expected regret.

:::{prf:proposition} Equipartition of Meaning
:label: prop-equipartition

At metabolic equilibrium, the marginal utility per bit is uniform across the ontological hierarchy:

$$
\frac{\partial U}{\partial H(K_{\text{chart}})} \approx \frac{\partial U}{\partial H(K_{\text{code}})} \approx \text{const.}
$$
where $U$ is the total utility functional (value minus complexity cost).

*Interpretation:* The agent allocates representational capacity such that one additional bit of chart-level information provides the same marginal value as one additional bit of symbol-level information. This is the information-theoretic analogue of thermodynamic equipartition.
:::



(sec-thermodynamic-hysteresis-calibration)=
### 30.15 Thermodynamic Calibration of Ontological Hysteresis

We derive the hysteresis constant $\epsilon_{\text{hysteresis}}$ appearing in the Fusion Criterion ({prf:ref}`thm-fusion-criterion`) as a thermodynamic necessity arising from the computational metabolism of the agent ({ref}`Section 31 <sec-computational-metabolism-the-landauer-bound-and-deliberation-dynamics>`).

:::{prf:theorem} Thermodynamic Lower Bound on Hysteresis
:label: thm-thermodynamic-hysteresis-bound

Let $\mathcal{C}$ be a cycle of ontological operations consisting of a fission event $N_c \to N_c + 1$ followed immediately by a fusion event $N_c + 1 \to N_c$. Let $T_c$ be the cognitive temperature and $\mathcal{W}_{\text{comp}}$ be the metabolic work of parameter instantiation. To satisfy the generalized Second Law of Thermodynamics for open cognitive systems (Theorem {prf:ref}`thm-generalized-landauer-bound`), the hysteresis threshold must satisfy:

$$
\epsilon_{\text{hysteresis}} \geq \frac{1}{\beta_{\text{eff}}} \left( \Delta H_{\text{Shannon}} + \frac{1}{T_c}\mathcal{W}_{\text{comp}} \right)
$$
where $\beta_{\text{eff}} = 1/T_c$ is the inverse cognitive temperature and $\Delta H_{\text{Shannon}}$ is the entropy reduction associated with the discarded distinction.

*Proof.*
Consider the free energy functional $\mathcal{F} = E - T_c S$.

1. **Fission Cost:** The creation of a new chart requires initializing a set of parameters $\theta_{\text{new}}$. By Landauer's Principle ({prf:ref}`pi-landauer-principle`), the erasure of the previous random state of these memory units to a low-entropy initialization requires work $\mathcal{W}_{\text{init}} \geq k T_c \ln 2 \cdot |\theta_{\text{new}}|$.

2. **Fusion Cost:** The merger of two charts implies the erasure of the mutual information $I(X; \{K_i, K_j\}) - I(X; K_{i \cup j})$, defined as the Discrimination Gain $G_\Delta$ ({prf:ref}`def-discrimination-gain`). This is an irreversible logical operation, dissipating heat $Q_{\text{fus}} \geq T_c G_\Delta$.

3. **Cycle Condition:** For the cycle $\mathcal{C}$ to be non-spontaneous (preventing chattering), the total free energy change must be positive. The Governor imposes a metabolic efficiency constraint $\eta_{\text{ROI}} > \eta_{\min}$ ({ref}`Section 26 <sec-theory-of-meta-stability-the-universal-governor-as-homeostatic-controller>`).

4. **Derivation:** The utility gain of the cycle is zero (the topology is unchanged). The cost is $\mathcal{W}_{\text{init}} + Q_{\text{fus}}$. For the cycle to be rejected by the Fusion Criterion ({prf:ref}`thm-fusion-criterion`), the hysteresis term must exceed the minimum metabolic dissipation of the cycle:

$$
\epsilon_{\text{hysteresis}} \geq \inf_{\mathcal{C}} \oint \dot{\mathcal{M}}(s) ds
$$
Substituting the Landauer bound yields the stated inequality. $\square$
:::

*Units:* $[\epsilon_{\text{hysteresis}}] = \text{nat}$, consistent with the complexity cost functional.

*Cross-references:* This resolves the hysteresis calibration question by grounding it in the Landauer thermodynamics of {ref}`Section 31 <sec-computational-metabolism-the-landauer-bound-and-deliberation-dynamics>`.



(sec-hyperbolic-coalescence)=
### 30.16 Intrinsic Coalescence on Hyperbolic Manifolds

The Query Coalescence operation ({prf:ref}`def-query-coalescence`) uses a Euclidean barycenter $\bar{q} = \frac{1}{N}\sum q_i$. In the Poincare disk $\mathbb{D}$, this induces geometric distortion since straight lines in $\mathbb{R}^n$ are not geodesics in $\mathbb{D}$. The rigorous fusion operator is the **Fréchet Mean**, following the pattern established in {prf:ref}`def-class-centroid-in-poincar-disk`.

:::{prf:definition} Hyperbolic Fréchet Mean for Query Coalescence
:label: def-hyperbolic-frechet-coalescence

Let $\{q_i\}_{i=1}^k \subset \mathbb{D}$ be a set of chart query vectors with associated usage weights $\bar{w}_i := \mathbb{E}[w_i(x)]$ from the Attentive Atlas ({prf:ref}`def-attentive-routing-law`). The **Intrinsic Merged Query** is:

$$
q_{\text{merged}} := \operatorname*{arg\,min}_{q \in \mathbb{D}} \sum_{i=1}^k \bar{w}_i \cdot d^2_{\mathbb{D}}(q, q_i),
$$
where $d_{\mathbb{D}}(x, y) = \operatorname{arccosh}\left(1 + \frac{2\|x-y\|^2}{(1-\|x\|^2)(1-\|y\|^2)}\right)$ is the hyperbolic distance.

*Units:* $[q_{\text{merged}}] = [q_i]$ (dimensionless in the unit disk).

*Cross-reference:* This definition supersedes {prf:ref}`def-query-coalescence` for hyperbolic embeddings.
:::

:::{prf:theorem} Existence and Uniqueness of Fusion Center
:label: thm-frechet-fusion-uniqueness

Since the Poincare disk $(\mathbb{D}, G)$ is a complete, simply connected Riemannian manifold with non-positive sectional curvature ($K=-1$), it is a Hadamard space (global CAT(0) space). The squared distance function $d^2_{\mathbb{D}}(\cdot, y)$ is strictly convex. Therefore, the functional $F(q) = \sum \bar{w}_i d^2_{\mathbb{D}}(q, q_i)$ admits a unique global minimizer.

*Proof.* By Cartan's theorem on Hadamard manifolds, the distance function from any point is strictly convex along geodesics. The weighted sum of strictly convex functions is strictly convex, ensuring the minimizer exists and is unique. $\square$
:::

:::{prf:remark} Computational Algorithm
:label: rem-frechet-algorithm

The minimizer can be computed via Riemannian gradient descent:

$$
q_{t+1} = \operatorname{Exp}_{q_t}\left( -\eta \sum_i \bar{w}_i \operatorname{Log}_{q_t}(q_i) \right)
$$
where:
- $\operatorname{Exp}_p: T_p\mathbb{D} \to \mathbb{D}$ is the exponential map at $p$
- $\operatorname{Log}_p: \mathbb{D} \to T_p\mathbb{D}$ is the logarithmic map (inverse of exponential)

For the Poincare disk, these have closed-form expressions via Möbius operations ({ref}`Section 21.3 <sec-bulk-boundary-independence>`).

*Complexity:* $O(k \cdot d)$ per iteration, where $k$ is the number of charts being merged and $d$ is the embedding dimension.
:::

*Cross-references:* This resolves the geometric inconsistency by ensuring coalescence respects the intrinsic hyperbolic geometry.



(sec-fission-inhibition-corollary)=
### 30.17 The Fission Inhibition Corollary (Hierarchical Metabolism Resolution)

We prove that the Stacked TopoEncoder architecture ({ref}`Section 7.12 <sec-stacked-topoencoders-deep-renormalization-group-flow>`) enforces **top-down stability** via the properties of the residual variance in the Renormalization Group (RG) flow. A fission event at layer $\ell$ does not trigger cascading fission at higher layers.

:::{prf:theorem} Fission Inhibition Corollary
:label: thm-fission-inhibition

Let $\mathcal{E}^{(\ell)}$ be the encoder at scale $\ell$. A Topological Fission event at layer $\ell$ (increasing chart count $N_c^{(\ell)} \to N_c^{(\ell)}+1$) strictly reduces the probability of fission at layer $\ell+1$.

*Proof.*
1. **Residual Coupling:** The input to layer $\ell+1$ is the normalized residual of layer $\ell$: $x^{(\ell+1)} = z_{\text{tex}}^{(\ell)} / \sigma^{(\ell)}$.

2. **Approximation Theory:** Fission adds a centroid to the Voronoi partition at layer $\ell$. By standard quantization theory (Zador's theorem), increasing codebook size strictly reduces the mean squared quantization error (distortion), provided the data is not uniform.

3. **Variance Reduction:** The reconstruction error $\|z_{\text{tex}}^{(\ell)}\|^2$ decreases, implying the scale factor $\sigma^{(\ell)}$ decreases.

4. **Stress Damping:** Ontological Stress at layer $\ell+1$ is upper-bounded by the mutual information of its input. Since the input variance is reduced (relative to the pre-fission state), the extractable structure $I(x^{(\ell+1)}_t; x^{(\ell+1)}_{t+1})$ decreases.

5. **Conclusion:** Macro-scale adaptation absorbs structural variance, starving the micro-scale of the stress required to trigger bifurcation. $\square$
:::

:::{prf:corollary} Hierarchical Stability
:label: cor-hierarchical-stability

The stacked architecture is **inherently stable** against fission cascades. Ontological expansion at coarse scales (low $\ell$) pre-empts the need for expansion at fine scales (high $\ell$).

*Interpretation:* If the agent learns a new high-level concept (e.g., "mammal"), the residual variance available to learn low-level distinctions (e.g., specific breeds) is reduced. The hierarchy self-regulates, preventing runaway complexity growth.
:::

*Cross-references:* This resolves the hierarchical metabolism question by showing that the RG structure naturally dampens topological perturbations from propagating upward.



(sec-computational-metabolism-the-landauer-bound-and-deliberation-dynamics)=
