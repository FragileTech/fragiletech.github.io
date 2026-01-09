## 25. Supervised Topology: Semantic Potentials and Metric Segmentation

:::{admonition} Researcher Bridge: Metric Learning for Classification
:class: info
:name: rb-metric-learning
This section recasts supervised labels as geometric constraints. It is the same idea as contrastive or metric learning, but expressed as class-conditioned potentials and separation in the latent manifold.
:::

We rigorously define the role of discrete class labels $\mathcal{Y}$ within the continuous latent geometry. Rather than treating classification as "predicting a target variable," we define classification via the **Manifold Hypothesis** {cite}`carlsson2009tda`: Class labels identify topologically coherent regions of the latent manifold, and classification is **equilibrium chart assignment under class-conditioned gradient flow**.

This section **extends** the context-conditioned framework of Section 23.6, providing the topological constraints that make classification geometrically meaningful. The approach integrates ideas from topological data analysis {cite}`carlsson2009tda`, mixture-of-experts routing {cite}`shazeer2017moe`, hyperbolic embeddings {cite}`nickel2017poincare`, and Riemannian optimization {cite}`bonnabel2013rsgd`.

(sec-relationship-to-the-context-conditioned-framework)=
### 25.1 Relationship to the Context-Conditioned Framework

:::{prf:remark} Extension, Not Replacement
:label: rem-extension-not-replacement

Section 23.6 establishes classification as selecting a context $c \in \mathcal{Y}$ (the label space), with effective potential $\Phi_{\text{eff}} = -\log p(y|z)$ (Theorem {prf:ref}`thm-universal-context-structure`). This section specifies the **topological constraints** that enforce geometric coherence of this classification:

1. Charts should be semantically pure (one class per chart, modulo transition regions)
2. Different classes should be metrically separated (long geodesics between class regions)
3. Classification should be stable under dynamics (regions of attraction)

:::
:::{prf:definition} Semantic Partition
:label: def-semantic-partition

Let $\mathcal{Y} = \{1, \ldots, C\}$ be the set of class labels and $\mathcal{K}$ the macro-state register (Definition 2.2.1). A labeling $Y: \mathcal{X} \to \mathcal{Y}$ induces a **soft partition** of the chart atlas:

$$
\mathcal{A}_y := \{k \in \mathcal{K} : P(Y=y \mid K=k) > 1 - \epsilon_{\text{purity}}\},
$$
where $\epsilon_{\text{purity}} \in (0, 0.5)$ is the purity threshold.

*Interpretation:* $\mathcal{A}_y$ is the **sub-atlas** of charts predominantly associated with class $y$. A chart $k$ belongs to $\mathcal{A}_y$ if, given that a sample routes to chart $k$, the probability of class $y$ exceeds $1 - \epsilon_{\text{purity}}$.

:::
:::{prf:proposition} Soft Injectivity
:label: prop-soft-injectivity

The sub-atlases need not be disjoint. Charts in $\mathcal{A}_i \cap \mathcal{A}_j$ for $i \neq j$ are **transition regions** characterized by:

1. **Low purity:** $\max_y P(Y=y \mid K=k) < 1 - \epsilon_{\text{purity}}$ for all $y$
2. **High entropy:** $H(Y \mid K=k) > H_{\text{transition}}$ (conditional entropy; see {cite}`cover1991elements`)
3. **Low information content:** These charts carry less semantic information per the information bottleneck principle {cite}`tishby2015ib`

*Remark (Geometric Interpretation).* Transition charts correspond to saddle regions of the semantic potential landscape—unstable fixed points between class regions of attraction.

**Cross-references:** Section 23.6 (Context-Conditioned Policies), Definition 2.2.1 (Macro-State Register), Section 7.8 (Router Weights).

:::
(sec-the-semantic-potential)=
### 25.2 The Semantic Potential

We embed class labels into the dynamics via a **class-conditioned potential** that shapes the energy landscape.

:::{prf:definition} Class-Conditioned Potential
:label: def-class-conditioned-potential

Given a target class $y \in \mathcal{Y}$, define the semantic potential:

$$
V_y(z, K) := -\beta_{\text{class}} \log P(Y=y \mid K) + V_{\text{base}}(z, K),
$$
where:
- $P(Y=y \mid K) = \text{softmax}(\Theta_{K,:})_y$ with learnable parameters $\Theta \in \mathbb{R}^{N_c \times C}$
- $V_{\text{base}}(z, K)$ is the unconditioned critic (Section 2.7)
- $\beta_{\text{class}} > 0$ is the **class temperature** (inverse of semantic diffusion)
- Units: $[V_y] = \mathrm{nat}$

*Remark (Chart-to-Class Mapping).* The learnable parameter $\Theta_{k,y}$ represents the log-affinity of chart $k$ for class $y$. After training, $P(Y=y \mid K=k) = \text{softmax}(\Theta_{k,:})_y$ approximates the empirical conditional distribution.

*Remark (Alternative: Empirical Estimation).* Instead of learnable parameters, one may estimate $P(Y|K)$ empirically via exponential moving average:

$$
\hat{P}(Y=y \mid K=k) = \frac{\text{EMA}[\mathbb{I}[Y=y, K=k]]}{\text{EMA}[\mathbb{I}[K=k]]}.
$$
This is non-differentiable w.r.t. chart assignment but more grounded in observations. A hybrid approach initializes learnable $\Theta$ from empirical estimates after warmup.

:::
:::{prf:definition} Region of Attraction
:label: def-region-of-attraction

The **region of attraction** for class $y$ is:

$$
\mathcal{B}_y := \{z \in \mathcal{Z} : \lim_{t \to \infty} \phi_t(z) \in \mathcal{A}_y\},
$$
where $\phi_t$ denotes the flow of the gradient dynamical system $\dot{z} = -G^{-1}(z)\nabla V_y(z)$.

*Interpretation:* $\mathcal{B}_y$ is the set of initial conditions from which the deterministic gradient flow on $V_y$ converges to the class-$y$ region.

:::
:::{prf:theorem} Classification as Relaxation
:label: thm-classification-as-relaxation

Under the overdamped dynamics (Section 22.5) with potential $V_y$:

$$
dz = -G^{-1}(z) \nabla V_y(z, K)\, ds + \sqrt{2T_c}\, G^{-1/2}(z)\, dW_s,
$$
the limiting chart assignment satisfies:

$$
\lim_{s \to \infty} K(z(s)) \in \mathcal{A}_y \quad \text{almost surely},
$$
provided:
1. $z(0) \in \mathcal{B}_y$ (initial condition in the basin)
2. $T_c$ is sufficiently small (low temperature limit)
3. The basins have positive measure and are separated by finite barriers

*Proof sketch.* Define the Lyapunov function $L(z) := V_y(z, K(z))$ (see {cite}`khalil2002nonlinear` for Lyapunov theory, {cite}`lasalle1960invariance` for the invariance principle). Under the overdamped dynamics:

$$
\frac{dL}{ds} = \nabla V_y \cdot \dot{z} = -\|\nabla V_y\|_G^2 + \text{noise terms}.
$$
For small $T_c$, the deterministic term dominates, ensuring $L$ decreases until $z$ reaches a local minimum. The class-$y$ region is the global minimum of $V_y$ by construction. Full proof in Appendix A.5. $\square$

:::
:::{prf:corollary} Inference via Relaxation
:label: cor-inference-via-relaxation

Classification inference proceeds as:
1. Encode: $z_0 = \text{Enc}(x)$
2. Relax under neutral potential $V_{\text{base}}$ (no class conditioning) to equilibrium $z^*$
3. Read out: $\hat{y} = \arg\max_y P(Y=y \mid K(z^*))$

*Remark (Fast Path).* In practice, we often skip the relaxation and use direct readout: $\hat{y} = \arg\max_y \sum_k w_k(x) \cdot P(Y=y \mid K=k)$, where $w_k(x)$ are the router weights (Section 7.8). The relaxation interpretation justifies this as the $T_c \to 0$, $s \to \infty$ limit.

**Cross-references:** Section 22.5 (Overdamped Limit), Definition {prf:ref}`def-effective-potential`, Section 2.7 (Critic).

:::

::::{note} Connection to RL #21: Imitation Learning as Degenerate Supervised Topology
**The General Law (Fragile Agent):**
Class labels define **Semantic Partitions** with **Class-Conditioned Potentials**:

$$
V_y(z, K) = -\beta_{\text{class}} \log P(Y=y \mid K) + V_{\text{base}}(z, K)
$$
Trajectories relax into class-specific basins via gradient flow on the learned metric.

**The Degenerate Limit:**
Set $V_{\text{base}} = 0$. Interpret labels as expert actions. Use Euclidean minimization.

**The Special Case (Standard RL):**

$$
\mathcal{L}_{\text{BC}} = \mathbb{E}_{(s,a^*) \sim \mathcal{D}_{\text{expert}}}[-\log \pi(a^*|s)]
$$
This recovers **Behavioral Cloning** and **Imitation Learning** {cite}`pomerleau1991bc`.

**What the generalization offers:**
- **Metric segmentation:** Classes are metrically separated on the learned manifold
- **Potential landscape:** $V_y$ creates basins of attraction, not just classification boundaries
- **Jump modulation:** Cross-class transitions suppressed by separation penalty $\gamma_{\text{sep}}$
- **Relaxation dynamics:** Classification emerges from physics, not discrete argmax
::::

(sec-metric-segmentation-via-jump-rate-modulation)=
### 25.3 Metric Segmentation via Jump Rate Modulation

To enforce separation between classes, we modulate the **jump rates** in the WFR dynamics {cite}`chizat2018wfr` (Section 20), which govern transitions between chart coordinate systems defined in Section 7.13.

:::{prf:definition} Class-Consistent Jump Rate
:label: def-class-consistent-jump-rate

For the WFR reaction term (Definition {prf:ref}`def-the-wfr-action`), modulate the inter-chart transition rate:

$$
\lambda_{i \to j}^{\text{sup}} := \lambda_{i \to j}^{(0)} \cdot \exp\left(-\gamma_{\text{sep}} \cdot D_{\text{class}}(i, j)\right),
$$
where:
- $\lambda^{(0)}_{i \to j}$ is the **base transition rate** from the GKSL master equation {cite}`lindblad1976gksl,gorini1976gksl` (Section 20.5), derived from the overlap consistency of jump operators (Section 7.13)
- $\gamma_{\text{sep}} \geq 0$ is the **separation strength** (hyperparameter)
- $D_{\text{class}}(i, j) = \mathbb{I}[\text{Class}(i) \neq \text{Class}(j)]$ is the class disagreement indicator
- $\text{Class}(k) := \arg\max_y P(Y=y \mid K=k)$ is the dominant class of chart $k$

*Remark (Rate vs Operator).* Section 7.13 defines the **transition function** $L_{i \to j}$ (the coordinate change map). The **transition rate** $\lambda_{i \to j}$ is a separate quantity from the GKSL/master equation framework (Section 20.5, Equation 20.5.2) that governs *how often* jumps occur, not *where* they go. The rate is typically derived from the overlap structure: $\lambda_{i \to j}^{(0)} \propto \mathbb{E}_{x}[w_i(x) w_j(x)]$, measuring how much probability mass lies in the overlap $U_i \cap U_j$.

*Interpretation:* Transitions between charts of the same class proceed at the base rate $\lambda^{(0)}$. Transitions between charts of different classes are exponentially suppressed by factor $e^{-\gamma_{\text{sep}}}$.

:::
:::{prf:proposition} Effective Disconnection
:label: prop-effective-disconnection

As $\gamma_{\text{sep}} \to \infty$, the effective WFR distance between charts of different classes diverges:

$$
d_{\text{WFR}}(\mathcal{A}_{y_1}, \mathcal{A}_{y_2}) \to \infty \quad \text{for } y_1 \neq y_2.
$$
*Proof sketch.* The WFR distance (Definition {prf:ref}`def-the-wfr-action`) involves minimizing over paths that may use both transport (continuous flow within charts) and reaction (jumps between charts). Consider a path from $\mathcal{A}_{y_1}$ to $\mathcal{A}_{y_2}$:

1. **Transport-only paths:** If $\mathcal{A}_{y_1}$ and $\mathcal{A}_{y_2}$ are not geometrically adjacent (no shared chart boundary), pure transport paths have infinite cost.

2. **Jump paths:** Any path using cross-class jumps incurs reaction cost. In the GKSL interpretation (Section 20.5), the suppressed jump rate $\lambda^{\text{sup}} = \lambda^{(0)} e^{-\gamma_{\text{sep}}}$ means mass transfer between unlike-class charts requires longer dwell times, increasing the action.

3. **Divergence:** As $\gamma_{\text{sep}} \to \infty$, cross-class jumps become arbitrarily rare. The optimal path cost diverges because: (a) pure transport is blocked by chart boundaries, and (b) the reaction term penalizes staying in transition states waiting for rare jumps.

The precise scaling (exponential, polynomial, etc.) depends on the manifold geometry, but divergence is guaranteed. $\square$

:::
:::{prf:remark} Tunneling as Anomaly Detection
:label: rem-tunneling-as-anomaly-detection

Cross-class transitions are not forbidden, merely exponentially suppressed. A detected cross-class jump indicates:

1. **Anomaly:** The sample lies in a transition region not well-covered by training
2. **Distribution shift:** The test distribution differs from training
3. **Adversarial input:** Deliberate perturbation to cross class boundaries

This provides a natural **out-of-distribution detection** mechanism: monitor the rate of cross-class transitions.

:::
:::{prf:definition} Class-Modulated Jump Operator
:label: def-class-modulated-jump-operator

Modify the jump operator (Definition {prf:ref}`def-factorized-jump-operator`) to incorporate class consistency:

```python
def class_modulated_jump_rate(
    lambda_base: torch.Tensor,    # [N_c, N_c] base jump rates
    chart_to_class: torch.Tensor, # [N_c, C] learnable logits
    gamma_sep: float = 5.0,       # Separation strength
) -> torch.Tensor:
    """
    Compute class-modulated jump rates.

    Cross-ref:
        - Definition 25.3.1 (Class-Consistent Jump Rate)
        - Definition 7.13.1 (Jump Operator)
    """
    # Get dominant class per chart
    p_y_given_k = F.softmax(chart_to_class, dim=1)  # [N_c, C]
    dominant_class = p_y_given_k.argmax(dim=1)       # [N_c]

    # Compute class disagreement matrix
    class_match = (dominant_class.unsqueeze(1) == dominant_class.unsqueeze(0)).float()  # [N_c, N_c]
    D_class = 1.0 - class_match  # 1 if classes differ, 0 if same

    # Modulate rates
    lambda_sup = lambda_base * torch.exp(-gamma_sep * D_class)

    return lambda_sup
```

**Cross-references:** Section 20.2 (WFR Metric), Definition {prf:ref}`def-factorized-jump-operator`, Section 20.5 (GKSL Connection).

:::
(sec-the-supervised-topology-loss)=
### 25.4 The Supervised Topology Loss

We define training losses that enforce the geometric structure described above.

(sec-chart-purity-loss)=
#### 25.4.1 Chart Purity Loss (Conditional Entropy)

:::{prf:definition} Purity Loss
:label: def-purity-loss

The purity loss measures how well charts separate classes:

$$
\mathcal{L}_{\text{purity}} = \sum_{k=1}^{N_c} P(K=k) \cdot H(Y \mid K=k),
$$
where:
- $P(K=k) = \mathbb{E}_{x \sim \mathcal{D}}[w_k(x)]$ is the marginal chart probability
- $H(Y \mid K=k) = -\sum_y P(Y=y \mid K=k) \log P(Y=y \mid K=k)$ is the class entropy within chart $k$

*Interpretation:* $\mathcal{L}_{\text{purity}} = H(Y \mid K)$, the conditional entropy of class given chart. Minimizing this encourages each chart to be associated with a single class.

:::
:::{prf:proposition} Purity-Information Duality
:label: prop-purity-information-duality

Minimizing $\mathcal{L}_{\text{purity}}$ is equivalent to maximizing the mutual information $I(K; Y)$:

$$
\mathcal{L}_{\text{purity}} = H(Y) - I(K; Y).
$$
Since $H(Y)$ is fixed by the data, $\min \mathcal{L}_{\text{purity}} \Leftrightarrow \max I(K; Y)$.

:::
(sec-load-balance-loss)=
#### 25.4.2 Load Balance Loss (Uniform Coverage)

{cite}`shazeer2017moe`

:::{prf:definition} Balance Loss
:label: def-balance-loss

Prevent degenerate solutions where all samples route to few charts:

$$
\mathcal{L}_{\text{balance}} = D_{\text{KL}}\left(\bar{w} \;\|\; \text{Uniform}(N_c)\right),
$$
where $\bar{w} = \mathbb{E}_{x \sim \mathcal{D}}[w(x)]$ is the average router weight vector.

*Interpretation:* Encourages all charts to be used, preventing "dead charts" and ensuring the atlas covers the label space.

:::
(sec-metric-contrastive-loss)=
#### 25.4.3 Metric Contrastive Loss (Geometric Separation)

{cite}`schroff2015facenet,khosla2020supcon`

:::{prf:definition} Contrastive Loss
:label: def-contrastive-loss

Enforce that different-class samples are geometrically separated:

$$
\mathcal{L}_{\text{metric}} = \frac{1}{|\mathcal{P}|} \sum_{(i,j) \in \mathcal{P}: y_i \neq y_j} w_i^\top w_j \cdot \max(0, m - d_{\text{jump}}(z_i, z_j))^2,
$$
where:
- $\mathcal{P}$ is the set of sample pairs in the batch
- $w_i, w_j$ are router weight vectors
- $m > 0$ is the margin (minimum desired separation)
- $d_{\text{jump}}(z_i, z_j)$ is the minimum jump cost (Section 7.13)

*Interpretation:* If two samples have different labels but high router overlap ($w_i^\top w_j$ large), they must be separated by at least margin $m$ in jump distance. Otherwise, the loss penalizes the configuration.

:::
(sec-route-alignment-loss)=
#### 25.4.4 Route Alignment Loss (Prediction Consistency)

:::{prf:definition} Route Alignment Loss
:label: def-route-alignment-loss

The primary classification loss:

$$
\mathcal{L}_{\text{route}} = \mathbb{E}_{x, y_{\text{true}}}\left[\text{CE}\left(\sum_k w_k(x) \cdot P(Y=\cdot \mid K=k), \; y_{\text{true}}\right)\right],
$$
where $\text{CE}$ denotes cross-entropy.

*Interpretation:* The predicted class distribution is the router-weighted average of per-chart class distributions. This must match the true label.

:::
(sec-combined-supervised-topology-loss)=
#### 25.4.5 Combined Supervised Topology Loss

:::{prf:definition} Total Loss
:label: def-total-loss

The full supervised topology loss:

$$
\mathcal{L}_{\text{sup-topo}} = \mathcal{L}_{\text{route}} + \lambda_{\text{pur}} \mathcal{L}_{\text{purity}} + \lambda_{\text{bal}} \mathcal{L}_{\text{balance}} + \lambda_{\text{met}} \mathcal{L}_{\text{metric}}.
$$
Typical hyperparameters: $\lambda_{\text{pur}} = 0.1$, $\lambda_{\text{bal}} = 0.01$, $\lambda_{\text{met}} = 0.01$.

**Algorithm 25.4.7 (SupervisedTopologyLoss Implementation).**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class SupervisedTopologyLoss(nn.Module):
    """
    Supervised topology loss enforcing chart purity, balance, and separation.

    Cross-ref:
        - Definition 25.4.6 (Total Loss)
        - Section 7.8 (Router Weights)
    """

    def __init__(
        self,
        num_charts: int,
        num_classes: int,
        lambda_purity: float = 0.1,
        lambda_balance: float = 0.01,
        lambda_metric: float = 0.01,
        margin: float = 1.0,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.num_charts = num_charts
        self.num_classes = num_classes
        self.lambda_purity = lambda_purity
        self.lambda_balance = lambda_balance
        self.lambda_metric = lambda_metric
        self.margin = margin

        # Learnable chart-to-class mapping (Definition 25.2.1)
        self.chart_to_class = nn.Parameter(
            torch.randn(num_charts, num_classes) * 0.01
        )
        self.temperature = temperature

    @property
    def p_y_given_k(self) -> torch.Tensor:
        """P(Y|K) distribution [N_c, C]."""
        return F.softmax(self.chart_to_class / self.temperature, dim=1)

    def forward(
        self,
        router_weights: torch.Tensor,  # [B, N_c]
        y_true: torch.Tensor,          # [B] class labels
        z_latent: torch.Tensor = None, # [B, D] optional for metric loss
    ) -> Dict[str, torch.Tensor]:
        """
        Compute supervised topology losses.

        Returns dict with individual losses and total.
        """
        B = router_weights.shape[0]
        p_y_k = self.p_y_given_k  # [N_c, C]

        # === Route Alignment Loss (Definition 25.4.5) ===
        # P(Y|x) = sum_k w_k(x) * P(Y|K=k)
        p_y_x = torch.matmul(router_weights, p_y_k)  # [B, C]
        loss_route = F.cross_entropy(
            torch.log(p_y_x + 1e-8), y_true
        )

        # === Purity Loss (Definition 25.4.1) ===
        # H(Y|K=k) for each chart
        entropy_per_chart = -(p_y_k * torch.log(p_y_k + 1e-8)).sum(dim=1)  # [N_c]
        # P(K=k) = average router weight
        p_k = router_weights.mean(dim=0)  # [N_c]
        # L_purity = sum_k P(K=k) * H(Y|K=k)
        loss_purity = (p_k * entropy_per_chart).sum()

        # === Balance Loss (Definition 25.4.3) ===
        # KL(p_k || Uniform) = sum_k p_k * log(p_k / (1/N_c)) = sum_k p_k * (log(p_k) + log(N_c))
        uniform = torch.ones_like(p_k) / self.num_charts
        # Manual KL computation: KL(P||Q) = sum P * log(P/Q)
        loss_balance = (p_k * (torch.log(p_k + 1e-8) - torch.log(uniform))).sum()

        # === Metric Contrastive Loss (Definition 25.4.4) ===
        loss_metric = torch.tensor(0.0, device=router_weights.device)
        if self.lambda_metric > 0 and B > 1:
            # Router overlap as proxy for proximity
            # w_i^T w_j measures routing similarity
            overlap = torch.matmul(router_weights, router_weights.t())  # [B, B]

            # Class disagreement mask
            y_match = (y_true.unsqueeze(1) == y_true.unsqueeze(0)).float()
            y_diff = 1.0 - y_match  # 1 if different classes

            # Penalize high overlap for different-class pairs
            # Using overlap as proxy for d_jump (lower overlap ~ larger distance)
            pseudo_dist = 1.0 - overlap  # Rough proxy
            hinge = F.relu(self.margin - pseudo_dist)
            loss_metric = (y_diff * overlap * hinge ** 2).sum() / (y_diff.sum() + 1e-8)

        # === Total Loss ===
        loss_total = (
            loss_route
            + self.lambda_purity * loss_purity
            + self.lambda_balance * loss_balance
            + self.lambda_metric * loss_metric
        )

        return {
            'loss_total': loss_total,
            'loss_route': loss_route,
            'loss_purity': loss_purity,
            'loss_balance': loss_balance,
            'loss_metric': loss_metric,
        }
```

**Cross-references:** Section 7.8 (Router Weights), Section 7.13 (Jump Operators), Section 3 (Diagnostic Nodes).

:::
(sec-thermodynamics-conditioned-generation)=
### 25.5 Thermodynamics: Conditioned Generation

This framework unifies classification with the generative law (Section 21).

:::{prf:remark} Connection to Möbius Re-centering
:label: rem-connection-to-m-bius-re-centering

The Möbius re-centering $\phi_c$ for conditioned generation (Definition {prf:ref}`ax-bulk-boundary-decoupling`) can be interpreted as centering at the **class centroid**:

$$
c_y := \mathbb{E}_{x: Y(x)=y}[\text{Enc}(x)],
$$
i.e., the average latent position of class-$y$ samples. Conditioned generation "starts" the holographic expansion from this centroid.

:::
:::{prf:proposition} Class-Conditioned Langevin
:label: prop-class-conditioned-langevin

The generative Langevin equation {cite}`welling2011sgld,song2019ncsn` (Definition {prf:ref}`prop-so-d-symmetry-at-origin`) with class conditioning becomes:

$$
dz = -\nabla_G V_y(z, K)\,d\tau + \sqrt{2T_c}\,G^{-1/2}(z)\,dW_\tau,
$$
where $V_y$ is the class-conditioned potential (Definition {prf:ref}`def-class-conditioned-potential`).

*Interpretation:* To generate a sample of class $y$, we run Langevin dynamics with the $V_y$ potential. The semantic term $-\beta_{\text{class}} \log P(Y=y \mid K)$ biases the flow toward class-$y$ charts.

:::
:::{prf:corollary} Label as Symmetry-Breaking Field, cf. classifier-free guidance {cite}`ho2022cfg`
:label: cor-label-as-symmetry-breaking-field-cf-classifier-free-guidance

The class label $y$ breaks the $SO(2)$ symmetry of the unconditioned flow in the Poincare disk. At the origin:

1. **Unconditioned:** $\nabla V_{\text{base}}(0) = 0$ (symmetric saddle)
2. **Conditioned:** $\nabla V_y(0) = -\beta_{\text{class}} \nabla_z \log P(Y=y \mid K(z))|_{z=0} \neq 0$

The non-zero gradient aligns the initial "kick" direction with the class-$y$ basin.

:::
:::{prf:definition} Class Centroid in Poincare Disk
:label: def-class-centroid-in-poincar-disk

For the Poincare disk embedding {cite}`nickel2017poincare,ganea2018hnn`, define the class centroid using the **Fréchet mean** {cite}`lou2020frechet`:

$$
c_y := \arg\min_{c \in \mathbb{D}} \sum_{x: Y(x)=y} d_{\mathbb{D}}(c, \text{Enc}(x))^2.
$$
This is well-defined since the Poincare disk has negative curvature (unique Fréchet means).

**Cross-references:** Section 21.2 (Langevin Dynamics), Section 21.3 (Möbius Re-centering), Definition {prf:ref}`prop-so-d-symmetry-at-origin`.

:::
:::{prf:remark} Integration with TopologicalDecoder
:label: rem-integration-with-topologicaldecoder

The TopologicalDecoder (Section 7.10) receives the geometric content $z_{\text{geo}} = e_K + z_n$ and routes through chart-specific projectors. For class-conditioned generation:

1. **Class determines charts:** The class label $y$ biases chart selection toward $\mathcal{A}_y$ via the semantic potential $V_y$
2. **Decoder routing:** The TopologicalDecoder's inverse router (Section 7.10.1) can either:
   - Accept an explicit chart index $K$ (from the generative flow)
   - Infer routing from $z_{\text{geo}}$ (autonomous mode)
3. **Consistency constraint:** The decoder's inferred routing should agree with the encoder's class-conditioned routing:

   $$
   \mathcal{L}_{\text{route-consistency}} = \mathbb{E}_{x,y}\left[\text{CE}\left(w_{\text{dec}}(z_{\text{geo}}), w_{\text{enc}}(x)\right)\right]
   $$
   where $w_{\text{dec}}$ are the decoder's soft router weights and $w_{\text{enc}}$ are the encoder's.

This ensures that class-conditioned generation produces samples that the encoder would classify correctly—a form of **cycle consistency** between encoding and decoding under the semantic topology.

:::
(sec-hierarchical-classification-via-scale-decomposition)=
### 25.6 Hierarchical Classification via Scale Decomposition

Real-world categories are hierarchical (e.g., Animal → Dog → Terrier). The **stacked TopoEncoder** (Section 7.12) naturally reflects this.

:::{prf:definition} Hierarchical Labels
:label: def-hierarchical-labels

A **label hierarchy** is a sequence of label spaces:

$$
\mathcal{Y}_0 \twoheadrightarrow \mathcal{Y}_1 \twoheadrightarrow \cdots \twoheadrightarrow \mathcal{Y}_L,
$$
where $\twoheadrightarrow$ denotes a surjection (coarsening). $\mathcal{Y}_0$ are coarse labels (super-categories), $\mathcal{Y}_L$ are fine labels (leaf categories).

*Example:* $\mathcal{Y}_0 = \{\text{Animal}, \text{Vehicle}\}$, $\mathcal{Y}_1 = \{\text{Dog}, \text{Cat}, \text{Car}, \text{Bike}\}$, $\mathcal{Y}_2 = \{\text{Terrier}, \text{Poodle}, \ldots\}$.

:::
:::{prf:proposition} Scale-Label Alignment
:label: prop-scale-label-alignment

In the stacked TopoEncoder (Section 7.12), enforce purity at each scale:

- **Layer 0 (Bulk/Slow):** Charts at level 0 correspond to coarse classes. Enforce:

  $$
  \mathcal{L}_{\text{purity}}^{(0)} = H(\mathcal{Y}_0 \mid K^{(0)})
  $$
- **Layer $\ell$ (Intermediate):** Charts at level $\ell$ correspond to level-$\ell$ classes. Enforce:

  $$
  \mathcal{L}_{\text{purity}}^{(\ell)} = H(\mathcal{Y}_\ell \mid K^{(\ell)})
  $$
- **Layer $L$ (Boundary/Fast):** Charts at level $L$ correspond to fine classes. Enforce:

  $$
  \mathcal{L}_{\text{purity}}^{(L)} = H(\mathcal{Y}_L \mid K^{(L)})
  $$
:::
:::{prf:remark} Renormalization Group Interpretation
:label: rem-renormalization-group-interpretation

The semantic hierarchy matches the physical renormalization scale:

| Scale                | Latent Structure              | Semantic Structure |
|----------------------|-------------------------------|--------------------|
| Bulk (Layer 0)       | Slow modes, large wavelengths | Super-categories   |
| Intermediate         | Medium modes                  | Categories         |
| Boundary (Layer $L$) | Fast modes, fine details      | Sub-categories     |

This is the **semantic RG flow**: coarse-graining in the label space corresponds to flowing toward the bulk in latent space.

:::
:::{prf:definition} Hierarchical Supervised Loss
:label: def-hierarchical-supervised-loss

The total hierarchical loss:

$$
\mathcal{L}_{\text{hier}} = \sum_{\ell=0}^{L} \alpha_\ell \left(\mathcal{L}_{\text{route}}^{(\ell)} + \lambda_{\text{pur}} \mathcal{L}_{\text{purity}}^{(\ell)}\right),
$$
where $\alpha_\ell$ weights the contribution of each scale (typically $\alpha_\ell = 1$ or decaying with $\ell$).

**Cross-references:** Section 7.12 (Stacked TopoEncoder), Definition {prf:ref}`def-the-peeling-step`, Section 7.12.3 (RG Interpretation).

:::
(sec-summary-and-diagnostic-nodes)=
### 25.7 Summary and Diagnostic Nodes

**Table 25.7.1 (Summary of Supervised Topology Laws).**

| Aspect             | Formula                                                                                   | Units       | Reference                                      |
|--------------------|-------------------------------------------------------------------------------------------|-------------|------------------------------------------------|
| Semantic Partition | $\mathcal{A}_y = \{k: P(Y=y\mid K=k) > 1-\epsilon\}$                                      | —           | Def {prf:ref}`def-semantic-partition`          |
| Class Potential    | $V_y = -\beta_{\text{class}} \log P(Y=y\mid K) + V_{\text{base}}$                         | nat         | Def {prf:ref}`def-class-conditioned-potential` |
| Jump Modulation    | $\lambda_{i\to j}^{\text{sup}} = \lambda^{(0)} e^{-\gamma_{\text{sep}} D_{\text{class}}}$ | step$^{-1}$ | Def {prf:ref}`def-class-consistent-jump-rate`  |
| Purity Loss        | $\sum_k P(K=k) H(Y\mid K=k)$                                                              | nat         | Def {prf:ref}`def-purity-loss`                 |
| Route Alignment    | $\text{CE}(\sum_k w_k P(Y\mid K=k), y_{\text{true}})$                                     | nat         | Def {prf:ref}`def-route-alignment-loss`        |

(node-40)=
**Node 40: PurityCheck (CapacitySaturationCheck)**

Following the diagnostic node convention (Section 3.1), we define:

| **#**  | **Name**        | **Component** | **Type**                | **Interpretation**     | **Proxy**     | **Cost** |
|--------|-----------------|---------------|-------------------------|------------------------|---------------|----------|
| **40** | **PurityCheck** | **Router**    | **Semantic Clustering** | Are charts class-pure? | $H(Y \mid K)$ | $O(BC)$  |

**Trigger conditions:**
- High PurityCheck: Charts contain mixed classes; classification boundaries fall within charts.
- Remedy: Increase purity loss weight $\lambda_{\text{pur}}$; increase number of charts; check for insufficient training data per class.

(node-41)=
**Node 41: ClassSeparationCheck (SupervisedTopologyChecks)**

| **#**  | **Name**                 | **Component** | **Type**             | **Interpretation**                          | **Proxy**                                                                  | **Cost**     |
|--------|--------------------------|---------------|----------------------|---------------------------------------------|----------------------------------------------------------------------------|--------------|
| **41** | **ClassSeparationCheck** | **Jump Op**   | **Class Separation** | Are different classes metrically separated? | $\min_{y_1 \neq y_2} d_{\text{WFR}}(\mathcal{A}_{y_1}, \mathcal{A}_{y_2})$ | $O(C^2 N_c)$ |

**Trigger conditions:**
- Low ClassSeparationCheck: Different classes are metrically close; cross-class transitions are too frequent.
- Remedy: Increase separation strength $\gamma_{\text{sep}}$; add metric contrastive loss; check for class imbalance.

**Cross-references:** Section 3 (Sieve Diagnostic Nodes), Section 24.7 (Scalar Field Diagnostics).



(sec-theory-of-meta-stability-the-universal-governor-as-homeostatic-controller)=
