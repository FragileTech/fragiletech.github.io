## 8. Infeasible Implementation Replacements

:::{admonition} Researcher Bridge: Practical Substitutions for Idealized Laws
:class: tip
:name: rb-practical-substitutions
Many theoretical constraints are too expensive to compute directly. This section provides the RL-engineering replacements (surrogate losses, probes, and bounds) that preserve the same failure detection in practice.
:::

Several regularization terms from the theoretical framework are computationally infeasible for standard training. This section provides practical alternatives with full PyTorch implementations.

(sec-barrierbode-temporal-gain-margin)=
### 8.1 BarrierBode → Temporal Gain Margin

**Original (Infeasible):**

$$
\int_{0}^{\infty} \log \lvert S(j\omega) \rvert d\omega = \text{const.} \quad \text{(Bode sensitivity integral)}
$$
**Problem:** Requires frequency-domain analysis of the closed-loop transfer function $S(j\omega)$. Neural policies don't have closed-form transfer functions, and FFT requires long stationary trajectories.

**Replacement: Temporal Gain Margin**

$$
\mathcal{L}_{\text{gain}} = \sum_{k=1}^{K} \max\left(0, \frac{\Vert e_{t+k} \Vert}{\Vert e_t \Vert + \epsilon} - G_{\max}\right)^2
$$
This surrogate loss penalizes error amplification and oscillatory instability without requiring LTI assumptions.

```python
def compute_gain_margin_loss(
    errors: torch.Tensor,  # Shape: [B, T] - tracking errors over time
    G_max: float = 2.0,     # Maximum allowed gain
    K: int = 5,             # Lookahead horizon
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    BarrierBode replacement: Temporal gain margin constraint.

    Penalizes trajectories where errors amplify over time,
    corresponding to the loop instability detected by Bode sensitivity analysis.

    Args:
        errors: [B, T] tensor of error magnitudes at each timestep
        G_max: Maximum allowed amplification ratio
        K: Number of steps to check ahead
        eps: Numerical stability

    Returns:
        Scalar loss penalizing gain violations
    """
    B, T = errors.shape
    if T <= K:
        return torch.tensor(0.0, device=errors.device)

    total_violation = 0.0
    for k in range(1, min(K + 1, T)):
        # Gain at lag k: ||e_{t+k}|| / ||e_t||
        e_t = errors[:, :-k]  # [B, T-k]
        e_t_plus_k = errors[:, k:]  # [B, T-k]

        gain = e_t_plus_k / (e_t + eps)
        violation = torch.relu(gain - G_max).pow(2)
        total_violation = total_violation + violation.mean()

    return total_violation / K


# Alternative: Peak gain detection
def compute_peak_gain_loss(
    errors: torch.Tensor,  # [B, T]
    G_max: float = 2.0,
) -> torch.Tensor:
    """Simplified variant: penalizes maximum gain ratio."""
    B, T = errors.shape
    e_ratios = errors[:, 1:] / (errors[:, :-1] + 1e-6)
    max_gain = e_ratios.max(dim=-1).values  # [B]
    return torch.relu(max_gain - G_max).pow(2).mean()
```

(sec-bifurcatecheck-stochastic-jacobian-probing)=
### 8.2 BifurcateCheck → Stochastic Jacobian Probing

**Original (Infeasible):**

$$
\det(J_{S_t}) \quad \text{where } J_{S_t} = \frac{\partial S_t(z)}{\partial z}
$$
**Problem:** Computing the full Jacobian is $O(Z^3)$. For $Z = 256$, this is ~16M operations per sample.

**Replacement: Hutchinson-style Jacobian Probing**

$$
\mathcal{L}_{\text{bifurcate}} = \text{Var}_v\left[\Vert J_{S_t} v \Vert^2\right] \quad \text{where } v \sim \mathcal{N}(0, I)
$$
High variance in the Jacobian-vector product norm indicates instability (eigenvalue spread).

```python
def compute_bifurcation_loss(
    world_model: nn.Module,
    z: torch.Tensor,           # [B, Z] - current latent states
    a: torch.Tensor,           # [B, A] - actions (if needed)
    n_probes: int = 5,
    instability_threshold: float = 1.0,
) -> torch.Tensor:
    """
    BifurcateCheck replacement: Stochastic Jacobian probing.

    Uses Hutchinson trace estimator principle: instead of computing
    full Jacobian, probe with random vectors. High variance in
    ||J @ v|| indicates eigenvalue spread → bifurcation sensitivity.

    Args:
        world_model: S_t(z, a) -> z_next
        z: Current latent states [B, Z]
        a: Actions [B, A]
        n_probes: Number of random direction probes
        instability_threshold: Variance threshold for penalty

    Returns:
        Scalar loss penalizing high Jacobian variance
    """
    B, Z = z.shape
    z = z.requires_grad_(True)

    # Forward through world model
    z_next = world_model(z, a)  # [B, Z]

    jvp_norms = []
    for _ in range(n_probes):
        # Random probe direction
        v = torch.randn_like(z)  # [B, Z]

        # Jacobian-vector product via autodiff (efficient: O(Z))
        jvp = torch.autograd.grad(
            outputs=z_next,
            inputs=z,
            grad_outputs=v,
            create_graph=True,
            retain_graph=True,
        )[0]  # [B, Z]

        jvp_norm = jvp.norm(dim=-1)  # [B]
        jvp_norms.append(jvp_norm)

    # Stack and compute variance across probes
    jvp_norms = torch.stack(jvp_norms, dim=0)  # [n_probes, B]
    variance = jvp_norms.var(dim=0).mean()  # Average variance across batch

    # Penalize high variance (indicates instability)
    loss = torch.relu(variance - instability_threshold).pow(2)

    return loss
```

(sec-tamecheck-lipschitz-gradient-proxy)=
### 8.3 TameCheck → Lipschitz Gradient Proxy

**Original (Infeasible):**

$$
\Vert \nabla^2 S_t \Vert \quad \text{(Hessian norm)}
$$
**Problem:** Full Hessian is $O(Z^2 \times P_{WM})$ — prohibitive for large world models.

**Replacement: Lipschitz of Gradient**

$$
\mathcal{L}_{\text{tame}} = \frac{\Vert \nabla_z S_t(z_1) - \nabla_z S_t(z_2) \Vert}{\Vert z_1 - z_2 \Vert + \epsilon}
$$
Bounded gradient Lipschitz constant implies bounded Hessian (by definition).

```python
def compute_tame_loss(
    world_model: nn.Module,
    z: torch.Tensor,         # [B, Z]
    a: torch.Tensor,         # [B, A]
    perturbation_scale: float = 0.01,
    lipschitz_target: float = 1.0,
) -> torch.Tensor:
    """
    TameCheck replacement: Lipschitz gradient constraint.

    Instead of computing full Hessian, we estimate the Lipschitz
    constant of the gradient via finite differences. This bounds
    the Hessian spectral norm (tameness).

    Args:
        world_model: S_t(z, a) -> z_next
        z: Current latent states [B, Z]
        a: Actions [B, A]
        perturbation_scale: Size of random perturbation
        lipschitz_target: Target Lipschitz constant

    Returns:
        Scalar loss penalizing non-tame dynamics
    """
    B, Z = z.shape

    # Two nearby points
    z1 = z.requires_grad_(True)
    delta = torch.randn_like(z) * perturbation_scale
    z2 = (z + delta).requires_grad_(True)

    # Forward passes
    z1_next = world_model(z1, a)
    z2_next = world_model(z2, a)

    # Compute gradients at both points
    # Sum over output dims to get [B, Z] gradient
    grad1 = torch.autograd.grad(
        z1_next.sum(), z1, create_graph=True, retain_graph=True
    )[0]  # [B, Z]

    grad2 = torch.autograd.grad(
        z2_next.sum(), z2, create_graph=True, retain_graph=True
    )[0]  # [B, Z]

    # Lipschitz estimate: ||grad1 - grad2|| / ||z1 - z2||
    grad_diff = (grad1 - grad2).norm(dim=-1)  # [B]
    z_diff = delta.norm(dim=-1) + 1e-6  # [B]

    lipschitz_estimate = grad_diff / z_diff  # [B]

    # Penalize exceeding target Lipschitz constant
    loss = torch.relu(lipschitz_estimate - lipschitz_target).pow(2).mean()

    return loss
```

(sec-topocheck-value-gradient-alignment)=
### 8.4 TopoCheck → Value Gradient Alignment

**Original (Infeasible):**

$$
T_{\text{reach}}(z_{\text{goal}}) \quad \text{(Reachability time)}
$$
**Problem:** Requires multi-step planning through world model: $O(H \times B \times Z)$ with potentially large horizon $H$.

**Replacement: Value Gradient Alignment**

$$
\mathcal{L}_{\text{topo}} = -\left\langle \nabla_z V(z), \frac{z_{\text{goal}} - z}{\Vert z_{\text{goal}} - z \Vert} \right\rangle
$$
When $\nabla_z V(z)$ aligns with the goal direction, gradient ascent on $V$ yields a path to $z_{\text{goal}}$.

```python
def compute_topo_loss(
    critic: nn.Module,
    states: torch.Tensor,      # [B, Z] - current states
    goal_states: torch.Tensor,  # [B, Z] or [Z] - goal states
) -> torch.Tensor:
    """
    TopoCheck replacement: Value gradient alignment.

    Instead of computing multi-step reachability, we check if
    the critic's value gradient points toward the goal. This is
    a necessary condition for gradient-based reachability.

    Args:
        critic: V(z) -> scalar value
        states: Current states [B, Z]
        goal_states: Target states [B, Z] or [Z]

    Returns:
        Scalar loss (negative = gradient points toward goal)
    """
    B, Z = states.shape
    states = states.requires_grad_(True)

    # Compute value and its gradient
    values = critic(states)  # [B]
    grad_v = torch.autograd.grad(
        values.sum(), states, create_graph=True
    )[0]  # [B, Z]

    # Direction to goal
    if goal_states.dim() == 1:
        goal_states = goal_states.unsqueeze(0).expand(B, -1)

    to_goal = goal_states - states  # [B, Z]
    to_goal_normalized = to_goal / (to_goal.norm(dim=-1, keepdim=True) + 1e-6)

    # Alignment: should be negative (V decreases toward goal)
    alignment = (grad_v * to_goal_normalized).sum(dim=-1)  # [B]

    # Loss: penalize positive alignment (wrong direction)
    loss = torch.relu(alignment).mean()

    return loss
```

(sec-geomcheck-efficient-infonce)=
### 8.5 GeomCheck → Efficient InfoNCE

**Original (Expensive):**

$$
\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp(\text{sim}(z_t, z_{t+k}))}{\sum_{j=1}^{B} \exp(\text{sim}(z_t, z_j))}
$$
**Problem:** Full pairwise computation is $O(B^2 \times Z)$.

**Replacement: Sampled InfoNCE**

$$
\mathcal{L}_{\text{InfoNCE}}^{\text{eff}} = -\log \frac{\exp(\text{sim}(z_t, z_{t+k}))}{\exp(\text{sim}(z_t, z_{t+k})) + \sum_{j=1}^{K} \exp(\text{sim}(z_t, z_{\text{neg},j}))}
$$
Use $K \ll B$ sampled negatives instead of full batch.

```python
class EfficientInfoNCE(nn.Module):
    """
    GeomCheck replacement: Efficient contrastive loss.

    Uses K sampled negatives instead of full batch pairwise.
    Reduces O(B²Z) to O(KBZ) where K << B.
    """

    def __init__(
        self,
        latent_dim: int,
        n_negatives: int = 128,
        tau: float = 0.1,  # softmax scale
    ):
        super().__init__()
        self.n_negatives = n_negatives
        self.tau = tau

        # Projection head (optional, improves quality)
        self.projector = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(
        self,
        z_anchor: torch.Tensor,   # [B, Z] - z_t
        z_positive: torch.Tensor,  # [B, Z] - z_{t+k} (temporally close)
        z_bank: torch.Tensor = None,  # [M, Z] - memory bank for negatives
    ) -> torch.Tensor:
        """
        Compute efficient InfoNCE loss.

        Args:
            z_anchor: Anchor embeddings [B, Z]
            z_positive: Positive pairs [B, Z]
            z_bank: Optional memory bank for negatives [M, Z]

        Returns:
            Scalar contrastive loss
        """
        B, Z = z_anchor.shape

        # Project
        anchor = F.normalize(self.projector(z_anchor), dim=-1)  # [B, Z]
        positive = F.normalize(self.projector(z_positive), dim=-1)  # [B, Z]

        # Sample negatives
        if z_bank is not None and z_bank.shape[0] >= self.n_negatives:
            # Sample from memory bank
            indices = torch.randperm(z_bank.shape[0])[:self.n_negatives]
            negatives = z_bank[indices]  # [K, Z]
            negatives = F.normalize(self.projector(negatives), dim=-1)
        else:
            # Use other batch elements as negatives (in-batch)
            K = min(self.n_negatives, B - 1)
            # Shuffle and take first K (excluding self)
            perm = torch.randperm(B, device=z_anchor.device)
            negatives = anchor[perm[:K]]  # [K, Z]

        # Positive similarity: [B]
        pos_sim = (anchor * positive).sum(dim=-1) / self.tau

        # Negative similarities: [B, K]
        neg_sim = torch.mm(anchor, negatives.T) / self.tau

        # InfoNCE: log(exp(pos) / (exp(pos) + sum(exp(neg))))
        # = pos - log(exp(pos) + sum(exp(neg)))
        # = pos - logsumexp([pos, neg1, neg2, ...])

        # Combine for logsumexp: [B, K+1]
        all_sim = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1)

        # Loss: -pos + logsumexp(all)
        loss = -pos_sim + torch.logsumexp(all_sim, dim=-1)

        return loss.mean()


# Usage example:
def compute_geom_loss(
    vae_encoder: nn.Module,
    x_t: torch.Tensor,      # [B, D] - observation at time t
    x_t_plus_k: torch.Tensor,  # [B, D] - observation at time t+k
    info_nce: EfficientInfoNCE,
) -> torch.Tensor:
    """GeomCheck: Contrastive anchoring for latent space."""
    z_t = vae_encoder(x_t)
    z_t_k = vae_encoder(x_t_plus_k)
    return info_nce(z_t, z_t_k)
```

(sec-summary-replacement-mapping)=
### 8.6 Summary: Replacement Mapping

| Original                  | Replacement        | Speedup  | Preserved Property              |
|---------------------------|--------------------|----------|---------------------------------|
| BarrierBode (FFT)         | Temporal Gain      | ~100×    | Detects oscillatory instability |
| BifurcateCheck ($O(Z^3)$) | Jacobian Probing   | ~$Z^2/K$ | Detects eigenvalue spread       |
| TameCheck ($O(Z^2 P)$)    | Lipschitz Gradient | ~$ZP$    | Bounds Hessian norm             |
| TopoCheck ($O(HBZ)$)      | Value Alignment    | ~$H$     | Ensures goal reachability       |
| GeomCheck ($O(B^2 Z)$)    | Sampled NCE        | ~$B/K$   | Preserves slow features         |



(sec-the-disentangled-variational-architecture-hierarchical-latent-separation)=
