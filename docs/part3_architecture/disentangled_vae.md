## 9. The Disentangled Variational Architecture: Hierarchical Latent Separation

:::{admonition} Researcher Bridge: World Models with Typed Latents
:class: info
:name: rb-world-models
Extends world-model architectures (Dreamer, MuZero) with explicit separation between control-relevant symbols, structured nuisance, and texture. The objective is to prevent policies from depending on high-frequency detail while preserving reconstruction fidelity.
:::

This section provides a practical guide to implementing a **split-latent** architecture that separates a *predictive* macro register from two distinct residual channels: **structured nuisance** $z_n$ (pose/basis/disturbance coordinates that can be modeled and audited) and **texture** $z_{\mathrm{tex}}$ (reconstruction-only detail). This targets **BarrierEpi** (information overload) and **BarrierOmin** (model mismatch) by preventing the World Model and policy from silently depending on texture while still representing nuisance explicitly.

(sec-the-core-concept-split-brain-architecture)=
### 9.1 The Core Concept: Split-Brain Architecture

**Standard Agent:** Encodes the state into a single vector $z$.

**Disentangled Agent:** Encodes the state into a **discrete macro-symbol** plus two distinct continuous residual channels (Section 2.2b):

1. **$K_t \in \mathcal{K}$ (The Macro Symbol / Law Register):** Low-frequency, causal, predictable. This is the *quantized* macrostate (a code index). For downstream continuous networks we also use its embedding $z_{\text{macro}}:=e_{K_t}\in\mathbb{R}^{d_m}$, but the *information-carrying* object is the discrete symbol $K_t$.

2. **$z_{n,t}$ (Structured Nuisance / Gauge Residual):** A continuous latent for pose/basis/disturbance coordinates. This is *not* “noise”: it is structured variation that may be needed for actuation and for explaining boundary-driven deviations, but it must remain disentangled from macro identity and must not be required for predicting macro transitions beyond $(K_t,a_t)$.

3. **$z_{\mathrm{tex},t}$ (Texture Residual):** A high-rate continuous latent for reconstruction detail. Texture is treated as an **emission residual**: it may be needed to reconstruct $x_t$ but must not be required for macro closure or for control.

**The Golden Rule of Causal Enclosure:**

$$
P(K_{t+1}\mid K_t,a_t)\ \text{is sharply concentrated (ideally deterministic), and}\ I(K_{t+1};Z_{\mathrm{tex},t}\mid K_t,a_t)=0.
$$
(Optionally, and in the strongest form, also $I(K_{t+1};Z_{n,t}\mid K_t,a_t)=0$: nuisance should not be needed to predict the next macro symbol once action is accounted for.)

The macro symbol must be predictable **solely from its own history** (plus action). If the World Model needs micro-residuals to predict the next macro symbol, then $K_t$ is not a sufficient macro statistic (Section 2.8).

(sec-architecture-the-disentangled-vq-vae-rnn)=
### 9.2 Architecture: The Disentangled VQ-VAE-RNN

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class DisentangledConfig:
    """Configuration for split-latent (macro + nuisance + texture) agent."""
    obs_dim: int = 64 * 64 * 3      # Observation dimension
    hidden_dim: int = 256            # Encoder hidden dimension
    macro_embed_dim: int = 32        # Macro embedding dim (code vectors e_k)
    codebook_size: int = 512         # Number of discrete macrostates |𝒦|
    nuisance_dim: int = 32           # Structured nuisance latent dimension
    tex_dim: int = 96                # Texture latent dimension (reconstruction-only)
    action_dim: int = 4              # Action dimension
    rnn_hidden_dim: int = 256        # Dynamics model RNN hidden

    # Loss weights
    lambda_closure: float = 1.0      # Causal enclosure weight
    lambda_slowness: float = 0.1     # Temporal smoothness weight
    lambda_nuis_kl: float = 0.01     # Nuisance KL weight (regularize, not “trash”)
    lambda_tex_kl: float = 0.05      # Texture KL weight (reconstruction residual)
    lambda_vq: float = 1.0           # VQ codebook + commitment weight
    lambda_recon: float = 1.0        # Reconstruction weight

    # Training
    tex_dropout_prob: float = 0.5    # Probability of dropping texture (forces macro+nuisance decoding)
    warmup_steps: int = 1000         # Warmup for closure loss


class Encoder(nn.Module):
    """Shared encoder backbone."""

    def __init__(self, obs_dim: int, hidden_dim: int):
        super().__init__()
        # For image observations, use CNN
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Compute flattened size (for 64x64 input: 256 * 4 * 4 = 4096)
        self.fc = nn.Linear(4096, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] or [B, T, C, H, W]
        if x.dim() == 5:
            B, T = x.shape[:2]
            x = x.view(B * T, *x.shape[2:])
            h = F.relu(self.fc(self.conv(x)))
            return h.view(B, T, -1)
        return F.relu(self.fc(self.conv(x)))


class VectorQuantizer(nn.Module):
    """
    VQ layer: maps a continuous encoder output z_e to a discrete code K and
    its corresponding embedding e_K using a straight-through estimator.
    """

    def __init__(self, codebook_size: int, embed_dim: int, beta: float = 0.25):
        super().__init__()
        self.codebook = nn.Embedding(codebook_size, embed_dim)
        self.beta = beta
        nn.init.uniform_(self.codebook.weight, -1.0 / codebook_size, 1.0 / codebook_size)

    def forward(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # z_e: [B, D]
        # Compute squared L2 distances to codebook vectors
        codebook = self.codebook.weight  # [|𝒦|, D]
        distances = (
            z_e.pow(2).sum(dim=-1, keepdim=True)
            - 2 * z_e @ codebook.t()
            + codebook.pow(2).sum(dim=-1, keepdim=True).t()
        )
        K = torch.argmin(distances, dim=-1)          # [B]
        z_q = self.codebook(K)                       # [B, D]

        # VQ-VAE losses
        codebook_loss = (z_q - z_e.detach()).pow(2).mean()
        commit_loss = self.beta * (z_e - z_q.detach()).pow(2).mean()
        vq_loss = codebook_loss + commit_loss

        # Straight-through estimator: pass gradients to encoder as if identity
        z_q_st = z_e + (z_q - z_e).detach()
        return z_q_st, K, vq_loss

    def embed(self, K: torch.Tensor) -> torch.Tensor:
        return self.codebook(K)


class Decoder(nn.Module):
    """Decoder using macro + nuisance + texture latents."""

    def __init__(self, macro_dim: int, nuisance_dim: int, tex_dim: int, obs_channels: int = 3):
        super().__init__()
        self.fc = nn.Linear(macro_dim + nuisance_dim + tex_dim, 4096)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, obs_channels, 4, stride=2, padding=1),
            nn.Sigmoid(),  # Normalize to [0, 1]
        )

    def forward(self, z_macro: torch.Tensor, z_nuis: torch.Tensor, z_tex: torch.Tensor) -> torch.Tensor:
        z = torch.cat([z_macro, z_nuis, z_tex], dim=-1)
        h = F.relu(self.fc(z))
        h = h.view(-1, 256, 4, 4)
        return self.deconv(h)


class MacroDynamicsModel(nn.Module):
    """
    Macro dynamics model (micro-blind).

    Important: this module only sees z_macro (it is blind to z_nuis and z_tex). This
    forces causally/predictively relevant information into the macro channel.
    """

    def __init__(self, macro_embed_dim: int, action_dim: int, hidden_dim: int, codebook_size: int):
        super().__init__()
        self.macro_embed_dim = macro_embed_dim
        self.codebook_size = codebook_size

        # GRU for temporal dynamics
        self.gru = nn.GRUCell(macro_embed_dim + action_dim, hidden_dim)

        # Project hidden state to a distribution over next macro symbol K_{t+1}
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, codebook_size),  # logits over 𝒦
        )

        # Uncertainty estimation (optional but recommended)
        self.uncertainty = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        z_macro: torch.Tensor,       # [B, macro_embed_dim] (= code embedding e_K)
        action: torch.Tensor,        # [B, action_dim]
        h_prev: torch.Tensor,        # [B, hidden_dim]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict the next macro symbol distribution from the current macro embedding only.

        Returns:
            logits_next: Logits for K_{t+1} over 𝒦
            h_next: Updated hidden state
            uncertainty: Scalar uncertainty proxy
        """
        # Concatenate macro state and action (NO micro!)
        x = torch.cat([z_macro, action], dim=-1)

        # Update hidden state
        h_next = self.gru(x, h_prev)

        # Predict next macro symbol (logits over 𝒦)
        logits_next = self.predictor(h_next)
        uncertainty = self.uncertainty(h_next)

        return logits_next, h_next, uncertainty


class DisentangledAgent(nn.Module):
    """
    Split-latent VQ-VAE + macro dynamics model.

    Separates:
    - K_t (macro): a discrete symbol in 𝒦 (predictive/controllable state)
    - z_n (nuisance): a structured residual (pose/basis/disturbance coordinates)
    - z_tex (texture): a reconstruction residual (detail), excluded from macro closure/control
    """

    def __init__(self, config: DisentangledConfig):
        super().__init__()
        self.config = config

        # Shared encoder
        self.encoder = Encoder(config.obs_dim, config.hidden_dim)

        # Macro: continuous pre-quantization → discrete code K via VQ
        self.head_macro = nn.Linear(config.hidden_dim, config.macro_embed_dim)
        self.vq = VectorQuantizer(config.codebook_size, config.macro_embed_dim)

        # Nuisance: structured Gaussian residual
        self.head_nuis_mean = nn.Linear(config.hidden_dim, config.nuisance_dim)
        self.head_nuis_logvar = nn.Linear(config.hidden_dim, config.nuisance_dim)

        # Texture: reconstruction-only Gaussian residual
        self.head_tex_mean = nn.Linear(config.hidden_dim, config.tex_dim)
        self.head_tex_logvar = nn.Linear(config.hidden_dim, config.tex_dim)

        # Macro dynamics model (blind to micro)
        self.macro_dynamics = MacroDynamicsModel(
            config.macro_embed_dim,
            config.action_dim,
            config.rnn_hidden_dim,
            config.codebook_size,
        )

        # Decoder (uses all three channels)
        self.decoder = Decoder(config.macro_embed_dim, config.nuisance_dim, config.tex_dim)

    def encode(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Encode observation into a discrete macro symbol plus nuisance + texture residuals.

        Returns:
            K: Macro code index in 𝒦
            z_macro: Quantized macro embedding e_K (straight-through)
            z_nuis: Nuisance latent
            z_tex: Texture latent
            nuis_dist: (mean, logvar) for nuisance
            tex_dist: (mean, logvar) for texture
            vq_loss: codebook + commitment loss
        """
        features = self.encoder(x)

        # Macro: quantize into a discrete symbol K and embedding z_macro := e_K
        z_e = self.head_macro(features)
        z_macro, K, vq_loss = self.vq(z_e)

        # Nuisance: structured residual
        nuis_mean = self.head_nuis_mean(features)
        nuis_logvar = self.head_nuis_logvar(features)
        nuis_logvar = torch.clamp(nuis_logvar, min=-7, max=2)
        z_nuis = self._reparameterize(nuis_mean, nuis_logvar)

        # Texture: reconstruction-only residual
        tex_mean = self.head_tex_mean(features)
        tex_logvar = self.head_tex_logvar(features)
        tex_logvar = torch.clamp(tex_logvar, min=-7, max=2)
        z_tex = self._reparameterize(tex_mean, tex_logvar)

        return K, z_macro, z_nuis, z_tex, (nuis_mean, nuis_logvar), (tex_mean, tex_logvar), vq_loss

    def _reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(
        self,
        x_t: torch.Tensor,           # [B, C, H, W] current observation
        action: torch.Tensor,         # [B, action_dim]
        h_prev: torch.Tensor,         # [B, rnn_hidden_dim]
        training: bool = True,
    ) -> dict:
        """
        Full forward pass with information dropout.

        Returns dict with all intermediate values for loss computation.
        """
        # 1. Encode current observation
        K_t, z_macro, z_nuis, z_tex, nuis_dist, tex_dist, vq_loss = self.encode(x_t)

        # 2. Macro dynamics step (blind to micro)
        logits_next, h_next, uncertainty = self.macro_dynamics(z_macro, action, h_prev)
        K_pred = torch.argmax(logits_next, dim=-1)
        z_macro_pred = self.vq.embed(K_pred)

        # 3. Texture dropout for reconstruction (forces macro+nuisance to carry structure)
        if training and torch.rand(1).item() < self.config.tex_dropout_prob:
            z_tex_for_decode = torch.zeros_like(z_tex)
        else:
            z_tex_for_decode = z_tex

        # 4. Reconstruct
        x_recon = self.decoder(z_macro, z_nuis, z_tex_for_decode)

        return {
            'K_t': K_t,
            'z_macro': z_macro,
            'z_nuis': z_nuis,
            'z_tex': z_tex,
            'z_macro_logits': logits_next,
            'K_pred': K_pred,
            'z_macro_pred': z_macro_pred,
            'nuis_dist': nuis_dist,
            'tex_dist': tex_dist,
            'vq_loss': vq_loss,
            'h_next': h_next,
            'uncertainty': uncertainty,
            'x_recon': x_recon,
            'tex_dropped': z_tex_for_decode.sum() == 0,
        }

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.config.rnn_hidden_dim, device=device)


```

(sec-loss-function-enforcing-macro-micro-separation)=
### 9.3 Loss Function: Enforcing Macro/Micro Separation

The Disentangled Agent cannot be trained with a reconstruction loss alone. It requires a compound loss that enforces the **learnability threshold**: the macro channel must be predictive/closed, and the micro channel must be treated as nuisance rather than state.

$$
\mathcal{L}_{\text{total}}
=
\lambda_{\text{recon}}\mathcal{L}_{\text{recon}}
\;+\;\lambda_{\text{vq}}\mathcal{L}_{\text{vq}}
\;+\;\lambda_{\text{closure}}\mathcal{L}_{\text{closure}}
\;+\;\lambda_{\text{slowness}}\mathcal{L}_{\text{slowness}}
\;+\;\lambda_{\text{nuis}}\mathcal{L}_{\text{nuis-KL}}
\;+\;\lambda_{\text{tex}}\mathcal{L}_{\text{tex-KL}}
$$
where $\mathcal{L}_{\text{closure}}$ is a cross-entropy on the discrete macro symbols (an estimator of $H(K_{t+1}\mid K_t,a_t)$) and $\mathcal{L}_{\text{vq}}$ is the codebook+commitment term from the VQ layer.

```python
class DisentangledLoss(nn.Module):
    """
    Compound loss for training the split-latent agent.

	    Implements the five key constraints:
	    1. Closure: Macro must predict itself (causal enclosure)
	    2. Slowness: Macro should change slowly (prevents symbol churn)
	    3. Nuisance prior: nuisance should be regularized (but is not “trash”)
	    4. Texture prior: texture should stay close to a simple prior (reconstruction residual)
	    4. VQ: Macro must remain quantized (symbolic)
	    5. Reconstruction: Both channels needed for full reconstruction
	    """

    def __init__(self, config: DisentangledConfig):
        super().__init__()
        self.config = config

    def forward(
        self,
        outputs: dict,              # From DisentangledAgent.forward()
        x_target: torch.Tensor,     # Target observation (x_{t+1} for recon)
        K_next: torch.Tensor,       # Target macro code at t+1 (LongTensor)
        z_macro_prev: Optional[torch.Tensor] = None,  # e_{K_{t-1}} (for slowness)
        step: int = 0,              # Training step for warmup
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute all loss components.

        Returns:
            total_loss: Scalar loss for backprop
            loss_dict: Individual losses for logging
        """
        losses = {}

        # === A. RECONSTRUCTION LOSS ===
        # Standard pixel-wise reconstruction
        losses['recon'] = F.mse_loss(outputs['x_recon'], x_target)

        # === B. CAUSAL ENCLOSURE LOSS (SYMBOLIC) ===
        # Predict the next macro symbol K_{t+1} from the current macro only.
        losses['closure'] = F.cross_entropy(outputs['z_macro_logits'], K_next)

        # Warmup: gradually increase closure weight
        closure_weight = min(1.0, step / self.config.warmup_steps)

        # === C. SLOWNESS LOSS ===
        # Penalize rapid changes in the macro embedding e_K (proxy for symbol churn)
        if z_macro_prev is not None:
            losses['slowness'] = (outputs['z_macro'] - z_macro_prev).pow(2).mean()
        else:
            losses['slowness'] = torch.tensor(0.0, device=outputs['z_macro'].device)

	        # === D. RESIDUAL PRIORS ===
	        # Regularize nuisance and texture toward simple priors (macro closure remains micro-blind).
	        nuis_mean, nuis_logvar = outputs['nuis_dist']
	        tex_mean, tex_logvar = outputs['tex_dist']
	        losses['nuis_kl'] = self._kl_divergence(nuis_mean, nuis_logvar)
	        losses['tex_kl'] = self._kl_divergence(tex_mean, tex_logvar)

        # === E. VQ LOSS (CODEBOOK + COMMITMENT) ===
        losses['vq'] = outputs['vq_loss']

        # === TOTAL LOSS ===
	        total = (
	            self.config.lambda_recon * losses['recon']
	            + self.config.lambda_vq * losses['vq']
	            + self.config.lambda_closure * closure_weight * losses['closure']
	            + self.config.lambda_slowness * losses['slowness']
	            + self.config.lambda_nuis_kl * losses['nuis_kl']
	            + self.config.lambda_tex_kl * losses['tex_kl']
	        )

        losses['total'] = total
        losses['closure_weight'] = closure_weight

        return total, losses

    def _kl_divergence(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """KL(N(mean, var) || N(0, I))"""
        return -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
```

(sec-the-complete-training-loop)=
### 9.4 The Complete Training Loop

```python
class DisentangledTrainer:
    """
    Complete training loop for the split-latent agent.

    Handles:
    - Temporal sequence processing
    - Gradient isolation between macro/micro paths
    - Warmup schedules
    - Diagnostic monitoring
    """

    def __init__(
        self,
        agent: DisentangledAgent,
        learning_rate: float = 3e-4,
        device: str = 'cuda',
    ):
        self.agent = agent.to(device)
        self.device = device
        self.loss_fn = DisentangledLoss(agent.config)

        # Separate optimizers for different components (optional)
        self.optimizer = torch.optim.Adam(agent.parameters(), lr=learning_rate)

        # Diagnostics
        self.closure_history = []
        self.dispersion_history = []
        self.step = 0

    def train_step(
        self,
        observations: torch.Tensor,  # [B, T, C, H, W] sequence
        actions: torch.Tensor,        # [B, T-1, action_dim]
    ) -> dict:
        """
        Train on a sequence of observations.

        Args:
            observations: Temporal sequence of observations
            actions: Actions taken between observations

        Returns:
            Dictionary of losses and diagnostics
        """
        B, T = observations.shape[:2]
        observations = observations.to(self.device)
        actions = actions.to(self.device)

        self.optimizer.zero_grad()

        # Initialize hidden state
        h = self.agent.init_hidden(B, self.device)

        total_loss = 0.0
        all_losses = {k: 0.0 for k in ['recon', 'vq', 'closure', 'slowness', 'dispersion']}

        z_macro_prev = None
        K_nexts = []          # Targets K_{t+1}
        logits_nexts = []     # Predicted logits for K_{t+1}

        for t in range(T - 1):
            x_t = observations[:, t]
            x_t1 = observations[:, t + 1]
            a_t = actions[:, t]

            # Forward pass at time t
            outputs_t = self.agent(x_t, a_t, h, training=True)
            h = outputs_t['h_next']

            # Encode t+1 for closure target
            K_t1, _, _, _, _ = self.agent.encode(x_t1)

            # Compute loss
            loss, losses = self.loss_fn(
                outputs_t,
                x_t1,
                K_t1,
                z_macro_prev,
                self.step,
            )

            total_loss = total_loss + loss
            for k in all_losses:
                if k in losses:
                    all_losses[k] = all_losses[k] + losses[k].item()

            z_macro_prev = outputs_t['z_macro'].detach()
            K_nexts.append(K_t1.detach())
            logits_nexts.append(outputs_t['z_macro_logits'].detach())

        # Backprop
        total_loss = total_loss / (T - 1)
        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.agent.parameters(), max_norm=1.0)

        self.optimizer.step()
        self.step += 1

        # Compute diagnostics
        avg_losses = {k: v / (T - 1) for k, v in all_losses.items()}
        avg_losses['total'] = total_loss.item()

        # Closure ratio (see 9.5): symbolic cross-entropy gain vs baseline
        closure_ratio = self._compute_closure_ratio(K_nexts, logits_nexts)
        avg_losses['closure_ratio'] = closure_ratio

        self.closure_history.append(closure_ratio)

        return avg_losses

    def _compute_closure_ratio(
        self,
        K_nexts: List[torch.Tensor],
        logits_nexts: List[torch.Tensor],
    ) -> float:
        """
        Closure ratio diagnostic (discrete macro).

        Ratio < 1  : predictive macro model beats a baseline (law-like)
        Ratio ≈ 1  : macro is no more predictable than baseline (no learned law)
        Ratio > 1  : predictor is worse than baseline (bug/degenerate training)
        """
        if not K_nexts:
            return float('nan')

        logits = torch.cat(logits_nexts, dim=0)  # [(T-1)B, |𝒦|]
        K_next = torch.cat(K_nexts, dim=0)       # [(T-1)B]

        # Model conditional cross entropy ≈ H(K_{t+1}|K_t,a_t)
        ce_model = F.cross_entropy(logits, K_next, reduction='mean')

        # Baseline: marginal code model (no conditioning)
        K_size = self.agent.config.codebook_size
        hist = torch.bincount(K_next, minlength=K_size).float()
        p = (hist + 1.0) / (hist.sum() + K_size)  # Laplace smoothing
        ce_baseline = (-torch.log(p[K_next])).mean()

        return (ce_model / (ce_baseline + 1e-8)).item()
```

(sec-runtime-diagnostics-the-closure-ratio)=
### 9.5 Runtime Diagnostics: The Closure Ratio

The key diagnostic for macro/micro separation is the **Closure Ratio**:

$$
\text{Closure Ratio}
=
\frac{\mathbb{E}\big[-\log p_\psi(K_{t+1}\mid K_t,a_t)\big]}{\mathbb{E}\big[-\log p_{\text{base}}(K_{t+1})\big]}.
$$
With $p_{\text{base}}$ chosen as the marginal symbol model, the numerator estimates $H(K_{t+1}\mid K_t,a_t)$ and the denominator estimates $H(K_{t+1})$, so the *gap* is a direct estimate of predictive information $I(K_{t+1};K_t,a_t)$.

| Closure Ratio | Interpretation                                                 | Action                                                                            |
|---------------|----------------------------------------------------------------|-----------------------------------------------------------------------------------|
| $\approx 1$   | **No predictive law** — macro dynamics no better than marginal | Increase model capacity or improve macro encoder; check that actions are provided |
| $\ll 1$       | **Success** — conditional entropy collapses vs marginal        | Sufficient macro statistic learned                                                |
| $> 1$         | **Worse than baseline**                                        | Bug/degeneracy (labels, codebook collapse, optimizer instability)                 |

```python
class ClosureMonitor:
    """
    Monitor for split-latent training.

    Integrates with Sieve nodes to detect failure modes.
    """

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.closure_ratios = []
        self.macro_predictability = []
        self.micro_entropy = []

    def update(
        self,
        logits_next: torch.Tensor,   # [B, |𝒦|]
        K_next: torch.Tensor,        # [B]
        micro_logvar: torch.Tensor,  # [B, d_μ]
    ):
        """Update diagnostics with new batch."""
        # Macro predictability: cross entropy (nats)
        ce_model = F.cross_entropy(logits_next, K_next, reduction='mean').item()

        # Baseline: marginal code model (per-batch)
        K_size = logits_next.shape[-1]
        hist = torch.bincount(K_next, minlength=K_size).float()
        p = (hist + 1.0) / (hist.sum() + K_size)  # Laplace smoothing
        ce_base = (-torch.log(p[K_next])).mean().item()

        # Closure ratio
        ratio = ce_model / (ce_base + 1e-8)
        self.closure_ratios.append(ratio)

        # Micro entropy proxy (Gaussian differential entropy)
        import math
        micro_ent = (0.5 * (micro_logvar + math.log(2 * math.pi * math.e))).sum(dim=-1).mean().item()

        # Individual metrics
        self.macro_predictability.append(ce_model)
        self.micro_entropy.append(micro_ent)

        # Keep window
        if len(self.closure_ratios) > self.window_size:
            self.closure_ratios.pop(0)
            self.macro_predictability.pop(0)
            self.micro_entropy.pop(0)

    def get_diagnostics(self) -> dict:
        """Get current diagnostic summary."""
        import numpy as np

        if not self.closure_ratios:
            return {}

        ratios = np.array(self.closure_ratios)
        macro_pred = np.array(self.macro_predictability)
        micro_ent = np.array(self.micro_entropy)

        return {
            'closure_ratio_mean': ratios.mean(),
            'closure_ratio_std': ratios.std(),
            'macro_predictability': macro_pred.mean(),
            'micro_entropy': micro_ent.mean(),
            'macro_model_learned': ratios.mean() < 0.5,  # Success threshold
            'recommendation': self._get_recommendation(ratios.mean()),
        }

    def _get_recommendation(self, ratio: float) -> str:
        if ratio < 0.3:
            return "Excellent: Clear separation of predictive macro and nuisance"
        elif ratio < 0.7:
            return "Good: Macro closure partially learned, consider increasing closure weight"
        elif ratio < 1.0:
            return "Warning: Macro/micro entanglement detected, increase lambda_closure"
        else:
            return "Error: Micro residual more predictive than macro, check architecture"

    def check_sieve_nodes(self) -> dict:
        """
        Map diagnostics to Sieve node checks.

        Returns status for relevant nodes.
        """
        diag = self.get_diagnostics()
        if not diag:
            return {}

        return {
            # TameCheck: Is the world model interpretable?
            'TameCheck': 'PASS' if diag['macro_predictability'] < 1.0 else 'WARN',

            # ComplexCheck: Is model capacity appropriate?
            'ComplexCheck': 'PASS' if diag['micro_entropy'] > -2.0 else 'WARN',

            # GeomCheck: Are latent spaces well-separated?
            'GeomCheck': 'PASS' if diag['closure_ratio_mean'] < 0.5 else 'WARN',

            # ParamCheck: Is the macro dynamics stable?
            'ParamCheck': 'PASS' if diag['closure_ratio_std'] < 0.5 else 'WARN',
        }
```

(sec-advanced-hierarchical-multi-scale-latents)=
### 9.6 Advanced: Hierarchical Multi-Scale Latents

For complex environments, a single macro/micro split may be insufficient. A macro hierarchy extends the split-latent idea to multiple discrete scales:

$$
Z_t = (K_t^{(1)}, K_t^{(2)}, \ldots, K_t^{(L)}, Z_{\mu,t}),
\qquad
z_{\text{macro}}^{(i)} := e^{(i)}_{K_t^{(i)}}\in\mathbb{R}^{d_i}.
$$
Where $K^{(1)}$ is the slowest (most abstract) and $K^{(L)}$ is the fastest (most detailed) macro symbol; $z_{\text{macro}}^{(i)}$ denotes its code embedding.

```python
class HierarchicalDisentangled(nn.Module):
    """
    Multi-scale split-latent architecture.

    Each level operates at a different timescale:
    - Level 1: Slowest (global game state, long-term goals)
    - Level 2: Medium (object positions, velocities)
    - Level 3: Fast (fine motor control, reactions)
    - Micro: Noise (textures, particles)

    Inspired by Clockwork VAE (Saxena et al.) and
    Hierarchical World Models (Hafner et al.)
    """

    def __init__(
        self,
        config: DisentangledConfig,
        n_levels: int = 3,
        level_dims: List[int] = [8, 16, 32],
        level_codebook_sizes: List[int] = [64, 128, 256],
        level_update_freqs: List[int] = [8, 4, 1],  # Update every N steps
    ):
        super().__init__()
        self.n_levels = n_levels
        self.level_dims = level_dims
        self.level_codebook_sizes = level_codebook_sizes
        self.update_freqs = level_update_freqs

        self.encoder = Encoder(config.obs_dim, config.hidden_dim)

        # Separate heads for each macro level (pre-quantization)
        self.macro_heads = nn.ModuleList([
            nn.Linear(config.hidden_dim, dim) for dim in level_dims
        ])

        # VQ quantizers and macro dynamics models at each level
        self.vq_levels = nn.ModuleList([
            VectorQuantizer(level_codebook_sizes[i], level_dims[i])
            for i in range(n_levels)
        ])
        self.macro_dynamics_models = nn.ModuleList([
            MacroDynamicsModel(
                level_dims[i],
                config.action_dim,
                config.rnn_hidden_dim // n_levels,
                level_codebook_sizes[i],
            )
            for i in range(n_levels)
        ])

        # Cross-level connections (top-down modulation)
        self.cross_level = nn.ModuleList([
            nn.Linear(level_dims[i], level_dims[i + 1])
            for i in range(n_levels - 1)
        ])

        # Texture head (reconstruction residual)
        self.tex_head = nn.Linear(config.hidden_dim, config.tex_dim)

        # Decoder uses macro levels + texture (no nuisance channel in this sketch)
        macro_total_dim = sum(level_dims)
        self.decoder = Decoder(macro_total_dim, nuisance_dim=0, tex_dim=config.tex_dim)

        self.step_counter = 0

    def forward(
        self,
        x_t: torch.Tensor,
        action: torch.Tensor,
        h_prevs: List[torch.Tensor],  # Hidden state for each level
        training: bool = True,
    ) -> dict:
        """
        Hierarchical forward pass with clockwork updates.
        """
        features = self.encoder(x_t)

        z_macros = []
        z_macro_preds = []
        h_nexts = []

        top_down_context = None

        for i in range(self.n_levels):
            # Encode this level (pre-quantization)
            z_e_i = self.macro_heads[i](features)

            # Add top-down modulation from slower levels
            if top_down_context is not None:
                z_e_i = z_e_i + self.cross_level[i - 1](top_down_context)

            # Quantize into a discrete macro symbol K^{(i)} and embedding z_macro^{(i)} := e_{K^{(i)}}
            z_macro_i, K_i, _ = self.vq_levels[i](z_e_i)

            # Update macro dynamics only at appropriate frequency
            if self.step_counter % self.update_freqs[i] == 0:
                logits_i, h_next_i, _ = self.macro_dynamics_models[i](
                    z_macro_i, action, h_prevs[i]
                )
                K_pred_i = torch.argmax(logits_i, dim=-1)
                z_pred_i = self.vq_levels[i].embed(K_pred_i)
            else:
                # Hold state
                z_pred_i = z_macro_i
                h_next_i = h_prevs[i]

            z_macros.append(z_macro_i)
            z_macro_preds.append(z_pred_i)
            h_nexts.append(h_next_i)

            top_down_context = z_macro_i

        # Texture (always updates)
        z_tex = self.tex_head(features)

        # Texture dropout: force the macro stack to carry structure
        if training and torch.rand(1).item() < self.config.tex_dropout_prob:
            z_tex = torch.zeros_like(z_tex)

        # Decode from macro levels + texture
        z_macro_all = torch.cat(z_macros, dim=-1)
        z_nuis_empty = torch.zeros(z_tex.shape[0], 0, device=z_tex.device)
        x_recon = self.decoder(z_macro_all, z_nuis_empty, z_tex)

        self.step_counter += 1

        return {
            'z_macros': z_macros,
            'z_macro_preds': z_macro_preds,
            'z_tex': z_tex,
            'h_nexts': h_nexts,
            'x_recon': x_recon,
        }
```

(sec-literature-connections)=
### 9.7 Literature Connections (Mapping + Differences)

This framework is largely a **recomposition** of known ideas. What differs is the insistence on (i) a *discrete* macro state used for prediction/control, and (ii) explicit, online-auditable contracts tying representation, dynamics, and safety.

| Construct in this document                         | Closest literature                                                                  | What is different here                                                                                                                                                                          | Representative references                                                        |
|----------------------------------------------------|-------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| **Discrete macro register $K_t$**                  | discrete latents / vector quantization                                              | $K_t$ is treated as the *control-relevant* state so closure/capacity checks are well-typed, an enabler for audit-friendly information constraints rather than merely a compression mechanism                                                                        | {cite}`oord2017vqvae`                                                            |
| **Causal enclosure / closure loss**                | predictive state representations, state abstraction, bisimulation-style sufficiency | closure is used as an explicit defect functional to certify “macro sufficiency” for predicting macro dynamics                                                                                   | {cite}`littman2001predictive,singh2004predictive,li2006towards,ferns2004metrics` |
| **Typed residual split $(z_n, z_{\mathrm{tex}})$** | rate–distortion / information bottleneck views of representation learning           | nuisance $z_n$ is modeled and auditable (and may be control-relevant), while texture $z_{\mathrm{tex}}$ is explicitly reconstruction-only and prohibited from influencing macro closure/control | {cite}`tishby2015deep`                                                           |
| **Micro-blind macro dynamics $\bar P$**            | latent world models / predictive models                                             | macro dynamics is constrained to depend on $K$ (and action) only; violation is diagnosed as enclosure failure                                                                                   | {cite}`hafner2019dreamer,ha2018worldmodels,lecun2022path`                        |
| **Gate Nodes + Barriers**                          | CMDPs and safe RL constraints                                                       | constraints include *representation* and *interface* diagnostics (grounding, mixing, saturation, switching), not only expected cost                                                             | {cite}`altman1999constrained,achiam2017constrained,chow2018lyapunov`             |
| **MaxEnt/KL control + path-entropy exploration**   | entropy-regularized RL, KL-control, linearly-solvable control                       | exploration is defined on the discrete macro register and tied to capacity/grounding diagnostics                                                                                                | {cite}`haarnoja2018soft,todorov2009efficient,kappen2005path`                     |
| **State-space sensitivity metric $G$**             | information geometry / natural gradient                                             | emphasizes **state-space** sensitivity as a runtime regulator (in addition to parameter-space natural gradients)                                                                                | {cite}`amari1998natural,schulman2015trpo,martens2015kfac`                        |
| **Belief evolution as filter + projection**        | Bayesian filtering + constrained inference                                          | maps Sieve events to explicit belief-space projections/reweightings (predict → update → project)                                                                                                | {cite}`rabiner1989tutorial`                                                      |
| **Entropic OT bridge (optional)**                  | entropic optimal transport / Schrödinger bridge                                     | used only as a unifying *view* of KL-regularized path measures, not as a required ontology                                                                                                      | {cite}`cuturi2013sinkhorn,leonard2014schrodinger`                                |

**Key pointers.**
- Causal emergence / macro closure as a modeling advantage: {cite}`hoel2017map,rosas2020reconciling`
- Information bottleneck perspective: {cite}`tishby2015deep`

(sec-computational-costs)=
### 9.8 Computational Costs

| Loss Component | Formula | Time Complexity | Notes |
|----------------|---------|-----------------|-------|
| $\mathcal{L}_{\text{recon}}$ | $\Vert x - \hat{x} \Vert^2$ | $O(BD)$ | Baseline reconstruction term |
| $\mathcal{L}_{\text{vq}}$ | $\lVert \operatorname{sg}[z_e]-e_{K}\rVert^2 + \beta\lVert z_e-\operatorname{sg}[e_K]\rVert^2$ | $O(B\lvert\mathcal{K}\rvert)$ | Codebook lookup/update |
| $\mathcal{L}_{\text{closure}}$ | $-\log p_\psi(K_{t+1}\mid K_t,a_t)$ | $O(B\lvert\mathcal{K}\rvert)$ | Macro-prediction head + cross-entropy |
| $\mathcal{L}_{\text{slowness}}$ | $\Vert e_{K_t} - e_{K_{t-1}} \Vert^2$ | $O(Bd_m)$ | Embedding drift penalty |
| $\mathcal{L}_{\text{dispersion}}$ | $D_{\mathrm{KL}}(q_{\text{micro}} \Vert \mathcal{N}(0,I))$ | $O(BZ_\mu)$ | Micro KL / nuisance regularizer |

Total cost depends on $|\mathcal{K}|$, whether closure is computed online or intermittently, and whether heavy diagnostics are amortized across steps.

**When to Use Split-Latent vs Standard:**

| Scenario                              | Recommendation                                                                  |
|---------------------------------------|---------------------------------------------------------------------------------|
| High-frequency texture (games, video) | Use split-latent — separate predictive state from nuisance texture              |
| Low-noise simulation (MuJoCo)         | Standard may suffice — dynamics are already clean                               |
| Real-world robotics                   | Use split-latent — sensor noise is significant                                  |
| Long-horizon planning                 | Use hierarchical split-latent — multiple timescales                             |
| Compute-constrained                   | Standard may be preferable if codebook + closure diagnostics are not affordable |

(sec-control-theory-translation-dictionary)=
### 9.9 Control Theory Translation: Dictionary

To ensure rigorous connections to the established literature, we explicitly map Hypostructure components to their Control Theory and optimization equivalents.

| Hypostructure Component                                    | Control / Optimization Term                       | Role                                                          |
|:-----------------------------------------------------------|:--------------------------------------------------|:--------------------------------------------------------------|
| **Critic**                                                 | **Lyapunov / value function** ($V$)               | Defines stability regions and cost contours.                  |
| **Policy**                                                 | **Lie Derivative Controller** ($\mathcal{L}_f V$) | Actuator that maximizes negative definiteness of $\dot{V}$.   |
| **World Model**                                            | **System Dynamics** ($f(x, u)$)                   | The vector field governing the flow.                          |
| **Fragile Index**                                          | **State-space sensitivity metric** ($G_{ij}$)     | Local conditioning from Hessian/Fisher structure.             |
| **StiffnessCheck**                                         | **LaSalle's Invariance Principle**                | Guarantee that the system does not get stuck in limit cycles. |
| **BarrierAction**                                          | **Controllability Gramian**                       | Measure of whether the actuator can affect the state.         |
| **Scaling coefficients** ($\alpha, \beta, \gamma, \delta$) | **Learning-dynamics coefficients**                | Relative update/volatility scales per component.              |
| **BarrierTypeII**                                          | **Scaling Hierarchy**                             | Ensures faster components don't outrun slower ones.           |

**Related Work:**
- {cite}`chang2019neural` — Neural Lyapunov Control
- {cite}`berkenkamp2017safe` — Safe Model-Based RL with Stability Guarantees
- {cite}`lasalle1960extent` — The Extent of Asymptotic Stability

(sec-differential-geometry-view-curvature-as-conditioning)=
### 9.10 Differential-Geometry View (No Physics): Curvature as Conditioning

The Fragile Agent uses differential geometry as a **regulation tool**: the metric $G$ (from Hessian curvature and/or Fisher information) defines a local notion of distance/conditioning in latent space, and therefore how aggressively the controller should update.

**Core relationship.** Objective curvature defines $G$; $G$ defines how updates are scaled:

$$
\theta_{t+1} = \theta_t + \eta\,G^{-1}\nabla_\theta \mathcal{L}.
$$
**Dictionary (geometry → optimization).**

| Differential geometry / optimization | Fragile Agent            | Interpretation                              |
|:-------------------------------------|:-------------------------|:--------------------------------------------|
| Metric tensor                        | $G$                      | Local sensitivity / conditioning            |
| Metric distance                      | $d_G(\cdot,\cdot)$       | Trust-region measure in latent space        |
| Natural gradient                     | $G^{-1}\nabla$           | Preconditioned update direction             |
| Ill-conditioning                     | $\lambda_{\max}(G)\gg 1$ | Updates should shrink to maintain stability |

**Adaptive conditioning (“freeze-out”).** When $G$ becomes extremely ill-conditioned, geometry-aware updates shrink ($G^{-1}\to 0$ along stiff directions), preventing destabilizing steps and signaling that the current regime is hard to model/control with available capacity.

**Related Work:**
- {cite}`bronstein2021geometric` — Geometric Deep Learning (geometry for inductive bias)
- {cite}`amari1998natural` — Natural Gradient (information geometry)

**Key Distinction from Geometric Deep Learning:**

Geometric Deep Learning uses geometry to design **architectures** (equivariant neural networks). The Fragile Agent uses geometry for **runtime regulation**: curvature is estimated from the critic/policy sensitivity and used to adapt update magnitudes online.

(sec-the-entropy-regularized-objective-functional)=
### 9.11 The Entropy-Regularized Objective Functional

We use a regularized objective (in nats) that trades off task cost, representation complexity, and control effort.

Define an instantaneous regularized objective:

$$
F_t
:=
V(Z_t)
+ \beta_K\big(-\log p_\psi(K_t)\big)
+ \beta_n D_{\mathrm{KL}}\!\left(q(z_{n,t}\mid x_t)\ \Vert\ p(z_n)\right)
+ \beta_{\mathrm{tex}} D_{\mathrm{KL}}\!\left(q(z_{\mathrm{tex},t}\mid x_t)\ \Vert\ p(z_{\mathrm{tex}})\right)
+ T_c D_{\mathrm{KL}}\!\left(\pi(\cdot\mid K_t)\ \Vert\ \pi_0(\cdot\mid K_t)\right),
$$
where {math}`Z_t=(K_t,z_{n,t},z_{\mathrm{tex},t})` and all terms are measured in nats.

A practical monotonicity surrogate is:

$$
\mathcal{L}_{\downarrow F}
:=
\mathbb{E}\!\left[\mathrm{ReLU}\!\left(F_{t+1}-F_t\right)^2\right].
$$
**Interpretation.**
- $V(Z_t)$: task-aligned cost-to-go (critic).
- $\beta_K(-\log p_\psi(K_t))$: macro codelength penalty (MDL / rate).
- $\beta_n D_{\mathrm{KL}}(q(z_{n,t}\mid x_t)\Vert p(z_n))$: nuisance regularizer (structured residual; not “trash”).
- $\beta_{\mathrm{tex}} D_{\mathrm{KL}}(q(z_{\mathrm{tex},t}\mid x_t)\Vert p(z_{\mathrm{tex}}))$: texture-as-residual (likelihood/reconstruction-only).
- $T_c D_{\mathrm{KL}}(\pi\Vert\pi_0)$: KL-regularized control effort (deviation from prior/constraints).

This is the standard structure behind information bottlenecks/MDL (representation) and KL-regularized control / MaxEnt RL (policy), stated without additional metaphors.

(sec-atlas-manifold-dictionary-from-topology-to-neural-networks)=
### 9.12 Atlas-Manifold Dictionary: From Topology to Neural Networks

This section provides a translation dictionary connecting **manifold theory** to the **neural network implementations** described in Sections 7.7–7.8 (Attentive Atlas routing assumed unless noted).

(sec-core-correspondences)=
#### Core Correspondences

| Manifold Theory                     | Neural Implementation                                          | Role                     | Section Reference |
|-------------------------------------|----------------------------------------------------------------|--------------------------|-------------------|
| **Manifold $M$**                    | Input data distribution                                        | The space to be embedded | —                 |
| **Chart $(U_i, \phi_i)$**           | Local VQ codebook $i$                                          | Local embedding function | 7.8.3             |
| **Atlas $\mathcal{A} = \{U_i\}$**   | Attentive router + chart codebooks                             | Global coverage          | 7.8.3             |
| **Transition function $\tau_{ij}$** | Attention-weighted blending                                    | Chart overlap handling   | 7.8.3             |
| **Riemannian metric $g$**           | Orthogonality regularizer $\lVert W^TW - I\rVert^2$ (optional) | Distance preservation    | 7.7.2             |
| **Geodesic $\gamma(t)$**            | Latent space trajectory                                        | Optimal path             | 2.4               |
| **Curvature $R$**                   | Hessian of loss landscape                                      | Local complexity         | 2.5               |
| **Chart separation**                | Separation loss                                                | Chart partitioning       | 7.7.4             |

(sec-self-supervised-learning-correspondences)=
#### Self-Supervised Learning Correspondences

| SSL Concept                 | VICReg Term                | Geometric Interpretation | Failure Mode Prevented    |
|-----------------------------|----------------------------|--------------------------|---------------------------|
| **Augmentation invariance** | $\mathcal{L}_{\text{inv}}$ | Metric tensor stability  | Sensitivity to noise      |
| **Non-collapse**            | $\mathcal{L}_{\text{var}}$ | Non-degenerate metric    | Trivial constant solution |
| **Decorrelation**           | $\mathcal{L}_{\text{cov}}$ | Coordinate independence  | Redundant dimensions      |
| **Negative sampling**       | (Not needed in VICReg)     | Contrastive boundary     | —                         |

(sec-mixture-of-experts-correspondences)=
#### Mixture of Experts Correspondences

| MoE Concept {cite}`jacobs1991adaptive` | Atlas Concept         | Implementation                                  |
|----------------------------------------|-----------------------|-------------------------------------------------|
| **Gating network**                     | Chart selector        | Cross-attention over chart queries              |
| **Expert networks**                    | Local charts $\phi_i$ | Chart-specific VQ codebooks                     |
| **Expert specialization**              | Chart coverage $U_i$  | Learned via separation loss                     |
| **Load balancing**                     | Atlas completeness    | Balance loss $\lVert\text{usage} - 1/K\rVert^2$ |
| **Expert capacity**                    | Chart dimension       | Latent dimension $d$                            |

(sec-loss-function-decomposition)=
#### Loss Function Decomposition

The **Universal Loss** (Section 7.7.4) decomposes into geometric objectives, using attentive router weights $w_i(x)$ from Section 7.8.1:

| Loss Component                 | Geometric Objective | Manifold Property Enforced         |
|--------------------------------|---------------------|------------------------------------|
| $\mathcal{L}_{\text{inv}}$     | Metric stability    | Local isometry                     |
| $\mathcal{L}_{\text{var}}$     | Non-degeneracy      | Full rank Jacobian                 |
| $\mathcal{L}_{\text{cov}}$     | Orthonormality      | Riemannian normal coordinates      |
| $\mathcal{L}_{\text{entropy}}$ | Sharp boundaries    | Distinct chart domains             |
| $\mathcal{L}_{\text{balance}}$ | Complete coverage   | Atlas covers all of $M$            |
| $\mathcal{L}_{\text{sep}}$     | Disjoint interiors  | $U_i \cap U_j$ minimal             |
| $\mathcal{L}_{\text{orth}}$    | Isometric embedding | $\lVert Wx\rVert = \lVert x\rVert$ |

(sec-when-to-use-atlas-architecture)=
#### When to Use Atlas Architecture

| Data Topology            | Single Chart | Atlas Required | Why                   |
|--------------------------|--------------|----------------|-----------------------|
| Euclidean $\mathbb{R}^n$ | ✓            | —              | Trivially covered     |
| Sphere $S^2$             | ✗            | ≥2 charts      | Hairy Ball Theorem    |
| Torus $T^2$              | ✗            | ≥4 charts      | Non-trivial $H_1$     |
| Swiss Roll               | ✓*           | —              | Topologically trivial |
| Disconnected components  | ✗            | ≥k charts      | k components          |
| Mixed topology           | ✗            | Adaptive       | Data-dependent        |

*Swiss Roll is topologically trivial but may benefit from multiple charts for geometric reasons (unrolling).

(sec-key-citations)=
#### Key Citations

| Concept                  | Citation                          | Contribution                                 |
|--------------------------|-----------------------------------|----------------------------------------------|
| **Manifold Atlas**       | {cite}`lee2012smooth`             | *Smooth Manifolds* textbook                  |
| **Embedding Theorem**    | {cite}`whitney1936differentiable` | Any $n$-manifold embeds in $\mathbb{R}^{2n}$ |
| **Mixture of Experts**   | {cite}`jacobs1991adaptive`        | Gated expert networks                        |
| **VICReg**               | {cite}`bardes2022vicreg`          | Collapse prevention without negatives        |
| **Barlow Twins**         | {cite}`zbontar2021barlow`         | Redundancy reduction                         |
| **InfoNCE**              | {cite}`oord2018cpc`               | Contrastive predictive coding                |
| **Information Geometry** | {cite}`saxe2019information`       | Fisher information in NNs                    |



(sec-intrinsic-motivation-maximum-entropy-exploration)=
