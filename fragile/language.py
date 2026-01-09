"""
Chapter 37: The Inter-Subjective Metric — PyTorch Implementation
================================================================

This module provides PyTorch implementations of all equations from Chapter 37
of the Fragile monograph. Each function/class is documented with its
corresponding mathematical definition.

References:
    - Section 37.1: Metric Friction (Definition 37.1, Lemma 37.1)
    - Section 37.2: Locking Operator (Definitions 37.2-37.3, Theorem 37.1)
    - Section 37.3: Spontaneous Gauge Locking (Definition 37.4, Theorem 37.2)
    - Section 37.4: Language as Gauge Transport (Definitions 37.5-37.8, Theorem 37.3)
    - Section 37.5: Babel Limit (Theorem 37.4, Corollary 37.2)
    - Section 37.6: Spectral Analysis (Definition 37.9, Theorem 37.5)
    - Section 37.7: Emergence of Objective Reality (Theorem 37.6, Corollary 37.3)
    - Section 37.8: Multi-Agent Scaling (Definition 37.10)
    - Section 37.9: Physics Isomorphisms (Kuramoto Model)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Callable
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class LockingConfig:
    """Configuration for the Inter-Subjective Metric system.

    Parameters correspond to the theoretical constants defined in Chapter 37.
    """
    # Dimensions
    latent_dim: int = 64          # D: dimension of latent manifold Z
    gauge_dim: int = 8            # dim(g): dimension of gauge algebra
    spacetime_dim: int = 4        # D in d^D z integration

    # Coupling constants
    g_lock: float = 1.0           # g_lock: inter-agent coupling constant
    lambda_lock: float = 1.0      # λ_lock: locking strength
    beta: float = 1.0             # β: interaction coupling strength

    # Capacity parameters
    sigma: float = 0.1            # σ: cognitive action scale (Def 29.21)
    ell_L: float = 1.0            # ℓ_L: Planck-like length scale
    nu_D: float = 0.25            # ν_D: area law coefficient

    # Friction parameters
    friction_scale: float = 1.0   # F_0: characteristic friction scale
    friction_thresh: float = 0.1  # F_thresh: friction threshold for diagnostics

    # Training parameters
    use_procrustes: bool = True   # Use efficient Procrustes alignment


# =============================================================================
# Section 37.1: Metric Friction
# =============================================================================

def metric_friction(
    G_A: Tensor,
    G_B: Tensor,
    phi: Optional[Callable[[Tensor], Tensor]] = None
) -> Tensor:
    """
    Definition 37.1: Metric Friction

    Computes the squared Frobenius norm of the pullback metric distortion:

        F_AB(z) := || G_A(z) - φ*G_B(φ(z)) ||_F^2

    where φ*G_B denotes the pullback metric.

    Args:
        G_A: [B, D, D] or [D, D] - Metric tensor of Agent A
        G_B: [B, D, D] or [D, D] - Metric tensor of Agent B
        phi: Optional map φ: Z_A → Z_B. If None, assumes identity.

    Returns:
        F_AB: [B] or scalar - Metric friction

    Mathematical form:
        F_AB = ||G_A - φ*G_B||_F^2 = Tr[(G_A - φ*G_B)^T (G_A - φ*G_B)]
    """
    if phi is not None:
        # Apply pullback: φ*G_B = J^T G_B J where J is Jacobian of φ
        # For simplicity, assume φ is identity or handled externally
        pass

    # Compute Frobenius norm squared
    diff = G_A - G_B

    if diff.dim() == 2:
        # Single metric tensors
        return torch.sum(diff ** 2)
    else:
        # Batch of metric tensors [B, D, D]
        return torch.sum(diff ** 2, dim=(-2, -1))


def metric_friction_from_embeddings(
    z_A: Tensor,
    z_B: Tensor,
    use_procrustes: bool = True
) -> Tensor:
    """
    Definition 37.1 (Computational Form): Metric Friction from Embeddings

    Computes metric friction using distance matrix correlation as a
    Gromov-Hausdorff proxy:

        F_AB ≈ ||D_A - D_B||_F^2 / (||D_A||_F ||D_B||_F)

    where D_i[j,k] = ||z_i[j] - z_i[k]||.

    Args:
        z_A: [B, D] - Batch of states from Agent A
        z_B: [B, D] - Batch of states from Agent B
        use_procrustes: If True, use O(D³) Procrustes; else O(B²) GW

    Returns:
        friction: scalar - Metric friction loss
    """
    if use_procrustes:
        # Procrustes alignment: min_R ||z_A - z_B @ R||_F^2 s.t. R^T R = I
        # Solution: R = V @ U^T where z_A^T @ z_B = U @ S @ V^T
        U, S, Vt = torch.linalg.svd(z_A.T @ z_B)
        R = U @ Vt
        z_B_aligned = z_B @ R
        friction = F.mse_loss(z_A, z_B_aligned)
    else:
        # Gromov-Wasserstein proxy via distance matrices
        # D_A[i,j] = ||z_A[i] - z_A[j]||
        dist_A = torch.cdist(z_A, z_A)
        dist_B = torch.cdist(z_B, z_B)

        # Normalize to scale-invariant
        dist_A = dist_A / (dist_A.mean() + 1e-8)
        dist_B = dist_B / (dist_B.mean() + 1e-8)

        friction = F.mse_loss(dist_A, dist_B)

    return friction


def friction_bounds_utility(
    V_max: Tensor,
    friction: Tensor,
    friction_scale: float = 1.0
) -> Tensor:
    """
    Lemma 37.1: Metric Friction Bounds Cooperative Utility

        V_coop ≤ V_max · exp(-F_AB / F_0)

    Args:
        V_max: Maximum cooperative value under perfect alignment
        friction: F_AB metric friction
        friction_scale: F_0 characteristic friction scale

    Returns:
        V_coop_bound: Upper bound on cooperative value
    """
    return V_max * torch.exp(-friction / friction_scale)


# =============================================================================
# Section 37.2: The Locking Operator
# =============================================================================

class GaugeConnection(nn.Module):
    """
    Strategic Connection A_μ from Definition 29.14.

    Represents a gauge connection on the latent manifold Z.
    The connection is parameterized as a learnable field.
    """

    def __init__(self, latent_dim: int, gauge_dim: int, spacetime_dim: int = 4):
        """
        Args:
            latent_dim: D - dimension of latent space
            gauge_dim: dim(g) - dimension of gauge algebra
            spacetime_dim: number of spacetime indices μ
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.gauge_dim = gauge_dim
        self.spacetime_dim = spacetime_dim

        # A_μ^a: connection components [spacetime_dim, gauge_dim, latent_dim]
        # Parameterized via neural network for flexibility
        self.connection_net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.Tanh(),
            nn.Linear(latent_dim * 2, spacetime_dim * gauge_dim)
        )

    def forward(self, z: Tensor) -> Tensor:
        """
        Compute connection A_μ^a(z).

        Args:
            z: [B, D] - Points in latent space

        Returns:
            A: [B, spacetime_dim, gauge_dim] - Connection at each point
        """
        B = z.shape[0]
        A_flat = self.connection_net(z)
        return A_flat.view(B, self.spacetime_dim, self.gauge_dim)


def inter_agent_connection(
    A_mu_A: Tensor,
    A_mu_B: Tensor,
    lambda_lock: float,
    C_AB_mu: Optional[Tensor] = None
) -> Tensor:
    """
    Definition 37.2: The Inter-Agent Connection

        A_AB^μ(z_A, z_B) := A_μ^(A)(z_A) ⊗ 1_B + 1_A ⊗ A_μ^(B)(z_B) + λ_lock C_AB^μ

    Args:
        A_mu_A: [B, spacetime_dim, gauge_dim] - Agent A's connection
        A_mu_B: [B, spacetime_dim, gauge_dim] - Agent B's connection
        lambda_lock: λ_lock - locking strength
        C_AB_mu: [B, spacetime_dim, gauge_dim] - Coupling connection (optional)

    Returns:
        A_AB: [B, spacetime_dim, gauge_dim, 2] - Inter-agent connection
              Last dim: [A contribution, B contribution]
    """
    B, S, G = A_mu_A.shape

    # A_μ^(A) ⊗ 1_B: Agent A's connection extended to product space
    A_term = A_mu_A.unsqueeze(-1)  # [B, S, G, 1]

    # 1_A ⊗ A_μ^(B): Agent B's connection extended to product space
    B_term = A_mu_B.unsqueeze(-1)  # [B, S, G, 1]

    # Combine: simple sum for joint connection
    A_AB = A_term + B_term  # [B, S, G, 1]

    # Add coupling term if provided
    if C_AB_mu is not None:
        coupling = lambda_lock * C_AB_mu.unsqueeze(-1)
        A_AB = A_AB + coupling

    return A_AB.squeeze(-1)


def locking_curvature(
    A_AB: Tensor,
    g_lock: float = 1.0,
    eps: float = 1e-6
) -> Tensor:
    """
    Definition 37.3: The Locking Curvature

        F_AB^μν := ∂^μ A_AB^ν - ∂^ν A_AB^μ - i g_lock [A_AB^μ, A_AB^ν]

    For computational purposes, we compute this via finite differences
    and the Lie bracket (commutator) of the connection.

    Args:
        A_AB: [B, spacetime_dim, gauge_dim] - Inter-agent connection
        g_lock: Coupling constant
        eps: Finite difference step size

    Returns:
        F_AB: [B, spacetime_dim, spacetime_dim, gauge_dim] - Curvature tensor
    """
    B, S, G = A_AB.shape

    # Initialize curvature tensor
    F_AB = torch.zeros(B, S, S, G, device=A_AB.device, dtype=A_AB.dtype)

    # For a discrete approximation, compute [A^μ, A^ν] via commutator
    # In gauge theory: [A^μ, A^ν]^a = f^abc A^μ_b A^ν_c
    # Here we use a simplified scalar commutator proxy

    for mu in range(S):
        for nu in range(S):
            if mu != nu:
                # Antisymmetric: F^μν = -F^νμ
                # Abelian approximation: F^μν ≈ A^μ - A^ν (simplified)
                # Non-Abelian: includes commutator term
                commutator = A_AB[:, mu, :] * A_AB[:, nu, :] - A_AB[:, nu, :] * A_AB[:, mu, :]
                F_AB[:, mu, nu, :] = -g_lock * commutator

    return F_AB


def integrated_friction(
    F_AB: Tensor,
    G_shared: Optional[Tensor] = None
) -> Tensor:
    """
    Definition 37.3 (cont.): Integrated Friction (gauge-invariant scalar)

        Ψ_sync := ∫ Tr(F_AB^μν F_AB,μν) √|G_shared| d^D z

    Args:
        F_AB: [B, S, S, G] - Locking curvature tensor
        G_shared: [B, D, D] - Shared metric tensor (optional, for volume element)

    Returns:
        Psi_sync: [B] - Integrated synchronization potential
    """
    # Tr(F^μν F_μν) = sum over μ,ν,a of (F^μν_a)^2
    # Contract: F^μν F_μν (sum over repeated indices)
    F_squared = torch.einsum('bmna,bmna->b', F_AB, F_AB)

    # Volume element √|G|
    if G_shared is not None:
        # det(G) and sqrt
        det_G = torch.linalg.det(G_shared)
        sqrt_det_G = torch.sqrt(torch.abs(det_G) + 1e-8)
        F_squared = F_squared * sqrt_det_G

    return F_squared


def locking_operator(
    F_AB: Tensor,
    g_lock: float = 1.0,
    G_shared: Optional[Tensor] = None
) -> Tensor:
    """
    Theorem 37.1: The Locking Operator (Yang-Mills Energy)

        L_sync(G_A, G_B) := -1/(4 g_lock²) ∫ Tr(F_AB^μν F_AB,μν) √|G_AB| d^D z

    Args:
        F_AB: [B, S, S, G] - Locking curvature tensor
        g_lock: Inter-agent coupling constant
        G_shared: [B, D, D] - Shared metric tensor (optional)

    Returns:
        L_sync: [B] - Locking operator value (energy)
    """
    Psi_sync = integrated_friction(F_AB, G_shared)

    # Yang-Mills normalization
    normalization = -1.0 / (4.0 * g_lock ** 2)

    return normalization * Psi_sync


class FiniteCommunicationBandwidth:
    """
    Axiom 37.1: Finite Communication Bandwidth

        C_L ≤ ν_D · Area(∂L) / ℓ_L^(D-1)

    Implements the Causal Information Bound (Theorem 33.3) for language channels.
    """

    def __init__(self, config: LockingConfig):
        self.nu_D = config.nu_D
        self.ell_L = config.ell_L
        self.D = config.spacetime_dim

    def channel_capacity(self, boundary_area: Tensor) -> Tensor:
        """
        Compute Shannon capacity of the language channel.

        Args:
            boundary_area: Area of the channel boundary ∂L

        Returns:
            C_L: Channel capacity in nats
        """
        return self.nu_D * boundary_area / (self.ell_L ** (self.D - 1))


# =============================================================================
# Section 37.3: Spontaneous Gauge Locking
# =============================================================================

def gauge_alignment_order_parameter(
    U_A: Tensor,
    U_B: Tensor
) -> Tensor:
    """
    Definition 37.4: Gauge Alignment Order Parameter

        φ_AB(z) := Tr(U_A(z) U_B†(z)) ∈ ℂ

    where U_A, U_B ∈ G_Fragile are local gauge transformations.

    Args:
        U_A: [B, G, G] - Agent A's gauge transformation (unitary matrix)
        U_B: [B, G, G] - Agent B's gauge transformation (unitary matrix)

    Returns:
        phi_AB: [B] (complex) - Order parameter
    """
    # U_B† = conjugate transpose
    U_B_dag = U_B.conj().transpose(-2, -1)

    # Tr(U_A @ U_B†)
    product = torch.bmm(U_A, U_B_dag)
    phi_AB = torch.diagonal(product, dim1=-2, dim2=-1).sum(dim=-1)

    return phi_AB


def locking_potential(
    phi_AB: Tensor,
    mu_lock_sq: float,
    lambda_lock: float
) -> Tensor:
    """
    Definition 37.4 (cont.): Locking Potential

        V_lock(φ_AB) = -μ_lock² |φ_AB|² + λ_lock |φ_AB|⁴

    This is a Mexican hat potential that drives spontaneous symmetry breaking.

    Args:
        phi_AB: [B] (complex) - Order parameter
        mu_lock_sq: μ_lock² = β - β_c (effective mass parameter)
        lambda_lock: λ_lock (quartic coupling)

    Returns:
        V: [B] - Potential energy
    """
    phi_sq = torch.abs(phi_AB) ** 2

    return -mu_lock_sq * phi_sq + lambda_lock * phi_sq ** 2


def joint_prediction_loss(
    x_pred_A: Tensor,
    x_pred_B: Tensor,
    x_true: Tensor,
    Psi_sync: Tensor,
    beta: float
) -> Tensor:
    """
    Theorem 37.2: Joint Prediction Error (for Spontaneous Gauge Locking)

        L_joint = ||x̂_{t+1}^A - x_{t+1}||² + ||x̂_{t+1}^B - x_{t+1}||² + β Ψ_sync

    Minimizing this loss drives gauge locking as β → ∞.

    Args:
        x_pred_A: [B, ...] - Agent A's prediction
        x_pred_B: [B, ...] - Agent B's prediction
        x_true: [B, ...] - Ground truth
        Psi_sync: [B] - Synchronization potential
        beta: β - Interaction coupling strength

    Returns:
        L_joint: scalar - Joint loss
    """
    # Prediction errors
    loss_A = F.mse_loss(x_pred_A, x_true, reduction='none')
    loss_B = F.mse_loss(x_pred_B, x_true, reduction='none')

    # Sum over all dimensions except batch
    while loss_A.dim() > 1:
        loss_A = loss_A.sum(dim=-1)
        loss_B = loss_B.sum(dim=-1)

    # Total loss
    L_joint = loss_A + loss_B + beta * Psi_sync

    return L_joint.mean()


def critical_coupling(
    sigma: float,
    volume: Tensor,
    g_lock: float
) -> Tensor:
    """
    Corollary 37.1: Critical Coupling for Locking

        β_c = σ² Vol(Z_shared) / (2 g_lock²)

    Args:
        sigma: σ - Cognitive action scale
        volume: Vol(Z_shared) - Volume of shared manifold
        g_lock: g_lock - Inter-agent coupling constant

    Returns:
        beta_c: Critical coupling strength
    """
    return (sigma ** 2) * volume / (2.0 * g_lock ** 2)


def relative_gauge(U_A: Tensor, U_B: Tensor) -> Tensor:
    """
    Step 3 of Theorem 37.2: Relative Gauge Transformation

        ΔU(z) := U_A(z) U_B^{-1}(z)

    Args:
        U_A: [B, G, G] - Agent A's gauge transformation
        U_B: [B, G, G] - Agent B's gauge transformation

    Returns:
        Delta_U: [B, G, G] - Relative gauge
    """
    U_B_inv = torch.linalg.inv(U_B)
    return torch.bmm(U_A, U_B_inv)


def phase_transition_order_parameter(
    beta: float,
    beta_c: float,
    lambda_lock: float
) -> float:
    """
    Step 11 of Theorem 37.2: Phase Transition Order Parameter

        ⟨|φ_AB|⟩ = { 0                                  if β < β_c
                   { √((β - β_c) / λ_lock)             if β > β_c

    Args:
        beta: Current coupling strength
        beta_c: Critical coupling
        lambda_lock: Quartic coupling

    Returns:
        expectation: Expected value of order parameter magnitude
    """
    if beta < beta_c:
        return 0.0
    else:
        return math.sqrt((beta - beta_c) / lambda_lock)


# =============================================================================
# Section 37.4: Language as Gauge-Covariant Transport
# =============================================================================

class LieAlgebraMessage(nn.Module):
    """
    Definition 37.5: Message as Lie Algebra Element

        m_{A→B} ∈ g = Lie(G_Fragile), m = m^a T_a

    where {T_a} are generators satisfying [T_a, T_b] = i f^{abc} T_c.
    """

    def __init__(self, gauge_dim: int):
        """
        Args:
            gauge_dim: dim(g) - Dimension of Lie algebra
        """
        super().__init__()
        self.gauge_dim = gauge_dim

        # Generators T_a (for SU(n), these are generalized Gell-Mann matrices)
        # Here we use a simplified real representation
        self.register_buffer(
            'generators',
            self._init_generators(gauge_dim)
        )

    def _init_generators(self, dim: int) -> Tensor:
        """Initialize Lie algebra generators."""
        # For simplicity, use antisymmetric matrices as generators
        # Full SU(n) would use Gell-Mann matrices
        generators = []
        idx = 0
        n = int(math.ceil(math.sqrt(dim)))

        for i in range(n):
            for j in range(i + 1, n):
                if idx >= dim:
                    break
                T = torch.zeros(n, n)
                T[i, j] = 1.0
                T[j, i] = -1.0
                generators.append(T)
                idx += 1

        # Pad if needed
        while len(generators) < dim:
            generators.append(torch.zeros(n, n))

        return torch.stack(generators[:dim])  # [gauge_dim, n, n]

    def encode(self, coefficients: Tensor) -> Tensor:
        """
        Encode message coefficients m^a into Lie algebra element.

            m = m^a T_a

        Args:
            coefficients: [B, gauge_dim] - Coefficients m^a

        Returns:
            m: [B, n, n] - Lie algebra element (matrix)
        """
        # m = sum_a m^a T_a
        # Einstein summation: m_ij = m^a T_a_ij
        return torch.einsum('ba,aij->bij', coefficients, self.generators)

    def decode(self, m: Tensor) -> Tensor:
        """
        Decode Lie algebra element back to coefficients.

            m^a = Tr(T_a m) / Tr(T_a T_a)

        Args:
            m: [B, n, n] - Lie algebra element

        Returns:
            coefficients: [B, gauge_dim] - Coefficients m^a
        """
        # Project onto each generator
        # m^a ∝ Tr(T_a @ m)
        return torch.einsum('aij,bij->ba', self.generators, m)


class LanguageChannel(nn.Module):
    """
    Definition 37.6: The Language Channel

        L: g → g_L ⊂ g

    where dim(g_L) << dim(g). The channel satisfies the bandwidth
    constraint of Axiom 37.1.
    """

    def __init__(self, gauge_dim: int, channel_dim: int):
        """
        Args:
            gauge_dim: dim(g) - Full gauge algebra dimension
            channel_dim: dim(g_L) - Channel (compressed) dimension
        """
        super().__init__()
        self.gauge_dim = gauge_dim
        self.channel_dim = channel_dim

        # Projection L: g → g_L
        self.encoder = nn.Linear(gauge_dim, channel_dim)

        # Lifting g_L → g
        self.decoder = nn.Linear(channel_dim, gauge_dim)

    def project(self, m: Tensor) -> Tensor:
        """
        Project full message to language channel.

        Args:
            m: [B, gauge_dim] - Full message coefficients

        Returns:
            m_L: [B, channel_dim] - Projected message
        """
        return self.encoder(m)

    def lift(self, m_L: Tensor) -> Tensor:
        """
        Lift channel message back to full algebra.

        Args:
            m_L: [B, channel_dim] - Channel message

        Returns:
            m: [B, gauge_dim] - Lifted message (approximate)
        """
        return self.decoder(m_L)

    def forward(self, m: Tensor) -> Tensor:
        """
        Full channel: project then lift (lossy compression).

        Args:
            m: [B, gauge_dim] - Input message

        Returns:
            m_reconstructed: [B, gauge_dim] - Reconstructed message
        """
        return self.lift(self.project(m))


def translation_operator(
    m: Tensor,
    A_mu: Tensor,
    path: Tensor,
    g: float = 1.0
) -> Tensor:
    """
    Definition 37.7: Gauge-Covariant Translation Operator

        T_{A→B}(m) := exp(-ig ∫_{γ_AB} m^a A_μ^a dz^μ) · P exp(-ig ∫_{γ_AB} A_μ dz^μ)

    where:
        - First factor encodes message content
        - Second factor is Wilson line (parallel transport)
        - P denotes path-ordering

    Args:
        m: [B, gauge_dim] - Message coefficients
        A_mu: [B, path_len, spacetime_dim, gauge_dim] - Connection along path
        path: [B, path_len, spacetime_dim] - Path coordinates
        g: Coupling constant

    Returns:
        T: [B, gauge_dim, gauge_dim] - Translation operator (matrix)
    """
    B, L, S, G = A_mu.shape

    # Compute path differentials dz^μ
    dz = torch.diff(path, dim=1)  # [B, L-1, S]

    # Contract: m^a A_μ^a dz^μ along path
    # First term: message-weighted integral
    A_mu_mid = (A_mu[:, :-1] + A_mu[:, 1:]) / 2  # Midpoint rule

    # m^a A_μ^a
    m_A = torch.einsum('bg,blsg->bls', m, A_mu_mid)  # [B, L-1, S]

    # Integrate: sum over path segments
    integral_m = torch.einsum('bls,bls->b', m_A, dz)  # [B]

    # Wilson line: P exp(-ig ∫ A_μ dz^μ)
    # For simplicity, use product of exponentials (valid for small segments)
    wilson_phases = -g * torch.einsum('blsg,bls->blg', A_mu_mid, dz)  # [B, L-1, G]

    # Exponentiate (simplified: diagonal approximation)
    # Full implementation would use path-ordered product
    total_phase = wilson_phases.sum(dim=1)  # [B, G]

    # Message phase
    message_phase = -g * integral_m  # [B]

    # Combined translation (simplified as phase factors)
    # Full: matrix exponential of Lie algebra element
    T_diag = torch.exp(1j * (total_phase + message_phase.unsqueeze(-1)))

    # Return as diagonal matrix
    T = torch.diag_embed(T_diag.real)  # [B, G, G]

    return T


def semantic_alignment(
    friction_before: Tensor,
    friction_after: Tensor
) -> Tensor:
    """
    Definition 37.8: Semantic Alignment

        Understanding(m) ⟺ F_AB(z; t+Δt) < F_AB(z; t)

    Args:
        friction_before: F_AB(t) - Friction before message
        friction_after: F_AB(t+Δt) - Friction after message

    Returns:
        understanding: [B] bool - Whether understanding occurred
    """
    return friction_after < friction_before


def untranslatability_bound(
    m_norm: Tensor,
    F_AB: Tensor,
    surface_area: Tensor
) -> Tensor:
    """
    Theorem 37.3: The Untranslatability Bound

        U_AB(m) ≤ ||m|| · ∮_{∂Σ} ||F_AB||_F dA

    Args:
        m_norm: ||m|| - Norm of message
        F_AB: [B, S, S, G] - Locking curvature
        surface_area: Area of boundary surface

    Returns:
        U_bound: [B] - Upper bound on untranslatability
    """
    # ||F_AB||_F = sqrt(sum of squares)
    F_norm = torch.sqrt(torch.sum(F_AB ** 2, dim=(-3, -2, -1)))

    # Surface integral (simplified: F_norm * area)
    integral = F_norm * surface_area

    return m_norm * integral


def holonomy(
    A_mu: Tensor,
    path: Tensor,
    g: float = 1.0
) -> Tensor:
    """
    Step 1 of Theorem 37.3: Holonomy

        H_γ = P exp(-ig ∮_γ A_μ dz^μ)

    The holonomy measures the "memory" of parallel transport around a closed loop.

    Args:
        A_mu: [B, path_len, spacetime_dim, gauge_dim] - Connection along closed path
        path: [B, path_len, spacetime_dim] - Closed path coordinates
        g: Coupling constant

    Returns:
        H: [B, gauge_dim, gauge_dim] - Holonomy matrix
    """
    B, L, S, G = A_mu.shape

    # Initialize holonomy as identity
    H = torch.eye(G, device=A_mu.device, dtype=A_mu.dtype).unsqueeze(0).expand(B, -1, -1)
    H = H.clone()

    # Path differentials
    dz = torch.diff(path, dim=1)  # [B, L-1, S]

    # Product of exponentials along path (path-ordered)
    for i in range(L - 1):
        A_segment = A_mu[:, i]  # [B, S, G]
        dz_segment = dz[:, i]   # [B, S]

        # Phase: A_μ dz^μ (contract over spacetime)
        phase = torch.einsum('bsg,bs->bg', A_segment, dz_segment)  # [B, G]

        # Exponentiate (diagonal approximation)
        exp_phase = torch.exp(-1j * g * phase)

        # Multiply into holonomy (simplified: element-wise for diagonal)
        H = H * exp_phase.unsqueeze(-1).real

    return H


# =============================================================================
# Section 37.5: The Babel Limit
# =============================================================================

def babel_limit_satisfied(
    gauge_dim: int,
    metric_entropy: Tensor,
    channel_capacity: Tensor
) -> Tuple[Tensor, Tensor]:
    """
    Theorem 37.4: The Babel Limit

        dim(g) · H(G_A) ≤ C_L

    Complete gauge locking is achievable only if this condition is satisfied.

    Args:
        gauge_dim: dim(g) - Dimension of gauge algebra
        metric_entropy: H(G_A) - Differential entropy of metric tensor
        channel_capacity: C_L - Channel capacity

    Returns:
        satisfied: bool - Whether Babel limit allows complete locking
        I_required: Information required for complete locking
    """
    I_required = gauge_dim * metric_entropy
    satisfied = I_required <= channel_capacity

    return satisfied, I_required


def unlocked_dimension(
    gauge_dim: int,
    metric_entropy: Tensor,
    channel_capacity: Tensor
) -> Tensor:
    """
    Theorem 37.4 (cont.): Unlocked Subspace Dimension

        d_unlocked = dim(g) - ⌊C_L / H(G_A)⌋

    Args:
        gauge_dim: dim(g)
        metric_entropy: H(G_A)
        channel_capacity: C_L

    Returns:
        d_unlocked: Dimension of unlocked (private) subspace
    """
    lockable = torch.floor(channel_capacity / (metric_entropy + 1e-8))
    d_unlocked = gauge_dim - lockable

    return torch.clamp(d_unlocked, min=0)


def private_qualia_dimension(
    gauge_dim: int,
    metric_entropy: Tensor,
    channel_capacity: Tensor
) -> Tensor:
    """
    Corollary 37.2: The Ineffability Theorem

        dim(q) = dim(g) - ⌊C_L / H(G_A)⌋ > 0

    This subspace q ⊂ g corresponds to Private Qualia: aspects of experience
    that cannot be communicated regardless of symbol system.

    Args:
        gauge_dim: dim(g)
        metric_entropy: H(G_A)
        channel_capacity: C_L

    Returns:
        dim_q: Dimension of private qualia subspace
    """
    return unlocked_dimension(gauge_dim, metric_entropy, channel_capacity)


# =============================================================================
# Section 37.6: Spectral Analysis
# =============================================================================

def metric_eigendecomposition(G: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Definition 37.9: Metric Eigendecomposition

        G_A = Σ_{k=1}^D σ_k^(A) v_k^(A) ⊗ v_k^(A)

    where σ_1 ≥ σ_2 ≥ ... ≥ σ_D > 0 are eigenvalues (principal curvatures).

    Args:
        G: [B, D, D] or [D, D] - Metric tensor

    Returns:
        eigenvalues: [B, D] or [D] - Sorted eigenvalues (descending)
        eigenvectors: [B, D, D] or [D, D] - Corresponding eigenvectors
    """
    # Symmetric eigendecomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(G)

    # Sort descending
    idx = torch.argsort(eigenvalues, dim=-1, descending=True)

    if G.dim() == 2:
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
    else:
        # Batch case
        B, D = eigenvalues.shape
        batch_idx = torch.arange(B, device=G.device).unsqueeze(-1).expand(-1, D)
        eigenvalues = eigenvalues[batch_idx, idx]
        eigenvectors = eigenvectors[batch_idx, :, idx].transpose(-2, -1)
        eigenvectors = eigenvectors.transpose(-2, -1)

    return eigenvalues, eigenvectors


def spectral_locking_order(
    eigenvalues: Tensor,
    channel_capacity: Tensor,
    time: float
) -> int:
    """
    Theorem 37.5: Spectral Locking Order

        k_max = max{k : Σ_{j=1}^k H(σ_j v_j) ≤ C_L · T}

    Locking proceeds from high-curvature (salient) to low-curvature (subtle) features.

    Args:
        eigenvalues: [D] - Sorted eigenvalues (descending)
        channel_capacity: C_L - Channel capacity
        time: T - Communication time

    Returns:
        k_max: Maximum number of lockable components
    """
    D = eigenvalues.shape[-1]
    total_capacity = channel_capacity * time

    # Entropy per component: H(σ_k v_k) ≈ log(σ_k) (simplified)
    entropies = torch.log(eigenvalues + 1e-8)

    # Cumulative entropy
    cumsum = torch.cumsum(entropies, dim=-1)

    # Find k_max
    k_max = torch.sum(cumsum <= total_capacity).item()

    return min(k_max, D)


def core_vs_nuance_split(
    eigenvalues: Tensor,
    threshold: Optional[float] = None
) -> Tuple[Tensor, Tensor, int]:
    """
    Definition 37.9 (cont.): Core Concepts vs Nuance

    - Core Concepts: σ_k > σ_thresh (high information density)
    - Nuance: σ_k ≤ σ_thresh (low information density)

    Args:
        eigenvalues: [D] - Sorted eigenvalues
        threshold: σ_thresh (if None, uses median)

    Returns:
        core_eigenvalues: Eigenvalues above threshold
        nuance_eigenvalues: Eigenvalues below threshold
        split_idx: Index separating core from nuance
    """
    if threshold is None:
        threshold = torch.median(eigenvalues)

    core_mask = eigenvalues > threshold
    split_idx = torch.sum(core_mask).item()

    return eigenvalues[:split_idx], eigenvalues[split_idx:], split_idx


# =============================================================================
# Section 37.7: Emergence of Objective Reality
# =============================================================================

def objective_reality_quotient(
    z_A: Tensor,
    z_B: Tensor,
    G_A: Tensor,
    G_B: Tensor,
    friction_threshold: float = 1e-6
) -> Tuple[Tensor, Tensor]:
    """
    Theorem 37.6: Emergence of Objective Reality

        Z_shared := (Z_A ⊔ Z_B) / ~_isometry

    where ~_isometry identifies points with vanishing metric friction.

    Args:
        z_A: [B, D] - Points in Agent A's manifold
        z_B: [B, D] - Points in Agent B's manifold
        G_A: [B, D, D] - Agent A's metric
        G_B: [B, D, D] - Agent B's metric
        friction_threshold: Threshold for identifying isometric points

    Returns:
        z_shared: [B, D] - Points in shared manifold
        equivalence_mask: [B] - Which points are identified (friction < threshold)
    """
    # Compute pointwise friction
    friction = metric_friction(G_A, G_B)

    # Identify equivalent points
    equivalence_mask = friction < friction_threshold

    # For equivalent points, use average as representative
    z_shared = torch.where(
        equivalence_mask.unsqueeze(-1),
        (z_A + z_B) / 2,
        z_A  # For non-equivalent, keep A's representation
    )

    return z_shared, equivalence_mask


def echo_chamber_loss(
    friction_AB: Tensor,
    friction_AE: Tensor,
    friction_BE: Tensor,
    lambda_lock: float,
    lambda_ground: float
) -> Tensor:
    """
    Remark 37.1: Echo Chamber Effect (Corrected Loss)

        L_total = λ_lock F_AB + λ_ground (F_AE + F_BE)

    This prevents agents from drifting into shared hallucination by
    maintaining grounding to the environment.

    Args:
        friction_AB: F_AB - Inter-agent friction
        friction_AE: F_AE - Agent A to environment friction
        friction_BE: F_BE - Agent B to environment friction
        lambda_lock: Locking coefficient
        lambda_ground: Grounding coefficient

    Returns:
        L_total: Total loss with grounding
    """
    return lambda_lock * friction_AB + lambda_ground * (friction_AE + friction_BE)


def critical_mass_for_consensus(
    sigma: float,
    lambda_lock: float,
    avg_friction: Tensor
) -> Tensor:
    """
    Corollary 37.3: Critical Mass for Consensus

        N_c = σ² / (λ_lock · ⟨F_ij⟩)

    Args:
        sigma: Cognitive action scale
        lambda_lock: Locking coefficient
        avg_friction: Average pairwise friction ⟨F_ij⟩

    Returns:
        N_c: Critical number of agents for consensus emergence
    """
    return (sigma ** 2) / (lambda_lock * avg_friction + 1e-8)


# =============================================================================
# Section 37.8: Multi-Agent Scaling
# =============================================================================

def institutional_friction(
    G_agents: Tensor,
    G_inst: Tensor
) -> Tensor:
    """
    Definition 37.10: The Institutional Manifold

        F_{A,Inst} + F_{B,Inst}  replaces  F_AB

    Institution-mediated locking is O(N) instead of O(N²).

    Args:
        G_agents: [N, D, D] - Metric tensors for N agents
        G_inst: [D, D] - Institutional reference metric

    Returns:
        total_friction: Sum of agent-institution frictions
    """
    N = G_agents.shape[0]

    # Broadcast G_inst for batch computation
    G_inst_expanded = G_inst.unsqueeze(0).expand(N, -1, -1)

    # Compute friction for each agent to institution
    frictions = metric_friction(G_agents, G_inst_expanded)

    return frictions.sum()


def money_distance(
    z1: Tensor,
    z2: Tensor,
    price_fn: Callable[[Tensor], Tensor],
    n_steps: int = 100
) -> Tensor:
    """
    Remark 37.2: Money as Universal Metric

        d_money(z1, z2) = inf_γ ∫_γ Price(ż) dt

    Money quantifies the "cost distance" between states.

    Args:
        z1: [B, D] - Starting points
        z2: [B, D] - Ending points
        price_fn: Function mapping velocity to price
        n_steps: Number of discretization steps

    Returns:
        d_money: [B] - Money distance
    """
    # Linear interpolation as initial path
    t = torch.linspace(0, 1, n_steps, device=z1.device).unsqueeze(0).unsqueeze(-1)
    path = z1.unsqueeze(1) * (1 - t) + z2.unsqueeze(1) * t  # [B, n_steps, D]

    # Compute velocities
    velocities = torch.diff(path, dim=1) * n_steps  # [B, n_steps-1, D]

    # Compute prices along path
    prices = price_fn(velocities)  # [B, n_steps-1]

    # Integrate
    d_money = prices.sum(dim=-1) / n_steps

    return d_money


# =============================================================================
# Section 37.9: Physics Isomorphisms — Kuramoto Model
# =============================================================================

class KuramotoAgents(nn.Module):
    """
    Physics Isomorphism: Kuramoto Model

    Agent gauge parameters satisfy Kuramoto-like dynamics:

        dθ^(i)/dt = ω^(i) + β Σ_{j≠i} ∇_θ F_ij

    This models synchronization of coupled oscillators.
    """

    def __init__(self, n_agents: int, latent_dim: int, beta: float = 1.0):
        """
        Args:
            n_agents: N - Number of agents
            latent_dim: D - Dimension per agent
            beta: Coupling strength
        """
        super().__init__()
        self.n_agents = n_agents
        self.latent_dim = latent_dim
        self.beta = beta

        # Natural frequencies ω^(i) (private drift rates)
        self.omega = nn.Parameter(torch.randn(n_agents))

        # Agent phases θ^(i)
        self.theta = nn.Parameter(torch.randn(n_agents, latent_dim))

    def pairwise_friction_gradient(self) -> Tensor:
        """
        Compute ∇_θ F_ij for all pairs.

        Returns:
            grad_F: [N, N, D] - Gradient of friction w.r.t. θ
        """
        N, D = self.theta.shape

        # Compute pairwise differences
        theta_i = self.theta.unsqueeze(1)  # [N, 1, D]
        theta_j = self.theta.unsqueeze(0)  # [1, N, D]

        # Simplified friction gradient: sin(θ_j - θ_i)
        # Full version would use actual metric friction
        diff = theta_j - theta_i  # [N, N, D]
        grad_F = torch.sin(diff)  # [N, N, D]

        return grad_F

    def dynamics(self) -> Tensor:
        """
        Compute dθ/dt according to Kuramoto dynamics.

            dθ^(i)/dt = ω^(i) + β Σ_{j≠i} ∇_θ F_ij

        Returns:
            d_theta: [N, D] - Time derivative of phases
        """
        N = self.n_agents

        # Natural frequency term
        d_theta = self.omega.unsqueeze(-1).expand(-1, self.latent_dim).clone()

        # Coupling term: sum over j ≠ i
        grad_F = self.pairwise_friction_gradient()  # [N, N, D]

        # Zero out diagonal (j ≠ i)
        mask = ~torch.eye(N, dtype=torch.bool, device=self.theta.device)
        grad_F = grad_F * mask.unsqueeze(-1)

        # Sum over j
        coupling = self.beta * grad_F.sum(dim=1) / N  # [N, D]

        d_theta = d_theta + coupling

        return d_theta

    def step(self, dt: float = 0.01):
        """
        Euler step for Kuramoto dynamics.

        Args:
            dt: Time step
        """
        with torch.no_grad():
            d_theta = self.dynamics()
            self.theta.add_(dt * d_theta)

    def order_parameter(self) -> Tuple[Tensor, Tensor]:
        """
        Compute Kuramoto order parameter r e^{iψ}.

            r e^{iψ} = (1/N) Σ_j e^{iθ_j}

        In our framework, this corresponds to the consensus metric G_shared.

        Returns:
            r: [D] - Synchronization magnitude per dimension
            psi: [D] - Mean phase per dimension
        """
        # Complex representation
        z = torch.exp(1j * self.theta)  # [N, D]

        # Mean
        z_mean = z.mean(dim=0)  # [D]

        r = torch.abs(z_mean)
        psi = torch.angle(z_mean)

        return r, psi

    def is_synchronized(self, threshold: float = 0.9) -> bool:
        """
        Check if agents are synchronized.

        Args:
            threshold: r > threshold indicates synchronization

        Returns:
            synchronized: True if r > threshold
        """
        r, _ = self.order_parameter()
        return (r.mean() > threshold).item()


# =============================================================================
# Complete Module: GaugeCovariantMetricSynchronizer (Enhanced)
# =============================================================================

class GaugeCovariantMetricSynchronizer(nn.Module):
    """
    Complete implementation of Chapter 37: Inter-Subjective Metric.

    Implements:
        - Metric Friction (Definition 37.1)
        - Locking Operator (Theorem 37.1)
        - Gauge-Covariant Translation (Definition 37.7)
        - Babel Limit checking (Theorem 37.4)
        - Spectral Locking (Theorem 37.5)

    This module aligns the latent geometries of two agents via gauge-covariant
    transport, enabling the emergence of shared objective reality.
    """

    def __init__(self, config: LockingConfig):
        """
        Args:
            config: Configuration parameters for the locking system
        """
        super().__init__()
        self.config = config

        # Gauge connections for each agent
        self.connection_A = GaugeConnection(
            config.latent_dim,
            config.gauge_dim,
            config.spacetime_dim
        )
        self.connection_B = GaugeConnection(
            config.latent_dim,
            config.gauge_dim,
            config.spacetime_dim
        )

        # Language channel
        self.language = LanguageChannel(
            config.gauge_dim,
            config.gauge_dim // 2  # 50% compression
        )

        # Lie algebra message encoding
        self.message_algebra = LieAlgebraMessage(config.gauge_dim)

        # Learnable gauge transform (for Procrustes alignment)
        self.gauge_transform = nn.Linear(
            config.latent_dim,
            config.latent_dim,
            bias=False
        )
        nn.init.orthogonal_(self.gauge_transform.weight)

        # Bandwidth tracking
        self.bandwidth = FiniteCommunicationBandwidth(config)

    def compute_metric_friction(
        self,
        z_A: Tensor,
        z_B: Tensor
    ) -> Tensor:
        """
        Definition 37.1: Metric Friction (computational form).
        """
        return metric_friction_from_embeddings(
            z_A, z_B,
            self.config.use_procrustes
        )

    def compute_locking_curvature(
        self,
        z_A: Tensor,
        z_B: Tensor
    ) -> Tensor:
        """
        Definition 37.3: Locking Curvature.
        """
        # Get connections at these points
        A_mu_A = self.connection_A(z_A)
        A_mu_B = self.connection_B(z_B)

        # Inter-agent connection
        A_AB = inter_agent_connection(
            A_mu_A, A_mu_B,
            self.config.lambda_lock
        )

        # Curvature
        F_AB = locking_curvature(A_AB, self.config.g_lock)

        return F_AB

    def compute_locking_loss(
        self,
        z_A: Tensor,
        z_B: Tensor,
        x_pred_A: Optional[Tensor] = None,
        x_pred_B: Optional[Tensor] = None,
        x_true: Optional[Tensor] = None
    ) -> Tuple[Tensor, dict]:
        """
        Theorem 37.1 & 37.2: Locking Operator and Joint Loss.

        Args:
            z_A: Agent A's latent states
            z_B: Agent B's latent states
            x_pred_A: Agent A's predictions (optional)
            x_pred_B: Agent B's predictions (optional)
            x_true: Ground truth (optional)

        Returns:
            loss: Total locking loss
            metrics: Dictionary of component metrics
        """
        # Metric friction
        friction = self.compute_metric_friction(z_A, z_B)

        # Locking curvature and operator
        F_AB = self.compute_locking_curvature(z_A, z_B)
        L_sync = locking_operator(F_AB, self.config.g_lock)
        Psi_sync = integrated_friction(F_AB)

        metrics = {
            'friction': friction.mean().item(),
            'L_sync': L_sync.mean().item(),
            'Psi_sync': Psi_sync.mean().item()
        }

        # Joint prediction loss if predictions provided
        if x_pred_A is not None and x_pred_B is not None and x_true is not None:
            loss = joint_prediction_loss(
                x_pred_A, x_pred_B, x_true,
                Psi_sync, self.config.beta
            )
            metrics['joint_loss'] = loss.item()
        else:
            loss = self.config.lambda_lock * friction + L_sync.mean()

        return loss, metrics

    def send_message(
        self,
        m_full: Tensor
    ) -> Tensor:
        """
        Definition 37.6: Send message through language channel.

        Args:
            m_full: [B, gauge_dim] - Full message

        Returns:
            m_received: [B, gauge_dim] - Message after channel (lossy)
        """
        return self.language(m_full)

    def check_babel_limit(
        self,
        G_A: Tensor,
        boundary_area: Tensor
    ) -> Tuple[bool, dict]:
        """
        Theorem 37.4: Check Babel Limit.

        Args:
            G_A: Metric tensor of Agent A
            boundary_area: Area of language channel boundary

        Returns:
            achievable: Whether complete locking is achievable
            info: Dictionary with detailed information
        """
        # Channel capacity
        C_L = self.bandwidth.channel_capacity(boundary_area)

        # Metric entropy (simplified: log det)
        H_G = 0.5 * torch.logdet(G_A + 1e-6 * torch.eye(G_A.shape[-1], device=G_A.device))

        # Check limit
        satisfied, I_required = babel_limit_satisfied(
            self.config.gauge_dim, H_G, C_L
        )

        # Unlocked dimension
        d_unlocked = unlocked_dimension(self.config.gauge_dim, H_G, C_L)

        info = {
            'channel_capacity': C_L.item() if isinstance(C_L, Tensor) else C_L,
            'metric_entropy': H_G.item() if isinstance(H_G, Tensor) else H_G,
            'I_required': I_required.item() if isinstance(I_required, Tensor) else I_required,
            'd_unlocked': d_unlocked.item() if isinstance(d_unlocked, Tensor) else d_unlocked,
            'private_qualia_dim': d_unlocked.item() if isinstance(d_unlocked, Tensor) else d_unlocked
        }

        achievable = satisfied.item() if isinstance(satisfied, Tensor) else satisfied

        return achievable, info

    def spectral_analysis(
        self,
        G: Tensor,
        channel_capacity: float,
        time: float
    ) -> dict:
        """
        Theorem 37.5: Spectral Locking Analysis.

        Args:
            G: Metric tensor
            channel_capacity: C_L
            time: Communication time T

        Returns:
            analysis: Dictionary with spectral locking information
        """
        eigenvalues, eigenvectors = metric_eigendecomposition(G)

        k_max = spectral_locking_order(
            eigenvalues,
            torch.tensor(channel_capacity),
            time
        )

        core, nuance, split_idx = core_vs_nuance_split(eigenvalues)

        return {
            'eigenvalues': eigenvalues,
            'eigenvectors': eigenvectors,
            'k_max': k_max,
            'core_eigenvalues': core,
            'nuance_eigenvalues': nuance,
            'core_nuance_split': split_idx
        }

    def forward(
        self,
        z_A: Tensor,
        z_B: Tensor
    ) -> Tuple[Tensor, Tensor, dict]:
        """
        Full forward pass: align agents and compute loss.

        Args:
            z_A: [B, D] - Agent A's states
            z_B: [B, D] - Agent B's states

        Returns:
            loss: Locking loss
            z_B_aligned: Agent B's states after alignment
            metrics: Dictionary of metrics
        """
        # Apply gauge transform
        z_B_aligned = self.gauge_transform(z_B)

        # Compute loss
        loss, metrics = self.compute_locking_loss(z_A, z_B_aligned)

        # Check semantic alignment
        friction_before = self.compute_metric_friction(z_A, z_B)
        friction_after = self.compute_metric_friction(z_A, z_B_aligned)

        understanding = semantic_alignment(friction_before, friction_after)
        metrics['understanding_rate'] = understanding.float().mean().item()
        metrics['friction_reduction'] = (friction_before - friction_after).mean().item()

        return loss, z_B_aligned, metrics


# =============================================================================
# Diagnostic Functions (Nodes 69-70)
# =============================================================================

def metric_alignment_check(
    friction: Tensor,
    threshold: float
) -> Tuple[bool, str]:
    """
    Node 69: MetricAlignmentCheck

    Monitors whether agents see the same world.

    Args:
        friction: F_AB metric friction
        threshold: F_thresh

    Returns:
        passed: Whether check passed
        message: Diagnostic message
    """
    passed = (friction < threshold).all().item()

    if passed:
        message = f"MetricAlignmentCheck PASSED: F_AB = {friction.mean():.4f} < {threshold}"
    else:
        message = f"MetricAlignmentCheck FAILED: F_AB = {friction.mean():.4f} > {threshold}. " \
                  f"Agents talking past each other. Remediation: increase bandwidth or trigger sync."

    return passed, message


def babel_check(
    friction_current: Tensor,
    friction_previous: Tensor,
    friction_env_A: Optional[Tensor] = None,
    friction_env_B: Optional[Tensor] = None
) -> Tuple[bool, str]:
    """
    Node 70: BabelCheck

    Monitors whether language is drifting.

    Args:
        friction_current: F_AB(t)
        friction_previous: F_AB(t-1)
        friction_env_A: F_AE (optional, for echo chamber detection)
        friction_env_B: F_BE (optional)

    Returns:
        passed: Whether check passed
        message: Diagnostic message
    """
    d_friction = friction_current - friction_previous

    # Main check: divergence
    diverging = (d_friction > 0).any().item()

    # Echo chamber check
    echo_chamber = False
    if friction_env_A is not None and friction_env_B is not None:
        env_increasing = (friction_env_A.mean() + friction_env_B.mean()) > 0
        inter_agent_decreasing = d_friction.mean() < 0
        echo_chamber = env_increasing and inter_agent_decreasing

    if echo_chamber:
        passed = False
        message = "BabelCheck WARNING: Echo Chamber detected. " \
                  "Agents aligning with each other but drifting from environment."
    elif diverging:
        passed = False
        message = f"BabelCheck FAILED: ∂F_AB/∂t = {d_friction.mean():.4f} > 0. " \
                  f"Language losing grounding. Force ostensive definitions."
    else:
        passed = True
        message = f"BabelCheck PASSED: ∂F_AB/∂t = {d_friction.mean():.4f} ≤ 0"

    return passed, message


# =============================================================================
# Utility Functions
# =============================================================================

def estimate_metric_from_samples(
    z: Tensor,
    eps: float = 1e-4
) -> Tensor:
    """
    Estimate metric tensor G from samples via local covariance.

    The Fisher-Rao metric can be approximated as the inverse covariance
    of the gradient of the log-likelihood.

    Args:
        z: [B, D] - Samples from the latent space
        eps: Regularization for numerical stability

    Returns:
        G: [D, D] - Estimated metric tensor
    """
    # Center the data
    z_centered = z - z.mean(dim=0, keepdim=True)

    # Covariance
    cov = (z_centered.T @ z_centered) / (z.shape[0] - 1)

    # Metric is inverse covariance (Fisher information)
    G = torch.linalg.inv(cov + eps * torch.eye(z.shape[1], device=z.device))

    return G


def gromov_wasserstein_distance(
    z_A: Tensor,
    z_B: Tensor,
    p: int = 2,
    n_iter: int = 100
) -> Tensor:
    """
    Compute Gromov-Wasserstein distance between two point clouds.

    This is the proper metric for comparing manifolds without correspondence.
    Approximated via Sinkhorn iterations.

    Args:
        z_A: [N, D] - Points from manifold A
        z_B: [M, D] - Points from manifold B
        p: Power for distance (2 = quadratic GW)
        n_iter: Sinkhorn iterations

    Returns:
        gw_dist: Gromov-Wasserstein distance
    """
    N, D = z_A.shape
    M = z_B.shape[0]

    # Intra-manifold distance matrices
    C_A = torch.cdist(z_A, z_A) ** p  # [N, N]
    C_B = torch.cdist(z_B, z_B) ** p  # [M, M]

    # Normalize
    C_A = C_A / C_A.max()
    C_B = C_B / C_B.max()

    # Initialize transport plan (uniform)
    T = torch.ones(N, M, device=z_A.device) / (N * M)

    # Sinkhorn iterations for entropic GW
    for _ in range(n_iter):
        # GW cost matrix
        # L(C_A, C_B, T) = sum_{i,j,k,l} |C_A[i,k] - C_B[j,l]|^2 T[i,j] T[k,l]
        # Simplified: use tensor contraction
        cost = torch.einsum('ik,jl,ij,kl->', C_A, C_B, T, T)

        # Update T (simplified Sinkhorn step)
        row_sum = T.sum(dim=1, keepdim=True)
        T = T / (row_sum + 1e-8)
        col_sum = T.sum(dim=0, keepdim=True)
        T = T / (col_sum + 1e-8)
        T = T / T.sum()

    # Final GW distance
    gw_dist = torch.einsum('ik,jl,ij,kl->', (C_A.unsqueeze(1) - C_B.unsqueeze(0)) ** 2,
                           torch.ones_like(C_A).unsqueeze(1).unsqueeze(-1),
                           T.unsqueeze(-1).unsqueeze(-1),
                           T.unsqueeze(0).unsqueeze(0)).sqrt()

    # Simplified computation
    gw_dist = ((C_A.unsqueeze(1).unsqueeze(-1) - C_B.unsqueeze(0).unsqueeze(2)) ** 2 *
               T.unsqueeze(-1).unsqueeze(-1) * T.unsqueeze(0).unsqueeze(2)).sum().sqrt()

    return gw_dist


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    # Configuration
    config = LockingConfig(
        latent_dim=32,
        gauge_dim=8,
        spacetime_dim=4,
        beta=2.0,
        lambda_lock=1.0
    )

    # Create synchronizer
    sync = GaugeCovariantMetricSynchronizer(config)

    # Example: Two agents with different representations
    batch_size = 64
    z_A = torch.randn(batch_size, config.latent_dim)
    z_B = torch.randn(batch_size, config.latent_dim) + 0.5  # Shifted

    # Forward pass
    loss, z_B_aligned, metrics = sync(z_A, z_B)

    print("=== Chapter 37: Inter-Subjective Metric ===")
    print(f"Initial friction: {metrics['friction']:.4f}")
    print(f"Locking loss: {loss.item():.4f}")
    print(f"Understanding rate: {metrics['understanding_rate']:.2%}")
    print(f"Friction reduction: {metrics['friction_reduction']:.4f}")

    # Diagnostic checks
    friction = sync.compute_metric_friction(z_A, z_B)
    passed, msg = metric_alignment_check(friction, config.friction_thresh)
    print(f"\n{msg}")

    # Babel limit check
    G_A = estimate_metric_from_samples(z_A)
    boundary_area = torch.tensor(100.0)  # Example area
    achievable, info = sync.check_babel_limit(G_A, boundary_area)
    print(f"\nBabel Limit Check:")
    print(f"  Complete locking achievable: {achievable}")
    print(f"  Private qualia dimension: {info['private_qualia_dim']}")

    # Kuramoto simulation
    print("\n=== Kuramoto Synchronization ===")
    kuramoto = KuramotoAgents(n_agents=10, latent_dim=config.latent_dim, beta=2.0)

    for step in range(100):
        kuramoto.step(dt=0.1)
        if step % 20 == 0:
            r, psi = kuramoto.order_parameter()
            print(f"Step {step}: Order parameter r = {r.mean():.3f}")

    print(f"Synchronized: {kuramoto.is_synchronized()}")
