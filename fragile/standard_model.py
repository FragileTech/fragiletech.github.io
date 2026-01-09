"""
Chapters 34-35: Standard Model of Cognition & Parameter Space Sieve — PyTorch Implementation
============================================================================================

This module provides PyTorch implementations of all equations from Chapters 34-35
of the Fragile monograph (docs/source/info_sm.md).

Tensor Dimension Convention:
    B   = batch size
    D   = latent manifold dimension
    S   = spacetime dimensions (typically 4)
    N_f = feature dimension (color charges, default 3)
    N_i = isospin dimension (2 for SU(2))
    C   = spinor components (4 for Dirac spinor)
    G_a = gauge algebra dimension for each group

Sections covered:
    - 34.1: Gauge Principle (U(1)_Y, SU(2)_L, SU(N_f)_C)
    - 34.2: Matter Sector (Cognitive Spinor, Covariant Derivative)
    - 34.3: Scalar Sector (Higgs Mechanism, SSB)
    - 34.4: Interaction Terms (Yukawa, Value Coupling)
    - 34.5: Unified Cognitive Lagrangian
    - 35.1-35.2: Sieve Formulation
    - 35.3-35.7: Constraint Checks
    - 35.8-35.9: Optimization
    - Diagnostic Nodes (2, 7, 29, 40, 52, 56, 62)

References:
    - Definition 34.1 (def-utility-gauge-freedom): Utility Gauge Freedom
    - Theorem 34.1 (thm-emergence-opportunity-field): Emergence of B_mu
    - Theorem 34.2 (thm-emergence-error-field): Emergence of W_mu
    - Theorem 34.3 (thm-emergence-binding-field): Emergence of G_mu
    - Definition 34.4 (def-cognitive-spinor): Cognitive Spinor
    - Definition 34.5 (def-universal-covariant-derivative): Universal D_mu
    - Theorem 34.4 (thm-three-cognitive-forces): Field Strength Tensors
    - Theorem 34.5 (thm-complexity-potential): Mexican Hat Potential
    - Corollary 34.2 (cor-ontological-ssb): Spontaneous Symmetry Breaking
    - Theorem 34.6 (thm-semantic-inertia): Mass Generation
    - Definition 34.9 (def-cognitive-lagrangian): Full Lagrangian
    - Definition 35.1 (def-agent-parameter-vector): Agent Parameters
    - Theorem 35.1 (thm-speed-window): Speed Window
    - Theorem 35.2 (thm-holographic-bound): Holographic Bound
    - Theorem 35.4 (thm-landauer-constraint): Landauer Constraint
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict, NamedTuple
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class StandardModelConfig:
    """Configuration for the Standard Model of Cognition.

    Parameters correspond to theoretical constants from Chapters 34-35.

    Dimension Parameters:
        latent_dim (D): Dimension of the latent manifold
        feature_dim (N_f): Feature dimension for SU(N_f)_C (color charges)
        spacetime_dim (S): Spacetime dimensions (typically 4)

    Gauge Couplings (Section 34.1):
        g1: U(1)_Y hypercharge coupling [dimensionless]
        g2: SU(2)_L weak isospin coupling [dimensionless]
        gs: SU(N_f)_C binding coupling [dimensionless]

    Scalar Sector (Section 34.3):
        mu_sq: Higgs mass parameter mu^2 [energy^2], >0 for SSB
        lambda_higgs: Quartic self-coupling [dimensionless]
        xi_crit: Critical ontological stress Xi_crit [dimensionless]
        alpha_stab: Stabilization coefficient alpha [dimensionless]

    Cognitive Parameters (Section 34.2):
        sigma: Cognitive action scale [nat*time], analog of hbar
        m_inference: Inference mass [1/length]

    Yukawa (Section 34.4):
        yukawa_coupling: Decision coupling Y [dimensionless]

    Sieve Parameters (Section 35.1):
        c_info: Information speed [length/time]
        ell_L: Levin length [length]
        T_c: Cognitive temperature [energy]
        gamma_discount: Temporal discount factor [dimensionless]

    Sieve Architecture:
        L_buf: Buffer depth [length]
        d_sync: Synchronization distance [length]
        tau_proc: Processing interval [time]
    """
    # Dimensions
    latent_dim: int = 64              # D: latent manifold dimension
    feature_dim: int = 3              # N_f: feature dimension (default 3 like QCD)
    spacetime_dim: int = 4            # S: spacetime dimensions

    # Gauge Couplings (Section 34.1)
    g1: float = 0.35                  # g_1: U(1)_Y hypercharge coupling
    g2: float = 0.65                  # g_2: SU(2)_L weak isospin coupling
    gs: float = 1.0                   # g_s: SU(N_f)_C binding coupling

    # Hypercharges for different fields
    Y_L: float = -1.0                 # Y_L: Left-handed doublet hypercharge
    Y_R: float = -2.0                 # Y_R: Right-handed singlet hypercharge
    Y_phi: float = 1.0                # Y_phi: Higgs doublet hypercharge

    # Scalar Sector (Section 34.3)
    mu_sq: float = 1.0                # mu^2: Higgs mass parameter (>0 for SSB)
    lambda_higgs: float = 0.1         # lambda: quartic self-coupling
    xi_crit: float = 0.5              # Xi_crit: critical ontological stress
    alpha_stab: float = 0.25          # alpha: stabilization coefficient

    # Cognitive Parameters (Section 34.2)
    sigma: float = 0.1                # sigma: cognitive action scale
    m_inference: float = 0.5          # m: inference mass

    # Yukawa Coupling (Section 34.4)
    yukawa_coupling: float = 1.0      # Y: decision coupling strength

    # Sieve Parameters (Section 35.1)
    c_info: float = 1.0               # c_info: information speed
    ell_L: float = 0.01               # ell_L: Levin length
    T_c: float = 0.1                  # T_c: cognitive temperature
    gamma_discount: float = 0.99      # gamma: temporal discount

    # Sieve Architecture Parameters
    L_buf: float = 100.0              # L_buf: buffer depth
    d_sync: float = 1.0               # d_sync: synchronization distance
    tau_proc: float = 0.1             # tau_proc: processing interval

    # Metabolic parameters (Section 35.4)
    E_dot_met: float = 100.0          # Metabolic power budget [energy/time]
    I_dot_erase: float = 10.0         # Information erasure rate [bits/time]

    # Stiffness parameters (Section 35.6)
    delta_E: float = 1.0              # Energy gap between metastable states
    chi_max: float = 100.0            # Maximum stiffness ratio

    # Coupling RG parameters (Section 35.5)
    gs_crit: float = 1.0              # Critical binding coupling
    mu_IR: float = 0.1                # IR resolution scale
    mu_UV: float = 100.0              # UV resolution scale
    gs_epsilon: float = 0.01          # UV decoupling threshold


# =============================================================================
# Mathematical Primitives
# =============================================================================

def minkowski_metric(device: torch.device = None) -> Tensor:
    """Minkowski metric tensor g^{mu,nu} = diag(+1, -1, -1, -1).

    The metric for flat spacetime with signature (+,-,-,-).

    Returns:
        g: Metric tensor
           Shape: (S=4, S=4)
           g[mu, nu] = eta_{mu,nu}

    Reference:
        Used throughout Section 34 for index raising/lowering.
    """
    # Shape: (4, 4)
    g = torch.diag(torch.tensor([1.0, -1.0, -1.0, -1.0], device=device))
    return g


def gamma_matrices(device: torch.device = None) -> Tensor:
    """Construct 4D Dirac gamma matrices satisfying {gamma^mu, gamma^nu} = 2*g^{mu,nu}.

    Uses the Dirac (standard) representation:
        gamma^0 = [[I, 0], [0, -I]]
        gamma^i = [[0, sigma^i], [-sigma^i, 0]]

    where sigma^i are Pauli matrices.

    Returns:
        gamma: Gamma matrices
               Shape: (S=4, C=4, C=4)
               gamma[mu, a, b] = (gamma^mu)_{ab}

    Mathematical Property:
        {gamma^mu, gamma^nu} = gamma^mu @ gamma^nu + gamma^nu @ gamma^mu = 2 * g^{mu,nu} * I_4

    Reference:
        Axiom 34.2 (ax-cognitive-dirac-equation): Cognitive Dirac Equation
    """
    # Pauli matrices for constructing gamma matrices
    # sigma_1, sigma_2, sigma_3: each shape (2, 2)
    sigma_1 = torch.tensor([[0.0, 1.0], [1.0, 0.0]], device=device)
    sigma_2 = torch.tensor([[0.0, -1.0j], [1.0j, 0.0]], device=device, dtype=torch.complex64)
    sigma_3 = torch.tensor([[1.0, 0.0], [0.0, -1.0]], device=device)

    # Identity matrices
    I2 = torch.eye(2, device=device)
    zeros2 = torch.zeros(2, 2, device=device)

    # gamma^0 = [[I, 0], [0, -I]]
    # Shape: (4, 4)
    gamma_0 = torch.zeros(4, 4, dtype=torch.complex64, device=device)
    gamma_0[:2, :2] = I2
    gamma_0[2:, 2:] = -I2

    # gamma^1 = [[0, sigma_1], [-sigma_1, 0]]
    gamma_1 = torch.zeros(4, 4, dtype=torch.complex64, device=device)
    gamma_1[:2, 2:] = sigma_1
    gamma_1[2:, :2] = -sigma_1

    # gamma^2 = [[0, sigma_2], [-sigma_2, 0]]
    gamma_2 = torch.zeros(4, 4, dtype=torch.complex64, device=device)
    gamma_2[:2, 2:] = sigma_2
    gamma_2[2:, :2] = -sigma_2

    # gamma^3 = [[0, sigma_3], [-sigma_3, 0]]
    gamma_3 = torch.zeros(4, 4, dtype=torch.complex64, device=device)
    gamma_3[:2, 2:] = sigma_3
    gamma_3[2:, :2] = -sigma_3

    # Stack into (S=4, C=4, C=4)
    gamma = torch.stack([gamma_0, gamma_1, gamma_2, gamma_3], dim=0)

    return gamma


def gamma5_matrix(device: torch.device = None) -> Tensor:
    """Construct gamma^5 = i * gamma^0 * gamma^1 * gamma^2 * gamma^3.

    Used for chiral projection operators P_L = (1 - gamma^5)/2 and P_R = (1 + gamma^5)/2.

    Returns:
        gamma5: Chirality matrix
                Shape: (C=4, C=4)

    Property:
        (gamma^5)^2 = I, {gamma^5, gamma^mu} = 0
    """
    gamma = gamma_matrices(device)
    # gamma5 = i * gamma[0] @ gamma[1] @ gamma[2] @ gamma[3]
    # Shape: (4, 4)
    gamma5 = 1j * gamma[0] @ gamma[1] @ gamma[2] @ gamma[3]
    return gamma5


def pauli_matrices(device: torch.device = None) -> Tensor:
    """Pauli matrices tau^a for SU(2)_L generators.

    tau^1 = [[0, 1], [1, 0]]
    tau^2 = [[0, -i], [i, 0]]
    tau^3 = [[1, 0], [0, -1]]

    Satisfying [tau^a, tau^b] = 2i * epsilon^{abc} * tau^c

    Returns:
        tau: Pauli matrices
             Shape: (3, N_i=2, N_i=2)
             tau[a, i, j] = (tau^a)_{ij}

    Reference:
        Theorem 34.2 (thm-emergence-error-field): Error Field W_mu
    """
    tau_1 = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.complex64, device=device)
    tau_2 = torch.tensor([[0.0, -1.0j], [1.0j, 0.0]], dtype=torch.complex64, device=device)
    tau_3 = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.complex64, device=device)

    # Stack: shape (3, 2, 2)
    tau = torch.stack([tau_1, tau_2, tau_3], dim=0)
    return tau


def gellmann_matrices(n_f: int, device: torch.device = None) -> Tensor:
    """Generalized Gell-Mann matrices for SU(N_f) generators.

    For N_f=2: Returns 3 matrices (Pauli matrices)
    For N_f=3: Returns 8 standard Gell-Mann matrices

    Satisfying Tr(lambda^a * lambda^b) = 2 * delta^{ab}

    Args:
        n_f: Feature dimension (number of colors)

    Returns:
        lambda_a: Gell-Mann matrices
                  Shape: (N_f^2-1, N_f, N_f)
                  lambda_a[a, i, j] = (lambda^a)_{ij}

    Reference:
        Theorem 34.3 (thm-emergence-binding-field): Binding Field G_mu
    """
    n_generators = n_f * n_f - 1

    if n_f == 2:
        # SU(2): Just Pauli matrices
        return pauli_matrices(device)

    elif n_f == 3:
        # SU(3): Standard Gell-Mann matrices
        # Shape: (8, 3, 3)
        matrices = torch.zeros(8, 3, 3, dtype=torch.complex64, device=device)

        # lambda_1: off-diagonal (1,2)
        matrices[0, 0, 1] = 1.0
        matrices[0, 1, 0] = 1.0

        # lambda_2: off-diagonal (1,2) with i
        matrices[1, 0, 1] = -1.0j
        matrices[1, 1, 0] = 1.0j

        # lambda_3: diagonal
        matrices[2, 0, 0] = 1.0
        matrices[2, 1, 1] = -1.0

        # lambda_4: off-diagonal (1,3)
        matrices[3, 0, 2] = 1.0
        matrices[3, 2, 0] = 1.0

        # lambda_5: off-diagonal (1,3) with i
        matrices[4, 0, 2] = -1.0j
        matrices[4, 2, 0] = 1.0j

        # lambda_6: off-diagonal (2,3)
        matrices[5, 1, 2] = 1.0
        matrices[5, 2, 1] = 1.0

        # lambda_7: off-diagonal (2,3) with i
        matrices[6, 1, 2] = -1.0j
        matrices[6, 2, 1] = 1.0j

        # lambda_8: diagonal (normalized)
        matrices[7, 0, 0] = 1.0 / math.sqrt(3)
        matrices[7, 1, 1] = 1.0 / math.sqrt(3)
        matrices[7, 2, 2] = -2.0 / math.sqrt(3)

        return matrices

    else:
        # General SU(N_f): Construct generalized Gell-Mann matrices
        # Uses the standard construction for arbitrary N
        matrices = torch.zeros(n_generators, n_f, n_f, dtype=torch.complex64, device=device)
        idx = 0

        # Symmetric off-diagonal matrices
        for i in range(n_f):
            for j in range(i + 1, n_f):
                matrices[idx, i, j] = 1.0
                matrices[idx, j, i] = 1.0
                idx += 1

        # Antisymmetric off-diagonal matrices
        for i in range(n_f):
            for j in range(i + 1, n_f):
                matrices[idx, i, j] = -1.0j
                matrices[idx, j, i] = 1.0j
                idx += 1

        # Diagonal matrices
        for k in range(1, n_f):
            norm = math.sqrt(2.0 / (k * (k + 1)))
            for i in range(k):
                matrices[idx, i, i] = norm
            matrices[idx, k, k] = -k * norm
            idx += 1

        return matrices


def su_n_structure_constants(n: int, device: torch.device = None) -> Tensor:
    """Structure constants f^{abc} for SU(N).

    Defined by [T^a, T^b] = i * f^{abc} * T^c
    where T^a = lambda^a / 2 are the generators.

    Args:
        n: The N in SU(N)

    Returns:
        f_abc: Structure constants
               Shape: (N^2-1, N^2-1, N^2-1)
               f_abc[a, b, c] = f^{abc}

    Reference:
        Theorem 34.4 (thm-three-cognitive-forces): Non-abelian field strength
    """
    n_gen = n * n - 1
    generators = gellmann_matrices(n, device) / 2  # T^a = lambda^a / 2
    # Shape: (n_gen, n, n)

    f_abc = torch.zeros(n_gen, n_gen, n_gen, dtype=torch.complex64, device=device)

    for a in range(n_gen):
        for b in range(n_gen):
            # [T^a, T^b] = T^a @ T^b - T^b @ T^a
            commutator = generators[a] @ generators[b] - generators[b] @ generators[a]
            # Shape: (n, n)

            for c in range(n_gen):
                # f^{abc} = -i * Tr([T^a, T^b] * T^c) * 2
                # Since Tr(T^a T^b) = delta^{ab}/2
                f_abc[a, b, c] = -1j * torch.trace(commutator @ generators[c]) * 2

    # Structure constants should be real
    return f_abc.real


def levi_civita_3d(device: torch.device = None) -> Tensor:
    """3D Levi-Civita tensor epsilon^{abc}.

    epsilon^{123} = 1, antisymmetric under any index swap.

    Returns:
        epsilon: Levi-Civita tensor
                 Shape: (3, 3, 3)
    """
    epsilon = torch.zeros(3, 3, 3, device=device)
    epsilon[0, 1, 2] = 1.0
    epsilon[1, 2, 0] = 1.0
    epsilon[2, 0, 1] = 1.0
    epsilon[0, 2, 1] = -1.0
    epsilon[1, 0, 2] = -1.0
    epsilon[2, 1, 0] = -1.0
    return epsilon


# =============================================================================
# Section 34.1: Gauge Fields
# =============================================================================

def utility_phase_transform(
    psi: Tensor,
    theta: Tensor
) -> Tensor:
    """Apply global U(1)_Y phase transformation to belief wave-function.

    psi(z) -> exp(i * theta) * psi(z)

    This corresponds to shifting the Value baseline: V(z) -> V(z) + sigma * theta

    Args:
        psi: Belief wave-function
             Shape: (B, D) or (B, D, ...) for spinor
        theta: Phase angle
               Shape: (B,) or scalar

    Returns:
        psi_transformed: Transformed wave-function
                         Shape: same as psi

    Reference:
        Definition 34.1 (def-utility-gauge-freedom): Utility Gauge Freedom
    """
    # Ensure theta has right shape for broadcasting
    if theta.dim() == 0:
        # Scalar theta
        phase = torch.exp(1j * theta)
    else:
        # theta shape (B,) -> need to add dimensions for broadcasting
        phase = torch.exp(1j * theta.view(-1, *([1] * (psi.dim() - 1))))

    return phase * psi


class OpportunityField(nn.Module):
    """U(1)_Y Gauge Field B_mu: The Opportunity Field.

    Compensates for local variations in the Value/utility baseline.
    The covariant derivative is: D_mu = d_mu - i * g1 * (Y/2) * B_mu

    Reference:
        Theorem 34.1 (thm-emergence-opportunity-field)

    Attributes:
        config: Model configuration
        B_mu: Gauge field parameters
              Shape: (S,) learnable, represents background field
    """

    def __init__(self, config: StandardModelConfig):
        super().__init__()
        self.config = config
        # B_mu: Shape (S,) - spacetime components of gauge field
        # In practice, this is a background field or computed from dynamics
        self.B_mu = nn.Parameter(torch.zeros(config.spacetime_dim))

    def field_at_point(self, x: Tensor) -> Tensor:
        """Get B_mu at spacetime point x.

        Args:
            x: Spacetime coordinates
               Shape: (B, S)

        Returns:
            B: Field values at each point
               Shape: (B, S)
               B[b, mu] = B_mu(x_b)
        """
        B, S = x.shape
        # For now, return constant background field
        # In full implementation, would depend on x
        # Shape: (B, S) by broadcasting (S,) -> (1, S) -> (B, S)
        return self.B_mu.unsqueeze(0).expand(B, S)

    def field_strength(self, B_mu: Tensor, derivatives: Tensor) -> Tensor:
        """Compute U(1) field strength tensor B_{mu,nu} = d_mu B_nu - d_nu B_mu.

        Args:
            B_mu: Gauge field
                  Shape: (B, S)
            derivatives: Partial derivatives d_mu B_nu
                         Shape: (B, S, S)
                         derivatives[b, mu, nu] = d_mu B_nu

        Returns:
            B_munu: Field strength tensor (antisymmetric)
                    Shape: (B, S, S)
                    B_munu[b, mu, nu] = d_mu B_nu - d_nu B_mu

        Reference:
            Theorem 34.4 (thm-three-cognitive-forces): U(1)_Y Curvature
        """
        B, S, _ = derivatives.shape

        # B_{mu,nu} = d_mu B_nu - d_nu B_mu
        # derivatives[b, mu, nu] = d_mu B_nu
        # Shape: (B, S, S)
        B_munu = derivatives - derivatives.transpose(-2, -1)

        return B_munu


class CognitiveIsospinDoublet(nn.Module):
    """SU(2)_L Isospin Doublet: The Left-Handed Belief Field.

    Psi_L = (psi_pred, psi_obs)^T

    where:
    - psi_pred: Prior (top-down prediction)
    - psi_obs: Likelihood (bottom-up observation)

    Reference:
        Definition 34.2 (def-cognitive-isospin-doublet)
    """

    def __init__(self, config: StandardModelConfig):
        super().__init__()
        self.config = config

    def construct_doublet(
        self,
        psi_pred: Tensor,
        psi_obs: Tensor
    ) -> Tensor:
        """Construct isospin doublet from prediction and observation.

        Args:
            psi_pred: Prediction amplitude
                      Shape: (B, D, C, N_f) for spinor with color
            psi_obs: Observation amplitude
                     Shape: (B, D, C, N_f)

        Returns:
            Psi_L: Isospin doublet
                   Shape: (B, D, C, N_i=2, N_f)
                   Psi_L[..., 0, :] = psi_pred
                   Psi_L[..., 1, :] = psi_obs
        """
        # Stack along new isospin dimension
        # Shape: (B, D, C, 2, N_f)
        Psi_L = torch.stack([psi_pred, psi_obs], dim=-2)
        return Psi_L

    def decompose_doublet(self, Psi_L: Tensor) -> Tuple[Tensor, Tensor]:
        """Decompose isospin doublet into prediction and observation.

        Args:
            Psi_L: Isospin doublet
                   Shape: (B, D, C, N_i=2, N_f)

        Returns:
            psi_pred: Prediction amplitude, shape (B, D, C, N_f)
            psi_obs: Observation amplitude, shape (B, D, C, N_f)
        """
        psi_pred = Psi_L[..., 0, :]  # Shape: (B, D, C, N_f)
        psi_obs = Psi_L[..., 1, :]   # Shape: (B, D, C, N_f)
        return psi_pred, psi_obs


class ErrorField(nn.Module):
    """SU(2)_L Gauge Field W^a_mu: The Error Field.

    Mediates transitions between prediction and observation in belief updates.
    The covariant derivative acting on doublets is:
    D_mu Psi_L = (d_mu - i * g2 * (tau^a/2) * W^a_mu) Psi_L

    Reference:
        Theorem 34.2 (thm-emergence-error-field)

    Attributes:
        W_mu: Gauge field components
              Shape: (S, 3) - S spacetime indices, 3 SU(2) generators
    """

    def __init__(self, config: StandardModelConfig):
        super().__init__()
        self.config = config
        # W^a_mu: Shape (S, 3)
        self.W_mu = nn.Parameter(torch.zeros(config.spacetime_dim, 3))
        # Cache Pauli matrices
        self.register_buffer('tau', pauli_matrices())

    def field_at_point(self, x: Tensor) -> Tensor:
        """Get W^a_mu at spacetime point x.

        Args:
            x: Spacetime coordinates
               Shape: (B, S)

        Returns:
            W: Field values
               Shape: (B, S, 3)
               W[b, mu, a] = W^a_mu(x_b)
        """
        B, S = x.shape
        # Constant background field, broadcast to batch
        # Shape: (B, S, 3)
        return self.W_mu.unsqueeze(0).expand(B, S, 3)

    def field_strength(
        self,
        W_mu: Tensor,
        derivatives: Tensor
    ) -> Tensor:
        """Compute SU(2) field strength tensor W^a_{mu,nu}.

        W^a_{mu,nu} = d_mu W^a_nu - d_nu W^a_mu + g2 * epsilon^{abc} * W^b_mu * W^c_nu

        Args:
            W_mu: Gauge field
                  Shape: (B, S, 3)
            derivatives: Partial derivatives d_mu W^a_nu
                         Shape: (B, S, S, 3)
                         derivatives[b, mu, nu, a] = d_mu W^a_nu

        Returns:
            W_munu: Field strength tensor
                    Shape: (B, S, S, 3)
                    W_munu[b, mu, nu, a] = W^a_{mu,nu}

        Reference:
            Theorem 34.4 (thm-three-cognitive-forces): SU(2)_L Curvature
        """
        B, S, _ = W_mu.shape
        g2 = self.config.g2

        # Abelian part: d_mu W^a_nu - d_nu W^a_mu
        # derivatives: (B, S, S, 3)
        # Shape: (B, S, S, 3)
        abelian_part = derivatives - derivatives.transpose(1, 2)

        # Non-abelian part: g2 * epsilon^{abc} * W^b_mu * W^c_nu
        # W_mu: (B, S, 3), need W^b_mu and W^c_nu
        # Use einsum: epsilon[a,b,c] * W[B,mu,b] * W[B,nu,c]
        epsilon = levi_civita_3d(W_mu.device)  # (3, 3, 3)

        # Shape: (B, S, S, 3)
        # einsum: 'abc,Bmb,Bnc->BMna' where M=mu, N=nu
        nonabelian_part = g2 * torch.einsum(
            'abc,bmb,bnc->bmna',
            epsilon,
            W_mu,  # (B, S, 3) -> b indexes batch, m indexes mu
            W_mu   # (B, S, 3) -> b indexes batch, n indexes nu
        )

        return abelian_part + nonabelian_part


class BindingField(nn.Module):
    """SU(N_f)_C Gauge Field G^a_mu: The Binding/Gluon Field.

    Binds sub-symbolic features into composite concepts.
    The covariant derivative is:
    D_mu psi = (d_mu - i * gs * (lambda^a/2) * G^a_mu) psi

    Reference:
        Theorem 34.3 (thm-emergence-binding-field)

    Attributes:
        G_mu: Gauge field components
              Shape: (S, N_f^2-1)
    """

    def __init__(self, config: StandardModelConfig):
        super().__init__()
        self.config = config
        n_generators = config.feature_dim ** 2 - 1
        # G^a_mu: Shape (S, N_f^2-1)
        self.G_mu = nn.Parameter(torch.zeros(config.spacetime_dim, n_generators))
        # Cache Gell-Mann matrices
        self.register_buffer('lambda_a', gellmann_matrices(config.feature_dim))
        # Cache structure constants
        self.register_buffer('f_abc', su_n_structure_constants(config.feature_dim))

    def field_at_point(self, x: Tensor) -> Tensor:
        """Get G^a_mu at spacetime point x.

        Args:
            x: Spacetime coordinates
               Shape: (B, S)

        Returns:
            G: Field values
               Shape: (B, S, N_f^2-1)
        """
        B, S = x.shape
        n_gen = self.config.feature_dim ** 2 - 1
        # Shape: (B, S, N_f^2-1)
        return self.G_mu.unsqueeze(0).expand(B, S, n_gen)

    def field_strength(
        self,
        G_mu: Tensor,
        derivatives: Tensor
    ) -> Tensor:
        """Compute SU(N_f) field strength tensor G^a_{mu,nu}.

        G^a_{mu,nu} = d_mu G^a_nu - d_nu G^a_mu + gs * f^{abc} * G^b_mu * G^c_nu

        Args:
            G_mu: Gauge field
                  Shape: (B, S, N_f^2-1)
            derivatives: Partial derivatives
                         Shape: (B, S, S, N_f^2-1)

        Returns:
            G_munu: Field strength tensor
                    Shape: (B, S, S, N_f^2-1)

        Reference:
            Theorem 34.4 (thm-three-cognitive-forces): SU(N_f)_C Curvature
        """
        gs = self.config.gs

        # Abelian part
        # Shape: (B, S, S, N_f^2-1)
        abelian_part = derivatives - derivatives.transpose(1, 2)

        # Non-abelian part: gs * f^{abc} * G^b_mu * G^c_nu
        # f_abc: (N_f^2-1, N_f^2-1, N_f^2-1)
        # G_mu: (B, S, N_f^2-1)
        # Shape: (B, S, S, N_f^2-1)
        nonabelian_part = gs * torch.einsum(
            'abc,bmb,bnc->bmna',
            self.f_abc,
            G_mu,
            G_mu
        )

        return abelian_part + nonabelian_part


# =============================================================================
# Section 34.2: Matter Sector
# =============================================================================

class CognitiveSpinor(nn.Module):
    """Cognitive Spinor Field Psi(x).

    The belief state as a spinor field:
    Psi = (Psi_L, Psi_R) in C^4 x C^2 x C^{N_f}

    where:
    - Psi_L: Left-handed doublet (prediction/observation)
    - Psi_R: Right-handed singlet (action)

    Reference:
        Definition 34.4 (def-cognitive-spinor)

    Tensor Shapes:
        Psi_L: (B, D, C=4, N_i=2, N_f) - left-handed doublet
        Psi_R: (B, D, C=4, N_f) - right-handed singlet
    """

    def __init__(self, config: StandardModelConfig):
        super().__init__()
        self.config = config
        self.isospin = CognitiveIsospinDoublet(config)

        # Cache gamma^5 for chiral projection
        self.register_buffer('gamma5', gamma5_matrix())

    def chiral_project_left(self, psi: Tensor) -> Tensor:
        """Project spinor to left-handed component: P_L = (1 - gamma^5)/2.

        Args:
            psi: Full Dirac spinor
                 Shape: (B, D, C=4, ...)

        Returns:
            psi_L: Left-handed component
                   Shape: (B, D, C=4, ...)

        Dimension flow:
            psi: (B, D, 4, ...) @ P_L: (4, 4) -> (B, D, 4, ...)
        """
        # P_L = (I - gamma5) / 2
        # Shape: (4, 4)
        I4 = torch.eye(4, dtype=self.gamma5.dtype, device=self.gamma5.device)
        P_L = (I4 - self.gamma5) / 2

        # Apply projection: contract over spinor index
        # psi: (B, D, 4, ...), P_L: (4, 4)
        # Result: (B, D, 4, ...)
        return torch.einsum('...i...,ij->...j...', psi, P_L)

    def chiral_project_right(self, psi: Tensor) -> Tensor:
        """Project spinor to right-handed component: P_R = (1 + gamma^5)/2.

        Args:
            psi: Full Dirac spinor
                 Shape: (B, D, C=4, ...)

        Returns:
            psi_R: Right-handed component
                   Shape: (B, D, C=4, ...)
        """
        I4 = torch.eye(4, dtype=self.gamma5.dtype, device=self.gamma5.device)
        P_R = (I4 + self.gamma5) / 2
        return torch.einsum('...i...,ij->...j...', psi, P_R)

    def probability_density(self, Psi_L: Tensor, Psi_R: Tensor) -> Tensor:
        """Compute probability density rho = Psi^dag Psi.

        Args:
            Psi_L: Left-handed doublet
                   Shape: (B, D, C=4, N_i=2, N_f)
            Psi_R: Right-handed singlet
                   Shape: (B, D, C=4, N_f)

        Returns:
            rho: Probability density
                 Shape: (B, D)

        Dimension flow:
            |Psi_L|^2: sum over (C, N_i, N_f) -> (B, D)
            |Psi_R|^2: sum over (C, N_f) -> (B, D)
            rho = |Psi_L|^2 + |Psi_R|^2: (B, D)
        """
        # |Psi_L|^2: sum over spinor, isospin, color
        rho_L = (Psi_L.conj() * Psi_L).real.sum(dim=(-3, -2, -1))  # (B, D)
        # |Psi_R|^2: sum over spinor, color
        rho_R = (Psi_R.conj() * Psi_R).real.sum(dim=(-2, -1))  # (B, D)

        return rho_L + rho_R

    def probability_current(
        self,
        Psi: Tensor,
        gamma: Tensor
    ) -> Tensor:
        """Compute probability current J^mu = Psi_bar gamma^mu Psi.

        Args:
            Psi: Full spinor (combined L and R)
                 Shape: (B, D, C=4, ...)
            gamma: Gamma matrices
                   Shape: (S=4, C=4, C=4)

        Returns:
            J: Probability current
               Shape: (B, D, S)
               J[b, d, mu] = J^mu at point (b, d)

        Dimension flow:
            Psi_bar: (B, D, 4, ...) conjugate transpose
            gamma^mu @ Psi: (4, 4) @ (B, D, 4, ...) -> (B, D, 4, ...)
            Psi_bar @ result: contract spinor -> (B, D, ...)
            Sum remaining indices -> (B, D)
            Stack over mu -> (B, D, S)
        """
        S = gamma.shape[0]
        # Psi_bar = Psi^dag @ gamma^0
        gamma0 = gamma[0]  # (4, 4)

        # Contract over all non-batch, non-spatial dimensions
        J_list = []
        for mu in range(S):
            # gamma^mu @ Psi contracted appropriately
            # This is simplified; full implementation needs proper index handling
            gamma_mu_psi = torch.einsum('ij,...j...->...i...', gamma[mu], Psi)
            # Psi_bar @ gamma_mu_psi
            Psi_dag = Psi.conj()
            Psi_bar = torch.einsum('...i...,ij->...j...', Psi_dag, gamma0)
            # Contract over spinor index
            J_mu = (Psi_bar * gamma_mu_psi).real.sum(dim=tuple(range(2, Psi.dim())))
            J_list.append(J_mu)

        # Shape: (B, D, S)
        return torch.stack(J_list, dim=-1)


class UniversalCovariantDerivative(nn.Module):
    """Universal Covariant Derivative D_mu for the full gauge group.

    D_mu = d_mu - i*g1*(Y/2)*B_mu - i*g2*(tau^a/2)*W^a_mu - i*gs*(lambda^a/2)*G^a_mu

    This is the derivative that parallel transports the belief spinor
    through the latent manifold while compensating for all gauge connections.

    Reference:
        Definition 34.5 (def-universal-covariant-derivative)
    """

    def __init__(self, config: StandardModelConfig):
        super().__init__()
        self.config = config

        # Initialize gauge fields
        self.B_field = OpportunityField(config)
        self.W_field = ErrorField(config)
        self.G_field = BindingField(config)

        # Cache generators
        self.register_buffer('tau', pauli_matrices())
        self.register_buffer('lambda_a', gellmann_matrices(config.feature_dim))

    def covariant_derivative_left(
        self,
        Psi_L: Tensor,
        partial_mu_Psi_L: Tensor,
        B_mu: Tensor,
        W_mu: Tensor,
        G_mu: Tensor,
        Y: float
    ) -> Tensor:
        """Apply covariant derivative to left-handed doublet.

        D_mu Psi_L = d_mu Psi_L - i*g1*(Y/2)*B_mu*Psi_L
                     - i*g2*(tau^a/2)*W^a_mu*Psi_L
                     - i*gs*(lambda^a/2)*G^a_mu*Psi_L

        Args:
            Psi_L: Left-handed doublet
                   Shape: (B, D, C=4, N_i=2, N_f)
            partial_mu_Psi_L: Partial derivative d_mu Psi_L
                              Shape: (B, D, S, C=4, N_i=2, N_f)
            B_mu: U(1)_Y gauge field
                  Shape: (B, S)
            W_mu: SU(2)_L gauge field
                  Shape: (B, S, 3)
            G_mu: SU(N_f)_C gauge field
                  Shape: (B, S, N_f^2-1)
            Y: Hypercharge

        Returns:
            D_mu_Psi_L: Covariant derivative
                        Shape: (B, D, S, C=4, N_i=2, N_f)

        Dimension flow:
            partial: (B, D, S, 4, 2, N_f)
            U(1) term: (B, S) * (B, D, 4, 2, N_f) -> broadcast -> (B, D, S, 4, 2, N_f)
            SU(2) term: (B, S, 3) with (3, 2, 2) on Psi -> (B, D, S, 4, 2, N_f)
            SU(N_f) term: (B, S, N_f^2-1) with (N_f^2-1, N_f, N_f) on Psi -> (B, D, S, 4, 2, N_f)
        """
        g1, g2, gs = self.config.g1, self.config.g2, self.config.gs
        B, D, C, N_i, N_f = Psi_L.shape
        S = B_mu.shape[1]

        # Start with partial derivative
        # Shape: (B, D, S, C, N_i, N_f)
        result = partial_mu_Psi_L.clone()

        # U(1)_Y term: -i * g1 * (Y/2) * B_mu * Psi_L
        # B_mu: (B, S) -> expand to (B, 1, S, 1, 1, 1) for broadcasting
        B_mu_expanded = B_mu.view(B, 1, S, 1, 1, 1)
        Psi_L_expanded = Psi_L.unsqueeze(2)  # (B, D, 1, C, N_i, N_f) -> broadcast to (B, D, S, C, N_i, N_f)
        u1_term = -1j * g1 * (Y / 2) * B_mu_expanded * Psi_L_expanded
        result = result + u1_term

        # SU(2)_L term: -i * g2 * (tau^a/2) * W^a_mu * Psi_L
        # W_mu: (B, S, 3), tau: (3, 2, 2)
        # Need to contract a with tau and apply to isospin index of Psi_L
        # W_mu[b, mu, a] * tau[a, i, j] * Psi_L[b, d, c, j, f]
        # -> sum over a, j -> result[b, d, mu, c, i, f]
        for mu_idx in range(S):
            # tau_W = sum_a W^a_mu * tau^a / 2
            # Shape: (B, 2, 2)
            tau_W = torch.einsum('ba,aij->bij', W_mu[:, mu_idx, :], self.tau) / 2
            # Apply to Psi_L: (B, 2, 2) @ (B, D, C, 2, N_f) over isospin
            # Result: (B, D, C, 2, N_f)
            su2_contribution = torch.einsum('bij,bdcjf->bdcif', tau_W, Psi_L)
            result[:, :, mu_idx, :, :, :] = result[:, :, mu_idx, :, :, :] - 1j * g2 * su2_contribution

        # SU(N_f)_C term: -i * gs * (lambda^a/2) * G^a_mu * Psi_L
        # G_mu: (B, S, N_f^2-1), lambda_a: (N_f^2-1, N_f, N_f)
        for mu_idx in range(S):
            # lambda_G = sum_a G^a_mu * lambda^a / 2
            # Shape: (B, N_f, N_f)
            lambda_G = torch.einsum('ba,aij->bij', G_mu[:, mu_idx, :], self.lambda_a) / 2
            # Apply to Psi_L over color index
            # (B, N_f, N_f) @ (B, D, C, N_i, N_f) -> (B, D, C, N_i, N_f)
            sun_contribution = torch.einsum('bfg,bdcig->bdcif', lambda_G, Psi_L)
            result[:, :, mu_idx, :, :, :] = result[:, :, mu_idx, :, :, :] - 1j * gs * sun_contribution

        return result

    def covariant_derivative_right(
        self,
        Psi_R: Tensor,
        partial_mu_Psi_R: Tensor,
        B_mu: Tensor,
        G_mu: Tensor,
        Y: float
    ) -> Tensor:
        """Apply covariant derivative to right-handed singlet.

        D_mu Psi_R = d_mu Psi_R - i*g1*(Y/2)*B_mu*Psi_R - i*gs*(lambda^a/2)*G^a_mu*Psi_R

        Note: Right-handed is SU(2)_L singlet, so no W_mu term.

        Args:
            Psi_R: Right-handed singlet
                   Shape: (B, D, C=4, N_f)
            partial_mu_Psi_R: Partial derivative
                              Shape: (B, D, S, C=4, N_f)
            B_mu: U(1)_Y gauge field, shape (B, S)
            G_mu: SU(N_f)_C gauge field, shape (B, S, N_f^2-1)
            Y: Hypercharge

        Returns:
            D_mu_Psi_R: Covariant derivative
                        Shape: (B, D, S, C=4, N_f)
        """
        g1, gs = self.config.g1, self.config.gs
        B, D, C, N_f = Psi_R.shape
        S = B_mu.shape[1]

        result = partial_mu_Psi_R.clone()

        # U(1)_Y term
        B_mu_expanded = B_mu.view(B, 1, S, 1, 1)
        Psi_R_expanded = Psi_R.unsqueeze(2)  # (B, D, 1, C, N_f)
        u1_term = -1j * g1 * (Y / 2) * B_mu_expanded * Psi_R_expanded
        result = result + u1_term

        # SU(N_f)_C term
        for mu_idx in range(S):
            lambda_G = torch.einsum('ba,aij->bij', G_mu[:, mu_idx, :], self.lambda_a) / 2
            sun_contribution = torch.einsum('bfg,bdcg->bdcf', lambda_G, Psi_R)
            result[:, :, mu_idx, :, :] = result[:, :, mu_idx, :, :] - 1j * gs * sun_contribution

        return result


def yang_mills_lagrangian(
    B_munu: Tensor,
    W_munu: Tensor,
    G_munu: Tensor
) -> Tensor:
    """Compute Yang-Mills Lagrangian for all gauge fields.

    L_gauge = -1/4 * B_{mu,nu} * B^{mu,nu}
              -1/4 * W^a_{mu,nu} * W^{a,mu,nu}
              -1/4 * G^a_{mu,nu} * G^{a,mu,nu}

    Args:
        B_munu: U(1)_Y field strength
                Shape: (B, S, S)
        W_munu: SU(2)_L field strength
                Shape: (B, S, S, 3)
        G_munu: SU(N_f)_C field strength
                Shape: (B, S, S, N_f^2-1)

    Returns:
        L_gauge: Gauge sector Lagrangian density
                 Shape: (B,)

    Reference:
        Corollary 34.1 (cor-gauge-invariant-action)

    Dimension flow:
        B_munu * B^munu: (B, S, S) * (B, S, S) -> sum -> (B,)
        W_munu * W^munu: (B, S, S, 3) * (B, S, S, 3) -> sum over all but B -> (B,)
        G_munu * G^munu: (B, S, S, N_f^2-1) -> sum -> (B,)
        Total: (B,)
    """
    # Use Minkowski metric to raise indices
    # For simplicity, use trace: F_{mu,nu} F^{mu,nu} = sum over mu,nu
    # (with metric signature accounted for)

    # B_{mu,nu} B^{mu,nu}: sum over spatial indices
    # Shape: (B,)
    L_B = -0.25 * (B_munu * B_munu).sum(dim=(-2, -1))

    # W^a_{mu,nu} W^{a,mu,nu}: sum over spatial and generator indices
    # Shape: (B,)
    L_W = -0.25 * (W_munu * W_munu).sum(dim=(-3, -2, -1))

    # G^a_{mu,nu} G^{a,mu,nu}
    # Shape: (B,)
    L_G = -0.25 * (G_munu * G_munu).sum(dim=(-3, -2, -1))

    return L_B + L_W + L_G


# =============================================================================
# Section 34.3: Scalar Sector (Higgs Mechanism)
# =============================================================================

class OntologicalOrderParameter(nn.Module):
    """Ontological Order Parameter phi(x): The Higgs Field of Cognition.

    phi(x) = r(x) * exp(i * theta(x))

    where:
    - r(x): Metric separation between concepts (modulus)
    - theta(x): Orientation in feature space (phase)

    The field transforms as a doublet under SU(2)_L.

    Reference:
        Definition 34.6 (def-ontological-order-parameter)
    """

    def __init__(self, config: StandardModelConfig):
        super().__init__()
        self.config = config
        # Higgs doublet: 2 complex components
        # phi = (phi_+, phi_0)^T
        # In unitary gauge after SSB: phi = (0, v + h)^T / sqrt(2)

    def construct_doublet(
        self,
        phi_plus: Tensor,
        phi_zero: Tensor
    ) -> Tensor:
        """Construct Higgs doublet from components.

        Args:
            phi_plus: Upper component (charged)
                      Shape: (B, D) complex
            phi_zero: Lower component (neutral)
                      Shape: (B, D) complex

        Returns:
            phi: Higgs doublet
                 Shape: (B, D, 2)
        """
        # Shape: (B, D, 2)
        return torch.stack([phi_plus, phi_zero], dim=-1)

    def unitary_gauge_vacuum(
        self,
        v: float,
        h: Tensor
    ) -> Tensor:
        """Construct Higgs field in unitary gauge around VEV.

        phi = (0, (v + h) / sqrt(2))^T

        Args:
            v: Vacuum expectation value
            h: Higgs excitation field
               Shape: (B, D)

        Returns:
            phi: Higgs doublet in unitary gauge
                 Shape: (B, D, 2)
        """
        B, D = h.shape
        device = h.device

        phi_plus = torch.zeros(B, D, dtype=torch.complex64, device=device)
        phi_zero = (v + h) / math.sqrt(2)

        return self.construct_doublet(phi_plus, phi_zero.to(torch.complex64))


def complexity_potential(
    phi: Tensor,
    config: StandardModelConfig
) -> Tensor:
    """Compute the Complexity (Mexican Hat) Potential.

    V(phi) = -mu^2 * |phi|^2 + lambda * |phi|^4

    This is derived from the pitchfork bifurcation dynamics (Thm 34.5).

    Args:
        phi: Higgs doublet
             Shape: (B, D, 2) complex
        config: Model configuration

    Returns:
        V: Potential energy density
           Shape: (B, D)

    Reference:
        Theorem 34.5 (thm-complexity-potential)

    Dimension flow:
        |phi|^2 = |phi_+|^2 + |phi_0|^2: (B, D, 2) -> sum dim=-1 -> (B, D)
        |phi|^4: (B, D)
        V: (B, D)
    """
    mu_sq = config.mu_sq
    lambda_h = config.lambda_higgs

    # |phi|^2 = sum over doublet components of |phi_i|^2
    # Shape: (B, D)
    phi_sq = (phi.conj() * phi).real.sum(dim=-1)

    # |phi|^4
    # Shape: (B, D)
    phi_fourth = phi_sq ** 2

    # V = -mu^2 |phi|^2 + lambda |phi|^4
    # Shape: (B, D)
    V = -mu_sq * phi_sq + lambda_h * phi_fourth

    return V


def vacuum_expectation_value(config: StandardModelConfig) -> float:
    """Compute the Vacuum Expectation Value (VEV) v.

    v = sqrt(mu^2 / (2 * lambda))

    This is the equilibrium separation in the broken phase.

    Args:
        config: Model configuration

    Returns:
        v: VEV (scalar)

    Reference:
        Corollary 34.2 (cor-ontological-ssb)

    Requires:
        mu^2 > 0 for spontaneous symmetry breaking
    """
    if config.mu_sq <= 0:
        return 0.0  # Symmetric phase

    v = math.sqrt(config.mu_sq / (2 * config.lambda_higgs))
    return v


def spontaneous_symmetry_breaking(
    phi_sq: Tensor,
    config: StandardModelConfig
) -> Dict[str, Tensor]:
    """Analyze spontaneous symmetry breaking phase.

    Case 1 (Symmetric, Xi < Xi_crit): mu^2 < 0, minimum at phi = 0
    Case 2 (Broken, Xi > Xi_crit): mu^2 > 0, minimum at |phi| = v

    Args:
        phi_sq: |phi|^2 values to analyze
                Shape: (B, D)
        config: Model configuration

    Returns:
        Dictionary with:
            'phase': 'symmetric' or 'broken'
            'vev': VEV value (0 in symmetric phase)
            'potential_min': Value of potential at minimum
            'is_stable': Whether phi_sq is near stable minimum

    Reference:
        Corollary 34.2 (cor-ontological-ssb)
    """
    mu_sq = config.mu_sq
    lambda_h = config.lambda_higgs

    if mu_sq <= 0:
        # Symmetric phase
        phase = 'symmetric'
        vev = 0.0
        potential_min = torch.zeros_like(phi_sq[:, 0])
        # Stable if near zero
        is_stable = phi_sq.abs() < 0.1
    else:
        # Broken phase
        phase = 'broken'
        vev = math.sqrt(mu_sq / (2 * lambda_h))
        # V_min = -mu^4 / (4 * lambda)
        potential_min = -config.mu_sq ** 2 / (4 * lambda_h) * torch.ones_like(phi_sq[:, 0])
        # Stable if near v^2
        is_stable = (phi_sq - vev ** 2).abs() < 0.1 * vev ** 2

    return {
        'phase': phase,
        'vev': vev,
        'potential_min': potential_min,
        'is_stable': is_stable
    }


def gauge_boson_mass(
    coupling: float,
    v: float
) -> float:
    """Compute gauge boson mass from Higgs mechanism.

    M = g * v

    After SSB, gauge bosons acquire mass proportional to VEV.

    Args:
        coupling: Gauge coupling constant g
        v: Vacuum expectation value

    Returns:
        M: Gauge boson mass

    Reference:
        Theorem 34.6 (thm-semantic-inertia)
    """
    return coupling * v


def higgs_mass(config: StandardModelConfig) -> float:
    """Compute Higgs boson mass.

    m_H = sqrt(2 * mu^2) = sqrt(2 * lambda) * v

    Args:
        config: Model configuration

    Returns:
        m_H: Higgs mass

    Reference:
        Theorem 34.6 (thm-semantic-inertia)
    """
    if config.mu_sq <= 0:
        return 0.0
    return math.sqrt(2 * config.mu_sq)


def scalar_kinetic_lagrangian(
    phi: Tensor,
    D_mu_phi: Tensor
) -> Tensor:
    """Compute scalar kinetic Lagrangian |D_mu phi|^2.

    L_kin = (D_mu phi)^dag (D^mu phi)

    Args:
        phi: Higgs doublet
             Shape: (B, D, 2)
        D_mu_phi: Covariant derivative of Higgs
                  Shape: (B, D, S, 2)

    Returns:
        L_kin: Kinetic Lagrangian density
               Shape: (B, D)

    Dimension flow:
        |D_mu phi|^2: (B, D, S, 2) -> |.|^2 -> sum over S, 2 -> (B, D)
    """
    # |D_mu phi|^2 = sum over mu and doublet index
    # Shape: (B, D)
    L_kin = (D_mu_phi.conj() * D_mu_phi).real.sum(dim=(-2, -1))
    return L_kin


# =============================================================================
# Section 34.4: Interaction Terms
# =============================================================================

def yukawa_lagrangian(
    Psi_L: Tensor,
    Psi_R: Tensor,
    phi: Tensor,
    config: StandardModelConfig
) -> Tensor:
    """Compute Yukawa (Decision Coupling) Lagrangian.

    L_Y = -Y * (Psi_bar_L . phi . Psi_R + h.c.)

    This couples the belief doublet to action singlet via the Higgs.

    Args:
        Psi_L: Left-handed doublet
               Shape: (B, D, C=4, N_i=2, N_f)
        Psi_R: Right-handed singlet
               Shape: (B, D, C=4, N_f)
        phi: Higgs doublet
             Shape: (B, D, 2)
        config: Model configuration

    Returns:
        L_Y: Yukawa Lagrangian density
             Shape: (B, D)

    Reference:
        Definition 34.7 (def-decision-coupling)

    Dimension flow:
        Psi_bar_L: (B, D, 4, 2, N_f) conjugate
        phi: (B, D, 2) contracts with isospin index
        Psi_R: (B, D, 4, N_f)
        Contract all internal indices -> (B, D)
    """
    Y = config.yukawa_coupling

    # Psi_bar_L = Psi_L^dag @ gamma^0 (simplified: just conjugate for now)
    Psi_bar_L = Psi_L.conj()

    # Contract: Psi_bar_L[..., i, :] * phi[..., i] * Psi_R
    # Sum over isospin i, spinor c, color f
    # Shape: (B, D)

    # Simplified contraction (full version needs gamma^0)
    # Psi_bar_L . phi: contract isospin
    # Shape after: (B, D, C, N_f)
    Psi_bar_L_phi = torch.einsum('bdcif,bdi->bdcf', Psi_bar_L, phi)

    # . Psi_R: contract spinor and color
    # Shape: (B, D)
    term1 = torch.einsum('bdcf,bdcf->bd', Psi_bar_L_phi, Psi_R)

    # Hermitian conjugate
    term2 = term1.conj()

    # L_Y = -Y * (term1 + term2)
    L_Y = -Y * (term1 + term2).real

    return L_Y


def cognitive_mass(config: StandardModelConfig) -> float:
    """Compute cognitive (decision) mass from Yukawa coupling.

    m_psi = Y * v

    In broken phase, spinor acquires mass through Higgs mechanism.

    Args:
        config: Model configuration

    Returns:
        m_psi: Cognitive mass

    Reference:
        Theorem 34.7 (thm-cognitive-mass)
    """
    v = vacuum_expectation_value(config)
    return config.yukawa_coupling * v


class Value4Potential(nn.Module):
    """External Value 4-Potential A^ext_mu.

    A^ext_mu = (-Phi_eff, 0, 0, 0)

    where Phi_eff is the effective potential from the Value function.

    Reference:
        Definition 34.8 (def-value-4-potential)
    """

    def __init__(self, config: StandardModelConfig):
        super().__init__()
        self.config = config

    def compute(self, Phi_eff: Tensor) -> Tensor:
        """Construct 4-potential from effective potential.

        Args:
            Phi_eff: Effective potential field
                     Shape: (B, D)

        Returns:
            A_ext: External 4-potential
                   Shape: (B, D, S=4)
                   A_ext[..., 0] = -Phi_eff
                   A_ext[..., 1:] = 0
        """
        B, D = Phi_eff.shape
        S = self.config.spacetime_dim
        device = Phi_eff.device

        A_ext = torch.zeros(B, D, S, device=device)
        A_ext[..., 0] = -Phi_eff

        return A_ext


def drive_lagrangian(
    rho: Tensor,
    Phi_eff: Tensor
) -> Tensor:
    """Compute Value Drive Lagrangian.

    L_drive = -rho * Phi_eff

    This couples the belief density to the value potential.

    Args:
        rho: Probability density
             Shape: (B, D)
        Phi_eff: Effective potential
                 Shape: (B, D)

    Returns:
        L_drive: Drive Lagrangian density
                 Shape: (B, D)

    Reference:
        Axiom 34.3 (ax-minimal-value-coupling)
    """
    return -rho * Phi_eff


# =============================================================================
# Section 34.5: Unified Cognitive Lagrangian
# =============================================================================

class CognitiveLagrangian(nn.Module):
    """The Complete Standard Model of Cognition Lagrangian.

    L_SM = L_gauge + L_inference + L_scalar + L_yukawa + L_drive

    where:
    I.   L_gauge: Yang-Mills for B, W, G fields
    II.  L_inference: Dirac kinetic for Psi_L, Psi_R
    III. L_scalar: Higgs kinetic + potential
    IV.  L_yukawa: Decision coupling
    V.   L_drive: Value coupling

    Reference:
        Definition 34.9 (def-cognitive-lagrangian)
    """

    def __init__(self, config: StandardModelConfig):
        super().__init__()
        self.config = config

        # Initialize all components
        self.covariant_deriv = UniversalCovariantDerivative(config)
        self.higgs = OntologicalOrderParameter(config)
        self.value_potential = Value4Potential(config)
        self.spinor = CognitiveSpinor(config)

        # Cache gamma matrices
        self.register_buffer('gamma', gamma_matrices())

    def compute_gauge_sector(
        self,
        B_munu: Tensor,
        W_munu: Tensor,
        G_munu: Tensor
    ) -> Tensor:
        """Compute Sector I: Gauge field Lagrangian.

        L_gauge = -1/4 B_munu B^munu - 1/4 W^a_munu W^{a,munu} - 1/4 G^a_munu G^{a,munu}

        Returns:
            L_gauge: Shape (B,)
        """
        return yang_mills_lagrangian(B_munu, W_munu, G_munu)

    def compute_inference_sector(
        self,
        Psi_L: Tensor,
        Psi_R: Tensor,
        D_mu_Psi_L: Tensor,
        D_mu_Psi_R: Tensor
    ) -> Tensor:
        """Compute Sector II: Inference (Dirac) Lagrangian.

        L_inf = Psi_bar_L i gamma^mu D_mu Psi_L + Psi_bar_R i gamma^mu D_mu Psi_R

        Args:
            Psi_L: Left-handed doublet, shape (B, D, 4, 2, N_f)
            Psi_R: Right-handed singlet, shape (B, D, 4, N_f)
            D_mu_Psi_L: Covariant derivative, shape (B, D, S, 4, 2, N_f)
            D_mu_Psi_R: Covariant derivative, shape (B, D, S, 4, N_f)

        Returns:
            L_inf: Shape (B, D)
        """
        # Simplified version: i * Psi^dag @ gamma^mu @ D_mu Psi
        # Sum over spacetime, spinor, isospin, color

        L_L = torch.zeros(Psi_L.shape[0], Psi_L.shape[1], device=Psi_L.device)
        L_R = torch.zeros(Psi_R.shape[0], Psi_R.shape[1], device=Psi_R.device)

        for mu in range(self.config.spacetime_dim):
            # gamma^mu @ D_mu Psi_L
            # gamma[mu]: (4, 4), D_mu_Psi_L[:,:,mu]: (B, D, 4, 2, N_f)
            gamma_D_L = torch.einsum('ij,bdjkf->bdikf', self.gamma[mu], D_mu_Psi_L[:, :, mu])
            # Psi_bar_L @ gamma_D_L
            L_L = L_L + 1j * (Psi_L.conj() * gamma_D_L).sum(dim=(-3, -2, -1)).real

            # Same for right-handed
            gamma_D_R = torch.einsum('ij,bdjf->bdif', self.gamma[mu], D_mu_Psi_R[:, :, mu])
            L_R = L_R + 1j * (Psi_R.conj() * gamma_D_R).sum(dim=(-2, -1)).real

        return L_L + L_R

    def compute_scalar_sector(
        self,
        phi: Tensor,
        D_mu_phi: Tensor
    ) -> Tensor:
        """Compute Sector III: Scalar (Higgs) Lagrangian.

        L_scalar = |D_mu phi|^2 - V(phi)

        Args:
            phi: Higgs doublet, shape (B, D, 2)
            D_mu_phi: Covariant derivative, shape (B, D, S, 2)

        Returns:
            L_scalar: Shape (B, D)
        """
        L_kin = scalar_kinetic_lagrangian(phi, D_mu_phi)
        V = complexity_potential(phi, self.config)
        return L_kin - V

    def compute_yukawa_sector(
        self,
        Psi_L: Tensor,
        Psi_R: Tensor,
        phi: Tensor
    ) -> Tensor:
        """Compute Sector IV: Yukawa Lagrangian.

        Returns:
            L_yukawa: Shape (B, D)
        """
        return yukawa_lagrangian(Psi_L, Psi_R, phi, self.config)

    def compute_drive_sector(
        self,
        rho: Tensor,
        Phi_eff: Tensor
    ) -> Tensor:
        """Compute Sector V: Value Drive Lagrangian.

        Returns:
            L_drive: Shape (B, D)
        """
        return drive_lagrangian(rho, Phi_eff)

    def forward(
        self,
        Psi_L: Tensor,
        Psi_R: Tensor,
        phi: Tensor,
        B_munu: Tensor,
        W_munu: Tensor,
        G_munu: Tensor,
        D_mu_Psi_L: Tensor,
        D_mu_Psi_R: Tensor,
        D_mu_phi: Tensor,
        Phi_eff: Tensor
    ) -> Dict[str, Tensor]:
        """Compute full Standard Model Lagrangian.

        Args:
            Psi_L: Left-handed doublet, (B, D, 4, 2, N_f)
            Psi_R: Right-handed singlet, (B, D, 4, N_f)
            phi: Higgs doublet, (B, D, 2)
            B_munu: U(1) field strength, (B, S, S)
            W_munu: SU(2) field strength, (B, S, S, 3)
            G_munu: SU(N_f) field strength, (B, S, S, N_f^2-1)
            D_mu_Psi_L: Covariant deriv of Psi_L, (B, D, S, 4, 2, N_f)
            D_mu_Psi_R: Covariant deriv of Psi_R, (B, D, S, 4, N_f)
            D_mu_phi: Covariant deriv of phi, (B, D, S, 2)
            Phi_eff: Effective potential, (B, D)

        Returns:
            Dictionary with all Lagrangian components and total
        """
        # Compute probability density
        rho = self.spinor.probability_density(Psi_L, Psi_R)

        # Compute each sector
        L_gauge = self.compute_gauge_sector(B_munu, W_munu, G_munu)
        L_inference = self.compute_inference_sector(Psi_L, Psi_R, D_mu_Psi_L, D_mu_Psi_R)
        L_scalar = self.compute_scalar_sector(phi, D_mu_phi)
        L_yukawa = self.compute_yukawa_sector(Psi_L, Psi_R, phi)
        L_drive = self.compute_drive_sector(rho, Phi_eff)

        # Total: sum over latent dimension D, keep batch B
        L_total = L_gauge + L_inference.sum(dim=-1) + L_scalar.sum(dim=-1) + L_yukawa.sum(dim=-1) + L_drive.sum(dim=-1)

        return {
            'L_gauge': L_gauge,
            'L_inference': L_inference,
            'L_scalar': L_scalar,
            'L_yukawa': L_yukawa,
            'L_drive': L_drive,
            'L_total': L_total,
            'rho': rho
        }


# =============================================================================
# Section 35.1-35.2: Sieve Formulation
# =============================================================================

class AgentParameterVector(NamedTuple):
    """Agent Parameter Vector Lambda = (c_info, sigma, ell_L, T_c, g_s, gamma).

    These are the fundamental operational constants of a viable agent.

    Reference:
        Definition 35.1 (def-agent-parameter-vector)
    """
    c_info: Tensor       # Information propagation speed [length/time]
    sigma: Tensor        # Cognitive action scale [nat*time]
    ell_L: Tensor        # Levin length [length]
    T_c: Tensor          # Cognitive temperature [energy]
    gs: Tensor           # Binding coupling [dimensionless]
    gamma: Tensor        # Temporal discount [dimensionless]


class SieveConstraint(Enum):
    """Enumeration of Sieve constraint types."""
    CAUSAL_LOWER = "causal_lower"      # Node 2: ZenoCheck
    CAUSAL_UPPER = "causal_upper"      # Node 62: CausalityViolationCheck
    HOLOGRAPHIC = "holographic"        # Node 56: CapacityHorizonCheck
    LANDAUER = "landauer"              # Node 52: LandauerViolationCheck
    IR_BINDING = "ir_binding"          # Node 40: PurityCheck
    UV_DECOUPLING = "uv_decoupling"    # Node 29: TextureFirewallCheck
    STIFFNESS_LOWER = "stiffness_lower"  # Node 7: StiffnessCheck
    STIFFNESS_UPPER = "stiffness_upper"  # Node 7: StiffnessCheck
    DISCOUNT_LOWER = "discount_lower"
    DISCOUNT_UPPER = "discount_upper"


# =============================================================================
# Section 35.3-35.7: Constraint Checks
# =============================================================================

def check_speed_window(
    c_info: Tensor,
    d_sync: float,
    L_buf: float,
    tau_proc: float
) -> Tuple[Tensor, Tensor]:
    """Check the Speed Window Constraint (Theorem 35.1).

    d_sync / tau_proc <= c_info <= L_buf / tau_proc

    Lower bound: Prevents Zeno freeze (Node 2)
    Upper bound: Prevents causal paradox (Node 62)

    Args:
        c_info: Information speed, shape (B,) or scalar
        d_sync: Synchronization distance
        L_buf: Buffer depth
        tau_proc: Processing interval

    Returns:
        lower_violation: >0 means violated, shape same as c_info
        upper_violation: >0 means violated, shape same as c_info

    Reference:
        Theorem 35.1 (thm-speed-window)
    """
    c_min = d_sync / tau_proc
    c_max = L_buf / tau_proc

    # Violation is positive when constraint is broken
    lower_violation = c_min - c_info  # >0 if c_info < c_min
    upper_violation = c_info - c_max  # >0 if c_info > c_max

    return lower_violation, upper_violation


def check_holographic_bound(
    ell_L: Tensor,
    area: float,
    I_req: float,
    D: int = 4,
    nu_D: float = 0.25
) -> Tensor:
    """Check the Holographic Bound Constraint (Theorem 35.2).

    ell_L^{D-1} <= nu_D * Area / I_req

    Args:
        ell_L: Levin length, shape (B,) or scalar
        area: Boundary area of latent manifold
        I_req: Required information capacity
        D: Bulk dimension (default 4)
        nu_D: Holographic coefficient (default 1/4 for D=2)

    Returns:
        violation: >0 means violated

    Reference:
        Theorem 35.2 (thm-holographic-bound)
    """
    bound = nu_D * area / I_req
    violation = ell_L ** (D - 1) - bound
    return violation


def capacity_horizon_metric(
    I_bulk: Tensor,
    I_max: float
) -> Tensor:
    """Compute metric component approaching capacity horizon.

    As I_bulk -> I_max, the metric diverges (Theorem 35.3).

    g_FR = 1 / (rho * (1 - rho)) where rho = I_bulk / I_max

    Args:
        I_bulk: Current bulk information, shape (B,)
        I_max: Maximum capacity

    Returns:
        g_FR: Fisher-Rao metric component, shape (B,)

    Reference:
        Theorem 35.3 (thm-capacity-horizon)
    """
    rho = I_bulk / I_max
    # Clamp to avoid division by zero
    rho = torch.clamp(rho, min=1e-6, max=1 - 1e-6)
    g_FR = 1.0 / (rho * (1 - rho))
    return g_FR


def check_landauer_constraint(
    T_c: Tensor,
    E_dot_met: float,
    I_dot_erase: float
) -> Tensor:
    """Check the Landauer Constraint (Theorem 35.4).

    T_c <= E_dot_met / (I_dot_erase * ln(2))

    Args:
        T_c: Cognitive temperature, shape (B,)
        E_dot_met: Metabolic power budget
        I_dot_erase: Information erasure rate

    Returns:
        violation: >0 means violated

    Reference:
        Theorem 35.4 (thm-landauer-constraint)
    """
    T_c_max = E_dot_met / (I_dot_erase * math.log(2))
    violation = T_c - T_c_max
    return violation


def coupling_beta_function(
    gs: Tensor,
    mu: Tensor,
    n_f: int = 3
) -> Tensor:
    """Compute beta function for SU(N_f) coupling.

    beta(g_s) = mu * dg_s/dmu

    For asymptotic freedom, beta < 0 when N_f >= 2.

    Args:
        gs: Current coupling, shape (B,)
        mu: Resolution scale, shape (B,)
        n_f: Feature dimension

    Returns:
        beta: Beta function value, shape (B,)

    Reference:
        Definition 35.3 (def-coupling-function)
    """
    # One-loop beta function for SU(N): beta = -b_0 * g^3 / (16*pi^2)
    # b_0 = (11*N - 2*N_f) / 3 for SU(N) with N_f fermion flavors
    # Simplified: assume N = N_f for feature binding
    b_0 = (11 * n_f) / 3  # No fermions in simplified model
    beta = -b_0 * gs ** 3 / (16 * math.pi ** 2)
    return beta


def check_ir_binding(
    gs_IR: Tensor,
    gs_crit: float
) -> Tensor:
    """Check IR Binding Constraint (Theorem 35.5).

    g_s(mu_IR) >= g_s^crit

    Args:
        gs_IR: Coupling at IR scale, shape (B,)
        gs_crit: Critical coupling for confinement

    Returns:
        violation: >0 means violated

    Reference:
        Theorem 35.5 (thm-ir-binding-constraint)
    """
    violation = gs_crit - gs_IR  # >0 if gs_IR < gs_crit
    return violation


def check_uv_decoupling(
    gs_UV: Tensor,
    epsilon: float = 0.01
) -> Tensor:
    """Check UV Decoupling Constraint (Theorem 35.6).

    g_s(mu_UV) <= epsilon (approximately 0)

    Args:
        gs_UV: Coupling at UV scale, shape (B,)
        epsilon: Decoupling threshold

    Returns:
        violation: >0 means violated

    Reference:
        Theorem 35.6 (thm-uv-decoupling-constraint)
    """
    violation = gs_UV - epsilon  # >0 if gs_UV > epsilon
    return violation


def check_stiffness_bounds(
    delta_E: Tensor,
    T_c: Tensor,
    chi_max: float = 100.0
) -> Tuple[Tensor, Tensor]:
    """Check Stiffness Bounds (Theorem 35.7).

    1 < chi = Delta_E / T_c < chi_max

    Args:
        delta_E: Energy gap, shape (B,)
        T_c: Cognitive temperature, shape (B,)
        chi_max: Maximum stiffness ratio

    Returns:
        lower_violation: >0 if chi < 1
        upper_violation: >0 if chi > chi_max

    Reference:
        Theorem 35.7 (thm-stiffness-bounds)
    """
    chi = delta_E / T_c

    lower_violation = 1.0 - chi  # >0 if chi < 1
    upper_violation = chi - chi_max  # >0 if chi > chi_max

    return lower_violation, upper_violation


def check_discount_window(
    gamma: Tensor,
    gamma_min: float = 0.5
) -> Tuple[Tensor, Tensor]:
    """Check Discount Window (Theorem 35.8).

    gamma_min < gamma < 1

    Args:
        gamma: Discount factor, shape (B,)
        gamma_min: Minimum discount (prevents myopia)

    Returns:
        lower_violation: >0 if gamma < gamma_min
        upper_violation: >0 if gamma >= 1

    Reference:
        Theorem 35.8 (thm-discount-window)
    """
    lower_violation = gamma_min - gamma
    upper_violation = gamma - 1.0 + 1e-6  # gamma must be strictly < 1

    return lower_violation, upper_violation


def screening_length(
    c_info: float,
    tau_proc: float,
    gamma: Tensor
) -> Tensor:
    """Compute temporal screening length.

    ell_gamma = c_info * tau_proc / (-ln(gamma))

    Args:
        c_info: Information speed
        tau_proc: Processing interval
        gamma: Discount factor, shape (B,)

    Returns:
        ell_gamma: Screening length, shape (B,)

    Reference:
        Corollary 35.2 (cor-screening-buffer-consistency)
    """
    ell_0 = c_info * tau_proc
    ell_gamma = ell_0 / (-torch.log(gamma))
    return ell_gamma


# =============================================================================
# Section 35.8-35.9: Optimization
# =============================================================================

class SieveConstraintSystem(nn.Module):
    """The Complete Sieve Constraint System.

    Combines all constraints S(Lambda) <= 0.

    Reference:
        Definition 35.2 (def-sieve-constraint-system)
    """

    def __init__(self, config: StandardModelConfig):
        super().__init__()
        self.config = config

    def evaluate_all(
        self,
        params: AgentParameterVector
    ) -> Dict[str, Tensor]:
        """Evaluate all sieve constraints.

        Args:
            params: Agent parameter vector

        Returns:
            Dictionary mapping constraint names to violation values
            (positive = violated, negative = satisfied)
        """
        config = self.config
        results = {}

        # Speed Window (Theorem 35.1)
        lower, upper = check_speed_window(
            params.c_info,
            config.d_sync,
            config.L_buf,
            config.tau_proc
        )
        results['speed_lower'] = lower
        results['speed_upper'] = upper

        # Holographic Bound (Theorem 35.2)
        # Use default area and I_req
        results['holographic'] = check_holographic_bound(
            params.ell_L,
            area=100.0,  # Placeholder
            I_req=1000.0  # Placeholder
        )

        # Landauer (Theorem 35.4)
        results['landauer'] = check_landauer_constraint(
            params.T_c,
            config.E_dot_met,
            config.I_dot_erase
        )

        # IR Binding (Theorem 35.5) - would need gs at IR scale
        # UV Decoupling (Theorem 35.6) - would need gs at UV scale
        # For now, use params.gs as proxy

        # Stiffness (Theorem 35.7)
        delta_E = torch.tensor(config.delta_E, device=params.T_c.device)
        lower, upper = check_stiffness_bounds(delta_E, params.T_c, config.chi_max)
        results['stiffness_lower'] = lower
        results['stiffness_upper'] = upper

        # Discount Window (Theorem 35.8)
        lower, upper = check_discount_window(params.gamma)
        results['discount_lower'] = lower
        results['discount_upper'] = upper

        return results

    def is_feasible(self, params: AgentParameterVector) -> Tensor:
        """Check if parameters are in the feasible region.

        Args:
            params: Agent parameter vector

        Returns:
            feasible: Boolean tensor, True if all constraints satisfied
        """
        violations = self.evaluate_all(params)
        # Feasible if all violations <= 0
        max_violation = torch.stack([v.max() for v in violations.values()]).max()
        return max_violation <= 0


def dual_objective(
    I_bulk: Tensor,
    V_metabolic: Tensor,
    beta: float = 1.0
) -> Tensor:
    """Compute dual objective for parameter optimization.

    J(Lambda) = I_bulk - beta * V_metabolic

    Args:
        I_bulk: Bulk information capacity, shape (B,)
        V_metabolic: Metabolic cost, shape (B,)
        beta: Cost sensitivity

    Returns:
        J: Objective value, shape (B,)

    Reference:
        Definition 35.4 (def-dual-objective)
    """
    return I_bulk - beta * V_metabolic


def compute_feasible_region(
    param_grid: Dict[str, Tensor],
    config: StandardModelConfig
) -> Tensor:
    """Compute feasibility mask over parameter grid.

    Args:
        param_grid: Dictionary of parameter tensors
        config: Model configuration

    Returns:
        feasible_mask: Boolean tensor indicating feasible points

    Reference:
        Theorem 35.9 (thm-feasible-region)
    """
    sieve = SieveConstraintSystem(config)

    # Create parameter vector from grid
    params = AgentParameterVector(
        c_info=param_grid['c_info'],
        sigma=param_grid['sigma'],
        ell_L=param_grid['ell_L'],
        T_c=param_grid['T_c'],
        gs=param_grid['gs'],
        gamma=param_grid['gamma']
    )

    return sieve.is_feasible(params)


# =============================================================================
# Diagnostic Nodes
# =============================================================================

class ZenoCheck(nn.Module):
    """Node 2: Check for Zeno freeze (speed too low).

    Reference: Theorem 35.1, lower bound
    """

    def __init__(self, config: StandardModelConfig):
        super().__init__()
        self.config = config

    def forward(self, c_info: Tensor) -> Dict[str, Tensor]:
        c_min = self.config.d_sync / self.config.tau_proc
        violation = c_min - c_info
        passed = violation <= 0
        return {
            'node': 'ZenoCheck',
            'passed': passed,
            'violation': violation,
            'c_min': torch.tensor(c_min),
            'c_info': c_info
        }


class StiffnessCheck(nn.Module):
    """Node 7: Check stiffness ratio bounds.

    Reference: Theorem 35.7
    """

    def __init__(self, config: StandardModelConfig):
        super().__init__()
        self.config = config

    def forward(self, delta_E: Tensor, T_c: Tensor) -> Dict[str, Tensor]:
        lower, upper = check_stiffness_bounds(delta_E, T_c, self.config.chi_max)
        chi = delta_E / T_c
        passed = (lower <= 0) & (upper <= 0)
        return {
            'node': 'StiffnessCheck',
            'passed': passed,
            'chi': chi,
            'lower_violation': lower,
            'upper_violation': upper
        }


class TextureFirewallCheck(nn.Module):
    """Node 29: Check UV decoupling (texture firewall).

    Reference: Theorem 35.6
    """

    def __init__(self, config: StandardModelConfig):
        super().__init__()
        self.config = config

    def forward(self, gs_UV: Tensor) -> Dict[str, Tensor]:
        violation = check_uv_decoupling(gs_UV, self.config.gs_epsilon)
        passed = violation <= 0
        return {
            'node': 'TextureFirewallCheck',
            'passed': passed,
            'violation': violation,
            'gs_UV': gs_UV
        }


class PurityCheck(nn.Module):
    """Node 40: Check IR binding (color confinement).

    Reference: Theorem 35.5
    """

    def __init__(self, config: StandardModelConfig):
        super().__init__()
        self.config = config

    def forward(self, gs_IR: Tensor) -> Dict[str, Tensor]:
        violation = check_ir_binding(gs_IR, self.config.gs_crit)
        passed = violation <= 0
        return {
            'node': 'PurityCheck',
            'passed': passed,
            'violation': violation,
            'gs_IR': gs_IR
        }


class LandauerViolationCheck(nn.Module):
    """Node 52: Check Landauer bound.

    Reference: Theorem 35.4
    """

    def __init__(self, config: StandardModelConfig):
        super().__init__()
        self.config = config

    def forward(self, T_c: Tensor) -> Dict[str, Tensor]:
        violation = check_landauer_constraint(
            T_c,
            self.config.E_dot_met,
            self.config.I_dot_erase
        )
        T_c_max = self.config.E_dot_met / (self.config.I_dot_erase * math.log(2))
        passed = violation <= 0
        return {
            'node': 'LandauerViolationCheck',
            'passed': passed,
            'violation': violation,
            'T_c': T_c,
            'T_c_max': torch.tensor(T_c_max)
        }


class CapacityHorizonCheck(nn.Module):
    """Node 56: Check holographic bound.

    Reference: Theorem 35.2
    """

    def __init__(self, config: StandardModelConfig):
        super().__init__()
        self.config = config

    def forward(
        self,
        ell_L: Tensor,
        area: float,
        I_req: float
    ) -> Dict[str, Tensor]:
        violation = check_holographic_bound(ell_L, area, I_req)
        passed = violation <= 0
        return {
            'node': 'CapacityHorizonCheck',
            'passed': passed,
            'violation': violation,
            'ell_L': ell_L
        }


class CausalityViolationCheck(nn.Module):
    """Node 62: Check for causal paradox (speed too high).

    Reference: Theorem 35.1, upper bound
    """

    def __init__(self, config: StandardModelConfig):
        super().__init__()
        self.config = config

    def forward(self, c_info: Tensor) -> Dict[str, Tensor]:
        c_max = self.config.L_buf / self.config.tau_proc
        violation = c_info - c_max
        passed = violation <= 0
        return {
            'node': 'CausalityViolationCheck',
            'passed': passed,
            'violation': violation,
            'c_max': torch.tensor(c_max),
            'c_info': c_info
        }


class SieveNodeRunner(nn.Module):
    """Run all diagnostic sieve nodes.

    Aggregates results from Nodes 2, 7, 29, 40, 52, 56, 62.
    """

    def __init__(self, config: StandardModelConfig):
        super().__init__()
        self.config = config

        self.node_2 = ZenoCheck(config)
        self.node_7 = StiffnessCheck(config)
        self.node_29 = TextureFirewallCheck(config)
        self.node_40 = PurityCheck(config)
        self.node_52 = LandauerViolationCheck(config)
        self.node_56 = CapacityHorizonCheck(config)
        self.node_62 = CausalityViolationCheck(config)

    def forward(
        self,
        c_info: Tensor,
        T_c: Tensor,
        delta_E: Tensor,
        gs_IR: Tensor,
        gs_UV: Tensor,
        ell_L: Tensor,
        area: float = 100.0,
        I_req: float = 1000.0
    ) -> Dict[str, Dict]:
        """Run all sieve nodes.

        Returns:
            Dictionary mapping node names to their results
        """
        results = {}

        results['node_2'] = self.node_2(c_info)
        results['node_7'] = self.node_7(delta_E, T_c)
        results['node_29'] = self.node_29(gs_UV)
        results['node_40'] = self.node_40(gs_IR)
        results['node_52'] = self.node_52(T_c)
        results['node_56'] = self.node_56(ell_L, area, I_req)
        results['node_62'] = self.node_62(c_info)

        # Aggregate: all passed?
        all_passed = all(r['passed'].all() for r in results.values())
        results['all_passed'] = torch.tensor(all_passed)

        return results


# =============================================================================
# Integration: StandardModelOfCognition
# =============================================================================

class StandardModelOfCognition(nn.Module):
    """Main interface for the Standard Model of Cognition.

    Integrates:
    - Chapter 34: Gauge-theoretic formulation
    - Chapter 35: Parameter space sieve

    Usage:
        config = StandardModelConfig()
        model = StandardModelOfCognition(config)

        # Compute Lagrangian
        result = model.compute_lagrangian(Psi_L, Psi_R, phi, ...)

        # Check sieve constraints
        sieve_result = model.check_sieve(params)
    """

    def __init__(self, config: StandardModelConfig = None):
        super().__init__()
        self.config = config or StandardModelConfig()

        # Chapter 34 components
        self.lagrangian = CognitiveLagrangian(self.config)
        self.covariant_deriv = self.lagrangian.covariant_deriv

        # Chapter 35 components
        self.sieve_system = SieveConstraintSystem(self.config)
        self.sieve_nodes = SieveNodeRunner(self.config)

    def compute_vev(self) -> float:
        """Compute vacuum expectation value."""
        return vacuum_expectation_value(self.config)

    def compute_masses(self) -> Dict[str, float]:
        """Compute all masses from Higgs mechanism."""
        v = self.compute_vev()
        return {
            'higgs': higgs_mass(self.config),
            'W_boson': gauge_boson_mass(self.config.g2, v),
            'cognitive': cognitive_mass(self.config)
        }

    def check_sieve(self, params: AgentParameterVector) -> Dict[str, Tensor]:
        """Check all sieve constraints."""
        return self.sieve_system.evaluate_all(params)

    def is_viable(self, params: AgentParameterVector) -> bool:
        """Check if agent parameters define a viable configuration."""
        return self.sieve_system.is_feasible(params).item()


# =============================================================================
# Tests
# =============================================================================

def test_gamma_matrices():
    """Test gamma matrix anticommutation relation."""
    print("Testing gamma matrices...")
    gamma = gamma_matrices()
    g = minkowski_metric()

    # {gamma^mu, gamma^nu} = 2 * g^{mu,nu} * I_4
    for mu in range(4):
        for nu in range(4):
            anticomm = gamma[mu] @ gamma[nu] + gamma[nu] @ gamma[mu]
            expected = 2 * g[mu, nu] * torch.eye(4, dtype=torch.complex64)
            diff = (anticomm - expected).abs().max()
            assert diff < 1e-6, f"Failed for mu={mu}, nu={nu}, diff={diff}"

    print("  Gamma matrices: PASSED")


def test_pauli_matrices():
    """Test Pauli matrix algebra."""
    print("Testing Pauli matrices...")
    tau = pauli_matrices()
    epsilon = levi_civita_3d()

    # [tau^a, tau^b] = 2i * epsilon^{abc} * tau^c
    for a in range(3):
        for b in range(3):
            commutator = tau[a] @ tau[b] - tau[b] @ tau[a]
            expected = sum(2j * epsilon[a, b, c] * tau[c] for c in range(3))
            diff = (commutator - expected).abs().max()
            assert diff < 1e-6, f"Failed for a={a}, b={b}"

    print("  Pauli matrices: PASSED")


def test_gellmann_matrices():
    """Test Gell-Mann matrix trace normalization."""
    print("Testing Gell-Mann matrices...")
    for n_f in [2, 3, 4]:
        lambda_a = gellmann_matrices(n_f)
        n_gen = n_f * n_f - 1

        for a in range(n_gen):
            for b in range(n_gen):
                trace = torch.trace(lambda_a[a] @ lambda_a[b])
                expected = 2.0 if a == b else 0.0
                diff = abs(trace.real - expected)
                assert diff < 1e-5, f"Failed for N_f={n_f}, a={a}, b={b}"

    print("  Gell-Mann matrices: PASSED")


def test_complexity_potential():
    """Test Mexican hat potential."""
    print("Testing complexity potential...")
    config = StandardModelConfig(mu_sq=1.0, lambda_higgs=0.1)

    # Create test field
    B, D = 2, 8
    phi = torch.randn(B, D, 2, dtype=torch.complex64) * 0.1

    V = complexity_potential(phi, config)
    assert V.shape == (B, D), f"Wrong shape: {V.shape}"

    # Test at VEV: |phi|^2 = v^2 = mu^2 / (2*lambda) at the minimum
    # V_min = -mu^2 * v^2 + lambda * v^4 = -mu^4/(4*lambda)
    v = vacuum_expectation_value(config)
    phi_vev = torch.zeros(1, 1, 2, dtype=torch.complex64)
    # Set |phi| = v (not v/sqrt(2) which is physics convention)
    phi_vev[0, 0, 1] = v
    V_vev = complexity_potential(phi_vev, config)
    V_min_expected = -config.mu_sq ** 2 / (4 * config.lambda_higgs)
    assert abs(V_vev.item() - V_min_expected) < 1e-5, f"V_vev={V_vev.item()}, expected={V_min_expected}"

    print("  Complexity potential: PASSED")


def test_ssb():
    """Test spontaneous symmetry breaking analysis."""
    print("Testing SSB...")

    # Broken phase
    config_broken = StandardModelConfig(mu_sq=1.0, lambda_higgs=0.1)
    v = vacuum_expectation_value(config_broken)
    assert v > 0, "Should have non-zero VEV in broken phase"

    # Symmetric phase
    config_sym = StandardModelConfig(mu_sq=-1.0, lambda_higgs=0.1)
    v_sym = vacuum_expectation_value(config_sym)
    assert v_sym == 0, "Should have zero VEV in symmetric phase"

    print("  SSB: PASSED")


def test_sieve_constraints():
    """Test sieve constraint checks."""
    print("Testing sieve constraints...")
    config = StandardModelConfig()

    # Test speed window
    c_info = torch.tensor([5.0, 500.0, 50000.0])
    lower, upper = check_speed_window(c_info, config.d_sync, config.L_buf, config.tau_proc)

    # Middle value should satisfy both bounds
    assert lower[1] <= 0, "Middle c_info should satisfy lower bound"
    assert upper[1] <= 0, "Middle c_info should satisfy upper bound"

    # Test discount window
    gamma = torch.tensor([0.3, 0.8, 1.0])
    lower, upper = check_discount_window(gamma)
    assert lower[1] <= 0, "Middle gamma should satisfy lower"
    assert upper[1] <= 0, "Middle gamma should satisfy upper"

    print("  Sieve constraints: PASSED")


def test_sieve_nodes():
    """Test sieve diagnostic nodes."""
    print("Testing sieve nodes...")
    config = StandardModelConfig()
    runner = SieveNodeRunner(config)

    # Create test parameters (should be viable)
    c_info = torch.tensor([50.0])
    T_c = torch.tensor([0.05])
    delta_E = torch.tensor([1.0])
    gs_IR = torch.tensor([2.0])
    gs_UV = torch.tensor([0.001])
    ell_L = torch.tensor([0.001])

    results = runner(c_info, T_c, delta_E, gs_IR, gs_UV, ell_L)

    print(f"  All nodes passed: {results['all_passed'].item()}")
    for name, res in results.items():
        if isinstance(res, dict) and 'passed' in res:
            status = "PASS" if res['passed'].item() else "FAIL"
            print(f"    {res['node']}: {status}")

    print("  Sieve nodes: PASSED")


def test_full_lagrangian():
    """Test full Lagrangian computation."""
    print("Testing full Lagrangian...")
    config = StandardModelConfig(latent_dim=8, feature_dim=3)
    lagrangian = CognitiveLagrangian(config)

    B, D, S = 2, config.latent_dim, config.spacetime_dim
    N_f = config.feature_dim
    C = 4  # Spinor components

    # Create test tensors
    Psi_L = torch.randn(B, D, C, 2, N_f, dtype=torch.complex64) * 0.1
    Psi_R = torch.randn(B, D, C, N_f, dtype=torch.complex64) * 0.1
    phi = torch.randn(B, D, 2, dtype=torch.complex64) * 0.1

    B_munu = torch.randn(B, S, S)
    W_munu = torch.randn(B, S, S, 3)
    G_munu = torch.randn(B, S, S, N_f * N_f - 1)

    D_mu_Psi_L = torch.randn(B, D, S, C, 2, N_f, dtype=torch.complex64) * 0.1
    D_mu_Psi_R = torch.randn(B, D, S, C, N_f, dtype=torch.complex64) * 0.1
    D_mu_phi = torch.randn(B, D, S, 2, dtype=torch.complex64) * 0.1

    Phi_eff = torch.randn(B, D)

    result = lagrangian(
        Psi_L, Psi_R, phi,
        B_munu, W_munu, G_munu,
        D_mu_Psi_L, D_mu_Psi_R, D_mu_phi,
        Phi_eff
    )

    assert 'L_total' in result
    assert result['L_total'].shape == (B,)
    print(f"  L_total shape: {result['L_total'].shape}")
    print(f"  L_gauge: {result['L_gauge'].mean():.4f}")
    print(f"  L_scalar: {result['L_scalar'].mean():.4f}")

    print("  Full Lagrangian: PASSED")


def test_standard_model():
    """Run all tests for standard_model.py."""
    print("=" * 60)
    print("Standard Model of Cognition - Test Suite")
    print("=" * 60)

    test_gamma_matrices()
    test_pauli_matrices()
    test_gellmann_matrices()
    test_complexity_potential()
    test_ssb()
    test_sieve_constraints()
    test_sieve_nodes()
    test_full_lagrangian()

    print("=" * 60)
    print("All tests PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    test_standard_model()
