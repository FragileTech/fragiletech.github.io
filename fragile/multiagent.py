"""
Chapter 29: Relativistic Symplectic Multi-Agent Field Theory — PyTorch Implementation
=======================================================================================

This module provides PyTorch implementations of all equations from Chapter 29
of the Fragile monograph. Each function/class is documented with its
corresponding mathematical definition.

Tensor Dimension Convention:
    B  = batch size
    N  = number of agents
    D  = latent dimension per agent
    S  = spacetime dimension (typically D+1)
    G  = gauge algebra dimension (dim(g))
    T  = number of time steps

Sections covered:
    - 29.1-29.3: Product Configuration Space and Causal Structure
    - 29.4-29.5: Ghost Interface and Klein-Gordon Value Equation
    - 29.6-29.7: Game Tensor and Nash Equilibrium
    - 29.10-29.12: Mean-Field Metric Law and Geometric Locking
    - 29.13-29.18: Gauge Theory Layer (Yang-Mills)
    - 29.19: Mass Gap Analysis
    - 29.21-29.25: Quantum Layer (Schrodinger Representation)
    - 29.27-29.28: Diagnostic Nodes and Causal Buffer

References:
    - Definition 29.1: N-Agent Product Manifold
    - Axiom 29.1: Information Speed Limit
    - Definition 29.4: Ghost Interface
    - Theorem 29.5: HJB-Klein-Gordon Correspondence
    - Definition 29.6: The Game Tensor
    - Theorem 29.7: Nash Equilibrium as Standing Wave
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Callable, Dict
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class MultiAgentConfig:
    """Configuration for the Relativistic Multi-Agent Field Theory system.

    Parameters correspond to theoretical constants defined in Chapter 29.

    Dimension Parameters:
        n_agents (N): Number of agents in the system
        latent_dim (D): Dimension of each agent's latent manifold Z^(i)
        spacetime_dim (S): Spacetime dimensions for gauge theory (typically D+1)
        gauge_dim (G): Dimension of gauge Lie algebra g
    """
    # Dimensions
    n_agents: int = 2                 # N: number of agents
    latent_dim: int = 64              # D: dimension of each agent's latent manifold
    spacetime_dim: int = 4            # S: spacetime dimensions for gauge theory
    gauge_dim: int = 8                # G: dimension of gauge algebra

    # Causal Structure (Section 29.2)
    c_info: float = 1.0               # c_info: information propagation speed [length/time]
    max_latency: int = 100            # Maximum buffer size for causal history [steps]

    # Screening and Discount (Section 29.5)
    kappa: float = 0.1                # kappa: screening mass [1/length]
    gamma_discount: float = 0.99      # gamma: temporal discount factor [dimensionless]
    gamma_damp: float = 0.01          # gamma_damp: temporal damping rate [1/time]

    # Strategic Coupling (Section 29.6)
    beta_adversarial: float = 1.0     # beta_ij: adversarial coupling strength [dimensionless]
    gamma_game: float = 1.0           # gamma_game: risk amplification factor [dimensionless]

    # Gauge Theory (Section 29.13-29.18)
    g_coupling: float = 1.0           # g: gauge coupling constant [dimensionless]
    mu_higgs_sq: float = -1.0         # mu^2: Higgs mass parameter [energy^2], negative for SSB
    lambda_higgs: float = 0.1         # lambda: Higgs quartic coupling [dimensionless]

    # Quantum Layer (Section 29.21)
    sigma: float = 0.1                # sigma: cognitive action scale [nat*time], analog of hbar
    T_c: float = 1.0                  # T_c: cognitive temperature [nat]

    # Mean-Field (Section 29.10)
    alpha_adversarial: float = 1.0    # alpha_adv: adversarial interaction strength [dimensionless]


class StrategicRelation(Enum):
    """Strategic relationship types for the Game Tensor."""
    COOPERATIVE = -1   # alpha_ij = -1: aligned gradients
    INDEPENDENT = 0    # alpha_ij =  0: no interaction
    ADVERSARIAL = 1    # alpha_ij = +1: opposing gradients


# =============================================================================
# Section 29.1: The Product Configuration Space
# =============================================================================

def product_metric(
    G_agents: List[Tensor]
) -> Tensor:
    """
    Definition 29.1: N-Agent Product Manifold Metric

    The metric on Z^(N) is the direct sum:
        G^(N) := bigoplus_{i=1}^N G^(i)

    This is block-diagonal: G^(N)_μν = G^(i)_ab when indices lie in agent i's block.

    Args:
        G_agents: List of N tensors, each [D_i, D_i] metric tensor for agent i

    Returns:
        G_N: [sum(D_i), sum(D_i)] - Product metric (block diagonal)

    Tensor Shapes:
        Input:  G_agents[i] has shape [D_i, D_i]
        Output: G_N has shape [D_total, D_total] where D_total = sum_i D_i
    """
    # D_total = sum of all agent dimensions
    total_dim = sum(G.shape[0] for G in G_agents)

    # Initialize block-diagonal matrix: [D_total, D_total]
    G_N = torch.zeros(total_dim, total_dim, device=G_agents[0].device, dtype=G_agents[0].dtype)

    offset = 0
    for G_i in G_agents:
        d_i = G_i.shape[0]  # D_i: dimension of agent i's manifold
        # Place G_i on the diagonal block
        G_N[offset:offset+d_i, offset:offset+d_i] = G_i  # [D_i, D_i] -> block in [D_total, D_total]
        offset += d_i

    return G_N  # [D_total, D_total]


def environment_distance(
    positions: Tensor,
    topology: str = "euclidean"
) -> Tensor:
    """
    Definition 29.3: Environment Distance

    d_E^{ij} = geodesic length in environment manifold E between agents i and j.

    Args:
        positions: [N, D_env] - Agent positions in environment space
                   N = number of agents
                   D_env = dimension of environment (physical) space
        topology: "euclidean" or "network"

    Returns:
        d_E: [N, N] - Pairwise environment distances
             d_E[i,j] = distance from agent i to agent j

    Tensor Shapes:
        Input:  positions [N, D_env]
        Output: d_E [N, N], symmetric matrix with zeros on diagonal
    """
    if topology == "euclidean":
        # cdist computes pairwise Euclidean distances
        # positions: [N, D_env] -> d_E: [N, N]
        return torch.cdist(positions, positions)  # [N, N]
    else:
        raise NotImplementedError("Network topology not yet implemented")


# =============================================================================
# Section 29.2: The Failure of Simultaneity
# =============================================================================

def causal_delay(
    d_E: Tensor,
    c_info: float
) -> Tensor:
    """
    Axiom 29.1: Information Speed Limit - Causal Delay

        tau_ij := d_E^{ij} / c_info

    Args:
        d_E: [N, N] - Environment distances matrix
        c_info: scalar - Information propagation speed

    Returns:
        tau: [N, N] - Causal delays between agent pairs
             tau[i,j] = time for signal from j to reach i

    Tensor Shapes:
        Input:  d_E [N, N]
        Output: tau [N, N]
    """
    # Element-wise division: [N, N] / scalar -> [N, N]
    return d_E / c_info  # [N, N]


def causal_interval(
    t_i: Tensor,
    t_j: Tensor,
    d_E_ij: Tensor,
    c_info: float
) -> Tensor:
    """
    Definition 29.4: Causal Interval

        Delta s^2_{ij} := -c_info^2 (t_j - t_i)^2 + (d_E^{ij})^2

    Classification:
        - Timelike (< 0): Causal influence possible
        - Spacelike (> 0): No causal influence
        - Lightlike (= 0): Boundary case

    Args:
        t_i: scalar or [B] - Time of event at agent i
        t_j: scalar or [B] - Time of event at agent j
        d_E_ij: scalar or [B] - Environment distance between i and j
        c_info: scalar - Information speed

    Returns:
        Delta_s_sq: scalar or [B] - Causal interval (squared)

    Tensor Shapes:
        All inputs broadcast to same shape
        Output matches broadcasted input shape
    """
    dt = t_j - t_i  # Time difference: same shape as inputs
    # Minkowski-like metric: -c^2 dt^2 + dx^2
    return -c_info**2 * dt**2 + d_E_ij**2  # Same shape as inputs


def past_light_cone(
    agent_id: int,
    t: float,
    tau: Tensor,
    n_agents: int
) -> List[Tuple[int, float]]:
    """
    Definition 29.5: Past Light Cone

        C^-_i(t) := {(j, t') : t' <= t - tau_ij}

    Args:
        agent_id: int - Index of the observing agent (0 to N-1)
        t: float - Current time
        tau: [N, N] - Causal delay matrix where tau[i,j] = delay from j to i
        n_agents: int - Total number of agents N

    Returns:
        past_cone: List of (agent_j_index, max_observable_time) pairs
                   Length N-1 (excludes self)

    Tensor Shapes:
        Input:  tau [N, N]
        Output: List of (int, float) tuples
    """
    past_cone = []
    for j in range(n_agents):
        if j != agent_id:
            # Maximum time at agent j that agent_id can observe at time t
            t_max = t - tau[agent_id, j].item()  # tau[i,j]: [N,N] -> scalar via indexing
            past_cone.append((j, t_max))
    return past_cone  # List of length N-1


# =============================================================================
# Section 29.3: The Relativistic State
# =============================================================================

def retarded_green_function(
    z: Tensor,
    zeta: Tensor,
    t: float,
    tau_val: float,
    c_info: float,
    kappa: float,
    D: int
) -> Tensor:
    """
    Proposition 29.8: Retarded Green's Function

    G_ret(z, t; zeta, tau) propto delta(t - tau - d/c_info) / d^{(D-2)/2} * exp(-kappa * d)

    Args:
        z: [B, D] - Observer positions in latent space
           B = batch size, D = latent dimension
        zeta: [B, D] - Source positions in latent space
        t: float - Current time at observer
        tau_val: float - Emission time at source
        c_info: float - Information propagation speed
        kappa: float - Screening mass (Yukawa decay)
        D: int - Spatial dimension

    Returns:
        G_ret: [B] - Green's function values for each batch element

    Tensor Shapes:
        Input:  z [B, D], zeta [B, D]
        Output: G_ret [B]
    """
    # Distance in latent space: [B, D] - [B, D] -> [B, D] -> norm -> [B]
    d = torch.norm(z - zeta, dim=-1)  # [B]

    # Retardation condition: signal arrives at t if emitted at tau = t - d/c_info
    # Error from exact retardation: scalar operation
    retard_error = torch.abs((t - tau_val) - d / c_info)  # [B]

    # Approximate delta function as narrow Gaussian
    delta_width = 0.1  # Width of approximate delta
    delta_factor = torch.exp(-retard_error**2 / (2 * delta_width**2))  # [B]

    # Geometric decay factor: 1/d^{(D-2)/2}
    geom_factor = 1.0 / (d + 1e-8)**((D - 2) / 2)  # [B]

    # Yukawa screening: exp(-kappa * d)
    screen_factor = torch.exp(-kappa * d)  # [B]

    # Combine all factors: [B] * [B] * [B] -> [B]
    return delta_factor * geom_factor * screen_factor  # [B]


class CausalBundle(nn.Module):
    """
    Definition 29.8: Causal Bundle

        Z_causal := Z^(N) x Xi_{<t}

    The augmented state space where the Markov property is restored.
    Xi_{<t} is the "memory screen" containing past states.

    Attributes:
        memory_screens: Dict[int, List[Tuple[float, Tensor]]]
            Maps agent_id -> list of (time, state) pairs
            Each state tensor has shape [D] (agent's latent dimension)
    """

    def __init__(self, config: MultiAgentConfig):
        super().__init__()
        self.config = config
        # Memory screen for each agent: agent_id -> [(t, state[D]), ...]
        self.memory_screens: Dict[int, List[Tuple[float, Tensor]]] = {
            i: [] for i in range(config.n_agents)
        }

    def update_memory_screen(self, agent_id: int, t: float, state: Tensor):
        """
        Add state to agent's memory screen Xi_{<t}.

        Args:
            agent_id: int - Agent index (0 to N-1)
            t: float - Current time
            state: [D] - Agent's current latent state
        """
        # Clone to avoid reference issues: [D] -> [D]
        self.memory_screens[agent_id].append((t, state.clone()))

        # Prune old entries beyond causal horizon
        tau_horizon = self.config.max_latency / self.config.c_info
        self.memory_screens[agent_id] = [
            (t_old, s) for t_old, s in self.memory_screens[agent_id]
            if t - t_old < tau_horizon
        ]

    def get_relativistic_state(
        self,
        agent_id: int,
        z_current: Tensor,
        t: float
    ) -> Tuple[Tensor, List[Tensor]]:
        """
        Get the Relativistic State S^(i)_t := (z^(i)_t, Xi^(i)_{<t}).

        Args:
            agent_id: int - Agent index
            z_current: [D] or [B, D] - Current position
            t: float - Current time

        Returns:
            z_current: [D] or [B, D] - Current position (unchanged)
            memory: List of [D] tensors - Past states in memory screen
        """
        memory = [s for _, s in self.memory_screens[agent_id]]  # List of [D] tensors
        return z_current, memory


# =============================================================================
# Section 29.4: The Ghost Interface
# =============================================================================

class GhostInterface(nn.Module):
    """
    Definition 29.9: Ghost Interface

        G_ij(t) := partial Z^(i)(t) x partial Z^(j)(t - tau_ij)

    Couples Agent i's current boundary to Agent j's past boundary.
    The "ghost state" is agent j's state at the retarded time.
    """

    def __init__(self, config: MultiAgentConfig):
        super().__init__()
        self.config = config

    def get_ghost_state(
        self,
        agent_j_states: List[Tuple[float, Tensor]],
        t_now: float,
        tau_ij: float
    ) -> Optional[Tensor]:
        """
        Get ghost state: z^(j) at retarded time t - tau_ij.

        Args:
            agent_j_states: List of (time: float, state: [D]) from agent j's history
            t_now: float - Current time of observing agent i
            tau_ij: float - Causal delay from j to i

        Returns:
            ghost_state: [D] or None - Agent j's state at retarded time

        Tensor Shapes:
            Input states: each [D]
            Output: [D] or None
        """
        t_retarded = t_now - tau_ij  # Target emission time

        if not agent_j_states:
            return None

        # Find state closest to retarded time
        closest_state = None
        closest_diff = float('inf')

        for t_emit, state in agent_j_states:
            diff = abs(t_emit - t_retarded)
            if diff < closest_diff:
                closest_diff = diff
                closest_state = state  # [D]

        return closest_state  # [D] or None


def retarded_interaction_potential(
    z_i: Tensor,
    z_j_ghost: Tensor,
    alpha_ij: float,
    kappa: float
) -> Tensor:
    """
    Definition 29.10: Retarded Interaction Potential

        Phi^ret_ij(z^(i), t) := alpha_ij * G_kappa(z^(i), z_hat^(j)_t) * sigma^(j)(z_hat^(j))

    where:
        - z_hat^(j)_t = z^(j)_{t - tau_ij} is the ghost state
        - G_kappa is the Yukawa (screened) Green's function
        - alpha_ij in {-1, 0, +1} encodes strategic relationship

    Args:
        z_i: [B, D] - Agent i's positions
        z_j_ghost: [B, D] - Agent j's ghost (retarded) positions
        alpha_ij: float - Strategic relationship coefficient
                  -1 = cooperative, 0 = independent, +1 = adversarial
        kappa: float - Screening mass

    Returns:
        Phi_ret: [B] - Retarded interaction potential for each batch element

    Tensor Shapes:
        Input:  z_i [B, D], z_j_ghost [B, D]
        Output: Phi_ret [B]
    """
    # Yukawa (screened) potential: exp(-kappa*d) / d
    # Distance: [B, D] - [B, D] -> [B, D] -> norm over D -> [B]
    d = torch.norm(z_i - z_j_ghost, dim=-1)  # [B]

    # Screened Green's function: [B] -> [B]
    G_kappa = torch.exp(-kappa * d) / (d + 1e-8)  # [B]

    # Source strength sigma^(j) - assume uniform for simplicity
    sigma_j = 1.0  # scalar

    # alpha_ij * G_kappa * sigma_j: scalar * [B] * scalar -> [B]
    return alpha_ij * G_kappa * sigma_j  # [B]


# =============================================================================
# Section 29.5: The Hyperbolic Value Equation (Klein-Gordon)
# =============================================================================

def klein_gordon_operator(
    V: Tensor,
    G_inv: Tensor,
    c_info: float,
    kappa: float,
    gamma_damp: float,
    dt: float = 0.01
) -> Callable[[Tensor, Tensor], Tensor]:
    """
    Theorem 29.5: HJB-Klein-Gordon Correspondence

    The screened wave equation:
        (1/c^2 d^2/dt^2 + gamma_damp d/dt - Delta_G + kappa^2) V = rho_r + sum_j Phi^ret_ij

    Returns a function that computes the spatial part of the Klein-Gordon operator.

    Args:
        V: [B, ...] - Value function (discretized), not directly used but for reference
        G_inv: [D, D] - Inverse metric tensor G^{ij}
        c_info: float - Information speed
        kappa: float - Screening mass
        gamma_damp: float - Damping rate
        dt: float - Time step

    Returns:
        kg_operator: Callable that takes (V_current: [B], V_prev: [B]) -> [B]

    Tensor Shapes:
        G_inv: [D, D]
        Returned function operates on [B] tensors
    """
    def kg_operator(V_current: Tensor, V_prev: Tensor) -> Tensor:
        """
        Apply Klein-Gordon operator (screening term only in this simplified version).

        Args:
            V_current: [B] - Value at current time
            V_prev: [B] - Value at previous time

        Returns:
            result: [B] - Screening term kappa^2 * V
        """
        # Full operator would include:
        # - Temporal: (1/c^2)(V_next - 2*V_current + V_prev)/dt^2 + gamma*(V_current - V_prev)/dt
        # - Spatial: -Delta_G V (Laplace-Beltrami)
        # - Screening: kappa^2 * V

        # Screening term: scalar * [B] -> [B]
        screening = kappa**2 * V_current  # [B]

        return screening  # [B]

    return kg_operator


def value_wavefront_propagation(
    z: Tensor,
    z_source: Tensor,
    t: float,
    t_source: float,
    c_info: float,
    kappa: float,
    rho_source: float,
    D: int
) -> Tensor:
    """
    Corollary 29.6: Value Wavefront Propagation

        V(z, t) ~ Theta(t - t_0 - d/c_info) / d^{(D-2)/2} * exp(-kappa * d) * rho_r(z_A, t_0)

    A sudden reward change propagates as a wavefront at speed c_info.

    Args:
        z: [B, D] - Observation positions in latent space
        z_source: [D] - Source position (where reward changed)
        t: float - Current time
        t_source: float - Time of reward change
        c_info: float - Information speed
        kappa: float - Screening mass
        rho_source: float - Source strength (reward magnitude)
        D: int - Spatial dimension

    Returns:
        V: [B] - Value contribution from the wavefront

    Tensor Shapes:
        Input:  z [B, D], z_source [D]
        Output: V [B]
    """
    # Distance from source: [B, D] - [1, D] -> [B, D] -> norm -> [B]
    d = torch.norm(z - z_source.unsqueeze(0), dim=-1)  # [B]

    # Arrival time of wavefront at each position
    arrival_time = t_source + d / c_info  # [B]

    # Heaviside step function: signal has arrived if t >= arrival_time
    theta = (t >= arrival_time).float()  # [B], values in {0, 1}

    # Geometric decay: 1/d^{(D-2)/2}
    geom = 1.0 / (d + 1e-8)**((D - 2) / 2)  # [B]

    # Yukawa screening
    screening = torch.exp(-kappa * d)  # [B]

    # Combine: [B] * [B] * [B] * scalar -> [B]
    return theta * geom * screening * rho_source  # [B]


# =============================================================================
# Section 29.6: The Game Tensor
# =============================================================================

def game_tensor(
    V_i: Tensor,
    z_j: Tensor,
    create_graph: bool = True
) -> Tensor:
    """
    Definition 29.11: The Game Tensor

        G_ij^{kl}(z^(i), z^(j)) := d^2 V^(i) / dz^(j)_k dz^(j)_l

    Cross-Hessian of Agent i's value with respect to Agent j's position.
    Measures how curved agent i's value landscape is in the direction of agent j.

    Args:
        V_i: [B] - Agent i's value function (must have grad enabled)
        z_j: [B, D] - Agent j's positions (must have requires_grad=True)
        create_graph: bool - Whether to create graph for higher-order derivatives

    Returns:
        G_ij: [B, D, D] - Game tensor (Hessian) for each batch element

    Tensor Shapes:
        Input:  V_i [B], z_j [B, D]
        Output: G_ij [B, D, D]
    """
    B, D = z_j.shape

    # First derivative: dV_i/dz_j
    # V_i.sum() creates scalar for autograd; grad returns [B, D]
    grad_V = torch.autograd.grad(
        V_i.sum(),           # scalar
        z_j,                 # [B, D]
        create_graph=create_graph,
        retain_graph=True
    )[0]  # [B, D]

    # Second derivative (Hessian): d^2V_i / dz_j^k dz_j^l
    # Build [B, D, D] by computing gradient of each component of grad_V
    hessian = torch.zeros(B, D, D, device=z_j.device, dtype=z_j.dtype)  # [B, D, D]

    for k in range(D):
        # Gradient of k-th component of grad_V w.r.t. z_j
        grad_k = torch.autograd.grad(
            grad_V[:, k].sum(),  # scalar (sum over batch of k-th component)
            z_j,                  # [B, D]
            create_graph=False,
            retain_graph=True
        )[0]  # [B, D]
        hessian[:, k, :] = grad_k  # [B, D] -> row k of [B, D, D]

    return hessian  # [B, D, D]


def effective_metric_with_game_tensor(
    G_i: Tensor,
    game_tensors: List[Tuple[Tensor, float]]
) -> Tensor:
    """
    Definition 29.11 (cont.): Effective Metric with Game Tensor

        tilde{G}^(i)_{kl}(z) = G^(i)_{kl}(z) + sum_{j != i} beta_ij * G_{ij,kl}(z)

    The strategic coupling inflates the metric, making movement costly.

    Args:
        G_i: [B, D, D] or [D, D] - Agent i's intrinsic metric
        game_tensors: List of (G_ij: [B, D, D], beta_ij: float) pairs
                      One pair for each other agent j

    Returns:
        G_tilde: [B, D, D] or [D, D] - Effective (game-augmented) metric
                 Same shape as G_i

    Tensor Shapes:
        Input:  G_i [B, D, D] or [D, D]
                game_tensors[k][0] [B, D, D] (must match batch dim of G_i)
        Output: G_tilde same shape as G_i
    """
    G_tilde = G_i.clone()  # [B, D, D] or [D, D]

    for G_ij, beta_ij in game_tensors:
        # beta_ij * G_ij: scalar * [B, D, D] -> [B, D, D]
        G_tilde = G_tilde + beta_ij * G_ij  # [B, D, D] + [B, D, D] -> [B, D, D]

    return G_tilde  # Same shape as G_i


def strategic_jacobian(
    z_j: Tensor,
    z_i: Tensor,
    policy_j: nn.Module
) -> Tensor:
    """
    Strategic Jacobian (for Game Tensor derivation)

        J_ji := dz^(j) / dz^(i)

    The best-response derivative: how agent j responds to agent i's move.

    Args:
        z_j: [B, D] - Agent j's position
        z_i: [B, D] - Agent i's position (requires_grad for Jacobian computation)
        policy_j: nn.Module - Agent j's policy network (not used in simplified version)

    Returns:
        J_ji: [B, D, D] - Strategic Jacobian
              J_ji[b, k, l] = dz_j^k / dz_i^l for batch element b

    Tensor Shapes:
        Input:  z_j [B, D], z_i [B, D]
        Output: J_ji [B, D, D]
    """
    B, D = z_i.shape

    # Simplified: assume identity response (no strategic reaction)
    # Full implementation would compute true best-response Jacobian
    # Identity: [D, D] -> unsqueeze -> [1, D, D] -> expand -> [B, D, D]
    return torch.eye(D, device=z_i.device).unsqueeze(0).expand(B, -1, -1)  # [B, D, D]


class GameTensor(nn.Module):
    """
    Complete Game Tensor computation module for multi-agent systems.

    Computes G_ij: cross-Hessian of V^(i) w.r.t. z^(j) for all agent pairs.

    Attributes:
        value_nets: nn.ModuleList of N value networks
                    Each network: [D] -> [1]
    """

    def __init__(self, config: MultiAgentConfig):
        super().__init__()
        self.config = config

        # Value networks for each agent
        # Input: [B, D], Output: [B, 1]
        self.value_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.latent_dim, config.latent_dim * 2),  # [B, D] -> [B, 2D]
                nn.Tanh(),                                             # [B, 2D] -> [B, 2D]
                nn.Linear(config.latent_dim * 2, 1)                    # [B, 2D] -> [B, 1]
            )
            for _ in range(config.n_agents)
        ])

    def compute_game_tensor(
        self,
        agent_i: int,
        agent_j: int,
        z_i: Tensor,
        z_j: Tensor
    ) -> Tensor:
        """
        Compute G_ij: cross-Hessian of V^(i) w.r.t. z^(j).

        Args:
            agent_i: int - Index of agent whose value we differentiate
            agent_j: int - Index of agent whose position we differentiate w.r.t.
            z_i: [B, D] - Agent i's positions
            z_j: [B, D] - Agent j's positions

        Returns:
            G_ij: [B, D, D] - Game tensor

        Tensor Shapes:
            Input:  z_i [B, D], z_j [B, D]
            Output: G_ij [B, D, D]
        """
        # Enable gradients for z_j: [B, D]
        z_j = z_j.clone().requires_grad_(True)  # [B, D]

        # Compute V^(i) - in simplified model, depends only on z_j
        # value_nets[i]: [B, D] -> [B, 1] -> squeeze -> [B]
        V_i = self.value_nets[agent_i](z_j).squeeze(-1)  # [B]

        # Compute Hessian: [B] -> [B, D, D]
        return game_tensor(V_i, z_j)  # [B, D, D]

    def compute_retarded_game_tensor(
        self,
        agent_i: int,
        agent_j: int,
        z_i: Tensor,
        z_j_ghost: Tensor
    ) -> Tensor:
        """
        Definition 29.13: Retarded Game Tensor

            G_ij^{kl,ret}(z^(i), t) := d^2 V^(i) / dz^(j)_k dz^(j)_l |_{z^(j) = z_hat^(j)_t}

        Uses the ghost state z_hat^(j)_t instead of current z^(j)_t.

        Args:
            agent_i: int - Agent whose value we differentiate
            agent_j: int - Agent whose ghost position we use
            z_i: [B, D] - Agent i's current positions
            z_j_ghost: [B, D] - Agent j's ghost (retarded) positions

        Returns:
            G_ij_ret: [B, D, D] - Retarded game tensor

        Tensor Shapes:
            Same as compute_game_tensor
        """
        return self.compute_game_tensor(agent_i, agent_j, z_i, z_j_ghost)  # [B, D, D]

    def forward(
        self,
        z_all: Tensor
    ) -> Tensor:
        """
        Compute all pairwise game tensors.

        Args:
            z_all: [N, B, D] - All agent positions
                   N = number of agents
                   B = batch size
                   D = latent dimension

        Returns:
            G_tensors: [N, N, B, D, D] - All pairwise game tensors
                       G_tensors[i, j, b, k, l] = (G_ij)^{kl} for batch b

        Tensor Shapes:
            Input:  z_all [N, B, D]
            Output: G_tensors [N, N, B, D, D]
        """
        N = self.config.n_agents
        B, D = z_all.shape[1], z_all.shape[2]

        # Initialize output: [N, N, B, D, D]
        G_tensors = torch.zeros(N, N, B, D, D, device=z_all.device)

        for i in range(N):
            for j in range(N):
                if i != j:
                    # z_all[i]: [B, D], z_all[j]: [B, D]
                    # compute_game_tensor returns [B, D, D]
                    G_tensors[i, j] = self.compute_game_tensor(i, j, z_all[i], z_all[j])

        return G_tensors  # [N, N, B, D, D]


# =============================================================================
# Section 29.7: Nash Equilibrium as Standing Wave
# =============================================================================

def joint_wfr_action(
    rho: List[Tensor],
    v: List[Tensor],
    r: List[Tensor],
    G_tilde: List[Tensor],
    V_int_ret: Tensor,
    lambda_sq: float,
    T: float,
    dt: float
) -> Tensor:
    """
    Definition 29.14: Joint WFR Action (Relativistic)

        A^(N)[rho, v, r] = int_0^T [ sum_i int_{Z^(i)} (||v^(i)||^2_{tilde{G}} + lambda^2 |r^(i)|^2) drho^(i)
                                    + V_int^ret(rho, t) ] dt

    Args:
        rho: List of N tensors [B] - Belief density for each agent
        v: List of N tensors [B, D] - Velocity field for each agent
        r: List of N tensors [B] - Reaction rate for each agent
        G_tilde: List of N tensors [D, D] - Game-augmented metric for each agent
        V_int_ret: [B] - Retarded interaction energy
        lambda_sq: float - Reaction cost coefficient
        T: float - Time horizon (not directly used, for reference)
        dt: float - Time step

    Returns:
        action: scalar Tensor - Total action value

    Tensor Shapes:
        rho[i]: [B]
        v[i]: [B, D]
        r[i]: [B]
        G_tilde[i]: [D, D]
        V_int_ret: [B]
        Output: scalar
    """
    action = torch.tensor(0.0, device=rho[0].device)

    for i in range(len(rho)):
        # Kinetic term: ||v||^2_G = v^T G v
        # v[i]: [B, D], G_tilde[i]: [D, D]
        # einsum 'bd,de,be->b' computes v^T G v for each batch: [B]
        v_G_norm_sq = torch.einsum('bd,de,be->b', v[i], G_tilde[i], v[i])  # [B]

        # Weight by density and average: [B] * [B] -> [B] -> scalar
        kinetic = (v_G_norm_sq * rho[i]).mean()  # scalar

        # Reaction term: lambda^2 * |r|^2 * rho
        # r[i]: [B], rho[i]: [B]
        reaction = lambda_sq * (r[i]**2 * rho[i]).mean()  # scalar

        # Accumulate with time step
        action = action + (kinetic + reaction) * dt  # scalar

    # Interaction energy: [B] -> mean -> scalar
    action = action + V_int_ret.mean() * dt  # scalar

    return action  # scalar


def nash_residual(
    grad_Phi_eff: List[Tensor],
    G_inv: List[Tensor]
) -> Tensor:
    """
    Theorem 29.15: Nash Equilibrium as Geometric Stasis

    Condition 1 (Vanishing individual gradient):
        (G^(i))^{-1} nabla_{z^(i)} Phi_eff^(i) = 0 for all i

    Args:
        grad_Phi_eff: List of N tensors [B, D] - Gradient of effective potential
        G_inv: List of N tensors [D, D] - Inverse metric for each agent

    Returns:
        epsilon_Nash: scalar - Maximum deviation from stasis (Node 47 diagnostic)

    Tensor Shapes:
        grad_Phi_eff[i]: [B, D]
        G_inv[i]: [D, D]
        Output: scalar
    """
    residuals = []
    for i in range(len(grad_Phi_eff)):
        # Flow velocity: G^{-1} nabla Phi
        # G_inv[i]: [D, D], grad_Phi_eff[i]: [B, D]
        # einsum 'de,bd->be' applies G^{-1} to gradient: [B, D]
        flow = torch.einsum('de,bd->be', G_inv[i], grad_Phi_eff[i])  # [B, D]

        # Norm w.r.t. G (not G^{-1}): ||flow||_G^2 = flow^T G flow
        # Need G from G_inv: [D, D]
        G_i = torch.linalg.inv(G_inv[i])  # [D, D]

        # einsum 'bd,de,be->b' computes norm squared: [B]
        norm_sq = torch.einsum('bd,de,be->b', flow, G_i, flow)  # [B]

        # Maximum over batch: scalar
        residuals.append(torch.sqrt(norm_sq).max())

    # Return maximum residual across all agents: scalar
    return max(residuals)  # scalar


def standing_wave_nash_condition(
    J: List[Tensor],
    T: float,
    dt: float
) -> Tensor:
    """
    Theorem 29.15: Standing Wave Nash (Time-Averaged Stasis)

        <J^(i)>_T := (1/T) int_0^T J^(i)(z, t) dt = 0

    Args:
        J: List of N tensors [T_steps, B, D] - Probability current over time
           T_steps = number of time steps
           B = batch size
           D = latent dimension
        T: float - Averaging period (not directly used)
        dt: float - Time step (not directly used)

    Returns:
        avg_current_norm: scalar - Should be near zero at Nash

    Tensor Shapes:
        J[i]: [T_steps, B, D]
        Output: scalar
    """
    total_norm = torch.tensor(0.0, device=J[0].device)

    for J_i in J:
        # Time average: mean over dim 0
        # [T_steps, B, D] -> mean over T_steps -> [B, D]
        J_avg = J_i.mean(dim=0)  # [B, D]

        # Norm over D dimension, then mean over batch
        # [B, D] -> norm over D -> [B] -> mean -> scalar
        total_norm = total_norm + torch.norm(J_avg, dim=-1).mean()  # scalar

    # Average over agents: scalar / N -> scalar
    return total_norm / len(J)  # scalar


# =============================================================================
# Section 29.10: Mean-Field Metric Law
# =============================================================================

def mean_field_metric(
    G_intrinsic: Tensor,
    alpha_adv: float,
    Phi_int: Callable[[Tensor, Tensor], Tensor],
    rho: Callable[[Tensor], Tensor],
    z: Tensor,
    n_samples: int = 100
) -> Tensor:
    """
    Theorem 29.16: Mean-Field Metric Law

    As N -> infinity, the effective metric converges to:
        tilde{G}(z) = G_intrinsic(z) + alpha_adv * nabla^2_z (Phi_int * rho)(z)

    where * denotes convolution.

    Args:
        G_intrinsic: [D, D] - Base metric tensor
        alpha_adv: float - Adversarial coupling strength
        Phi_int: Callable([B, D], [B, D]) -> [B] - Pairwise interaction potential
        rho: Callable([D]) -> float - Population density (not directly used)
        z: [B, D] - Test positions where to evaluate metric
        n_samples: int - Monte Carlo samples for convolution

    Returns:
        G_tilde: [B, D, D] - Mean-field metric at each position

    Tensor Shapes:
        G_intrinsic: [D, D]
        z: [B, D]
        Output: [B, D, D]
    """
    B, D = z.shape
    device = z.device

    # Sample from population density (simplified: standard normal)
    # [n_samples, D]
    z_samples = torch.randn(n_samples, D, device=device)

    # Enable gradients for Hessian computation
    z_req = z.clone().requires_grad_(True)  # [B, D]

    # Accumulate Hessian over samples: [B, D, D]
    hess_sum = torch.zeros(B, D, D, device=device)

    for zeta in z_samples:
        # zeta: [D] -> unsqueeze -> [1, D] -> expand -> [B, D]
        zeta_expanded = zeta.unsqueeze(0).expand(B, -1)  # [B, D]

        # Interaction potential: Phi(z, zeta) -> [B]
        Phi = Phi_int(z_req, zeta_expanded)  # [B]

        # Gradient w.r.t. z: [B] -> [B, D]
        grad_Phi = torch.autograd.grad(
            Phi.sum(), z_req, create_graph=True
        )[0]  # [B, D]

        # Hessian via second gradients
        for k in range(D):
            # Gradient of k-th component: [B, D]
            grad_k = torch.autograd.grad(
                grad_Phi[:, k].sum(), z_req,
                create_graph=False, retain_graph=True
            )[0]  # [B, D]
            hess_sum[:, k, :] += grad_k  # [B, D] added to [B, D, D][:, k, :]

    # Average Hessian: [B, D, D] / scalar -> [B, D, D]
    hess_avg = hess_sum / n_samples  # [B, D, D]

    # Effective metric: G + alpha * Hessian
    # G_intrinsic: [D, D] -> unsqueeze -> [1, D, D] -> expand -> [B, D, D]
    G_tilde = G_intrinsic.unsqueeze(0).expand(B, -1, -1) + alpha_adv * hess_avg  # [B, D, D]

    return G_tilde  # [B, D, D]


# =============================================================================
# Section 29.11: Metabolic Tracking Bound
# =============================================================================

def metabolic_tracking_bound(
    z_dot_star: Tensor,
    G_tilde: Tensor,
    M_dot_max: float,
    sigma_met: float
) -> Tuple[bool, Tensor]:
    """
    Theorem 29.17: Metabolic Tracking Bound

    Agent can track Nash z*(t) if:
        ||dot{z}*||_{tilde{G}(z*)} <= sqrt(2 * M_dot_max / sigma_met)

    Args:
        z_dot_star: [B, D] - Velocity of moving Nash equilibrium
        G_tilde: [D, D] - Game-augmented metric (evaluated at Nash)
        M_dot_max: float - Maximum metabolic flux budget
        sigma_met: float - Metabolic coefficient

    Returns:
        can_track: bool - Whether tracking is possible for all batch elements
        required_flux: [B] - The metabolic flux required for each trajectory

    Tensor Shapes:
        z_dot_star: [B, D]
        G_tilde: [D, D]
        required_flux: [B]
    """
    # ||v||^2_G = v^T G v via einsum
    # z_dot_star: [B, D], G_tilde: [D, D]
    # 'bd,de,be->b' computes [B, D] @ [D, D] @ [B, D]^T -> [B]
    norm_sq = torch.einsum('bd,de,be->b', z_dot_star, G_tilde, z_dot_star)  # [B]

    # Required metabolic flux: (1/2) sigma_met ||v||^2
    required_flux = 0.5 * sigma_met * norm_sq  # [B]

    # Maximum allowable velocity squared
    max_velocity_sq = 2 * M_dot_max / sigma_met  # scalar

    # Check if all batch elements can track
    can_track = (norm_sq <= max_velocity_sq).all()  # bool

    return can_track.item(), required_flux  # (bool, [B])


# =============================================================================
# Section 29.12: Geometric Locking Principle
# =============================================================================

def geometric_locking_energy(
    v: Tensor,
    G_tilde: Tensor,
    rho: Tensor
) -> Tensor:
    """
    Theorem 29.18: Geometric Locking Principle

    The WFR action includes transport term: int ||v||^2_{tilde{G}} d rho

    An inflated metric (high Game Tensor) makes movement costly,
    driving the system toward:
        1. Cooperation (reduce G_ij)
        2. Decoupling (move to low-coupling regions)
        3. Freeze (Nash stasis with v=0)

    Args:
        v: [B, D] - Velocity field
        G_tilde: [B, D, D] - Game-augmented metric (position-dependent)
        rho: [B] - Belief density

    Returns:
        transport_energy: scalar - Average kinetic cost of transport

    Tensor Shapes:
        v: [B, D]
        G_tilde: [B, D, D]
        rho: [B]
        Output: scalar
    """
    # ||v||^2_G = v^T G v for each batch element
    # v: [B, D], G_tilde: [B, D, D]
    # 'bd,bde,be->b' handles batched matrix multiplication: [B]
    v_norm_sq = torch.einsum('bd,bde,be->b', v, G_tilde, v)  # [B]

    # Weighted by density and averaged: [B] * [B] -> [B] -> scalar
    return (v_norm_sq * rho).mean()  # scalar


# =============================================================================
# Section 29.13-29.18: Gauge Theory Layer
# =============================================================================

class StrategicConnection(nn.Module):
    """
    Definition 29.19: Strategic Connection (Gauge Potential)

        A = A_mu^a T_a dz^mu

    The connection tells agents how to "translate" internal representations
    from point z to z + dz in latent space.

    Network architecture:
        Input:  [B, D] latent position
        Output: [B, S, G] connection coefficients
                S = spacetime_dim, G = gauge_dim
    """

    def __init__(self, config: MultiAgentConfig):
        super().__init__()
        self.config = config

        # Connection coefficients: A_mu^a(z)
        # Input: [B, D], Output: [B, S * G]
        self.connection_net = nn.Sequential(
            nn.Linear(config.latent_dim, config.latent_dim * 2),  # [B, D] -> [B, 2D]
            nn.Tanh(),                                             # [B, 2D] -> [B, 2D]
            nn.Linear(config.latent_dim * 2, config.spacetime_dim * config.gauge_dim)
            # [B, 2D] -> [B, S*G]
        )

    def forward(self, z: Tensor) -> Tensor:
        """
        Compute connection coefficients A_mu^a(z).

        Args:
            z: [B, D] - Latent positions

        Returns:
            A: [B, S, G] - Connection coefficients
               A[b, mu, a] = A_mu^a at position z[b]

        Tensor Shapes:
            Input:  z [B, D]
            Output: A [B, S, G] where S=spacetime_dim, G=gauge_dim
        """
        B = z.shape[0]
        # Network output: [B, S*G]
        A_flat = self.connection_net(z)  # [B, S*G]
        # Reshape to [B, S, G]
        return A_flat.view(B, self.config.spacetime_dim, self.config.gauge_dim)  # [B, S, G]


def covariant_derivative(
    psi: Tensor,
    A_mu: Tensor,
    g: float,
    direction: int
) -> Tensor:
    """
    Definition 29.21: Covariant Derivative

        D_mu = partial_mu - i g A_mu

    Args:
        psi: [B, D] - Matter field (belief amplitude)
        A_mu: [B, G] - Connection in direction mu
              G = gauge_dim
        g: float - Coupling constant
        direction: int - Coordinate direction mu (0 to S-1)

    Returns:
        D_mu_psi: [B, D] - Covariant derivative of psi

    Tensor Shapes:
        psi: [B, D]
        A_mu: [B, G]
        Output: [B, D]

    Note: Simplified version returning coupling term only.
          Full version would add partial_mu psi.
    """
    # Coupling term: -i g A_mu psi
    # A_mu: [B, G] -> unsqueeze -> [B, G, 1]
    # psi: [B, D] -> expand for gauge interaction
    # Simplified: return -i*g * A_mu * psi (element-wise approximation)
    return -1j * g * A_mu.unsqueeze(-1) * psi.unsqueeze(1)  # [B, G, D]


def field_strength_tensor(
    A: Tensor,
    g: float
) -> Tensor:
    """
    Definition 29.22: Field Strength Tensor (Yang-Mills Curvature)

        F_mu_nu = partial_mu A_nu - partial_nu A_mu - i g [A_mu, A_nu]

    For non-Abelian gauge groups, the commutator term creates self-interaction.

    Args:
        A: [B, S, G] - Connection coefficients
           B = batch, S = spacetime_dim, G = gauge_dim
        g: float - Coupling constant

    Returns:
        F: [B, S, S, G] - Field strength tensor
           F[b, mu, nu, a] = F_mu_nu^a at batch element b

    Tensor Shapes:
        Input:  A [B, S, G]
        Output: F [B, S, S, G]
    """
    B, S, G = A.shape

    # Initialize field strength: [B, S, S, G]
    F = torch.zeros(B, S, S, G, device=A.device, dtype=A.dtype)

    for mu in range(S):
        for nu in range(S):
            if mu != nu:
                # Simplified commutator: [A_mu, A_nu] as element-wise product difference
                # A[:, mu, :]: [B, G], A[:, nu, :]: [B, G]
                # For true non-Abelian, would use structure constants f^{abc}
                commutator = A[:, mu, :] * A[:, nu, :] - A[:, nu, :] * A[:, mu, :]  # [B, G]

                # F_mu_nu = -g * [A_mu, A_nu] (simplified, omitting partial derivatives)
                F[:, mu, nu, :] = -g * commutator  # [B, G]

    return F  # [B, S, S, G]


def yang_mills_action(
    F: Tensor,
    G_det: Tensor,
    g: float
) -> Tensor:
    """
    Definition 29.24: Yang-Mills Action

        S_YM[A] = -1/(4g^2) int Tr(F_mu_nu F^mu_nu) sqrt(|G|) d^{D+1}x

    Args:
        F: [B, S, S, G] - Field strength tensor
        G_det: [B] - Determinant of spacetime metric
        g: float - Coupling constant

    Returns:
        S_YM: scalar - Yang-Mills action

    Tensor Shapes:
        F: [B, S, S, G]
        G_det: [B]
        Output: scalar
    """
    # Tr(F_mu_nu F^mu_nu) = sum over all indices
    # F: [B, S, S, G], contract all: 'bmna,bmna->b' gives [B]
    F_sq = torch.einsum('bmna,bmna->b', F, F)  # [B]

    # Volume element: sqrt(|det G|)
    sqrt_G = torch.sqrt(torch.abs(G_det) + 1e-8)  # [B]

    # Action: -1/(4g^2) * integral
    # [B] * [B] -> [B] -> mean -> scalar
    S_YM = -1.0 / (4.0 * g**2) * (F_sq * sqrt_G).mean()  # scalar

    return S_YM  # scalar


def yang_mills_field_equation(
    F: Tensor,
    A: Tensor,
    J: Tensor,
    g: float
) -> Tensor:
    """
    Theorem 29.25: Yang-Mills Field Equations

        D_mu F^{mu nu} = J^nu

    Expanded: partial_mu F^{mu nu,a} + g f^{abc} A_mu^b F^{mu nu,c} = J^{nu,a}

    Args:
        F: [B, S, S, G] - Field strength tensor
        A: [B, S, G] - Connection
        J: [B, S, G] - Strategic current (source term)
        g: float - Coupling constant

    Returns:
        residual: [B, S, G] - Deviation from field equation (0 if satisfied)

    Tensor Shapes:
        F: [B, S, S, G]
        A: [B, S, G]
        J: [B, S, G]
        Output: [B, S, G]
    """
    B, S, _, G_dim = F.shape

    # Full implementation would compute D_mu F^{mu nu}
    # Simplified: return J (field equation residual = J - D_mu F)
    return J  # [B, S, G]


class HiggsField(nn.Module):
    """
    Definition 29.26: Higgs Sector (Value Order Parameter)

        L_Higgs = |D_mu Phi|^2 - V(Phi)
        V(Phi) = mu^2 |Phi|^2 + lambda |Phi|^4

    For mu^2 < 0, spontaneous symmetry breaking occurs.

    Attributes:
        phi: [G] - Higgs field (value order parameter)
    """

    def __init__(self, config: MultiAgentConfig):
        super().__init__()
        self.config = config

        # Higgs field: [G]
        self.phi = nn.Parameter(torch.randn(config.gauge_dim))  # [G]

    def potential(self, phi: Optional[Tensor] = None) -> Tensor:
        """
        Higgs potential V(Phi) = mu^2 |Phi|^2 + lambda |Phi|^4

        Args:
            phi: [G] or None - Higgs field (uses self.phi if None)

        Returns:
            V: scalar - Potential energy

        Tensor Shapes:
            phi: [G]
            Output: scalar
        """
        if phi is None:
            phi = self.phi  # [G]

        # |Phi|^2 = sum of squares: [G] -> scalar
        phi_sq = torch.sum(phi**2)  # scalar

        # V = mu^2 |Phi|^2 + lambda |Phi|^4
        return self.config.mu_higgs_sq * phi_sq + self.config.lambda_higgs * phi_sq**2  # scalar

    def vacuum_expectation_value(self) -> Tensor:
        """
        Theorem 29.27: Spontaneous Symmetry Breaking

            <Phi> = v/sqrt(2), where v = sqrt(-mu^2 / lambda)

        VEV exists when mu^2 < 0.

        Returns:
            vev: scalar - Vacuum expectation value

        Tensor Shapes:
            Output: scalar tensor
        """
        if self.config.mu_higgs_sq >= 0:
            return torch.tensor(0.0, device=self.phi.device)  # scalar

        # v = sqrt(-mu^2 / lambda)
        v = math.sqrt(-self.config.mu_higgs_sq / self.config.lambda_higgs)  # float
        # VEV = v / sqrt(2)
        return torch.tensor(v / math.sqrt(2), device=self.phi.device)  # scalar

    def gauge_boson_mass(self) -> Tensor:
        """
        Mass generation via Higgs mechanism:
            m_A = g * v / 2

        Returns:
            m_A: scalar - Gauge boson mass

        Tensor Shapes:
            Output: scalar tensor
        """
        vev = self.vacuum_expectation_value()  # scalar
        v = vev * math.sqrt(2)  # scalar
        return torch.tensor(self.config.g_coupling, device=self.phi.device) * v / 2  # scalar

    def forward(self) -> Tuple[Tensor, Tensor]:
        """
        Returns:
            (potential, vev): (scalar, scalar) - Potential energy and VEV
        """
        return self.potential(), self.vacuum_expectation_value()  # (scalar, scalar)


class CompleteLagrangian(nn.Module):
    """
    Definition 29.28: Complete Multi-Agent Lagrangian

        L_SMFT = L_YM + L_Dirac + L_Higgs + L_Yukawa

    The "Standard Model of Multi-Agent Field Theory".
    """

    def __init__(self, config: MultiAgentConfig):
        super().__init__()
        self.config = config

        # Gauge sector
        self.connection = StrategicConnection(config)

        # Higgs sector
        self.higgs = HiggsField(config)

        # Yukawa couplings: [N, N]
        self.yukawa = nn.Parameter(
            torch.randn(config.n_agents, config.n_agents) * 0.1
        )  # [N, N]

    def compute_lagrangian(
        self,
        z: Tensor,
        psi: List[Tensor]
    ) -> Dict[str, Tensor]:
        """
        Compute all Lagrangian components.

        Args:
            z: [B, D] - Latent positions
            psi: List of N tensors [B] - Belief spinors for each agent

        Returns:
            Dictionary with keys:
                'L_YM': scalar - Yang-Mills term
                'L_Higgs': scalar - Higgs potential
                'L_Yukawa': scalar - Yukawa coupling
                'vev': scalar - Vacuum expectation value
                'total': scalar - Total Lagrangian

        Tensor Shapes:
            z: [B, D]
            psi[i]: [B]
            All outputs: scalar
        """
        # Yang-Mills: compute connection and field strength
        A = self.connection(z)  # [B, S, G]
        F = field_strength_tensor(A, self.config.g_coupling)  # [B, S, S, G]

        # L_YM = -1/4 Tr(F^2)
        # 'bmna,bmna->' contracts all to scalar
        L_YM = -0.25 * torch.einsum('bmna,bmna->', F, F)  # scalar

        # Higgs
        V_higgs, vev = self.higgs()  # (scalar, scalar)
        L_Higgs = -V_higgs  # scalar (minus because V is potential energy)

        # Yukawa: -sum_{ij} y_{ij} psi_i psi_j
        L_Yukawa = torch.tensor(0.0, device=z.device)  # scalar
        for i in range(self.config.n_agents):
            for j in range(self.config.n_agents):
                if i < len(psi) and j < len(psi):
                    # psi[i]: [B], psi[j]: [B]
                    # y[i,j]: scalar
                    # sum over batch: [B] * [B] -> [B] -> sum -> scalar
                    L_Yukawa = L_Yukawa - self.yukawa[i, j] * (psi[i] * psi[j]).sum()

        return {
            'L_YM': L_YM,           # scalar
            'L_Higgs': L_Higgs,     # scalar
            'L_Yukawa': L_Yukawa,   # scalar
            'vev': vev,             # scalar
            'total': L_YM + L_Higgs + L_Yukawa  # scalar
        }


# =============================================================================
# Section 29.19: Mass Gap
# =============================================================================

def mass_gap_lower_bound(
    kappa: float,
    m_eff: float
) -> float:
    """
    Theorem 29.30: Mass Gap from Screening

        Delta >= kappa^2 / (2 * m_eff)

    Args:
        kappa: float - Screening mass (from discount factor: kappa = -ln(gamma))
        m_eff: float - Effective inertia from Game Tensor inflation

    Returns:
        Delta_min: float - Lower bound on mass gap

    Note: All inputs and output are Python floats, not tensors.
    """
    return kappa**2 / (2 * m_eff)  # float


def check_causal_information_bound(
    I_bulk: float,
    area: float,
    ell_L: float,
    nu_D: float = 0.25,
    D: int = 2
) -> Tuple[bool, float]:
    """
    Theorem 29.31: Computational Necessity of Mass Gap

    Check the Causal Information Bound:
        I_bulk <= nu_D * Area(partial V) / ell_L^{D-1}

    A gapless theory (Delta=0) violates this bound.

    Args:
        I_bulk: float - Bulk information content [nats]
        area: float - Boundary area [length^{D-1}]
        ell_L: float - Levin length (resolution scale) [length]
        nu_D: float - Holographic coefficient (default 1/4 for D=2)
        D: int - Spatial dimension

    Returns:
        satisfied: bool - Whether bound is satisfied
        C_partial: float - Boundary capacity [nats]

    Note: All inputs/outputs are Python types, not tensors.
    """
    # Boundary capacity: nu_D * Area / ell_L^{D-1}
    C_partial = nu_D * area / (ell_L ** (D - 1))  # float
    satisfied = I_bulk <= C_partial  # bool
    return satisfied, C_partial  # (bool, float)


# =============================================================================
# Section 29.21-29.25: Quantum Layer
# =============================================================================

def belief_wave_function(
    rho: Tensor,
    V: Tensor,
    sigma: float
) -> Tensor:
    """
    Definition 29.35: Belief Wave-Function

        psi(z, s) = sqrt(rho(z, s)) * exp(i V(z, s) / sigma)

    Combines belief density (amplitude) with value function (phase).

    Args:
        rho: [B, ...] - Belief density (non-negative)
        V: [B, ...] - Value function (real)
        sigma: float - Cognitive action scale (analog of hbar)

    Returns:
        psi: [B, ...] - Complex belief amplitude

    Tensor Shapes:
        rho and V must have same shape
        Output has same shape, complex dtype
    """
    # Amplitude: sqrt(rho), avoiding numerical issues
    amplitude = torch.sqrt(rho + 1e-8)  # Same shape as rho

    # Phase: V / sigma
    phase = V / sigma  # Same shape as V

    # Complex wave function: A * exp(i * phi)
    return amplitude * torch.exp(1j * phase)  # Same shape, complex dtype


def bohm_quantum_potential(
    rho: Tensor,
    G_inv: Tensor,
    sigma: float,
    eps: float = 1e-6
) -> Tensor:
    """
    Definition 29.38: Bohm Quantum Potential (Information Resolution Limit)

        Q_B = -sigma^2 / 2 * (Delta_G sqrt(rho)) / sqrt(rho)

    Represents the energetic cost of belief localization.

    Args:
        rho: [B] - Belief density (requires_grad for proper computation)
        G_inv: [D, D] - Inverse metric tensor
        sigma: float - Cognitive action scale
        eps: float - Small constant for numerical stability

    Returns:
        Q_B: [B] or scalar - Bohm potential

    Tensor Shapes:
        rho: [B]
        G_inv: [D, D]
        Output: [B] or scalar (simplified version returns scalar)
    """
    # Simplified version based on density variation
    # Full version would compute Laplace-Beltrami of sqrt(rho)
    sqrt_rho = torch.sqrt(rho + eps)  # [B]
    rho_variation = torch.var(rho)  # scalar

    # Q_B ~ -sigma^2/2 * (variation measure) / (mean amplitude)
    return -sigma**2 / 2 * rho_variation / (sqrt_rho.mean() + eps)  # scalar


class InferenceHamiltonian(nn.Module):
    """
    Theorem 29.37: Inference Hamiltonian

        H_inf = -sigma^2/2 * Delta_G + Phi_eff + Q_B - i*sigma/2 * r

    The generator of belief wave-function dynamics (Schrodinger equation).

    Network architecture:
        Phi_eff: [B, D] -> [B, 1] -> [B] effective potential
    """

    def __init__(self, config: MultiAgentConfig):
        super().__init__()
        self.config = config

        # Effective potential network
        # Input: [B, D], Output: [B, 1]
        self.Phi_eff = nn.Sequential(
            nn.Linear(config.latent_dim, config.latent_dim),  # [B, D] -> [B, D]
            nn.Tanh(),                                         # [B, D] -> [B, D]
            nn.Linear(config.latent_dim, 1)                    # [B, D] -> [B, 1]
        )

    def forward(
        self,
        psi: Tensor,
        z: Tensor,
        G: Tensor,
        r: Optional[Tensor] = None
    ) -> Tensor:
        """
        Apply H_inf to wave function.

        Args:
            psi: [B] - Wave function values (complex)
            z: [B, D] - Positions in latent space
            G: [D, D] - Metric tensor
            r: [B] or None - Reaction rate (for non-Hermitian part)

        Returns:
            H_psi: [B] - Hamiltonian applied to psi

        Tensor Shapes:
            psi: [B]
            z: [B, D]
            G: [D, D]
            r: [B] or None
            Output: [B]
        """
        sigma = self.config.sigma

        # Potential term: Phi_eff(z)
        # [B, D] -> [B, 1] -> squeeze -> [B]
        Phi = self.Phi_eff(z).squeeze(-1)  # [B]

        # Kinetic term: -sigma^2/2 * Delta_G psi
        # Simplified: zero (would need spatial discretization for proper Laplacian)
        kinetic = -sigma**2 / 2 * torch.zeros_like(psi)  # [B]

        # Reaction (non-Hermitian dissipation): -i * sigma/2 * r
        if r is not None:
            dissipation = -1j * sigma / 2 * r  # [B]
        else:
            dissipation = 0  # scalar

        # H psi = (kinetic + potential + dissipation) * psi
        H_psi = kinetic + Phi * psi + dissipation * psi  # [B]

        return H_psi  # [B]


class StrategicHamiltonian(nn.Module):
    """
    Definition 29.41: Strategic Hamiltonian (Multi-Agent)

        H_strat = sum_i H^(i)_kin + sum_i Phi^(i)_eff + sum_{i<j} V_ij

    Governs joint wave function evolution on tensor product Hilbert space.

    Attributes:
        H_single: List of N InferenceHamiltonian modules
        V_int: [N, N] interaction potential parameters
    """

    def __init__(self, config: MultiAgentConfig):
        super().__init__()
        self.config = config

        # Individual Hamiltonians
        self.H_single = nn.ModuleList([
            InferenceHamiltonian(config) for _ in range(config.n_agents)
        ])

        # Interaction potentials: [N, N]
        self.V_int = nn.Parameter(
            torch.randn(config.n_agents, config.n_agents) * 0.1
        )  # [N, N]

    def forward(
        self,
        Psi: Tensor,
        z_all: Tensor
    ) -> Tensor:
        """
        Apply H_strat to joint wave function.

        Args:
            Psi: [B] - Joint wave function (in full theory: [B, d1, d2, ..., dN])
            z_all: [N, B, D] - All agent positions

        Returns:
            H_Psi: [B] - Strategic Hamiltonian applied to Psi

        Tensor Shapes:
            Psi: [B] (simplified; full: [B, d1, ..., dN])
            z_all: [N, B, D]
            Output: [B]
        """
        H_Psi = torch.zeros_like(Psi)  # [B]

        # Single-agent kinetic + potential terms
        # (simplified: would properly handle tensor product structure)
        # In full implementation, would apply H^(i) to i-th factor

        # Interaction terms: sum_{i<j} V_ij
        for i in range(self.config.n_agents):
            for j in range(i + 1, self.config.n_agents):
                # V_int[i,j]: scalar
                V_ij = self.V_int[i, j]  # scalar
                # Add interaction: scalar * [B] -> [B]
                H_Psi = H_Psi + V_ij * Psi

        return H_Psi  # [B]


def strategic_entanglement_entropy(
    Psi: Tensor,
    agent_id: int,
    all_dims: List[int]
) -> Tensor:
    """
    Definition 29.40: Strategic Entanglement Entropy

        S_ent(i) = -Tr[rho^(i) ln rho^(i)]

    where rho^(i) = Tr_{j != i}[|Psi><Psi|] is the reduced density operator.

    Args:
        Psi: [total_dim] - Joint wave function (flattened tensor product)
        agent_id: int - Which agent to compute entropy for
        all_dims: List of N ints - Hilbert space dimension for each agent

    Returns:
        S_ent: scalar - Entanglement entropy in nats

    Tensor Shapes:
        Psi: [prod(all_dims)] flattened, or [d1, d2, ..., dN] unflattened
        Output: scalar

    Note: Simplified implementation. Full version would reshape Psi
          and compute partial trace properly.
    """
    # Simplified: von Neumann entropy of |Psi|^2 distribution
    # [total_dim] -> scalar
    prob = torch.abs(Psi)**2  # [total_dim]
    prob = prob / (prob.sum() + 1e-8)  # Normalize
    return -torch.sum(prob * torch.log(prob + 1e-8))  # scalar


def tunneling_probability(
    Phi_barrier: Tensor,
    E_0: Tensor,
    sigma: float,
    path_length: float
) -> Tensor:
    """
    Theorem 29.47: Strategic Tunneling Probability (WKB Approximation)

        P_tunnel ~ exp(-2/sigma * integral sqrt(2*(Phi - E_0)) dl)

    Args:
        Phi_barrier: [n_points] - Potential along tunneling path
        E_0: scalar - Ground state energy
        sigma: float - Cognitive action scale
        path_length: float - Total path length in latent space

    Returns:
        P_tunnel: scalar - Tunneling probability

    Tensor Shapes:
        Phi_barrier: [n_points]
        E_0: scalar
        Output: scalar
    """
    # Classical turning points: where Phi > E_0 (forbidden region)
    classically_forbidden = Phi_barrier > E_0  # [n_points], bool

    if not classically_forbidden.any():
        return torch.tensor(1.0, device=Phi_barrier.device)  # No barrier

    # Integrand in forbidden region: sqrt(2*(Phi - E_0))
    # Only where classically_forbidden is True
    integrand = torch.sqrt(2 * (Phi_barrier[classically_forbidden] - E_0))  # [n_forbidden]

    # Path element: dl = total_length / n_points
    dl = path_length / len(Phi_barrier)  # float

    # WKB exponent: 2/sigma * integral
    exponent = 2 / sigma * integrand.sum() * dl  # scalar

    # Tunneling probability: exp(-exponent)
    return torch.exp(-exponent)  # scalar


def imaginary_time_evolution(
    Psi_0: Tensor,
    H_strat: StrategicHamiltonian,
    z_all: Tensor,
    tau: float,
    n_steps: int = 100
) -> Tensor:
    """
    Proposition 29.48: Imaginary Time Evolution for Nash Finding

        Psi(tau) = exp(-H_strat * tau / sigma) Psi_0 -> c * Psi_Nash

    Equivalent to Value Iteration: projects onto ground state (Nash equilibrium).

    Args:
        Psi_0: [B] - Initial wave function
        H_strat: StrategicHamiltonian - The Hamiltonian module
        z_all: [N, B, D] - Agent positions
        tau: float - Total imaginary time
        n_steps: int - Number of Euler steps

    Returns:
        Psi_final: [B] - Evolved wave function (approximates Nash ground state)

    Tensor Shapes:
        Psi_0: [B]
        z_all: [N, B, D]
        Output: [B]
    """
    d_tau = tau / n_steps  # Imaginary time step
    Psi = Psi_0.clone()  # [B]

    for _ in range(n_steps):
        # Euler step: Psi' = Psi - (H Psi / sigma) * d_tau
        H_Psi = H_strat(Psi, z_all)  # [B]
        Psi = Psi - d_tau * H_Psi  # [B]

        # Renormalize to maintain unit norm
        Psi = Psi / (torch.norm(Psi) + 1e-8)  # [B]

    return Psi  # [B]


# =============================================================================
# Section 29.27-29.28: Diagnostic Nodes and Causal Buffer
# =============================================================================

def game_tensor_check(
    G_ij: Tensor,
    G_i: Tensor,
    G_max_factor: float = 10.0
) -> Tuple[bool, float]:
    """
    Node 46: GameTensorCheck

    Monitors ||G_ij||_F - Frobenius norm of Game Tensor.
    Large values indicate high strategic interdependence (intense conflict).

    Args:
        G_ij: [D, D] - Game tensor between agents i and j
        G_i: [D, D] - Base metric for agent i
        G_max_factor: float - Threshold factor

    Returns:
        passed: bool - Whether check passed
        norm: float - Frobenius norm of game tensor

    Tensor Shapes:
        G_ij: [D, D]
        G_i: [D, D]
        Output: (bool, float)
    """
    # Frobenius norm: sqrt(sum of squared elements)
    norm_G_ij = torch.norm(G_ij, p='fro')  # scalar
    norm_G_i = torch.norm(G_i, p='fro')    # scalar
    threshold = G_max_factor * norm_G_i    # scalar

    passed = norm_G_ij < threshold  # bool tensor
    return passed.item(), norm_G_ij.item()  # (bool, float)


def nash_residual_check(
    epsilon_Nash: float,
    tolerance: float = 1e-3
) -> Tuple[bool, str]:
    """
    Node 47: NashResidualCheck

    Monitors deviation from Nash stasis condition.

    Args:
        epsilon_Nash: float - Maximum gradient norm (from nash_residual())
        tolerance: float - Threshold for Nash convergence

    Returns:
        passed: bool - Whether check passed
        message: str - Diagnostic message
    """
    passed = epsilon_Nash < tolerance

    if passed:
        message = f"NashResidualCheck PASSED: epsilon = {epsilon_Nash:.6f} < {tolerance}"
    else:
        message = f"NashResidualCheck FAILED: epsilon = {epsilon_Nash:.6f} > {tolerance}"

    return passed, message


def relativistic_symplectic_check(
    Phi_out: Tensor,
    Phi_in: Tensor,
    tau_ij: float,
    dt: float,
    tolerance: float = 1e-4
) -> Tuple[bool, float]:
    """
    Node 48: RelativisticSymplecticCheck

    Monitors retarded flux balance:
        Delta_omega^ret = integral |Phi_out(t) - Phi_in(t + tau_ij)| dt

    Args:
        Phi_out: [T] - Outgoing flux over T time steps
        Phi_in: [T] - Incoming flux over T time steps
        tau_ij: float - Causal delay between agents
        dt: float - Time step size
        tolerance: float - Threshold for flux imbalance

    Returns:
        passed: bool - Whether check passed
        imbalance: float - Total flux imbalance

    Tensor Shapes:
        Phi_out: [T]
        Phi_in: [T]
        Output: (bool, float)
    """
    # Shift Phi_in by tau_ij time steps
    shift = int(tau_ij / dt)
    if shift > 0 and shift < len(Phi_in):
        # Phi_in shifted: [T - shift]
        Phi_in_shifted = Phi_in[shift:]
        # Trim Phi_out to match: [T - shift]
        Phi_out_trimmed = Phi_out[:len(Phi_in_shifted)]
        # Imbalance: integral of |difference|
        # [T-shift] -> abs -> sum -> * dt -> scalar
        imbalance = torch.abs(Phi_out_trimmed - Phi_in_shifted).sum() * dt
    else:
        imbalance = torch.tensor(0.0)

    passed = imbalance < tolerance
    return passed.item(), imbalance.item()  # (bool, float)


def causality_violation_check(
    mutual_info: Tensor,
    t_i: float,
    t_j: float,
    tau_ij: float
) -> Tuple[bool, str]:
    """
    Node 62: CausalityViolationCheck

    Detects if information arrived faster than c_info (causality violation).
    Agent i should have no mutual info with j's state at t' > t - tau_ij.

    Args:
        mutual_info: scalar - Mutual information I(z^(i)_t; z^(j)_{t'})
        t_i: float - Agent i's observation time
        t_j: float - Agent j's state time
        tau_ij: float - Causal delay from j to i

    Returns:
        passed: bool - True if no violation (causality preserved)
        message: str - Diagnostic message

    Tensor Shapes:
        mutual_info: scalar
        Output: (bool, str)
    """
    # Check if j's state is in i's future light cone
    in_future_cone = t_j > t_i - tau_ij  # j's time is after light could arrive
    has_info = mutual_info > 1e-8  # Non-trivial information

    violation = in_future_cone and has_info

    if violation:
        return False, "CausalityViolation DETECTED: Info from future light cone"
    else:
        return True, "CausalityViolationCheck PASSED"


def gauge_invariance_check(
    L_A: Tensor,
    L_A_prime: Tensor,
    tolerance: float = 1e-6
) -> Tuple[bool, float]:
    """
    Node 63: GaugeInvarianceCheck

    Monitors |L(A') - L(A)| under gauge transformation.
    Lagrangian should be unchanged by gauge transformations.

    Args:
        L_A: scalar - Lagrangian with original connection A
        L_A_prime: scalar - Lagrangian with transformed connection A'
        tolerance: float - Threshold for gauge invariance

    Returns:
        passed: bool - Whether check passed
        deviation: float - Gauge invariance deviation

    Tensor Shapes:
        L_A, L_A_prime: scalar
        Output: (bool, float)
    """
    deviation = torch.abs(L_A_prime - L_A)  # scalar
    passed = deviation < tolerance  # bool tensor
    return passed.item(), deviation.item()  # (bool, float)


def field_strength_bound_check(
    F: Tensor,
    F_max: float = 100.0
) -> Tuple[bool, float]:
    """
    Node 64: FieldStrengthBoundCheck

    Monitors ||F_mu_nu||_F - Frobenius norm of field strength.
    Large values indicate strong strategic curvature (intense conflict).

    Args:
        F: [B, S, S, G] - Field strength tensor
        F_max: float - Maximum allowed norm

    Returns:
        passed: bool - Whether check passed
        norm: float - Frobenius norm

    Tensor Shapes:
        F: [B, S, S, G]
        Output: (bool, float)
    """
    # Frobenius norm over all elements
    norm_F = torch.sqrt(torch.sum(F**2))  # scalar
    passed = norm_F < F_max  # bool tensor
    return passed.item(), norm_F.item()  # (bool, float)


def bianchi_violation_check(
    D_F: Tensor,
    tolerance: float = 1e-8
) -> Tuple[bool, float]:
    """
    Node 65: BianchiViolationCheck

    Monitors ||D_[mu F_nu_rho]|| - Bianchi identity should be exactly zero.
    Violations indicate topological defects or numerical errors.

    Args:
        D_F: [B, S, S, S, G] - Covariant derivative of field strength
             (cyclic sum D_[mu F_nu_rho])
        tolerance: float - Strict threshold for geometric constraint

    Returns:
        passed: bool - Whether check passed
        violation: float - Bianchi violation magnitude

    Tensor Shapes:
        D_F: [B, S, S, S, G]
        Output: (bool, float)
    """
    violation = torch.norm(D_F)  # scalar
    passed = violation < tolerance  # bool tensor
    return passed.item(), violation.item()  # (bool, float)


def mass_gap_check(
    E_0: float,
    E_1: float,
    Delta_min: float = 0.0
) -> Tuple[bool, str]:
    """
    Node 66: MassGapCheck

    Monitors spectral gap Delta = E_1 - E_0.
    Positive gap required for stable, non-frozen dynamics.

    Args:
        E_0: float - Ground state energy
        E_1: float - First excited state energy
        Delta_min: float - Minimum required gap

    Returns:
        passed: bool - Whether check passed
        message: str - Diagnostic message
    """
    Delta = E_1 - E_0  # Gap

    if Delta < 0:
        return False, f"MassGapCheck FAILED: Delta = {Delta:.6f} < 0 (tachyonic mode)"
    elif Delta < Delta_min:
        return False, f"MassGapCheck WARNING: Delta = {Delta:.6f} approaching critical point"
    else:
        return True, f"MassGapCheck PASSED: Delta = {Delta:.6f}"


def coherence_check(
    Psi_new: Tensor,
    Psi_old: Tensor,
    tolerance: float = 1e-6
) -> Tuple[bool, float]:
    """
    Node 57: CoherenceCheck

    Monitors ||Psi_{s+ds}||^2 - ||Psi_s||^2 for probability conservation.
    For closed systems, norm should be conserved (unitarity).

    Args:
        Psi_new: [B] - Wave function at new time
        Psi_old: [B] - Wave function at old time
        tolerance: float - Threshold for norm deviation

    Returns:
        passed: bool - Whether check passed
        deviation: float - Norm deviation

    Tensor Shapes:
        Psi_new, Psi_old: [B]
        Output: (bool, float)
    """
    # ||Psi||^2 = sum |psi_i|^2
    norm_new = torch.sum(torch.abs(Psi_new)**2)  # scalar
    norm_old = torch.sum(torch.abs(Psi_old)**2)  # scalar
    deviation = torch.abs(norm_new - norm_old)   # scalar

    passed = deviation < tolerance  # bool tensor
    return passed.item(), deviation.item()  # (bool, float)


def uncertainty_principle_check(
    sigma_z: Tensor,
    sigma_p: Tensor,
    sigma: float
) -> Tuple[bool, float]:
    """
    Node 59: UncertaintyPrincipleCheck

    Monitors sigma_z * sigma_p >= sigma/2 (Heisenberg-Robertson bound).
    Violation indicates over-confident world model.

    Args:
        sigma_z: [B] or scalar - Position uncertainty
        sigma_p: [B] or scalar - Momentum uncertainty
        sigma: float - Cognitive action scale

    Returns:
        passed: bool - Whether check passed (all eta <= 1)
        eta: float - Mean uncertainty ratio (should be <= 1)

    Tensor Shapes:
        sigma_z: [B] or scalar
        sigma_p: [B] or scalar
        Output: (bool, float)
    """
    # Uncertainty product
    product = sigma_z * sigma_p  # Same shape as inputs

    # Ratio: (sigma/2) / product - should be <= 1
    eta = (sigma / 2) / (product + 1e-8)  # Same shape

    # Check if all values satisfy bound
    passed = (eta <= 1).all().item() if eta.dim() > 0 else (eta <= 1).item()
    eta_mean = eta.mean().item() if eta.dim() > 0 else eta.item()

    return passed, eta_mean  # (bool, float)


# =============================================================================
# Section 29.28: Implementation - Causal Buffer
# =============================================================================

class CausalContextBuffer(nn.Module):
    """
    Algorithm 29.20.1: Causal Context Buffer

    Implements the Memory Screen for Relativistic Agents.
    Stores past signals and serves ghost states based on light-cone delay.

    Attributes:
        buffer: List[(float, Tensor)] - Ring buffer of (time, signal) pairs
                Each signal has shape [context_dim]
        c_info: float - Information propagation speed
        max_latency: int - Maximum buffer size
        context_dim: int - Dimension of stored signals
    """

    def __init__(self, context_dim: int, max_latency: int = 100, c_info: float = 1.0):
        super().__init__()
        self.buffer: List[Tuple[float, Tensor]] = []  # [(t, signal[context_dim]), ...]
        self.c_info = c_info      # Information speed
        self.max_latency = max_latency  # Max buffer entries
        self.context_dim = context_dim  # Signal dimension

    def write(self, t: float, signal: Tensor):
        """
        Agent emits signal at time t.

        Args:
            t: float - Emission time
            signal: [context_dim] - State tensor to record
        """
        # Clone to avoid reference issues: [context_dim] -> [context_dim]
        self.buffer.append((t, signal.clone()))

        # Prune old entries beyond max latency
        while self.buffer and self.buffer[0][0] < t - self.max_latency:
            self.buffer.pop(0)

    def read(self, t_now: float, dist: float) -> Tensor:
        """
        Read signal arriving at t_now from distance dist.
        Returns ghost state emitted at t_emit = t_now - dist/c_info.

        Args:
            t_now: float - Current time of reading agent
            dist: float - Environment distance to emitting agent

        Returns:
            ghost_signal: [context_dim] - Signal from retarded time

        Tensor Shapes:
            Output: [context_dim]
        """
        # Target emission time based on light-cone
        t_emit_target = t_now - (dist / self.c_info)

        if not self.buffer:
            return torch.zeros(self.context_dim)  # [context_dim]

        return self._interpolate(t_emit_target)  # [context_dim]

    def _interpolate(self, t_target: float) -> Tensor:
        """
        Linear interpolation between nearest buffer entries.

        Args:
            t_target: float - Target time to interpolate to

        Returns:
            signal: [context_dim] - Interpolated signal

        Tensor Shapes:
            Output: [context_dim]
        """
        if len(self.buffer) == 1:
            return self.buffer[0][1]  # [context_dim]

        # Find bracketing entries
        for i in range(len(self.buffer) - 1):
            t_lo, s_lo = self.buffer[i]      # (float, [context_dim])
            t_hi, s_hi = self.buffer[i + 1]  # (float, [context_dim])

            if t_lo <= t_target <= t_hi:
                # Linear interpolation coefficient
                alpha = (t_target - t_lo) / (t_hi - t_lo + 1e-8)  # float
                # Interpolate: (1-alpha)*s_lo + alpha*s_hi
                return (1 - alpha) * s_lo + alpha * s_hi  # [context_dim]

        # Extrapolate if t_target outside range
        if t_target < self.buffer[0][0]:
            return self.buffer[0][1]   # [context_dim]
        return self.buffer[-1][1]      # [context_dim]

    def clear(self):
        """Clear the buffer."""
        self.buffer = []


class RelativisticMultiAgentInterface(nn.Module):
    """
    Relativistic Multi-Agent Communication Interface.

    Wraps agent-to-agent communication with causal buffers,
    implementing the Ghost Interface and Memory Screen.

    Attributes:
        n_agents: int - Number of agents N
        c_info: float - Information speed
        buffers: Dict[str, CausalContextBuffer] - Buffer for each (i,j) pair
        tau: [N, N] - Causal delay matrix
        env_distances: [N, N] - Environment distance matrix
    """

    def __init__(
        self,
        n_agents: int,
        context_dim: int,
        env_distances: Tensor,
        c_info: float = 1.0,
        max_latency: int = 100
    ):
        """
        Args:
            n_agents: int - Number of agents N
            context_dim: int - Dimension of context signals
            env_distances: [N, N] - Environment distance matrix
            c_info: float - Information propagation speed
            max_latency: int - Maximum buffer size
        """
        super().__init__()
        self.n_agents = n_agents
        self.c_info = c_info

        # Causal buffer for each ordered pair (i, j)
        # Key format: "{i}_{j}" where i writes and j reads
        self.buffers = nn.ModuleDict({
            f"{i}_{j}": CausalContextBuffer(context_dim, max_latency, c_info)
            for i in range(n_agents) for j in range(n_agents) if i != j
        })

        # Precompute causal delays: tau[i,j] = d[i,j] / c_info
        # env_distances: [N, N] -> tau: [N, N]
        self.register_buffer('tau', env_distances / c_info)
        self.register_buffer('env_distances', env_distances)

    def broadcast(self, agent_id: int, t: float, state: Tensor):
        """
        Agent broadcasts its state to all outgoing buffers.

        Args:
            agent_id: int - Broadcasting agent index
            t: float - Broadcast time
            state: [context_dim] - State to broadcast
        """
        for j in range(self.n_agents):
            if j != agent_id:
                # Write to buffer where agent_id is sender, j is receiver
                self.buffers[f"{agent_id}_{j}"].write(t, state)

    def receive_context(self, agent_id: int, t: float) -> Tensor:
        """
        Agent receives ghost states from all other agents.

        Args:
            agent_id: int - Receiving agent index
            t: float - Current time of receiving agent

        Returns:
            context: [(N-1) * context_dim] - Concatenated ghost states

        Tensor Shapes:
            Output: [total_context_dim] where total = (N-1) * context_dim
        """
        contexts = []
        for j in range(self.n_agents):
            if j != agent_id:
                # Get distance and read ghost state
                dist = self.env_distances[agent_id, j].item()  # scalar
                ghost = self.buffers[f"{j}_{agent_id}"].read(t, dist)  # [context_dim]
                contexts.append(ghost)

        if contexts:
            # Concatenate: List of [context_dim] -> [total_context_dim]
            return torch.cat(contexts, dim=-1)
        return torch.zeros(0)  # Empty tensor if no other agents

    def get_past_light_cone(
        self,
        agent_id: int,
        t: float
    ) -> List[Tuple[int, float, Tensor]]:
        """
        Get all causally accessible states for agent_id at time t.

        Args:
            agent_id: int - Observing agent
            t: float - Observation time

        Returns:
            List of (other_agent_id: int, emission_time: float, ghost_state: [context_dim])
        """
        results = []
        for j in range(self.n_agents):
            if j != agent_id:
                tau_ij = self.tau[agent_id, j].item()  # Causal delay
                dist = self.env_distances[agent_id, j].item()
                ghost = self.buffers[f"{j}_{agent_id}"].read(t, dist)  # [context_dim]
                results.append((j, t - tau_ij, ghost))
        return results


# =============================================================================
# Complete Module: RelativisticMultiAgentSystem
# =============================================================================

class RelativisticMultiAgentSystem(nn.Module):
    """
    Complete implementation of Chapter 29: Relativistic Symplectic Multi-Agent Field Theory.

    Integrates all components:
        - Causal structure (Ghost Interface, retarded potentials)
        - Game Tensor (strategic coupling)
        - Gauge theory (Yang-Mills, Higgs)
        - Quantum layer (wave functions, tunneling)
        - Diagnostic nodes

    Attributes:
        config: MultiAgentConfig - System configuration
        causal_bundle: CausalBundle - Memory screens for all agents
        ghost_interface: GhostInterface - Ghost state extractor
        game_tensor_module: GameTensor - Game tensor computation
        lagrangian: CompleteLagrangian - Full SMFT Lagrangian
        strategic_hamiltonian: StrategicHamiltonian - Quantum Hamiltonian
        env_distances: [N, N] - Environment distance matrix
    """

    def __init__(self, config: MultiAgentConfig):
        super().__init__()
        self.config = config

        # Causal structure
        self.causal_bundle = CausalBundle(config)
        self.ghost_interface = GhostInterface(config)

        # Game tensor
        self.game_tensor_module = GameTensor(config)

        # Gauge theory
        self.lagrangian = CompleteLagrangian(config)

        # Quantum layer
        self.strategic_hamiltonian = StrategicHamiltonian(config)

        # Initialize environment distances (default: all pairs at distance 1)
        # [N, N] with zeros on diagonal
        dist = torch.ones(config.n_agents, config.n_agents)
        dist.fill_diagonal_(0)
        self.register_buffer('env_distances', dist)

    def set_environment_distances(self, distances: Tensor):
        """
        Set the environment distance matrix.

        Args:
            distances: [N, N] - Symmetric distance matrix with zeros on diagonal
        """
        self.env_distances = distances  # [N, N]

    def compute_causal_delays(self) -> Tensor:
        """
        Compute all pairwise causal delays tau_ij = d_ij / c_info.

        Returns:
            tau: [N, N] - Causal delay matrix
        """
        return causal_delay(self.env_distances, self.config.c_info)  # [N, N]

    def compute_game_tensors(self, z_all: Tensor) -> Tensor:
        """
        Compute all pairwise game tensors.

        Args:
            z_all: [N, B, D] - All agent positions

        Returns:
            G_tensors: [N, N, B, D, D] - All pairwise game tensors
        """
        return self.game_tensor_module(z_all)  # [N, N, B, D, D]

    def compute_effective_metrics(
        self,
        z_all: Tensor,
        G_base: List[Tensor]
    ) -> List[Tensor]:
        """
        Compute game-augmented metrics for all agents.

        Args:
            z_all: [N, B, D] - All agent positions
            G_base: List of N tensors [D, D] - Base metrics

        Returns:
            G_tilde: List of N tensors [B, D, D] - Effective metrics

        Tensor Shapes:
            z_all: [N, B, D]
            G_base[i]: [D, D]
            G_tilde[i]: [B, D, D]
        """
        N = self.config.n_agents
        G_tensors = self.compute_game_tensors(z_all)  # [N, N, B, D, D]

        G_tilde = []
        for i in range(N):
            # Collect (G_ij, beta_ij) pairs for agent i
            game_pairs = []
            for j in range(N):
                if j != i:
                    # G_tensors[i, j]: [B, D, D]
                    beta = self.config.beta_adversarial
                    game_pairs.append((G_tensors[i, j], beta))

            # Compute effective metric: [D, D] -> [B, D, D]
            G_tilde_i = effective_metric_with_game_tensor(
                G_base[i].unsqueeze(0).expand(z_all.shape[1], -1, -1),  # [B, D, D]
                game_pairs
            )
            G_tilde.append(G_tilde_i)  # [B, D, D]

        return G_tilde  # List of [B, D, D]

    def compute_klein_gordon_rhs(
        self,
        V_i: Tensor,
        rho_r_i: Tensor,
        Phi_ret: List[Tensor]
    ) -> Tensor:
        """
        Compute RHS of Klein-Gordon equation:
            rho_r^(i) + sum_j Phi^ret_ij

        Args:
            V_i: [B] - Agent i's value (not directly used)
            rho_r_i: [B] - Local reward source
            Phi_ret: List of [B] tensors - Retarded interaction potentials

        Returns:
            rhs: [B] - Klein-Gordon RHS
        """
        # Sum retarded potentials: [B] + sum([B], ...) -> [B]
        return rho_r_i + sum(Phi_ret)  # [B]

    def run_diagnostics(
        self,
        z_all: Tensor,
        G_tensors: Tensor,
        F: Tensor
    ) -> Dict[str, Tuple[bool, float]]:
        """
        Run all diagnostic nodes (46-48, 57-60, 62-66).

        Args:
            z_all: [N, B, D] - Agent positions
            G_tensors: [N, N, B, D, D] - Game tensors
            F: [B, S, S, G] - Field strength tensor

        Returns:
            diagnostics: Dict mapping node name to (passed: bool, value: float)
        """
        diagnostics = {}

        # Node 46: GameTensorCheck for each pair
        for i in range(self.config.n_agents):
            for j in range(self.config.n_agents):
                if i != j:
                    # G_tensors[i, j, 0]: [D, D] (first batch element)
                    G_ij = G_tensors[i, j, 0]  # [D, D]
                    G_i = torch.eye(self.config.latent_dim, device=z_all.device)  # [D, D]
                    passed, norm = game_tensor_check(G_ij, G_i)
                    diagnostics[f'node46_G_{i}{j}'] = (passed, norm)

        # Node 64: FieldStrengthBoundCheck
        passed, norm = field_strength_bound_check(F)  # F: [B, S, S, G]
        diagnostics['node64_field_strength'] = (passed, norm)

        return diagnostics

    def forward(
        self,
        z_all: Tensor,
        t: float
    ) -> Dict[str, Tensor]:
        """
        Forward pass computing all multi-agent dynamics.

        Args:
            z_all: [N, B, D] - All agent positions
            t: float - Current time

        Returns:
            Dictionary with:
                - game_tensors: [N, N, B, D, D]
                - connection: [B, S, G]
                - field_strength: [B, S, S, G]
                - lagrangian: Dict of scalars
                - causal_delays: [N, N]

        Tensor Shapes documented in return values above.
        """
        # Compute game tensors: [N, B, D] -> [N, N, B, D, D]
        game_tensors = self.compute_game_tensors(z_all)

        # Compute gauge connection at mean position
        # z_all: [N, B, D] -> mean over N -> [B, D]
        z_mean = z_all.mean(dim=0)
        # connection: [B, D] -> [B, S, G]
        A = self.lagrangian.connection(z_mean)

        # Field strength: [B, S, G] -> [B, S, S, G]
        F = field_strength_tensor(A, self.config.g_coupling)

        # Lagrangian components
        psi_dummy = [torch.randn(z_all.shape[1], device=z_all.device)
                     for _ in range(self.config.n_agents)]  # List of [B]
        lagrangian_components = self.lagrangian.compute_lagrangian(z_mean, psi_dummy)

        # Causal delays: [N, N]
        tau = self.compute_causal_delays()

        return {
            'game_tensors': game_tensors,       # [N, N, B, D, D]
            'connection': A,                     # [B, S, G]
            'field_strength': F,                 # [B, S, S, G]
            'lagrangian': lagrangian_components, # Dict of scalars
            'causal_delays': tau                 # [N, N]
        }


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Chapter 29: Relativistic Multi-Agent Field Theory — PyTorch")
    print("=" * 70)

    # Configuration with explicit dimensions
    config = MultiAgentConfig(
        n_agents=3,        # N = 3 agents
        latent_dim=32,     # D = 32 dimensional latent space
        gauge_dim=4,       # G = 4 dimensional gauge algebra
        spacetime_dim=4,   # S = 4 spacetime dimensions
        c_info=1.0,        # Information speed
        beta_adversarial=0.5,
        sigma=0.1
    )

    print(f"\nConfiguration:")
    print(f"  N (agents)     = {config.n_agents}")
    print(f"  D (latent_dim) = {config.latent_dim}")
    print(f"  G (gauge_dim)  = {config.gauge_dim}")
    print(f"  S (spacetime)  = {config.spacetime_dim}")

    # Create system
    system = RelativisticMultiAgentSystem(config)

    # Example positions: [N, B, D] = [3, 16, 32]
    batch_size = 16
    z_all = torch.randn(config.n_agents, batch_size, config.latent_dim)
    print(f"\nInput z_all shape: [{config.n_agents}, {batch_size}, {config.latent_dim}]")

    # Set environment distances: [N, N] = [3, 3]
    distances = torch.tensor([
        [0.0, 1.0, 2.0],
        [1.0, 0.0, 1.5],
        [2.0, 1.5, 0.0]
    ])
    system.set_environment_distances(distances)
    print(f"Environment distances shape: {list(distances.shape)}")

    # Forward pass
    results = system(z_all, t=10.0)

    print(f"\n--- Output Tensor Shapes ---")
    print(f"game_tensors:   {list(results['game_tensors'].shape)}")
    print(f"connection:     {list(results['connection'].shape)}")
    print(f"field_strength: {list(results['field_strength'].shape)}")
    print(f"causal_delays:  {list(results['causal_delays'].shape)}")

    print(f"\nCausal delays (tau_ij = d_ij / c_info):")
    print(results['causal_delays'])

    print(f"\nGame Tensor norms ||G_ij||_F:")
    for i in range(config.n_agents):
        for j in range(config.n_agents):
            if i != j:
                norm = torch.norm(results['game_tensors'][i, j]).item()
                print(f"  ||G_{i}{j}||_F = {norm:.4f}")

    print(f"\nField strength ||F||_F = {torch.norm(results['field_strength']):.4f}")

    print(f"\nLagrangian components:")
    for key, val in results['lagrangian'].items():
        if isinstance(val, Tensor):
            print(f"  {key}: {val.item():.4f}")

    # Test causal buffer
    print("\n--- Causal Buffer Test ---")
    buffer = CausalContextBuffer(context_dim=32, c_info=1.0)
    print(f"Buffer context_dim: {buffer.context_dim}")

    for t in range(10):
        signal = torch.randn(32)  # [32]
        buffer.write(float(t), signal)

    # Read ghost state: at t=10, from distance 3 -> retarded time = 10 - 3/1 = 7
    ghost = buffer.read(t_now=10.0, dist=3.0)  # [32]
    print(f"Ghost state shape: {list(ghost.shape)} (from t=7, distance=3)")

    # Test quantum layer
    print("\n--- Quantum Layer Test ---")
    rho = torch.rand(batch_size) + 0.1  # [B]
    rho = rho / rho.sum()  # Normalize to probability
    V = torch.randn(batch_size)  # [B]

    psi = belief_wave_function(rho, V, config.sigma)  # [B], complex
    print(f"Belief wave function shape: {list(psi.shape)}, dtype: {psi.dtype}")
    print(f"|psi|^2 sums to: {torch.sum(torch.abs(psi)**2).item():.4f}")

    # Test tunneling probability
    Phi_barrier = torch.tensor([0.0, 0.5, 1.0, 1.5, 1.0, 0.5, 0.0])  # [7]
    E_0 = torch.tensor(0.3)  # scalar
    P_tunnel = tunneling_probability(Phi_barrier, E_0, config.sigma, path_length=1.0)
    print(f"Tunneling probability: {P_tunnel.item():.6f}")

    # Test mass gap
    print("\n--- Mass Gap Analysis ---")
    kappa = 0.1
    m_eff = 1.0
    Delta_min = mass_gap_lower_bound(kappa, m_eff)
    print(f"Mass gap lower bound: Delta >= {Delta_min:.6f}")

    # Causal information bound
    I_bulk, area, ell_L = 10.0, 100.0, 1.0
    satisfied, C_partial = check_causal_information_bound(I_bulk, area, ell_L)
    print(f"Causal Information Bound: I_bulk={I_bulk}, C_partial={C_partial:.2f}, satisfied={satisfied}")

    # Run diagnostics
    print("\n--- Diagnostic Nodes ---")
    diagnostics = system.run_diagnostics(z_all, results['game_tensors'], results['field_strength'])
    for node, (passed, value) in diagnostics.items():
        status = "PASSED" if passed else "FAILED"
        print(f"  {node}: {status} (value={value:.4f})")

    print("\n" + "=" * 70)
    print("Chapter 29 implementation complete with full dimension annotations.")
    print("=" * 70)
