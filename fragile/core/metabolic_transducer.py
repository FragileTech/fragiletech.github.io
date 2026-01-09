"""
Section 36: The Metabolic Transducer - PyTorch Implementation
=============================================================

This module implements all equations from Section 36 of the Fragile Agent monograph,
providing a direct translation between mathematical formulas and executable code.

Each class and function is annotated with the corresponding Definition, Theorem,
or Equation number from the document.

References:
    - Szilard (1929): "On the decrease of entropy in a thermodynamic system..."
    - Landauer (1961): "Irreversibility and Heat Generation in the Computing Process"
    - Maturana & Varela (1980): "Autopoiesis and Cognition"
    - Friston (2010): "The free-energy principle: a unified brain theory?"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from dataclasses import dataclass, field
from typing import Tuple, Dict, Optional, Callable
from enum import Enum
import math


# =============================================================================
# Physical Constants
# =============================================================================

# Boltzmann constant in J/K
K_BOLTZMANN = 1.380649e-23  # SI units

# Natural logarithm of 2 (for bit-to-nat conversion)
LN2 = math.log(2)


# =============================================================================
# Section 36.1: The Thermodynamics of Information Harvesting
# =============================================================================

@dataclass
class ThermodynamicConfig:
    """
    Configuration for thermodynamic parameters.

    All parameters have physical units as specified in Section 36.
    """
    # Environmental temperature (Kelvin) - T_env
    T_env: float = 300.0

    # Cognitive temperature (Kelvin) - T_c
    T_c: float = 240.0

    # Boltzmann constant times T_env (Joules/nat) - k_B * T_env
    # For dimensionless analysis, set to 1.0
    k_B_T_env: float = 1.0

    # Transduction efficiency - η ∈ [0, 1]
    eta: float = 0.5

    # Battery capacity (Joules) - B_max
    B_max: float = 100.0

    # Initial battery (Joules) - B_0
    B_0: float = 50.0

    # Critical energy threshold (Joules) - B_crit
    B_crit: float = 10.0

    # Passive leak rate (1/step) - γ_leak
    gamma_leak: float = 0.01

    # Survival weight (dimensionless) - λ_surv
    lambda_surv: float = 10.0

    # Regularization constant for homeostatic potential - ε
    epsilon: float = 1e-3

    # Discount factor - γ (for screening mass κ = -ln(γ))
    gamma_discount: float = 0.99

    # Maximum heat dissipation rate (Watts) - Q_dot_radiate_max
    Q_radiate_max: float = 10.0


def reward_flux(
    reward_1form: Tensor,  # R(z): reward 1-form at current state
    velocity: Tensor,       # v_t = dz/dt: velocity in latent space
    metric: Tensor          # G: metric tensor
) -> Tensor:
    """
    Definition 36.1.1: The Reward Flux

    J_r(t) = ⟨R(z_t), v_t⟩_G = r_t

    The instantaneous rate of reward accumulation.

    Args:
        reward_1form: Reward 1-form R(z) ∈ R^d, shape (batch, d)
        velocity: Velocity v_t ∈ R^d, shape (batch, d)
        metric: Metric tensor G ∈ R^(d×d), shape (batch, d, d) or (d, d)

    Returns:
        Reward flux J_r(t), shape (batch,)

    Units: [nats/step] or [utility/step]
    """
    # Inner product ⟨R, v⟩_G = R^T G v
    if metric.dim() == 2:
        # Shared metric across batch
        Gv = torch.einsum('ij,bj->bi', metric, velocity)
    else:
        # Per-sample metric
        Gv = torch.einsum('bij,bj->bi', metric, velocity)

    return torch.einsum('bi,bi->b', reward_1form, Gv)


def information_utility(
    reward: Tensor,
    entropy_reward: Optional[Tensor] = None,
    entropy_reward_given_state: Optional[Tensor] = None
) -> Tensor:
    """
    Definition 36.1.2: Information Utility

    I_util(r_t) := I(Z_t; R_t) = H[R_t] - H[R_t | Z_t]

    Quantifies the actionable information content of the reward signal.

    Args:
        reward: Reward signal r_t, shape (batch,)
        entropy_reward: H[R_t] - marginal entropy of reward (optional)
        entropy_reward_given_state: H[R_t | Z_t] - conditional entropy (optional)

    Returns:
        Information utility I_util, shape (batch,)

    Units: [nats]

    Simplification: When reward is deterministic given state,
    H[R_t | Z_t] = 0, so I_util(r_t) ≈ |r_t| for rewards in natural units.
    """
    if entropy_reward is not None and entropy_reward_given_state is not None:
        # Full mutual information computation
        return entropy_reward - entropy_reward_given_state
    else:
        # Simplified approximation: I_util ≈ |r_t|
        return torch.abs(reward)


def szilard_work_bound(
    mutual_information: Tensor,  # I: mutual information in nats
    T_env: float                  # T_env: environmental temperature (K)
) -> Tensor:
    """
    Axiom 36.1.3: The Szilard Correspondence (Information-Work Duality)

    W_max = k_B T_env · I

    Maximum work extractable from I nats of mutual information.

    Args:
        mutual_information: Information I in nats, shape (batch,)
        T_env: Environmental temperature in Kelvin

    Returns:
        Maximum extractable work in Joules, shape (batch,)

    Physical basis: Inverse of Landauer's principle.
    - Landauer: erasing 1 bit costs k_B T ln(2) joules
    - Szilard: acquiring 1 bit enables extracting k_B T ln(2) joules
    """
    return K_BOLTZMANN * T_env * mutual_information


def transducer_bound(
    I_util: Tensor,    # I_util(r_t): information utility
    T_env: float,      # T_env: environmental temperature
    k_B: float = K_BOLTZMANN
) -> Tensor:
    """
    Theorem 36.1.4: The Transducer Bound

    Ė_in^max(t) = k_B T_env · I_util(r_t)

    Maximum free energy extractable per unit time.

    Args:
        I_util: Information utility in nats, shape (batch,)
        T_env: Environmental temperature in Kelvin
        k_B: Boltzmann constant (set to 1.0 for dimensionless analysis)

    Returns:
        Maximum energy flux in Joules/step, shape (batch,)
    """
    return k_B * T_env * I_util


class MetabolicTransducer(nn.Module):
    """
    Definition 36.1.5: The Metabolic Transducer Operator

    Ė_in(t) = T_harvest(r_t) := η · k_B T_env · I_util(r_t)

    Converts reward flux to free energy flux.

    Attributes:
        config: ThermodynamicConfig with all parameters

    Units: [Joules/step] (power)
    """

    def __init__(self, config: ThermodynamicConfig):
        super().__init__()
        self.config = config

        # Store as buffers for device compatibility
        self.register_buffer('k_B_T_env', torch.tensor(config.k_B_T_env))
        self.register_buffer('eta', torch.tensor(config.eta))

    def forward(
        self,
        reward: Tensor,
        I_util: Optional[Tensor] = None
    ) -> Tensor:
        """
        Compute transduced energy from reward.

        T_harvest(r_t) = η · k_B T_env · I_util(r_t)

        Args:
            reward: Reward signal r_t, shape (batch,)
            I_util: Pre-computed information utility (optional)

        Returns:
            Energy flux Ė_in in Joules/step, shape (batch,)
        """
        if I_util is None:
            I_util = information_utility(reward)

        # Only positive information content yields energy
        I_util_positive = torch.clamp(I_util, min=0.0)

        # Equation: T_harvest(r_t) = η · k_B T_env · I_util(r_t)
        return self.eta * self.k_B_T_env * I_util_positive

    def simplified_form(self, reward_nats: Tensor) -> Tensor:
        """
        Simplified form for dimensionless analysis (k_B = 1):

        T_harvest(r_t) = η · T_env · r_t

        Args:
            reward_nats: Reward measured in nats, shape (batch,)

        Returns:
            Energy flux, shape (batch,)
        """
        return self.eta * self.config.T_env * reward_nats


# =============================================================================
# Section 36.2: The Internal Battery and Autopoietic Dynamics
# =============================================================================

class InternalBattery(nn.Module):
    """
    Definition 36.2.1: The Internal Battery

    B: [0, ∞) → [0, B_max]

    Scalar state variable representing the agent's stored free energy.

    Units: [Joules] (energy)

    Interpretation: Capacity for future computation.
    - Biological: ATP/glucose reserves
    - Artificial: Available compute budget
    """

    def __init__(self, config: ThermodynamicConfig):
        super().__init__()
        self.config = config

        # Battery state B(t)
        self.register_buffer('B', torch.tensor(config.B_0))

        # Death flag
        self.register_buffer('is_dead', torch.tensor(False))

        # History for integral computations
        self.B_history: list = []
        self.harvest_history: list = []
        self.cost_history: list = []

    def get_battery(self) -> Tensor:
        """Get current battery level B(t)."""
        return self.B

    def is_alive(self) -> bool:
        """Check if agent is alive (B > 0)."""
        return not self.is_dead.item()


def battery_dynamics(
    B: Tensor,           # B(t): current battery level
    E_harvest: Tensor,   # T_harvest(r_t): transduced energy (income)
    M_dot: Tensor,       # Ṁ(t): metabolic cost
    gamma_leak: float    # γ_leak: passive dissipation rate
) -> Tensor:
    """
    Axiom 36.2.2: Energy Conservation (First Law)

    dB/dt = T_harvest(r_t) - Ṁ(t) - γ_leak · B(t)
           \_____________/   \___/   \__________/
              Income         Cost    Passive Leak

    Args:
        B: Current battery level, shape (batch,)
        E_harvest: Energy harvested from rewards, shape (batch,)
        M_dot: Metabolic cost (Landauer dissipation), shape (batch,)
        gamma_leak: Passive self-discharge rate (basal metabolic rate)

    Returns:
        Rate of change dB/dt, shape (batch,)

    Terminal Condition: If B(t) ≤ 0, agent undergoes Thermodynamic Death.
    """
    return E_harvest - M_dot - gamma_leak * B


def autopoietic_inequality(
    harvest_integral: Tensor,  # ∫₀^τ T_harvest(r_t) dt
    cost_integral: Tensor,     # ∫₀^τ Ṁ(t) dt
    B_integral: Tensor,        # ∫₀^τ B(t) dt
    B_0: float,                # Initial battery
    gamma_leak: float          # Passive leak rate
) -> Tensor:
    """
    Theorem 36.2.3: The Autopoietic Inequality

    Sufficient condition for survival at time τ (B(τ) > 0):

    ∫₀^τ (T_harvest(r_t) - Ṁ(t)) dt > γ_leak ∫₀^τ B(t) dt - B₀

    Args:
        harvest_integral: Integrated harvest energy
        cost_integral: Integrated metabolic cost
        B_integral: Integrated battery level
        B_0: Initial endowment
        gamma_leak: Passive dissipation rate

    Returns:
        Boolean tensor: True if inequality satisfied (survival likely)

    Physical interpretation: Agent must harvest more energy than it dissipates.
    This is the autopoietic closure condition.
    """
    lhs = harvest_integral - cost_integral
    rhs = gamma_leak * B_integral - B_0
    return lhs > rhs


def net_harvest_rate_condition(
    harvest_mean: Tensor,   # ⟨T⟩_τ: time-averaged harvest
    cost_mean: Tensor,      # ⟨Ṁ⟩_τ: time-averaged cost
    B_mean: Tensor,         # ⟨B⟩_τ: time-averaged battery
    B_0: float,             # Initial battery
    tau: float,             # Time horizon
    gamma_leak: float       # Passive leak rate
) -> Tensor:
    """
    Equivalent form of Theorem 36.2.3:

    ⟨T - Ṁ⟩_τ > γ_leak ⟨B⟩_τ - B₀/τ

    The time-averaged Net Harvest Rate must be positive.
    """
    return (harvest_mean - cost_mean) > (gamma_leak * B_mean - B_0 / tau)


def survival_objective(
    harvest_sequence: Tensor,  # T_harvest(r_t) for t = 0, 1, ..., T
    cost_sequence: Tensor,     # Ṁ(t) for t = 0, 1, ..., T
    gamma_leak: float,         # Passive leak rate
    dt: float = 1.0            # Time step
) -> Tensor:
    """
    Corollary 36.2.4: The Survival Objective

    J_survival = E[∫₀^∞ (T_harvest(r_t) - Ṁ(t)) e^(-γ_leak t) dt]

    The agent's fundamental objective: energy surplus maximization.

    Standard RL emerges when:
    1. Ṁ → 0 (free computation)
    2. η → 1 (perfect conversion)
    3. B_max → ∞ (unlimited storage)

    Args:
        harvest_sequence: Harvest at each timestep, shape (batch, T)
        cost_sequence: Cost at each timestep, shape (batch, T)
        gamma_leak: Discount/leak rate
        dt: Integration timestep

    Returns:
        Discounted survival objective, shape (batch,)
    """
    T = harvest_sequence.shape[-1]
    t = torch.arange(T, device=harvest_sequence.device, dtype=harvest_sequence.dtype)

    # Discount factors: e^(-γ_leak t)
    discount = torch.exp(-gamma_leak * t * dt)

    # Net energy at each step
    net_energy = harvest_sequence - cost_sequence

    # Discounted integral
    return (net_energy * discount).sum(dim=-1) * dt


# =============================================================================
# Section 36.3: The Fading Metric - Energy-Dependent Geometry
# =============================================================================

def information_maintenance_cost(
    T_c: float,        # T_c: cognitive temperature
    I_F: Tensor        # I_F: Fisher Information
) -> Tensor:
    """
    Theorem 36.3.1: The Information-Maintenance Cost

    Ė_maintain ≥ (1/2) T_c · I_F

    Maintaining Fisher Information requires continuous energy expenditure.

    Derivation:
    1. Fisher Information: I_F = E_ρ[||∇ln ρ||²_G]
    2. de Bruijn identity: dH[ρ]/dt = (1/2) I_F[ρ] under diffusion
    3. Landauer cost: Ė_maintain ≥ T_c |dH/dt| = (1/2) T_c · I_F

    Interpretation: Sharp distributions (high I_F) cost more to maintain.

    Args:
        T_c: Cognitive temperature (controls diffusion rate)
        I_F: Fisher Information of belief distribution

    Returns:
        Minimum maintenance energy rate, shape matches I_F
    """
    return 0.5 * T_c * I_F


def fading_function(x: Tensor) -> Tensor:
    """
    Theorem 36.3.2: The Fading Function

    f(x) = 1 - e^(-x)

    Satisfies:
    - f(0) = 0 (no energy → no metric)
    - lim_{x→∞} f(x) = 1 (saturation)
    - f(x) ≈ x for x ≪ 1 (linear regime)
    - f(x) ≈ 1 for x ≫ 1 (saturation regime)

    Args:
        x: Normalized energy B/B_crit, shape arbitrary

    Returns:
        Fading factor in [0, 1], same shape as x
    """
    return 1.0 - torch.exp(-x)


def fading_function_derivative(x: Tensor) -> Tensor:
    """
    Derivative of the fading function:

    f'(x) = e^(-x)

    Useful for gradient computations.
    """
    return torch.exp(-x)


class FadingMetric(nn.Module):
    """
    Theorem 36.3.2: The Fading Metric Law

    G_ij^eff(z, B) = f(B/B_crit) · G_ij(z)

    When available energy falls below maintenance requirement,
    the effective metric contracts.

    Attributes:
        B_crit: Critical energy for full metric resolution
    """

    def __init__(self, B_crit: float):
        super().__init__()
        self.B_crit = B_crit

    def forward(
        self,
        G: Tensor,   # Full-capacity metric G_ij(z)
        B: Tensor    # Current battery level B(t)
    ) -> Tensor:
        """
        Compute effective (faded) metric.

        G_eff = f(B/B_crit) · G

        Args:
            G: Full metric tensor, shape (..., d, d)
            B: Battery level, shape (batch,) or scalar

        Returns:
            Effective metric, shape (..., d, d)
        """
        # Normalized energy
        x = B / self.B_crit

        # Fading factor
        f = fading_function(x)

        # Scale metric
        if f.dim() == 0:
            return f * G
        else:
            # Broadcast f over metric dimensions
            while f.dim() < G.dim():
                f = f.unsqueeze(-1)
            return f * G

    def get_scaling(self, B: Tensor) -> Tensor:
        """Get the current metric scaling factor f(B/B_crit)."""
        return fading_function(B / self.B_crit)


def effective_geodesic_distance(
    d_G: Tensor,      # d_G(z, z'): geodesic distance in full metric
    B: Tensor,        # B(t): battery level
    B_crit: float     # B_crit: critical energy
) -> Tensor:
    """
    Corollary 36.3.3, Item 1: Resolution Loss

    d_G^eff(z, z') = √f(B/B_crit) · d_G(z, z') → 0 as B → 0

    Geodesic distances collapse; distinct concepts become indistinguishable.

    Args:
        d_G: Geodesic distance in full metric
        B: Battery level
        B_crit: Critical energy threshold

    Returns:
        Effective geodesic distance
    """
    f = fading_function(B / B_crit)
    return torch.sqrt(f) * d_G


def effective_causal_information_bound(
    I_max_full: Tensor,   # I_max = Area/(4 ℓ_L²): full capacity
    B: Tensor,            # B(t): battery level
    B_crit: float         # B_crit: critical energy
) -> Tensor:
    """
    Corollary 36.3.3, Item 3: Causal Dissolution

    I_max^eff = (Area(∂Z) / 4ℓ_L²) · f(B/B_crit) → 0 as B → 0

    The agent's representational capacity vanishes.

    Cross-reference: Section 33 (Causal Information Bound)

    Args:
        I_max_full: Full causal information bound
        B: Battery level
        B_crit: Critical energy threshold

    Returns:
        Effective causal information bound
    """
    f = fading_function(B / B_crit)
    return I_max_full * f


def dynamics_snr(
    velocity: Tensor,     # v: velocity in latent space
    G_eff: Tensor,        # G_eff: effective metric
    T_c: float            # T_c: cognitive temperature
) -> Tensor:
    """
    Corollary 36.3.4: SNR of Internal Dynamics

    SNR_dynamics = ||v||²_{G_eff} / (2 T_c) ∝ f(B/B_crit) → 0 as B → 0

    In the starvation regime:
    - Drift vanishes relative to diffusion
    - Agent performs random walk (hallucination)

    Args:
        velocity: Velocity v ∈ R^d, shape (batch, d)
        G_eff: Effective metric, shape (batch, d, d) or (d, d)
        T_c: Cognitive temperature

    Returns:
        Signal-to-noise ratio, shape (batch,)
    """
    # ||v||²_{G_eff} = v^T G_eff v
    if G_eff.dim() == 2:
        Gv = torch.einsum('ij,bj->bi', G_eff, velocity)
    else:
        Gv = torch.einsum('bij,bj->bi', G_eff, velocity)

    norm_sq = torch.einsum('bi,bi->b', velocity, Gv)

    return norm_sq / (2 * T_c)


def is_hallucinating(
    B: Tensor,
    B_crit: float,
    threshold: float = 0.1
) -> Tensor:
    """
    Check if agent is in starvation-hallucination regime.

    Hallucination occurs when f(B/B_crit) < threshold,
    i.e., SNR is too low for coherent inference.

    Args:
        B: Battery level
        B_crit: Critical energy
        threshold: Minimum scaling factor for coherent operation

    Returns:
        Boolean tensor: True if hallucinating
    """
    f = fading_function(B / B_crit)
    return f < threshold


# =============================================================================
# Section 36.4: Homeostatic Control - The Battery Potential
# =============================================================================

def homeostatic_potential(
    z: Tensor,                    # z: current state in latent space
    B: Tensor,                    # B(t): battery level
    food_region_mask: Tensor,     # 1[z ∈ Z_food]: indicator for food region
    lambda_surv: float,           # λ_surv: survival weight
    epsilon: float = 1e-3         # ε: regularization
) -> Tensor:
    """
    Definition 36.4.1: The Homeostatic Potential

    Φ_homeo(z, B) = (λ_surv / (B + ε)) · 1[z ∈ Z_food]

    Battery level induces a scalar potential field acting on policy.

    Args:
        z: State in latent space, shape (batch, d)
        B: Battery level, shape (batch,) or scalar
        food_region_mask: Binary mask for food region, shape (batch,)
        lambda_surv: Survival weight (dimensionless priority)
        epsilon: Regularization to prevent singularity

    Returns:
        Homeostatic potential Φ_homeo, shape (batch,)

    Units: [nats] (log-probability scale)
    """
    # Homeostatic drive: inversely proportional to battery
    drive = lambda_surv / (B + epsilon)

    # Only active in food region
    return drive * food_region_mask.float()


def total_potential(
    Phi_task: Tensor,    # Φ_task(z): task potential
    Phi_homeo: Tensor    # Φ_homeo(z, B): homeostatic potential
) -> Tensor:
    """
    Theorem 36.4.2: The Augmented Value Equation

    Φ_total(z, B) = Φ_task(z) + Φ_homeo(z, B)

    Total effective potential combines task and homeostatic contributions.

    The value function satisfies:
    (-Δ_{G_eff} + κ²) V = ρ_r + ρ_homeo

    where:
    - G_eff = f(B/B_crit) · G is the faded metric
    - ρ_homeo = -Δ Φ_homeo is the homeostatic source
    - κ = -ln(γ) is the screening mass

    Args:
        Phi_task: Task-related potential
        Phi_homeo: Homeostatic potential

    Returns:
        Total potential
    """
    return Phi_task + Phi_homeo


def priority_inversion_ratio(
    Phi_task: Tensor,    # Φ_task: task potential
    Phi_homeo: Tensor    # Φ_homeo: homeostatic potential
) -> Tensor:
    """
    Corollary 36.4.3: Priority Inversion at Low Battery

    As B → 0:
    - Φ_homeo ∝ 1/B → ∞ dominates Φ_task
    - Gradient steering: ∇Φ_total ≈ ∇Φ_homeo → Z_food
    - Task objectives become irrelevant; survival dominates

    Returns ratio indicating degree of priority inversion:
    - ratio > 1: Survival dominates
    - ratio < 1: Task dominates

    Args:
        Phi_task: Task potential (bounded)
        Phi_homeo: Homeostatic potential (∝ 1/B)

    Returns:
        Priority ratio Φ_homeo / (|Φ_task| + ε)
    """
    return Phi_homeo / (torch.abs(Phi_task) + 1e-6)


# =============================================================================
# Section 36.5: Thermal Management and the Carnot Bound
# =============================================================================

def carnot_efficiency(T_c: Tensor, T_env: float) -> Tensor:
    """
    Theorem 36.5.1: The Carnot Bound on Transduction

    η ≤ η_Carnot = 1 - T_c / T_env

    The transduction efficiency is bounded by the Carnot limit.

    Consequence: Agent must maintain T_c < T_env (thermal gradient)
    to extract any work. If T_c ≥ T_env, then η ≤ 0.

    Args:
        T_c: Cognitive temperature, shape arbitrary
        T_env: Environmental temperature (scalar)

    Returns:
        Carnot efficiency bound, clamped to [0, 1]
    """
    eta = 1.0 - T_c / T_env
    return torch.clamp(eta, 0.0, 1.0)


def waste_heat_flux(
    E_harvest_gross: Tensor,  # T_gross(r_t): gross transduction
    M_dot: Tensor,            # Ṁ(t): metabolic cost
    eta: Tensor               # η: transduction efficiency
) -> Tensor:
    """
    Definition 36.5.2: The Waste Heat Flux

    Q̇_waste = (1 - η) · T_gross(r_t) + Ṁ(t)

    Rate at which agent must dump entropy to environment.
    All non-useful energy becomes waste heat.

    Args:
        E_harvest_gross: Gross transduction before efficiency loss
        M_dot: Metabolic cost (Landauer dissipation)
        eta: Current transduction efficiency

    Returns:
        Waste heat flux in Watts

    Units: [Watts] (power)
    """
    return (1.0 - eta) * E_harvest_gross + M_dot


def check_thermal_runaway(
    Q_waste: Tensor,       # Q̇_waste: waste heat flux
    Q_radiate_max: float   # Q̇_radiate: maximum dissipation rate
) -> Tensor:
    """
    Corollary 36.5.3: The Thermal Runaway Condition

    If Q̇_waste > Q̇_radiate, then T_c increases.

    This triggers positive feedback:
    1. T_c ↑ ⟹ η_Carnot = 1 - T_c/T_env ↓
    2. Lower η ⟹ more waste heat
    3. More waste ⟹ T_c ↑

    Terminal state: T_c → T_env, η → 0, death by thermal runaway.

    Args:
        Q_waste: Current waste heat flux
        Q_radiate_max: Maximum heat dissipation capacity

    Returns:
        Boolean: True if thermal runaway is occurring
    """
    return Q_waste > Q_radiate_max


def thermal_margin(T_c: Tensor, T_env: float) -> Tensor:
    """
    Thermal safety margin: T_env - T_c

    Positive = safe, zero/negative = thermal runaway
    """
    return T_env - T_c


class ThermalDynamics(nn.Module):
    """
    Thermal dynamics model for cognitive temperature evolution.

    Definition 36.5.4: The Thermal Operating Envelope

    Agent is thermally viable if ∃ steady-state solution to:
    Q̇_waste(T_c) = Q̇_radiate(T_c)

    with T_c < T_env and η(T_c) > η_min.
    """

    def __init__(
        self,
        T_env: float,
        T_c_init: float,
        Q_radiate_max: float,
        thermal_mass: float = 1.0,  # Thermal inertia
        cooling_coefficient: float = 0.1  # Heat transfer coefficient
    ):
        super().__init__()
        self.T_env = T_env
        self.Q_radiate_max = Q_radiate_max
        self.thermal_mass = thermal_mass
        self.cooling_coefficient = cooling_coefficient

        self.register_buffer('T_c', torch.tensor(T_c_init))

    def compute_radiation(self, T_c: Tensor) -> Tensor:
        """
        Compute heat radiation rate (simplified Stefan-Boltzmann).

        Q̇_radiate = k · (T_c - T_ambient) clamped by Q_radiate_max
        """
        # Linear cooling model (Newton's law of cooling)
        Q_rad = self.cooling_coefficient * (T_c - 0.5 * self.T_env)
        return torch.clamp(Q_rad, 0.0, self.Q_radiate_max)

    def update(self, Q_waste: Tensor, dt: float = 1.0) -> Tensor:
        """
        Update cognitive temperature based on thermal balance.

        dT_c/dt = (Q̇_waste - Q̇_radiate) / C_thermal

        Args:
            Q_waste: Waste heat generated
            dt: Time step

        Returns:
            New cognitive temperature
        """
        Q_radiate = self.compute_radiation(self.T_c)
        dT_dt = (Q_waste - Q_radiate) / self.thermal_mass

        # Update temperature
        new_T_c = self.T_c + dT_dt * dt

        # Clamp to physical bounds
        new_T_c = torch.clamp(new_T_c, min=0.0, max=self.T_env)

        self.T_c.copy_(new_T_c)
        return self.T_c

    def is_in_operating_envelope(self, eta_min: float = 0.1) -> bool:
        """Check if current state is within thermal operating envelope."""
        eta = carnot_efficiency(self.T_c, self.T_env)
        return (self.T_c < self.T_env).item() and (eta > eta_min).item()


# =============================================================================
# Section 36.6 & 36.7: Complete Implementation with Diagnostics
# =============================================================================

class DiagnosticResult(Enum):
    """Diagnostic node status."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    DEAD = "dead"


@dataclass
class AutopoiesisDiagnostics:
    """
    Section 36.7: Diagnostic Nodes 67-70

    Monitoring autopoietic viability.
    """
    # Node 67: AutopoiesisCheck
    alive: bool = True
    battery_level: float = 0.0

    # Node 68: HarvestEfficiencyCheck
    harvest_efficiency: float = float('inf')  # ⟨T⟩/⟨Ṁ⟩
    is_sustainable: bool = True

    # Node 69: ThermalRunawayCheck
    thermal_margin: float = 0.0  # T_env - T_c
    thermal_stable: bool = True

    # Node 70: MetricFadingCheck
    metric_scaling: float = 1.0  # f(B/B_crit)
    is_hallucinating: bool = False

    def overall_status(self) -> DiagnosticResult:
        """Get overall system status."""
        if not self.alive:
            return DiagnosticResult.DEAD
        if self.is_hallucinating or not self.thermal_stable:
            return DiagnosticResult.CRITICAL
        if not self.is_sustainable or self.metric_scaling < 0.5:
            return DiagnosticResult.WARNING
        return DiagnosticResult.HEALTHY


class MetabolicSystem(nn.Module):
    """
    Complete Metabolic System implementing all of Section 36.

    Integrates:
    - Metabolic Transducer (36.1)
    - Internal Battery (36.2)
    - Fading Metric (36.3)
    - Homeostatic Control (36.4)
    - Thermal Management (36.5)
    - Diagnostic Nodes (36.7)

    This is the reference implementation linking the Sieve,
    the Governor, and the Reward signal.
    """

    def __init__(self, config: ThermodynamicConfig):
        super().__init__()
        self.config = config

        # Core components
        self.transducer = MetabolicTransducer(config)
        self.fading_metric = FadingMetric(config.B_crit)
        self.thermal = ThermalDynamics(
            T_env=config.T_env,
            T_c_init=config.T_c,
            Q_radiate_max=config.Q_radiate_max
        )

        # State variables
        self.register_buffer('battery', torch.tensor(config.B_0))
        self.register_buffer('is_dead', torch.tensor(False))

        # Running statistics for diagnostics
        self.register_buffer('harvest_ema', torch.tensor(0.0))
        self.register_buffer('cost_ema', torch.tensor(0.0))
        self.ema_decay = 0.99

        # History for analysis
        self.history: Dict[str, list] = {
            'battery': [],
            'harvest': [],
            'cost': [],
            'T_c': [],
            'metric_scaling': []
        }

    def step(
        self,
        reward: Tensor,
        metabolic_cost: Tensor,
        food_region_mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, AutopoiesisDiagnostics]:
        """
        Execute one step of the thermodynamic loop.

        Implements the complete energy flow from Section 36.8:

        Environment → Reward → Transducer → Battery → Metric → Inference → Action
                                               ↓
                                         Metabolism (cost)

        Args:
            reward: Reward signal r_t, shape (batch,) or scalar
            metabolic_cost: Ṁ(t) from Landauer bound, shape (batch,) or scalar
            food_region_mask: Optional indicator for food states

        Returns:
            delta_B: Net energy change
            diagnostics: AutopoiesisDiagnostics with all node values
        """
        if self.is_dead:
            return torch.tensor(0.0), AutopoiesisDiagnostics(
                alive=False, battery_level=0.0
            )

        # Ensure tensor types
        if not isinstance(reward, Tensor):
            reward = torch.tensor(reward)
        if not isinstance(metabolic_cost, Tensor):
            metabolic_cost = torch.tensor(metabolic_cost)

        # === Section 36.1: Transduction ===
        # E_in = T_harvest(r_t) = η · k_B T_env · I_util(r_t)
        I_util = information_utility(reward)
        E_harvest = self.transducer(reward, I_util)

        # Gross transduction (before efficiency)
        E_gross = self.config.k_B_T_env * I_util

        # === Section 36.5: Thermal Management ===
        # Current Carnot efficiency
        eta_current = carnot_efficiency(self.thermal.T_c, self.config.T_env)

        # Waste heat: Q̇_waste = (1 - η) · T_gross + Ṁ
        Q_waste = waste_heat_flux(E_gross, metabolic_cost, eta_current)

        # Update thermal state
        self.thermal.update(Q_waste)

        # === Section 36.2: Battery Dynamics ===
        # dB/dt = T_harvest - Ṁ - γ_leak · B
        dB_dt = battery_dynamics(
            self.battery,
            E_harvest,
            metabolic_cost,
            self.config.gamma_leak
        )

        # Update battery with Euler integration
        new_battery = torch.clamp(
            self.battery + dB_dt,
            min=0.0,
            max=self.config.B_max
        )
        self.battery.copy_(new_battery)

        # === Check death condition ===
        if self.battery <= 0:
            self.is_dead.copy_(torch.tensor(True))

        # === Update EMAs for diagnostics ===
        self.harvest_ema.copy_(
            self.ema_decay * self.harvest_ema +
            (1 - self.ema_decay) * E_harvest
        )
        self.cost_ema.copy_(
            self.ema_decay * self.cost_ema +
            (1 - self.ema_decay) * metabolic_cost
        )

        # === Section 36.3: Metric Scaling ===
        metric_scaling = self.fading_metric.get_scaling(self.battery)

        # === Section 36.7: Compute Diagnostics ===
        diagnostics = self._compute_diagnostics(metric_scaling, Q_waste)

        # === Store history ===
        self._update_history(E_harvest, metabolic_cost, metric_scaling)

        return dB_dt, diagnostics

    def _compute_diagnostics(
        self,
        metric_scaling: Tensor,
        Q_waste: Tensor
    ) -> AutopoiesisDiagnostics:
        """Compute all diagnostic nodes (67-70)."""

        # Node 67: AutopoiesisCheck - B(t) > 0
        alive = not self.is_dead.item()

        # Node 68: HarvestEfficiencyCheck - ⟨T⟩/⟨Ṁ⟩ > 1
        if self.cost_ema > 1e-6:
            efficiency = (self.harvest_ema / self.cost_ema).item()
        else:
            efficiency = float('inf')
        is_sustainable = efficiency > 1.0

        # Node 69: ThermalRunawayCheck - T_c < T_env
        t_margin = thermal_margin(self.thermal.T_c, self.config.T_env).item()
        thermal_ok = not check_thermal_runaway(
            Q_waste, self.config.Q_radiate_max
        ).item()

        # Node 70: MetricFadingCheck - f(B/B_crit) > ε_fade
        scaling = metric_scaling.item() if isinstance(metric_scaling, Tensor) else metric_scaling
        hallucinating = scaling < 0.1  # ε_fade threshold

        return AutopoiesisDiagnostics(
            alive=alive,
            battery_level=self.battery.item(),
            harvest_efficiency=efficiency,
            is_sustainable=is_sustainable,
            thermal_margin=t_margin,
            thermal_stable=thermal_ok,
            metric_scaling=scaling,
            is_hallucinating=hallucinating
        )

    def _update_history(
        self,
        E_harvest: Tensor,
        metabolic_cost: Tensor,
        metric_scaling: Tensor
    ):
        """Update history buffers."""
        self.history['battery'].append(self.battery.item())
        self.history['harvest'].append(E_harvest.item() if isinstance(E_harvest, Tensor) else E_harvest)
        self.history['cost'].append(metabolic_cost.item() if isinstance(metabolic_cost, Tensor) else metabolic_cost)
        self.history['T_c'].append(self.thermal.T_c.item())
        self.history['metric_scaling'].append(
            metric_scaling.item() if isinstance(metric_scaling, Tensor) else metric_scaling
        )

    def get_effective_metric(self, G: Tensor) -> Tensor:
        """
        Get the current effective (faded) metric.

        G_eff = f(B/B_crit) · G
        """
        return self.fading_metric(G, self.battery)

    def get_homeostatic_potential(
        self,
        z: Tensor,
        food_mask: Tensor
    ) -> Tensor:
        """
        Get current homeostatic potential.

        Φ_homeo(z, B) = λ_surv / (B + ε) · 1[z ∈ Z_food]
        """
        return homeostatic_potential(
            z, self.battery, food_mask,
            self.config.lambda_surv,
            self.config.epsilon
        )

    def check_autopoietic_inequality(self, tau: int) -> bool:
        """
        Check if autopoietic inequality is satisfied over recent history.

        ∫(T - Ṁ) dt > γ_leak ∫B dt - B_0
        """
        if len(self.history['harvest']) < tau:
            return True  # Not enough data

        recent_harvest = sum(self.history['harvest'][-tau:])
        recent_cost = sum(self.history['cost'][-tau:])
        recent_B = sum(self.history['battery'][-tau:])

        return autopoietic_inequality(
            torch.tensor(recent_harvest),
            torch.tensor(recent_cost),
            torch.tensor(recent_B),
            self.config.B_0,
            self.config.gamma_leak
        ).item()

    def reset(self):
        """Reset system to initial state."""
        self.battery.copy_(torch.tensor(self.config.B_0))
        self.is_dead.copy_(torch.tensor(False))
        self.thermal.T_c.copy_(torch.tensor(self.config.T_c))
        self.harvest_ema.zero_()
        self.cost_ema.zero_()
        self.history = {k: [] for k in self.history}


# =============================================================================
# Utility Functions for Analysis
# =============================================================================

def compute_survival_probability(
    system: MetabolicSystem,
    reward_distribution: Callable[[], Tensor],
    cost_distribution: Callable[[], Tensor],
    horizon: int,
    n_simulations: int = 1000
) -> float:
    """
    Monte Carlo estimation of survival probability.

    Simulates the autopoietic dynamics and estimates P(B(τ) > 0).

    Args:
        system: MetabolicSystem instance
        reward_distribution: Callable returning reward samples
        cost_distribution: Callable returning cost samples
        horizon: Time horizon τ
        n_simulations: Number of Monte Carlo runs

    Returns:
        Estimated survival probability
    """
    survived = 0

    for _ in range(n_simulations):
        system.reset()

        for _ in range(horizon):
            reward = reward_distribution()
            cost = cost_distribution()
            _, diagnostics = system.step(reward, cost)

            if not diagnostics.alive:
                break

        if system.battery > 0:
            survived += 1

    return survived / n_simulations


def analyze_phase_diagram(
    config: ThermodynamicConfig,
    eta_range: Tuple[float, float] = (0.1, 0.9),
    harvest_rate_range: Tuple[float, float] = (0.5, 2.0),
    n_points: int = 20
) -> Dict[str, Tensor]:
    """
    Compute phase diagram of autopoietic viability.

    Analyzes regions in (η, harvest_rate) space where:
    - Survival is possible (autopoietic region)
    - Thermal runaway occurs
    - Starvation occurs

    Returns tensors for visualization.
    """
    etas = torch.linspace(eta_range[0], eta_range[1], n_points)
    rates = torch.linspace(harvest_rate_range[0], harvest_rate_range[1], n_points)

    # Grid for analysis
    survival_map = torch.zeros(n_points, n_points)
    thermal_map = torch.zeros(n_points, n_points)

    for i, eta in enumerate(etas):
        for j, rate in enumerate(rates):
            # Check steady-state conditions
            # Survival: η · rate > cost + γ_leak · B_ss
            # Thermal: (1-η) · rate + cost < Q_radiate_max

            cost = 1.0  # Normalized
            harvest = eta * rate
            net = harvest - cost - config.gamma_leak * config.B_0

            survival_map[i, j] = 1.0 if net > 0 else 0.0

            Q_waste = (1 - eta) * rate + cost
            thermal_map[i, j] = 1.0 if Q_waste < config.Q_radiate_max else 0.0

    return {
        'etas': etas,
        'rates': rates,
        'survival': survival_map,
        'thermal': thermal_map,
        'viable': survival_map * thermal_map  # Both conditions
    }


# =============================================================================
# Summary Table (Section 36.8)
# =============================================================================

EQUATION_REFERENCE = """
Section 36: Equation Reference
==============================

36.1 Thermodynamics of Information Harvesting
---------------------------------------------
Def 36.1.1  Reward Flux:           J_r(t) = ⟨R(z_t), v_t⟩_G
Def 36.1.2  Information Utility:   I_util(r_t) = I(Z_t; R_t) = H[R_t] - H[R_t|Z_t]
Ax  36.1.3  Szilard Correspondence: W_max = k_B T_env · I
Thm 36.1.4  Transducer Bound:      Ė_in^max = k_B T_env · I_util(r_t)
Def 36.1.5  Metabolic Transducer:  T_harvest(r_t) = η · k_B T_env · I_util(r_t)

36.2 Internal Battery and Autopoietic Dynamics
----------------------------------------------
Def 36.2.1  Internal Battery:      B: [0,∞) → [0, B_max]
Ax  36.2.2  Energy Conservation:   dB/dt = T_harvest - Ṁ - γ_leak·B
Thm 36.2.3  Autopoietic Inequality: ∫(T - Ṁ)dt > γ_leak∫B dt - B_0
Cor 36.2.4  Survival Objective:    J = E[∫(T - Ṁ)e^{-γt} dt]

36.3 The Fading Metric
----------------------
Thm 36.3.1  Maintenance Cost:      Ė_maintain ≥ ½ T_c · I_F
Thm 36.3.2  Fading Metric Law:     G_eff = f(B/B_crit) · G, f(x) = 1-e^{-x}
Cor 36.3.3  Resolution Loss:       d_eff = √f · d_G → 0
Cor 36.3.4  Starvation-Hallucination: SNR = ||v||²_G / 2T_c → 0

36.4 Homeostatic Control
------------------------
Def 36.4.1  Homeostatic Potential: Φ_homeo = λ_surv/(B+ε) · 1[z ∈ Z_food]
Thm 36.4.2  Augmented Value:       (-Δ_{G_eff} + κ²)V = ρ_r + ρ_homeo
Cor 36.4.3  Priority Inversion:    Φ_homeo ∝ 1/B → ∞ as B → 0

36.5 Thermal Management
-----------------------
Thm 36.5.1  Carnot Bound:          η ≤ 1 - T_c/T_env
Def 36.5.2  Waste Heat:            Q̇_waste = (1-η)T_gross + Ṁ
Cor 36.5.3  Thermal Runaway:       Q̇_waste > Q̇_radiate ⟹ feedback loop
Def 36.5.4  Operating Envelope:    ∃ steady-state with T_c < T_env, η > η_min

36.7 Diagnostic Nodes
---------------------
Node 67: AutopoiesisCheck      - B(t) > 0
Node 68: HarvestEfficiencyCheck - ⟨T⟩/⟨Ṁ⟩ > 1
Node 69: ThermalRunawayCheck   - T_c < T_env
Node 70: MetricFadingCheck     - f(B/B_crit) > ε_fade
"""


if __name__ == "__main__":
    print(EQUATION_REFERENCE)

    # Example usage
    config = ThermodynamicConfig()
    system = MetabolicSystem(config)

    print("\n" + "="*60)
    print("Example Simulation")
    print("="*60)

    # Simulate 100 steps
    for t in range(100):
        # Varying reward and cost
        reward = torch.tensor(1.0 + 0.5 * math.sin(t * 0.1))
        cost = torch.tensor(0.8)

        delta_B, diagnostics = system.step(reward, cost)

        if t % 20 == 0:
            print(f"\nStep {t}:")
            print(f"  Battery: {diagnostics.battery_level:.2f}")
            print(f"  Harvest Efficiency: {diagnostics.harvest_efficiency:.2f}")
            print(f"  Metric Scaling: {diagnostics.metric_scaling:.3f}")
            print(f"  Status: {diagnostics.overall_status().value}")

        if not diagnostics.alive:
            print(f"\nAgent died at step {t}")
            break

    print("\n" + "="*60)
    print("Final State")
    print("="*60)
    print(f"Battery: {system.battery.item():.2f}")
    print(f"Cognitive Temperature: {system.thermal.T_c.item():.1f} K")
    print(f"Autopoietic Inequality (τ=50): {system.check_autopoietic_inequality(50)}")
