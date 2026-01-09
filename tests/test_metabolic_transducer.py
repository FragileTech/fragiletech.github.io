"""
Tests for Section 36: The Metabolic Transducer

These tests verify that the PyTorch implementation correctly implements
all equations from Section 36 of the Fragile Agent monograph.
"""

import torch
import math
import pytest
import sys
sys.path.insert(0, '.')

from fragile.core.metabolic_transducer import (
    # Constants
    K_BOLTZMANN, LN2,
    # Config
    ThermodynamicConfig,
    # Section 36.1
    reward_flux, information_utility, szilard_work_bound,
    transducer_bound, MetabolicTransducer,
    # Section 36.2
    InternalBattery, battery_dynamics, autopoietic_inequality,
    net_harvest_rate_condition, survival_objective,
    # Section 36.3
    information_maintenance_cost, fading_function, fading_function_derivative,
    FadingMetric, effective_geodesic_distance, effective_causal_information_bound,
    dynamics_snr, is_hallucinating,
    # Section 36.4
    homeostatic_potential, total_potential, priority_inversion_ratio,
    # Section 36.5
    carnot_efficiency, waste_heat_flux, check_thermal_runaway,
    thermal_margin, ThermalDynamics,
    # Complete system
    MetabolicSystem, DiagnosticResult, AutopoiesisDiagnostics,
)


class TestSection361_ThermodynamicsOfHarvesting:
    """Tests for Section 36.1: The Thermodynamics of Information Harvesting"""

    def test_reward_flux(self):
        """Test Definition 36.1.1: J_r(t) = ⟨R(z), v⟩_G"""
        # Simple case with identity metric
        R = torch.tensor([[1.0, 0.0, 0.0]])  # Reward 1-form
        v = torch.tensor([[1.0, 0.0, 0.0]])  # Velocity
        G = torch.eye(3)  # Identity metric

        J_r = reward_flux(R, v, G)
        assert torch.allclose(J_r, torch.tensor([1.0]))

        # Non-trivial metric
        G = torch.tensor([[2.0, 0.0], [0.0, 1.0]])
        R = torch.tensor([[1.0, 1.0]])
        v = torch.tensor([[1.0, 1.0]])
        J_r = reward_flux(R, v, G)
        # ⟨R, v⟩_G = R^T G v = [1,1] @ [[2,0],[0,1]] @ [1,1]^T = [1,1] @ [2,1] = 3
        assert torch.allclose(J_r, torch.tensor([3.0]))

    def test_information_utility_approximation(self):
        """Test Definition 36.1.2: I_util ≈ |r_t| (simplified form)"""
        rewards = torch.tensor([1.0, -2.0, 0.5, -0.3])
        I_util = information_utility(rewards)
        expected = torch.abs(rewards)
        assert torch.allclose(I_util, expected)

    def test_szilard_work_bound(self):
        """Test Axiom 36.1.3: W_max = k_B T_env · I"""
        I = torch.tensor([1.0])  # 1 nat
        T_env = 300.0  # Room temperature

        W = szilard_work_bound(I, T_env)
        expected = K_BOLTZMANN * T_env * 1.0
        assert torch.allclose(W, torch.tensor([expected]))

        # 1 bit = ln(2) nats should give k_B T ln(2)
        I_bit = torch.tensor([LN2])
        W_bit = szilard_work_bound(I_bit, T_env)
        assert torch.allclose(W_bit, torch.tensor([K_BOLTZMANN * T_env * LN2]))

    def test_transducer_bound(self):
        """Test Theorem 36.1.4: Ė_in^max = k_B T_env · I_util"""
        I_util = torch.tensor([2.0])
        T_env = 300.0

        E_max = transducer_bound(I_util, T_env)
        expected = K_BOLTZMANN * T_env * 2.0
        assert torch.allclose(E_max, torch.tensor([expected]))

    def test_metabolic_transducer(self):
        """Test Definition 36.1.5: T_harvest = η · k_B T_env · I_util"""
        config = ThermodynamicConfig(eta=0.5, k_B_T_env=1.0)
        transducer = MetabolicTransducer(config)

        reward = torch.tensor([2.0])
        E_harvest = transducer(reward)

        # η · k_B T_env · |r| = 0.5 · 1.0 · 2.0 = 1.0
        assert torch.allclose(E_harvest, torch.tensor([1.0]))

    def test_transducer_only_positive_harvest(self):
        """Verify only positive information yields energy"""
        config = ThermodynamicConfig(eta=0.5, k_B_T_env=1.0)
        transducer = MetabolicTransducer(config)

        # Negative reward should still give positive I_util (|r|)
        # but if we clamp at I_util level, we get 0 for negative
        reward = torch.tensor([-1.0])
        E_harvest = transducer(reward)
        # |r| = 1.0, so E = 0.5 * 1.0 * 1.0 = 0.5
        assert E_harvest.item() >= 0  # Energy is non-negative


class TestSection362_BatteryDynamics:
    """Tests for Section 36.2: The Internal Battery and Autopoietic Dynamics"""

    def test_battery_dynamics_equilibrium(self):
        """Test Axiom 36.2.2: dB/dt = T_harvest - Ṁ - γ_leak·B"""
        B = torch.tensor(50.0)
        E_harvest = torch.tensor(1.0)
        M_dot = torch.tensor(0.5)
        gamma_leak = 0.01

        dB_dt = battery_dynamics(B, E_harvest, M_dot, gamma_leak)
        # dB/dt = 1.0 - 0.5 - 0.01 * 50 = 0.5 - 0.5 = 0.0
        assert torch.allclose(dB_dt, torch.tensor(0.0))

    def test_battery_dynamics_growth(self):
        """Test battery grows when harvest > cost + leak"""
        B = torch.tensor(10.0)
        E_harvest = torch.tensor(2.0)
        M_dot = torch.tensor(0.5)
        gamma_leak = 0.01

        dB_dt = battery_dynamics(B, E_harvest, M_dot, gamma_leak)
        # dB/dt = 2.0 - 0.5 - 0.1 = 1.4 > 0
        assert dB_dt > 0

    def test_battery_dynamics_decay(self):
        """Test battery decays when harvest < cost + leak"""
        B = torch.tensor(100.0)
        E_harvest = torch.tensor(0.5)
        M_dot = torch.tensor(0.5)
        gamma_leak = 0.01

        dB_dt = battery_dynamics(B, E_harvest, M_dot, gamma_leak)
        # dB/dt = 0.5 - 0.5 - 1.0 = -1.0 < 0
        assert dB_dt < 0

    def test_autopoietic_inequality(self):
        """Test Theorem 36.2.3: ∫(T - Ṁ)dt > γ_leak∫B dt - B_0"""
        # Survival case
        harvest_integral = torch.tensor(100.0)
        cost_integral = torch.tensor(50.0)
        B_integral = torch.tensor(500.0)
        B_0 = 50.0
        gamma_leak = 0.01

        # LHS = 100 - 50 = 50
        # RHS = 0.01 * 500 - 50 = 5 - 50 = -45
        # 50 > -45 is True
        survives = autopoietic_inequality(
            harvest_integral, cost_integral, B_integral, B_0, gamma_leak
        )
        assert survives.item() == True

    def test_survival_objective(self):
        """Test Corollary 36.2.4: J = E[∫(T - Ṁ)e^{-γt} dt]"""
        # Constant harvest and cost over 10 steps
        T = 10
        harvest = torch.ones(1, T) * 2.0
        cost = torch.ones(1, T) * 1.0
        gamma_leak = 0.1

        J = survival_objective(harvest, cost, gamma_leak, dt=1.0)

        # Should be positive (harvesting more than spending)
        assert J.item() > 0


class TestSection363_FadingMetric:
    """Tests for Section 36.3: The Fading Metric"""

    def test_fading_function_properties(self):
        """Test f(x) = 1 - e^{-x} satisfies required properties"""
        # f(0) = 0
        assert torch.allclose(fading_function(torch.tensor(0.0)), torch.tensor(0.0))

        # f(∞) → 1
        assert fading_function(torch.tensor(100.0)).item() > 0.99

        # Monotonically increasing
        x = torch.linspace(0, 10, 100)
        f = fading_function(x)
        assert torch.all(f[1:] >= f[:-1])

        # f(x) ≈ x for small x
        x_small = torch.tensor(0.01)
        f_small = fading_function(x_small)
        assert torch.allclose(f_small, x_small, atol=1e-3)

    def test_fading_function_derivative(self):
        """Test f'(x) = e^{-x}"""
        x = torch.tensor([0.0, 1.0, 2.0])
        df = fading_function_derivative(x)
        expected = torch.exp(-x)
        assert torch.allclose(df, expected)

    def test_fading_metric_scaling(self):
        """Test Theorem 36.3.2: G_eff = f(B/B_crit) · G"""
        B_crit = 10.0
        fading = FadingMetric(B_crit)

        G = torch.eye(3) * 2.0  # Scaled identity metric

        # At B = B_crit, f(1) = 1 - e^{-1} ≈ 0.632
        B = torch.tensor(10.0)
        G_eff = fading(G, B)
        expected_scaling = 1 - math.exp(-1)
        assert torch.allclose(G_eff, G * expected_scaling, atol=1e-5)

        # At B = 0, f(0) = 0, so G_eff = 0
        B_zero = torch.tensor(0.0)
        G_eff_zero = fading(G, B_zero)
        assert torch.allclose(G_eff_zero, torch.zeros_like(G))

    def test_effective_geodesic_distance(self):
        """Test Corollary 36.3.3: d_eff = √f · d_G"""
        d_G = torch.tensor(10.0)  # Geodesic distance
        B = torch.tensor(5.0)
        B_crit = 10.0

        d_eff = effective_geodesic_distance(d_G, B, B_crit)

        f = 1 - math.exp(-0.5)  # B/B_crit = 0.5
        expected = math.sqrt(f) * 10.0
        assert torch.allclose(d_eff, torch.tensor(expected))

    def test_hallucination_detection(self):
        """Test Corollary 36.3.4: Hallucination when f < threshold"""
        B_crit = 10.0

        # Low battery → hallucinating
        B_low = torch.tensor(0.5)  # f(0.05) ≈ 0.05 < 0.1
        assert is_hallucinating(B_low, B_crit, threshold=0.1).item()

        # High battery → not hallucinating
        B_high = torch.tensor(50.0)  # f(5) ≈ 0.99 > 0.1
        assert not is_hallucinating(B_high, B_crit, threshold=0.1).item()


class TestSection364_HomeostaticControl:
    """Tests for Section 36.4: Homeostatic Control"""

    def test_homeostatic_potential(self):
        """Test Definition 36.4.1: Φ_homeo = λ_surv/(B+ε) · 1[z ∈ Z_food]"""
        z = torch.randn(3, 5)  # 3 samples, 5 dims
        food_mask = torch.tensor([1, 0, 1])
        lambda_surv = 10.0
        epsilon = 0.001

        # High battery → low homeostatic drive
        B_high = torch.tensor(100.0)
        Phi_high = homeostatic_potential(z, B_high, food_mask, lambda_surv, epsilon)
        # 10 / (100 + 0.001) ≈ 0.1 for food states
        assert torch.allclose(Phi_high[0], torch.tensor(lambda_surv / (100 + epsilon)), atol=1e-3)
        assert Phi_high[1].item() == 0  # Not in food region

        # Low battery → high homeostatic drive
        B_low = torch.tensor(1.0)
        Phi_low = homeostatic_potential(z, B_low, food_mask, lambda_surv, epsilon)
        # 10 / (1 + 0.001) ≈ 10 for food states
        assert Phi_low[0] > Phi_high[0]  # Higher drive at low battery

    def test_priority_inversion(self):
        """Test Corollary 36.4.3: Φ_homeo dominates as B → 0"""
        Phi_task = torch.tensor(1.0)

        # High battery → task dominates
        Phi_homeo_high = torch.tensor(0.1)
        ratio_high = priority_inversion_ratio(Phi_task, Phi_homeo_high)
        assert ratio_high < 1

        # Low battery → survival dominates
        Phi_homeo_low = torch.tensor(10.0)
        ratio_low = priority_inversion_ratio(Phi_task, Phi_homeo_low)
        assert ratio_low > 1


class TestSection365_ThermalManagement:
    """Tests for Section 36.5: Thermal Management and Carnot Bound"""

    def test_carnot_efficiency(self):
        """Test Theorem 36.5.1: η ≤ 1 - T_c/T_env"""
        T_env = 300.0

        # T_c = 0 → η = 1 (maximum efficiency)
        T_c_zero = torch.tensor(0.0)
        eta = carnot_efficiency(T_c_zero, T_env)
        assert torch.allclose(eta, torch.tensor(1.0))

        # T_c = T_env → η = 0 (no extraction possible)
        T_c_equal = torch.tensor(300.0)
        eta = carnot_efficiency(T_c_equal, T_env)
        assert torch.allclose(eta, torch.tensor(0.0))

        # T_c = 240K → η = 1 - 240/300 = 0.2
        T_c = torch.tensor(240.0)
        eta = carnot_efficiency(T_c, T_env)
        assert torch.allclose(eta, torch.tensor(0.2))

    def test_waste_heat_flux(self):
        """Test Definition 36.5.2: Q̇_waste = (1-η)T_gross + Ṁ"""
        E_gross = torch.tensor(10.0)
        M_dot = torch.tensor(2.0)
        eta = torch.tensor(0.5)

        Q_waste = waste_heat_flux(E_gross, M_dot, eta)
        # (1 - 0.5) * 10 + 2 = 5 + 2 = 7
        assert torch.allclose(Q_waste, torch.tensor(7.0))

    def test_thermal_runaway_detection(self):
        """Test Corollary 36.5.3: Thermal runaway when Q̇_waste > Q̇_radiate"""
        Q_radiate_max = 5.0

        # Below threshold → safe
        Q_waste_safe = torch.tensor(3.0)
        assert not check_thermal_runaway(Q_waste_safe, Q_radiate_max).item()

        # Above threshold → runaway
        Q_waste_danger = torch.tensor(7.0)
        assert check_thermal_runaway(Q_waste_danger, Q_radiate_max).item()


class TestCompleteSystem:
    """Tests for the complete MetabolicSystem"""

    def test_system_initialization(self):
        """Test system initializes correctly"""
        config = ThermodynamicConfig()
        system = MetabolicSystem(config)

        assert system.battery.item() == config.B_0
        assert not system.is_dead.item()

    def test_system_step(self):
        """Test single step of the system"""
        config = ThermodynamicConfig()
        system = MetabolicSystem(config)

        reward = torch.tensor(1.0)
        cost = torch.tensor(0.5)

        delta, diag = system.step(reward, cost)

        assert diag.alive
        assert diag.battery_level > 0
        assert diag.metric_scaling > 0

    def test_system_death(self):
        """Test system death when battery depletes"""
        config = ThermodynamicConfig(B_0=1.0, B_crit=1.0)
        system = MetabolicSystem(config)

        # High cost, no reward → death
        for _ in range(100):
            _, diag = system.step(torch.tensor(0.0), torch.tensor(1.0))
            if not diag.alive:
                break

        assert not diag.alive
        assert diag.overall_status() == DiagnosticResult.DEAD

    def test_diagnostic_nodes(self):
        """Test diagnostic nodes 67-70"""
        config = ThermodynamicConfig()
        system = MetabolicSystem(config)

        # Run a few steps
        for _ in range(10):
            system.step(torch.tensor(1.0), torch.tensor(0.5))

        _, diag = system.step(torch.tensor(1.0), torch.tensor(0.5))

        # Node 67: Should be alive
        assert diag.alive

        # Node 68: Harvest efficiency should be computable
        assert diag.harvest_efficiency > 0

        # Node 69: Thermal margin should be positive
        assert diag.thermal_margin > 0

        # Node 70: Metric scaling should be close to 1
        assert diag.metric_scaling > 0.5

    def test_system_reset(self):
        """Test system reset functionality"""
        config = ThermodynamicConfig()
        system = MetabolicSystem(config)

        # Run some steps
        for _ in range(10):
            system.step(torch.tensor(1.0), torch.tensor(0.5))

        # Reset
        system.reset()

        assert system.battery.item() == config.B_0
        assert not system.is_dead.item()
        assert len(system.history['battery']) == 0


class TestEquationMapping:
    """Verify direct mapping between equations and code"""

    def test_equation_36_1_1_reward_flux(self):
        """J_r(t) = ⟨R(z_t), v_t⟩_G = r_t"""
        # The reward flux is the inner product under metric G
        R = torch.tensor([[1.0, 2.0]])
        v = torch.tensor([[3.0, 4.0]])
        G = torch.eye(2)

        result = reward_flux(R, v, G)
        manual = (R * v).sum(dim=-1)  # Standard inner product
        assert torch.allclose(result, manual)

    def test_equation_36_2_2_battery_ode(self):
        """dB/dt = T_harvest(r_t) - Ṁ(t) - γ_leak · B(t)"""
        B = torch.tensor(50.0)
        T_harvest = torch.tensor(2.0)
        M_dot = torch.tensor(1.0)
        gamma_leak = 0.02

        # Direct computation
        expected = T_harvest - M_dot - gamma_leak * B
        result = battery_dynamics(B, T_harvest, M_dot, gamma_leak)

        assert torch.allclose(result, expected)

    def test_equation_36_3_2_fading_metric(self):
        """G_eff = f(B/B_crit) · G where f(x) = 1 - e^{-x}"""
        B_crit = 10.0
        B = torch.tensor(5.0)
        G = torch.tensor([[1.0, 0.5], [0.5, 1.0]])

        fading = FadingMetric(B_crit)
        G_eff = fading(G, B)

        x = B / B_crit
        f_x = 1 - torch.exp(-x)
        expected = f_x * G

        assert torch.allclose(G_eff, expected)

    def test_equation_36_4_1_homeostatic_potential(self):
        """Φ_homeo(z, B) = λ_surv/(B + ε) · 1[z ∈ Z_food]"""
        z = torch.randn(2, 3)
        B = torch.tensor(10.0)
        food_mask = torch.tensor([1.0, 0.0])
        lambda_surv = 5.0
        epsilon = 0.01

        result = homeostatic_potential(z, B, food_mask, lambda_surv, epsilon)

        expected = (lambda_surv / (B + epsilon)) * food_mask
        assert torch.allclose(result, expected)

    def test_equation_36_5_1_carnot(self):
        """η ≤ η_Carnot = 1 - T_c/T_env"""
        T_c = torch.tensor(250.0)
        T_env = 300.0

        result = carnot_efficiency(T_c, T_env)
        expected = 1 - T_c / T_env

        assert torch.allclose(result, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
