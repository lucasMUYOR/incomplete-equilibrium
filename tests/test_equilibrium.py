import numpy as np
import pytest

from incomplete_equilibrium import (EquilibriumSolver, benchmarks as bm, example_market, montecarlo as mc,
                                    separable_market, separable_parts, spanned_market)


def test_separable_incomplete_matches_closed_form():
    m = separable_market()
    s = EquilibriumSolver(m, N=121, M=400).solve()
    exact, lam = bm.separable_solution(separable_parts(), m.alphas, m.T, m.theta, 0.0, 0.0, 0.0)
    assert s.lambda_at_origin() == pytest.approx(lam, abs=1e-8)
    np.testing.assert_allclose(s.ce0(), exact, atol=2e-4)


def test_separable_with_positive_supply():
    m = separable_market(theta=0.3)
    s = EquilibriumSolver(m, N=121, M=400).solve()
    exact, lam = bm.separable_solution(separable_parts(), m.alphas, m.T, m.theta, 0.0, 0.0, 0.0)
    assert s.lambda_at_origin() == pytest.approx(lam, abs=1e-8)
    np.testing.assert_allclose(s.ce0(), exact, atol=2e-4)


def test_complete_market_is_pareto():
    m = spanned_market()
    s = EquilibriumSolver(m, N=121, M=400).solve()
    ys, lam = bm.complete_market_solution([np.tanh, lambda x: -0.5 * np.sin(x)], m.alphas, m.T)
    np.testing.assert_allclose(s.ce0(), ys, atol=2e-4)
    assert s.lambda_at_origin() == pytest.approx(float(lam(0.0)), abs=5e-4)


def test_grid_convergence_example():
    m = example_market()
    coarse = EquilibriumSolver(m, N=121, M=400).solve()
    fine = EquilibriumSolver(m, N=241, M=800).solve()
    assert abs(coarse.lambda_at_origin() - fine.lambda_at_origin()) < 1e-4
    np.testing.assert_allclose(coarse.ce0(), fine.ce0(), atol=2e-4)


def test_domain_truncation_is_harmless():
    m = example_market()
    a = EquilibriumSolver(m, L=6, N=121, M=200).solve()
    b = EquilibriumSolver(m, L=9, N=181, M=200).solve()
    np.testing.assert_allclose(a.ce0(), b.ce0(), atol=1e-6)


def test_monte_carlo_optimality_and_clearing():
    m = example_market()
    s = EquilibriumSolver(m, N=81, M=100).solve(store_path=True)
    r = mc.verify(s, n_paths=40_000, seed=3)
    assert r.max_clearing_error < 1e-10
    # equilibrium strategy reproduces the PDE certainty equivalents
    assert np.all(np.abs(r.ce["equilibrium"] - s.ce0()) < 4 * r.se["equilibrium"] + 2e-3)
    # and beats not trading for every agent
    assert np.all(r.ce["equilibrium"] > r.ce["no_trade"])


def test_trading_improves_on_autarky():
    m = example_market()
    s = EquilibriumSolver(m, N=121, M=400).solve()
    aut = np.array([bm.autarky_ce(a.endowment, a.alpha, m.T) for a in m.agents])
    assert np.all(s.ce0() >= aut - 1e-4)
