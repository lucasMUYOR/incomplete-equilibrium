"""Closed-form benchmarks used to validate the numerical solver.

1. Separable endowments  xi_i = a_i B_T + f_i(W_T)   (incomplete market)
   Theorem 2 (README): lambda = (theta + sum a_i) / tau is constant and
       y_i(t, b, w) = a_i b + phi_i(t, w) + (T - t) (lambda^2 / (2 alpha_i) - lambda a_i),
       phi_i(t, w) = -(1/alpha_i) log E[exp(-alpha_i f_i(w + W_{T-t}))]   (Cole-Hopf).

2. Spanned endowments  xi_i = g_i(B_T)   (complete market, theta = 0)
   The equilibrium is Pareto optimal; the pricing density is
   Z = exp(-G(B_T)/tau) / E[exp(-G(B_T)/tau)], G = sum_i g_i, and
       lambda(t, b) = -d_b log E[exp(-G(b + B_{T-t})/tau)],
       y_i(0) = E^Q[g_i(B_T)] + H(Q|P) / alpha_i.
"""
from __future__ import annotations

from typing import Callable, Sequence, Tuple

import numpy as np

_X, _WTS = np.polynomial.hermite_e.hermegauss(120)   # probabilists' Gauss-Hermite
_WTS = _WTS / _WTS.sum()


def gauss_expect(func: Callable[[np.ndarray], np.ndarray], mean, sd: float) -> np.ndarray:
    """E[func(mean + sd * Z)], Z ~ N(0,1), vectorised over ``mean``."""
    mean = np.asarray(mean, dtype=float)
    pts = mean[..., None] + sd * _X
    return (func(pts) * _WTS).sum(axis=-1)


def separable_solution(parts: Sequence[Tuple[float, Callable]], alphas: Sequence[float],
                       T: float, theta: float, t: float, b, w):
    """Exact (y_i, lambda) for separable endowments."""
    alphas = np.asarray(alphas, dtype=float)
    tau = (1.0 / alphas).sum()
    lam = (theta + sum(a for a, _ in parts)) / tau
    b, w = np.asarray(b, float), np.asarray(w, float)
    ys = []
    for (a, f), al in zip(parts, alphas):
        if T - t > 0:
            phi = -np.log(gauss_expect(lambda x: np.exp(-al * f(x)), w, np.sqrt(T - t))) / al
        else:
            phi = f(w)
        ys.append(a * b + phi + (T - t) * (lam ** 2 / (2 * al) - lam * a))
    return np.array(ys), lam


def complete_market_solution(gs: Sequence[Callable], alphas: Sequence[float], T: float):
    """Pareto benchmark at t = 0, b = 0: returns (y_i(0), lambda_fn(b))."""
    alphas = np.asarray(alphas, dtype=float)
    tau = (1.0 / alphas).sum()
    G = lambda x: sum(g(x) for g in gs)
    sd = np.sqrt(T)
    norm = gauss_expect(lambda x: np.exp(-G(x) / tau), 0.0, sd)
    dens = lambda x: np.exp(-G(x) / tau) / norm                         # dQ/dP as a function of B_T
    H = gauss_expect(lambda x: dens(x) * np.log(dens(x)), 0.0, sd)      # relative entropy H(Q|P)
    ys = np.array([gauss_expect(lambda x: dens(x) * g(x), 0.0, sd) + H / al for g, al in zip(gs, alphas)])

    def lam(b, eps: float = 1e-4):
        logM = lambda z: np.log(gauss_expect(lambda x: np.exp(-G(x) / tau), z, sd))
        return -(logM(np.asarray(b) + eps) - logM(np.asarray(b) - eps)) / (2 * eps)

    return ys, lam


def autarky_ce(g: Callable, alpha: float, T: float) -> float:
    """Certainty equivalent of holding the endowment without trading."""
    sd = np.sqrt(T)
    inner = lambda bb: gauss_expect(lambda ww: np.exp(-alpha * g(bb[..., None], ww)), np.zeros_like(bb), sd)
    val = gauss_expect(lambda bb: inner(bb), 0.0, sd)
    return float(-np.log(val) / alpha)
