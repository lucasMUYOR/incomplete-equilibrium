"""Independent Monte Carlo check of a computed equilibrium.

For each agent we simulate (B, W), trade the stock dS = lambda dt + dB with a
given strategy and estimate the certainty equivalent

    CE = -(1/alpha) log E[exp(-alpha (X_T + xi))].

If the PDE solution is an equilibrium then, for every agent,
  * the equilibrium strategy pi_i = lambda/alpha_i - d_b y_i attains CE ~ y_i(0,0,0);
  * perturbed strategies (no trading, scaled positions) do worse;
  * holdings clear the market: sum_i pi_i = theta on every path.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from .solver import Solution


def _bilinear(field: np.ndarray, grid: np.ndarray, b: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Bilinear interpolation of field[..., N, N] on a uniform grid (clipped to the domain)."""
    h = grid[1] - grid[0]
    n = len(grid)
    xb = np.clip((b - grid[0]) / h, 0, n - 1 - 1e-9)
    xw = np.clip((w - grid[0]) / h, 0, n - 1 - 1e-9)
    i, j = xb.astype(int), xw.astype(int)
    fb, fw = xb - i, xw - j
    f00, f10 = field[..., i, j], field[..., i + 1, j]
    f01, f11 = field[..., i, j + 1], field[..., i + 1, j + 1]
    return (1 - fb) * (1 - fw) * f00 + fb * (1 - fw) * f10 + (1 - fb) * fw * f01 + fb * fw * f11


@dataclass
class MCResult:
    ce: Dict[str, np.ndarray]        # strategy -> CE per agent
    se: Dict[str, np.ndarray]        # strategy -> standard error per agent
    max_clearing_error: float        # max over paths/times of |sum_i pi_i - theta|


def verify(sol: Solution, n_paths: int = 100_000, seed: int = 0) -> MCResult:
    m = sol.market
    rng = np.random.default_rng(seed)
    I = len(m.agents)
    dt = m.T / (len(sol.times) - 1)
    al = m.alphas[:, None]
    strategies = {"equilibrium": 1.0, "no_trade": 0.0, "scaled_0.5": 0.5, "scaled_1.5": 1.5}

    b = np.zeros(n_paths)
    w = np.zeros(n_paths)
    X = {k: np.zeros((I, n_paths)) for k in strategies}
    clearing = 0.0
    for n in range(len(sol.times) - 1):
        lam_f, yb_f = sol.fields(n)
        lam = _bilinear(lam_f, sol.grid, b, w)                     # (P,)
        yb = _bilinear(yb_f, sol.grid, b, w)                       # (I, P)
        pi = lam[None] / al - yb                                   # equilibrium holdings
        clearing = max(clearing, float(np.max(np.abs(pi.sum(axis=0) - m.theta))))
        dB = rng.standard_normal(n_paths) * np.sqrt(dt)
        dW = rng.standard_normal(n_paths) * np.sqrt(dt)
        dS = lam * dt + dB
        for k, s in strategies.items():
            X[k] += s * pi * dS[None]
        b += dB
        w += dW

    xi = np.stack([a.endowment(b, w) for a in m.agents])
    ce, se = {}, {}
    for k in strategies:
        z = -al * (X[k] + xi)
        zmax = z.max(axis=1, keepdims=True)
        e = np.exp(z - zmax)
        mean = e.mean(axis=1)
        ce[k] = -(np.log(mean) + zmax[:, 0]) / m.alphas
        se[k] = e.std(axis=1) / (mean * np.sqrt(n_paths)) / m.alphas
    return MCResult(ce, se, clearing)
