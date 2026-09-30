r"""Finite-difference solver for the incomplete-market (Radner) equilibrium.

Characterisation (see README, Theorem 1)
----------------------------------------
Write y_i(t, b, w) for agent i's certainty equivalent. A Markovian equilibrium
is a solution of the coupled system of semilinear parabolic PDEs

    d_t y_i + 1/2 (d_bb y_i + d_ww y_i)
        - lambda d_b y_i + lambda^2 / (2 alpha_i) - alpha_i / 2 (d_w y_i)^2 = 0,
    y_i(T, b, w) = g_i(b, w),

coupled through the market-clearing market price of risk

    lambda = (theta + sum_j d_b y_j) / tau,        tau = sum_j 1 / alpha_j.

Optimal holdings are pi_i = lambda / alpha_i - d_b y_i and sum_i pi_i = theta.

Numerical scheme
----------------
Backward in time (s = T - t), first-order IMEX splitting per step:
  1. explicit Euler step for the nonlinear coupling terms, with adaptive
     sub-steps chosen from a CFL condition;
  2. implicit (backward Euler) diffusion along b, then along w
     (tridiagonal solves, unconditionally stable).
Boundary condition: d_bb y = 0 (resp. d_ww y = 0) on the edges of the
truncated domain, which is exact for endowments that are affine far out.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.linalg import solve_banded

from .model import Market


def _implicit_band(n: int, r: float) -> np.ndarray:
    """Banded matrix of I - r * D2 with D2 = [1, -2, 1] inside, 0 on edges."""
    ab = np.zeros((3, n))
    ab[1, :] = 1.0 + 2.0 * r
    ab[1, 0] = ab[1, -1] = 1.0
    ab[0, 2:] = -r          # super-diagonal A[i, i+1], rows 1..n-2
    ab[2, :-2] = -r         # sub-diagonal   A[i+1, i], rows 1..n-2
    return ab


@dataclass
class Solution:
    market: Market
    grid: np.ndarray            # 1-D grid shared by b and w
    times: np.ndarray           # t_0 = 0 < ... < t_M = T
    y0: np.ndarray              # (I, N, N) certainty equivalents at t = 0
    lam0: np.ndarray            # (N, N) market price of risk at t = 0
    path: Optional[np.ndarray]  # (M+1, I, N, N) float32, y at every t_n (if stored)

    @property
    def h(self) -> float:
        return float(self.grid[1] - self.grid[0])

    def center_index(self) -> int:
        return int(np.argmin(np.abs(self.grid)))

    def ce0(self) -> np.ndarray:
        """Certainty equivalents y_i(0, 0, 0)."""
        c = self.center_index()
        return self.y0[:, c, c].astype(float)

    def lambda_at_origin(self) -> float:
        c = self.center_index()
        return float(self.lam0[c, c])

    # fields used by the Monte Carlo verification
    def fields(self, n: int):
        """(lambda, d_b y_i) at time index n as float64 arrays."""
        if self.path is None:
            raise RuntimeError("solve(store_path=True) is required")
        y = self.path[n].astype(float)
        yb = np.gradient(y, self.h, axis=1, edge_order=2)
        lam = (self.market.theta + yb.sum(axis=0)) / self.market.risk_tolerance
        return lam, yb


class EquilibriumSolver:
    """Solve the coupled PDE system on [-L, L]^2 with N points per axis and M time steps."""

    def __init__(self, market: Market, L: float = 6.0, N: int = 121, M: int = 400, scheme: str = "central"):
        """``scheme``: "central" (2nd order in space, accurate for moderate endowments) or
        "upwind" (monotone upwind/Godunov, 1st order, more robust for large endowments)."""
        if N < 5 or M < 1:
            raise ValueError("grid too small")
        if scheme not in ("central", "upwind"):
            raise ValueError("scheme must be 'central' or 'upwind'")
        self.scheme = scheme
        self.market = market
        self.L, self.N, self.M = L, N, M
        self.grid = np.linspace(-L, L, N)
        self.h = self.grid[1] - self.grid[0]
        self.dt = market.T / M

    def _one_sided(self, y: np.ndarray, axis: int):
        """Backward and forward differences along ``axis`` (one-sided copy at the edges)."""
        d = np.diff(y, axis=axis) / self.h
        n = y.shape[axis]
        back = np.concatenate([np.take(d, [0], axis=axis), d], axis=axis)      # D- y
        fwd = np.concatenate([d, np.take(d, [n - 2], axis=axis)], axis=axis)   # D+ y
        return back, fwd

    def _coupling(self, y: np.ndarray):
        """Right-hand side of d_s y = 1/2 Laplacian(y) + rhs (s = T - t) and lambda.

        lambda always uses centred differences. With scheme="central" every
        derivative is centred; with scheme="upwind" the transport terms use
        monotone discretisations:
          * -lambda d_b y_i         : upwind (information travels with speed lambda in +b);
          * -alpha_i/2 (d_w y_i)^2  : Godunov Hamiltonian for the convex H(p) = alpha p^2 / 2.
        """
        m = self.market
        yb = np.gradient(y, self.h, axis=1, edge_order=2)
        lam = (m.theta + yb.sum(axis=0)) / m.risk_tolerance
        al = m.alphas[:, None, None]
        if self.scheme == "central":
            yw = np.gradient(y, self.h, axis=2, edge_order=2)
            return -lam[None] * yb + lam[None] ** 2 / (2.0 * al) - 0.5 * al * yw ** 2, lam
        bb, bf = self._one_sided(y, axis=1)
        wb, wf = self._one_sided(y, axis=2)
        lp, lm = np.maximum(lam, 0.0)[None], np.minimum(lam, 0.0)[None]
        transport_b = lp * bb + lm * bf
        hamilton_w = 0.5 * al * np.maximum(np.maximum(wb, 0.0) ** 2, np.minimum(wf, 0.0) ** 2)
        rhs = -transport_b + lam[None] ** 2 / (2.0 * al) - hamilton_w
        return rhs, lam

    def solve(self, store_path: bool = False, cfl: float = 0.4, max_substeps: int = 10_000) -> Solution:
        """March backward from T to 0.

        Each of the M output steps is split into adaptive sub-steps so that the
        explicit nonlinear terms satisfy a CFL condition
            dt_sub * max(|lambda|, alpha_i |d_w y_i|) <= cfl * h,
        which keeps the scheme stable for large endowments / risk aversions.
        """
        m = self.market
        B, Wg = np.meshgrid(self.grid, self.grid, indexing="ij")
        y = np.stack([a.endowment(B, Wg) for a in m.agents]).astype(float)  # (I, N, N), axis1 = b, axis2 = w
        if not np.all(np.isfinite(y)):
            raise ValueError("endowment is not finite on the grid")
        al = m.alphas[:, None, None]
        self.substeps = 0

        path = np.empty((self.M + 1,) + y.shape, dtype=np.float32) if store_path else None
        if store_path:
            path[self.M] = y
        for n in range(self.M - 1, -1, -1):          # from t_{n+1} to t_n
            remaining = self.dt
            while remaining > 1e-15:
                rhs, lam = self._coupling(y)
                yw = np.gradient(y, self.h, axis=2, edge_order=2)
                speed = max(float(np.max(np.abs(lam))), float(np.max(al * np.abs(yw))), 1e-12)
                dt = min(remaining, cfl * self.h / speed)
                y = y + dt * rhs
                if not np.all(np.isfinite(y)):
                    raise FloatingPointError(f"blow-up at t = {n * self.dt:.4f}")
                ab = _implicit_band(self.N, 0.5 * dt / self.h ** 2)
                for i in range(y.shape[0]):
                    y[i] = solve_banded((1, 1), ab, y[i])            # implicit in b
                    y[i] = solve_banded((1, 1), ab, y[i].T).T        # implicit in w
                remaining -= dt
                self.substeps += 1
                if self.substeps > max_substeps * self.M:
                    raise FloatingPointError("time step collapsed (solution appears to blow up)")
            if store_path:
                path[n] = y
        _, lam0 = self._coupling(y)
        return Solution(m, self.grid, np.linspace(0.0, m.T, self.M + 1), y, lam0, path)
