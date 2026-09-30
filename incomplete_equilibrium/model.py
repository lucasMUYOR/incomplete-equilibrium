"""Model specification: exponential-utility agents in an incomplete Brownian market.

State variables
---------------
B : Brownian motion driving the traded asset (spanned risk)
W : independent Brownian motion that no asset loads on (unspanned risk)

Traded assets
-------------
* a riskless bond with zero interest rate (numeraire);
* one stock in net supply ``theta`` with Bachelier dynamics
      dS_t = lambda_t dt + dB_t,
  whose market price of risk ``lambda_t = lambda(t, B_t, W_t)`` is endogenous.

Agents
------
Agent i has CARA utility U_i(x) = -exp(-alpha_i x) over terminal wealth
X_T^i + xi_i, where xi_i = g_i(B_T, W_T) is a (bounded) random endowment.

Because W is not traded, endowments that depend on W_T cannot be hedged:
the market is incomplete.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Sequence

import numpy as np

Endowment = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass
class Agent:
    """A CARA investor with terminal endowment ``g(B_T, W_T)``."""

    name: str
    alpha: float  # absolute risk aversion
    endowment: Endowment

    def __post_init__(self) -> None:
        if self.alpha <= 0:
            raise ValueError("risk aversion must be positive")


@dataclass
class Market:
    """Economy = horizon, stock supply and a list of agents."""

    agents: List[Agent]
    T: float = 1.0
    theta: float = 0.0  # net supply of the stock

    @property
    def risk_tolerance(self) -> float:
        """Aggregate risk tolerance tau = sum_i 1/alpha_i."""
        return float(sum(1.0 / a.alpha for a in self.agents))

    @property
    def alphas(self) -> np.ndarray:
        return np.array([a.alpha for a in self.agents])


# --------------------------------------------------------------------- examples
def example_market(scale: float = 1.0, T: float = 1.0, theta: float = 0.0) -> Market:
    """Three agents with bounded, *non-separable* endowments.

    The endowments mix spanned (B) and unspanned (W) risk nonlinearly, so no
    closed form is available and the market is genuinely incomplete.
    """
    return Market(
        agents=[
            Agent("A", 1.0, lambda b, w: scale * np.tanh(b + w)),
            Agent("B", 2.0, lambda b, w: scale * 0.5 * np.sin(b - w)),
            Agent("C", 0.5, lambda b, w: -scale * 0.8 * np.tanh(b) * np.tanh(w)),
        ],
        T=T,
        theta=theta,
    )


def separable_market(a: Sequence[float] = (1.0, -0.4), alphas: Sequence[float] = (1.0, 2.0),
                     T: float = 1.0, theta: float = 0.0) -> Market:
    """Endowments xi_i = a_i B_T + f_i(W_T): closed-form equilibrium exists.

    Used as an exact benchmark (see :mod:`incomplete_equilibrium.benchmarks`).
    """
    fs = [lambda w: np.tanh(w), lambda w: 0.5 * np.sin(w)]
    agents = []
    for i, (ai, al) in enumerate(zip(a, alphas)):
        f = fs[i % len(fs)]
        agents.append(Agent(f"S{i}", al, (lambda ai_, f_: (lambda b, w: ai_ * b + f_(w)))(ai, f)))
    return Market(agents=agents, T=T, theta=theta)


def separable_parts(market_a: Sequence[float] = (1.0, -0.4)):
    """(a_i, f_i) pairs matching :func:`separable_market`."""
    fs = [lambda w: np.tanh(w), lambda w: 0.5 * np.sin(w)]
    return [(ai, fs[i % len(fs)]) for i, ai in enumerate(market_a)]


def spanned_market(alphas: Sequence[float] = (1.0, 3.0), T: float = 1.0) -> Market:
    """Endowments depending on B_T only: the market is *complete*."""
    return Market(
        agents=[
            Agent("P", alphas[0], lambda b, w: np.tanh(b) + 0.0 * w),
            Agent("Q", alphas[1], lambda b, w: -0.5 * np.sin(b) + 0.0 * w),
        ],
        T=T,
    )
