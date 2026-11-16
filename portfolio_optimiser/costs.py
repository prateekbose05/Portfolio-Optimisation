"""Transaction costs and periodic rebalancing.

The headline backtest holds the optimised weights fixed and ignores trading
costs. A real portfolio cannot do both: weights drift as prices move, and
pulling them back to target means trading, which costs money. This module
simulates that directly so the trade-off can be measured:

- never rebalance: cheapest, but the portfolio drifts away from the weights
  the optimiser chose and ends up owning whatever went up most;
- rebalance monthly or quarterly: stays on target, pays for it in turnover.

Unlike the headline backtest, which sums weighted log returns, the simulation
here compounds simple returns on actual holdings, so "never rebalance" is a
true buy-and-hold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import TRADING_DAYS
from .metrics import performance_summary

FREQUENCIES = {"never": None, "quarterly": "Q", "monthly": "M"}


def _rebalance_days(index: pd.DatetimeIndex, rebalance: str) -> np.ndarray:
    """True on the last trading day of each period, excluding the final day."""
    if rebalance not in FREQUENCIES:
        raise ValueError(f"rebalance must be one of {', '.join(FREQUENCIES)}")
    freq = FREQUENCIES[rebalance]
    flags = np.zeros(len(index), dtype=bool)
    if freq is not None:
        periods = index.to_period(freq)
        flags[:-1] = periods[:-1] != periods[1:]
    return flags


def simulate_portfolio(prices: pd.DataFrame, target_weights: np.ndarray,
                       rebalance: str = "never", cost_bps: float = 10.0) -> dict:
    """Simulate holding a portfolio through `prices`, net of trading costs.

    The portfolio is bought at `target_weights` on the first day and left to
    drift. On each rebalance day it is traded back to target at the close.
    Every dollar bought or sold costs `cost_bps` basis points, including the
    initial purchase, and the cost comes out of portfolio value that day.

    Turnover is reported one-way: half the sum of absolute weight changes, so
    selling 5% of the portfolio to buy 5% of something else is 5% turnover.
    """
    if cost_bps < 0:
        raise ValueError("cost_bps cannot be negative")
    target = np.asarray(target_weights, dtype=float)
    if len(target) != prices.shape[1]:
        raise ValueError("one weight per price column is required")
    if not np.isclose(target.sum(), 1.0, atol=1e-6):
        raise ValueError("target weights must sum to one")

    simple = prices.pct_change().dropna(how="any")
    rebalance_today = _rebalance_days(simple.index, rebalance)
    rate = cost_bps / 10_000

    weights = target.copy()
    entry_cost = rate * 1.0                     # buying the whole portfolio once
    gross_log, net_log = [], []
    traded, costs, n_rebalances, max_drift = 0.0, entry_cost, 0, 0.0

    for day, returns in enumerate(simple.to_numpy()):
        growth = 1.0 + float(weights @ returns)
        weights = weights * (1.0 + returns) / growth
        max_drift = max(max_drift, float(np.abs(weights - target).max()))

        cost = entry_cost if day == 0 else 0.0
        if rebalance_today[day]:
            trade = float(np.abs(target - weights).sum())   # buys plus sells
            cost += rate * trade
            costs += rate * trade
            traded += trade
            weights = target.copy()
            n_rebalances += 1

        gross_log.append(np.log(growth))
        net_log.append(np.log(growth * (1.0 - cost)))

    years = len(simple) / TRADING_DAYS
    return {
        "gross_returns": pd.Series(gross_log, index=simple.index),
        "net_returns": pd.Series(net_log, index=simple.index),
        "final_weights": weights,
        "max_drift": max_drift,
        "annual_turnover": (traded / 2) / years,
        "total_cost": costs,
        "rebalances": n_rebalances,
    }


def rebalancing_study(prices: pd.DataFrame, weights: dict[str, np.ndarray],
                      risk_free: float, cost_bps: float = 10.0) -> pd.DataFrame:
    """Compare never / quarterly / monthly rebalancing for each strategy.

    One row per strategy and frequency: gross and net annualised return, the
    cost drag between them, net Sharpe, one-way annual turnover, and the
    furthest any holding got from its target weight at any point.
    """
    rows = []
    for name, target in weights.items():
        target = np.asarray(target, dtype=float)
        for rebalance in FREQUENCIES:
            run = simulate_portfolio(prices, target, rebalance, cost_bps)
            gross = float(run["gross_returns"].mean() * TRADING_DAYS)
            net = performance_summary(run["net_returns"], risk_free)
            rows.append({
                "Strategy": name,
                "Rebalance": rebalance,
                "Gross return": gross,
                "Net return": net["Annualised return"],
                "Cost drag": gross - net["Annualised return"],
                "Net Sharpe": net["Sharpe ratio"],
                "Annual turnover": run["annual_turnover"],
                "Rebalances": run["rebalances"],
                "Max weight drift": run["max_drift"],
            })
    return pd.DataFrame(rows).set_index(["Strategy", "Rebalance"])
