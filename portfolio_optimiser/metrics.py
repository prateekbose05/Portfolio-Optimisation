"""Risk and performance metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import TRADING_DAYS


def portfolio_return(weights: np.ndarray, mean_returns: pd.Series) -> float:
    """Annualised expected portfolio return."""
    return float(np.dot(weights, mean_returns.values))


def portfolio_volatility(weights: np.ndarray, cov_matrix: pd.DataFrame) -> float:
    """Annualised portfolio standard deviation."""
    return float(np.sqrt(weights @ cov_matrix.values @ weights))


def sharpe_ratio(ret: float, vol: float, risk_free: float) -> float:
    """Sharpe ratio using excess return over the risk-free rate."""
    return (ret - risk_free) / vol if vol > 0 else 0.0


def sortino_ratio(daily_returns: pd.Series, risk_free: float) -> float:
    """Sortino ratio: excess return over annualised downside deviation."""
    ann_ret = daily_returns.mean() * TRADING_DAYS
    downside = daily_returns[daily_returns < 0]
    if downside.empty:
        return float("inf")
    downside_dev = downside.std() * np.sqrt(TRADING_DAYS)
    return float((ann_ret - risk_free) / downside_dev) if downside_dev > 0 else 0.0


def max_drawdown(daily_returns: pd.Series) -> float:
    """Largest peak-to-trough decline in wealth (negative number).

    Input is daily log returns, so wealth is exp of the cumulative sum.
    """
    wealth = np.exp(daily_returns.cumsum())
    return float((wealth / wealth.cummax() - 1).min())


def value_at_risk(daily_returns: pd.Series, level: float = 0.95) -> float:
    """Historical one-day VaR at the given confidence level (negative number)."""
    return float(np.percentile(daily_returns, (1 - level) * 100))


def conditional_var(daily_returns: pd.Series, level: float = 0.95) -> float:
    """Expected shortfall: mean loss on days beyond the VaR threshold."""
    var = value_at_risk(daily_returns, level)
    tail = daily_returns[daily_returns <= var]
    return float(tail.mean()) if not tail.empty else var


def beta(daily_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Portfolio beta against a benchmark, on overlapping dates."""
    joined = pd.concat([daily_returns, benchmark_returns], axis=1, join="inner").dropna()
    if len(joined) < 2:
        return float("nan")
    cov = np.cov(joined.iloc[:, 0], joined.iloc[:, 1])
    return float(cov[0, 1] / cov[1, 1]) if cov[1, 1] > 0 else float("nan")


def performance_summary(daily_returns: pd.Series, risk_free: float,
                        benchmark_returns: pd.Series | None = None) -> dict[str, float]:
    """Full realised-performance table for a daily return series."""
    ann_ret = float(daily_returns.mean() * TRADING_DAYS)
    ann_vol = float(daily_returns.std() * np.sqrt(TRADING_DAYS))
    summary = {
        "Annualised return": ann_ret,
        "Annualised volatility": ann_vol,
        "Sharpe ratio": sharpe_ratio(ann_ret, ann_vol, risk_free),
        "Sortino ratio": sortino_ratio(daily_returns, risk_free),
        "Max drawdown": max_drawdown(daily_returns),
        "Daily VaR (95%)": value_at_risk(daily_returns),
        "Daily CVaR (95%)": conditional_var(daily_returns),
    }
    if benchmark_returns is not None:
        summary["Beta vs benchmark"] = beta(daily_returns, benchmark_returns)
    return summary
