"""Portfolio optimisation: Monte Carlo simulation and analytical (SciPy) solutions.

Monte Carlo maps the feasible risk-return space and gives an intuitive picture
of the efficient frontier. The analytical optimiser (SLSQP) then finds the exact
maximum-Sharpe and minimum-volatility portfolios, which random sampling can only
approximate. Reporting both, and the gap between them, is the point.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .metrics import portfolio_return, portfolio_volatility, sharpe_ratio


def monte_carlo_portfolios(mean_returns: pd.Series, cov_matrix: pd.DataFrame,
                           risk_free: float, n_portfolios: int = 10_000,
                           seed: int | None = 42) -> tuple[pd.DataFrame, np.ndarray]:
    """Simulate random fully-invested long-only portfolios.

    Returns a results DataFrame (return, volatility, Sharpe) and the weights array.
    """
    rng = np.random.default_rng(seed)
    n_assets = len(mean_returns)
    weights = rng.dirichlet(np.ones(n_assets), n_portfolios)

    rets = weights @ mean_returns.values
    vols = np.sqrt(np.einsum("ij,jk,ik->i", weights, cov_matrix.values, weights))
    sharpes = (rets - risk_free) / vols

    results = pd.DataFrame({"Return": rets, "Volatility": vols, "Sharpe": sharpes})
    return results, weights


def _optimise(objective, n_assets: int) -> np.ndarray:
    """Minimise an objective over long-only weights that sum to one."""
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1},)
    bounds = tuple((0.0, 1.0) for _ in range(n_assets))
    start = np.full(n_assets, 1 / n_assets)
    result = minimize(objective, start, method="SLSQP", bounds=bounds, constraints=constraints)
    if not result.success:
        raise RuntimeError(f"Optimisation failed: {result.message}")
    return result.x


def max_sharpe_weights(mean_returns: pd.Series, cov_matrix: pd.DataFrame,
                       risk_free: float) -> np.ndarray:
    """Exact maximum-Sharpe (tangency) portfolio weights."""
    def neg_sharpe(w):
        return -sharpe_ratio(portfolio_return(w, mean_returns),
                             portfolio_volatility(w, cov_matrix), risk_free)
    return _optimise(neg_sharpe, len(mean_returns))


def min_volatility_weights(mean_returns: pd.Series, cov_matrix: pd.DataFrame) -> np.ndarray:
    """Exact minimum-volatility portfolio weights."""
    return _optimise(lambda w: portfolio_volatility(w, cov_matrix), len(mean_returns))


def efficient_frontier(mean_returns: pd.Series, cov_matrix: pd.DataFrame,
                       n_points: int = 50) -> pd.DataFrame:
    """The analytical efficient frontier: minimum volatility for each target return."""
    min_ret = portfolio_return(min_volatility_weights(mean_returns, cov_matrix), mean_returns)
    targets = np.linspace(min_ret, mean_returns.max(), n_points)
    n_assets = len(mean_returns)

    frontier = []
    for target in targets:
        constraints = (
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w, t=target: portfolio_return(w, mean_returns) - t},
        )
        result = minimize(lambda w: portfolio_volatility(w, cov_matrix),
                          np.full(n_assets, 1 / n_assets), method="SLSQP",
                          bounds=tuple((0.0, 1.0) for _ in range(n_assets)),
                          constraints=constraints)
        if result.success:
            frontier.append({"Return": target, "Volatility": result.fun})

    return pd.DataFrame(frontier)


def weights_table(weights: np.ndarray, tickers: list[str], label: str) -> pd.DataFrame:
    """Weights as a tidy percentage table, sorted largest first."""
    table = pd.DataFrame({label: weights * 100}, index=tickers)
    return table.sort_values(label, ascending=False).round(2)
