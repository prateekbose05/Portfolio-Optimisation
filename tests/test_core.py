"""Unit tests on synthetic data: no network required."""

import numpy as np
import pandas as pd
import pytest

from portfolio_optimiser.backtest import run_backtest
from portfolio_optimiser.data import annualised_inputs, log_returns, train_test_split
from portfolio_optimiser.metrics import (conditional_var, max_drawdown, performance_summary,
                                         portfolio_return, portfolio_volatility, sharpe_ratio,
                                         value_at_risk)
from portfolio_optimiser.optimisation import (efficient_frontier, max_sharpe_weights,
                                              min_volatility_weights, monte_carlo_portfolios)


@pytest.fixture
def prices() -> pd.DataFrame:
    """Synthetic GBM prices for four assets, 3 years of daily data."""
    rng = np.random.default_rng(0)
    n_days, n_assets = 756, 4
    drifts = np.array([0.08, 0.12, 0.05, 0.10]) / 252
    vols = np.array([0.15, 0.25, 0.10, 0.20]) / np.sqrt(252)
    shocks = rng.standard_normal((n_days, n_assets)) * vols + drifts
    prices = 100 * np.exp(np.cumsum(shocks, axis=0))
    index = pd.bdate_range("2021-01-01", periods=n_days)
    return pd.DataFrame(prices, index=index, columns=["AAA", "BBB", "CCC", "DDD"])


@pytest.fixture
def inputs(prices):
    return annualised_inputs(log_returns(prices))


def test_log_returns_shape(prices):
    returns = log_returns(prices)
    assert len(returns) == len(prices) - 1
    assert not returns.isna().any().any()


def test_monte_carlo_weights_valid(inputs):
    mean_returns, cov = inputs
    results, weights = monte_carlo_portfolios(mean_returns, cov, 0.04, 500)
    assert len(results) == 500
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-9)
    assert (weights >= 0).all()
    assert (results["Volatility"] > 0).all()


def test_analytical_beats_monte_carlo(inputs):
    """The exact optimiser must never lose to random sampling."""
    mean_returns, cov = inputs
    results, _ = monte_carlo_portfolios(mean_returns, cov, 0.04, 2000)
    w = max_sharpe_weights(mean_returns, cov, 0.04)
    exact = sharpe_ratio(portfolio_return(w, mean_returns), portfolio_volatility(w, cov), 0.04)
    assert exact >= results["Sharpe"].max() - 1e-6


def test_min_vol_below_all_assets(inputs):
    """Diversified minimum volatility cannot exceed the least volatile asset."""
    mean_returns, cov = inputs
    w = min_volatility_weights(mean_returns, cov)
    assert portfolio_volatility(w, cov) <= np.sqrt(np.diag(cov)).min() + 1e-9


def test_frontier_is_increasing(inputs):
    mean_returns, cov = inputs
    frontier = efficient_frontier(mean_returns, cov, 20)
    assert len(frontier) >= 15
    assert frontier["Return"].is_monotonic_increasing


def test_max_drawdown_signs():
    up_only = pd.Series([0.01] * 50)
    assert max_drawdown(up_only) == 0
    crash = pd.Series([0.01] * 10 + [-0.5] + [0.0] * 10)
    assert max_drawdown(crash) <= -0.5 + 1e-9


def test_var_ordering():
    rng = np.random.default_rng(1)
    returns = pd.Series(rng.standard_normal(2000) * 0.01)
    var = value_at_risk(returns)
    cvar = conditional_var(returns)
    assert cvar <= var < 0


def test_performance_summary_keys():
    rng = np.random.default_rng(2)
    daily = pd.Series(rng.standard_normal(500) * 0.01 + 0.0004,
                      index=pd.bdate_range("2023-01-01", periods=500))
    summary = performance_summary(daily, 0.04)
    assert {"Annualised return", "Sharpe ratio", "Max drawdown"} <= set(summary)


def test_train_test_split_chronological(prices):
    train, test = train_test_split(prices, 0.7)
    assert train.index.max() < test.index.min()
    assert len(train) + len(test) == len(prices)


def test_backtest_end_to_end(prices):
    result = run_backtest(prices, 0.04, 0.7)
    assert set(result["weights"]) == {"Max Sharpe", "Min Volatility", "Equal Weight"}
    assert result["summary"].shape[1] == 3
    assert not result["cumulative"].isna().any().any()
