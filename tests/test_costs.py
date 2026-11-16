"""Tests for transaction costs and rebalancing: synthetic data, no network."""

import numpy as np
import pandas as pd
import pytest

from portfolio_optimiser.costs import rebalancing_study, simulate_portfolio


@pytest.fixture
def prices() -> pd.DataFrame:
    """Synthetic GBM prices for three assets with very different drifts."""
    rng = np.random.default_rng(7)
    n_days = 504
    drifts = np.array([0.30, 0.02, -0.10]) / 252
    vols = np.array([0.25, 0.10, 0.20]) / np.sqrt(252)
    shocks = rng.standard_normal((n_days, 3)) * vols + drifts
    index = pd.bdate_range("2022-01-03", periods=n_days)
    return pd.DataFrame(100 * np.exp(np.cumsum(shocks, axis=0)), index=index,
                        columns=["AAA", "BBB", "CCC"])


TARGET = np.array([0.5, 0.3, 0.2])


def test_free_buy_and_hold_matches_price_relatives(prices):
    """With no costs and no rebalancing, wealth is the weighted price relative."""
    run = simulate_portfolio(prices, TARGET, "never", cost_bps=0)
    wealth = np.exp(run["net_returns"].sum())
    expected = float(TARGET @ (prices.iloc[-1] / prices.iloc[0]).to_numpy())
    assert wealth == pytest.approx(expected)
    assert run["rebalances"] == 0 and run["annual_turnover"] == 0


def test_costs_never_help(prices):
    for rebalance in ("never", "quarterly", "monthly"):
        run = simulate_portfolio(prices, TARGET, rebalance, cost_bps=25)
        assert run["net_returns"].sum() < run["gross_returns"].sum()


def test_entry_cost_is_charged_once(prices):
    """Buy-and-hold pays the cost rate on the initial purchase and nothing else."""
    run = simulate_portfolio(prices, TARGET, "never", cost_bps=50)
    drag = run["gross_returns"].sum() - run["net_returns"].sum()
    assert drag == pytest.approx(-np.log(1 - 0.005))
    assert run["total_cost"] == pytest.approx(0.005)


def test_more_frequent_rebalancing_trades_more(prices):
    quarterly = simulate_portfolio(prices, TARGET, "quarterly")
    monthly = simulate_portfolio(prices, TARGET, "monthly")
    assert monthly["rebalances"] > quarterly["rebalances"] > 0
    assert monthly["annual_turnover"] > quarterly["annual_turnover"] > 0


def test_rebalancing_limits_drift(prices):
    """Left alone the winner takes over; rebalanced, weights stay near target."""
    drifted = simulate_portfolio(prices, TARGET, "never")
    held = simulate_portfolio(prices, TARGET, "monthly")
    assert held["max_drift"] < drifted["max_drift"]
    assert drifted["final_weights"].sum() == pytest.approx(1.0)


def test_no_trading_when_assets_move_together():
    """Identical returns mean no drift, so rebalancing has nothing to trade."""
    index = pd.bdate_range("2023-01-02", periods=260)
    path = 100 * np.exp(np.linspace(0, 0.2, 260))
    prices = pd.DataFrame({"AAA": path, "BBB": path * 2}, index=index)
    run = simulate_portfolio(prices, np.array([0.6, 0.4]), "monthly", cost_bps=10)
    assert run["rebalances"] > 0
    assert run["annual_turnover"] == pytest.approx(0, abs=1e-12)


def test_rebalancing_study_shape(prices):
    study = rebalancing_study(prices, {"A": TARGET, "B": np.full(3, 1 / 3)}, 0.04, 10)
    assert len(study) == 6
    assert (study["Cost drag"] > 0).all()
    assert study.loc[("A", "never"), "Rebalances"] == 0


def test_invalid_inputs(prices):
    with pytest.raises(ValueError):
        simulate_portfolio(prices, TARGET, "weekly")
    with pytest.raises(ValueError):
        simulate_portfolio(prices, np.array([0.5, 0.3]), "never")
    with pytest.raises(ValueError):
        simulate_portfolio(prices, TARGET, "never", cost_bps=-1)


def test_report_handles_a_single_holding_and_shows_backtest_weights(prices):
    """A one-stock max-Sharpe portfolio has nothing to rebalance; the report must say so."""
    from portfolio_optimiser.report import _backtest_weights, _cost_section

    n = prices.shape[1]
    single = np.zeros(n)
    single[0] = 1.0
    weights = {"Max Sharpe": single, "Equal Weight": np.full(n, 1 / n)}
    study = rebalancing_study(prices, weights, 0.04, 10.0)
    context = {"cost_study": study, "cost_bps": 10.0, "tickers": list(prices.columns),
               "backtest": {"weights": weights}}

    text = "\n".join(_cost_section(context))
    assert "single holding" in text
    assert "drift as far as" not in text

    table = "\n".join(_backtest_weights(context))
    assert "Max Sharpe (%)" in table and "100" in table
