"""Out-of-sample backtest.

Optimise weights on the training window only, hold them fixed through the test
window, and compare against an equal-weight portfolio and a market benchmark.
This is the honest test: in-sample frontier statistics always flatter.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import annualised_inputs, log_returns, train_test_split
from .metrics import performance_summary
from .optimisation import max_sharpe_weights, min_volatility_weights


def run_backtest(prices: pd.DataFrame, risk_free: float, split: float = 0.7,
                 benchmark_prices: pd.Series | None = None) -> dict:
    """Train on the first `split` of history, evaluate buy-and-hold weights on the rest."""
    train, test = train_test_split(prices, split)

    mean_returns, cov_matrix = annualised_inputs(log_returns(train))
    strategies = {
        "Max Sharpe": max_sharpe_weights(mean_returns, cov_matrix, risk_free),
        "Min Volatility": min_volatility_weights(mean_returns, cov_matrix),
        "Equal Weight": np.full(prices.shape[1], 1 / prices.shape[1]),
    }

    test_returns = log_returns(test)
    benchmark_returns = None
    if benchmark_prices is not None:
        benchmark_returns = np.log(benchmark_prices / benchmark_prices.shift(1)).dropna()
        benchmark_returns = benchmark_returns.loc[benchmark_returns.index.isin(test_returns.index)]

    summaries, cumulative = {}, {}
    for name, weights in strategies.items():
        daily = test_returns @ weights
        summaries[name] = performance_summary(daily, risk_free, benchmark_returns)
        cumulative[name] = (1 + daily).cumprod()

    if benchmark_returns is not None and not benchmark_returns.empty:
        summaries["Benchmark"] = performance_summary(benchmark_returns, risk_free,
                                                     benchmark_returns)
        cumulative["Benchmark"] = (1 + benchmark_returns).cumprod()

    return {
        "weights": {name: w for name, w in strategies.items()},
        "summary": pd.DataFrame(summaries),
        "cumulative": pd.DataFrame(cumulative).ffill(),
        "train_range": (train.index[0], train.index[-1]),
        "test_range": (test.index[0], test.index[-1]),
    }
