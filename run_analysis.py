"""Run the full portfolio analysis for any tickers, on live data.

Examples:
    python run_analysis.py
    python run_analysis.py --tickers CBA.AX CSL.AX WES.AX BHP.AX XRO.AX
    python run_analysis.py --tickers AAPL MSFT NVDA JPM --benchmark ^GSPC --rf 0.043
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from portfolio_optimiser.backtest import run_backtest
from portfolio_optimiser.data import annualised_inputs, fetch_prices, log_returns
from portfolio_optimiser.metrics import portfolio_return, portfolio_volatility, sharpe_ratio
from portfolio_optimiser.optimisation import (efficient_frontier, max_sharpe_weights,
                                              min_volatility_weights, monte_carlo_portfolios,
                                              weights_table)
from portfolio_optimiser.plots import (correlation_plot, cumulative_returns_plot,
                                       efficient_frontier_plot, weights_plot)
from portfolio_optimiser.report import build_report

DEFAULT_TICKERS = ["CBA.AX", "MQG.AX", "NAB.AX", "XRO.AX", "BHP.AX"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MPT portfolio optimisation on live data.")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS,
                        help="Any Yahoo Finance tickers (ASX needs .AX suffix).")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default=None, help="Defaults to today (live data).")
    parser.add_argument("--benchmark", default="^AXJO", help="Benchmark index; '' to disable.")
    parser.add_argument("--rf", type=float, default=0.0435,
                        help="Annual risk-free rate (default: RBA cash rate 4.35%%).")
    parser.add_argument("--portfolios", type=int, default=10_000)
    parser.add_argument("--split", type=float, default=0.7,
                        help="Fraction of history used for training in the backtest.")
    parser.add_argument("--outdir", default="docs/results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    charts = outdir / "charts"
    charts.mkdir(parents=True, exist_ok=True)

    end = args.end or pd.Timestamp.now().strftime("%Y-%m-%d")
    print(f"Fetching {', '.join(args.tickers)} from {args.start} to {end}...")
    prices = fetch_prices(args.tickers, args.start, args.end)
    tickers = list(prices.columns)
    returns = log_returns(prices)
    mean_returns, cov_matrix = annualised_inputs(returns)

    print(f"Running {args.portfolios:,} Monte Carlo simulations and analytical optimisation...")
    mc_results, _ = monte_carlo_portfolios(mean_returns, cov_matrix, args.rf, args.portfolios)
    ms_weights = max_sharpe_weights(mean_returns, cov_matrix, args.rf)
    mv_weights = min_volatility_weights(mean_returns, cov_matrix)
    frontier = efficient_frontier(mean_returns, cov_matrix)

    ms_ret = portfolio_return(ms_weights, mean_returns)
    ms_vol = portfolio_volatility(ms_weights, cov_matrix)
    points = {
        "Max Sharpe": (ms_vol, ms_ret),
        "Min Volatility": (portfolio_volatility(mv_weights, cov_matrix),
                           portfolio_return(mv_weights, mean_returns)),
    }

    benchmark_prices = None
    if args.benchmark:
        try:
            benchmark_prices = fetch_prices([args.benchmark, args.tickers[0]],
                                            args.start, args.end)[args.benchmark]
        except (ValueError, KeyError):
            print(f"Warning: benchmark {args.benchmark} unavailable; skipping comparison.")

    print("Backtesting out of sample...")
    backtest = run_backtest(prices, args.rf, args.split, benchmark_prices)

    efficient_frontier_plot(mc_results, frontier, points).savefig(
        charts / "efficient_frontier.png", dpi=200)
    weights_plot({"Max Sharpe": ms_weights, "Min Volatility": mv_weights}, tickers).savefig(
        charts / "weights.png", dpi=200)
    correlation_plot(returns).savefig(charts / "correlations.png", dpi=200)
    cumulative_returns_plot(backtest["cumulative"], backtest["test_range"][0]).savefig(
        charts / "backtest.png", dpi=200)

    table = weights_table(ms_weights, tickers, "Max Sharpe (%)").join(
        weights_table(mv_weights, tickers, "Min Volatility (%)"))

    context = {
        "tickers": tickers, "start": args.start, "end": end, "risk_free": args.rf,
        "benchmark": args.benchmark, "n_portfolios": args.portfolios, "split": args.split,
        "max_sharpe": {"ret": ms_ret, "vol": ms_vol,
                       "sharpe": sharpe_ratio(ms_ret, ms_vol, args.rf)},
        "mc_best_sharpe": float(mc_results["Sharpe"].max()),
        "weights_table": table, "backtest": backtest,
    }
    (outdir / "RESULTS.md").write_text(build_report(context))

    with pd.ExcelWriter(outdir / "portfolio_results.xlsx") as writer:
        mc_results.to_excel(writer, sheet_name="Monte Carlo", index=False)
        table.to_excel(writer, sheet_name="Optimal Weights")
        backtest["summary"].to_excel(writer, sheet_name="Backtest")

    print(f"\nDone. Report: {outdir / 'RESULTS.md'}, charts: {charts}/")
    print(f"Max Sharpe {context['max_sharpe']['sharpe']:.2f} "
          f"(return {ms_ret:.1%}, volatility {ms_vol:.1%})")


if __name__ == "__main__":
    main()
