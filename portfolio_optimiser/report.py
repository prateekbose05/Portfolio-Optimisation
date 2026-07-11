"""Markdown results report, generated from a completed analysis run."""

from __future__ import annotations

import pandas as pd


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _format_summary(summary: pd.DataFrame) -> pd.DataFrame:
    formatted = summary.astype(object)
    percent_rows = ["Annualised return", "Annualised volatility", "Max drawdown",
                    "Daily VaR (95%)", "Daily CVaR (95%)"]
    for row in formatted.index:
        if row in percent_rows:
            formatted.loc[row] = summary.loc[row].map(_pct)
        else:
            formatted.loc[row] = summary.loc[row].map(lambda x: f"{x:.2f}")
    return formatted


def build_report(context: dict) -> str:
    """Render RESULTS.md from the analysis context produced by run_analysis.py."""
    summary = context["backtest"]["summary"]
    train_start, train_end = context["backtest"]["train_range"]
    test_start, test_end = context["backtest"]["test_range"]
    ms = context["max_sharpe"]

    best = summary.loc["Sharpe ratio"].astype(float).idxmax()
    benchmark_note = ""
    if "Benchmark" in summary.columns:
        excess = float(summary.loc["Annualised return", "Max Sharpe"]) - \
                 float(summary.loc["Annualised return", "Benchmark"])
        direction = "outperformed" if excess > 0 else "underperformed"
        benchmark_note = (
            f"Out of sample, the max-Sharpe portfolio {direction} the benchmark by "
            f"{_pct(abs(excess))} annualised.")

    lines = [
        "# Results",
        "",
        f"*Generated {pd.Timestamp.now().date()} on live Yahoo Finance data.*",
        "",
        "## Run parameters",
        "",
        f"- Tickers: {', '.join(context['tickers'])}",
        f"- History: {context['start']} to {context['end']}",
        f"- Risk-free rate: {_pct(context['risk_free'])} (RBA cash rate, configurable)",
        f"- Benchmark: {context['benchmark'] or 'none'}",
        f"- Monte Carlo simulations: {context['n_portfolios']:,}",
        f"- Train window: {train_start.date()} to {train_end.date()} "
        f"({int(context['split'] * 100)}%), test window: {test_start.date()} to {test_end.date()}",
        "",
        "## Optimal portfolios (in-sample, full history)",
        "",
        f"Maximum-Sharpe portfolio: expected return {_pct(ms['ret'])}, "
        f"volatility {_pct(ms['vol'])}, Sharpe {ms['sharpe']:.2f}.",
        "",
        context["weights_table"].to_markdown(),
        "",
        "The analytical (SLSQP) optimum had a Sharpe ratio of "
        f"{ms['sharpe']:.4f} against {context['mc_best_sharpe']:.4f} for the best of "
        f"{context['n_portfolios']:,} random portfolios, which is why the Monte Carlo cloud "
        "is used to map the feasible space while the exact weights come from the optimiser.",
        "",
        "![Efficient frontier](charts/efficient_frontier.png)",
        "",
        "![Allocations](charts/weights.png)",
        "",
        "![Correlations](charts/correlations.png)",
        "",
        "## Out-of-sample backtest",
        "",
        "Weights were fitted on the training window only, then held fixed (buy and hold) "
        "through the unseen test window.",
        "",
        _format_summary(summary).to_markdown(),
        "",
        "![Out-of-sample performance](charts/backtest.png)",
        "",
        "## Conclusion",
        "",
        f"On the test window, the strongest risk-adjusted performer was the {best} portfolio "
        f"(Sharpe {float(summary.loc['Sharpe ratio', best]):.2f}). {benchmark_note}",
        "",
        "The wider takeaway matches the literature: in-sample mean-variance weights are "
        "estimates, not guarantees. Expected returns are noisy, so optimised weights can "
        "concentrate in whatever happened to do well historically, and out-of-sample results "
        "routinely narrow the gap to naive equal weighting. The minimum-volatility portfolio, "
        "which depends only on the covariance matrix, tends to transfer out of sample more "
        "reliably than the max-Sharpe portfolio.",
        "",
        "## Limitations",
        "",
        "- Long-only, fully invested, no transaction costs, taxes or rebalancing.",
        "- Historical returns and covariances are weak predictors of future ones.",
        "- Single train/test split rather than walk-forward validation.",
        "- Log returns are summed for portfolio aggregation, a standard approximation.",
        "",
    ]
    return "\n".join(lines)
