"""Markdown results report, generated from a completed analysis run."""

from __future__ import annotations

import numpy as np
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


def _backtest_weights(context: dict) -> list[str]:
    """The weights the backtest actually held, which differ from the full-history optimum."""
    weights = context["backtest"].get("weights")
    if not weights:
        return []
    table = pd.DataFrame({f"{name} (%)": np.round(np.asarray(w) * 100, 2)
                          for name, w in weights.items()}, index=context["tickers"])
    table = table.sort_values(table.columns[0], ascending=False)
    return [
        "Weights held in the backtest (fitted on the training window, so they differ from "
        "the full-history optimum above):",
        "",
        table.to_markdown(),
        "",
    ]


def _cost_section(context: dict) -> list[str]:
    """The rebalancing and transaction-cost section, if the study was run."""
    study = context.get("cost_study")
    if study is None:
        return []
    cost_bps = context["cost_bps"]

    table = study.reset_index()
    for column in ["Gross return", "Net return", "Cost drag", "Annual turnover",
                   "Max weight drift"]:
        table[column] = table[column].map(_pct)
    table["Net Sharpe"] = table["Net Sharpe"].map(lambda x: f"{x:.2f}")

    lines = [
        "## Rebalancing and transaction costs",
        "",
        "The backtest above holds the fitted weights fixed and trades for free. Here each "
        "portfolio is bought on the first day of the test window and simulated on its actual "
        f"holdings, paying {cost_bps:g} basis points on every dollar bought or sold "
        "(including the initial purchase). Left alone, weights drift with prices; "
        "rebalancing trades them back to target at each month or quarter end.",
        "",
        table.to_markdown(index=False, disable_numparse=True),
        "",
    ]

    if "Max Sharpe" in study.index.get_level_values(0):
        ms = study.loc["Max Sharpe"]
        never, monthly = ms.loc["never"], ms.loc["monthly"]
        labels = {"never": "never rebalancing", "quarterly": "quarterly rebalancing",
                  "monthly": "monthly rebalancing"}
        if float(ms["Annual turnover"].astype(float).max()) < 1e-9:
            # One holding (or assets that never diverge): nothing to trade back to target.
            lines += [
                "The max-Sharpe portfolio fitted on the training window is a single holding, "
                "so there is nothing to rebalance: all three frequencies give the same "
                f"{_pct(float(never['Net return']))} a year after the "
                f"{_pct(float(never['Cost drag']))} cost of the initial purchase. The "
                "min-volatility and equal-weight rows show the effect of rebalancing on "
                "diversified portfolios.",
                "",
            ]
        else:
            best = ms["Net return"].astype(float).idxmax()
            lines += [
                f"For the max-Sharpe portfolio, the best net return came from "
                f"**{labels[best]}** ({_pct(float(ms.loc[best, 'Net return']))} a year "
                "after costs). "
                f"Never rebalancing cost {_pct(float(never['Cost drag']))} a year and let "
                f"one holding drift as far as {float(never['Max weight drift']) * 100:.1f} "
                "points from its target weight; "
                f"monthly rebalancing turned over {_pct(float(monthly['Annual turnover']))} "
                f"of the portfolio a year and cost {_pct(float(monthly['Cost drag']))}.",
                "",
            ]
        lines += [
            "Turnover is one-way (selling 5% to buy 5% is 5%). Gross returns differ by "
            "frequency because rebalancing changes what is held, not only what is paid: it "
            "sells winners and buys laggards, which helps when prices mean-revert and hurts "
            "when they trend.",
            "",
        ]
    return lines


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
        "through the unseen test window, with no trading costs.",
        "",
        *_backtest_weights(context),
        _format_summary(summary).to_markdown(),
        "",
        "![Out-of-sample performance](charts/backtest.png)",
        "",
        *_cost_section(context),
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
        "- Long-only and fully invested. Trading costs are a flat rate per dollar traded: "
        "no bid-ask spread model, market impact, minimum brokerage or tax.",
        "- Historical returns and covariances are weak predictors of future ones.",
        "- Single train/test split rather than walk-forward validation.",
        "- The headline backtest sums weighted log returns, a standard approximation; the "
        "cost study compounds simple returns on actual holdings.",
        "",
    ]
    return "\n".join(lines)
