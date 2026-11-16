"""Interactive portfolio optimiser. Run with: python3 run.py"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from portfolio_optimiser.backtest import run_backtest
from portfolio_optimiser.costs import rebalancing_study
from portfolio_optimiser.data import (annualised_inputs, fetch_prices, log_returns,
                                      train_test_split)
from portfolio_optimiser.metrics import portfolio_return, portfolio_volatility, sharpe_ratio
from portfolio_optimiser.optimisation import (efficient_frontier, max_sharpe_weights,
                                              min_volatility_weights, monte_carlo_portfolios,
                                              weights_table)
from portfolio_optimiser.plots import (correlation_plot, cumulative_returns_plot,
                                       efficient_frontier_plot, weights_plot)

st.set_page_config(page_title="Portfolio Optimiser", page_icon="📈", layout="wide")
st.title("Portfolio Optimiser")
st.caption("Modern Portfolio Theory on live market data: Monte Carlo simulation, "
           "analytical optimisation and an out-of-sample backtest. Built by Prateek Bose.")

with st.sidebar:
    st.header("Inputs")
    tickers_text = st.text_input("Tickers (comma-separated, ASX needs .AX)",
                                 "CBA.AX, MQG.AX, NAB.AX, XRO.AX, BHP.AX")
    start = st.date_input("Start date", pd.Timestamp("2020-01-01"))
    benchmark = st.text_input("Benchmark index", "^AXJO")
    risk_free = st.number_input("Risk-free rate (annual)", 0.0, 0.15, 0.0435, 0.0025,
                                format="%.4f", help="Default: RBA cash rate.")
    n_sims = st.select_slider("Monte Carlo simulations", [1_000, 5_000, 10_000, 25_000], 10_000)
    split = st.slider("Backtest training share", 0.5, 0.9, 0.7, 0.05)
    cost_bps = st.number_input("Trading cost (basis points per dollar traded)", 0.0, 100.0,
                               10.0, 5.0)
    run = st.button("Run analysis", type="primary")


@st.cache_data(show_spinner="Fetching live prices...")
def load_prices(tickers: tuple[str, ...], start: str, benchmark: str):
    prices = fetch_prices(list(tickers), start)
    benchmark_prices = None
    if benchmark:
        try:
            benchmark_prices = fetch_prices([benchmark, tickers[0]], start)[benchmark]
        except (ValueError, KeyError):
            st.warning(f"Benchmark {benchmark} unavailable; running without it.")
    return prices, benchmark_prices


if run:
    tickers = tuple(t.strip().upper() for t in tickers_text.split(",") if t.strip())
    if len(tickers) < 2:
        st.error("Enter at least two tickers.")
        st.stop()

    try:
        prices, benchmark_prices = load_prices(tickers, str(start), benchmark.strip())
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    returns = log_returns(prices)
    mean_returns, cov_matrix = annualised_inputs(returns)
    mc_results, _ = monte_carlo_portfolios(mean_returns, cov_matrix, risk_free, n_sims)
    ms_weights = max_sharpe_weights(mean_returns, cov_matrix, risk_free)
    mv_weights = min_volatility_weights(mean_returns, cov_matrix)
    frontier = efficient_frontier(mean_returns, cov_matrix)
    backtest = run_backtest(prices, risk_free, split, benchmark_prices)
    _, test_prices = train_test_split(prices, split)
    cost_study = rebalancing_study(test_prices, backtest["weights"], risk_free, cost_bps)

    ms_ret = portfolio_return(ms_weights, mean_returns)
    ms_vol = portfolio_volatility(ms_weights, cov_matrix)

    col1, col2, col3 = st.columns(3)
    col1.metric("Max Sharpe ratio", f"{sharpe_ratio(ms_ret, ms_vol, risk_free):.2f}")
    col2.metric("Expected return", f"{ms_ret:.1%}")
    col3.metric("Volatility", f"{ms_vol:.1%}")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Efficient frontier", "Allocations", "Out-of-sample backtest", "Data"])

    with tab1:
        points = {"Max Sharpe": (ms_vol, ms_ret),
                  "Min Volatility": (portfolio_volatility(mv_weights, cov_matrix),
                                     portfolio_return(mv_weights, mean_returns))}
        st.pyplot(efficient_frontier_plot(mc_results, frontier, points))

    with tab2:
        left, right = st.columns([1, 2])
        table = weights_table(ms_weights, list(prices.columns), "Max Sharpe (%)").join(
            weights_table(mv_weights, list(prices.columns), "Min Volatility (%)"))
        left.dataframe(table, width="stretch")
        right.pyplot(weights_plot({"Max Sharpe": ms_weights, "Min Volatility": mv_weights},
                                  list(prices.columns)))
        st.pyplot(correlation_plot(returns))

    with tab3:
        train_start, train_end = backtest["train_range"]
        st.caption(f"Weights fitted on {train_start.date()} to {train_end.date()}, "
                   "then held fixed through the unseen test window.")
        st.dataframe(backtest["summary"].style.format("{:.4f}"), width="stretch")
        st.pyplot(cumulative_returns_plot(backtest["cumulative"], backtest["test_range"][0]))

        st.subheader("Weights held in the backtest")
        st.caption("Fitted on the training window, so they differ from the full-history "
                   "optimum on the Allocations tab.")
        held = pd.DataFrame({f"{name} (%)": np.round(np.asarray(w) * 100, 2)
                             for name, w in backtest["weights"].items()},
                            index=list(prices.columns))
        st.dataframe(held, width="stretch")

        st.subheader("Rebalancing and trading costs")
        st.caption(f"Each portfolio is bought on the first day of the test window and "
                   f"simulated on its actual holdings, paying {cost_bps:g} basis points on "
                   "every dollar traded. Turnover is one-way.")
        percent = ["Gross return", "Net return", "Cost drag", "Annual turnover",
                   "Max weight drift"]
        st.dataframe(cost_study.reset_index().style.format(
            {**{column: "{:.2%}" for column in percent}, "Net Sharpe": "{:.2f}"}),
            width="stretch", hide_index=True)

    with tab4:
        st.line_chart(prices / prices.iloc[0])
        st.dataframe(prices.tail(10), width="stretch")
else:
    st.info("Set your tickers in the sidebar and hit Run analysis. "
            "Any Yahoo Finance tickers work: ASX (CBA.AX), US (AAPL), ETFs (VAS.AX).")
