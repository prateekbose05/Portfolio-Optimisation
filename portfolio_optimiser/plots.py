"""Chart builders. Each returns a matplotlib Figure so the CLI and the
Streamlit app can share them."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def efficient_frontier_plot(mc_results: pd.DataFrame, frontier: pd.DataFrame,
                            points: dict[str, tuple[float, float]]) -> plt.Figure:
    """Monte Carlo cloud, analytical frontier line, and the optimal portfolios."""
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(mc_results["Volatility"], mc_results["Return"],
                         c=mc_results["Sharpe"], cmap="viridis", alpha=0.35, s=8)
    fig.colorbar(scatter, ax=ax, label="Sharpe ratio")

    if not frontier.empty:
        ax.plot(frontier["Volatility"], frontier["Return"], "r--", lw=2,
                label="Efficient frontier (analytical)")

    markers = {"Max Sharpe": ("*", "red", 350), "Min Volatility": ("X", "royalblue", 180)}
    for name, (vol, ret) in points.items():
        marker, colour, size = markers.get(name, ("o", "black", 120))
        ax.scatter(vol, ret, c=colour, marker=marker, s=size, zorder=5,
                   edgecolors="black", linewidths=0.8, label=name)

    ax.set_xlabel("Annualised volatility")
    ax.set_ylabel("Annualised expected return")
    ax.set_title(f"Efficient frontier: {len(mc_results):,} simulated portfolios "
                 "vs analytical optimum")
    ax.legend()
    fig.tight_layout()
    return fig


def cumulative_returns_plot(cumulative: pd.DataFrame, test_start) -> plt.Figure:
    """Out-of-sample growth of $1 for each strategy."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for column in cumulative.columns:
        style = "--" if column == "Benchmark" else "-"
        ax.plot(cumulative.index, cumulative[column], style, lw=1.8, label=column)
    ax.set_ylabel("Growth of $1 (out-of-sample)")
    ax.set_title(f"Out-of-sample performance from {pd.Timestamp(test_start).date()}")
    ax.legend()
    fig.tight_layout()
    return fig


def weights_plot(weights: dict[str, np.ndarray], tickers: list[str]) -> plt.Figure:
    """Side-by-side allocation bars for each optimised strategy."""
    frame = pd.DataFrame(weights, index=tickers) * 100
    fig, ax = plt.subplots(figsize=(10, 5))
    frame.plot(kind="bar", ax=ax)
    ax.set_ylabel("Weight (%)")
    ax.set_title("Optimised portfolio allocations")
    ax.legend(title=None)
    fig.tight_layout()
    return fig


def correlation_plot(returns: pd.DataFrame) -> plt.Figure:
    """Correlation heatmap of daily returns: the diversification story."""
    corr = returns.corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(corr, cmap="RdYlGn_r", vmin=-1, vmax=1)
    fig.colorbar(image, ax=ax, label="Correlation")
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr)), corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title("Daily return correlations")
    fig.tight_layout()
    return fig
