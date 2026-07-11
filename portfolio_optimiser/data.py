"""Price data retrieval and return calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import TRADING_DAYS


def fetch_prices(tickers: list[str], start: str, end: str | None = None) -> pd.DataFrame:
    """Download adjusted close prices from Yahoo Finance for any tickers.

    Works with any market Yahoo covers (ASX tickers use the .AX suffix,
    e.g. CBA.AX; US tickers need no suffix). Tickers that return no data
    are dropped with a warning rather than failing the whole run.
    """
    import yfinance as yf

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    if isinstance(raw, pd.Series):
        raw = raw.to_frame(name=tickers[0])

    prices = raw.dropna(axis=1, how="all")
    dropped = sorted(set(tickers) - set(prices.columns))
    if dropped:
        print(f"Warning: no data for {', '.join(dropped)}; continuing without them.")
    if prices.shape[1] < 2:
        raise ValueError("Need at least two tickers with valid data to build a portfolio.")

    return prices.dropna(how="any")


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns."""
    return np.log(prices / prices.shift(1)).dropna(how="any")


def annualised_inputs(returns: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Annualised mean returns and covariance matrix from daily returns."""
    return returns.mean() * TRADING_DAYS, returns.cov() * TRADING_DAYS


def train_test_split(prices: pd.DataFrame, split: float = 0.7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split: optimise on the first part, evaluate on the rest."""
    if not 0.5 <= split <= 0.9:
        raise ValueError("split should be between 0.5 and 0.9")
    cut = int(len(prices) * split)
    return prices.iloc[:cut], prices.iloc[cut:]
