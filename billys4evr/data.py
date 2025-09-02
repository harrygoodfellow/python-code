"""Data fetching utilities for Billys4Evr."""

from __future__ import annotations

import datetime as dt
from typing import Literal

import pandas as pd
import yfinance as yf

SYMBOLS = {
    "SPX": "^GSPC",
    "NASDAQ": "^IXIC",
}

INTERVALS = {
    "1m": "1m",
    "5m": "5m",
    "30m": "30m",
    "1h": "60m",
    "1d": "1d",
    "1wk": "1wk",
    "1mo": "1mo",
}


def fetch_history(
    symbol: Literal["SPX", "NASDAQ"],
    period: str = "60d",
    interval: str = "1d",
) -> pd.DataFrame:
    """Fetch historical data for the given symbol.

    Parameters
    ----------
    symbol:
        Logical name of the instrument.
    period:
        Period argument for yfinance (e.g. "60d", "1y").
    interval:
        Interval string supported by yfinance (e.g. "1m", "1d").
    """
    ticker = SYMBOLS[symbol]
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True)
    df.dropna(inplace=True)
    return df
