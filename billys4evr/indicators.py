"""Indicator calculation utilities."""

from __future__ import annotations

import pandas as pd
import pandas_ta as ta


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add a broad set of indicators to *df* using :mod:`pandas_ta`.

    This calls :func:`pandas_ta.strategy` to compute many indicators. The
    resulting dataframe contains the original OHLC columns plus hundreds of
    additional indicator columns. For performance reasons only the most recent
    500 rows are kept.
    """
    df = df.copy()
    # pandas_ta requires column names to be lowercase
    df.columns = [c.lower() for c in df.columns]
    df.ta.strategy("All")
    return df.tail(500)
