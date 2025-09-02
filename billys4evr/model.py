"""Simple ML model for next-candle prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


class CandleModel:
    """A very small wrapper around :class:`LinearRegression`.

    This is **not** a state-of-the-art trading model but serves as a placeholder
    to demonstrate how the application can integrate a predictive model.
    """

    def __init__(self) -> None:
        self.model = LinearRegression()

    def _prepare(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        features = df.drop(columns=["close"]).values
        target = df["close"].shift(-1).dropna().values
        features = features[:-1]
        return features, target

    def fit(self, df: pd.DataFrame) -> None:
        x, y = self._prepare(df)
        self.model.fit(x, y)

    def predict_next(self, df: pd.DataFrame) -> float:
        last_features = df.drop(columns=["close"]).iloc[-1:].values
        return float(self.model.predict(last_features)[0])

    def direction(self, current: float, predicted: float) -> str:
        if predicted > current:
            return "up"
        if predicted < current:
            return "down"
        return "flat"
