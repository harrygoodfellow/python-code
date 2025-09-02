"""Billys4Evr single-file application.

This desktop app fetches SPX 500 or NASDAQ historical data, computes a wide
range of technical indicators using pandas-ta, trains a simple linear-regression
model on the fly, and predicts the next closing price. The UI shows the latest
prices with an extra point for the predicted next candle along with direction
and a confidence score.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf
from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
from sklearn.linear_model import LinearRegression

# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------

SYMBOLS = {
    "SPX": "^GSPC",
    "NASDAQ": "^IXIC",
}

# Map logical interval names to yfinance intervals. "10m" is resampled from 1m.
INTERVALS = {
    "1m": "1m",
    "10m": "1m",
    "30m": "30m",
    "1h": "60m",
    "1d": "1d",
    "1wk": "1wk",
    "1mo": "1mo",
}


def fetch_history(
    symbol: Literal["SPX", "NASDAQ"],
    interval: Literal["1m", "10m", "30m", "1h", "1d", "1wk", "1mo"],
) -> pd.DataFrame:
    """Fetch recent price history for *symbol* at *interval*.

    For the special 10-minute interval, 1-minute data is resampled.
    """

    ticker = SYMBOLS[symbol]

    # Determine period based on interval (yfinance restrictions)
    if interval in ("1m", "10m"):
        period = "7d"
    elif interval in ("30m", "1h"):
        period = "60d"
    else:
        period = "2y"

    df = yf.download(ticker, period=period, interval=INTERVALS[interval], auto_adjust=True)

    if interval == "10m":
        df = df.resample("10T").agg(
            {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
        )

    df.dropna(inplace=True)
    df.rename(columns=str.lower, inplace=True)
    return df


# ---------------------------------------------------------------------------
# Indicator utilities
# ---------------------------------------------------------------------------

def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Append a broad set of technical indicators using pandas-ta."""

    df = df.copy()
    df.ta.strategy("All")
    return df.tail(500)


# ---------------------------------------------------------------------------
# Prediction model
# ---------------------------------------------------------------------------

class CandleModel:
    """Linear regression model predicting the next closing price."""

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

    @staticmethod
    def direction(current: float, predicted: float) -> str:
        if predicted > current:
            return "up"
        if predicted < current:
            return "down"
        return "flat"


# ---------------------------------------------------------------------------
# UI Components
# ---------------------------------------------------------------------------

@dataclass
class Prediction:
    value: float
    direction: str
    confidence: float


class SplashScreen(QtWidgets.QSplashScreen):
    """Simple splash screen with a "4" logo and app name."""

    def __init__(self) -> None:
        pixmap = QtGui.QPixmap(400, 300)
        pixmap.fill(QtCore.Qt.white)
        painter = QtGui.QPainter(pixmap)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.setPen(QtGui.QPen(QtCore.Qt.darkBlue, 6))
        painter.drawLine(200, 40, 200, 260)  # vertical of the "4"
        painter.drawLine(100, 160, 300, 160)  # horizontal of the "4"
        painter.drawLine(100, 160, 200, 40)   # diagonal of the "4"
        painter.setPen(QtGui.QPen(QtCore.Qt.black))
        font = QtGui.QFont("Sans", 24, QtGui.QFont.Bold)
        painter.setFont(font)
        painter.drawText(pixmap.rect(), QtCore.Qt.AlignCenter, "ever")
        painter.end()
        super().__init__(pixmap)
        self.showMessage(
            "Billys4Evr", QtCore.Qt.AlignBottom | QtCore.Qt.AlignHCenter, QtCore.Qt.black
        )


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Billys4Evr")
        self.resize(1000, 600)

        pg.setConfigOptions(antialias=True)

        # Widgets
        self.chart = pg.PlotWidget()
        self.side_panel = QtWidgets.QTextEdit()
        self.side_panel.setReadOnly(True)
        self.side_dock = QtWidgets.QDockWidget("Details", self)
        self.side_dock.setWidget(self.side_panel)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.side_dock)
        self.setCentralWidget(self.chart)

        # Toolbars
        toolbar = self.addToolBar("Toolbar")
        toolbar.addWidget(QtWidgets.QLabel("Symbol:"))
        self.symbol_box = QtWidgets.QComboBox()
        self.symbol_box.addItems(sorted(SYMBOLS.keys()))
        toolbar.addWidget(self.symbol_box)
        toolbar.addSeparator()
        toolbar.addWidget(QtWidgets.QLabel("Interval:"))
        self.timeframe = QtWidgets.QComboBox()
        self.timeframe.addItems(["1m", "10m", "30m", "1h", "1d", "1wk", "1mo"])
        toolbar.addWidget(self.timeframe)

        self.symbol_box.currentTextChanged.connect(self.reload)
        self.timeframe.currentTextChanged.connect(self.reload)

        self.model = CandleModel()
        self.reload()

    # ------------------------------------------------------------------
    def reload(self) -> None:
        symbol = self.symbol_box.currentText()
        interval = self.timeframe.currentText()
        df = fetch_history(symbol, interval)
        df = add_all_indicators(df)
        numeric_df = df.select_dtypes("number").dropna(axis=1)
        self.model.fit(numeric_df)
        pred = self.model.predict_next(numeric_df)
        current = numeric_df["close"].iloc[-1]
        direction = self.model.direction(current, pred)
        confidence = max(0.0, 1 - abs(pred - current) / current)
        self.update_ui(df, Prediction(pred, direction, confidence), symbol, interval)

    # ------------------------------------------------------------------
    def update_ui(
        self,
        df: pd.DataFrame,
        prediction: Prediction,
        symbol: str,
        interval: str,
    ) -> None:
        self.chart.clear()
        closes = df["close"].values
        x = np.arange(len(closes) + 1)
        y = np.append(closes, prediction.value)
        pen = pg.mkPen("b", width=2)
        self.chart.plot(x[:-1], y[:-1], pen=pen)
        self.chart.plot(x[-2:], y[-2:], pen=pg.mkPen("r", width=2))
        info = (
            f"Symbol: {symbol}\n"
            f"Interval: {interval}\n"
            f"Predicted next close: {prediction.value:.2f}\n"
            f"Direction: {prediction.direction}\n"
            f"Confidence: {prediction.confidence:.2%}"
        )
        self.side_panel.setPlainText(info)


# ---------------------------------------------------------------------------
# Application entry point
# ---------------------------------------------------------------------------

def run() -> None:
    app = QtWidgets.QApplication(sys.argv)
    splash = SplashScreen()
    splash.show()
    app.processEvents()

    window = MainWindow()
    window.show()
    splash.finish(window)

    sys.exit(app.exec())


if __name__ == "__main__":
    run()
