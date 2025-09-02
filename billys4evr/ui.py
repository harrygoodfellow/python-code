"""Graphical user interface for Billys4Evr."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

from .data import fetch_history
from .indicators import add_all_indicators
from .model import CandleModel


@dataclass
class Prediction:
    value: float
    direction: str
    confidence: float


class SplashScreen(QtWidgets.QSplashScreen):
    def __init__(self) -> None:
        pixmap = QtGui.QPixmap(400, 300)
        pixmap.fill(QtCore.Qt.white)
        painter = QtGui.QPainter(pixmap)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.setPen(QtGui.QPen(QtCore.Qt.darkBlue, 6))
        # Draw a simple "4" shape reminiscent of the requested logo
        painter.drawLine(200, 40, 200, 260)
        painter.drawLine(100, 160, 300, 160)
        painter.drawLine(100, 160, 200, 40)
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

        # Widgets
        self.chart = pg.PlotWidget()
        self.side_panel = QtWidgets.QTextEdit()
        self.side_panel.setReadOnly(True)
        self.side_dock = QtWidgets.QDockWidget("Details", self)
        self.side_dock.setWidget(self.side_panel)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.side_dock)
        self.setCentralWidget(self.chart)

        # Timeframe selector
        self.timeframe = QtWidgets.QComboBox()
        self.timeframe.addItems(["1d", "1wk", "1mo"])
        self.timeframe.currentTextChanged.connect(self.reload)
        toolbar = self.addToolBar("Toolbar")
        toolbar.addWidget(QtWidgets.QLabel("Interval:"))
        toolbar.addWidget(self.timeframe)

        self.model = CandleModel()
        self.reload()

    def reload(self) -> None:
        interval = self.timeframe.currentText()
        df = fetch_history("SPX", period="2y", interval=interval)
        df = add_all_indicators(df)
        self.model.fit(df.select_dtypes(float))
        pred = self.model.predict_next(df.select_dtypes(float))
        current = df["close"].iloc[-1]
        direction = self.model.direction(current, pred)
        confidence = float(np.abs(pred - current) / current)
        self.update_ui(df, Prediction(pred, direction, confidence))

    def update_ui(self, df: pd.DataFrame, prediction: Prediction) -> None:
        self.chart.clear()
        closes = df["close"].values
        x = np.arange(len(closes) + 1)
        y = np.append(closes, prediction.value)
        pen = pg.mkPen("b", width=2)
        self.chart.plot(x[:-1], y[:-1], pen=pen)
        self.chart.plot(x[-2:], y[-2:], pen=pg.mkPen("r", width=2))
        info = (
            f"Predicted next close: {prediction.value:.2f}\n"
            f"Direction: {prediction.direction}\n"
            f"Confidence: {prediction.confidence:.2%}\n"
        )
        self.side_panel.setPlainText(info)


def run() -> None:
    app = QtWidgets.QApplication(sys.argv)
    splash = SplashScreen()
    splash.show()
    QtCore.QTimer.singleShot(2000, splash.close)

    window = MainWindow()
    # Delay showing the main window until splash is closed
    splash.destroyed.connect(window.show)
    sys.exit(app.exec())


if __name__ == "__main__":
    run()
