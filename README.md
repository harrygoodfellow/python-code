# Billys4Evr

A desktop prototype that fetches SPX 500 or NASDAQ data, computes a large set
of technical indicators and uses a lightweight machine learning model to
predict the next candle. The prediction along with its direction and a
confidence heuristic are displayed in a PySide6 interface.

## Features

* Fetches market data from Yahoo Finance via `yfinance`
* Computes hundreds of indicators using `pandas-ta`
* Trains a simple `scikit-learn` model for next close prediction
* Splash screen with a minimalist "4" logo before loading the main dashboard
* Main dashboard shows a line chart with the predicted next close and a
  collapsible side panel with additional information

## Usage

Install requirements and run the application:

```bash
pip install -r requirements.txt
python -m billys4evr
```

The app will start with a splash screen while data and indicators are loaded,
then show the main dashboard.
