# Billys4Evr

A desktop application that monitors the SPX 500 and NASDAQ, computes hundreds of
technical indicators, and uses a lightweight machine-learning model to predict
the next candle. Predictions are shown alongside direction and a confidence
score in a PySide6 interface.

## Features

* Fetches market data from Yahoo Finance via `yfinance`
* Computes a wide range of indicators using `pandas-ta`
* Trains a simple `scikit-learn` model for next close prediction
* Splash screen with a minimalist "4" logo before loading the main dashboard
* Main dashboard shows a chart with the predicted next close and a collapsible
  side panel with detailed information
* Symbol selector (SPX or NASDAQ) and interval selector (1m to 1mo)

## Usage

Install requirements and run the application:

```bash
pip install -r requirements.txt
python billys4evr.py
```

The app starts with a splash screen while data and indicators load and then
switches to the main dashboard displaying the latest data and the next predicted
candle.

> **Note**
> `pandas-ta` currently requires `numpy` 1.x. If you see an error like
> `ImportError: cannot import name 'NaN' from 'numpy'`, ensure your environment
> uses `numpy<2` as specified in `requirements.txt`.
