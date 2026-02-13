"""Fetch OHLCV stock data from Yahoo Finance via yfinance."""

import pandas as pd
import yfinance as yf


def fetch_ohlcv(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """Download OHLCV data for a ticker.

    Args:
        ticker: Stock ticker symbol (e.g. "AAPL").
        period: yfinance period string (e.g. "1y", "6mo", "2y").
        interval: yfinance interval string (e.g. "1d", "1h").

    Returns:
        DataFrame with DatetimeIndex and columns [Open, High, Low, Close, Volume].

    Raises:
        ValueError: If no data is returned for the given ticker/period.
    """
    df = yf.download(ticker, period=period, interval=interval, progress=False)

    if df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}' with period='{period}'")

    # yfinance >= 0.2.51 returns MultiIndex columns for single tickers
    if isinstance(df.columns, pd.MultiIndex):
        df = df.droplevel("Ticker", axis=1)

    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in downloaded data: {missing}")

    df = df[required].dropna()
    df = df.sort_index()

    if df.empty:
        raise ValueError(f"All rows were NaN for ticker '{ticker}'")

    return df
