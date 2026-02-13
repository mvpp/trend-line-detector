"""Volume context classification for OHLCV candles.

Classifies each candle as high-volume or normal-volume, then selects the
appropriate price for support/resistance pivot detection:
  - High volume: High (resistance) / Low (support) — wicks are meaningful
  - Normal volume: max(Open,Close) (resistance) / min(Open,Close) (support) — body edges
"""

import numpy as np
import pandas as pd


def classify_volume_context(
    df: pd.DataFrame,
    lookback: int = 20,
    high_vol_multiplier: float = 1.5,
) -> pd.DataFrame:
    """Add volume-adaptive price columns to the OHLCV DataFrame.

    Args:
        df: DataFrame with columns [Open, High, Low, Close, Volume].
        lookback: Window for rolling volume SMA (default 20 = ~1 trading month).
        high_vol_multiplier: Volume must exceed this multiple of SMA to be
            classified as high-volume (default 1.5).

    Returns:
        The same DataFrame with 4 new columns added:
          VolSMA           - rolling volume average
          IsHighVolume     - bool flag
          ResistancePrice  - price to use for resistance pivots
          SupportPrice     - price to use for support pivots
    """
    vol_sma = df["Volume"].rolling(window=lookback, min_periods=lookback).mean()

    is_high_vol = df["Volume"] > (high_vol_multiplier * vol_sma)
    is_high_vol = is_high_vol.fillna(False).astype(bool)

    body_top = np.maximum(df["Open"].values, df["Close"].values)
    body_bottom = np.minimum(df["Open"].values, df["Close"].values)

    df["VolSMA"] = vol_sma
    df["IsHighVolume"] = is_high_vol

    df["ResistancePrice"] = np.where(is_high_vol, df["High"].values, body_top)
    df["SupportPrice"] = np.where(is_high_vol, df["Low"].values, body_bottom)

    return df
