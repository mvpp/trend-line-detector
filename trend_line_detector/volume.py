"""Volume context: which price each bar offers to pivots and lines.

pivot_price="volume" (default): high-volume bars (volume > multiplier ×
rolling SMA) offer their wicks (High for resistance, Low for support) —
heavy trade at an extreme carries weight; normal bars offer their body edges
(max/min of Open and Close).

pivot_price="wick": every bar offers its wicks. The volume flags are still
computed (pivot quality and line scores use them).
"""

from typing import NamedTuple

import numpy as np


class VolumeContext(NamedTuple):
    vol_sma: np.ndarray            # NaN until `lookback` bars exist
    is_high_volume: np.ndarray     # bool
    resistance_price: np.ndarray
    support_price: np.ndarray


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Trailing mean over `window` bars; NaN for the first window − 1."""
    out = np.full(len(values), np.nan)
    if window > 0 and len(values) >= window:
        out[window - 1:] = np.lib.stride_tricks.sliding_window_view(values, window).mean(axis=1)
    return out


def classify_volume(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray,
    volume: np.ndarray, lookback: int, multiplier: float, pivot_price: str = "volume",
) -> VolumeContext:
    vol_sma = rolling_mean(volume, lookback)
    with np.errstate(invalid="ignore"):
        is_high = np.nan_to_num(volume > multiplier * vol_sma, nan=0).astype(bool)
    if pivot_price not in ("volume", "wick"):
        raise ValueError(f"pivot_price must be 'volume' or 'wick', got {pivot_price!r}")
    wick = np.ones_like(is_high) if pivot_price == "wick" else is_high
    body_top = np.maximum(open_, close)
    body_bottom = np.minimum(open_, close)
    return VolumeContext(
        vol_sma=vol_sma,
        is_high_volume=is_high,
        resistance_price=np.where(wick, high, body_top),
        support_price=np.where(wick, low, body_bottom),
    )
