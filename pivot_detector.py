"""Williams Fractal pivot detection on volume-adaptive prices.

Detects local maxima of ResistancePrice (resistance pivots) and local minima
of SupportPrice (support pivots) using a left/right span window.  If too few
pivots are found, the span is reduced adaptively down to a minimum of 2.

Boundary bars (first left_span and last right_span) are checked with partial
context so that edge extremes (e.g. an all-time high near the chart end) are
not missed.

Each pivot is assigned a quality score based on three signals:
  - Prominence: how far the pivot stands out from its neighbors
  - Volume strength: volume at the pivot relative to the rolling SMA
  - Bounce: how strongly price reversed in the 3 bars after the pivot
"""

from typing import Tuple

import numpy as np
import pandas as pd

from config import (
    BOUNCE_FLOOR,
    BOUNCE_LOOKAHEAD_BARS,
    MIN_PIVOTS_FALLBACK,
    MIN_SPAN_FALLBACK,
    PIVOT_QUALITY_BOUNCE_EXP,
    PIVOT_QUALITY_PROMINENCE_EXP,
    PIVOT_QUALITY_VOLUME_EXP,
    VOLUME_STRENGTH_MAX,
    VOLUME_STRENGTH_MIN,
)


# ──────────────────────────────────────────────────────────────────────
# Pivot quality helpers
# ──────────────────────────────────────────────────────────────────────

def _compute_prominence(
    bar_idx: int,
    price: float,
    pivot_type: str,
    price_col: np.ndarray,
    price_range: float,
    left_span: int,
    right_span: int,
    n: int,
) -> float:
    """How much the pivot stands out from its neighbors.

    For resistance: (price - mean(neighbors)) / price_range
    For support:    (mean(neighbors) - price) / price_range

    Neighbors are same-type prices within the fractal window.
    Returns a value clamped to [0, 1].
    """
    left_start = max(0, bar_idx - left_span)
    right_end = min(n, bar_idx + right_span + 1)

    neighbors = []
    for idx in range(left_start, bar_idx):
        neighbors.append(price_col[idx])
    for idx in range(bar_idx + 1, right_end):
        neighbors.append(price_col[idx])

    if not neighbors or price_range <= 0:
        return 0.0

    mean_neighbors = sum(neighbors) / len(neighbors)

    if pivot_type == "resistance":
        raw = (price - mean_neighbors) / price_range
    else:  # support
        raw = (mean_neighbors - price) / price_range

    return float(np.clip(raw, 0.0, 1.0))


def _compute_volume_strength(
    bar_idx: int,
    vol_array: np.ndarray,
    vol_sma_array: np.ndarray,
) -> float:
    """Volume at the pivot bar relative to the rolling SMA.

    Returns a value clamped to [VOLUME_STRENGTH_MIN, VOLUME_STRENGTH_MAX]
    then normalized to [0, 1] by dividing by VOLUME_STRENGTH_MAX.
    """
    sma_val = vol_sma_array[bar_idx]
    if np.isnan(sma_val) or sma_val <= 0:
        return VOLUME_STRENGTH_MIN / VOLUME_STRENGTH_MAX

    ratio = vol_array[bar_idx] / sma_val
    clamped = float(np.clip(ratio, VOLUME_STRENGTH_MIN, VOLUME_STRENGTH_MAX))
    return clamped / VOLUME_STRENGTH_MAX


def _compute_bounce(
    bar_idx: int,
    price: float,
    pivot_type: str,
    close_array: np.ndarray,
    price_range: float,
    lookahead: int,
    n: int,
) -> float:
    """Measure the post-pivot price reversal over the next few bars.

    For resistance: (price - min(next closes)) / price_range
    For support:    (max(next closes) - price) / price_range

    Returns a value clamped to [0, 1].  Caller applies BOUNCE_FLOOR
    in the composite score so edge pivots don't get zeroed out.
    """
    if bar_idx + lookahead >= n:
        return 0.0

    post_prices = close_array[bar_idx + 1 : bar_idx + lookahead + 1]

    if len(post_prices) == 0 or price_range <= 0:
        return 0.0

    if pivot_type == "resistance":
        raw = (price - float(min(post_prices))) / price_range
    else:  # support
        raw = (float(max(post_prices)) - price) / price_range

    return float(np.clip(raw, 0.0, 1.0))


# ──────────────────────────────────────────────────────────────────────
# Main pivot detection
# ──────────────────────────────────────────────────────────────────────

def detect_pivots(
    df: pd.DataFrame,
    left_span: int = 5,
    right_span: int = 5,
    min_pivots: int = MIN_PIVOTS_FALLBACK,
    ohlcv_df: pd.DataFrame | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Find resistance and support pivot points with quality scores.

    Args:
        df: DataFrame with columns [ResistancePrice, SupportPrice, Volume, IsHighVolume]
            (output of volume_classifier.classify_volume_context).
        left_span: Bars to the left that must be strictly lower/higher.
        right_span: Bars to the right that must be strictly lower/higher.
        min_pivots: If total pivots < this and span > 2, retry with span - 1.
        ohlcv_df: Full OHLCV DataFrame (with VolSMA, Close) for computing
            pivot quality signals.  If None, quality defaults to 1.0.

    Returns:
        (resistance_pivots_df, support_pivots_df) each with columns:
        [bar_index, date, price, volume, is_high_volume, type,
         prominence, volume_strength, bounce, pivot_quality]
    """
    res_price = df["ResistancePrice"].values
    sup_price = df["SupportPrice"].values
    volumes = df["Volume"].values
    high_vol = df["IsHighVolume"].values
    dates = df.index
    n = len(df)

    resistance_pivots = []
    support_pivots = []

    # --- Main loop: bars with full left and right context ---
    for i in range(left_span, n - right_span):
        # Check resistance pivot (local max of ResistancePrice)
        rp = res_price[i]
        is_res = True
        for j in range(1, left_span + 1):
            if res_price[i - j] >= rp:
                is_res = False
                break
        if is_res:
            for j in range(1, right_span + 1):
                if res_price[i + j] >= rp:
                    is_res = False
                    break
        if is_res:
            resistance_pivots.append({
                "bar_index": i,
                "date": dates[i],
                "price": rp,
                "volume": volumes[i],
                "is_high_volume": bool(high_vol[i]),
                "type": "resistance",
            })

        # Check support pivot (local min of SupportPrice)
        sp = sup_price[i]
        is_sup = True
        for j in range(1, left_span + 1):
            if sup_price[i - j] <= sp:
                is_sup = False
                break
        if is_sup:
            for j in range(1, right_span + 1):
                if sup_price[i + j] <= sp:
                    is_sup = False
                    break
        if is_sup:
            support_pivots.append({
                "bar_index": i,
                "date": dates[i],
                "price": sp,
                "volume": volumes[i],
                "is_high_volume": bool(high_vol[i]),
                "type": "support",
            })

    # --- Left boundary: bars [0, left_span) with partial left, full right ---
    for i in range(0, min(left_span, n - right_span)):
        # Resistance pivot
        rp = res_price[i]
        is_res = True
        for j in range(1, i + 1):  # partial left (vacuous if i == 0)
            if res_price[i - j] >= rp:
                is_res = False
                break
        if is_res:
            for j in range(1, right_span + 1):  # full right
                if res_price[i + j] >= rp:
                    is_res = False
                    break
        if is_res:
            resistance_pivots.append({
                "bar_index": i,
                "date": dates[i],
                "price": rp,
                "volume": volumes[i],
                "is_high_volume": bool(high_vol[i]),
                "type": "resistance",
            })

        # Support pivot
        sp = sup_price[i]
        is_sup = True
        for j in range(1, i + 1):
            if sup_price[i - j] <= sp:
                is_sup = False
                break
        if is_sup:
            for j in range(1, right_span + 1):
                if sup_price[i + j] <= sp:
                    is_sup = False
                    break
        if is_sup:
            support_pivots.append({
                "bar_index": i,
                "date": dates[i],
                "price": sp,
                "volume": volumes[i],
                "is_high_volume": bool(high_vol[i]),
                "type": "support",
            })

    # --- Right boundary: bars [n - right_span, n) with full left, partial right ---
    for i in range(max(left_span, n - right_span), n):
        # Resistance pivot
        rp = res_price[i]
        is_res = True
        for j in range(1, left_span + 1):  # full left
            if res_price[i - j] >= rp:
                is_res = False
                break
        if is_res:
            for j in range(1, n - i):  # partial right (vacuous if i == n-1)
                if res_price[i + j] >= rp:
                    is_res = False
                    break
        if is_res:
            resistance_pivots.append({
                "bar_index": i,
                "date": dates[i],
                "price": rp,
                "volume": volumes[i],
                "is_high_volume": bool(high_vol[i]),
                "type": "resistance",
            })

        # Support pivot
        sp = sup_price[i]
        is_sup = True
        for j in range(1, left_span + 1):
            if sup_price[i - j] <= sp:
                is_sup = False
                break
        if is_sup:
            for j in range(1, n - i):
                if sup_price[i + j] <= sp:
                    is_sup = False
                    break
        if is_sup:
            support_pivots.append({
                "bar_index": i,
                "date": dates[i],
                "price": sp,
                "volume": volumes[i],
                "is_high_volume": bool(high_vol[i]),
                "type": "support",
            })

    # --- Compute pivot quality scores ---
    if ohlcv_df is not None:
        close_array = ohlcv_df["Close"].values
        vol_array = ohlcv_df["Volume"].values
        vol_sma_array = ohlcv_df["VolSMA"].values
        price_range = float(close_array.max() - close_array.min())
        if price_range <= 0:
            price_range = 1.0

        for pivot_list, price_col in [
            (resistance_pivots, res_price),
            (support_pivots, sup_price),
        ]:
            for p in pivot_list:
                bi = p["bar_index"]

                prom = _compute_prominence(
                    bi, p["price"], p["type"], price_col, price_range,
                    left_span, right_span, n,
                )
                vol_str = _compute_volume_strength(bi, vol_array, vol_sma_array)
                bnc = _compute_bounce(
                    bi, p["price"], p["type"], close_array, price_range,
                    BOUNCE_LOOKAHEAD_BARS, n,
                )

                bounce_safe = max(bnc, BOUNCE_FLOOR)
                quality = (
                    prom ** PIVOT_QUALITY_PROMINENCE_EXP
                    * vol_str ** PIVOT_QUALITY_VOLUME_EXP
                    * bounce_safe ** PIVOT_QUALITY_BOUNCE_EXP
                )

                p["prominence"] = prom
                p["volume_strength"] = vol_str
                p["bounce"] = bnc
                p["pivot_quality"] = quality
    else:
        for pivot_list in (resistance_pivots, support_pivots):
            for p in pivot_list:
                p["prominence"] = 1.0
                p["volume_strength"] = 1.0
                p["bounce"] = 1.0
                p["pivot_quality"] = 1.0

    # --- Adaptive span reduction fallback ---
    total = len(resistance_pivots) + len(support_pivots)
    if total < min_pivots and left_span > MIN_SPAN_FALLBACK:
        return detect_pivots(df, left_span - 1, right_span - 1, min_pivots, ohlcv_df)

    cols = ["bar_index", "date", "price", "volume", "is_high_volume", "type",
            "prominence", "volume_strength", "bounce", "pivot_quality"]
    res_df = pd.DataFrame(resistance_pivots) if resistance_pivots else pd.DataFrame(columns=cols)
    sup_df = pd.DataFrame(support_pivots) if support_pivots else pd.DataFrame(columns=cols)
    return res_df, sup_df
