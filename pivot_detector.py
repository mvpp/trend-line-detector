"""Williams Fractal pivot detection on volume-adaptive prices.

Detects local maxima of ResistancePrice (resistance pivots) and local minima
of SupportPrice (support pivots) using a left/right span window.  If too few
pivots are found, the span is reduced adaptively down to a minimum of 2.

Boundary bars (first left_span and last right_span) are checked with partial
context so that edge extremes (e.g. an all-time high near the chart end) are
not missed.
"""

from typing import Tuple

import pandas as pd

from config import MIN_PIVOTS_FALLBACK, MIN_SPAN_FALLBACK


def detect_pivots(
    df: pd.DataFrame,
    left_span: int = 5,
    right_span: int = 5,
    min_pivots: int = MIN_PIVOTS_FALLBACK,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Find resistance and support pivot points.

    Args:
        df: DataFrame with columns [ResistancePrice, SupportPrice, Volume, IsHighVolume]
            (output of volume_classifier.classify_volume_context).
        left_span: Bars to the left that must be strictly lower/higher.
        right_span: Bars to the right that must be strictly lower/higher.
        min_pivots: If total pivots < this and span > 2, retry with span - 1.

    Returns:
        (resistance_pivots_df, support_pivots_df) each with columns:
        [bar_index, date, price, volume, is_high_volume, type]
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

    # --- Adaptive span reduction fallback ---
    total = len(resistance_pivots) + len(support_pivots)
    if total < min_pivots and left_span > MIN_SPAN_FALLBACK:
        return detect_pivots(df, left_span - 1, right_span - 1, min_pivots)

    res_df = pd.DataFrame(resistance_pivots) if resistance_pivots else pd.DataFrame(
        columns=["bar_index", "date", "price", "volume", "is_high_volume", "type"]
    )
    sup_df = pd.DataFrame(support_pivots) if support_pivots else pd.DataFrame(
        columns=["bar_index", "date", "price", "volume", "is_high_volume", "type"]
    )
    return res_df, sup_df
