"""Trend line fitting, scoring, validation, and deduplication.

Given a set of pivot points (support or resistance), fits candidate lines
through every pair, validates them against the OHLCV price action, scores
them by touch count / volume / span / recency, and deduplicates near-parallel
or overlapping lines.

Key design decisions:
  - First and last detected pivots are excluded from fitting (boundary artifacts).
  - Candle-through validation is tiered: zero violations for < 4 touches,
    5% allowed for >= 4 touches or near-horizontal lines.
  - Deduplication uses four criteria (slope/intercept similarity, pivot overlap,
    crossing lines, adjacent pivots) and keeps the longer-spanning line.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd

from config import (
    ADJACENT_PIVOT_BAR_TOLERANCE,
    DEDUP_OVERLAP_RATIO,
    INTERCEPT_DEDUP_THRESHOLD,
    MIN_SLOPE_FACTOR,
    RECENCY_WEIGHT_COEFF,
    SLOPE_DEDUP_THRESHOLD,
    TOUCH_WEIGHT_BASE,
    VIOLATION_RELAXED_MIN_TOUCHES,
    VIOLATION_TOLERANCE_RELAXED,
    VIOLATION_TOLERANCE_STRICT,
)


@dataclass
class TrendLine:
    slope: float
    intercept: float
    pivot_indices: List[int]
    touch_count: int
    line_type: str  # "support" or "resistance"
    score: float
    start_bar: int
    end_bar: int
    start_date: datetime
    end_date: datetime
    ssr: float
    avg_volume_at_touches: float


def fit_trend_lines(
    pivots: pd.DataFrame,
    line_type: str,
    ohlcv_df: pd.DataFrame,
    tolerance_pct: float = 0.01,
    min_touches: int = 3,
    max_lines: int = 5,
) -> List[TrendLine]:
    """Fit, validate, score, and return the best trend lines.

    Args:
        pivots: DataFrame with columns [bar_index, price, volume, ...].
        line_type: "support" or "resistance".
        ohlcv_df: Full OHLCV DataFrame (with VolSMA column) for validation/scoring.
        tolerance_pct: Fraction of pivot price range used as touch tolerance.
        min_touches: Minimum pivot points a valid line must touch.
        max_lines: Maximum lines to return.

    Returns:
        List of TrendLine objects sorted by score descending.
    """
    if len(pivots) < min_touches:
        return []

    # --- Fix 2: Exclude first and last pivot from fitting ---
    sorted_pivots = pivots.sort_values("bar_index").reset_index(drop=True)
    if len(sorted_pivots) > 2:
        fitting_pivots = sorted_pivots.iloc[1:-1].reset_index(drop=True)
    else:
        return []  # removing first+last leaves <= 0 pivots

    if len(fitting_pivots) < min_touches:
        return []

    prices = fitting_pivots["price"].values
    bar_indices = fitting_pivots["bar_index"].values.astype(int)
    volumes = fitting_pivots["volume"].values

    price_range = prices.max() - prices.min()
    if price_range == 0:
        return []
    tolerance = tolerance_pct * price_range

    total_bars = len(ohlcv_df)
    P = len(fitting_pivots)

    # --- Candidate generation O(P^2) pairs, O(P) check each = O(P^3) ---
    candidates: List[TrendLine] = []

    for i in range(P):
        for j in range(i + 1, P):
            dx = bar_indices[j] - bar_indices[i]
            if dx == 0:
                continue

            slope = (prices[j] - prices[i]) / dx
            intercept = prices[i] - slope * bar_indices[i]

            touching = []
            residuals = []
            for k in range(P):
                expected = slope * bar_indices[k] + intercept
                residual = prices[k] - expected
                if abs(residual) <= tolerance:
                    touching.append(k)
                    residuals.append(residual)

            if len(touching) < min_touches:
                continue

            ssr = sum(r * r for r in residuals)
            touch_bars = bar_indices[touching]
            touch_vols = volumes[touching]

            candidates.append(TrendLine(
                slope=slope,
                intercept=intercept,
                pivot_indices=touching,
                touch_count=len(touching),
                line_type=line_type,
                score=0.0,
                start_bar=int(touch_bars.min()),
                end_bar=int(touch_bars.max()),
                start_date=ohlcv_df.index[int(touch_bars.min())],
                end_date=ohlcv_df.index[int(touch_bars.max())],
                ssr=ssr,
                avg_volume_at_touches=float(touch_vols.mean()),
            ))

    # --- Fix 3: Tiered candle-through validation (BEFORE dedup) ---
    valid = []
    highs = ohlcv_df["High"].values
    lows = ohlcv_df["Low"].values

    for c in candidates:
        span = c.end_bar - c.start_bar + 1
        if span == 0:
            continue

        # Near-horizontal: total price change over span < tolerance
        is_near_horizontal = abs(c.slope * span) < tolerance

        violations = 0
        for bar_idx in range(c.start_bar, c.end_bar + 1):
            line_price = c.slope * bar_idx + c.intercept
            if line_type == "support" and line_price > highs[bar_idx]:
                violations += 1
            elif line_type == "resistance" and line_price < lows[bar_idx]:
                violations += 1

        if is_near_horizontal:
            # Near-horizontal lines: allow up to VIOLATION_TOLERANCE_RELAXED violations
            if violations / span <= VIOLATION_TOLERANCE_RELAXED:
                valid.append(c)
        elif c.touch_count >= VIOLATION_RELAXED_MIN_TOUCHES:
            # 4+ touches: allow up to VIOLATION_TOLERANCE_RELAXED violations
            if violations / span <= VIOLATION_TOLERANCE_RELAXED:
                valid.append(c)
        else:
            # < 4 touches: ZERO violations allowed
            if violations <= VIOLATION_TOLERANCE_STRICT:
                valid.append(c)

    # --- Scoring ---
    vol_sma = ohlcv_df["VolSMA"].values
    vol_vals = ohlcv_df["Volume"].values

    for c in valid:
        touch_bars = bar_indices[c.pivot_indices]

        # Touch weight: exponential
        touch_weight = TOUCH_WEIGHT_BASE ** c.touch_count

        # Volume weight: avg ratio of volume to SMA at each touch
        ratios = []
        for bi in touch_bars:
            sma_val = vol_sma[bi]
            if sma_val and not np.isnan(sma_val) and sma_val > 0:
                ratios.append(vol_vals[bi] / sma_val)
        volume_weight = float(np.mean(ratios)) if ratios else 1.0

        # Span weight
        span_weight = (c.end_bar - c.start_bar) / total_bars

        # Recency weight
        recency_weight = 1.0 + RECENCY_WEIGHT_COEFF * (c.end_bar / total_bars)

        c.score = touch_weight * volume_weight * span_weight * recency_weight

    # --- Fix 4: Enhanced deduplication ---
    deduped = _deduplicate(valid, price_range, bar_indices)

    deduped.sort(key=lambda ln: ln.score, reverse=True)
    top_lines = deduped[:max_lines]

    # Extend rightmost line to chart edge (with fallback to next-rightmost)
    _extend_rightmost_with_fallback(top_lines, line_type, ohlcv_df, tolerance_pct)

    return top_lines


def _deduplicate(
    candidates: List[TrendLine],
    price_range: float,
    bar_indices: np.ndarray,
    slope_threshold: float = SLOPE_DEDUP_THRESHOLD,
    intercept_threshold: float = INTERCEPT_DEDUP_THRESHOLD,
) -> List[TrendLine]:
    """Remove duplicate lines using four criteria, keeping the longer line.

    Criteria (any one triggers removal):
      1. Slope + intercept similarity
      2. Pivot overlap >= 50%
      3. Lines cross within their overlapping range
      4. Adjacent pivots >= 50% (pivots within ±2 bars)
      5. Enclosure: same slope sign and shorter line's bar range inside longer's
    """
    # Sort by span length descending — longest lines survive
    candidates.sort(key=lambda ln: (ln.end_bar - ln.start_bar), reverse=True)
    kept: List[TrendLine] = []

    for c in candidates:
        is_dup = False
        c_bars = set(int(bar_indices[pi]) for pi in c.pivot_indices)

        for k in kept:
            k_bars = set(int(bar_indices[pi]) for pi in k.pivot_indices)

            # --- Check 1: Slope + intercept similarity ---
            slope_diff = abs(c.slope - k.slope)
            max_slope = max(abs(c.slope), abs(k.slope), 1e-10)
            min_slope_abs = MIN_SLOPE_FACTOR * price_range
            if max_slope < min_slope_abs:
                slope_similar = slope_diff < min_slope_abs
            else:
                slope_similar = (slope_diff / max_slope) < slope_threshold
            intercept_similar = (
                abs(c.intercept - k.intercept) / price_range < intercept_threshold
            )
            if slope_similar and intercept_similar:
                is_dup = True
                break

            # --- Check 2: Pivot overlap >= 50% ---
            overlap = len(c_bars & k_bars)
            min_len = min(len(c_bars), len(k_bars))
            if min_len > 0 and overlap / min_len >= DEDUP_OVERLAP_RATIO:
                is_dup = True
                break

            # --- Check 3: Crossing check ---
            overlap_start = max(c.start_bar, k.start_bar)
            overlap_end = min(c.end_bar, k.end_bar)
            if overlap_start < overlap_end:
                c_price_start = c.slope * overlap_start + c.intercept
                k_price_start = k.slope * overlap_start + k.intercept
                c_price_end = c.slope * overlap_end + c.intercept
                k_price_end = k.slope * overlap_end + k.intercept
                diff_start = c_price_start - k_price_start
                diff_end = c_price_end - k_price_end
                if diff_start * diff_end < 0:  # sign change = crossing
                    is_dup = True
                    break

            # --- Check 4: Adjacent pivots >= 50% ---
            adjacent_count = 0
            for cb in c_bars:
                for kb in k_bars:
                    if abs(cb - kb) <= ADJACENT_PIVOT_BAR_TOLERANCE:
                        adjacent_count += 1
                        break  # count each c pivot at most once
            if len(c_bars) > 0 and adjacent_count / len(c_bars) >= DEDUP_OVERLAP_RATIO:
                is_dup = True
                break

            # --- Check 5: Enclosure — same direction, c inside k ---
            # Since candidates are sorted longest-first, k is always >= c in span.
            # If c's bar range fits entirely within k's and they share slope sign,
            # c is redundant.
            same_direction = c.slope * k.slope >= 0
            enclosed = c.start_bar >= k.start_bar and c.end_bar <= k.end_bar
            if same_direction and enclosed:
                is_dup = True
                break

        if not is_dup:
            kept.append(c)

    return kept


# ──────────────────────────────────────────────────────────────────────
# Rightmost-line extension (with fallback)
# ──────────────────────────────────────────────────────────────────────

def _extend_rightmost_with_fallback(
    lines: List[TrendLine],
    line_type: str,
    ohlcv_df: pd.DataFrame,
    tolerance_pct: float,
) -> None:
    """Try extending lines to chart edge, starting from the rightmost.

    Iterates lines in descending end_bar order.  The first line whose
    extension passes candle-through validation gets extended in-place.
    If only one line exists and it fails, it is kept as-is.
    """
    last_bar = len(ohlcv_df) - 1
    if not lines:
        return

    candidates = sorted(lines, key=lambda ln: ln.end_bar, reverse=True)

    for candidate in candidates:
        if candidate.end_bar >= last_bar:
            return  # already at chart edge

        ext_start = candidate.end_bar + 1
        ext_end = last_bar

        if _validate_extended_segment(
            candidate, line_type, ohlcv_df, ext_start, ext_end, tolerance_pct,
        ):
            candidate.end_bar = ext_end
            candidate.end_date = ohlcv_df.index[ext_end]
            return  # first success — done
    # No candidate passed → all kept at original length


def _validate_extended_segment(
    line: TrendLine,
    line_type: str,
    ohlcv_df: pd.DataFrame,
    ext_start: int,
    ext_end: int,
    tolerance_pct: float,
) -> bool:
    """Apply the same tiered candle-through rules to an extension segment.

    Returns True if the extension is clean enough to draw.
    """
    highs = ohlcv_df["High"].values
    lows = ohlcv_df["Low"].values

    # Use original span for tolerance & near-horizontal check
    original_span = line.end_bar - line.start_bar + 1
    prices = ohlcv_df["Close"].values
    price_range = prices.max() - prices.min()
    if price_range == 0:
        return False
    tolerance = tolerance_pct * price_range
    is_near_horizontal = abs(line.slope * original_span) < tolerance

    ext_span = ext_end - ext_start + 1
    if ext_span <= 0:
        return True

    violations = 0
    for bar_idx in range(ext_start, ext_end + 1):
        line_price = line.slope * bar_idx + line.intercept
        if line_type == "support" and line_price > highs[bar_idx]:
            violations += 1
        elif line_type == "resistance" and line_price < lows[bar_idx]:
            violations += 1

    if is_near_horizontal or line.touch_count >= VIOLATION_RELAXED_MIN_TOUCHES:
        return violations / ext_span <= VIOLATION_TOLERANCE_RELAXED
    else:
        return violations <= VIOLATION_TOLERANCE_STRICT
