"""Line fitting: candidates → candle-through validation → score → dedup →
extend one line to the last bar.

  candidates  a line through every pair of pivots (the newest pivot is left
              out: it is a boundary artifact). A pivot touches the line when
              its residual ≤ tolerance; a line that passes through the pivot
              bar's wick (capped at one body height) has zero residual.
              Lines with ≥ min_touches touches survive. All pairs × pivots
              are evaluated as one broadcast — O(P³) work, numpy speed.
  validation  bars the line spans must not be crossed: support above a
              bar's High / resistance below its Low is a violation. < 4
              touches → zero allowed; ≥ 4 touches or near-horizontal → 5%.
  score       touch_base^Σquality × mean(volume/SMA at touches)
              × (span / n) × (1 + recency_coeff × end / n)
  dedup       longest-first; a line is dropped if it matches a kept line on
              slope+intercept, shares ≥ 50% pivots, crosses it, has ≥ 50%
              pivots within ±2 bars of it, sits inside it with the same
              slope sign, or (when time_overlap_dedup is set) overlaps its
              bar range by more than that share of the two ranges' union
  extension   the rightmost surviving line whose projection to the last bar
              passes the same validation (and, when max_projection_gap is
              set, ends within that fraction of the last close) is extended
              (first success only)
"""

from dataclasses import dataclass

import numpy as np

from .params import Params
from .pivots import PivotSet


@dataclass
class FittedLine:
    slope: float
    intercept: float
    touch_bars: list[int]
    touch_pivots: list[int]   # positions in the fitting PivotSet (for quality)
    start_bar: int
    end_bar: int
    score: float = 0.0
    extended: bool = False

    @property
    def touch_count(self) -> int:
        return len(self.touch_bars)


@dataclass(frozen=True)
class _Bars:
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    vol_sma: np.ndarray


def _violations(line: FittedLine, support: bool, bars: _Bars, start: int, end: int) -> int:
    idx = np.arange(start, end + 1)
    level = line.slope * idx + line.intercept
    return int(np.count_nonzero(level > bars.high[idx] if support else level < bars.low[idx]))


def _passes(violations: int, span: int, lenient: bool, p: Params) -> bool:
    if lenient:
        return violations / span <= p.violation_relaxed
    return violations <= p.violation_strict


def _candidates(piv: PivotSet, support: bool, bars: _Bars, tol: float, p: Params) -> list[FittedLine]:
    prices, at = piv.price, piv.bar_index
    body_bottom = np.minimum(bars.open[at], bars.close[at])
    body_top = np.maximum(bars.open[at], bars.close[at])
    body = body_top - body_bottom
    if support:
        band_lo, band_hi = np.maximum(bars.low[at], body_bottom - body), body_bottom
    else:
        band_lo, band_hi = body_top, np.minimum(bars.high[at], body_top + body)

    i, j = np.triu_indices(len(at), 1)              # row-major pair order
    slope = (prices[j] - prices[i]) / (at[j] - at[i])
    intercept = prices[i] - slope * at[i]
    expected = slope[:, None] * at[None, :] + intercept[:, None]
    on_wick = (band_lo <= expected) & (expected <= band_hi)
    residual = np.where(on_wick, 0.0, prices - expected)
    touching = np.abs(residual) <= tol

    out = []
    for row in np.flatnonzero(touching.sum(axis=1) >= p.min_touches):
        k = np.flatnonzero(touching[row])
        touch = at[k]
        out.append(FittedLine(float(slope[row]), float(intercept[row]), [int(b) for b in touch],
                              [int(x) for x in k], int(touch.min()), int(touch.max())))
    return out


def _score(line: FittedLine, quality: np.ndarray, bars: _Bars, n: int, p: Params) -> float:
    touch_weight = p.touch_weight_base ** sum(quality[k] for k in line.touch_pivots)
    ratios = [bars.volume[b] / bars.vol_sma[b] for b in line.touch_bars
              if bars.vol_sma[b] and not np.isnan(bars.vol_sma[b]) and bars.vol_sma[b] > 0]
    volume_weight = float(np.mean(ratios)) if ratios else 1.0
    span_weight = (line.end_bar - line.start_bar) / n
    recency_weight = 1.0 + p.recency_coeff * (line.end_bar / n)
    return touch_weight * volume_weight * span_weight * recency_weight


def _is_duplicate(c: FittedLine, k: FittedLine, price_range: float, p: Params) -> bool:
    # 1. slope + intercept similarity (absolute slope diff when both ~flat)
    slope_diff = abs(c.slope - k.slope)
    max_slope = max(abs(c.slope), abs(k.slope), 1e-10)
    flat = p.min_slope_factor * price_range
    slope_similar = slope_diff < flat if max_slope < flat else slope_diff / max_slope < p.slope_dedup
    if slope_similar and abs(c.intercept - k.intercept) / price_range < p.intercept_dedup:
        return True
    # 2. pivot overlap
    c_bars, k_bars = set(c.touch_bars), set(k.touch_bars)
    shortest = min(len(c_bars), len(k_bars))
    if shortest and len(c_bars & k_bars) / shortest >= p.dedup_overlap_ratio:
        return True
    # 3. the two lines cross inside their common range
    lo, hi = max(c.start_bar, k.start_bar), min(c.end_bar, k.end_bar)
    if lo < hi:
        d_lo = (c.slope * lo + c.intercept) - (k.slope * lo + k.intercept)
        d_hi = (c.slope * hi + c.intercept) - (k.slope * hi + k.intercept)
        if d_lo * d_hi < 0:
            return True
    # 4. most of c's pivots sit next to k's
    adjacent = sum(1 for cb in c_bars if any(abs(cb - kb) <= p.adjacent_pivot_bars for kb in k_bars))
    if c_bars and adjacent / len(c_bars) >= p.dedup_overlap_ratio:
        return True
    # 5. same direction and c's range inside k's (k is never shorter)
    if c.slope * k.slope >= 0 and c.start_bar >= k.start_bar and c.end_bar <= k.end_bar:
        return True
    # 6. time-range overlap beyond that share of the union (k is never shorter)
    return p.time_overlap_dedup is not None and _range_overlap(c, k) > p.time_overlap_dedup


def _range_overlap(a: FittedLine, b: FittedLine) -> float:
    """Overlap of two lines' bar ranges as a share of their union."""
    union = max(a.end_bar, b.end_bar) - min(a.start_bar, b.start_bar)
    if union <= 0:
        return 0.0
    return max(0, min(a.end_bar, b.end_bar) - max(a.start_bar, b.start_bar)) / union


def _deduplicate(lines: list[FittedLine], price_range: float, p: Params) -> list[FittedLine]:
    kept: list[FittedLine] = []
    for c in sorted(lines, key=lambda ln: ln.end_bar - ln.start_bar, reverse=True):
        if not any(_is_duplicate(c, k, price_range, p) for k in kept):
            kept.append(c)
    return kept


def _near(price: float, ref: float, gap: float) -> bool:
    return ref > 0 and abs(price / ref - 1) <= gap


def _extend_rightmost(lines: list[FittedLine], support: bool, bars: _Bars, p: Params) -> None:
    last = len(bars.close) - 1
    last_close = float(bars.close[last])
    close_range = float(bars.close.max() - bars.close.min())
    if close_range == 0:
        return
    tol = p.tolerance_pct * close_range
    for line in sorted(lines, key=lambda ln: ln.end_bar, reverse=True):
        if line.end_bar >= last:
            return
        if p.max_projection_gap is not None and not _near(line.slope * last + line.intercept,
                                                          last_close, p.max_projection_gap):
            continue
        near_flat = abs(line.slope * (line.end_bar - line.start_bar + 1)) < tol
        lenient = near_flat or line.touch_count >= p.violation_relaxed_min_touches
        v = _violations(line, support, bars, line.end_bar + 1, last)
        if _passes(v, last - line.end_bar, lenient, p):
            line.end_bar, line.extended = last, True
            return


def fit_lines(piv: PivotSet, support: bool, bars: _Bars, p: Params) -> list[FittedLine]:
    """Best lines of one kind, by score descending."""
    if len(piv.bar_index) < p.min_touches or len(piv.bar_index) <= 1:
        return []
    fit = PivotSet(piv.bar_index[:-1], piv.price[:-1], piv.quality[:-1])
    if len(fit.bar_index) < p.min_touches:
        return []
    price_range = float(fit.price.max() - fit.price.min())
    if price_range == 0:
        return []
    tol = p.tolerance_pct * price_range
    n = len(bars.close)

    valid = []
    for line in _candidates(fit, support, bars, tol, p):
        span = line.end_bar - line.start_bar + 1
        lenient = abs(line.slope * span) < tol or line.touch_count >= p.violation_relaxed_min_touches
        if _passes(_violations(line, support, bars, line.start_bar, line.end_bar), span, lenient, p):
            line.score = _score(line, fit.quality, bars, n, p)
            valid.append(line)

    top = sorted(_deduplicate(valid, price_range, p), key=lambda ln: ln.score, reverse=True)[:p.max_lines]
    _extend_rightmost(top, support, bars, p)
    return top
