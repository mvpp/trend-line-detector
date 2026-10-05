"""Williams-fractal pivots on the volume-adaptive prices, with quality.

A bar is a resistance pivot when its resistance price is strictly above the
`left_span` bars before it and the `right_span` bars after it (support:
strictly below). Edge bars are checked with the partial window they have,
as long as one side is complete, so an extreme near either end of the chart
is not missed. Too few pivots overall → both spans shrink by one and the
scan reruns (down to `min_span`).

Quality (0..1) = prominence^a × volume_strength^b × max(bounce, floor)^c:
  prominence       how far the pivot stands out from its window neighbours
  volume_strength  pivot volume / its rolling SMA, clamped and normalised
  bounce           reversal over the next `bounce_lookahead` closes
"""

from typing import NamedTuple

import numpy as np

from .params import Params
from .volume import VolumeContext


class PivotSet(NamedTuple):
    """One kind's pivots, ascending by bar index."""
    bar_index: np.ndarray   # int
    price: np.ndarray
    quality: np.ndarray


def _is_pivot(prices: np.ndarray, i: int, left: int, right: int, above: bool) -> bool:
    lo, hi = max(0, i - left), min(len(prices), i + right + 1)
    neighbours = np.concatenate((prices[lo:i], prices[i + 1:hi]))
    p = prices[i]
    return bool(np.all(neighbours < p) if above else np.all(neighbours > p))


def _quality(
    i: int, price: float, above: bool, prices: np.ndarray, close: np.ndarray,
    volume: np.ndarray, vol_sma: np.ndarray, price_range: float,
    left: int, right: int, p: Params,
) -> float:
    n = len(prices)
    # Prominence — python sum keeps the original's summation order exactly.
    neighbours = prices[max(0, i - left):i].tolist() + prices[i + 1:min(n, i + right + 1)].tolist()
    prominence = 0.0
    if neighbours and price_range > 0:
        mean = sum(neighbours) / len(neighbours)
        prominence = float(np.clip(((price - mean) if above else (mean - price)) / price_range, 0.0, 1.0))

    sma = vol_sma[i]
    if np.isnan(sma) or sma <= 0:
        strength = p.volume_strength_min / p.volume_strength_max
    else:
        strength = float(np.clip(volume[i] / sma, p.volume_strength_min, p.volume_strength_max)) / p.volume_strength_max

    bounce = 0.0
    if i + p.bounce_lookahead < n and price_range > 0:
        after = close[i + 1:i + p.bounce_lookahead + 1]
        raw = (price - float(min(after))) if above else (float(max(after)) - price)
        bounce = float(np.clip(raw / price_range, 0.0, 1.0))

    return (prominence ** p.quality_prominence_exp
            * strength ** p.quality_volume_exp
            * max(bounce, p.bounce_floor) ** p.quality_bounce_exp)


def _scan(prices: np.ndarray, left: int, right: int, above: bool) -> list[int]:
    """Bars that are pivots; a bar needs at least one complete side."""
    n = len(prices)
    return [i for i in range(n) if (i >= left or i < n - right) and _is_pivot(prices, i, left, right, above)]


def detect_pivots(
    ctx: VolumeContext, close: np.ndarray, volume: np.ndarray, p: Params,
) -> tuple[PivotSet, PivotSet, np.ndarray]:
    """(resistance, support, is_high_volume) with the adaptive span fallback."""
    price_range = float(close.max() - close.min()) if len(close) else 0.0
    if price_range <= 0:
        price_range = 1.0
    left, right = p.left_span, p.right_span
    while True:
        res = _scan(ctx.resistance_price, left, right, above=True)
        sup = _scan(ctx.support_price, left, right, above=False)
        if len(res) + len(sup) >= p.min_pivots or left <= p.min_span:
            break
        left, right = left - 1, right - 1

    def build(idx: list[int], prices: np.ndarray, above: bool) -> PivotSet:
        q = [_quality(i, float(prices[i]), above, prices, close, volume, ctx.vol_sma,
                      price_range, left, right, p) for i in idx]
        return PivotSet(np.array(idx, dtype=int), prices[idx].astype(float), np.array(q, dtype=float))

    return build(res, ctx.resistance_price, True), build(sup, ctx.support_price, False), ctx.is_high_volume
