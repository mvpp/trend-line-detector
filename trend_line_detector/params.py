"""Tunable parameters of the detector, as one frozen value object.

Defaults are the values the algorithm was tuned with. Callers override any
field with ``Params(left_span=3, ...)`` or ``dataclasses.replace``; nothing
is read from globals, files or the environment.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Params:
    # ── Volume context ────────────────────────────────────────────────
    # Rolling volume SMA window, in bars.
    vol_lookback: int = 20
    # A bar is high-volume when volume > multiplier × SMA. High-volume bars
    # use High/Low (wicks); normal bars use the body edges.
    vol_multiplier: float = 1.5

    # ── Pivot detection (Williams fractal) ────────────────────────────
    # Bars left/right that must be strictly lower (support) or higher
    # (resistance) than the pivot. Larger spans → fewer, stronger pivots.
    left_span: int = 5
    right_span: int = 5
    # Fewer pivots than this (both kinds together) → retry with span − 1,
    # down to min_span.
    min_pivots: int = 6
    min_span: int = 2

    # ── Pivot quality (geometric mean of three signals) ───────────────
    quality_prominence_exp: float = 0.40
    quality_volume_exp: float = 0.35
    quality_bounce_exp: float = 0.25
    # Bars after the pivot used to measure the reversal ("bounce").
    bounce_lookahead: int = 3
    # Volume / SMA is clamped to [min, max], then divided by max.
    volume_strength_min: float = 0.1
    volume_strength_max: float = 5.0
    # Floor for bounce so the newest pivots (no look-ahead) aren't zeroed.
    bounce_floor: float = 0.1

    # ── Line fitting ──────────────────────────────────────────────────
    # A pivot touches a line when its residual ≤ tolerance × pivot price range.
    tolerance_pct: float = 0.01
    min_touches: int = 3
    # Lines returned per kind (support / resistance).
    max_lines: int = 5

    # Candle-through validation: < relaxed_min_touches touches → zero
    # violations; ≥ that, or near-horizontal, → up to `relaxed` share of bars.
    violation_strict: float = 0.0
    violation_relaxed: float = 0.05
    violation_relaxed_min_touches: int = 4

    # Projection: a line extended to the last bar must end within this
    # fraction of the last close (|end / close − 1| ≤ gap), otherwise the
    # next-rightmost line is tried. None = no distance limit.
    max_projection_gap: float | None = None

    # Score = touch_weight_base ^ Σ(touch quality) × volume × span × recency,
    # recency = 1 + coeff × end_index / n.
    touch_weight_base: float = 2.0
    recency_coeff: float = 0.5

    # Deduplication (see fitter._deduplicate).
    slope_dedup: float = 0.10
    intercept_dedup: float = 0.02
    min_slope_factor: float = 0.001
    dedup_overlap_ratio: float = 0.50
    adjacent_pivot_bars: int = 2
