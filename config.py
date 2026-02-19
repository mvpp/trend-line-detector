"""Central configuration for all tunable constants.

Every numeric constant used across the stock charting pipeline lives here.
Modules import what they need instead of embedding magic numbers.

Organisation follows the pipeline order:
  CLI defaults → Pivot detection → Trend fitting → Visualisation
"""

# ──────────────────────────────────────────────────────────────────────
# CLI / main.py defaults
# ──────────────────────────────────────────────────────────────────────

# Williams Fractal window: bars to the left/right that must be strictly
# lower (support) or higher (resistance) than the candidate pivot.
# Larger spans → fewer but more significant pivots.  Typical: 3–7.
DEFAULT_LEFT_SPAN = 5
DEFAULT_RIGHT_SPAN = 5

# Rolling volume SMA window (trading days).  20 ≈ one calendar month.
DEFAULT_VOL_LOOKBACK = 20

# A candle is "high-volume" when Volume > multiplier × VolSMA.
# 1.5 is a common institutional-flow threshold.
DEFAULT_VOL_MULTIPLIER = 1.5

# Touch tolerance: fraction of the fitting-pivot price range.
# A pivot counts as "touching" a candidate line when its residual
# is within tolerance × (max_price − min_price).  0.01 = 1%.
DEFAULT_TOLERANCE_PCT = 0.01

# Minimum pivot points that must lie on a valid trend line.
DEFAULT_MIN_TOUCHES = 3

# Maximum trend lines returned per type (support / resistance).
DEFAULT_MAX_LINES = 5

# ──────────────────────────────────────────────────────────────────────
# Pivot detector  (pivot_detector.py)
# ──────────────────────────────────────────────────────────────────────

# If total detected pivots < MIN_PIVOTS_FALLBACK and the current span
# is still above MIN_SPAN_FALLBACK, the detector retries with span − 1.
MIN_PIVOTS_FALLBACK = 6
MIN_SPAN_FALLBACK = 2

# ──────────────────────────────────────────────────────────────────────
# Pivot quality scoring  (pivot_detector.py)
# ──────────────────────────────────────────────────────────────────────

# Exponents for the geometric-mean composite quality score.
# Prominence gets the most weight (structural significance), volume
# strength next (institutional participation), bounce least (confirmation).
PIVOT_QUALITY_PROMINENCE_EXP = 0.40
PIVOT_QUALITY_VOLUME_EXP = 0.35
PIVOT_QUALITY_BOUNCE_EXP = 0.25

# Bars after the pivot to measure post-pivot reversal strength.
BOUNCE_LOOKAHEAD_BARS = 3

# Volume strength is clamped to [min, max] then normalized to [0, 1]
# by dividing by max.  Min prevents division-by-zero artifacts on
# extremely low-volume bars; max caps outlier spikes.
VOLUME_STRENGTH_MIN = 0.1
VOLUME_STRENGTH_MAX = 5.0

# Floor for bounce so rightmost pivots (no look-ahead data) don't
# zero out the entire quality via 0^0.25.  0.1^0.25 ≈ 0.56 = neutral.
BOUNCE_FLOOR = 0.1

# ──────────────────────────────────────────────────────────────────────
# Trend fitter  (trend_fitter.py)
# ──────────────────────────────────────────────────────────────────────

# --- Candle-through validation ---
# Lines with < 4 touches must have ZERO violations (strict).
VIOLATION_TOLERANCE_STRICT = 0.0
# Lines with ≥ 4 touches, or near-horizontal lines, may have up to 5%
# of their spanned bars violating the candle-through rule (relaxed).
VIOLATION_TOLERANCE_RELAXED = 0.05
# Minimum touch count before the relaxed threshold applies.
VIOLATION_RELAXED_MIN_TOUCHES = 4

# --- Scoring weights ---
# Touch weight = TOUCH_WEIGHT_BASE ^ touch_count.
# Higher base → stronger preference for many-touch lines.  2.0 means
# each additional touch doubles the weight.
TOUCH_WEIGHT_BASE = 2.0
# Recency bonus: score *= 1.0 + RECENCY_WEIGHT_COEFF × (end_bar / total_bars).
# 0.5 gives up to +50% boost for a line ending at the chart's right edge.
RECENCY_WEIGHT_COEFF = 0.5

# --- Deduplication thresholds ---
# Check 1 – slope similarity: relative difference threshold.
SLOPE_DEDUP_THRESHOLD = 0.10
# Check 1 – intercept similarity: difference as fraction of price range.
INTERCEPT_DEDUP_THRESHOLD = 0.02
# Near-zero slope normalisation: slopes below MIN_SLOPE_FACTOR × price_range
# are compared by absolute difference instead of relative.
MIN_SLOPE_FACTOR = 0.001
# Check 2 & 4 – overlap / adjacency ratio that triggers dedup.
DEDUP_OVERLAP_RATIO = 0.50
# Check 4 – two pivots are "adjacent" if their bar indices differ by
# at most this many bars.
ADJACENT_PIVOT_BAR_TOLERANCE = 2

# ──────────────────────────────────────────────────────────────────────
# Visualiser  (visualizer.py)
# ──────────────────────────────────────────────────────────────────────

# Chart dimensions (inches).
FIGURE_SIZE = (16, 10)
# Trend-line drawing: width thinned from original 1.5 for cleaner look.
LINE_WIDTH = 0.3
LINE_ALPHA = 0.8
# Pivot scatter markers.
PIVOT_MARKER_SIZE = 80
# Saved image resolution.
SAVE_DPI = 150

# MACD parameters (standard 12-26-9).
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Panel ratios: main candlestick : volume : MACD.
# (4, 1, 2) gives the price chart most space, MACD twice the volume bar.
PANEL_RATIOS = (4, 1, 2)
