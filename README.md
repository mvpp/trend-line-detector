# trend-line-detector

Volume-adaptive support and resistance trend lines from OHLCV bars, as a pure Python library.

Give it bars and it returns pivots and trend lines. It does no data fetching, plotting, file I/O or global state, and its only dependency is numpy. Callers own the data and the drawing.

## Install

```bash
pip install -e .              # library (numpy only)
pip install -e ".[dev]"       # + pytest
pip install -e ".[examples]"  # + pandas, yfinance, mplfinance for examples/plot_cli.py
```

## Usage

```python
from trend_line_detector import Bar, Params, detect

bars = [Bar("2026-01-02", 10.0, 10.6, 9.8, 10.4, 1_200_000), ...]   # time order, one per period
result = detect(bars)                        # or detect(rows, Params(left_span=3))

for pv in result.pivots:                     # both kinds, by bar_index
    print(pv.kind, pv.time, pv.price, pv.quality)

for ln in result.lines:                      # resistance then support, each best-first
    print(ln.kind, ln.start_time, ln.start_price, ln.end_time, ln.end_price,
          ln.touch_count, ln.score, ln.extended)
```

- **Input:** any iterable of `Bar` or `(time, open, high, low, close[, volume])` tuples, so a DB row works as is.
  - `time` is an opaque label that is echoed back; it can be a date string, an epoch or an index.
  - OHLC must be finite, otherwise `detect` raises `ValueError`.
  - Missing volume counts as 0: no bar is then high-volume, and pivots are read on body edges.
- **Output:** frozen dataclasses, JSON-ready via `dataclasses.asdict`.
  - `TrendLine` carries both endpoint prices, so a chart draws it as a two-point segment.
  - `TrendLine.price_at(i)` projects the line to any bar.
  - `touch_end_*` is the line's last touch. It equals `end_*` unless `extended` is true, in which case `end_*` is the projection to the last bar.
- **Window:** the result depends on the bars you pass in. Tolerances are a fraction of the window's price range, and scores reward span and recency relative to the window. Pass the window you intend to chart (for example the last 252 daily bars).

## Algorithm

1. **Volume context.** A bar is high-volume when its volume is above `vol_multiplier` × the 20-bar volume SMA.
   - High-volume bars offer their wicks: High for resistance, Low for support.
   - Normal bars offer their body edges: max/min of Open and Close.
2. **Pivots (Williams fractal).**
   - A bar is a pivot when its price is strictly beyond the `left_span` bars before it and the `right_span` bars after it.
   - Edge bars count if one side of the window is complete.
   - If there are fewer than `min_pivots` pivots in total, both spans shrink by one (down to `min_span`).
   - Each pivot gets a quality from 0 to 1: prominence^0.40 × volume strength^0.35 × max(bounce, 0.1)^0.25.
3. **Candidate lines.**
   - A line is drawn through every pair of pivots, leaving out the newest pivot.
   - A pivot touches the line within `tolerance_pct` of the pivots' price range. Passing through the pivot bar's wick (capped at one body height) counts as an exact touch.
   - Lines need at least `min_touches` touches.
4. **Candle-through validation.** A support line above a bar's High, or a resistance line below a bar's Low, is a violation.
   - Lines with fewer than 4 touches are allowed no violations.
   - Lines with 4 or more touches, or near-horizontal lines, may violate up to 5% of the bars they span.
5. **Score.** 2^Σ(touch quality) × mean(volume/SMA at the touches) × (span / bars) × (1 + 0.5 × end / bars).
6. **Deduplication.** Lines are compared longest first. A line is dropped if, against a kept line, it:
   - has a similar slope and intercept,
   - shares ≥ 50% of its pivots,
   - crosses it,
   - has ≥ 50% of its pivots within ±2 bars of it,
   - sits inside it with the same slope sign, or
   - with `Params(time_overlap_dedup=0.7)`, overlaps its bar range by more than 70% of the shorter range (the longer line is kept; off by default).
7. **Extension.** The rightmost surviving line whose projection to the last bar passes the same validation is extended (`extended=True`). At most one line per kind is extended.
   - With `Params(max_projection_gap=0.05)`, the projection must also end within ±5% of the last close; otherwise the next-rightmost line is tried.
   - The default (`None`) has no distance limit.

Each kind is then cut to `max_lines`, best first. Line fitting costs O(P³) in the number of pivots P. It runs as one numpy broadcast, about 5–20 ms for a 250-bar window.

All tunables live on `Params` with documented defaults. Override any of them with `Params(field=value)`.

## Tests

```bash
pytest
```

- `tests/test_parity.py`: the library reproduces the original scripts' pivots, lines and scores exactly, on recorded synthetic fixtures.
- `tests/test_api.py`: input handling, edge cases (empty, short, flat, no volume, non-finite) and the output contract.
- `tests/test_projection_gap.py`: projections respect `max_projection_gap`, and the gap changes only the projection, never which lines are found.
- `tests/test_time_overlap_dedup.py`: with `time_overlap_dedup`, no two same-kind lines overlap beyond the limit, and the longer one survives.

## Example CLI

```bash
MPLBACKEND=Agg python examples/plot_cli.py --ticker AAPL --period 1y --show-pivots --savefig chart.png
```

Every `Params` field is also a flag, for example `--left-span 3` or `--tolerance-pct 0.02`.
