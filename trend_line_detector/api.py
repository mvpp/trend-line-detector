"""The library's one entry point: ``detect(bars) -> Result``.

Pure: no I/O, no clock, no globals. Bars must be in time order, one per
period, with finite OHLC; missing volume counts as 0 (so no bar is
high-volume and every pivot is read on body edges).
"""

from typing import Iterable

import numpy as np

from .fitter import _Bars, FittedLine, fit_lines
from .models import Bar, Kind, Pivot, Result, TrendLine
from .params import Params
from .pivots import PivotSet, detect_pivots
from .volume import classify_volume


def _arrays(bars: list[Bar]) -> tuple[np.ndarray, ...]:
    ohlc = np.array([(b[1], b[2], b[3], b[4]) for b in bars], dtype=float).reshape(-1, 4)
    if not np.isfinite(ohlc).all():
        raise ValueError("bars must have finite open/high/low/close")
    volume = np.array([b[5] if len(b) > 5 and b[5] is not None else 0.0 for b in bars], dtype=float)
    return (*ohlc.T, np.nan_to_num(volume, nan=0.0))


def _pivots(ps: PivotSet, kind: Kind, times: list, high_volume: np.ndarray) -> list[Pivot]:
    return [Pivot(int(i), times[i], kind, float(price), bool(high_volume[i]), float(q))
            for i, price, q in zip(ps.bar_index, ps.price, ps.quality)]


def _line(f: FittedLine, kind: Kind, times: list) -> TrendLine:
    touch_end = max(f.touch_bars)
    return TrendLine(
        kind=kind, slope=f.slope, intercept=f.intercept,
        start_index=f.start_bar, end_index=f.end_bar,
        start_time=times[f.start_bar], end_time=times[f.end_bar],
        start_price=f.slope * f.start_bar + f.intercept, end_price=f.slope * f.end_bar + f.intercept,
        touch_end_index=touch_end, touch_end_time=times[touch_end],
        touch_end_price=f.slope * touch_end + f.intercept,
        touch_count=f.touch_count, touch_indices=tuple(f.touch_bars),
        score=f.score, extended=f.extended,
    )


def detect(bars: Iterable[Bar | tuple], params: Params = Params()) -> Result:
    """Pivots and support/resistance lines for one OHLCV series.

    `bars` is any iterable of ``Bar`` or ``(time, open, high, low, close,
    volume)`` tuples (a DB row works as is). Lines are resistance then
    support, each best-first; at most ``params.max_lines`` per kind.
    """
    bars = list(bars)
    if not bars:
        return Result((), ())
    times = [b[0] for b in bars]
    open_, high, low, close, volume = _arrays(bars)
    ctx = classify_volume(open_, high, low, close, volume, params.vol_lookback, params.vol_multiplier)
    res, sup, high_volume = detect_pivots(ctx, close, volume, params)
    ohlcv = _Bars(open_, high, low, close, volume, ctx.vol_sma)

    pivots = sorted(_pivots(res, "resistance", times, high_volume) + _pivots(sup, "support", times, high_volume),
                    key=lambda pv: (pv.bar_index, pv.kind))
    lines = [_line(f, "resistance", times) for f in fit_lines(res, False, ohlcv, params)] \
        + [_line(f, "support", times) for f in fit_lines(sup, True, ohlcv, params)]
    return Result(tuple(pivots), tuple(lines))
