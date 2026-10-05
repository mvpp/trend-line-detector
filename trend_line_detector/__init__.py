"""trend_line_detector — volume-adaptive support/resistance trend lines.

Pure library: give it OHLCV bars, get pivots and trend lines back.

    from trend_line_detector import Bar, Params, detect
    result = detect(rows)                       # rows: (time, o, h, l, c, v)
    for line in result.lines:
        print(line.kind, line.start_time, line.start_price, line.end_time, line.end_price)

Modules: params (tunables), models (Bar/Pivot/TrendLine/Result),
volume (volume context), pivots (fractal pivots + quality),
fitter (lines), api (detect).
"""

from .api import detect
from .models import Bar, Pivot, Result, TrendLine
from .params import Params

__all__ = ["Bar", "Params", "Pivot", "Result", "TrendLine", "detect"]
__version__ = "0.1.0"
