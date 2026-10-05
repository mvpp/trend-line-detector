"""Input and output value types. All frozen; outputs are JSON-ready via
``dataclasses.asdict``.

  Bar        one OHLCV candle; ``time`` is an opaque sortable label
             (date string, epoch, index) that is only echoed back
  Pivot      a fractal high (resistance) or low (support)
  TrendLine  a validated support/resistance line, with endpoint prices so a
             chart draws it as a two-point segment
  Result     everything ``detect`` found for one series
"""

from dataclasses import dataclass
from typing import Hashable, Literal, NamedTuple

Kind = Literal["support", "resistance"]


class Bar(NamedTuple):
    time: Hashable
    open: float
    high: float
    low: float
    close: float
    volume: float | None = None   # None → 0 (no bar is high-volume)


@dataclass(frozen=True)
class Pivot:
    bar_index: int
    time: Hashable
    kind: Kind
    price: float              # the volume-adaptive price the pivot was found on
    is_high_volume: bool
    quality: float            # 0..1, weights the pivot's touch in line scores


@dataclass(frozen=True)
class TrendLine:
    kind: Kind
    slope: float              # price per bar
    intercept: float          # price at bar 0
    start_index: int          # first touch
    end_index: int            # last touch, or the last bar when extended
    start_time: Hashable
    end_time: Hashable
    start_price: float
    end_price: float
    touch_end_index: int      # last touch (== end_index unless extended)
    touch_end_time: Hashable
    touch_end_price: float
    touch_count: int
    touch_indices: tuple[int, ...]   # bar indices of the touching pivots
    score: float
    extended: bool            # True → projected past its last touch to the last bar

    def price_at(self, bar_index: int) -> float:
        return self.slope * bar_index + self.intercept


@dataclass(frozen=True)
class Result:
    pivots: tuple[Pivot, ...]       # both kinds, by bar_index
    lines: tuple[TrendLine, ...]    # resistance then support, each by score desc
