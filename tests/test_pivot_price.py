"""pivot_price: "volume" (default, adaptive) vs "wick" (always High / Low)."""

import json
from pathlib import Path

import pytest

from trend_line_detector import Bar, Params, detect

FIXTURES = sorted((Path(__file__).parent / "fixtures").glob("*.json"))


def bars_of(path):
    return [Bar(*b) for b in json.loads(path.read_text())["bars"]]


@pytest.mark.parametrize("path", FIXTURES, ids=[p.stem for p in FIXTURES])
def test_wick_mode_reads_every_pivot_on_high_or_low(path):
    bars = bars_of(path)
    for pv in detect(bars, Params(pivot_price="wick")).pivots:
        b = bars[pv.bar_index]
        assert pv.price == (b.high if pv.kind == "resistance" else b.low)


def test_volume_mode_reads_normal_bars_on_body_edges():
    bars = bars_of(FIXTURES[0])
    normal = [pv for pv in detect(bars).pivots if not pv.is_high_volume]
    assert normal
    for pv in normal:
        b = bars[pv.bar_index]
        edge = max(b.open, b.close) if pv.kind == "resistance" else min(b.open, b.close)
        assert pv.price == edge


def test_wick_mode_changes_the_fit_without_volume():
    # No volume → "volume" mode reads bodies only; "wick" must differ.
    bars = bars_of(next(p for p in FIXTURES if "novolume" in p.stem))
    assert [p.price for p in detect(bars).pivots] != [p.price for p in detect(bars, Params(pivot_price="wick")).pivots]


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        detect(bars_of(FIXTURES[0]), Params(pivot_price="close"))
