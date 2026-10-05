"""max_projection_gap: a projected line must end near the last close."""

import json
from pathlib import Path

import pytest

from trend_line_detector import Bar, Params, detect

FIXTURES = sorted((Path(__file__).parent / "fixtures").glob("*.json"))
GAP = 0.05


def bars_of(path):
    return [Bar(*b) for b in json.loads(path.read_text())["bars"]]


def far_projections(bars, result, gap):
    last_close = bars[-1].close
    return [ln for ln in result.lines if ln.extended and abs(ln.end_price / last_close - 1) > gap]


@pytest.mark.parametrize("path", FIXTURES, ids=[p.stem for p in FIXTURES])
def test_projections_end_within_gap(path):
    bars = bars_of(path)
    assert far_projections(bars, detect(bars, Params(max_projection_gap=GAP)), GAP) == []


def test_gap_changes_something_on_the_fixtures():
    # Guards the test above from passing vacuously: unlimited projection
    # does produce far-off lines on at least one fixture.
    assert any(far_projections(b, detect(b), GAP) for b in map(bars_of, FIXTURES))


def test_gap_only_changes_projection_not_lines():
    for bars in map(bars_of, FIXTURES):
        free, capped = detect(bars), detect(bars, Params(max_projection_gap=GAP))
        assert [(l.kind, l.start_index, l.touch_end_index, l.score) for l in free.lines] == \
            [(l.kind, l.start_index, l.touch_end_index, l.score) for l in capped.lines]


@pytest.mark.parametrize("path", FIXTURES, ids=[p.stem for p in FIXTURES])
def test_touch_end_fields(path):
    bars = bars_of(path)
    for ln in detect(bars, Params(max_projection_gap=GAP)).lines:
        assert ln.touch_end_index == max(ln.touch_indices)
        assert ln.touch_end_time == bars[ln.touch_end_index].time
        assert ln.touch_end_price == pytest.approx(ln.price_at(ln.touch_end_index))
        assert ln.end_index > ln.touch_end_index if ln.extended else ln.end_index == ln.touch_end_index
