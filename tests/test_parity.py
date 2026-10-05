"""Behaviour parity: the library reproduces the original scripts' output.

Fixtures were recorded from the pre-library code (main @ 64091a9) on
synthetic series (adhoc record_golden.py): input bars + pivots + lines.
"""

import json
from pathlib import Path

import pytest

from trend_line_detector import detect

FIXTURES = sorted((Path(__file__).parent / "fixtures").glob("*.json"))


def _load(path):
    data = json.loads(path.read_text())
    return [tuple(b) for b in data["bars"]], data


@pytest.mark.parametrize("path", FIXTURES, ids=[p.stem for p in FIXTURES])
def test_matches_original(path):
    bars, golden = _load(path)
    result = detect(bars)

    want_pivots = sorted(golden["pivots"], key=lambda pv: (pv["bar_index"], pv["kind"]))
    assert [(pv.bar_index, pv.kind, pv.is_high_volume) for pv in result.pivots] == \
        [(pv["bar_index"], pv["kind"], pv["is_high_volume"]) for pv in want_pivots]
    for got, want in zip(result.pivots, want_pivots):
        assert got.price == want["price"]
        assert got.quality == pytest.approx(want["quality"], rel=1e-9, abs=1e-12)

    assert len(result.lines) == len(golden["lines"])
    for got, want in zip(result.lines, golden["lines"]):
        assert (got.kind, got.start_index, got.end_index, got.touch_count, list(got.touch_indices)) == \
            (want["kind"], want["start_index"], want["end_index"], want["touch_count"], want["touch_bars"])
        assert got.slope == pytest.approx(want["slope"], rel=1e-12)
        assert got.intercept == pytest.approx(want["intercept"], rel=1e-12)
        assert got.score == pytest.approx(want["score"], rel=1e-9)
