"""time_overlap_dedup: same-kind lines overlapping > X of the shorter range collapse to the longer."""

import json
from itertools import combinations
from pathlib import Path

import pytest

from trend_line_detector import Bar, Params, detect

FIXTURES = sorted((Path(__file__).parent / "fixtures").glob("*.json"))
LIMIT = 0.70


def bars_of(path):
    return [Bar(*b) for b in json.loads(path.read_text())["bars"]]


def overlapping_pairs(lines, limit):
    """Same-kind pairs whose first→last-touch ranges overlap beyond `limit`."""
    out = []
    for a, b in combinations(lines, 2):
        if a.kind != b.kind:
            continue
        shorter = min(a.touch_end_index - a.start_index, b.touch_end_index - b.start_index)
        overlap = min(a.touch_end_index, b.touch_end_index) - max(a.start_index, b.start_index)
        if shorter > 0 and max(0, overlap) / shorter > limit:
            out.append((a, b))
    return out


@pytest.mark.parametrize("path", FIXTURES, ids=[p.stem for p in FIXTURES])
def test_no_same_kind_pair_overlaps_beyond_limit(path):
    result = detect(bars_of(path), Params(time_overlap_dedup=LIMIT))
    assert overlapping_pairs(result.lines, LIMIT) == []


def test_rule_bites_on_the_fixtures():
    # Guards the test above from passing vacuously.
    assert any(overlapping_pairs(detect(b).lines, LIMIT) for b in map(bars_of, FIXTURES))


def test_longer_line_survives():
    for bars in map(bars_of, FIXTURES):
        free = detect(bars).lines
        kept = detect(bars, Params(time_overlap_dedup=LIMIT)).lines
        for a, b in overlapping_pairs(free, LIMIT):
            longer = max((a, b), key=lambda ln: ln.touch_end_index - ln.start_index)
            shorter = b if longer is a else a
            if (longer.touch_end_index - longer.start_index) != (shorter.touch_end_index - shorter.start_index):
                assert not any(k.kind == shorter.kind and k.start_index == shorter.start_index
                               and k.touch_end_index == shorter.touch_end_index and k.slope == shorter.slope
                               for k in kept)
