"""time_overlap_dedup: same-kind lines overlapping > X of their ranges' union collapse to the longer."""

import json
from itertools import combinations
from pathlib import Path

import pytest

from trend_line_detector import Bar, Params, detect
from trend_line_detector.fitter import FittedLine, _is_duplicate, _range_overlap

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
        union = max(a.touch_end_index, b.touch_end_index) - min(a.start_index, b.start_index)
        overlap = min(a.touch_end_index, b.touch_end_index) - max(a.start_index, b.start_index)
        if union > 0 and max(0, overlap) / union > limit:
            out.append((a, b))
    return out


@pytest.mark.parametrize("path", FIXTURES, ids=[p.stem for p in FIXTURES])
def test_no_same_kind_pair_overlaps_beyond_limit(path):
    result = detect(bars_of(path), Params(time_overlap_dedup=LIMIT))
    assert overlapping_pairs(result.lines, LIMIT) == []


def line(start, end, slope=0.0, intercept=100.0, touches=(0,)):
    return FittedLine(slope, intercept, list(touches), list(range(len(touches))), start, end)


@pytest.mark.parametrize("a, b, share", [
    ((0, 100), (20, 100), 0.80),    # inside, union = longer range
    ((0, 100), (50, 150), 1 / 3),   # half-shifted: overlap 50 of union 150
    ((0, 100), (100, 200), 0.0),    # touching ends
    ((0, 100), (150, 200), 0.0),    # disjoint
])
def test_range_overlap_is_share_of_union(a, b, share):
    assert _range_overlap(line(*a), line(*b)) == pytest.approx(share)


def test_overlap_rule_flags_beyond_limit_only():
    # Opposite slopes and disjoint pivots, so only rule 6 can fire.
    p = Params(time_overlap_dedup=LIMIT)
    longer = line(0, 100, slope=0.5, touches=(0, 50, 100))
    assert _is_duplicate(line(20, 100, slope=-0.5, intercept=200, touches=(30, 70, 95)), longer, 100.0, p)   # 0.80
    assert not _is_duplicate(line(35, 100, slope=-0.5, intercept=200, touches=(40, 70, 95)), longer, 100.0, p)  # 0.65
    assert not _is_duplicate(line(20, 100, slope=-0.5, intercept=200, touches=(30, 70, 95)), longer, 100.0, Params())


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
