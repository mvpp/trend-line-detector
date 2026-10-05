"""detect(): input handling, edge cases, output contract."""

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from trend_line_detector import Bar, Params, detect

SAMPLE = Path(__file__).parent / "fixtures" / "synthetic_5_504.json"


def sample_bars():
    return [Bar(*b) for b in json.loads(SAMPLE.read_text())["bars"]]


def test_empty_series():
    assert detect([]) == detect([], Params())
    assert detect([]).pivots == () and detect([]).lines == ()


@pytest.mark.parametrize("n", [1, 2, 5, 11])
def test_short_series_never_raises(n):
    result = detect(sample_bars()[:n])
    assert all(0 <= pv.bar_index < n for pv in result.pivots)


def test_flat_prices_have_no_lines():
    bars = [(i, 10.0, 10.0, 10.0, 10.0, 100.0) for i in range(100)]
    assert detect(bars).lines == ()


def test_missing_volume_reads_body_edges_only():
    bars = [(t, o, h, l, c, None) for t, o, h, l, c, _ in sample_bars()]
    result = detect(bars)
    assert result.pivots and not any(pv.is_high_volume for pv in result.pivots)


def test_five_tuple_bars_mean_no_volume():
    five = [b[:5] for b in sample_bars()]
    none = [(*b[:5], None) for b in sample_bars()]
    assert detect(five) == detect(none)


def test_non_finite_ohlc_is_rejected():
    bars = sample_bars()
    bars[10] = bars[10]._replace(high=float("nan"))
    with pytest.raises(ValueError):
        detect(bars)


def test_params_override():
    assert sum(1 for ln in detect(sample_bars(), Params(max_lines=1)).lines if ln.kind == "support") <= 1


def test_line_contract():
    bars = sample_bars()
    result = detect(bars)
    assert result.lines
    kinds = [ln.kind for ln in result.lines]
    assert kinds == sorted(kinds, key=lambda k: k != "resistance")      # resistance first
    for ln in result.lines:
        assert ln.start_time == bars[ln.start_index].time and ln.end_time == bars[ln.end_index].time
        assert ln.start_price == pytest.approx(ln.price_at(ln.start_index))
        assert ln.end_price == pytest.approx(ln.price_at(ln.end_index))
        assert ln.touch_count == len(ln.touch_indices) >= Params().min_touches
        assert ln.end_index == len(bars) - 1 if ln.extended else ln.end_index == max(ln.touch_indices)
    assert sum(ln.extended for ln in result.lines if ln.kind == "support") <= 1


def test_output_is_json_ready():
    json.dumps(asdict(detect(sample_bars())))
