#!/usr/bin/env python3
"""Volume-Adaptive Stock Trend Line Detection — proof of concept.

Usage:
    python main.py --ticker AAPL --period 1y --verbose
    python main.py --ticker TSLA --period 6mo --left-span 3 --right-span 3
    python main.py --ticker AAPL --period 1y --show-pivots --dash-lines --savefig chart.png
"""

import argparse

from config import (
    DEFAULT_LEFT_SPAN,
    DEFAULT_MAX_LINES,
    DEFAULT_MIN_TOUCHES,
    DEFAULT_RIGHT_SPAN,
    DEFAULT_TOLERANCE_PCT,
    DEFAULT_VOL_LOOKBACK,
    DEFAULT_VOL_MULTIPLIER,
)
from data_fetcher import fetch_ohlcv
from volume_classifier import classify_volume_context
from pivot_detector import detect_pivots
from trend_fitter import fit_trend_lines
from visualizer import plot_trend_lines


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Volume-adaptive stock trend line detection")
    p.add_argument("--ticker", default="AAPL", help="Stock ticker symbol")
    p.add_argument("--period", default="1y", help="yfinance period (e.g. 1y, 6mo, 2y)")
    p.add_argument("--interval", default="1d", help="yfinance interval (e.g. 1d, 1h)")
    p.add_argument("--left-span", type=int, default=DEFAULT_LEFT_SPAN, help="Williams Fractal left span")
    p.add_argument("--right-span", type=int, default=DEFAULT_RIGHT_SPAN, help="Williams Fractal right span")
    p.add_argument("--vol-lookback", type=int, default=DEFAULT_VOL_LOOKBACK, help="Rolling volume SMA window")
    p.add_argument("--vol-multiplier", type=float, default=DEFAULT_VOL_MULTIPLIER, help="High-volume threshold multiplier")
    p.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE_PCT, help="Trend line touch tolerance (fraction of price range)")
    p.add_argument("--min-touches", type=int, default=DEFAULT_MIN_TOUCHES, help="Minimum points for a valid trend line")
    p.add_argument("--max-lines", type=int, default=DEFAULT_MAX_LINES, help="Max trend lines per type to display")
    p.add_argument("--verbose", action="store_true", help="Print detailed pivot/line info")
    p.add_argument("--savefig", default=None, help="Save chart to file instead of displaying")
    p.add_argument("--show-pivots", action="store_true", help="Show pivot-point markers on chart")
    p.add_argument("--dash-lines", action="store_true", help="Draw trend lines as dashed instead of solid")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Step 1: Fetch data
    print(f"Fetching {args.ticker} ({args.period}, {args.interval})...")
    df = fetch_ohlcv(args.ticker, args.period, args.interval)
    print(f"  {len(df)} bars from {df.index[0].date()} to {df.index[-1].date()}")

    # Step 2: Volume classification
    df = classify_volume_context(df, args.vol_lookback, args.vol_multiplier)
    hv_count = df["IsHighVolume"].sum()
    print(f"  High-volume bars: {hv_count}/{len(df)} ({100 * hv_count / len(df):.1f}%)")

    # Step 3: Pivot detection
    res_pivots, sup_pivots = detect_pivots(df, args.left_span, args.right_span, ohlcv_df=df)
    print(f"  Resistance pivots: {len(res_pivots)}")
    print(f"  Support pivots:    {len(sup_pivots)}")

    if args.verbose and not res_pivots.empty:
        print("\n  Resistance pivots:")
        for _, p in res_pivots.iterrows():
            flag = " [HIGH VOL]" if p["is_high_volume"] else ""
            print(f"    {p['date'].date()}  price={p['price']:.2f}  vol={p['volume']:.0f}"
                  f"  q={p['pivot_quality']:.3f}"
                  f" (prom={p['prominence']:.3f} vs={p['volume_strength']:.3f}"
                  f" bnc={p['bounce']:.3f}){flag}")

    if args.verbose and not sup_pivots.empty:
        print("\n  Support pivots:")
        for _, p in sup_pivots.iterrows():
            flag = " [HIGH VOL]" if p["is_high_volume"] else ""
            print(f"    {p['date'].date()}  price={p['price']:.2f}  vol={p['volume']:.0f}"
                  f"  q={p['pivot_quality']:.3f}"
                  f" (prom={p['prominence']:.3f} vs={p['volume_strength']:.3f}"
                  f" bnc={p['bounce']:.3f}){flag}")

    # Step 4: Fit trend lines
    res_lines = fit_trend_lines(
        res_pivots, "resistance", df, args.tolerance, args.min_touches, args.max_lines,
    )
    sup_lines = fit_trend_lines(
        sup_pivots, "support", df, args.tolerance, args.min_touches, args.max_lines,
    )

    print(f"\n  Resistance lines: {len(res_lines)}")
    for i, ln in enumerate(res_lines):
        print(f"    #{i+1}: {ln.touch_count} touches, score={ln.score:.2f}, "
              f"slope={ln.slope:.4f}/bar, {ln.start_date.date()} to {ln.end_date.date()}")

    print(f"  Support lines:    {len(sup_lines)}")
    for i, ln in enumerate(sup_lines):
        print(f"    #{i+1}: {ln.touch_count} touches, score={ln.score:.2f}, "
              f"slope={ln.slope:.4f}/bar, {ln.start_date.date()} to {ln.end_date.date()}")

    # Step 5: Visualize
    title = f"{args.ticker} — Volume-Adaptive Trend Lines ({args.period})"
    plot_trend_lines(
        df, sup_pivots, res_pivots, sup_lines, res_lines, title, args.savefig,
        show_pivots=args.show_pivots, dash_lines=args.dash_lines,
    )
    if args.savefig:
        print(f"\n  Chart saved to {args.savefig}")


if __name__ == "__main__":
    main()
