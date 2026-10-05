#!/usr/bin/env python3
"""Fetch a ticker, detect trend lines with the library, save a PNG chart.

The original command-line proof of concept, rebuilt on the library. Needs
the optional extras:  pip install -e ".[examples]"

    MPLBACKEND=Agg python examples/plot_cli.py --ticker AAPL --period 1y --savefig chart.png
    MPLBACKEND=Agg python examples/plot_cli.py --ticker TSLA --show-pivots --dash-lines --savefig chart.png
"""

import argparse
from dataclasses import fields

import mplfinance as mpf
import numpy as np
import pandas as pd
import yfinance as yf

from trend_line_detector import Params, detect

# MACD (12-26-9) drawn under the candles, as in the original charts.
MACD = (12, 26, 9)


def fetch_ohlcv(ticker: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df = df.droplevel("Ticker", axis=1)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna().sort_index()
    if df.empty:
        raise SystemExit(f"No data for {ticker} ({period}, {interval})")
    return df


def macd_panels(close: pd.Series) -> list:
    fast, slow, signal = MACD
    line = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    sig = line.ewm(span=signal, adjust=False).mean()
    hist = line - sig
    return [
        mpf.make_addplot(line, panel=2, color="blue", width=0.8, ylabel="MACD"),
        mpf.make_addplot(sig, panel=2, color="orange", width=0.8),
        mpf.make_addplot(hist.where(hist >= 0), panel=2, type="bar", color="green", width=0.7),
        mpf.make_addplot(hist.where(hist < 0), panel=2, type="bar", color="red", width=0.7),
    ]


def pivot_markers(df: pd.DataFrame, pivots, kind: str, marker: str, color: str):
    series = pd.Series(np.nan, index=df.index)
    for pv in pivots:
        if pv.kind == kind:
            series.iloc[pv.bar_index] = pv.price
    return mpf.make_addplot(series, type="scatter", marker=marker, markersize=80, color=color) \
        if series.notna().any() else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ticker", default="AAPL")
    ap.add_argument("--period", default="1y")
    ap.add_argument("--interval", default="1d")
    ap.add_argument("--savefig", default=None)
    ap.add_argument("--show-pivots", action="store_true")
    ap.add_argument("--dash-lines", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    # Every Params field is a flag: --left-span 3, --tolerance-pct 0.02, ...
    for f in fields(Params):
        ap.add_argument(f"--{f.name.replace('_', '-')}", type=type(f.default), default=f.default)
    args = ap.parse_args()
    params = Params(**{f.name: getattr(args, f.name) for f in fields(Params)})

    df = fetch_ohlcv(args.ticker, args.period, args.interval)
    rows = zip(df.index, df["Open"], df["High"], df["Low"], df["Close"], df["Volume"])
    result = detect(rows, params)
    print(f"{args.ticker}: {len(df)} bars, {len(result.pivots)} pivots, {len(result.lines)} lines")
    if args.verbose:
        for ln in result.lines:
            print(f"  {ln.kind:10} touches={ln.touch_count} score={ln.score:.2f} "
                  f"{ln.start_time.date()} → {ln.end_time.date()}{' (extended)' if ln.extended else ''}")

    addplots = macd_panels(df["Close"])
    if args.show_pivots:
        addplots += [m for m in (pivot_markers(df, result.pivots, "support", "^", "green"),
                                 pivot_markers(df, result.pivots, "resistance", "v", "red")) if m]
    kwargs = dict(type="candle", style="charles", volume=True, figsize=(16, 10), tight_layout=True,
                  panel_ratios=(4, 1, 2), addplot=addplots,
                  title=f"{args.ticker} — volume-adaptive trend lines ({args.period})")
    if result.lines:
        kwargs["alines"] = dict(
            alines=[[(ln.start_time, ln.start_price), (ln.end_time, ln.end_price)] for ln in result.lines],
            colors=["green" if ln.kind == "support" else "red" for ln in result.lines],
            linewidths=0.3, alpha=0.8, linestyle="--" if args.dash_lines else "-")
    if args.savefig:
        kwargs["savefig"] = dict(fname=args.savefig, dpi=150, bbox_inches="tight")
    mpf.plot(df, **kwargs)


if __name__ == "__main__":
    main()
