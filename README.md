# Trend Line Detector

Volume-adaptive stock trend line detection using Williams Fractal pivot points.

## Overview

Automatically detects support and resistance trend lines on stock price charts. Unlike traditional approaches that only use closing prices, this algorithm adapts its price selection based on volume context:

- **High-volume candles** (> 1.5x 20-day SMA): uses High/Low (wicks carry institutional significance)
- **Normal-volume candles**: uses max/min of Open and Close (body edges are more meaningful)

Trend lines require at least 3 pivot-point touches and are validated against candle-through violations before being scored and deduplicated.

## Algorithm Pipeline

1. **Fetch** OHLCV data via yfinance
2. **Classify** each candle's volume context (high vs. normal)
3. **Detect pivots** using Williams Fractal with adaptive span fallback and boundary scanning
4. **Fit trend lines** through pivot combinations (O(P^3) where P = pivot count, typically 10-25), with tiered candle-through validation, scoring by touch count / volume / span / recency, and 5-criteria deduplication
5. **Extend** the rightmost line to the chart edge (with fallback to next-rightmost if validation fails)
6. **Visualize** with mplfinance candlesticks, volume bars, MACD subplot, and trend line overlays

## Installation

```bash
git clone https://github.com/mvpp/trend-line-detector.git
cd trend-line-detector
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# Basic usage (saves chart to file)
MPLBACKEND=Agg python main.py --ticker AAPL --period 1y --savefig chart.png

# Show pivot point markers and use dashed lines
MPLBACKEND=Agg python main.py --ticker TSLA --period 1y --show-pivots --dash-lines --savefig chart.png

# Interactive display (requires GUI backend)
python main.py --ticker AAPL --period 1y

# Verbose output with custom parameters
MPLBACKEND=Agg python main.py --ticker MUSI --period 6mo --left-span 3 --right-span 3 --verbose --savefig chart.png
```

### CLI Options

| Flag | Default | Description |
|---|---|---|
| `--ticker` | AAPL | Stock ticker symbol |
| `--period` | 1y | yfinance period (e.g. 6mo, 1y, 2y) |
| `--interval` | 1d | yfinance interval (e.g. 1d, 1h) |
| `--left-span` | 5 | Williams Fractal left window |
| `--right-span` | 5 | Williams Fractal right window |
| `--vol-lookback` | 20 | Volume SMA window (days) |
| `--vol-multiplier` | 1.5 | High-volume threshold multiplier |
| `--tolerance` | 0.01 | Touch tolerance (fraction of price range) |
| `--min-touches` | 3 | Minimum pivot touches for a valid line |
| `--max-lines` | 5 | Max lines per type (support/resistance) |
| `--show-pivots` | off | Show pivot-point markers |
| `--dash-lines` | off | Draw dashed trend lines |
| `--savefig` | none | Save chart to file |
| `--verbose` | off | Print detailed pivot/line info |

## Configuration

All tunable constants are centralized in `config.py` with documentation. Key categories:

- **Pivot detection**: fractal span, fallback thresholds
- **Trend fitting**: candle-through tolerance, scoring weights, deduplication thresholds
- **Visualization**: line width, figure size, MACD parameters, panel ratios
