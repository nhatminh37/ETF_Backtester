#!/usr/bin/env python3
"""
Plot β₂₀ **dollar price spread** (not return): price_A − β₂₀×price_B.

Large charts for readability. Uses helpers in ``silver_backtesting.spread``.

Usage::

    python plot_price_spread.py --date 20250910
    python plot_price_spread.py --start 20250901 --end 20250915 --date 20250910 --output spread_price.html
    python plot_price_spread.py --date 20250910 --level-only
"""

from __future__ import annotations

import argparse
import os
import tempfile
import webbrowser
from datetime import datetime, timedelta

from silver_backtesting.config import Paths
from silver_backtesting.pair_quick import load_pair_panel
from silver_backtesting.spread import attach_beta20_price_spread, beta20_price_spread_figure


def _parse_args() -> argparse.Namespace:
    today = datetime.now() - timedelta(days=3)
    default_end = int(today.strftime("%Y%m%d"))
    default_start = int((today - timedelta(days=45)).strftime("%Y%m%d"))

    p = argparse.ArgumentParser(description="Plot β₂₀ dollar price spread (large figure).")
    p.add_argument("--ticker-a", default="PAAS")
    p.add_argument("--ticker-b", default="SLV")
    p.add_argument("--start", type=int, default=default_start)
    p.add_argument("--end", type=int, default=default_end)
    p.add_argument("--date", type=int, default=None, help="Session to plot (YYYYMMDD)")
    p.add_argument("--beta-lookback", type=int, default=20)
    p.add_argument("--data-folder", default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--no-browser", action="store_true")
    p.add_argument("--level-only", action="store_true", help="Only raw spread level panel")
    p.add_argument("--vs-open-only", action="store_true", help="Only vs-open panel")
    p.add_argument("--height", type=int, default=920)
    p.add_argument("--width", type=int, default=1500)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    folder = args.data_folder or Paths().min_folder
    focus = args.date or args.end

    print(f"Loading {args.ticker_a}/{args.ticker_b} …")
    panel = load_pair_panel(args.ticker_a, args.ticker_b, folder, args.start, args.end)
    df = attach_beta20_price_spread(
        panel,
        args.ticker_a,
        args.ticker_b,
        lookback_days=args.beta_lookback,
    )

    if focus not in set(df["Trading date"].astype(int)):
        raise SystemExit(f"No data for --date {focus} in loaded range.")

    beta_col = f"{args.ticker_a}-{args.ticker_b}_beta20"
    beta = df.loc[df["Trading date"] == focus, beta_col].dropna()
    print(f"  β₂₀ on {focus}: {beta.iloc[0]:.4f}" if not beta.empty else f"  β₂₀ on {focus}: n/a")

    show_vs_open = not args.level_only
    show_level = not args.vs_open_only
    fig = beta20_price_spread_figure(
        df,
        args.ticker_a,
        args.ticker_b,
        trading_date=focus,
        show_vs_open=show_vs_open,
        show_level=show_level,
        height=args.height,
        width=args.width,
    )

    if args.output:
        out = args.output
    else:
        tmp = tempfile.NamedTemporaryFile(suffix=".html", prefix="price_spread_", delete=False)
        out = tmp.name
        tmp.close()

    fig.write_html(out, include_plotlyjs="cdn")
    print(f"Chart: {out}")
    if not args.no_browser:
        webbrowser.open(f"file://{os.path.abspath(out)}")


if __name__ == "__main__":
    main()
