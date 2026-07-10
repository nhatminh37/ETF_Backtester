#!/usr/bin/env python3
"""
Plot **dollar price spread** (not return): price_A − β×price_B.

Large charts for readability. Uses helpers in ``silver_backtesting.spread``.

Usage::

    # Full year with β_D (same hedge as backtest):
    python plot_price_spread.py --start 20250101 --end 20251231 --mode range --beta D

    # β₂₀ (20-day daily lookback):
    python plot_price_spread.py --start 20250101 --end 20251231 --mode range --beta 20

    # Single session (3 panels: vs 9:45, level, β):
    python plot_price_spread.py --start 20250101 --end 20251231 --date 20250910 --mode day --beta D

    # β hedge ratio panel only:
    python plot_price_spread.py --date 20250910 --mode day --beta 20 --beta-only
"""

from __future__ import annotations

import argparse
import os
import tempfile
import webbrowser
from datetime import datetime, timedelta

from silver_backtesting.config import Paths
from silver_backtesting.pair_quick import load_pair_panel
from silver_backtesting.spread import (
    attach_beta20_price_spread,
    attach_beta_d_price_spread,
    beta_price_spread_figure,
)


def _parse_args() -> argparse.Namespace:
    today = datetime.now() - timedelta(days=3)
    default_end = int(today.strftime("%Y%m%d"))
    default_start = int((today - timedelta(days=45)).strftime("%Y%m%d"))

    p = argparse.ArgumentParser(
        description="Plot dollar price spread (β₂₀ or β_D hedge, large figure)."
    )
    p.add_argument("--ticker-a", default="PAAS")
    p.add_argument("--ticker-b", default="SLV")
    p.add_argument("--start", type=int, default=default_start)
    p.add_argument("--end", type=int, default=default_end)
    p.add_argument("--date", type=int, default=None, help="Single session (mode=day only)")
    p.add_argument(
        "--mode",
        choices=("day", "range"),
        default=None,
        help="'range' = all sessions in start–end (default if --date omitted); "
        "'day' = one session (--date or last day in range)",
    )
    p.add_argument(
        "--beta",
        choices=("20", "D"),
        default="20",
        help="Hedge: '20' = 20-day daily OLS; 'D' = prior session 15m OLS (backtest β_D)",
    )
    p.add_argument("--beta-lookback", type=int, default=20, help="Days for --beta 20 only")
    p.add_argument("--data-folder", default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--no-browser", action="store_true")
    p.add_argument("--level-only", action="store_true", help="Only raw spread level panel")
    p.add_argument("--vs-945-only", action="store_true", help="Only vs-9:45 panel")
    p.add_argument("--beta-only", action="store_true", help="Only chosen β hedge ratio panel")
    p.add_argument("--no-beta-panel", action="store_true", help="Hide β hedge ratio panel")
    p.add_argument("--height", type=int, default=920)
    p.add_argument("--width", type=int, default=1500)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    folder = args.data_folder or Paths().min_folder
    mode = args.mode or ("day" if args.date is not None else "range")
    beta_label = "β_D" if args.beta == "D" else "β₂₀"

    print(f"Loading {args.ticker_a}/{args.ticker_b} ({args.start}–{args.end}), {beta_label} …")
    panel = load_pair_panel(args.ticker_a, args.ticker_b, folder, args.start, args.end)
    if args.beta == "D":
        df = attach_beta_d_price_spread(panel, args.ticker_a, args.ticker_b)
    else:
        df = attach_beta20_price_spread(
            panel,
            args.ticker_a,
            args.ticker_b,
            lookback_days=args.beta_lookback,
        )
    n_sessions = df["Trading date"].nunique()
    print(f"  {n_sessions} sessions, {len(df):,} 1-min bars")

    if mode == "day":
        focus = args.date or int(df["Trading date"].max())
        if focus not in set(df["Trading date"].astype(int)):
            raise SystemExit(f"No data for --date {focus} in loaded range.")
        trading_date: int | None = focus
        print(f"  mode=day, plotting {focus}")
    else:
        trading_date = None
        print(f"  mode=range, plotting {args.start}–{args.end}")

    show_vs_anchor = not args.level_only and not args.beta_only
    show_level = not args.vs_945_only and not args.beta_only
    show_beta = not args.no_beta_panel and not (args.level_only or args.vs_945_only)
    height = args.height if mode == "day" else max(args.height, 1000)

    fig = beta_price_spread_figure(
        df,
        args.ticker_a,
        args.ticker_b,
        beta=args.beta,
        trading_date=trading_date,
        show_vs_anchor=show_vs_anchor,
        show_level=show_level,
        show_beta=show_beta,
        height=height,
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
