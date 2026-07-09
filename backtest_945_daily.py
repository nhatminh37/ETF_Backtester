#!/usr/bin/env python3
"""
Daily 9:45 spread-level mean reversion — enter same 9:45, exit next 9:45.

Workflow (default):
  1. Each day at **9:45 ET**, spread level:
     S = P_A − β_D×P_B  where **β_D is from the prior trading session** (15m OLS).
  2. Rolling mean/std of the last ``window`` days' **9:45 spread levels** (excludes today).
  3. z = (S_today − roll_mean) / roll_std at 9:45.
  4. **Mon 9:45**, z = −1.8 → **long spread at Mon 9:45**, **exit Tue 9:45**.

PnL (spread level % move × notional):
  long:  (S_exit − S_entry) / |S_entry|
  short: (S_entry − S_exit) / |S_entry|

Usage::

    python backtest_945_daily.py --start 20240101 --end 20251231 --notional 10000
    python backtest_945_daily.py --window 20 --entry-std 1.5
"""

from __future__ import annotations

import argparse
import os
import webbrowser
from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd

from silver_backtesting.config import Paths
from silver_backtesting.pair_quick import load_pair_panel
from silver_backtesting.spread import attach_beta_d_price_spread

ANCHOR_HOUR = 9
ANCHOR_MINUTE = 45


@dataclass
class Daily945Params:
    window: int = 20
    entry_std: float = 1.5
    entry_lag: int = 0
    hold_days: int = 1
    notional: float = 10_000.0


def _level_pnl(side: str, entry_level: float, exit_level: float) -> float:
    denom = abs(entry_level)
    if denom < 1e-9:
        return 0.0
    if side == "long":
        return (exit_level - entry_level) / denom
    return (entry_level - exit_level) / denom


def extract_945_panel(
    df: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
) -> pd.DataFrame:
    """One row per session: 9:45 spread level S = P_A − β_D×P_B (β_D from prior session)."""
    spread_col = f"{ticker_a}-{ticker_b}"
    level_col = f"{spread_col}_price_bD"
    beta_col = f"{spread_col}_beta_D"

    rth = df.loc[df["ind_nyse_rth"] == True].copy()
    mask = (rth["DT"].dt.hour == ANCHOR_HOUR) & (rth["DT"].dt.minute == ANCHOR_MINUTE)
    snap = rth.loc[mask].copy()

    if snap.empty:
        fallback_rows: list[pd.Series] = []
        for td, day in rth.groupby("Trading date"):
            day = day.sort_values("DT")
            t_anchor = day["DT"].iloc[0].normalize() + pd.Timedelta(
                hours=ANCHOR_HOUR, minutes=ANCHOR_MINUTE
            )
            after = day.loc[day["DT"] >= t_anchor]
            if after.empty:
                continue
            fallback_rows.append(after.iloc[0])
        snap = pd.DataFrame(fallback_rows)

    out = snap[
        ["Trading date", "DT", ticker_a, ticker_b, beta_col, level_col]
    ].rename(
        columns={
            beta_col: "beta_D",
            level_col: "spread_level",
        }
    )
    out = out.dropna(subset=["spread_level", "beta_D"]).sort_values("Trading date")
    out["Trading date"] = out["Trading date"].astype(int)
    return out.reset_index(drop=True)


def attach_signal_z(daily: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    At each 9:45: z = (today's spread − rolling mean of prior ``window`` 9:45 spreads) / std.
    """
    out = daily.copy()
    hist_mean = out["spread_level"].shift(1).rolling(window=window, min_periods=window).mean()
    hist_std = out["spread_level"].shift(1).rolling(window=window, min_periods=window).std()
    out["roll_mean"] = hist_mean
    out["roll_std"] = hist_std
    out["z"] = (out["spread_level"] - hist_mean) / hist_std
    return out


def run_backtest(daily: pd.DataFrame, p: Daily945Params) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Map signal rows to entry/exit rows by trading-date index offset.
    """
    trades: list[dict] = []

    for pos in range(len(daily)):
        row = daily.iloc[pos]
        z = row["z"]
        if pd.isna(z):
            continue

        side: str | None = None
        if z <= -p.entry_std:
            side = "long"
        elif z >= p.entry_std:
            side = "short"
        if side is None:
            continue

        entry_idx = pos + p.entry_lag
        exit_idx = entry_idx + p.hold_days

        if exit_idx >= len(daily):
            continue

        entry_row = daily.iloc[entry_idx]
        exit_row = daily.iloc[exit_idx]
        entry_level = float(entry_row["spread_level"])
        exit_level = float(exit_row["spread_level"])
        pnl_dec = _level_pnl(side, entry_level, exit_level)

        trades.append(
            {
                "Signal_date": int(row["Trading date"]),
                "Signal_DT": row["DT"],
                "Signal_z": float(z),
                "Roll_mean_$": float(row["roll_mean"]) if pd.notna(row["roll_mean"]) else None,
                "Signal_level_$": float(row["spread_level"]),
                "Side": side,
                "Entry_date": int(entry_row["Trading date"]),
                "Entry_DT": entry_row["DT"],
                "Entry_level_$": entry_level,
                "Entry_beta_D": float(entry_row["beta_D"]),
                "Exit_date": int(exit_row["Trading date"]),
                "Exit_DT": exit_row["DT"],
                "Exit_level_$": exit_level,
                "Exit_beta_D": float(exit_row["beta_D"]),
                "Hold_sessions": p.hold_days,
                "Spread_move_$": exit_level - entry_level,
                "Return_pct": pnl_dec * 100,
                "PnL_$": pnl_dec * p.notional,
            }
        )

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        return trades_df, pd.DataFrame()

    daily_pnl = (
        trades_df.groupby("Exit_date", as_index=False)["PnL_$"]
        .sum()
        .rename(columns={"Exit_date": "Trading date", "PnL_$": "Day_PnL_$"})
        .sort_values("Trading date")
    )
    daily_pnl["Cum_PnL_$"] = daily_pnl["Day_PnL_$"].cumsum()
    return trades_df, daily_pnl


def summary_stats(
    trades_df: pd.DataFrame,
    daily_pnl: pd.DataFrame,
    p: Daily945Params,
    ticker_a: str,
    ticker_b: str,
) -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame({"Metric": ["Trades"], "Value": [0]})

    total = trades_df["PnL_$"].sum()
    wins = (trades_df["PnL_$"] > 0).sum()
    n = len(trades_df)
    avg_ret = trades_df["Return_pct"].mean()

    rows = [
        ("Trades", n),
        ("Long", int((trades_df["Side"] == "long").sum())),
        ("Short", int((trades_df["Side"] == "short").sum())),
        ("Total PnL ($)", round(total, 2)),
        ("Avg return per trade (%)", round(avg_ret, 3)),
        ("Avg PnL per trade ($)", round(trades_df["PnL_$"].mean(), 2)),
        ("Win rate (%)", round(100 * wins / n, 1)),
        ("Best trade ($)", round(trades_df["PnL_$"].max(), 2)),
        ("Worst trade ($)", round(trades_df["PnL_$"].min(), 2)),
        ("Spread", f"{ticker_a} − β_D×{ticker_b} ($), β_D from prior session"),
        ("Signal / entry", "9:45 ET same day"),
        ("Exit", f"9:45 ET +{p.hold_days} session(s)"),
        ("Rolling window (sessions)", p.window),
        ("Entry z-band", f"±{p.entry_std} σ"),
        ("Notional", p.notional),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value"])


def build_report_html(
    summary: pd.DataFrame,
    trades_df: pd.DataFrame,
    daily_pnl: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    p: Daily945Params,
) -> str:
    sum_html = summary.to_html(index=False, classes="data-table")
    trades_html = (
        trades_df.to_html(index=False, classes="data-table", float_format=lambda x: f"{x:,.4f}")
        if not trades_df.empty
        else "<p>No trades</p>"
    )
    daily_html = (
        daily_pnl.to_html(index=False, classes="data-table", float_format=lambda x: f"{x:,.2f}")
        if not daily_pnl.empty
        else ""
    )
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>9:45 daily spread backtest</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 24px; max-width: 1400px; }}
h1 {{ font-size: 1.35rem; }}
.data-table {{ border-collapse: collapse; width: 100%; margin: 16px 0; font-size: 0.85rem; }}
.data-table th, .data-table td {{ border: 1px solid #ddd; padding: 5px 8px; text-align: right; }}
.data-table th {{ background: #f5f5f5; }}
.data-table td:first-child, .data-table th:first-child {{ text-align: left; }}
.note {{ background: #f8f9fa; padding: 12px; border-radius: 6px; font-size: 0.9rem; line-height: 1.55; }}
</style></head><body>
<h1>9:45 daily spread — {ticker_a} vs {ticker_b}</h1>
<div class="note">
<p><b>Spread at 9:45:</b> S = P_{ticker_a} − β_D×P_{ticker_b} ($). β_D from <b>prior trading session</b> (15m OLS).</p>
<p><b>Signal:</b> z = (S − rolling mean of prior {p.window} days' 9:45 spreads) / rolling std.</p>
<p><b>Trade:</b> e.g. Mon 9:45 z = −1.8 → <b>long at Mon 9:45</b>, <b>exit Tue 9:45</b>.</p>
<p><b>Long spread</b> = buy {ticker_a}, sell β_D×{ticker_b}. <b>Short</b> = opposite.</p>
</div>
<h2>Summary</h2>{sum_html}
<h2>Trades</h2>{trades_html}
<h2>PnL by exit date</h2>{daily_html}
</body></html>"""


def _parse_args() -> argparse.Namespace:
    today = datetime.now() - timedelta(days=3)
    default_end = int(today.strftime("%Y%m%d"))
    default_start = int((today - timedelta(days=365)).strftime("%Y%m%d"))

    p = argparse.ArgumentParser(
        description="9:45 daily spread: signal+enter same 9:45, exit next 9:45."
    )
    p.add_argument("--ticker-a", default="PAAS")
    p.add_argument("--ticker-b", default="SLV")
    p.add_argument("--start", type=int, default=default_start)
    p.add_argument("--end", type=int, default=default_end)
    p.add_argument("--window", type=int, default=20, help="Rolling lookback (9:45 levels, sessions)")
    p.add_argument("--entry-std", type=float, default=1.5, help="Enter when |z| >= this")
    p.add_argument(
        "--entry-lag",
        type=int,
        default=0,
        help="Sessions from signal to entry 9:45 (0 = same bar as signal)",
    )
    p.add_argument(
        "--hold-days",
        type=int,
        default=1,
        help="Sessions from entry 9:45 to exit 9:45",
    )
    p.add_argument("--notional", type=float, default=10_000.0)
    p.add_argument("--data-folder", default=None)
    p.add_argument("--output", default="backtest_945_daily.html")
    p.add_argument("--no-browser", action="store_true")
    p.add_argument("--export-csv", default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    folder = args.data_folder or Paths().min_folder
    params = Daily945Params(
        window=args.window,
        entry_std=args.entry_std,
        entry_lag=args.entry_lag,
        hold_days=args.hold_days,
        notional=args.notional,
    )

    print(f"Loading {args.ticker_a}/{args.ticker_b} ({args.start}–{args.end}) …")
    panel = load_pair_panel(args.ticker_a, args.ticker_b, folder, args.start, args.end)
    df = attach_beta_d_price_spread(panel, args.ticker_a, args.ticker_b)

    daily = extract_945_panel(df, args.ticker_a, args.ticker_b)
    daily = attach_signal_z(daily, params.window)
    print(f"  {len(daily)} sessions with 9:45 snapshot")

    trades_df, daily_pnl = run_backtest(daily, params)
    summary = summary_stats(trades_df, daily_pnl, params, args.ticker_a, args.ticker_b)
    print("\n=== Summary ===")
    print(summary.to_string(index=False))

    if not trades_df.empty:
        print("\n=== Last 5 trades ===")
        cols = [
            "Signal_date",
            "Side",
            "Entry_date",
            "Exit_date",
            "Return_pct",
            "PnL_$",
        ]
        print(trades_df[cols].tail(5).to_string(index=False))

    out = args.output
    with open(out, "w", encoding="utf-8") as f:
        f.write(
            build_report_html(
                summary, trades_df, daily_pnl, args.ticker_a, args.ticker_b, params
            )
        )
    print(f"\nReport: {out}")

    if args.export_csv:
        os.makedirs(args.export_csv, exist_ok=True)
        trades_df.to_csv(os.path.join(args.export_csv, "trades.csv"), index=False)
        daily_pnl.to_csv(os.path.join(args.export_csv, "daily_pnl.csv"), index=False)
        daily.to_csv(os.path.join(args.export_csv, "snapshots_945.csv"), index=False)

    if not args.no_browser:
        webbrowser.open(f"file://{os.path.abspath(out)}")


if __name__ == "__main__":
    main()
