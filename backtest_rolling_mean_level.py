#!/usr/bin/env python3
"""
Rolling-mean mean reversion on β_D **dollar spread level**.

Spread level: S = P_A − β_D×P_B  (same hedge as return backtests).

Mean reversion (one position at a time per session; re-entry after flat):
  Long spread  when z <= −entry_std   (level below rolling mean — "cheap")
  Short spread when z >= +entry_std   (level above rolling mean — "rich")
  Exit long    when z >= −exit_std    (default: back to mean, z=0)
  Exit short   when z <= +exit_std
  Stop (opt.)  when |z| exceeds stop_std against the position
  EOD flat any remainder at last 1-min bar.

z = (S − rolling_mean) / rolling_std  within session (1-min bars).

PnL scales spread level change as % of |entry level| × notional:
  long:  (S_exit − S_entry) / |S_entry|
  short: (S_entry − S_exit) / |S_entry|

Usage::

    python backtest_rolling_mean_level.py --start 20250101 --end 20251231 --notional 10000
    python backtest_rolling_mean_level.py --window 60 --entry-std 1.5 --exit-std 0
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


@dataclass
class RollingMeanParams:
    window: int = 30
    entry_std: float = 1.5
    exit_std: float = 0.0
    stop_std: float | None = 3.0
    notional: float = 10_000.0


@dataclass
class OpenPosition:
    side: str
    entry_level: float
    entry_z: float
    entry_dt: pd.Timestamp


def _level_pnl(side: str, entry_level: float, exit_level: float) -> float:
    """Return PnL as decimal fraction of notional (spread % move)."""
    denom = abs(entry_level)
    if denom < 1e-9:
        return 0.0
    if side == "long":
        return (exit_level - entry_level) / denom
    return (entry_level - exit_level) / denom


def _attach_rolling_z(
    df: pd.DataFrame,
    level_col: str,
    window: int,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for _, day in df.groupby("Trading date", sort=False):
        chunk = day.sort_values("DT").copy()
        roll = chunk[level_col].rolling(window=window, min_periods=window)
        chunk["roll_mean"] = roll.mean()
        chunk["roll_std"] = roll.std()
        chunk["z"] = (chunk[level_col] - chunk["roll_mean"]) / chunk["roll_std"]
        parts.append(chunk)
    return pd.concat(parts).sort_values("DT").reset_index(drop=True)


def _record_fill(
    fills: list[dict],
    *,
    td: int,
    side: str,
    reason: str,
    dt: pd.Timestamp,
    entry_level: float,
    exit_level: float,
    entry_z: float,
    exit_z: float,
    beta_d: float,
    pnl_dec: float,
    notional: float,
) -> None:
    fills.append(
        {
            "Trading date": td,
            "Side": side,
            "Reason": reason,
            "DT": dt,
            "Beta_D": beta_d,
            "Entry_level_$": entry_level,
            "Exit_level_$": exit_level,
            "Entry_z": entry_z,
            "Exit_z": exit_z,
            "PnL_pct": pnl_dec * 100,
            "PnL_$": pnl_dec * notional,
        }
    )


def simulate_session(
    day: pd.DataFrame,
    td: int,
    level_col: str,
    p: RollingMeanParams,
) -> tuple[list[dict], float]:
    fills: list[dict] = []
    pos: OpenPosition | None = None

    for _, row in day.iterrows():
        level = row[level_col]
        if pd.isna(level) or pd.isna(row["z"]) or pd.isna(row["beta_D"]):
            continue

        level = float(level)
        z = float(row["z"])
        dt = row["DT"]
        beta = float(row["beta_D"])

        if pos is None:
            if z <= -p.entry_std:
                pos = OpenPosition("long", level, z, dt)
                _record_fill(
                    fills,
                    td=td,
                    side="long",
                    reason=f"Entry z={z:.2f} (below mean)",
                    dt=dt,
                    entry_level=level,
                    exit_level=level,
                    entry_z=z,
                    exit_z=z,
                    beta_d=beta,
                    pnl_dec=0.0,
                    notional=p.notional,
                )
            elif z >= p.entry_std:
                pos = OpenPosition("short", level, z, dt)
                _record_fill(
                    fills,
                    td=td,
                    side="short",
                    reason=f"Entry z={z:.2f} (above mean)",
                    dt=dt,
                    entry_level=level,
                    exit_level=level,
                    entry_z=z,
                    exit_z=z,
                    beta_d=beta,
                    pnl_dec=0.0,
                    notional=p.notional,
                )
            continue

        exit_reason: str | None = None
        if pos.side == "long":
            if z >= -p.exit_std:
                exit_reason = f"Exit at mean (z={z:.2f})"
            elif p.stop_std is not None and z <= -p.stop_std:
                exit_reason = f"Stop (z={z:.2f})"
        else:
            if z <= p.exit_std:
                exit_reason = f"Exit at mean (z={z:.2f})"
            elif p.stop_std is not None and z >= p.stop_std:
                exit_reason = f"Stop (z={z:.2f})"

        if exit_reason:
            pnl = _level_pnl(pos.side, pos.entry_level, level)
            _record_fill(
                fills,
                td=td,
                side=pos.side,
                reason=exit_reason,
                dt=dt,
                entry_level=pos.entry_level,
                exit_level=level,
                entry_z=pos.entry_z,
                exit_z=z,
                beta_d=beta,
                pnl_dec=pnl,
                notional=p.notional,
            )
            pos = None

    if pos is not None:
        last = day.dropna(subset=[level_col, "z"]).iloc[-1]
        exit_level = float(last[level_col])
        exit_z = float(last["z"])
        pnl = _level_pnl(pos.side, pos.entry_level, exit_level)
        _record_fill(
            fills,
            td=td,
            side=pos.side,
            reason="EOD",
            dt=last["DT"],
            entry_level=pos.entry_level,
            exit_level=exit_level,
            entry_z=pos.entry_z,
            exit_z=exit_z,
            beta_d=float(last["beta_D"]),
            pnl_dec=pnl,
            notional=p.notional,
        )

    session_pnl = sum(
        f["PnL_$"] for f in fills if not str(f["Reason"]).startswith("Entry")
    )
    return fills, session_pnl


def run_backtest(
    df: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    p: RollingMeanParams,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    spread_col = f"{ticker_a}-{ticker_b}"
    level_col = f"{spread_col}_price_bD"
    beta_col = f"{spread_col}_beta_D"

    work = df.loc[df["ind_nyse_rth"] == True].copy()
    work = work.rename(columns={beta_col: "beta_D"})
    work = _attach_rolling_z(work, level_col, p.window)

    all_fills: list[dict] = []
    daily_rows: list[dict] = []

    for td, day in work.groupby("Trading date"):
        day = day.sort_values("DT")
        if day["beta_D"].isna().all():
            continue
        fills, session_pnl = simulate_session(day, int(td), level_col, p)
        if not fills:
            continue
        all_fills.extend(fills)
        entries = [f for f in fills if str(f["Reason"]).startswith("Entry")]
        daily_rows.append(
            {
                "Trading date": int(td),
                "Trades": len([f for f in fills if not str(f["Reason"]).startswith("Entry")]),
                "Long": sum(1 for f in entries if f["Side"] == "long"),
                "Short": sum(1 for f in entries if f["Side"] == "short"),
                "Session_PnL_$": session_pnl,
            }
        )

    fills_df = pd.DataFrame(all_fills)
    daily_df = pd.DataFrame(daily_rows)
    if not daily_df.empty:
        daily_df = daily_df.sort_values("Trading date")
        daily_df["Cum_PnL_$"] = daily_df["Session_PnL_$"].cumsum()
    return fills_df, daily_df


def summary_stats(
    daily_df: pd.DataFrame,
    fills_df: pd.DataFrame,
    p: RollingMeanParams,
    ticker_a: str,
    ticker_b: str,
) -> pd.DataFrame:
    if daily_df.empty:
        return pd.DataFrame({"Metric": ["Sessions"], "Value": [0]})

    exits = fills_df[~fills_df["Reason"].astype(str).str.startswith("Entry")]
    n_mean = exits["Reason"].astype(str).str.startswith("Exit at mean").sum()
    n_stop = exits["Reason"].astype(str).str.startswith("Stop").sum()
    n_eod = (exits["Reason"] == "EOD").sum()
    total = daily_df["Session_PnL_$"].sum()
    win_days = (daily_df["Session_PnL_$"] > 0).sum()
    n_days = len(daily_df)

    rows = [
        ("Sessions with activity", n_days),
        ("Total PnL ($)", round(total, 2)),
        ("Avg daily PnL ($)", round(daily_df["Session_PnL_$"].mean(), 2)),
        ("Win days (%)", round(100 * win_days / n_days, 1)),
        ("Round-trip exits", len(exits)),
        ("Exit at mean", int(n_mean)),
        ("Stop exits", int(n_stop)),
        ("EOD exits", int(n_eod)),
        ("Spread level", f"{ticker_a} − β_D×{ticker_b} ($)"),
        ("Rolling window (min)", p.window),
        ("Entry z-band", f"±{p.entry_std} σ"),
        ("Exit z", f"±{p.exit_std} σ"),
        ("Stop z", "off" if p.stop_std is None else f"±{p.stop_std} σ"),
        ("Notional", p.notional),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value"])


def build_report_html(
    summary: pd.DataFrame,
    daily_df: pd.DataFrame,
    fills_df: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
) -> str:
    sum_html = summary.to_html(index=False, classes="data-table")
    daily_html = (
        daily_df.to_html(index=False, classes="data-table", float_format=lambda x: f"{x:,.2f}")
        if not daily_df.empty
        else "<p>No trades</p>"
    )
    fills_html = (
        fills_df.to_html(index=False, classes="data-table", float_format=lambda x: f"{x:,.4f}")
        if not fills_df.empty
        else ""
    )
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Rolling mean spread level</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 24px; max-width: 1200px; }}
h1 {{ font-size: 1.4rem; }}
.data-table {{ border-collapse: collapse; width: 100%; margin: 16px 0; font-size: 0.9rem; }}
.data-table th, .data-table td {{ border: 1px solid #ddd; padding: 6px 10px; text-align: right; }}
.data-table th {{ background: #f5f5f5; }}
.data-table td:first-child, .data-table th:first-child {{ text-align: left; }}
.note {{ background: #f8f9fa; padding: 12px; border-radius: 6px; font-size: 0.9rem; line-height: 1.5; }}
</style></head><body>
<h1>Rolling-mean spread level — {ticker_a} vs {ticker_b}</h1>
<div class="note">
<p><b>Long spread</b> = buy {ticker_a}, sell β_D×{ticker_b} (dollar-neutral hedge).
You profit when spread level <i>S = P_{ticker_a} − β_D×P_{ticker_b}</i> <b>rises</b>.</p>
<p><b>Short spread</b> = sell {ticker_a}, buy β_D×{ticker_b}.
You profit when <i>S falls</i>.</p>
<p>This backtest is <b>mean reversion</b>: long when S is below its rolling mean, short when above.</p>
</div>
<h2>Summary</h2>{sum_html}
<h2>Daily PnL</h2>{daily_html}
<h2>Fills</h2>{fills_html}
</body></html>"""


def _parse_args() -> argparse.Namespace:
    today = datetime.now() - timedelta(days=3)
    default_end = int(today.strftime("%Y%m%d"))
    default_start = int((today - timedelta(days=45)).strftime("%Y%m%d"))

    p = argparse.ArgumentParser(
        description="Rolling-mean mean reversion on β_D dollar spread level."
    )
    p.add_argument("--ticker-a", default="PAAS")
    p.add_argument("--ticker-b", default="SLV")
    p.add_argument("--start", type=int, default=default_start)
    p.add_argument("--end", type=int, default=default_end)
    p.add_argument("--window", type=int, default=30, help="Rolling mean/std window (minutes)")
    p.add_argument("--entry-std", type=float, default=1.5, help="Enter when |z| >= this")
    p.add_argument("--exit-std", type=float, default=0.0, help="Exit long when z >= -exit; short when z <= +exit")
    p.add_argument("--stop-std", type=float, default=3.0, help="Stop when z moves further against; 0=off")
    p.add_argument("--notional", type=float, default=10_000.0)
    p.add_argument("--data-folder", default=None)
    p.add_argument("--output", default="rolling_mean_level_backtest.html")
    p.add_argument("--no-browser", action="store_true")
    p.add_argument("--export-csv", default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    folder = args.data_folder or Paths().min_folder
    stop_std = None if args.stop_std <= 0 else args.stop_std
    params = RollingMeanParams(
        window=args.window,
        entry_std=args.entry_std,
        exit_std=args.exit_std,
        stop_std=stop_std,
        notional=args.notional,
    )

    print(f"Loading {args.ticker_a}/{args.ticker_b} …")
    panel = load_pair_panel(args.ticker_a, args.ticker_b, folder, args.start, args.end)
    df = attach_beta_d_price_spread(panel, args.ticker_a, args.ticker_b)
    print(f"  {df['Trading date'].nunique()} sessions, {len(df):,} 1-min bars")

    fills_df, daily_df = run_backtest(df, args.ticker_a, args.ticker_b, params)
    summary = summary_stats(daily_df, fills_df, params, args.ticker_a, args.ticker_b)
    print("\n=== Summary ===")
    print(summary.to_string(index=False))

    out = args.output
    with open(out, "w", encoding="utf-8") as f:
        f.write(
            build_report_html(summary, daily_df, fills_df, args.ticker_a, args.ticker_b)
        )
    print(f"\nReport: {out}")

    if args.export_csv:
        os.makedirs(args.export_csv, exist_ok=True)
        daily_df.to_csv(os.path.join(args.export_csv, "daily_pnl.csv"), index=False)
        fills_df.to_csv(os.path.join(args.export_csv, "fills.csv"), index=False)

    if not args.no_browser:
        webbrowser.open(f"file://{os.path.abspath(out)}")


if __name__ == "__main__":
    main()
