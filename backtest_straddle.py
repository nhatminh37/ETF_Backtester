#!/usr/bin/env python3
"""
Backtest: simultaneous long + short on β_D hedged cum spread (straddle).

Each session:
  - Enter **both** long and short at the first valid return bar (same entry cum).
  - Take profit 50%% at +1%% and 50%% at +2%% **relative to entry** (each leg).
  - Close any remaining size at session close (EOD).

Usage::

    python backtest_straddle.py --start 20250101 --end 20251231 --notional 10000 --output straddle_backtest.html
"""

from __future__ import annotations

import argparse
import os
import webbrowser
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from silver_backtesting.config import Paths
from silver_backtesting.pair_quick import attach_diff_ret, load_pair_panel


def _pct(x: float) -> float:
    return x / 100.0


@dataclass
class StraddleParams:
    tp_pcts: tuple[float, ...] = (1.0, 2.0)
    tp_fracs: tuple[float, ...] = (0.5, 0.5)
    notional: float = 10_000.0
    return_minutes: int = 5


@dataclass
class Leg:
    side: str
    entry_cum: float
    entry_dt: pd.Timestamp
    remaining: float = 1.0
    tp_hit: set[float] = field(default_factory=set)


def _leg_pnl(side: str, entry_cum: float, exit_cum: float, frac: float) -> float:
    if side == "long":
        return frac * (exit_cum - entry_cum)
    return frac * (entry_cum - exit_cum)


def _tp_targets(side: str, entry_cum: float, p: StraddleParams) -> list[tuple[float, float]]:
    """Return (absolute cum level, size fraction) for each TP."""
    out: list[tuple[float, float]] = []
    for tp_pct, tp_frac in zip(p.tp_pcts, p.tp_fracs, strict=True):
        offset = _pct(tp_pct)
        level = entry_cum + offset if side == "long" else entry_cum - offset
        out.append((level, tp_frac))
    return out


def _record_fill(
    fills: list[dict],
    *,
    td: int,
    side: str,
    reason: str,
    dt: pd.Timestamp,
    frac: float,
    entry_cum: float,
    exit_cum: float,
    pnl_dec: float,
    remaining: float,
    notional: float,
) -> None:
    fills.append(
        {
            "Trading date": td,
            "Side": side,
            "Reason": reason,
            "DT": dt,
            "Frac": frac,
            "Entry_cum_pct": entry_cum * 100,
            "Exit_cum_pct": exit_cum * 100,
            "PnL_pct": pnl_dec * 100,
            "PnL_$": pnl_dec * notional,
            "Remaining": remaining,
        }
    )


def _manage_leg(
    leg: Leg,
    cum: float,
    dt: pd.Timestamp,
    td: int,
    p: StraddleParams,
    fills: list[dict],
) -> None:
    if leg.remaining <= 1e-9:
        leg.remaining = 0.0
        return

    for level, tp_frac in _tp_targets(leg.side, leg.entry_cum, p):
        if level in leg.tp_hit or leg.remaining <= 0:
            continue
        hit = cum >= level if leg.side == "long" else cum <= level
        if not hit:
            continue
        frac = min(tp_frac, leg.remaining)
        pnl = _leg_pnl(leg.side, leg.entry_cum, cum, frac)
        leg.remaining -= frac
        leg.tp_hit.add(level)
        rel = abs(level - leg.entry_cum) * 100
        _record_fill(
            fills,
            td=td,
            side=leg.side,
            reason=f"TP@{rel:.1f}%",
            dt=dt,
            frac=frac,
            entry_cum=leg.entry_cum,
            exit_cum=cum,
            pnl_dec=pnl,
            remaining=leg.remaining,
            notional=p.notional,
        )


def simulate_session(
    day: pd.DataFrame,
    td: int,
    p: StraddleParams,
) -> tuple[list[dict], float]:
    fills: list[dict] = []
    long_leg: Leg | None = None
    short_leg: Leg | None = None

    for row in day.itertuples(index=False):
        if pd.isna(row.diff_ret) or pd.isna(row.cum_diff_ret) or pd.isna(row.beta_D):
            continue

        cum = float(row.cum_diff_ret)
        dt = row.DT

        if long_leg is None and short_leg is None:
            long_leg = Leg("long", cum, dt)
            short_leg = Leg("short", cum, dt)
            for side in ("long", "short"):
                _record_fill(
                    fills,
                    td=td,
                    side=side,
                    reason=f"Entry (β_D={row.beta_D:.3f})",
                    dt=dt,
                    frac=0.0,
                    entry_cum=cum,
                    exit_cum=cum,
                    pnl_dec=0.0,
                    remaining=1.0,
                    notional=p.notional,
                )

        if long_leg is not None and long_leg.remaining > 0:
            _manage_leg(long_leg, cum, dt, td, p, fills)
        if short_leg is not None and short_leg.remaining > 0:
            _manage_leg(short_leg, cum, dt, td, p, fills)

    exit_cum = float(day["cum_diff_ret"].dropna().iloc[-1])
    exit_dt = day["DT"].iloc[-1]
    for leg in (long_leg, short_leg):
        if leg is None or leg.remaining <= 0:
            continue
        frac = leg.remaining
        pnl = _leg_pnl(leg.side, leg.entry_cum, exit_cum, frac)
        leg.remaining = 0.0
        _record_fill(
            fills,
            td=td,
            side=leg.side,
            reason="EOD",
            dt=exit_dt,
            frac=frac,
            entry_cum=leg.entry_cum,
            exit_cum=exit_cum,
            pnl_dec=pnl,
            remaining=0.0,
            notional=p.notional,
        )

    session_pnl = sum(f["PnL_$"] for f in fills if not str(f["Reason"]).startswith("Entry"))
    return fills, session_pnl


def run_backtest(df: pd.DataFrame, p: StraddleParams) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_fills: list[dict] = []
    daily_rows: list[dict] = []

    for td, day in df.groupby("Trading date"):
        day = day.sort_values("DT")
        if day["beta_D"].isna().all():
            continue
        fills, session_pnl = simulate_session(day, int(td), p)
        if not fills:
            continue
        all_fills.extend(fills)
        daily_rows.append(
            {
                "Trading date": int(td),
                "Entry_cum_pct": fills[0]["Entry_cum_pct"],
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
    p: StraddleParams,
) -> pd.DataFrame:
    if daily_df.empty:
        return pd.DataFrame({"Metric": ["Sessions"], "Value": [0]})

    exits = fills_df[~fills_df["Reason"].astype(str).str.startswith("Entry")]
    total = daily_df["Session_PnL_$"].sum()
    win_days = (daily_df["Session_PnL_$"] > 0).sum()
    n_days = len(daily_df)
    tp_pnl = exits[exits["Reason"].astype(str).str.startswith("TP")]["PnL_$"].sum()
    eod_pnl = exits[exits["Reason"] == "EOD"]["PnL_$"].sum()

    rows = [
        ("Sessions", n_days),
        ("Total PnL ($)", round(total, 2)),
        ("Avg daily PnL ($)", round(daily_df["Session_PnL_$"].mean(), 2)),
        ("Win days (%)", round(100 * win_days / n_days, 1)),
        ("TP PnL ($)", round(tp_pnl, 2)),
        ("EOD PnL ($)", round(eod_pnl, 2)),
        ("Notional per leg", p.notional),
        ("Entry", "both legs at first valid bar"),
        ("TP", f"{int(p.tp_fracs[0]*100)}% @ +{p.tp_pcts[0]}%, {int(p.tp_fracs[1]*100)}% @ +{p.tp_pcts[1]}% vs entry"),
        ("Return bar (min)", p.return_minutes),
        ("Hedge β", "β_D (prior day 15m)"),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value"])


def build_report_html(
    summary: pd.DataFrame,
    daily_df: pd.DataFrame,
    fills_df: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
) -> str:
    fig = make_subplots(rows=2, cols=1, row_heights=[0.45, 0.55], vertical_spacing=0.08)
    if not daily_df.empty:
        x_dates = pd.to_datetime(daily_df["Trading date"].astype(str), format="%Y%m%d")
        fig.add_trace(go.Bar(x=x_dates, y=daily_df["Session_PnL_$"], name="Daily PnL"), row=1, col=1)
        fig.add_trace(
            go.Scatter(x=x_dates, y=daily_df["Cum_PnL_$"], mode="lines", name="Cumulative PnL"),
            row=2,
            col=1,
        )
    fig.update_layout(
        template="plotly_white",
        height=700,
        title="Straddle backtest — long + short, TP 1% / 2%",
    )
    fig.update_xaxes(title_text="Trading date", row=1, col=1)
    fig.update_xaxes(title_text="Trading date", row=2, col=1)
    fig.update_yaxes(title_text="Daily PnL ($)", row=1, col=1)
    fig.update_yaxes(title_text="Cumulative PnL ($)", row=2, col=1)
    chart = fig.to_html(full_html=False, include_plotlyjs="cdn")

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Straddle backtest</title>
<style>
body {{ font-family: sans-serif; margin: 2rem; }}
h1,h2 {{ color: #0066cc; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
th {{ background: #0066cc; color: white; padding: 6px; text-align: left; }}
td {{ border: 1px solid #ddd; padding: 6px; }}
</style></head><body>
<h1>Straddle — {ticker_a} vs {ticker_b}</h1>
<h2>Summary</h2>{summary.to_html(index=False, classes="data-table", border=0)}
<div>{chart}</div>
<h2>Daily PnL</h2>{daily_df.to_html(index=False, classes="data-table", border=0) if not daily_df.empty else "<p>No trades</p>"}
<h2>Fills (first 200)</h2>{fills_df.head(200).to_html(index=False, classes="data-table", border=0) if not fills_df.empty else ""}
</body></html>"""


def _parse_args() -> argparse.Namespace:
    today = datetime.now() - timedelta(days=3)
    default_end = int(today.strftime("%Y%m%d"))
    default_start = int((today - timedelta(days=45)).strftime("%Y%m%d"))

    p = argparse.ArgumentParser(description="Backtest long+short straddle on cum diff_ret.")
    p.add_argument("--ticker-a", default="PAAS")
    p.add_argument("--ticker-b", default="SLV")
    p.add_argument("--start", type=int, default=default_start)
    p.add_argument("--end", type=int, default=default_end)
    p.add_argument("--notional", type=float, default=10_000.0)
    p.add_argument("--return-minutes", type=int, default=5)
    p.add_argument("--data-folder", default=None)
    p.add_argument("--output", default="straddle_backtest.html")
    p.add_argument("--no-browser", action="store_true")
    p.add_argument("--export-csv", default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    folder = args.data_folder or Paths().min_folder
    params = StraddleParams(notional=args.notional, return_minutes=args.return_minutes)

    print(f"Loading {args.ticker_a}/{args.ticker_b} …")
    df = load_pair_panel(args.ticker_a, args.ticker_b, folder, args.start, args.end)
    n_1m = len(df)
    df = attach_diff_ret(df, args.ticker_a, args.ticker_b, return_minutes=params.return_minutes)
    print(
        f"  {df['Trading date'].nunique()} sessions, "
        f"{len(df):,} valid {params.return_minutes}m bars "
        f"(from {n_1m:,} 1-min rows)"
    )

    fills_df, daily_df = run_backtest(df, params)
    summary = summary_stats(daily_df, fills_df, params)

    print("\n=== Summary ===")
    print(summary.to_string(index=False))

    out = args.output
    with open(out, "w", encoding="utf-8") as f:
        f.write(build_report_html(summary, daily_df, fills_df, args.ticker_a, args.ticker_b))
    print(f"\nReport: {out}")

    if args.export_csv:
        os.makedirs(args.export_csv, exist_ok=True)
        daily_df.to_csv(os.path.join(args.export_csv, "daily_pnl.csv"), index=False)
        fills_df.to_csv(os.path.join(args.export_csv, "fills.csv"), index=False)

    if not args.no_browser:
        webbrowser.open(f"file://{os.path.abspath(out)}")


if __name__ == "__main__":
    main()
