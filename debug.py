#!/usr/bin/env python3
"""
Debug Kalman pairs backtest — log diagnostics for PnL / hypothesis checks.

Studies:
  - Mean-revert vs momentum (flipped direction) side-by-side
  - PnL by direction, entry-z sign, exit reason
  - Slippage drag vs gross PnL
  - Prior β₀ / R₀ sample
  - Worst / best days with trade trail
  - Optional single-session deep dive

Usage::

    python debug.py --start 20240101 --end 20251231
    python debug.py --start 20240101 --end 20251231 --log kalman_debug.log
    python debug.py --start 20240101 --end 20251231 --date 20250315
    python debug.py --start 20240101 --end 20251231 --momentum
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from datetime import datetime, timedelta
from typing import TextIO

import numpy as np
import pandas as pd

from claude_surprise_5m import (
    TradeParams,
    _entry_slippage_usd,
    _prior_close_by_date,
    fit_prior,
    run_backtest,
    signal_mode_label,
    simulate_session,
    summary_stats,
)
from silver_backtesting.config import Paths
from silver_backtesting.pair_quick import load_pair_panel


def _header(title: str, out: TextIO) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}", file=out)


def _expected_rth_minutes() -> int:
    return 390


def session_1m_coverage(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for td, g in panel.groupby("Trading date"):
        g = g.sort_values("DT")
        gaps_min = g["DT"].diff().dt.total_seconds() / 60
        gap_mask = gaps_min > 1.5
        exp = _expected_rth_minutes()
        rows.append(
            {
                "Trading date": int(td),
                "bars_1m": len(g),
                "expected_1m": exp,
                "missing_1m": max(0, exp - len(g)),
                "gaps_gt_1min": int(gap_mask.sum()),
                "max_gap_min": float(gaps_min.max()) if len(g) > 1 else 0.0,
                "first_DT": g["DT"].iloc[0],
                "last_DT": g["DT"].iloc[-1],
            }
        )
    return pd.DataFrame(rows)


def prior_sample(
    panel: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    lookback: int,
    out: TextIO,
    n: int = 5,
) -> None:
    _header("2. PRIOR SAMPLE (intraday-only OLS)", out)
    dates = sorted(panel["Trading date"].unique())
    rows: list[dict] = []
    for td in dates[5:5 + n]:
        hist = panel[panel["Trading date"] < td]
        try:
            p = fit_prior(hist, ticker_a, ticker_b, lookback)
            rows.append(
                {
                    "Trading date": int(td),
                    "beta0": round(p.beta0, 4),
                    "P0": round(p.P0, 6),
                    "R0": round(p.R0, 8),
                }
            )
        except ValueError:
            rows.append({"Trading date": int(td), "beta0": None, "P0": None, "R0": None})
    if rows:
        print(pd.DataFrame(rows).to_string(index=False), file=out)
    else:
        print("  No prior rows.", file=out)


def trade_diagnostics(trades: pd.DataFrame, tp: TradeParams, label: str, out: TextIO) -> dict:
    _header(f"TRADES — {label}", out)
    if trades.empty:
        print("  No trades.", file=out)
        return {"label": label, "trades": 0, "total_pnl": 0.0}

    gross = trades["PnL_$"] + trades["Slippage_$"]
    total = float(trades["PnL_$"].sum())
    slip = float(trades["Slippage_$"].sum())
    wins = int((trades["PnL_$"] > 0).sum())
    n = len(trades)

    print(f"  Mode           : {signal_mode_label(tp.momentum)}", file=out)
    print(f"  Trades         : {n}", file=out)
    print(f"  Total PnL      : ${total:,.2f}", file=out)
    print(f"  Total slippage : ${slip:,.2f}  ({100*slip/max(abs(total)+slip,1):.1f}% of |gross|+slip)", file=out)
    print(f"  Gross (pre-slip): ${gross.sum():,.2f}", file=out)
    print(f"  Win rate       : {100*wins/n:.1f}%", file=out)
    print(f"  Avg PnL/trade  : ${trades['PnL_$'].mean():.2f}", file=out)
    print(f"  Flipped PnL*   : ${-total:,.2f}  (*if every trade took opposite direction)", file=out)

    print("\n  By direction:", file=out)
    by_dir = trades.groupby("Direction")["PnL_$"].agg(["count", "sum", "mean"]).round(2)
    print(by_dir.to_string(), file=out)

    print("\n  By entry-z sign:", file=out)
    t = trades.copy()
    t["z_sign"] = np.where(t["Entry_z"] > 0, "z>0 (PAAS rich)", "z<0 (PAAS cheap)")
    by_z = t.groupby("z_sign")["PnL_$"].agg(["count", "sum", "mean"]).round(2)
    print(by_z.to_string(), file=out)

    print("\n  By exit reason:", file=out)
    by_ex = trades.groupby("Exit_reason")["PnL_$"].agg(["count", "sum", "mean"]).round(2)
    print(by_ex.sort_values("sum", ascending=False).to_string(), file=out)

    print("\n  Entry z stats:", file=out)
    print(
        f"    mean={trades['Entry_z'].mean():.3f}  "
        f"median={trades['Entry_z'].median():.3f}  "
        f"min={trades['Entry_z'].min():.3f}  max={trades['Entry_z'].max():.3f}",
        file=out,
    )

    return {
        "label": label,
        "trades": n,
        "total_pnl": total,
        "gross_pnl": float(gross.sum()),
        "slippage": slip,
        "win_rate": wins / n,
        "flipped_pnl": -total,
    }


def hypothesis_comparison(mr: dict, mom: dict, out: TextIO) -> None:
    _header("HYPOTHESIS COMPARISON (mean-revert vs momentum)", out)
    if mr["trades"] == 0 and mom["trades"] == 0:
        print("  No trades in either mode.", file=out)
        return

    print(f"  Mean-revert total PnL : ${mr['total_pnl']:,.2f}  ({mr['trades']} trades)", file=out)
    print(f"  Momentum total PnL    : ${mom['total_pnl']:,.2f}  ({mom['trades']} trades)", file=out)

    mr_flip = mr.get("flipped_pnl", 0)
    print(f"\n  Mean-revert flipped   : ${mr_flip:,.2f}", file=out)
    if mr["total_pnl"] < 0 and mr_flip > 0:
        print("  → Fade loses; opposite direction wins → market may be MOMENTUM at this horizon.", file=out)
    elif mr["total_pnl"] > 0 and mr_flip < 0:
        print("  → Fade wins; opposite loses → mean-revert hypothesis supported.", file=out)
    elif abs(mr["total_pnl"]) < 500:
        print("  → Flat / noise; try longer range or tune entry_z / hold.", file=out)
    else:
        print("  → Both directions lose: likely costs / no edge, not simple sign flip.", file=out)

    if mom["trades"] > 0 and mr["trades"] > 0:
        better = "momentum" if mom["total_pnl"] > mr["total_pnl"] else "mean_revert"
        diff = abs(mom["total_pnl"] - mr["total_pnl"])
        print(f"\n  Better mode in-sample: {better}  (Δ ${diff:,.2f})", file=out)


def pnl_day_breakdown(trades: pd.DataFrame, daily: pd.DataFrame, out: TextIO, top_n: int = 8) -> None:
    _header("DAY BREAKDOWN (worst / best)", out)
    if daily.empty:
        print("  No daily rows.", file=out)
        return

    for title, subset in (
        ("WORST", daily.nsmallest(top_n, "Session_PnL_$")),
        ("BEST", daily.nlargest(top_n, "Session_PnL_$")),
    ):
        print(f"\n  --- {title} {top_n} ---", file=out)
        for _, drow in subset.iterrows():
            td = int(drow["Trading date"])
            day_tr = trades[trades["Trading date"] == td]
            print(
                f"\n  {td}  Session=${drow['Session_PnL_$']:.2f}  "
                f"Trades={int(drow['Trades'])}  Cum=${drow.get('Cum_PnL_$', float('nan')):.2f}",
                file=out,
            )
            if day_tr.empty:
                continue
            cols = [
                "Trade_id", "Direction", "Signal_time", "Entry_time", "Exit_time",
                "Entry_z", "Exit_reason", "Slippage_$", "PnL_$",
            ]
            cols = [c for c in cols if c in day_tr.columns]
            print(day_tr[cols].to_string(index=False), file=out)


def deep_dive_day(
    td: int,
    panel: pd.DataFrame,
    trades: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    tp: TradeParams,
    prior_lookback: int,
    mu: float,
    out: TextIO,
) -> None:
    _header(f"DEEP DIVE — session {td}", out)
    day = panel[panel["Trading date"] == td].sort_values("DT")
    print(f"  1-min bars: {len(day)}", file=out)
    if day.empty:
        return

    hist = panel[panel["Trading date"] < td]
    try:
        prior = fit_prior(hist, ticker_a, ticker_b, prior_lookback)
        print(f"  Prior: beta0={prior.beta0:.4f}  P0={prior.P0:.6f}  R0={prior.R0:.8f}", file=out)
    except ValueError as exc:
        print(f"  Prior failed: {exc}", file=out)
        return

    pc = _prior_close_by_date(panel, ticker_a, ticker_b).get(td)
    print(f"  Prior close: {pc}", file=out)

    fills = simulate_session(day, td, ticker_a, ticker_b, prior, mu, tp, prior_close=pc)
    print(f"  Re-sim fills: {len(fills)}  session PnL=${sum(f['PnL_$'] for f in fills):.2f}", file=out)
    for f in fills:
        print(
            f"    {f['Direction']}  z={f['Entry_z']}  "
            f"{f['Entry_time']}→{f['Exit_time']}  {f['Exit_reason']}  ${f['PnL_$']:.2f}",
            file=out,
        )

    if not trades.empty:
        day_tr = trades[trades["Trading date"] == td]
        if not day_tr.empty:
            print("\n  Backtest trades that day:", file=out)
            print(day_tr.to_string(index=False), file=out)


class _Tee:
    def __init__(self, *streams: TextIO):
        self.streams = streams

    def write(self, data: str) -> None:
        for s in self.streams:
            s.write(data)

    def flush(self) -> None:
        for s in self.streams:
            s.flush()


def _parse_args() -> argparse.Namespace:
    today = datetime.now() - timedelta(days=3)
    p = argparse.ArgumentParser(description="Debug Kalman pairs backtest.")
    p.add_argument("--ticker-a", default="PAAS")
    p.add_argument("--ticker-b", default="SLV")
    p.add_argument("--start", type=int, default=20240101)
    p.add_argument("--end", type=int, default=int(today.strftime("%Y%m%d")))
    p.add_argument("--date", type=int, default=None, help="Deep-dive one YYYYMMDD")
    p.add_argument("--entry-z", type=float, default=2.0)
    p.add_argument("--hold-minutes", type=int, default=15)
    p.add_argument("--tp-pct", type=float, default=1.0)
    p.add_argument("--mu", type=float, default=0.99)
    p.add_argument("--prior-lookback", type=int, default=60)
    p.add_argument("--notional", type=float, default=10_000.0)
    p.add_argument("--slippage-pct", type=float, default=0.3)
    p.add_argument("--same-bar-entry", action="store_true")
    p.add_argument("--momentum", action="store_true", help="Primary run uses momentum mode")
    p.add_argument("--no-open-gap", action="store_true")
    p.add_argument("--no-adaptive-r", action="store_true")
    p.add_argument("--data-folder", default=None)
    p.add_argument("--log", default="kalman_debug.log", help="Log file path")
    p.add_argument("--export-dir", default="debug_out")
    p.add_argument("--skip-compare", action="store_true", help="Only run primary mode")
    return p.parse_args()


def _base_params(args: argparse.Namespace) -> TradeParams:
    return TradeParams(
        entry_z=args.entry_z,
        hold_minutes=args.hold_minutes,
        tp_pct=args.tp_pct,
        notional=args.notional,
        slippage_pct=args.slippage_pct,
        same_bar_entry=args.same_bar_entry,
        include_open_gap=not args.no_open_gap,
        adaptive_r=not args.no_adaptive_r,
        momentum=args.momentum,
    )


def main() -> None:
    args = _parse_args()
    folder = args.data_folder or Paths().min_folder
    tp_primary = _base_params(args)

    log_path = args.log
    log_file = open(log_path, "w", encoding="utf-8")
    out: TextIO = _Tee(sys.stdout, log_file)

    try:
        _header("0. CONFIG", out)
        print(f"  Data folder    : {folder}", file=out)
        print(f"  Pair           : {args.ticker_a} / {args.ticker_b}", file=out)
        print(f"  Range          : {args.start} – {args.end}", file=out)
        print(f"  Primary mode   : {signal_mode_label(tp_primary.momentum)}", file=out)
        print(f"  entry_z        : {tp_primary.entry_z}", file=out)
        print(f"  hold / tp      : {tp_primary.hold_minutes}m / {tp_primary.tp_pct}%", file=out)
        print(f"  slippage       : {tp_primary.slippage_pct}% (${_entry_slippage_usd(tp_primary.notional, tp_primary.slippage_pct):.2f}/trade)", file=out)
        print(f"  same_bar_entry : {tp_primary.same_bar_entry}", file=out)
        print(f"  open_gap       : {tp_primary.include_open_gap}", file=out)
        print(f"  adaptive_R     : {tp_primary.adaptive_r}", file=out)

        print(f"\nLoading …", file=out)
        panel = load_pair_panel(args.ticker_a, args.ticker_b, folder, args.start, args.end)
        cov = session_1m_coverage(panel)
        print(f"  1-min rows: {len(panel):,}  sessions: {cov['Trading date'].nunique()}", file=out)
        print(
            f"  Avg missing bars/session: {cov['missing_1m'].mean():.1f}  "
            f"max gap>1min: {int(cov['gaps_gt_1min'].max())}",
            file=out,
        )

        _header("1. DATA COVERAGE (worst 10 sessions)", out)
        print(cov.sort_values("missing_1m", ascending=False).head(10).to_string(index=False), file=out)

        prior_sample(panel, args.ticker_a, args.ticker_b, args.prior_lookback, out)

        tp_mr = replace(tp_primary, momentum=False)
        tp_mom = replace(tp_primary, momentum=True)
        compare_exports: tuple[pd.DataFrame, pd.DataFrame] | None = None

        if args.skip_compare:
            trades, daily = run_backtest(
                panel, args.ticker_a, args.ticker_b, args.prior_lookback, args.mu, tp_primary
            )
            trade_diagnostics(trades, tp_primary, signal_mode_label(tp_primary.momentum), out)
            print("\n  Summary:", file=out)
            print(summary_stats(trades, daily, tp_primary, args.mu).to_string(index=False), file=out)
        else:
            tr_mr, daily_mr = run_backtest(
                panel, args.ticker_a, args.ticker_b, args.prior_lookback, args.mu, tp_mr
            )
            tr_mom, daily_mom = run_backtest(
                panel, args.ticker_a, args.ticker_b, args.prior_lookback, args.mu, tp_mom
            )
            mr_stats = trade_diagnostics(tr_mr, tp_mr, "mean_revert", out)
            mom_stats = trade_diagnostics(tr_mom, tp_mom, "momentum", out)
            hypothesis_comparison(mr_stats, mom_stats, out)
            trades, daily = (tr_mom, daily_mom) if tp_primary.momentum else (tr_mr, daily_mr)
            print("\n  Summary (primary mode):", file=out)
            print(summary_stats(trades, daily, tp_primary, args.mu).to_string(index=False), file=out)
            compare_exports = (tr_mr, tr_mom)

        if not trades.empty:
            trades = trades.copy()
            if "Trade_id" not in trades.columns:
                trades.insert(0, "Trade_id", range(1, len(trades) + 1))

        pnl_day_breakdown(trades, daily, out)

        if args.date:
            deep_dive_day(
                args.date, panel, trades, args.ticker_a, args.ticker_b,
                tp_primary, args.prior_lookback, args.mu, out,
            )
        elif not daily.empty:
            worst_td = int(daily.loc[daily["Session_PnL_$"].idxmin(), "Trading date"])
            print(f"\n  Tip: python debug.py --date {worst_td} ... for worst-day deep dive.", file=out)

        os.makedirs(args.export_dir, exist_ok=True)
        cov.to_csv(os.path.join(args.export_dir, "session_1m_coverage.csv"), index=False)
        if not trades.empty:
            export_tr = trades.drop(columns=["_legs"], errors="ignore")
            export_tr.to_csv(os.path.join(args.export_dir, "trades_primary.csv"), index=False)
        if compare_exports is not None:
            tr_mr, tr_mom = compare_exports
            if not tr_mr.empty:
                tr_mr.to_csv(os.path.join(args.export_dir, "trades_mean_revert.csv"), index=False)
            if not tr_mom.empty:
                tr_mom.to_csv(os.path.join(args.export_dir, "trades_momentum.csv"), index=False)
        if not daily.empty:
            daily.to_csv(os.path.join(args.export_dir, "daily_pnl_primary.csv"), index=False)
        print(f"\nCSV → ./{args.export_dir}/", file=out)
        print(f"Log → {log_path}", file=out)

    finally:
        log_file.close()


if __name__ == "__main__":
    main()
