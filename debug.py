#!/usr/bin/env python3
"""
Debug diff_ret ladder backtest — print logs to diagnose PnL / data issues.

Checks:
  - 1-min data coverage & gaps per session
  - strict vs naive 5m bar counts (gap-fill false signals)
  - large |diff_ret| / cum_diff jumps
  - entry / TP / SL / EOD fill breakdown
  - worst & best PnL days with full fill trail

Usage::

    python debug.py --start 20250101 --end 20251231
    python debug.py --start 20250101 --end 20251231 --date 20250315
    python debug.py --start 20250101 --end 20251231 --log debug.log
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from typing import TextIO

import numpy as np
import pandas as pd

from backtest_diff_ret import (
    StrategyParams,
    _pct,
    run_backtest,
    simulate_session,
    summary_stats,
)
from silver_backtesting.config import Paths
from silver_backtesting.pair_quick import (
    _valid_bar_returns,
    attach_diff_ret,
    load_pair_panel,
    prior_session_15m_beta_by_date,
)


def _header(title: str, out: TextIO) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}", file=out)


def _expected_rth_minutes(td: int) -> int:
    """NYSE RTH 9:30–15:59 inclusive ≈ 390 minutes."""
    return 390


def session_1m_coverage(panel_1m: pd.DataFrame) -> pd.DataFrame:
    """Per-session 1-min bar counts and gap statistics."""
    rows: list[dict] = []
    for td, g in panel_1m.groupby("Trading date"):
        g = g.sort_values("DT")
        gaps_min = g["DT"].diff().dt.total_seconds() / 60
        gap_mask = gaps_min > 1.5
        rows.append(
            {
                "Trading date": int(td),
                "bars_1m": len(g),
                "expected_1m": _expected_rth_minutes(int(td)),
                "missing_1m": max(0, _expected_rth_minutes(int(td)) - len(g)),
                "gaps_gt_1min": int(gap_mask.sum()),
                "max_gap_min": float(gaps_min.max()) if len(g) > 1 else 0.0,
                "first_DT": g["DT"].iloc[0],
                "last_DT": g["DT"].iloc[-1],
            }
        )
    return pd.DataFrame(rows)


def naive_bar_returns(
    panel_1m: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    return_minutes: int,
) -> pd.DataFrame:
    """Old resample-based returns (can span gaps — for comparison only)."""
    freq = f"{return_minutes}min"
    chunks: list[pd.DataFrame] = []
    for td, group in panel_1m.groupby("Trading date"):
        prices = (
            group.set_index("DT")[[ticker_a, ticker_b]]
            .sort_index()
            .resample(freq, label="right", closed="right")
            .last()
            .dropna(how="all")
        )
        rets = prices.pct_change().dropna(how="any")
        if rets.empty:
            continue
        chunk = rets.reset_index()
        chunk["Trading date"] = td
        chunks.append(chunk)
    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True)


def compare_strict_vs_naive(
    panel_1m: pd.DataFrame,
    bars: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    return_minutes: int,
    out: TextIO,
) -> pd.DataFrame:
    """Sessions where naive resample has more bars than strict (gap artifacts)."""
    strict = bars.groupby("Trading date").size().rename("strict_bars")
    naive = naive_bar_returns(panel_1m, ticker_a, ticker_b, return_minutes)
    naive_cnt = naive.groupby("Trading date").size().rename("naive_bars")
    cmp_df = pd.concat([strict, naive_cnt], axis=1).fillna(0).astype(int)
    cmp_df["extra_naive"] = cmp_df["naive_bars"] - cmp_df["strict_bars"]
    cmp_df = cmp_df.reset_index()

    _header("1. STRICT vs NAIVE bar counts (gap artifact check)", out)
    print(f"  Total strict {return_minutes}m bars: {int(strict.sum())}", file=out)
    print(f"  Total naive  {return_minutes}m bars: {int(naive_cnt.sum())}", file=out)
    print(f"  Extra naive-only bars:   {int(cmp_df['extra_naive'].clip(lower=0).sum())}", file=out)

    bad = cmp_df[cmp_df["extra_naive"] > 0].sort_values("extra_naive", ascending=False)
    if bad.empty:
        print("  OK — no sessions with extra naive-only bars.", file=out)
    else:
        print(f"  Sessions with extra naive bars: {len(bad)} (top 15):", file=out)
        print(bad.head(15).to_string(index=False), file=out)
    return cmp_df


def large_moves(bars: pd.DataFrame, out: TextIO, top_n: int = 20) -> None:
    """Flag unusually large single-bar returns or cum jumps."""
    _header("2. LARGE single-bar moves (possible bad prints / gaps)", out)
    if bars.empty:
        print("  No bars.", file=out)
        return

    b = bars.copy()
    b["abs_diff_ret_pct"] = b["diff_ret"].abs() * 100
    b["cum_step_pct"] = b.groupby("Trading date")["cum_diff_ret"].diff().abs() * 100

    for col, label, thresh in [
        ("ret_a", "r_A", 2.0),
        ("ret_b", "r_B", 2.0),
        ("abs_diff_ret_pct", "|diff_ret|", 2.0),
        ("cum_step_pct", "|Δcum| between bars", 2.0),
    ]:
        if col not in b.columns:
            continue
        hit = b[b[col] > thresh].sort_values(col, ascending=False)
        print(f"\n  {label} > {thresh}%: {len(hit)} rows", file=out)
        if not hit.empty:
            cols = ["Trading date", "DT", "ret_a", "ret_b", "beta_D", "diff_ret", "cum_diff_ret"]
            cols = [c for c in cols if c in hit.columns]
            show = hit.head(top_n)[cols].copy()
            for c in ("ret_a", "ret_b", "diff_ret", "cum_diff_ret"):
                if c in show.columns:
                    show[c] = (show[c] * 100).round(3)
            if "beta_D" in show.columns:
                show["beta_D"] = show["beta_D"].round(3)
            print(show.to_string(index=False), file=out)


def entry_diagnostics(bars: pd.DataFrame, p: StrategyParams, out: TextIO) -> None:
    """Sessions that cross entry threshold — context at first trigger."""
    _header("3. ENTRY triggers (first bar crossing ±entry)", out)
    entry = _pct(p.entry_pct)
    rows: list[dict] = []
    for td, day in bars.groupby("Trading date"):
        day = day.sort_values("DT")
        beta = day["beta_D"].iloc[0]
        for side, cond in (("long", lambda c: c >= entry), ("short", lambda c: c <= -entry)):
            for row in day.itertuples(index=False):
                if pd.isna(row.cum_diff_ret):
                    continue
                if cond(float(row.cum_diff_ret)):
                    rows.append(
                        {
                            "Trading date": int(td),
                            "Side": side,
                            "DT": row.DT,
                            "beta_D": round(float(row.beta_D), 3),
                            "cum_pct": round(float(row.cum_diff_ret) * 100, 3),
                            "ret_a_pct": round(float(row.ret_a) * 100, 3),
                            "ret_b_pct": round(float(row.ret_b) * 100, 3),
                            "diff_ret_pct": round(float(row.diff_ret) * 100, 3),
                        }
                    )
                    break

    if not rows:
        print("  No entries in range.", file=out)
        return
    ent = pd.DataFrame(rows)
    print(f"  Total entry events: {len(ent)}", file=out)
    print(ent.head(30).to_string(index=False), file=out)

    instant_tp = ent[ent["cum_pct"].abs() >= p.tp_start_pct]
    if not instant_tp.empty:
        print(f"\n  WARNING: {len(instant_tp)} entries already at/ past first TP ({p.tp_start_pct}%):", file=out)
        print(instant_tp.to_string(index=False), file=out)


def pnl_day_breakdown(
    fills_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    out: TextIO,
    top_n: int = 10,
) -> None:
    """Worst/best days with every fill."""
    _header("4. PnL DAY BREAKDOWN (worst / best)", out)
    if daily_df.empty:
        print("  No trading days.", file=out)
        return

    for label, subset in (
        ("WORST", daily_df.nsmallest(top_n, "Session_PnL_$")),
        ("BEST", daily_df.nlargest(top_n, "Session_PnL_$")),
    ):
        print(f"\n  --- {label} {top_n} days ---", file=out)
        for _, drow in subset.iterrows():
            td = int(drow["Trading date"])
            print(
                f"\n  Date {td}  Session_PnL=${drow['Session_PnL_$']:.2f}  "
                f"Cum=${drow.get('Cum_PnL_$', float('nan')):.2f}  "
                f"Entries L/S={drow.get('Long', '?')}/{drow.get('Short', '?')}",
                file=out,
            )
            day_fills = fills_df[fills_df["Trading date"] == td].copy()
            if day_fills.empty:
                print("    (no fills)", file=out)
                continue
            show = day_fills[
                ["DT", "Side", "Reason", "Frac", "Entry_cum_pct", "Exit_cum_pct", "PnL_$", "Remaining"]
            ].copy()
            show["Entry_cum_pct"] = show["Entry_cum_pct"].round(3)
            show["Exit_cum_pct"] = show["Exit_cum_pct"].round(3)
            show["PnL_$"] = show["PnL_$"].round(2)
            show["Remaining"] = show["Remaining"].round(2)
            print(show.to_string(index=False), file=out)


def fill_reason_summary(fills_df: pd.DataFrame, out: TextIO) -> None:
    _header("5. FILL REASON SUMMARY", out)
    if fills_df.empty:
        print("  No fills.", file=out)
        return
    exits = fills_df[~fills_df["Reason"].astype(str).str.startswith("Entry")]
    grp = exits.groupby("Reason").agg(
        count=("PnL_$", "size"),
        total_pnl=("PnL_$", "sum"),
        avg_pnl=("PnL_$", "mean"),
    )
    grp = grp.sort_values("total_pnl", ascending=False)
    print(grp.round(2).to_string(), file=out)


def both_sides_days(daily_df: pd.DataFrame, out: TextIO) -> None:
    _header("6. DAYS WITH BOTH LONG AND SHORT", out)
    if daily_df.empty:
        return
    both = daily_df[(daily_df["Long"] > 0) & (daily_df["Short"] > 0)]
    print(f"  Count: {len(both)}", file=out)
    if not both.empty:
        print(both.sort_values("Session_PnL_$").head(20).to_string(index=False), file=out)


def deep_dive_day(
    td: int,
    panel_1m: pd.DataFrame,
    bars: pd.DataFrame,
    fills_df: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    return_minutes: int,
    p: StrategyParams,
    out: TextIO,
) -> None:
    """Full minute + bar path for one session."""
    _header(f"7. DEEP DIVE — session {td}", out)

    d1 = panel_1m[panel_1m["Trading date"] == td].sort_values("DT")
    db = bars[bars["Trading date"] == td].sort_values("DT")
    print(f"  1-min bars: {len(d1)}  |  valid {return_minutes}m bars: {len(db)}", file=out)
    if not d1.empty:
        gaps = d1["DT"].diff().dt.total_seconds() / 60
        big = d1.loc[gaps > 1.5, ["DT"]].copy()
        big["gap_min"] = gaps[gaps > 1.5].values
        if not big.empty:
            print(f"  Gaps > 1 min in 1-min data ({len(big)}):", file=out)
            print(big.head(20).to_string(index=False), file=out)

    if not db.empty:
        show = db[
            ["DT", "ret_a", "ret_b", "beta_D", "diff_ret", "cum_diff_ret"]
        ].copy()
        for c in ("ret_a", "ret_b", "diff_ret", "cum_diff_ret"):
            show[c] = (show[c] * 100).round(4)
        show["beta_D"] = show["beta_D"].round(4)
        print(f"\n  Valid {return_minutes}m bar path:", file=out)
        print(show.to_string(index=False), file=out)

    fills = fills_df[fills_df["Trading date"] == td] if not fills_df.empty else pd.DataFrame()
    if not fills.empty:
        print("\n  Fills:", file=out)
        print(
            fills[
                ["DT", "Side", "Reason", "Frac", "Entry_cum_pct", "Exit_cum_pct", "PnL_$", "Remaining"]
            ].to_string(index=False),
            file=out,
        )

    _, session_pnl = simulate_session(db, td, p) if not db.empty else ([], 0.0)
    print(f"\n  Re-simulated session PnL: ${session_pnl:.2f}", file=out)


def data_folder_banner(folder: str, ticker_a: str, ticker_b: str, out: TextIO) -> None:
    _header("0. CONFIG", out)
    print(f"  Data folder : {folder}", file=out)
    print(f"  Pair        : {ticker_a} / {ticker_b}", file=out)


def _parse_args() -> argparse.Namespace:
    today = datetime.now() - timedelta(days=3)
    p = argparse.ArgumentParser(description="Debug diff_ret backtest PnL & data quality.")
    p.add_argument("--ticker-a", default="PAAS")
    p.add_argument("--ticker-b", default="SLV")
    p.add_argument("--start", type=int, default=20250101)
    p.add_argument("--end", type=int, default=int(today.strftime("%Y%m%d")))
    p.add_argument("--date", type=int, default=None, help="Deep-dive one YYYYMMDD session")
    p.add_argument("--return-minutes", type=int, default=5)
    p.add_argument("--entry", type=float, default=1.5)
    p.add_argument("--notional", type=float, default=10_000.0)
    p.add_argument("--data-folder", default=None)
    p.add_argument("--log", default=None, help="Write all output to this file too")
    p.add_argument("--export-dir", default="debug_out", help="CSV export folder")
    return p.parse_args()


class _Tee:
    def __init__(self, *streams: TextIO):
        self.streams = streams

    def write(self, data: str) -> None:
        for s in self.streams:
            s.write(data)

    def flush(self) -> None:
        for s in self.streams:
            s.flush()


def main() -> None:
    args = _parse_args()
    folder = args.data_folder or Paths().min_folder
    params = StrategyParams(
        entry_pct=args.entry,
        notional=args.notional,
        return_minutes=args.return_minutes,
    )

    log_file = open(args.log, "w", encoding="utf-8") if args.log else None
    out: TextIO = _Tee(sys.stdout, log_file) if log_file else sys.stdout

    try:
        data_folder_banner(folder, args.ticker_a, args.ticker_b, out)

        print(f"\nLoading {args.ticker_a}/{args.ticker_b} …", file=out)
        panel_1m = load_pair_panel(
            args.ticker_a, args.ticker_b, folder, args.start, args.end
        )
        cov = session_1m_coverage(panel_1m)
        print(f"  1-min rows: {len(panel_1m):,} across {cov['Trading date'].nunique()} sessions", file=out)
        print(
            f"  Avg missing 1-min/session: {cov['missing_1m'].mean():.1f}  "
            f"max gaps>1min/session: {cov['gaps_gt_1min'].max()}",
            file=out,
        )

        bars = attach_diff_ret(
            panel_1m, args.ticker_a, args.ticker_b, return_minutes=params.return_minutes
        )
        print(
            f"  Valid {params.return_minutes}m bars: {len(bars):,}  "
            f"({100 * len(bars) / max(len(panel_1m), 1):.1f}% of 1-min rows)",
            file=out,
        )

        beta = prior_session_15m_beta_by_date(panel_1m, args.ticker_a, args.ticker_b)
        print(
            f"  β_D sessions: {beta.notna().sum()}  "
            f"range [{beta.min():.3f}, {beta.max():.3f}]",
            file=out,
        )

        cmp_df = compare_strict_vs_naive(
            panel_1m, bars, args.ticker_a, args.ticker_b, params.return_minutes, out
        )
        large_moves(bars, out)
        entry_diagnostics(bars, params, out)

        fills_df, daily_df = run_backtest(bars, params)
        summary = summary_stats(daily_df, fills_df, params)

        _header("BACKTEST SUMMARY (same as backtest_diff_ret.py)", out)
        print(summary.to_string(index=False), file=out)

        pnl_day_breakdown(fills_df, daily_df, out)
        fill_reason_summary(fills_df, out)
        both_sides_days(daily_df, out)

        if args.date:
            deep_dive_day(
                args.date,
                panel_1m,
                bars,
                fills_df,
                args.ticker_a,
                args.ticker_b,
                params.return_minutes,
                params,
                out,
            )
        else:
            if not daily_df.empty:
                worst_td = int(daily_df.loc[daily_df["Session_PnL_$"].idxmin(), "Trading date"])
                print(f"\n  Tip: re-run with --date {worst_td} for worst-day deep dive.", file=out)

        import os

        os.makedirs(args.export_dir, exist_ok=True)
        cov.to_csv(os.path.join(args.export_dir, "session_1m_coverage.csv"), index=False)
        cmp_df.to_csv(os.path.join(args.export_dir, "strict_vs_naive_bars.csv"), index=False)
        if not bars.empty:
            bars.to_csv(os.path.join(args.export_dir, "valid_bars.csv"), index=False)
        if not daily_df.empty:
            daily_df.to_csv(os.path.join(args.export_dir, "daily_pnl.csv"), index=False)
        if not fills_df.empty:
            fills_df.to_csv(os.path.join(args.export_dir, "fills.csv"), index=False)
        print(f"\nCSV exported to ./{args.export_dir}/", file=out)
        if args.log:
            print(f"Log written to {args.log}", file=sys.stdout)

    finally:
        if log_file:
            log_file.close()


if __name__ == "__main__":
    main()
