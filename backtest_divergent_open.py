#!/usr/bin/env python3
"""
Divergent-day fade: PAAS↑ & SLV↓ on day t−1 → short PAAS / long SLV on day t.

Signal (close-to-close on day t−1):
  r_PAAS > 0 and r_SLV < 0  →  day t: SHORT PAAS, LONG SLV
  (optional opposite) r_PAAS < 0 and r_SLV > 0  →  day t: LONG PAAS, SHORT SLV

Execution:
  Enter at day-t open (first RTH print / Yahoo Open)
  Exit at day-t close (last RTH print / Yahoo Close)
  Equal $ notional both legs — no beta hedge.

Usage::

    python backtest_divergent_open.py --start 20240101 --end 20251231
    python backtest_divergent_open.py --source yahoo --both-sides
"""

from __future__ import annotations

import argparse
import os
import webbrowser
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from silver_backtesting.config import Paths
from silver_backtesting.pair_quick import load_pair_panel


def session_ohlc(panel: pd.DataFrame, ticker_a: str, ticker_b: str) -> pd.DataFrame:
    """First/last RTH print per day as open/close; high/low from minute path."""
    rows = []
    for td, day in panel.groupby("Trading date"):
        day = day.sort_values("DT")
        oa, ob = float(day[ticker_a].iloc[0]), float(day[ticker_b].iloc[0])
        ca, cb = float(day[ticker_a].iloc[-1]), float(day[ticker_b].iloc[-1])
        ha, hb = float(day[ticker_a].max()), float(day[ticker_b].max())
        la, lb = float(day[ticker_a].min()), float(day[ticker_b].min())
        if min(oa, ob, ca, cb) <= 0:
            continue
        rows.append(
            {
                "Trading date": int(td),
                "open_a": oa,
                "open_b": ob,
                "high_a": ha,
                "high_b": hb,
                "low_a": la,
                "low_b": lb,
                "close_a": ca,
                "close_b": cb,
            }
        )
    return pd.DataFrame(rows).sort_values("Trading date").reset_index(drop=True)


def session_ohlc_yahoo(ticker_a: str, ticker_b: str, start: int, end: int) -> pd.DataFrame:
    import yfinance as yf

    start_s = pd.Timestamp(str(start)).strftime("%Y-%m-%d")
    end_s = (pd.Timestamp(str(end)) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    raw = yf.download(
        [ticker_a, ticker_b],
        start=start_s,
        end=end_s,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if raw.empty:
        raise RuntimeError("yfinance returned no data")

    if not isinstance(raw.columns, pd.MultiIndex):
        raise RuntimeError("Unexpected yfinance shape")

    o = raw["Open"][[ticker_a, ticker_b]]
    h = raw["High"][[ticker_a, ticker_b]]
    low = raw["Low"][[ticker_a, ticker_b]]
    c = raw["Close"][[ticker_a, ticker_b]]
    idx = o.dropna().index.intersection(c.dropna().index)
    o, h, low, c = o.loc[idx], h.loc[idx], low.loc[idx], c.loc[idx]

    def _td_index(ix: pd.DatetimeIndex) -> np.ndarray:
        ix = ix.tz_localize(None) if ix.tz is not None else ix
        return ix.strftime("%Y%m%d").astype(int)

    out = pd.DataFrame(
        {
            "Trading date": _td_index(pd.DatetimeIndex(idx)),
            "open_a": o[ticker_a].to_numpy(dtype=float),
            "open_b": o[ticker_b].to_numpy(dtype=float),
            "high_a": h[ticker_a].to_numpy(dtype=float),
            "high_b": h[ticker_b].to_numpy(dtype=float),
            "low_a": low[ticker_a].to_numpy(dtype=float),
            "low_b": low[ticker_b].to_numpy(dtype=float),
            "close_a": c[ticker_a].to_numpy(dtype=float),
            "close_b": c[ticker_b].to_numpy(dtype=float),
        }
    )
    return out.sort_values("Trading date").reset_index(drop=True)


def load_ohlc(
    ticker_a: str,
    ticker_b: str,
    start: int,
    end: int,
    folder: str | None,
    source: str,
) -> tuple[pd.DataFrame, str]:
    if source == "yahoo":
        return session_ohlc_yahoo(ticker_a, ticker_b, start, end), "yahoo"
    if source == "local":
        panel = load_pair_panel(ticker_a, ticker_b, folder, start, end)
        return session_ohlc(panel, ticker_a, ticker_b), "local_rth"
    try:
        panel = load_pair_panel(ticker_a, ticker_b, folder, start, end)
        return session_ohlc(panel, ticker_a, ticker_b), "local_rth"
    except (FileNotFoundError, OSError, RuntimeError) as exc:
        print(f"Local data unavailable ({exc}); falling back to Yahoo.")
        return session_ohlc_yahoo(ticker_a, ticker_b, start, end), "yahoo"


def _leg_exits(
    *,
    side: str,
    oa: float,
    ob: float,
    ha: float,
    hb: float,
    la: float,
    lb: float,
    ca: float,
    cb: float,
    tp_pct: float,
) -> tuple[float, float, str, str]:
    """
    Per-leg take-profit at ``tp_pct`` favorable move from open.
    Uses day high/low to detect touch; else exit at close.
    Returns (exit_a, exit_b, reason_a, reason_b).
    """
    thr = tp_pct / 100.0
    if side == "short_a_long_b":
        tp_a = oa * (1.0 - thr)  # short: price down
        tp_b = ob * (1.0 + thr)  # long: price up
        if la <= tp_a:
            exit_a, ra = tp_a, "TP"
        else:
            exit_a, ra = ca, "EOD"
        if hb >= tp_b:
            exit_b, rb = tp_b, "TP"
        else:
            exit_b, rb = cb, "EOD"
    else:
        tp_a = oa * (1.0 + thr)  # long A
        tp_b = ob * (1.0 - thr)  # short B
        if ha >= tp_a:
            exit_a, ra = tp_a, "TP"
        else:
            exit_a, ra = ca, "EOD"
        if lb <= tp_b:
            exit_b, rb = tp_b, "TP"
        else:
            exit_b, rb = cb, "EOD"
    return exit_a, exit_b, ra, rb


def run_backtest(
    ohlc: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    notional: float,
    both_sides: bool,
    slippage_pct: float,
    tp_pct: float = 0.1,
) -> pd.DataFrame:
    df = ohlc.copy()
    df["ret_a"] = df["close_a"] / df["close_a"].shift(1) - 1.0
    df["ret_b"] = df["close_b"] / df["close_b"].shift(1) - 1.0

    trades: list[dict] = []
    for i in range(1, len(df)):
        prev = df.iloc[i - 1]
        cur = df.iloc[i]
        ra, rb = float(prev["ret_a"]), float(prev["ret_b"])
        if not np.isfinite(ra) or not np.isfinite(rb):
            continue

        side = None
        if ra > 0 and rb < 0:
            side = "short_a_long_b"
        elif both_sides and ra < 0 and rb > 0:
            side = "long_a_short_b"
        if side is None:
            continue

        oa, ob = float(cur["open_a"]), float(cur["open_b"])
        ca, cb = float(cur["close_a"]), float(cur["close_b"])
        ha = float(cur["high_a"]) if "high_a" in cur.index else max(oa, ca)
        hb = float(cur["high_b"]) if "high_b" in cur.index else max(ob, cb)
        la = float(cur["low_a"]) if "low_a" in cur.index else min(oa, ca)
        lb = float(cur["low_b"]) if "low_b" in cur.index else min(ob, cb)
        if min(oa, ob, ca, cb) <= 0:
            continue

        sh_a = notional / oa
        sh_b = notional / ob
        slip = notional * slippage_pct / 100.0

        exit_a, exit_b, reason_a, reason_b = _leg_exits(
            side=side,
            oa=oa,
            ob=ob,
            ha=ha,
            hb=hb,
            la=la,
            lb=lb,
            ca=ca,
            cb=cb,
            tp_pct=tp_pct,
        )

        if side == "short_a_long_b":
            pnl_a = sh_a * (oa - exit_a)
            pnl_b = sh_b * (exit_b - ob)
            label = f"Short {ticker_a} / Long {ticker_b}"
        else:
            pnl_a = sh_a * (exit_a - oa)
            pnl_b = sh_b * (ob - exit_b)
            label = f"Long {ticker_a} / Short {ticker_b}"

        pnl = pnl_a + pnl_b - slip
        trades.append(
            {
                "Signal_date": int(prev["Trading date"]),
                "Trade_date": int(cur["Trading date"]),
                "Side": label,
                "Signal_ret_A_%": round(100.0 * ra, 4),
                "Signal_ret_B_%": round(100.0 * rb, 4),
                "Open_A": round(oa, 4),
                "Open_B": round(ob, 4),
                "Exit_A": round(exit_a, 4),
                "Exit_B": round(exit_b, 4),
                "Exit_A_reason": reason_a,
                "Exit_B_reason": reason_b,
                "PnL_A_$": round(pnl_a, 2),
                "PnL_B_$": round(pnl_b, 2),
                "Slippage_$": round(slip, 2),
                "PnL_$": round(pnl, 2),
                "PnL_%": round(100.0 * pnl / notional, 4),
            }
        )

    return pd.DataFrame(trades)


def summary_stats(trades: pd.DataFrame, notional: float) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame([("Trades", 0)], columns=["Metric", "Value"])
    pnl = trades["PnL_$"]
    rows = [
        ("Trades", len(trades)),
        ("Win rate", f"{100.0 * (pnl > 0).mean():.1f}%"),
        ("Total PnL $", round(float(pnl.sum()), 2)),
        ("Avg PnL $", round(float(pnl.mean()), 2)),
        ("Median PnL $", round(float(pnl.median()), 2)),
        ("Std PnL $", round(float(pnl.std(ddof=1)), 2) if len(pnl) > 1 else 0.0),
        ("Best $", round(float(pnl.max()), 2)),
        ("Worst $", round(float(pnl.min()), 2)),
        ("Total return on notional", f"{100.0 * pnl.sum() / notional:.2f}%"),
        ("Slippage total $", round(float(trades["Slippage_$"].sum()), 2)),
    ]
    if "Exit_A_reason" in trades.columns:
        rows.append(("Leg A TP rate", f"{100.0 * (trades['Exit_A_reason'] == 'TP').mean():.1f}%"))
        rows.append(("Leg B TP rate", f"{100.0 * (trades['Exit_B_reason'] == 'TP').mean():.1f}%"))
        both_tp = (trades["Exit_A_reason"] == "TP") & (trades["Exit_B_reason"] == "TP")
        rows.append(("Both legs TP", f"{100.0 * both_tp.mean():.1f}%"))
    return pd.DataFrame(rows, columns=["Metric", "Value"])


def build_html(
    trades: pd.DataFrame,
    summary: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    both_sides: bool,
    source: str,
    tp_pct: float,
) -> str:
    if trades.empty:
        fig_html = "<p>No trades.</p>"
    else:
        dates = pd.to_datetime(trades["Trade_date"].astype(str), format="%Y%m%d")
        cum = trades["PnL_$"].cumsum()
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                            subplot_titles=("Trade PnL $", "Cumulative PnL $"))
        fig.add_trace(go.Bar(x=dates, y=trades["PnL_$"], name="PnL"), row=1, col=1)
        fig.add_trace(go.Scatter(x=dates, y=cum, mode="lines", name="Cum PnL"), row=2, col=1)
        fig.update_layout(height=700, width=1100, template="plotly_white", showlegend=False,
                          title_text=f"Divergent fade {ticker_a}/{ticker_b} · TP {tp_pct}%/leg")
        fig_html = fig.to_html(full_html=False, include_plotlyjs="cdn")

    rule = (
        f"t−1: {ticker_a}↑ & {ticker_b}↓ → t open short {ticker_a} / long {ticker_b}<br>"
        f"Per-leg TP at <b>{tp_pct}%</b> favorable move (else EOD close)"
    )
    if both_sides:
        rule += (
            f"<br>also t−1: {ticker_a}↓ & {ticker_b}↑ → t open long {ticker_a} / short {ticker_b}"
        )

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/><title>divergent open {ticker_a}/{ticker_b}</title>
<style>
body {{ font-family: -apple-system, Segoe UI, sans-serif; margin: 24px; max-width: 1140px; }}
table {{ border-collapse: collapse; margin: 12px 0; }}
th, td {{ border: 1px solid #ddd; padding: 6px 10px; }}
th {{ background: #f5f5f5; }}
.note {{ color: #555; }}
</style></head><body>
<h1>Divergent open fade · per-leg TP</h1>
<p class="note">Source: <b>{source}</b> · Equal notional · No beta<br>{rule}</p>
<h2>Summary</h2>
{summary.to_html(index=False)}
<h2>Equity</h2>
{fig_html}
<h2>Trades</h2>
{trades.to_html(index=False) if not trades.empty else "<p>None</p>"}
</body></html>"""


def parse_args() -> argparse.Namespace:
    today = datetime.now().date()
    p = argparse.ArgumentParser(description="PAAS↑ SLV↓ fade: open→close next day")
    p.add_argument("--ticker-a", default="PAAS")
    p.add_argument("--ticker-b", default="SLV")
    p.add_argument("--start", type=int, default=int((today - timedelta(days=800)).strftime("%Y%m%d")))
    p.add_argument("--end", type=int, default=int(today.strftime("%Y%m%d")))
    p.add_argument("--notional", type=float, default=10_000.0)
    p.add_argument("--tp-pct", type=float, default=0.1,
                   help="Per-leg take-profit as %% price move from entry (default 0.1)")
    p.add_argument("--slippage-pct", type=float, default=0.0,
                   help="Flat cost as %% of one-leg notional deducted per trade")
    p.add_argument("--both-sides", action="store_true",
                   help="Also trade opposite divergence (A↓ B↑)")
    p.add_argument("--source", choices=("auto", "local", "yahoo"), default="auto")
    p.add_argument("--data-folder", default=None)
    p.add_argument("--output", default="divergent_open_backtest.html")
    p.add_argument("--csv", default="divergent_open_trades.csv")
    p.add_argument("--no-browser", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    folder = args.data_folder or Paths().min_folder
    print(f"Loading {args.ticker_a}/{args.ticker_b} …")
    ohlc, src = load_ohlc(
        args.ticker_a, args.ticker_b, args.start, args.end, folder, args.source
    )
    print(f"Source: {src}  sessions={len(ohlc)}  tp_pct={args.tp_pct}%/leg")

    trades = run_backtest(
        ohlc,
        args.ticker_a,
        args.ticker_b,
        args.notional,
        both_sides=args.both_sides,
        slippage_pct=args.slippage_pct,
        tp_pct=args.tp_pct,
    )
    summary = summary_stats(trades, args.notional)
    print("\n=== Summary ===")
    print(summary.to_string(index=False))

    trades.to_csv(args.csv, index=False)
    print(f"\nCSV: {args.csv}")

    html = build_html(
        trades, summary, args.ticker_a, args.ticker_b, args.both_sides, src, args.tp_pct
    )
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Report: {args.output}")
    if not args.no_browser:
        webbrowser.open(f"file://{os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
