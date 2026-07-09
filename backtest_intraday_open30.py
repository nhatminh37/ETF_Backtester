#!/usr/bin/env python3
"""
Intraday open-window spread: β₂₀ return from 9:30 open, z-score vs history, exit 10:00.

Universe: first 30 minutes RTH only (9:30–10:00).

Spread return from session open (minute ``t``):
  r_A(t) = P_A(t) / open_A − 1
  r_B(t) = P_B(t) / open_B − 1
  spread_ret(t) = r_A(t) − β₂₀×r_B(t)

β₂₀ = OLS on prior 20 session **daily** returns (known at open).

Historical z at each clock minute (prior sessions, rolling ``hist_days``):
  z(t) = (spread_ret(t) − hist_mean) / hist_std

Signal window: **after first 10 min** (9:40–9:59). First bar where:
  z ≥ +entry_z → **long** (BUY A $10k, SELL β₂₀×B)
  z ≤ −entry_z → **short**

Always **exit both legs at 10:00** ET.

PnL = leg fills (same format as other backtesters).

Usage::

    python backtest_intraday_open30.py --start 20240101 --end 20251231 --notional 10000
    python backtest_intraday_open30.py --entry-z 1.5 --hist-days 20
"""

from __future__ import annotations

import argparse
import json
import os
import webbrowser
from dataclasses import dataclass
from datetime import datetime, time, timedelta

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from silver_backtesting.config import Paths
from silver_backtesting.pair_quick import load_pair_panel
from silver_backtesting.spread import lookback_beta_by_trading_date

OPEN_TIME = time(9, 30)
SIGNAL_EARLIEST = time(9, 40)
EXIT_TIME = time(10, 0)
WINDOW_END = time(10, 0)


@dataclass
class Open30Params:
    entry_z: float = 1.5
    hist_days: int = 20
    min_hist_days: int = 10
    beta_lookback: int = 20
    notional: float = 10_000.0


def _leg_breakdown(
    side: str,
    notional: float,
    beta_entry: float,
    price_a_entry: float,
    price_b_entry: float,
    price_a_exit: float,
    price_b_exit: float,
) -> dict:
    n_a = notional
    n_b = notional * beta_entry
    sh_a = n_a / price_a_entry if price_a_entry else 0.0
    sh_b = n_b / price_b_entry if price_b_entry else 0.0

    if side == "long":
        entry_a_action, entry_b_action = "BUY", "SELL"
        exit_a_action, exit_b_action = "SELL", "BUY"
        pnl_a = sh_a * (price_a_exit - price_a_entry)
        pnl_b = sh_b * (price_b_entry - price_b_exit)
    else:
        entry_a_action, entry_b_action = "SELL", "BUY"
        exit_a_action, exit_b_action = "BUY", "SELL"
        pnl_a = sh_a * (price_a_entry - price_a_exit)
        pnl_b = sh_b * (price_b_exit - price_b_entry)

    return {
        "entry": [
            {
                "Leg": "A",
                "Action": entry_a_action,
                "Price": round(price_a_entry, 4),
                "Notional_usd": round(n_a, 2),
                "Shares": round(sh_a, 4),
            },
            {
                "Leg": "B",
                "Action": entry_b_action,
                "Price": round(price_b_entry, 4),
                "Beta_20": round(beta_entry, 4),
                "Notional_usd": round(n_b, 2),
                "Shares": round(sh_b, 4),
            },
        ],
        "exit": [
            {
                "Leg": "A",
                "Action": exit_a_action,
                "Price": round(price_a_exit, 4),
                "Notional_usd": round(sh_a * price_a_exit, 2),
                "Shares": round(sh_a, 4),
            },
            {
                "Leg": "B",
                "Action": exit_b_action,
                "Price": round(price_b_exit, 4),
                "Notional_usd": round(sh_b * price_b_exit, 2),
                "Shares": round(sh_b, 4),
            },
        ],
        "PnL_A_usd": round(pnl_a, 2),
        "PnL_B_usd": round(pnl_b, 2),
        "PnL_total_usd": round(pnl_a + pnl_b, 2),
    }


def _clock_t(row_dt: pd.Timestamp) -> time:
    return row_dt.time().replace(second=0, microsecond=0)


def _in_open_window(t: time) -> bool:
    return OPEN_TIME <= t <= WINDOW_END


def build_minute_spread_panel(
    panel: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    beta_lookback: int,
) -> pd.DataFrame:
    """9:30–10:00 rows with open-anchored β₂₀ spread return per minute."""
    beta20 = lookback_beta_by_trading_date(
        panel, ticker_a, ticker_b, lookback_days=beta_lookback, lag_sessions=1
    )

    chunks: list[pd.DataFrame] = []
    for td, day in panel.groupby("Trading date"):
        day = day.sort_values("DT")
        day = day.loc[day["DT"].map(_clock_t).map(_in_open_window)]
        if day.empty:
            continue

        open_mask = day["DT"].map(_clock_t) == OPEN_TIME
        if not open_mask.any():
            open_row = day.iloc[0]
        else:
            open_row = day.loc[open_mask].iloc[0]

        open_a = float(open_row[ticker_a])
        open_b = float(open_row[ticker_b])
        beta = beta20.get(int(td), float("nan"))
        if pd.isna(beta) or open_a <= 0 or open_b <= 0:
            continue

        chunk = day.copy()
        chunk["open_a"] = open_a
        chunk["open_b"] = open_b
        chunk["beta_20"] = float(beta)
        chunk["ret_a_open"] = chunk[ticker_a] / open_a - 1.0
        chunk["ret_b_open"] = chunk[ticker_b] / open_b - 1.0
        chunk["spread_ret"] = chunk["ret_a_open"] - chunk["beta_20"] * chunk["ret_b_open"]
        chunk["clock"] = chunk["DT"].map(_clock_t)
        chunk["clock_str"] = chunk["clock"].astype(str).str.slice(0, 5)
        chunks.append(
            chunk[
                [
                    "Trading date",
                    "DT",
                    "clock",
                    "clock_str",
                    ticker_a,
                    ticker_b,
                    "beta_20",
                    "spread_ret",
                ]
            ]
        )

    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True).sort_values(["Trading date", "DT"])


def attach_historical_z(minute_df: pd.DataFrame, hist_days: int, min_hist_days: int) -> pd.DataFrame:
    """Per clock minute: z vs rolling prior-session mean/std of spread_ret."""
    parts: list[pd.DataFrame] = []
    for _, grp in minute_df.groupby("clock_str", sort=True):
        g = grp.sort_values("Trading date").copy()
        g["hist_mean"] = g["spread_ret"].shift(1).rolling(hist_days, min_periods=min_hist_days).mean()
        g["hist_std"] = g["spread_ret"].shift(1).rolling(hist_days, min_periods=min_hist_days).std()
        g["z"] = (g["spread_ret"] - g["hist_mean"]) / g["hist_std"]
        parts.append(g)
    return pd.concat(parts).sort_values(["Trading date", "DT"]).reset_index(drop=True)


def _signal_side(z: float, entry_z: float) -> str | None:
    if pd.isna(z):
        return None
    if z >= entry_z:
        return "long"
    if z <= -entry_z:
        return "short"
    return None


def simulate_session(
    day: pd.DataFrame,
    td: int,
    ticker_a: str,
    ticker_b: str,
    p: Open30Params,
) -> dict | None:
    exit_rows = day.loc[day["clock"] == EXIT_TIME]
    if exit_rows.empty:
        return None
    exit_row = exit_rows.iloc[0]

    signal_rows = day.loc[
        (day["clock"] >= SIGNAL_EARLIEST) & (day["clock"] < EXIT_TIME)
    ].sort_values("DT")

    side: str | None = None
    entry_row = None
    for _, row in signal_rows.iterrows():
        side = _signal_side(float(row["z"]), p.entry_z)
        if side is not None:
            entry_row = row
            break

    if side is None or entry_row is None:
        return None

    beta_e = float(entry_row["beta_20"])
    legs = _leg_breakdown(
        side,
        p.notional,
        beta_e,
        float(entry_row[ticker_a]),
        float(entry_row[ticker_b]),
        float(exit_row[ticker_a]),
        float(exit_row[ticker_b]),
    )
    pnl_usd = legs["PnL_total_usd"]
    return {
        "Trading date": td,
        "Side": side,
        "Signal_time": pd.Timestamp(entry_row["DT"]).strftime("%H:%M"),
        "Exit_time": pd.Timestamp(exit_row["DT"]).strftime("%H:%M"),
        "Signal_z": round(float(entry_row["z"]), 4),
        "Signal_spread_ret_pct": round(float(entry_row["spread_ret"]) * 100, 4),
        "Hist_mean_pct": round(float(entry_row["hist_mean"]) * 100, 4)
        if pd.notna(entry_row["hist_mean"])
        else None,
        "Hist_std_pct": round(float(entry_row["hist_std"]) * 100, 4)
        if pd.notna(entry_row["hist_std"])
        else None,
        "Entry_beta_20": round(beta_e, 4),
        "Entry_time": pd.Timestamp(entry_row["DT"]).strftime("%Y-%m-%d %H:%M"),
        "Exit_time_full": pd.Timestamp(exit_row["DT"]).strftime("%Y-%m-%d %H:%M"),
        f"Entry_{ticker_a}": round(float(entry_row[ticker_a]), 4),
        f"Entry_{ticker_b}": round(float(entry_row[ticker_b]), 4),
        f"Exit_{ticker_a}": round(float(exit_row[ticker_a]), 4),
        f"Exit_{ticker_b}": round(float(exit_row[ticker_b]), 4),
        "Return_pct": round(100.0 * pnl_usd / p.notional, 4),
        "PnL_$": round(pnl_usd, 2),
        "PnL_A_$": legs["PnL_A_usd"],
        "PnL_B_$": legs["PnL_B_usd"],
        "_legs": legs,
    }


def run_backtest(
    minute_df: pd.DataFrame,
    p: Open30Params,
    ticker_a: str,
    ticker_b: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    trades: list[dict] = []
    trade_id = 0

    for td, day in minute_df.groupby("Trading date"):
        rec = simulate_session(day, int(td), ticker_a, ticker_b, p)
        if rec is None:
            continue
        trade_id += 1
        rec["Trade_id"] = trade_id
        trades.append(rec)

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        return trades_df, pd.DataFrame()

    daily_df = trades_df[["Trading date", "PnL_$"]].copy()
    daily_df = daily_df.rename(columns={"PnL_$": "Session_PnL_$"})
    daily_df["Trades"] = 1
    daily_df["Cum_PnL_$"] = daily_df["Session_PnL_$"].cumsum()
    return trades_df, daily_df


def summary_stats(
    trades_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    p: Open30Params,
    ticker_a: str,
    ticker_b: str,
) -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame({"Metric": ["Trades"], "Value": [0]})

    total = trades_df["PnL_$"].sum()
    wins = (trades_df["PnL_$"] > 0).sum()
    n = len(trades_df)

    rows = [
        ("Sessions traded", n),
        ("Long", int((trades_df["Side"] == "long").sum())),
        ("Short", int((trades_df["Side"] == "short").sum())),
        ("Total PnL ($)", round(total, 2)),
        ("Avg daily PnL ($)", round(daily_df["Session_PnL_$"].mean(), 2)),
        ("Win rate (%)", round(100 * wins / n, 1)),
        ("Spread return", f"r_A(open) − β₂₀×r_B(open)"),
        ("β₂₀ lookback (sessions)", p.beta_lookback),
        ("Hist z lookback (sessions)", p.hist_days),
        ("Entry z (σ)", f"±{p.entry_z}"),
        ("Signal window", "9:40–9:59 ET"),
        ("Exit", "10:00 ET"),
        ("Notional leg A", p.notional),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value"])


def _trading_dates_for_plot(daily_df: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(daily_df["Trading date"].astype(str), format="%Y%m%d")


def _daily_table_html(daily_df: pd.DataFrame) -> str:
    if daily_df.empty:
        return "<p>No trades</p>"
    cols = list(daily_df.columns)
    parts = [
        "<table id='daily-pnl-table' class='data-table interactive-table'>",
        "<thead><tr>",
        *[f"<th>{c}</th>" for c in cols],
        "</tr></thead><tbody>",
    ]
    for _, row in daily_df.iterrows():
        td = int(row["Trading date"])
        parts.append(f"<tr class='daily-row' data-td='{td}' tabindex='0'>")
        for c in cols:
            val = row[c]
            if c in ("Session_PnL_$", "Cum_PnL_$") and pd.notna(val):
                text = f"{float(val):,.2f}"
            else:
                text = str(val)
            parts.append(f"<td>{text}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "".join(parts)


def _trades_table_html(trades_df: pd.DataFrame) -> str:
    if trades_df.empty:
        return "<p>No trades</p>"
    cols = [
        "Trade_id",
        "Trading date",
        "Side",
        "Signal_time",
        "Signal_z",
        "Signal_spread_ret_pct",
        "Return_pct",
        "PnL_$",
    ]
    parts = [
        "<table id='trades-table' class='data-table interactive-table'>",
        "<thead><tr>",
        *[f"<th>{c}</th>" for c in cols],
        "</tr></thead><tbody>",
    ]
    for _, row in trades_df.iterrows():
        tid = int(row["Trade_id"])
        parts.append(f"<tr class='trade-row' data-trade='{tid}' tabindex='0'>")
        for c in cols:
            val = row[c]
            if c == "PnL_$":
                text = f"{float(val):,.2f}"
            elif c in ("Signal_z", "Signal_spread_ret_pct", "Return_pct"):
                text = f"{float(val):,.4f}"
            else:
                text = str(val)
            parts.append(f"<td>{text}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "".join(parts)


def _trades_json(trades_df: pd.DataFrame, ticker_a: str, ticker_b: str) -> str:
    if trades_df.empty:
        return "{}"
    out: dict[str, dict] = {}
    for _, row in trades_df.iterrows():
        tid = str(int(row["Trade_id"]))
        out[tid] = {
            "Trade_id": int(row["Trade_id"]),
            "Trading_date": int(row["Trading date"]),
            "Side": row["Side"],
            "Signal_time": row["Signal_time"],
            "Signal_z": float(row["Signal_z"]),
            "Signal_spread_ret_pct": float(row["Signal_spread_ret_pct"]),
            "Hist_mean_pct": row["Hist_mean_pct"],
            "Hist_std_pct": row["Hist_std_pct"],
            "Entry_time": row["Entry_time"],
            "Exit_time_full": row["Exit_time_full"],
            "Entry_beta_20": float(row["Entry_beta_20"]),
            "Entry_A": float(row[f"Entry_{ticker_a}"]),
            "Entry_B": float(row[f"Entry_{ticker_b}"]),
            "Exit_A": float(row[f"Exit_{ticker_a}"]),
            "Exit_B": float(row[f"Exit_{ticker_b}"]),
            "Return_pct": float(row["Return_pct"]),
            "PnL_usd": float(row["PnL_$"]),
            "PnL_A_usd": float(row["PnL_A_$"]),
            "PnL_B_usd": float(row["PnL_B_$"]),
            "legs": row["_legs"],
            "ticker_a": ticker_a,
            "ticker_b": ticker_b,
        }
    return json.dumps(out)


def _trades_by_td_json(trades_df: pd.DataFrame) -> str:
    if trades_df.empty:
        return "{}"
    grouped: dict[str, list[int]] = {}
    for _, row in trades_df.iterrows():
        td = str(int(row["Trading date"]))
        grouped.setdefault(td, []).append(int(row["Trade_id"]))
    return json.dumps(grouped)


def build_report_html(
    summary: pd.DataFrame,
    trades_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    p: Open30Params,
) -> str:
    fig = make_subplots(rows=2, cols=1, row_heights=[0.45, 0.55], vertical_spacing=0.08)
    if not daily_df.empty:
        x_dates = _trading_dates_for_plot(daily_df)
        fig.add_trace(go.Bar(x=x_dates, y=daily_df["Session_PnL_$"], name="Daily PnL"), row=1, col=1)
        fig.add_trace(
            go.Scatter(x=x_dates, y=daily_df["Cum_PnL_$"], mode="lines", name="Cumulative PnL"),
            row=2,
            col=1,
        )
    fig.update_layout(template="plotly_white", height=700, title="Open 30m β₂₀ z-score intraday")
    chart = fig.to_html(full_html=False, include_plotlyjs="cdn", div_id="pnl-chart")

    td_dates = (
        {
            str(int(td)): pd.to_datetime(str(int(td)), format="%Y%m%d").strftime("%Y-%m-%d")
            for td in daily_df["Trading date"].unique()
        }
        if not daily_df.empty
        else {}
    )

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Open 30m backtest</title>
<style>
body {{ font-family: sans-serif; margin: 2rem; }}
h1,h2,h3 {{ color: #0066cc; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
th {{ background: #0066cc; color: white; padding: 6px; text-align: left; }}
td {{ border: 1px solid #ddd; padding: 6px; }}
.interactive-table tbody tr {{ cursor: pointer; }}
.interactive-table tbody tr:hover {{ background: #eef6ff; }}
.interactive-table tbody tr.selected {{ background: #cce0ff; font-weight: 600; }}
#explainer {{ background: #f7f9fc; border-left: 4px solid #0066cc; padding: 0.75rem 1rem; margin: 0.5rem 0 1rem; font-size: 13px; line-height: 1.5; }}
</style></head><body>
<h1>Open 30m β₂₀ z-score — {ticker_a} vs {ticker_b}</h1>
<h2>Summary</h2>{summary.to_html(index=False, classes="data-table", border=0)}
<div>{chart}</div>
<h2>Daily PnL</h2>
<p>Click row or trade for leg detail.</p>
{_daily_table_html(daily_df)}
<h2>All trades</h2>
{_trades_table_html(trades_df)}
<h2 id="legs-title">Trade legs</h2>
<div id="explainer">
  spread_ret = r_{ticker_a}(from 9:30 open) − β₂₀×r_{ticker_b}(from 9:30 open).<br>
  z vs prior {p.hist_days} sessions at same minute. Signal 9:40–9:59 if |z|≥{p.entry_z}. Exit 10:00.
</div>
<p id="trade-meta">No trade selected.</p>
<div id="legs-panel"></div>
<script>
const TRADES = {_trades_json(trades_df, ticker_a, ticker_b)};
const TRADES_BY_TD = {_trades_by_td_json(trades_df)};
const TD_LABELS = {json.dumps(td_dates)};

function legTable(title, time, rows, tickerA, tickerB) {{
  let html = '<div><h3>' + title + ' — ' + time + '</h3><table class="data-table"><thead><tr>';
  html += '<th>Leg</th><th>Ticker</th><th>Action</th><th>Price</th><th>Notional $</th><th>Shares</th><th>β₂₀</th></tr></thead><tbody>';
  for (const r of rows) {{
    const ticker = r.Leg === 'A' ? tickerA : tickerB;
    html += '<tr><td>'+r.Leg+'</td><td>'+ticker+'</td><td>'+r.Action+'</td><td>'+r.Price+'</td><td>'+r.Notional_usd+'</td><td>'+r.Shares+'</td><td>'+(r.Beta_20!==undefined?r.Beta_20:'—')+'</td></tr>';
  }}
  return html + '</tbody></table></div>';
}}

function renderTrade(id) {{
  const t = TRADES[String(id)];
  if (!t) return;
  const total = t.PnL_A_usd + t.PnL_B_usd;
  document.getElementById('legs-title').textContent = 'Trade #'+t.Trade_id+' — '+t.Side.toUpperCase();
  document.getElementById('trade-meta').textContent =
    t.Trading_date + ' signal '+t.Signal_time+' z='+t.Signal_z.toFixed(2)
    +' spread_ret='+t.Signal_spread_ret_pct.toFixed(3)+'% | PnL $'+total.toFixed(2);
  let html = '<p>β₂₀='+t.Entry_beta_20+' | hist mean '+(t.Hist_mean_pct??'—')+'% std '+(t.Hist_std_pct??'—')+'%</p>';
  html += legTable('Entry', t.Entry_time, t.legs.entry, t.ticker_a, t.ticker_b);
  html += legTable('Exit 10:00', t.Exit_time_full, t.legs.exit, t.ticker_a, t.ticker_b);
  html += '<table class="data-table"><tr><td>'+t.ticker_a+'</td><td>'+t.PnL_A_usd.toFixed(2)+'</td></tr>';
  html += '<tr><td>'+t.ticker_b+' hedge</td><td>'+t.PnL_B_usd.toFixed(2)+'</td></tr>';
  html += '<tr><td><b>Total</b></td><td><b>'+total.toFixed(2)+'</b></td></tr></table>';
  document.getElementById('legs-panel').innerHTML = html;
}}

function selectTrade(id) {{
  document.querySelectorAll('.trade-row').forEach(r => r.classList.toggle('selected', r.dataset.trade===String(id)));
  document.querySelectorAll('.daily-row').forEach(r => r.classList.remove('selected'));
  renderTrade(id);
}}
function selectDaily(td) {{
  document.querySelectorAll('.daily-row').forEach(r => r.classList.toggle('selected', r.dataset.td===String(td)));
  const ids = TRADES_BY_TD[String(td)] || [];
  if (ids.length) selectTrade(ids[0]);
}}
document.querySelectorAll('.trade-row').forEach(r => r.addEventListener('click', () => selectTrade(r.dataset.trade)));
document.querySelectorAll('.daily-row').forEach(r => r.addEventListener('click', () => selectDaily(r.dataset.td)));
const chartEl = document.getElementById('pnl-chart');
if (chartEl) chartEl.on('plotly_click', ev => {{
  if (!ev.points?.length) return;
  const d = new Date(ev.points[0].x);
  selectDaily(String(d.getFullYear())+String(d.getMonth()+1).padStart(2,'0')+String(d.getDate()).padStart(2,'0'));
}});
if (Object.keys(TRADES).length) selectTrade(Object.keys(TRADES).pop());
</script>
</body></html>"""


def _parse_args() -> argparse.Namespace:
    today = datetime.now() - timedelta(days=3)
    default_end = int(today.strftime("%Y%m%d"))
    default_start = int((today - timedelta(days=180)).strftime("%Y%m%d"))

    p = argparse.ArgumentParser(description="9:30–10:00 β₂₀ open spread z-score backtest.")
    p.add_argument("--ticker-a", default="PAAS")
    p.add_argument("--ticker-b", default="SLV")
    p.add_argument("--start", type=int, default=default_start)
    p.add_argument("--end", type=int, default=default_end)
    p.add_argument("--entry-z", type=float, default=1.5)
    p.add_argument("--hist-days", type=int, default=20)
    p.add_argument("--min-hist-days", type=int, default=10)
    p.add_argument("--beta-lookback", type=int, default=20)
    p.add_argument("--notional", type=float, default=10_000.0)
    p.add_argument("--data-folder", default=None)
    p.add_argument("--output", default="intraday_open30_backtest.html")
    p.add_argument("--no-browser", action="store_true")
    p.add_argument("--export-csv", default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    folder = args.data_folder or Paths().min_folder
    params = Open30Params(
        entry_z=args.entry_z,
        hist_days=args.hist_days,
        min_hist_days=args.min_hist_days,
        beta_lookback=args.beta_lookback,
        notional=args.notional,
    )

    print(f"Loading {args.ticker_a}/{args.ticker_b} ({args.start}–{args.end}) …")
    panel = load_pair_panel(args.ticker_a, args.ticker_b, folder, args.start, args.end)

    minute_df = build_minute_spread_panel(
        panel, args.ticker_a, args.ticker_b, params.beta_lookback
    )
    minute_df = attach_historical_z(minute_df, params.hist_days, params.min_hist_days)
    print(f"  {minute_df['Trading date'].nunique()} sessions, {len(minute_df):,} open-window bars")

    trades_df, daily_df = run_backtest(minute_df, params, args.ticker_a, args.ticker_b)
    summary = summary_stats(trades_df, daily_df, params, args.ticker_a, args.ticker_b)
    print("\n=== Summary ===")
    print(summary.to_string(index=False))

    out = args.output
    with open(out, "w", encoding="utf-8") as f:
        f.write(
            build_report_html(
                summary, trades_df, daily_df, args.ticker_a, args.ticker_b, params
            )
        )
    print(f"\nReport: {out}")

    if args.export_csv:
        os.makedirs(args.export_csv, exist_ok=True)
        trades_df.drop(columns=["_legs"], errors="ignore").to_csv(
            os.path.join(args.export_csv, "trades.csv"), index=False
        )
        daily_df.to_csv(os.path.join(args.export_csv, "daily_pnl.csv"), index=False)

    if not args.no_browser:
        webbrowser.open(f"file://{os.path.abspath(out)}")


if __name__ == "__main__":
    main()
