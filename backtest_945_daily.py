#!/usr/bin/env python3
"""
Daily 9:45 spread-level mean reversion — enter same 9:45, exit next 9:45.

Workflow (default):
  1. Each day at **9:45 ET**, spread level:
     S = P_A − β_D×P_B  where **β_D is from the prior trading session** (15m OLS).
  2. Rolling mean/std of the last ``window`` days' **9:45 spread levels** (excludes today).
  3. z = (S_today − roll_mean) / roll_std at 9:45.
  4. **Mon 9:45**, z = −1.8 → **long at Mon 9:45**, **exit Tue 9:45**.

Usage::

    python backtest_945_daily.py --start 20240101 --end 20251231 --notional 10000
"""

from __future__ import annotations

import argparse
import json
import os
import webbrowser
from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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


def _leg_breakdown(
    side: str,
    notional: float,
    beta_entry: float,
    price_a_entry: float,
    price_b_entry: float,
    price_a_exit: float,
    price_b_exit: float,
) -> dict:
    """Dollar PnL per leg for β_D-hedged spread book sized at entry."""
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
                "Notional_$": round(n_a, 2),
                "Shares": round(sh_a, 4),
            },
            {
                "Leg": "B",
                "Action": entry_b_action,
                "Price": round(price_b_entry, 4),
                "Beta_D": round(beta_entry, 4),
                "Notional_$": round(n_b, 2),
                "Shares": round(sh_b, 4),
            },
        ],
        "exit": [
            {
                "Leg": "A",
                "Action": exit_a_action,
                "Price": round(price_a_exit, 4),
                "Notional_$": round(sh_a * price_a_exit, 2),
                "Shares": round(sh_a, 4),
            },
            {
                "Leg": "B",
                "Action": exit_b_action,
                "Price": round(price_b_exit, 4),
                "Notional_$": round(sh_b * price_b_exit, 2),
                "Shares": round(sh_b, 4),
            },
        ],
        "PnL_A_$": round(pnl_a, 2),
        "PnL_B_$": round(pnl_b, 2),
        "PnL_total_$": round(pnl_a + pnl_b, 2),
    }


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
        for _, day in rth.groupby("Trading date"):
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
            ticker_a: "price_a",
            ticker_b: "price_b",
        }
    )
    out = out.dropna(subset=["spread_level", "beta_D", "price_a", "price_b"]).sort_values(
        "Trading date"
    )
    out["Trading date"] = out["Trading date"].astype(int)
    return out.reset_index(drop=True)


def attach_signal_z(daily: pd.DataFrame, window: int) -> pd.DataFrame:
    out = daily.copy()
    hist_mean = out["spread_level"].shift(1).rolling(window=window, min_periods=window).mean()
    hist_std = out["spread_level"].shift(1).rolling(window=window, min_periods=window).std()
    out["roll_mean"] = hist_mean
    out["roll_std"] = hist_std
    out["z"] = (out["spread_level"] - hist_mean) / hist_std
    return out


def run_backtest(
    daily: pd.DataFrame,
    p: Daily945Params,
    ticker_a: str,
    ticker_b: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    trades: list[dict] = []
    trade_id = 0

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

        pa_e = float(entry_row["price_a"])
        pb_e = float(entry_row["price_b"])
        pa_x = float(exit_row["price_a"])
        pb_x = float(exit_row["price_b"])
        beta_e = float(entry_row["beta_D"])
        beta_x = float(exit_row["beta_D"])
        legs = _leg_breakdown(side, p.notional, beta_e, pa_e, pb_e, pa_x, pb_x)

        trade_id += 1
        trades.append(
            {
                "Trade_id": trade_id,
                "Signal_date": int(row["Trading date"]),
                "Signal_z": round(float(z), 3),
                "Roll_mean_$": round(float(row["roll_mean"]), 4)
                if pd.notna(row["roll_mean"])
                else None,
                "Signal_spread_$": round(float(row["spread_level"]), 4),
                "Side": side,
                "Entry_date": int(entry_row["Trading date"]),
                "Entry_time": pd.Timestamp(entry_row["DT"]).strftime("%Y-%m-%d %H:%M"),
                "Entry_spread_$": round(entry_level, 4),
                "Entry_beta_D": round(beta_e, 4),
                f"Entry_{ticker_a}": round(pa_e, 4),
                f"Entry_{ticker_b}": round(pb_e, 4),
                "Exit_date": int(exit_row["Trading date"]),
                "Exit_time": pd.Timestamp(exit_row["DT"]).strftime("%Y-%m-%d %H:%M"),
                "Exit_spread_$": round(exit_level, 4),
                "Exit_beta_D": round(beta_x, 4),
                f"Exit_{ticker_a}": round(pa_x, 4),
                f"Exit_{ticker_b}": round(pb_x, 4),
                "Spread_move_$": round(exit_level - entry_level, 4),
                "Return_pct": round(pnl_dec * 100, 4),
                "PnL_$": round(pnl_dec * p.notional, 2),
                "PnL_legs_$": legs["PnL_total_$"],
                "PnL_A_$": legs["PnL_A_$"],
                "PnL_B_$": legs["PnL_B_$"],
                "_legs": legs,
            }
        )

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        return trades_df, pd.DataFrame()

    daily_df = (
        trades_df.groupby("Exit_date", as_index=False)
        .agg(trades=("Trade_id", "count"), session_pnl=("PnL_$", "sum"))
        .rename(
            columns={
                "Exit_date": "Trading date",
                "trades": "Trades",
                "session_pnl": "Session_PnL_$",
            }
        )
        .sort_values("Trading date")
    )
    daily_df["Cum_PnL_$"] = daily_df["Session_PnL_$"].cumsum()
    return trades_df, daily_df


def summary_stats(
    trades_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    p: Daily945Params,
    ticker_a: str,
    ticker_b: str,
) -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame({"Metric": ["Trades"], "Value": [0]})

    total = trades_df["PnL_$"].sum()
    wins = (trades_df["PnL_$"] > 0).sum()
    n = len(trades_df)

    rows = [
        ("Trades", n),
        ("Long", int((trades_df["Side"] == "long").sum())),
        ("Short", int((trades_df["Side"] == "short").sum())),
        ("Total PnL ($)", round(total, 2)),
        ("Avg return per trade (%)", round(trades_df["Return_pct"].mean(), 3)),
        ("Avg PnL per trade ($)", round(trades_df["PnL_$"].mean(), 2)),
        ("Win rate (%)", round(100 * wins / n, 1)),
        ("Best trade ($)", round(trades_df["PnL_$"].max(), 2)),
        ("Worst trade ($)", round(trades_df["PnL_$"].min(), 2)),
        ("Spread", f"{ticker_a} − β_D×{ticker_b} ($), β_D from prior session"),
        ("Signal / entry", "9:45 ET same day"),
        ("Exit", f"9:45 ET +{p.hold_days} session(s)"),
        ("Rolling window (sessions)", p.window),
        ("Entry z-band", f"±{p.entry_std} σ"),
        ("Notional on leg A", p.notional),
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


def _trades_table_html(trades_df: pd.DataFrame, ticker_a: str, ticker_b: str) -> str:
    if trades_df.empty:
        return "<p>No trades</p>"
    cols = [
        "Trade_id",
        "Signal_date",
        "Side",
        "Signal_z",
        "Entry_date",
        "Exit_date",
        "Entry_spread_$",
        "Exit_spread_$",
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
            if c in ("Entry_spread_$", "Exit_spread_$", "PnL_$", "Return_pct") and pd.notna(val):
                text = f"{float(val):,.4f}" if c != "PnL_$" else f"{float(val):,.2f}"
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
        legs = row["_legs"]
        tid = str(int(row["Trade_id"]))
        out[tid] = {
            "Trade_id": int(row["Trade_id"]),
            "Side": row["Side"],
            "Signal_date": int(row["Signal_date"]),
            "Signal_z": float(row["Signal_z"]),
            "Roll_mean_$": row["Roll_mean_$"],
            "Signal_spread_$": float(row["Signal_spread_$"]),
            "Entry_date": int(row["Entry_date"]),
            "Entry_time": row["Entry_time"],
            "Exit_date": int(row["Exit_date"]),
            "Exit_time": row["Exit_time"],
            "Entry_spread_$": float(row["Entry_spread_$"]),
            "Exit_spread_$": float(row["Exit_spread_$"]),
            "Entry_beta_D": float(row["Entry_beta_D"]),
            "Exit_beta_D": float(row["Exit_beta_D"]),
            "Entry_A": float(row[f"Entry_{ticker_a}"]),
            "Entry_B": float(row[f"Entry_{ticker_b}"]),
            "Exit_A": float(row[f"Exit_{ticker_a}"]),
            "Exit_B": float(row[f"Exit_{ticker_b}"]),
            "Spread_move_$": float(row["Spread_move_$"]),
            "Return_pct": float(row["Return_pct"]),
            "PnL_$": float(row["PnL_$"]),
            "PnL_A_$": float(row["PnL_A_$"]),
            "PnL_B_$": float(row["PnL_B_$"]),
            "legs": legs,
            "ticker_a": ticker_a,
            "ticker_b": ticker_b,
        }
    return json.dumps(out)


def _trades_by_exit_json(trades_df: pd.DataFrame) -> str:
    if trades_df.empty:
        return "{}"
    grouped: dict[str, list[int]] = {}
    for _, row in trades_df.iterrows():
        td = str(int(row["Exit_date"]))
        grouped.setdefault(td, []).append(int(row["Trade_id"]))
    return json.dumps(grouped)


def build_report_html(
    summary: pd.DataFrame,
    trades_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    p: Daily945Params,
) -> str:
    fig = make_subplots(rows=2, cols=1, row_heights=[0.45, 0.55], vertical_spacing=0.08)
    if not daily_df.empty:
        x_dates = _trading_dates_for_plot(daily_df)
        fig.add_trace(
            go.Bar(x=x_dates, y=daily_df["Session_PnL_$"], name="Daily PnL"),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x_dates,
                y=daily_df["Cum_PnL_$"],
                mode="lines",
                name="Cumulative PnL",
            ),
            row=2,
            col=1,
        )
    fig.update_layout(
        template="plotly_white",
        height=700,
        title="9:45 rolling-mean spread level — daily & cumulative PnL",
    )
    fig.update_xaxes(title_text="Exit date", row=1, col=1)
    fig.update_xaxes(title_text="Exit date", row=2, col=1)
    fig.update_yaxes(title_text="Daily PnL ($)", row=1, col=1)
    fig.update_yaxes(title_text="Cumulative PnL ($)", row=2, col=1)
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
<html><head><meta charset="utf-8"><title>9:45 spread backtest</title>
<style>
body {{ font-family: sans-serif; margin: 2rem; }}
h1,h2,h3 {{ color: #0066cc; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
th {{ background: #0066cc; color: white; padding: 6px; text-align: left; }}
td {{ border: 1px solid #ddd; padding: 6px; }}
.interactive-table tbody tr.daily-row,
.interactive-table tbody tr.trade-row {{ cursor: pointer; }}
.interactive-table tbody tr:hover {{ background: #eef6ff; }}
.interactive-table tbody tr.selected {{ background: #cce0ff; font-weight: 600; }}
#legs-panel {{ margin-top: 1rem; }}
#legs-hint {{ color: #666; font-style: italic; }}
#trade-meta {{ margin: 0.5rem 0 1rem; color: #333; }}
#explainer {{ background: #f7f9fc; border-left: 4px solid #0066cc; padding: 0.75rem 1rem; margin: 0.5rem 0 1rem; font-size: 13px; line-height: 1.5; }}
.leg-section {{ margin-bottom: 1rem; }}
</style></head><body>
<h1>9:45 rolling-mean spread — {ticker_a} vs {ticker_b}</h1>
<h2>Summary</h2>{summary.to_html(index=False, classes="data-table", border=0)}
<div>{chart}</div>
<h2>Daily PnL (by exit date)</h2>
<p id="legs-hint">Click a daily row, chart bar, or trade row to see entry/exit legs.</p>
{_daily_table_html(daily_df)}
<h2>All trades</h2>
{_trades_table_html(trades_df, ticker_a, ticker_b)}
<h2 id="legs-title">Trade legs</h2>
<div id="explainer">
  <strong>Spread level</strong> S = P_{ticker_a} − β_D×P_{ticker_b} ($). β_D from <b>prior session</b>.<br>
  <strong>Long spread</strong>: BUY {ticker_a}, SELL β_D×{ticker_b} at entry — reverse at exit.<br>
  <strong>Short spread</strong>: SELL {ticker_a}, BUY β_D×{ticker_b} at entry — reverse at exit.<br>
  Signal at 9:45 (z vs {p.window}-day rolling mean). Hold {p.hold_days} session(s) to next 9:45.
</div>
<p id="trade-meta">No trade selected.</p>
<div id="legs-panel"><p>Select a trade above.</p></div>
<script>
const TRADES = {_trades_json(trades_df, ticker_a, ticker_b)};
const TRADES_BY_EXIT = {_trades_by_exit_json(trades_df)};
const TD_LABELS = {json.dumps(td_dates)};

function legTable(title, time, rows, tickerA, tickerB) {{
  let html = '<div class="leg-section"><h3>' + title + ' — ' + time + '</h3>';
  html += '<table class="data-table"><thead><tr>';
  html += '<th>Leg</th><th>Ticker</th><th>Action</th><th>Price</th><th>Notional $</th><th>Shares</th><th>β_D</th>';
  html += '</tr></thead><tbody>';
  for (const r of rows) {{
    const ticker = r.Leg === 'A' ? tickerA : tickerB;
    html += '<tr><td>' + r.Leg + '</td><td>' + ticker + '</td><td>' + r.Action + '</td>';
    html += '<td>' + r.Price + '</td><td>' + r.Notional_$ + '</td><td>' + r.Shares + '</td>';
    html += '<td>' + (r.Beta_D !== undefined ? r.Beta_D : '—') + '</td></tr>';
  }}
  html += '</tbody></table></div>';
  return html;
}}

function renderTrade(tradeId) {{
  const t = TRADES[String(tradeId)];
  const title = document.getElementById('legs-title');
  const meta = document.getElementById('trade-meta');
  const panel = document.getElementById('legs-panel');
  if (!t) {{
    title.textContent = 'Trade legs';
    meta.textContent = 'Trade not found.';
    panel.innerHTML = '';
    return;
  }}

  title.textContent = 'Trade #' + t.Trade_id + ' — ' + t.Side.toUpperCase() + ' spread';
  meta.textContent =
    'Signal ' + t.Signal_date + ' (z=' + t.Signal_z + ') | '
    + 'Entry ' + t.Entry_date + ' ' + t.Entry_time + ' | '
    + 'Exit ' + t.Exit_date + ' ' + t.Exit_time + ' | '
    + 'PnL $' + t.PnL_$.toFixed(2) + ' (' + t.Return_pct.toFixed(3) + '%)';

  let html = '<p><b>Spread level:</b> $' + t.Entry_spread_$.toFixed(4)
    + ' → $' + t.Exit_spread_$.toFixed(4)
    + ' (Δ $' + t.Spread_move_$.toFixed(4) + ')</p>';
  html += '<p><b>β_D:</b> entry ' + t.Entry_beta_D + ', exit ' + t.Exit_beta_D + '</p>';
  html += legTable('Entry legs', t.Entry_time, t.legs.entry, t.ticker_a, t.ticker_b);
  html += legTable('Exit legs', t.Exit_time, t.legs.exit, t.ticker_a, t.ticker_b);
  html += '<table class="data-table"><thead><tr><th>Leg PnL</th><th>$</th></tr></thead><tbody>';
  html += '<tr><td>' + t.ticker_a + '</td><td>' + t.PnL_A_$.toFixed(2) + '</td></tr>';
  html += '<tr><td>' + t.ticker_b + ' (hedge)</td><td>' + t.PnL_B_$.toFixed(2) + '</td></tr>';
  html += '<tr><td><b>Total</b></td><td><b>' + t.PnL_$.toFixed(2) + '</b></td></tr>';
  html += '</tbody></table>';
  panel.innerHTML = html;
}}

function selectTrade(tradeId) {{
  document.querySelectorAll('.trade-row').forEach(tr => {{
    tr.classList.toggle('selected', tr.dataset.trade === String(tradeId));
  }});
  document.querySelectorAll('.daily-row').forEach(tr => tr.classList.remove('selected'));
  renderTrade(tradeId);
  document.getElementById('legs-panel').scrollIntoView({{ behavior: 'smooth', block: 'nearest' }});
}}

function selectDailyRow(td) {{
  document.querySelectorAll('.daily-row').forEach(tr => {{
    tr.classList.toggle('selected', tr.dataset.td === String(td));
  }});
  const ids = TRADES_BY_EXIT[String(td)] || [];
  if (ids.length === 1) {{
    selectTrade(ids[0]);
  }} else if (ids.length > 1) {{
    document.querySelectorAll('.trade-row').forEach(tr => tr.classList.remove('selected'));
    document.getElementById('legs-title').textContent = 'Trades exiting ' + (TD_LABELS[String(td)] || td);
    document.getElementById('trade-meta').textContent = ids.length + ' trades on this exit date — click a trade row for legs.';
    let html = '<ul>';
    ids.forEach(id => {{
      const t = TRADES[String(id)];
      html += '<li><a href="#" data-trade="' + id + '">#' + id + ' ' + t.Side + ' PnL $' + t.PnL_$.toFixed(2) + '</a></li>';
    }});
    html += '</ul>';
    document.getElementById('legs-panel').innerHTML = html;
    document.querySelectorAll('#legs-panel a[data-trade]').forEach(a => {{
      a.addEventListener('click', e => {{
        e.preventDefault();
        selectTrade(a.dataset.trade);
      }});
    }});
  }} else {{
    document.getElementById('legs-panel').innerHTML = '<p>No trades for this date.</p>';
  }}
}}

document.querySelectorAll('.trade-row').forEach(tr => {{
  tr.addEventListener('click', () => selectTrade(tr.dataset.trade));
  tr.addEventListener('keydown', e => {{
    if (e.key === 'Enter' || e.key === ' ') {{ e.preventDefault(); selectTrade(tr.dataset.trade); }}
  }});
}});

document.querySelectorAll('.daily-row').forEach(tr => {{
  tr.addEventListener('click', () => selectDailyRow(tr.dataset.td));
  tr.addEventListener('keydown', e => {{
    if (e.key === 'Enter' || e.key === ' ') {{ e.preventDefault(); selectDailyRow(tr.dataset.td); }}
  }});
}});

const chartEl = document.getElementById('pnl-chart');
if (chartEl) {{
  chartEl.on('plotly_click', ev => {{
    if (!ev.points || !ev.points.length) return;
    const x = ev.points[0].x;
    if (!x) return;
    const d = new Date(x);
    const td = String(d.getFullYear())
      + String(d.getMonth() + 1).padStart(2, '0')
      + String(d.getDate()).padStart(2, '0');
    selectDailyRow(td);
  }});
}}

if (Object.keys(TRADES).length) {{
  const lastId = Object.keys(TRADES).sort((a,b) => Number(a)-Number(b)).pop();
  selectTrade(lastId);
}}
</script>
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
    p.add_argument("--window", type=int, default=20)
    p.add_argument("--entry-std", type=float, default=1.5)
    p.add_argument("--entry-lag", type=int, default=0)
    p.add_argument("--hold-days", type=int, default=1)
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

    trades_df, daily_df = run_backtest(daily, params, args.ticker_a, args.ticker_b)
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
        export_trades = trades_df.drop(columns=["_legs"], errors="ignore")
        export_trades.to_csv(os.path.join(args.export_csv, "trades.csv"), index=False)
        daily_df.to_csv(os.path.join(args.export_csv, "daily_pnl.csv"), index=False)
        daily.to_csv(os.path.join(args.export_csv, "snapshots_945.csv"), index=False)

    if not args.no_browser:
        webbrowser.open(f"file://{os.path.abspath(out)}")


if __name__ == "__main__":
    main()
