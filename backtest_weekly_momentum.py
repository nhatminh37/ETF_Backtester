#!/usr/bin/env python3
"""
Weekly momentum on β₁₀-hedged **daily** return spread.

Signal (prior ISO calendar week):
  avg_daily_diff = mean( r_PAAS − β₁₀×r_SLV ) over last week's trading days
  β₁₀ = OLS on prior 10 session daily returns (known pre-open each day).

Position (this week):
  avg > +threshold  → long spread all week (BUY PAAS, SELL β₁₀×SLV)
  avg < −threshold  → short spread (SELL PAAS, BUY β₁₀×SLV)
  else flat

Execution:
  Enter first trading day of week at 9:45 ET
  Exit last trading day of week at last 1-min RTH bar (~15:59)
  $10k notional on PAAS leg; hedge sized β₁₀ at entry.

Usage::

    python backtest_weekly_momentum.py --start 20230101 --end 20251231 --notional 10000
    python backtest_weekly_momentum.py --threshold 1.0 --beta-lookback 10
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
from silver_backtesting.spread import lookback_beta_by_trading_date, pair_daily_returns

ANCHOR_HOUR = 9
ANCHOR_MINUTE = 45


@dataclass
class WeeklyMomentumParams:
    threshold_pct: float = 1.0
    beta_lookback: int = 10
    min_signal_days: int = 3
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
                "Beta_10": round(beta_entry, 4),
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


def build_daily_hedged_panel(
    panel: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    beta_lookback: int,
) -> pd.DataFrame:
    """One row per session: daily returns, β₁₀, hedged daily diff_ret."""
    daily_ret = pair_daily_returns(panel, ticker_a, ticker_b)
    if daily_ret is None or daily_ret.empty:
        return pd.DataFrame()

    beta10 = lookback_beta_by_trading_date(
        panel, ticker_a, ticker_b, lookback_days=beta_lookback, lag_sessions=1
    )

    daily = daily_ret.copy()
    daily["Trading date"] = daily.index.astype(int)
    daily = daily.reset_index(drop=True)
    daily["ret_a"] = daily[ticker_a]
    daily["ret_b"] = daily[ticker_b]
    daily["beta_10"] = daily["Trading date"].map(beta10)
    daily["diff_ret"] = daily["ret_a"] - daily["beta_10"] * daily["ret_b"]

    dt = pd.to_datetime(daily["Trading date"].astype(str), format="%Y%m%d")
    iso = dt.dt.isocalendar()
    daily["iso_year"] = iso.year.astype(int)
    daily["iso_week"] = iso.week.astype(int)
    daily["week_key"] = (
        daily["iso_year"].astype(str) + "-W" + daily["iso_week"].astype(str).str.zfill(2)
    )
    return daily.dropna(subset=["diff_ret", "beta_10"]).sort_values("Trading date")


def _prices_entry_exit(
    panel: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    entry_td: int,
    exit_td: int,
) -> dict | None:
    """Entry 9:45 on entry_td; exit last RTH bar on exit_td."""
    out: dict = {}
    for label, td, use_eod in (
        ("entry", entry_td, False),
        ("exit", exit_td, True),
    ):
        day = panel.loc[panel["Trading date"] == td].sort_values("DT")
        if day.empty:
            return None
        if use_eod:
            row = day.iloc[-1]
        else:
            mask = (day["DT"].dt.hour == ANCHOR_HOUR) & (day["DT"].dt.minute == ANCHOR_MINUTE)
            row = day.loc[mask].iloc[0] if mask.any() else day.iloc[0]
        out[f"{label}_dt"] = row["DT"]
        out[f"{label}_pa"] = float(row[ticker_a])
        out[f"{label}_pb"] = float(row[ticker_b])
    return out


def build_week_signals(daily: pd.DataFrame, p: WeeklyMomentumParams) -> pd.DataFrame:
    """Map each position week to signal from prior ISO week."""
    thresh = p.threshold_pct / 100.0
    week_stats = (
        daily.groupby("week_key", as_index=False)
        .agg(
            n_days=("Trading date", "count"),
            avg_diff_ret=("diff_ret", "mean"),
            week_start=("Trading date", "min"),
            week_end=("Trading date", "max"),
        )
        .sort_values("week_start")
    )

    rows: list[dict] = []
    for i in range(1, len(week_stats)):
        signal = week_stats.iloc[i - 1]
        position = week_stats.iloc[i]
        if int(signal["n_days"]) < p.min_signal_days:
            continue

        avg = float(signal["avg_diff_ret"])
        if avg > thresh:
            side = "long"
        elif avg < -thresh:
            side = "short"
        else:
            side = "flat"

        rows.append(
            {
                "Position_week": position["week_key"],
                "Signal_week": signal["week_key"],
                "Signal_days": int(signal["n_days"]),
                "Signal_avg_diff_pct": round(avg * 100, 4),
                "Side": side,
                "Entry_date": int(position["week_start"]),
                "Exit_date": int(position["week_end"]),
            }
        )
    return pd.DataFrame(rows)


def run_backtest(
    panel: pd.DataFrame,
    daily: pd.DataFrame,
    signals: pd.DataFrame,
    p: WeeklyMomentumParams,
    ticker_a: str,
    ticker_b: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    beta_by_td = daily.set_index("Trading date")["beta_10"]
    trades: list[dict] = []
    trade_id = 0

    for _, sig in signals.iterrows():
        if sig["Side"] == "flat":
            continue

        entry_td = int(sig["Entry_date"])
        exit_td = int(sig["Exit_date"])
        px = _prices_entry_exit(panel, ticker_a, ticker_b, entry_td, exit_td)
        if px is None:
            continue

        beta_e = float(beta_by_td.get(entry_td, float("nan")))
        if pd.isna(beta_e):
            continue

        side = sig["Side"]
        legs = _leg_breakdown(
            side,
            p.notional,
            beta_e,
            px["entry_pa"],
            px["entry_pb"],
            px["exit_pa"],
            px["exit_pb"],
        )
        pnl_usd = legs["PnL_total_usd"]
        return_pct = 100.0 * pnl_usd / p.notional if p.notional else 0.0

        trade_id += 1
        trades.append(
            {
                "Trade_id": trade_id,
                "Position_week": sig["Position_week"],
                "Signal_week": sig["Signal_week"],
                "Signal_avg_diff_pct": sig["Signal_avg_diff_pct"],
                "Side": side,
                "Entry_date": entry_td,
                "Entry_time": pd.Timestamp(px["entry_dt"]).strftime("%Y-%m-%d %H:%M"),
                "Exit_date": exit_td,
                "Exit_time": pd.Timestamp(px["exit_dt"]).strftime("%Y-%m-%d %H:%M"),
                "Entry_beta_10": round(beta_e, 4),
                f"Entry_{ticker_a}": round(px["entry_pa"], 4),
                f"Entry_{ticker_b}": round(px["entry_pb"], 4),
                f"Exit_{ticker_a}": round(px["exit_pa"], 4),
                f"Exit_{ticker_b}": round(px["exit_pb"], 4),
                "Return_pct": round(return_pct, 4),
                "PnL_$": round(pnl_usd, 2),
                "PnL_A_$": legs["PnL_A_usd"],
                "PnL_B_$": legs["PnL_B_usd"],
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
    signals: pd.DataFrame,
    p: WeeklyMomentumParams,
    ticker_a: str,
    ticker_b: str,
) -> pd.DataFrame:
    n_weeks = len(signals)
    n_flat = int((signals["Side"] == "flat").sum()) if not signals.empty else 0
    if trades_df.empty:
        return pd.DataFrame(
            {
                "Metric": ["Weeks", "Trades"],
                "Value": [n_weeks, 0],
            }
        )

    total = trades_df["PnL_$"].sum()
    wins = (trades_df["PnL_$"] > 0).sum()
    n = len(trades_df)

    rows = [
        ("ISO weeks evaluated", n_weeks),
        ("Flat weeks (|signal| ≤ threshold)", n_flat),
        ("Trades", n),
        ("Long weeks", int((trades_df["Side"] == "long").sum())),
        ("Short weeks", int((trades_df["Side"] == "short").sum())),
        ("Total PnL ($)", round(total, 2)),
        ("Avg PnL per trade ($)", round(trades_df["PnL_$"].mean(), 2)),
        ("Win rate (%)", round(100 * wins / n, 1)),
        ("Signal", f"Prior week avg(r_A − β₁₀×r_B)"),
        ("Threshold (%)", f"±{p.threshold_pct}"),
        ("β lookback (sessions)", p.beta_lookback),
        ("Entry", "Mon (or 1st day) 9:45"),
        ("Exit", "Fri (or last day) EOD"),
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
        "Position_week",
        "Signal_week",
        "Side",
        "Signal_avg_diff_pct",
        "Entry_date",
        "Exit_date",
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
            if c == "PnL_$" and pd.notna(val):
                text = f"{float(val):,.2f}"
            elif c == "Signal_avg_diff_pct" and pd.notna(val):
                text = f"{float(val):,.4f}"
            elif c == "Return_pct" and pd.notna(val):
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
            "Position_week": row["Position_week"],
            "Signal_week": row["Signal_week"],
            "Signal_avg_diff_pct": float(row["Signal_avg_diff_pct"]),
            "Side": row["Side"],
            "Entry_date": int(row["Entry_date"]),
            "Entry_time": row["Entry_time"],
            "Exit_date": int(row["Exit_date"]),
            "Exit_time": row["Exit_time"],
            "Entry_beta_10": float(row["Entry_beta_10"]),
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
    p: WeeklyMomentumParams,
) -> str:
    fig = make_subplots(rows=2, cols=1, row_heights=[0.45, 0.55], vertical_spacing=0.08)
    if not daily_df.empty:
        x_dates = _trading_dates_for_plot(daily_df)
        fig.add_trace(
            go.Bar(x=x_dates, y=daily_df["Session_PnL_$"], name="Weekly exit PnL"),
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
        title="Weekly β₁₀ momentum — cumulative PnL",
    )
    fig.update_xaxes(title_text="Exit date (week end)", row=1, col=1)
    fig.update_xaxes(title_text="Exit date", row=2, col=1)
    fig.update_yaxes(title_text="PnL ($)", row=1, col=1)
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
<html><head><meta charset="utf-8"><title>Weekly momentum backtest</title>
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
#explainer {{ background: #f7f9fc; border-left: 4px solid #0066cc; padding: 0.75rem 1rem; margin: 0.5rem 0 1rem; font-size: 13px; line-height: 1.5; }}
</style></head><body>
<h1>Weekly β₁₀ momentum — {ticker_a} vs {ticker_b}</h1>
<h2>Summary</h2>{summary.to_html(index=False, classes="data-table", border=0)}
<div>{chart}</div>
<h2>PnL by week-end date</h2>
<p>Click a row, chart bar, or trade for leg detail.</p>
{_daily_table_html(daily_df)}
<h2>All weekly trades</h2>
{_trades_table_html(trades_df)}
<h2 id="legs-title">Trade legs</h2>
<div id="explainer">
  <strong>Signal:</strong> prior ISO week avg daily (r_{ticker_a} − β₁₀×r_{ticker_b}).<br>
  <strong>Long week</strong> if avg &gt; +{p.threshold_pct}%: BUY {ticker_a}, SELL β₁₀×{ticker_b}.<br>
  <strong>Short week</strong> if avg &lt; −{p.threshold_pct}%: opposite.<br>
  Enter 9:45 first day of week; exit EOD last day. PnL = sum of leg fills.
</div>
<p id="trade-meta">No trade selected.</p>
<div id="legs-panel"><p>Select a trade.</p></div>
<script>
const TRADES = {_trades_json(trades_df, ticker_a, ticker_b)};
const TRADES_BY_EXIT = {_trades_by_exit_json(trades_df)};
const TD_LABELS = {json.dumps(td_dates)};

function legTable(title, time, rows, tickerA, tickerB) {{
  let html = '<div class="leg-section"><h3>' + title + ' — ' + time + '</h3>';
  html += '<table class="data-table"><thead><tr>';
  html += '<th>Leg</th><th>Ticker</th><th>Action</th><th>Price</th><th>Notional $</th><th>Shares</th><th>β₁₀</th>';
  html += '</tr></thead><tbody>';
  for (const r of rows) {{
    const ticker = r.Leg === 'A' ? tickerA : tickerB;
    html += '<tr><td>' + r.Leg + '</td><td>' + ticker + '</td><td>' + r.Action + '</td>';
    html += '<td>' + r.Price + '</td><td>' + r.Notional_usd + '</td><td>' + r.Shares + '</td>';
    html += '<td>' + (r.Beta_10 !== undefined ? r.Beta_10 : '—') + '</td></tr>';
  }}
  html += '</tbody></table></div>';
  return html;
}}

function renderTrade(tradeId) {{
  const t = TRADES[String(tradeId)];
  const meta = document.getElementById('trade-meta');
  const panel = document.getElementById('legs-panel');
  if (!t) {{ meta.textContent = 'Not found'; panel.innerHTML = ''; return; }}

  document.getElementById('legs-title').textContent =
    'Week ' + t.Position_week + ' — ' + t.Side.toUpperCase() + ' (signal ' + t.Signal_week + ')';
  const pnlTotal = t.PnL_A_usd + t.PnL_B_usd;
  meta.textContent =
    'Prior week avg diff_ret: ' + t.Signal_avg_diff_pct.toFixed(3) + '% | '
    + 'Entry ' + t.Entry_date + ' → Exit ' + t.Exit_date + ' | PnL $' + pnlTotal.toFixed(2);

  let html = '<p><b>β₁₀ at entry:</b> ' + t.Entry_beta_10 + '</p>';
  html += legTable('Entry legs', t.Entry_time, t.legs.entry, t.ticker_a, t.ticker_b);
  html += legTable('Exit legs', t.Exit_time, t.legs.exit, t.ticker_a, t.ticker_b);
  html += '<table class="data-table"><thead><tr><th>Leg PnL</th><th>$</th></tr></thead><tbody>';
  html += '<tr><td>' + t.ticker_a + '</td><td>' + t.PnL_A_usd.toFixed(2) + '</td></tr>';
  html += '<tr><td>' + t.ticker_b + ' (hedge)</td><td>' + t.PnL_B_usd.toFixed(2) + '</td></tr>';
  html += '<tr><td><b>Total</b></td><td><b>' + pnlTotal.toFixed(2) + '</b></td></tr>';
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
  if (ids.length === 1) selectTrade(ids[0]);
  else if (ids.length > 1) {{
    let html = '<ul>';
    ids.forEach(id => {{
      const t = TRADES[String(id)];
      html += '<li><a href="#" data-trade="' + id + '">#' + id + ' ' + t.Side + ' $' + (t.PnL_A_usd+t.PnL_B_usd).toFixed(2) + '</a></li>';
    }});
    html += '</ul>';
    document.getElementById('legs-panel').innerHTML = html;
    document.querySelectorAll('#legs-panel a[data-trade]').forEach(a => {{
      a.addEventListener('click', e => {{ e.preventDefault(); selectTrade(a.dataset.trade); }});
    }});
  }}
}}

document.querySelectorAll('.trade-row').forEach(tr => {{
  tr.addEventListener('click', () => selectTrade(tr.dataset.trade));
}});
document.querySelectorAll('.daily-row').forEach(tr => {{
  tr.addEventListener('click', () => selectDailyRow(tr.dataset.td));
}});
const chartEl = document.getElementById('pnl-chart');
if (chartEl) {{
  chartEl.on('plotly_click', ev => {{
    if (!ev.points || !ev.points.length) return;
    const d = new Date(ev.points[0].x);
    const td = String(d.getFullYear()) + String(d.getMonth()+1).padStart(2,'0') + String(d.getDate()).padStart(2,'0');
    selectDailyRow(td);
  }});
}}
if (Object.keys(TRADES).length) selectTrade(Object.keys(TRADES).sort((a,b)=>Number(a)-Number(b)).pop());
</script>
</body></html>"""


def _parse_args() -> argparse.Namespace:
    today = datetime.now() - timedelta(days=3)
    default_end = int(today.strftime("%Y%m%d"))
    default_start = int((today - timedelta(days=730)).strftime("%Y%m%d"))

    p = argparse.ArgumentParser(description="Weekly β₁₀ momentum spread backtest.")
    p.add_argument("--ticker-a", default="PAAS")
    p.add_argument("--ticker-b", default="SLV")
    p.add_argument("--start", type=int, default=default_start)
    p.add_argument("--end", type=int, default=default_end)
    p.add_argument("--threshold", type=float, default=1.0, help="Signal threshold in %% (default 1)")
    p.add_argument("--beta-lookback", type=int, default=10)
    p.add_argument("--min-signal-days", type=int, default=3)
    p.add_argument("--notional", type=float, default=10_000.0)
    p.add_argument("--data-folder", default=None)
    p.add_argument("--output", default="weekly_momentum_backtest.html")
    p.add_argument("--no-browser", action="store_true")
    p.add_argument("--export-csv", default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    folder = args.data_folder or Paths().min_folder
    params = WeeklyMomentumParams(
        threshold_pct=args.threshold,
        beta_lookback=args.beta_lookback,
        min_signal_days=args.min_signal_days,
        notional=args.notional,
    )

    print(f"Loading {args.ticker_a}/{args.ticker_b} ({args.start}–{args.end}) …")
    panel = load_pair_panel(args.ticker_a, args.ticker_b, folder, args.start, args.end)

    daily = build_daily_hedged_panel(
        panel, args.ticker_a, args.ticker_b, params.beta_lookback
    )
    print(f"  {len(daily)} sessions with daily hedged returns")

    signals = build_week_signals(daily, params)
    trades_df, daily_df = run_backtest(
        panel, daily, signals, params, args.ticker_a, args.ticker_b
    )
    summary = summary_stats(trades_df, signals, params, args.ticker_a, args.ticker_b)
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
        export = trades_df.drop(columns=["_legs"], errors="ignore")
        export.to_csv(os.path.join(args.export_csv, "trades.csv"), index=False)
        signals.to_csv(os.path.join(args.export_csv, "week_signals.csv"), index=False)
        daily_df.to_csv(os.path.join(args.export_csv, "daily_pnl.csv"), index=False)

    if not args.no_browser:
        webbrowser.open(f"file://{os.path.abspath(out)}")


if __name__ == "__main__":
    main()
