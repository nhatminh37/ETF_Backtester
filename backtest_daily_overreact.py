#!/usr/bin/env python3
"""
Daily overreaction mean-reversion — directional PAAS only (no SLV hedge).

Signal (known before today's open on session *t*):
  1. β_signal = OLS on daily returns from sessions *t−21 … t−2* (20 days, excludes yesterday).
  2. Yesterday (*t−1*) hedged return:
     diff_ret = r_PAAS − β_signal×r_SLV  (close vs prior close).
  3. **Overreacted** if diff_ret > +threshold → **short PAAS** today.
     **Underreacted** if diff_ret < −threshold → **long PAAS** today.

Execution (session *t*):
  Enter **9:30** open; exit **EOD** or when session PnL hits **−$100** loss cap.
  **$10k notional on PAAS only** (no β hedge leg).

Usage::

    python backtest_daily_overreact.py --start 20230101 --end 20251231
    python backtest_daily_overreact.py --threshold 3.0 --notional 10000 --max-session-loss 100
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
from silver_backtesting.spread import ols_beta, pair_daily_returns

OPEN_HOUR = 9
OPEN_MINUTE = 30

SIGNAL_BETA_OFFSET_START = 21  # day -21
SIGNAL_BETA_OFFSET_END = 2  # day -2 (inclusive)


@dataclass
class DailyOverreactParams:
    threshold_pct: float = 3.0
    notional: float = 10_000.0
    max_session_loss: float = 100.0


def _leg_breakdown(
    side: str,
    notional: float,
    ticker: str,
    price_entry: float,
    price_exit: float,
) -> dict:
    """Single-leg PAAS book: $notional long or short."""
    shares = notional / price_entry if price_entry else 0.0

    if side == "long":
        entry_action, exit_action = "BUY", "SELL"
        pnl = shares * (price_exit - price_entry)
    else:
        entry_action, exit_action = "SELL", "BUY"
        pnl = shares * (price_entry - price_exit)

    return {
        "entry": [
            {
                "Leg": ticker,
                "Action": entry_action,
                "Price": round(price_entry, 4),
                "Notional_usd": round(notional, 2),
                "Shares": round(shares, 4),
            },
        ],
        "exit": [
            {
                "Leg": ticker,
                "Action": exit_action,
                "Price": round(price_exit, 4),
                "Notional_usd": round(shares * price_exit, 2),
                "Shares": round(shares, 4),
            },
        ],
        "PnL_usd": round(pnl, 2),
    }


def _position_pnl_usd(
    side: str,
    notional: float,
    ticker: str,
    entry_price: float,
    mark_price: float,
) -> float:
    return _leg_breakdown(side, notional, ticker, entry_price, mark_price)["PnL_usd"]


def _cap_session_pnl(session_pnl: float, raw_pnl: float, max_session_loss: float) -> float:
    if max_session_loss <= 0:
        return raw_pnl
    return max(raw_pnl, -max_session_loss - session_pnl)


def _scale_leg_pnl(legs: dict, capped_total: float) -> dict:
    raw = legs["PnL_usd"]
    if raw == 0 or abs(raw - capped_total) < 1e-9:
        out = dict(legs)
        out["PnL_usd"] = round(capped_total, 2)
        return out
    scale = capped_total / raw
    out = dict(legs)
    for leg_list in ("entry", "exit"):
        scaled_legs = []
        for leg in legs[leg_list]:
            item = dict(leg)
            item["Notional_usd"] = round(item["Notional_usd"] * scale, 2)
            item["Shares"] = round(item["Shares"] * scale, 4)
            scaled_legs.append(item)
        out[leg_list] = scaled_legs
    out["PnL_usd"] = round(capped_total, 2)
    return out


def build_signal_panel(
    panel: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
) -> pd.DataFrame:
    """
    One row per tradable session with signal beta and yesterday's diff_ret.
    """
    daily_ret = pair_daily_returns(panel, ticker_a, ticker_b)
    if daily_ret is None or daily_ret.empty:
        return pd.DataFrame()

    dates = [int(d) for d in daily_ret.index]
    rows: list[dict] = []

    for i in range(SIGNAL_BETA_OFFSET_START, len(dates)):
        td = dates[i]
        signal_beta_dates = dates[i - SIGNAL_BETA_OFFSET_START : i - SIGNAL_BETA_OFFSET_END + 1]
        signal_date = dates[i - 1]

        if len(signal_beta_dates) != SIGNAL_BETA_OFFSET_START - SIGNAL_BETA_OFFSET_END + 1:
            continue

        win_signal = daily_ret.loc[signal_beta_dates]
        beta_signal = ols_beta(win_signal[ticker_a], win_signal[ticker_b])
        if pd.isna(beta_signal):
            continue

        r_a = float(daily_ret.loc[signal_date, ticker_a])
        r_b = float(daily_ret.loc[signal_date, ticker_b])
        diff_ret = r_a - beta_signal * r_b

        rows.append(
            {
                "Trading date": td,
                "Signal_date": signal_date,
                "beta_signal": float(beta_signal),
                "Signal_r_a_pct": round(r_a * 100, 4),
                "Signal_r_b_pct": round(r_b * 100, 4),
                "Signal_diff_ret_pct": round(diff_ret * 100, 4),
            }
        )

    return pd.DataFrame(rows).sort_values("Trading date").reset_index(drop=True)


def _session_entry(
    panel: pd.DataFrame,
    ticker_a: str,
    td: int,
) -> dict | None:
    day = panel.loc[panel["Trading date"] == td].sort_values("DT")
    if day.empty:
        return None

    open_mask = (day["DT"].dt.hour == OPEN_HOUR) & (day["DT"].dt.minute == OPEN_MINUTE)
    open_row = day.loc[open_mask].iloc[0] if open_mask.any() else day.iloc[0]
    return {
        "entry_dt": open_row["DT"],
        "entry_price": float(open_row[ticker_a]),
    }


def _resolve_intraday_exit(
    panel: pd.DataFrame,
    ticker_a: str,
    td: int,
    *,
    side: str,
    notional: float,
    max_session_loss: float,
    entry_dt: pd.Timestamp,
    entry_price: float,
) -> dict | None:
    """Walk 1-min bars; exit on session loss cap or EOD."""
    day = panel.loc[panel["Trading date"] == td].sort_values("DT")
    if day.empty:
        return None

    session_pnl = 0.0
    after_entry = day.loc[day["DT"] >= entry_dt]
    if after_entry.empty:
        return None

    for _, bar in after_entry.iterrows():
        mark = float(bar[ticker_a])
        unrealized = _position_pnl_usd(
            side, notional, ticker_a, entry_price, mark
        )
        if max_session_loss > 0 and session_pnl + unrealized <= -max_session_loss:
            raw_pnl = unrealized
            capped_pnl = _cap_session_pnl(session_pnl, raw_pnl, max_session_loss)
            legs = _leg_breakdown(side, notional, ticker_a, entry_price, mark)
            reason = "Session loss cap"
            if abs(capped_pnl - raw_pnl) > 1e-9:
                legs = _scale_leg_pnl(legs, capped_pnl)
                reason = f"{reason} (${max_session_loss:.0f})"
            return {
                "exit_dt": bar["DT"],
                "exit_price": mark,
                "exit_reason": reason,
                "legs": legs,
                "pnl_usd": capped_pnl,
            }

    close_row = day.iloc[-1]
    mark = float(close_row[ticker_a])
    legs = _leg_breakdown(side, notional, ticker_a, entry_price, mark)
    return {
        "exit_dt": close_row["DT"],
        "exit_price": mark,
        "exit_reason": "EOD",
        "legs": legs,
        "pnl_usd": legs["PnL_usd"],
    }


def run_backtest(
    panel: pd.DataFrame,
    signals: pd.DataFrame,
    p: DailyOverreactParams,
    ticker_a: str,
    ticker_b: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    thresh = p.threshold_pct / 100.0
    trades: list[dict] = []
    trade_id = 0

    for _, row in signals.iterrows():
        diff = float(row["Signal_diff_ret_pct"]) / 100.0
        if diff > thresh:
            side = "short"
            flag = "overreacted"
        elif diff < -thresh:
            side = "long"
            flag = "underreacted"
        else:
            continue

        td = int(row["Trading date"])
        entry = _session_entry(panel, ticker_a, td)
        if entry is None:
            continue

        exit_px = _resolve_intraday_exit(
            panel,
            ticker_a,
            td,
            side=side,
            notional=p.notional,
            max_session_loss=p.max_session_loss,
            entry_dt=entry["entry_dt"],
            entry_price=entry["entry_price"],
        )
        if exit_px is None:
            continue

        legs = exit_px["legs"]
        pnl_usd = exit_px["pnl_usd"]
        return_pct = 100.0 * pnl_usd / p.notional if p.notional else 0.0

        trade_id += 1
        trades.append(
            {
                "Trade_id": trade_id,
                "Signal_date": int(row["Signal_date"]),
                "Signal_flag": flag,
                "Signal_diff_ret_pct": row["Signal_diff_ret_pct"],
                "Beta_signal": round(float(row["beta_signal"]), 4),
                "Side": side,
                "Entry_date": td,
                "Entry_time": pd.Timestamp(entry["entry_dt"]).strftime("%Y-%m-%d %H:%M"),
                "Exit_date": td,
                "Exit_time": pd.Timestamp(exit_px["exit_dt"]).strftime("%Y-%m-%d %H:%M"),
                "Exit_reason": exit_px["exit_reason"],
                f"Entry_{ticker_a}": round(entry["entry_price"], 4),
                f"Exit_{ticker_a}": round(exit_px["exit_price"], 4),
                "Return_pct": round(return_pct, 4),
                "PnL_$": round(pnl_usd, 2),
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
    p: DailyOverreactParams,
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
        ("Long PAAS (underreacted)", int((trades_df["Side"] == "long").sum())),
        ("Short PAAS (overreacted)", int((trades_df["Side"] == "short").sum())),
        ("Total PnL ($)", round(total, 2)),
        ("Avg return per trade (%)", round(trades_df["Return_pct"].mean(), 3)),
        ("Avg PnL per trade ($)", round(trades_df["PnL_$"].mean(), 2)),
        ("Win rate (%)", round(100 * wins / n, 1)),
        ("Best trade ($)", round(trades_df["PnL_$"].max(), 2)),
        ("Worst trade ($)", round(trades_df["PnL_$"].min(), 2)),
        ("Signal β window", f"day −{SIGNAL_BETA_OFFSET_START} … −{SIGNAL_BETA_OFFSET_END}"),
        ("Signal metric", f"r_{ticker_a} − β_signal×r_{ticker_b} on prior session (%)"),
        ("Position", f"Directional {ticker_a} only (${p.notional:,.0f}, no {ticker_b} hedge)"),
        ("Threshold (%)", f"±{p.threshold_pct}"),
        ("Entry", "9:30 open"),
        ("Exit", "EOD or session loss cap"),
        ("Max session loss ($)", p.max_session_loss if p.max_session_loss > 0 else "off"),
        (
            "Cap exits",
            int(trades_df["Exit_reason"].astype(str).str.contains("loss cap").sum()),
        ),
        ("Notional", p.notional),
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
        "Signal_date",
        "Signal_flag",
        "Side",
        "Signal_diff_ret_pct",
        "Beta_signal",
        "Entry_date",
        "Exit_reason",
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
            elif c in ("Signal_diff_ret_pct", "Return_pct") and pd.notna(val):
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
            "Signal_date": int(row["Signal_date"]),
            "Signal_flag": row["Signal_flag"],
            "Signal_diff_ret_pct": float(row["Signal_diff_ret_pct"]),
            "Beta_signal": float(row["Beta_signal"]),
            "Side": row["Side"],
            "Entry_date": int(row["Entry_date"]),
            "Entry_time": row["Entry_time"],
            "Exit_date": int(row["Exit_date"]),
            "Exit_time": row["Exit_time"],
            "Exit_reason": row["Exit_reason"],
            f"Entry_{ticker_a}": float(row[f"Entry_{ticker_a}"]),
            f"Exit_{ticker_a}": float(row[f"Exit_{ticker_a}"]),
            "Return_pct": float(row["Return_pct"]),
            "PnL_usd": float(row["PnL_$"]),
            "legs": row["_legs"],
            "ticker_a": ticker_a,
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
    p: DailyOverreactParams,
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
        title="Daily overreaction backtest — cumulative PnL",
    )
    fig.update_xaxes(title_text="Trading date", row=1, col=1)
    fig.update_xaxes(title_text="Trading date", row=2, col=1)
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
<html><head><meta charset="utf-8"><title>Daily overreaction backtest</title>
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
<h1>Daily overreaction — {ticker_a} vs {ticker_b}</h1>
<h2>Summary</h2>{summary.to_html(index=False, classes="data-table", border=0)}
<div>{chart}</div>
<h2>Daily PnL</h2>
<p>Click a row, chart bar, or trade for leg detail.</p>
{_daily_table_html(daily_df)}
<h2>All trades</h2>
{_trades_table_html(trades_df)}
<h2 id="legs-title">Trade legs</h2>
<div id="explainer">
  <strong>Signal β</strong> (day −{SIGNAL_BETA_OFFSET_START}…−{SIGNAL_BETA_OFFSET_END}): OLS on daily returns.<br>
  <strong>Yesterday diff_ret</strong> = r_{ticker_a} − β_signal×r_{ticker_b}.<br>
  <strong>Overreacted</strong> (&gt; +{p.threshold_pct}%): PAAS rich vs β×{ticker_b} → <strong>SHORT {ticker_a}</strong> ${p.notional:,.0f}.<br>
  <strong>Underreacted</strong> (&lt; −{p.threshold_pct}%): PAAS cheap → <strong>LONG {ticker_a}</strong> ${p.notional:,.0f}.<br>
  No SLV hedge leg — directional {ticker_a} only.<br>
  <strong>Exit:</strong> EOD, or intraday if mark-to-market PnL ≤ −${p.max_session_loss:.0f} (``--max-session-loss``).
</div>
<p id="trade-meta">No trade selected.</p>
<div id="legs-panel"><p>Select a trade above.</p></div>
<script>
const TRADES = {_trades_json(trades_df, ticker_a, ticker_b)};
const TRADES_BY_EXIT = {_trades_by_exit_json(trades_df)};
const TD_LABELS = {json.dumps(td_dates)};

function legTable(title, timeLabel, legs) {{
  let html = '<h3>' + title + ' (' + timeLabel + ')</h3>';
  html += '<table class="data-table"><thead><tr><th>Ticker</th><th>Action</th><th>Price</th><th>Notional $</th><th>Shares</th></tr></thead><tbody>';
  for (const leg of legs) {{
    html += '<tr><td>' + leg.Leg + '</td><td>' + leg.Action + '</td><td>' + leg.Price + '</td><td>' + leg.Notional_usd + '</td><td>' + leg.Shares + '</td></tr>';
  }}
  html += '</tbody></table>';
  return html;
}}

function renderTrade(tradeId) {{
  const t = TRADES[String(tradeId)];
  if (!t) return;
  const title = document.getElementById('legs-title');
  const meta = document.getElementById('trade-meta');
  const panel = document.getElementById('legs-panel');
  const pnlTotal = t.PnL_usd;
  title.textContent = 'Trade #' + t.Trade_id + ' — ' + t.Side.toUpperCase() + ' ' + t.ticker_a + ' (' + t.Signal_flag + ')';
  meta.textContent =
    'Signal ' + t.Signal_date + ' diff_ret ' + t.Signal_diff_ret_pct.toFixed(3) + '%'
    + ' | β_signal ' + t.Beta_signal
    + ' | Entry ' + t.Entry_time + ' Exit ' + t.Exit_time
    + ' (' + t.Exit_reason + ')'
    + ' | PnL $' + pnlTotal.toFixed(2);

  let html = legTable('Entry', t.Entry_time, t.legs.entry);
  html += legTable('Exit', t.Exit_time, t.legs.exit);
  html += '<table class="data-table"><thead><tr><th>PnL</th><th>$</th></tr></thead><tbody>';
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
    document.querySelectorAll('.trade-row').forEach(tr => tr.classList.remove('selected'));
    document.getElementById('legs-title').textContent = 'Trades on ' + (TD_LABELS[String(td)] || td);
    document.getElementById('trade-meta').textContent = ids.length + ' trades — click a trade row.';
    let html = '<ul>';
    ids.forEach(id => {{
      const t = TRADES[String(id)];
      html += '<li><a href="#" data-trade="' + id + '">#' + id + ' ' + t.Side + ' PnL $' + t.PnL_usd.toFixed(2) + '</a></li>';
    }});
    html += '</ul>';
    document.getElementById('legs-panel').innerHTML = html;
    document.querySelectorAll('#legs-panel a[data-trade]').forEach(a => {{
      a.addEventListener('click', e => {{ e.preventDefault(); selectTrade(a.dataset.trade); }});
    }});
  }} else {{
    document.getElementById('legs-panel').innerHTML = '<p>No trades for this date.</p>';
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
    default_start = int((today - timedelta(days=365 * 2)).strftime("%Y%m%d"))

    p = argparse.ArgumentParser(
        description="Daily overreaction: directional PAAS signal, open-to-EOD trade."
    )
    p.add_argument("--ticker-a", default="PAAS")
    p.add_argument("--ticker-b", default="SLV")
    p.add_argument("--start", type=int, default=default_start)
    p.add_argument("--end", type=int, default=default_end)
    p.add_argument(
        "--threshold",
        type=float,
        default=3.0,
        help="Signal threshold on prior-day diff_ret (%%); default 3",
    )
    p.add_argument("--notional", type=float, default=10_000.0)
    p.add_argument(
        "--max-session-loss",
        type=float,
        default=100.0,
        help="Exit when session PnL hits this loss ($); 0 = off",
    )
    p.add_argument("--data-folder", default=None)
    p.add_argument("--output", default="backtest_daily_overreact.html")
    p.add_argument("--no-browser", action="store_true")
    p.add_argument("--export-csv", default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    folder = args.data_folder or Paths().min_folder
    params = DailyOverreactParams(
        threshold_pct=args.threshold,
        notional=args.notional,
        max_session_loss=args.max_session_loss,
    )

    print(f"Loading {args.ticker_a}/{args.ticker_b} ({args.start}–{args.end}) …")
    panel = load_pair_panel(args.ticker_a, args.ticker_b, folder, args.start, args.end)
    signals = build_signal_panel(panel, args.ticker_a, args.ticker_b)
    print(f"  {panel['Trading date'].nunique()} sessions, {len(signals)} days with signal betas")

    trades_df, daily_df = run_backtest(
        panel, signals, params, args.ticker_a, args.ticker_b
    )
    summary = summary_stats(
        trades_df, daily_df, params, args.ticker_a, args.ticker_b
    )
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
        signals.to_csv(os.path.join(args.export_csv, "signals.csv"), index=False)
        trades_df.drop(columns=["_legs"]).to_csv(
            os.path.join(args.export_csv, "trades.csv"), index=False
        )
        daily_df.to_csv(os.path.join(args.export_csv, "daily_pnl.csv"), index=False)
        print(f"CSV exported to {args.export_csv}")

    if not args.no_browser:
        webbrowser.open(f"file://{os.path.abspath(out)}")


if __name__ == "__main__":
    main()
