#!/usr/bin/env python3
"""
β₂₀ surprise pairs trade on 5-min bars (since prior close).

At each 5-min bar close:
  r_B      = P_SLV / prior_close_SLV − 1
  r_A      = P_PAAS / prior_close_PAAS − 1
  expected = β₂₀ × r_B
  surprise = r_A − expected

Signals (one position at a time; re-entry after flat):
  surprise > +threshold  → **Short PAAS, Long SLV**  (PAAS rich vs β expectation)
  surprise < −threshold  → **Long PAAS, Short SLV**

Exit:
  • 15 minutes after entry (3 × 5-min bars), or
  • Take profit when position PnL ≥ +tp_pct on $10k leg, or
  • EOD (last RTH bar) if < 15 min remain in session

$10k notional on PAAS; SLV leg sized **$10k × β₂₀** (beta-hedged book).

Usage::

    python backtest_surprise_5m.py --start 20240101 --end 20251231
    python backtest_surprise_5m.py --threshold 2.0 --hold-minutes 15 --tp-pct 1.0
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
from silver_backtesting.spread import lookback_beta_by_trading_date

BAR_MINUTES = 5


@dataclass
class SurpriseParams:
    threshold_pct: float = 2.0
    hold_minutes: int = 15
    tp_pct: float = 1.0
    beta_lookback: int = 20
    notional: float = 10_000.0


def _leg_breakdown_pair(
    side: str,
    notional: float,
    price_a_entry: float,
    price_b_entry: float,
    price_a_exit: float,
    price_b_exit: float,
    beta_20: float,
) -> dict:
    """
    β-hedged sizing: $notional on PAAS, $notional×β₂₀ on SLV.

    side ``short_a_long_b``: short PAAS, long SLV (fade rich PAAS).
    side ``long_a_short_b``: long PAAS, short SLV (fade cheap PAAS).
    """
    n_a = notional
    n_b = notional * beta_20
    sh_a = n_a / price_a_entry if price_a_entry else 0.0
    sh_b = n_b / price_b_entry if price_b_entry else 0.0

    if side == "short_a_long_b":
        entry_a, entry_b = "SELL", "BUY"
        exit_a, exit_b = "BUY", "SELL"
        pnl_a = sh_a * (price_a_entry - price_a_exit)
        pnl_b = sh_b * (price_b_exit - price_b_entry)
    else:
        entry_a, entry_b = "BUY", "SELL"
        exit_a, exit_b = "SELL", "BUY"
        pnl_a = sh_a * (price_a_exit - price_a_entry)
        pnl_b = sh_b * (price_b_entry - price_b_exit)

    return {
        "entry": [
            {
                "Leg": "A",
                "Action": entry_a,
                "Price": round(price_a_entry, 4),
                "Notional_usd": round(n_a, 2),
                "Shares": round(sh_a, 4),
            },
            {
                "Leg": "B",
                "Action": entry_b,
                "Price": round(price_b_entry, 4),
                "Beta_20": round(beta_20, 4),
                "Notional_usd": round(n_b, 2),
                "Shares": round(sh_b, 4),
            },
        ],
        "exit": [
            {
                "Leg": "A",
                "Action": exit_a,
                "Price": round(price_a_exit, 4),
                "Notional_usd": round(sh_a * price_a_exit, 2),
                "Shares": round(sh_a, 4),
            },
            {
                "Leg": "B",
                "Action": exit_b,
                "Price": round(price_b_exit, 4),
                "Notional_usd": round(sh_b * price_b_exit, 2),
                "Shares": round(sh_b, 4),
            },
        ],
        "PnL_A_usd": round(pnl_a, 2),
        "PnL_B_usd": round(pnl_b, 2),
        "PnL_total_usd": round(pnl_a + pnl_b, 2),
    }


def _prior_close_by_date(panel: pd.DataFrame, ticker_a: str, ticker_b: str) -> pd.DataFrame:
    """Map each session to prior session's last RTH close."""
    last = (
        panel.sort_values("DT")
        .groupby("Trading date")[[ticker_a, ticker_b]]
        .last()
        .reset_index()
    )
    last["next_td"] = last["Trading date"].shift(-1)
    last = last.dropna(subset=["next_td"])
    last["next_td"] = last["next_td"].astype(int)
    return last.rename(
        columns={
            ticker_a: "prev_close_a",
            ticker_b: "prev_close_b",
            "Trading date": "prev_td",
        }
    )[["next_td", "prev_close_a", "prev_close_b"]]


def build_5m_surprise_bars(
    panel: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    beta_lookback: int,
) -> pd.DataFrame:
    """5-min RTH grid with surprise vs β₂₀ expectation since prior close."""
    beta20 = lookback_beta_by_trading_date(
        panel, ticker_a, ticker_b, lookback_days=beta_lookback, lag_sessions=1
    )
    prev_map = _prior_close_by_date(panel, ticker_a, ticker_b)
    prev_lut = prev_map.set_index("next_td")

    lag = pd.Timedelta(minutes=BAR_MINUTES)
    step = lag
    one = pd.Timedelta(minutes=1)
    rows: list[dict] = []

    for td, group in panel.groupby("Trading date"):
        td = int(td)
        if td not in prev_lut.index:
            continue
        prev_a = float(prev_lut.at[td, "prev_close_a"])
        prev_b = float(prev_lut.at[td, "prev_close_b"])
        beta = beta20.get(td, float("nan"))
        if pd.isna(beta) or prev_a <= 0 or prev_b <= 0:
            continue

        g = group.sort_values("DT")
        dt_set = set(g["DT"])
        px = g.set_index("DT")[[ticker_a, ticker_b]]
        session_start = g["DT"].iloc[0]
        last_dt = g["DT"].iloc[-1]

        for dt in g["DT"]:
            if dt <= session_start:
                continue
            if (dt - session_start) % step != pd.Timedelta(0):
                continue

            pa = float(px.at[dt, ticker_a])
            pb = float(px.at[dt, ticker_b])
            if any(pd.isna(v) or v == 0 for v in (pa, pb)):
                continue

            r_b = pb / prev_b - 1.0
            r_a = pa / prev_a - 1.0
            expected_a = float(beta) * r_b
            surprise = r_a - expected_a

            rows.append(
                {
                    "Trading date": td,
                    "DT": dt,
                    "price_a": pa,
                    "price_b": pb,
                    "beta_20": float(beta),
                    "r_a_pct": r_a * 100,
                    "r_b_pct": r_b * 100,
                    "expected_a_pct": expected_a * 100,
                    "surprise_pct": surprise * 100,
                    "last_dt": last_dt,
                }
            )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["Trading date", "DT"]).reset_index(drop=True)


def _position_pnl_pct(
    side: str,
    notional: float,
    beta_20: float,
    entry_a: float,
    entry_b: float,
    exit_a: float,
    exit_b: float,
) -> float:
    legs = _leg_breakdown_pair(side, notional, entry_a, entry_b, exit_a, exit_b, beta_20)
    return 100.0 * legs["PnL_total_usd"] / notional if notional else 0.0


def simulate_session(
    day: pd.DataFrame,
    td: int,
    ticker_a: str,
    ticker_b: str,
    p: SurpriseParams,
) -> list[dict]:
    thresh = p.threshold_pct / 100.0
    tp = p.tp_pct / 100.0
    hold_delta = pd.Timedelta(minutes=p.hold_minutes)
    fills: list[dict] = []
    trade_id_base = 0

    pos: dict | None = None

    for _, row in day.iterrows():
        dt = row["DT"]
        pa = float(row["price_a"])
        pb = float(row["price_b"])

        if pos is not None:
            pnl_pct = _position_pnl_pct(
                pos["side"],
                p.notional,
                float(pos["beta_20"]),
                pos["entry_a"],
                pos["entry_b"],
                pa,
                pb,
            ) / 100.0

            time_exit = dt >= pos["entry_dt"] + hold_delta
            tp_exit = pnl_pct >= tp
            eod_exit = dt >= row["last_dt"]

            reason: str | None = None
            if tp_exit:
                reason = f"TP {pnl_pct*100:.2f}%"
            elif time_exit:
                reason = f"Hold {p.hold_minutes}m"
            elif eod_exit:
                reason = "EOD"

            if reason:
                beta_e = float(pos["beta_20"])
                legs = _leg_breakdown_pair(
                    pos["side"],
                    p.notional,
                    pos["entry_a"],
                    pos["entry_b"],
                    pa,
                    pb,
                    beta_e,
                )
                fills.append(
                    {
                        "Trading date": td,
                        "Side": pos["side_label"],
                        "Entry_time": pd.Timestamp(pos["entry_dt"]).strftime("%Y-%m-%d %H:%M"),
                        "Exit_time": pd.Timestamp(dt).strftime("%Y-%m-%d %H:%M"),
                        "Entry_surprise_pct": round(pos["surprise_pct"], 4),
                        "Exit_reason": reason,
                        "Entry_beta_20": round(beta_e, 4),
                        f"Entry_{ticker_a}": round(pos["entry_a"], 4),
                        f"Entry_{ticker_b}": round(pos["entry_b"], 4),
                        f"Exit_{ticker_a}": round(pa, 4),
                        f"Exit_{ticker_b}": round(pb, 4),
                        "Return_pct": round(pnl_pct * 100, 4),
                        "PnL_$": round(legs["PnL_total_usd"], 2),
                        "PnL_A_$": legs["PnL_A_usd"],
                        "PnL_B_$": legs["PnL_B_usd"],
                        "_legs": legs,
                    }
                )
                pos = None

        if pos is None:
            surprise = float(row["surprise_pct"]) / 100.0
            if surprise > thresh:
                pos = {
                    "side": "short_a_long_b",
                    "side_label": "Short PAAS / Long SLV",
                    "entry_dt": dt,
                    "entry_a": pa,
                    "entry_b": pb,
                    "beta_20": float(row["beta_20"]),
                    "surprise_pct": float(row["surprise_pct"]),
                }
            elif surprise < -thresh:
                pos = {
                    "side": "long_a_short_b",
                    "side_label": "Long PAAS / Short SLV",
                    "entry_dt": dt,
                    "entry_a": pa,
                    "entry_b": pb,
                    "beta_20": float(row["beta_20"]),
                    "surprise_pct": float(row["surprise_pct"]),
                }

    if pos is not None:
        last_row = day.iloc[-1]
        dt = last_row["DT"]
        pa = float(last_row["price_a"])
        pb = float(last_row["price_b"])
        pnl_pct = _position_pnl_pct(
            pos["side"],
            p.notional,
            float(pos["beta_20"]),
            pos["entry_a"],
            pos["entry_b"],
            pa,
            pb,
        ) / 100.0
        beta_e = float(pos["beta_20"])
        legs = _leg_breakdown_pair(
            pos["side"], p.notional, pos["entry_a"], pos["entry_b"], pa, pb, beta_e
        )
        fills.append(
            {
                "Trading date": td,
                "Side": pos["side_label"],
                "Entry_time": pd.Timestamp(pos["entry_dt"]).strftime("%Y-%m-%d %H:%M"),
                "Exit_time": pd.Timestamp(dt).strftime("%Y-%m-%d %H:%M"),
                "Entry_surprise_pct": round(pos["surprise_pct"], 4),
                "Exit_reason": "EOD",
                "Entry_beta_20": round(beta_e, 4),
                f"Entry_{ticker_a}": round(pos["entry_a"], 4),
                f"Entry_{ticker_b}": round(pos["entry_b"], 4),
                f"Exit_{ticker_a}": round(pa, 4),
                f"Exit_{ticker_b}": round(pb, 4),
                "Return_pct": round(pnl_pct * 100, 4),
                "PnL_$": round(legs["PnL_total_usd"], 2),
                "PnL_A_$": legs["PnL_A_usd"],
                "PnL_B_$": legs["PnL_B_usd"],
                "_legs": legs,
            }
        )

    return fills


def run_backtest(
    bars: pd.DataFrame,
    p: SurpriseParams,
    ticker_a: str,
    ticker_b: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_fills: list[dict] = []
    trade_id = 0

    for td, day in bars.groupby("Trading date"):
        session_fills = simulate_session(day, int(td), ticker_a, ticker_b, p)
        for f in session_fills:
            trade_id += 1
            f["Trade_id"] = trade_id
            all_fills.append(f)

    trades_df = pd.DataFrame(all_fills)
    if trades_df.empty:
        return trades_df, pd.DataFrame()

    daily_df = (
        trades_df.groupby("Trading date", as_index=False)
        .agg(trades=("Trade_id", "count"), session_pnl=("PnL_$", "sum"))
        .rename(columns={"trades": "Trades", "session_pnl": "Session_PnL_$"})
        .sort_values("Trading date")
    )
    daily_df["Cum_PnL_$"] = daily_df["Session_PnL_$"].cumsum()
    return trades_df, daily_df


def summary_stats(
    trades_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    p: SurpriseParams,
    ticker_a: str,
    ticker_b: str,
) -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame({"Metric": ["Trades"], "Value": [0]})

    total = trades_df["PnL_$"].sum()
    wins = (trades_df["PnL_$"] > 0).sum()
    n = len(trades_df)
    n_tp = trades_df["Exit_reason"].astype(str).str.startswith("TP").sum()
    n_hold = trades_df["Exit_reason"].astype(str).str.startswith("Hold").sum()
    n_eod = (trades_df["Exit_reason"] == "EOD").sum()

    rows = [
        ("Trades", n),
        ("Sessions", daily_df["Trading date"].nunique()),
        ("Short PAAS / Long SLV", int((trades_df["Side"].str.startswith("Short")).sum())),
        ("Long PAAS / Short SLV", int((trades_df["Side"].str.startswith("Long")).sum())),
        ("Total PnL ($)", round(total, 2)),
        ("Avg PnL per trade ($)", round(trades_df["PnL_$"].mean(), 2)),
        ("Win rate (%)", round(100 * wins / n, 1)),
        ("TP exits", int(n_tp)),
        ("Time exits (15m)", int(n_hold)),
        ("EOD exits", int(n_eod)),
        ("Bar size", f"{BAR_MINUTES} min"),
        ("Surprise threshold (%)", f"±{p.threshold_pct}"),
        ("β₂₀ lookback", p.beta_lookback),
        ("Hold (min)", p.hold_minutes),
        ("Take profit (%)", p.tp_pct),
        ("Notional leg A", p.notional),
        ("Hedge leg B", f"${p.notional:,.0f} × β₂₀"),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value"])


def _trading_dates_for_plot(daily_df: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(daily_df["Trading date"].astype(str), format="%Y%m%d")


def _daily_table_html(daily_df: pd.DataFrame) -> str:
    if daily_df.empty:
        return "<p>No trades</p>"
    cols = list(daily_df.columns)
    parts = ["<table class='data-table interactive-table'><thead><tr>"]
    parts.extend(f"<th>{c}</th>" for c in cols)
    parts.append("</tr></thead><tbody>")
    for _, row in daily_df.iterrows():
        td = int(row["Trading date"])
        parts.append(f"<tr class='daily-row' data-td='{td}' tabindex='0'>")
        for c in cols:
            val = row[c]
            text = f"{float(val):,.2f}" if c in ("Session_PnL_$", "Cum_PnL_$") else str(val)
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
        "Entry_time",
        "Exit_time",
        "Entry_surprise_pct",
        "Exit_reason",
        "Return_pct",
        "PnL_$",
    ]
    parts = ["<table class='data-table interactive-table'><thead><tr>"]
    parts.extend(f"<th>{c}</th>" for c in cols)
    parts.append("</tr></thead><tbody>")
    for _, row in trades_df.iterrows():
        tid = int(row["Trade_id"])
        parts.append(f"<tr class='trade-row' data-trade='{tid}' tabindex='0'>")
        for c in cols:
            val = row[c]
            if c == "PnL_$":
                text = f"{float(val):,.2f}"
            elif c in ("Entry_surprise_pct", "Return_pct"):
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
            "Entry_time": row["Entry_time"],
            "Exit_time": row["Exit_time"],
            "Entry_surprise_pct": float(row["Entry_surprise_pct"]),
            "Exit_reason": row["Exit_reason"],
            "Entry_beta_20": float(row["Entry_beta_20"]),
            "Entry_A": float(row[f"Entry_{ticker_a}"]),
            "Entry_B": float(row[f"Entry_{ticker_b}"]),
            "Exit_A": float(row[f"Exit_{ticker_a}"]),
            "Exit_B": float(row[f"Exit_{ticker_b}"]),
            "Return_pct": float(row["Return_pct"]),
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
    g: dict[str, list[int]] = {}
    for _, row in trades_df.iterrows():
        g.setdefault(str(int(row["Trading date"])), []).append(int(row["Trade_id"]))
    return json.dumps(g)


def build_report_html(
    summary: pd.DataFrame,
    trades_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    p: SurpriseParams,
) -> str:
    fig = make_subplots(rows=2, cols=1, row_heights=[0.45, 0.55], vertical_spacing=0.08)
    if not daily_df.empty:
        x = _trading_dates_for_plot(daily_df)
        fig.add_trace(go.Bar(x=x, y=daily_df["Session_PnL_$"], name="Daily PnL"), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=daily_df["Cum_PnL_$"], mode="lines", name="Cum PnL"), row=2, col=1)
    fig.update_layout(template="plotly_white", height=700, title="β₂₀ surprise 5m pairs")
    chart = fig.to_html(full_html=False, include_plotlyjs="cdn", div_id="pnl-chart")

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Surprise 5m</title>
<style>
body {{ font-family: sans-serif; margin: 2rem; }}
h1,h2,h3 {{ color: #0066cc; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
th {{ background: #0066cc; color: white; padding: 6px; }}
td {{ border: 1px solid #ddd; padding: 6px; }}
.interactive-table tbody tr {{ cursor: pointer; }}
.interactive-table tbody tr:hover {{ background: #eef6ff; }}
.interactive-table tbody tr.selected {{ background: #cce0ff; }}
#explainer {{ background: #f7f9fc; border-left: 4px solid #0066cc; padding: 0.75rem 1rem; font-size: 13px; }}
</style></head><body>
<h1>β₂₀ surprise — {ticker_a} vs {ticker_b}</h1>
<h2>Summary</h2>{summary.to_html(index=False, border=0)}
<div>{chart}</div>
<h2>Daily PnL</h2>{_daily_table_html(daily_df)}
<h2>Trades</h2>{_trades_table_html(trades_df)}
<h2 id="legs-title">Trade legs</h2>
<div id="explainer">
  surprise = r_{ticker_a}(since prior close) − β₂₀×r_{ticker_b}(since prior close) on {BAR_MINUTES}m bars.<br>
  surprise &gt; +{p.threshold_pct}% → Short {ticker_a}, Long {ticker_b}. &lt; −{p.threshold_pct}% → opposite.<br>
  Exit: {p.hold_minutes}m, TP {p.tp_pct}%, or EOD. ${p.notional:,.0f} on {ticker_a}, ${p.notional:,.0f}×β₂₀ on {ticker_b}.
</div>
<p id="trade-meta">Select a trade.</p>
<div id="legs-panel"></div>
<script>
const TRADES = {_trades_json(trades_df, ticker_a, ticker_b)};
const BY_TD = {_trades_by_td_json(trades_df)};

function legTable(title, time, rows, ta, tb) {{
  let h = '<h3>'+title+' — '+time+'</h3><table class="data-table"><tr><th>Leg</th><th>Ticker</th><th>Action</th><th>Price</th><th>$</th><th>Shares</th><th>β₂₀</th></tr>';
  for (const r of rows) {{
    const t = r.Leg==='A'?ta:tb;
    h += '<tr><td>'+r.Leg+'</td><td>'+t+'</td><td>'+r.Action+'</td><td>'+r.Price+'</td><td>'+r.Notional_usd+'</td><td>'+r.Shares+'</td><td>'+(r.Beta_20||'—')+'</td></tr>';
  }}
  return h+'</table>';
}}

function show(id) {{
  const t = TRADES[String(id)]; if (!t) return;
  const tot = t.PnL_A_usd + t.PnL_B_usd;
  document.getElementById('legs-title').textContent = '#'+t.Trade_id+' '+t.Side;
  document.getElementById('trade-meta').textContent =
    t.Trading_date+' surprise '+t.Entry_surprise_pct.toFixed(3)+'% → '+t.Exit_reason+' | $'+tot.toFixed(2);
  let h = legTable('Entry', t.Entry_time, t.legs.entry, t.ticker_a, t.ticker_b);
  h += legTable('Exit', t.Exit_time, t.legs.exit, t.ticker_a, t.ticker_b);
  h += '<table class="data-table"><tr><td>'+t.ticker_a+'</td><td>'+t.PnL_A_usd.toFixed(2)+'</td></tr>';
  h += '<tr><td>'+t.ticker_b+'</td><td>'+t.PnL_B_usd.toFixed(2)+'</td></tr>';
  h += '<tr><td><b>Total</b></td><td><b>'+tot.toFixed(2)+'</b></td></tr></table>';
  document.getElementById('legs-panel').innerHTML = h;
}}

document.querySelectorAll('.trade-row').forEach(r => r.onclick = () => show(r.dataset.trade));
document.querySelectorAll('.daily-row').forEach(r => r.onclick = () => {{
  const ids = BY_TD[r.dataset.td]||[]; if (ids.length) show(ids[0]);
}});
if (Object.keys(TRADES).length) show(Object.keys(TRADES).pop());
</script></body></html>"""


def _parse_args() -> argparse.Namespace:
    today = datetime.now() - timedelta(days=3)
    p = argparse.ArgumentParser(description="β₂₀ surprise 5m pairs backtest.")
    p.add_argument("--ticker-a", default="PAAS")
    p.add_argument("--ticker-b", default="SLV")
    p.add_argument("--start", type=int, default=int((today - timedelta(days=180)).strftime("%Y%m%d")))
    p.add_argument("--end", type=int, default=int(today.strftime("%Y%m%d")))
    p.add_argument("--threshold", type=float, default=2.0, help="Surprise %% trigger")
    p.add_argument("--hold-minutes", type=int, default=15)
    p.add_argument("--tp-pct", type=float, default=1.0)
    p.add_argument("--beta-lookback", type=int, default=20)
    p.add_argument("--notional", type=float, default=10_000.0)
    p.add_argument("--data-folder", default=None)
    p.add_argument("--output", default="surprise_5m_backtest.html")
    p.add_argument("--no-browser", action="store_true")
    p.add_argument("--export-csv", default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    params = SurpriseParams(
        threshold_pct=args.threshold,
        hold_minutes=args.hold_minutes,
        tp_pct=args.tp_pct,
        beta_lookback=args.beta_lookback,
        notional=args.notional,
    )
    folder = args.data_folder or Paths().min_folder

    print(f"Loading {args.ticker_a}/{args.ticker_b} …")
    panel = load_pair_panel(args.ticker_a, args.ticker_b, folder, args.start, args.end)
    bars = build_5m_surprise_bars(panel, args.ticker_a, args.ticker_b, params.beta_lookback)
    print(f"  {bars['Trading date'].nunique()} sessions, {len(bars):,} 5m bars")

    trades_df, daily_df = run_backtest(bars, params, args.ticker_a, args.ticker_b)
    summary = summary_stats(trades_df, daily_df, params, args.ticker_a, args.ticker_b)
    print("\n=== Summary ===")
    print(summary.to_string(index=False))

    out = args.output
    with open(out, "w", encoding="utf-8") as f:
        f.write(build_report_html(summary, trades_df, daily_df, args.ticker_a, args.ticker_b, params))
    print(f"\nReport: {out}")

    if args.export_csv:
        os.makedirs(args.export_csv, exist_ok=True)
        trades_df.drop(columns=["_legs"]).to_csv(f"{args.export_csv}/trades.csv", index=False)
        daily_df.to_csv(f"{args.export_csv}/daily_pnl.csv", index=False)

    if not args.no_browser:
        webbrowser.open(f"file://{os.path.abspath(out)}")


if __name__ == "__main__":
    main()
