#!/usr/bin/env python3
"""
Kalman-filter hedge-ratio pairs trade on 1-min bars (SLV / PAAS).

Three layers, kept strictly separate:

  Layer 0 — Offline prior (once/day, before open)
      OLS  y = beta*x + eps  over last N historical trading days
      -> beta0, P0 (inflated), R0 (frozen residual variance for the day)

  Layer 1 — Online filter (runs every single minute, all session,
            regardless of trade state; this is the "brain")
      P-  = P / mu                      (forgetting factor inflate)
      yhat= beta * x
      e   = y - yhat
      K   = P-*x / (x^2*P- + R)
      beta= beta + K*e
      P   = (1 - K*x) * P-
      S   = x^2*P + R
      z   = e / sqrt(S)

  Layer 2 — Trade state machine (gated overlay, only "live" after
            the observation window; entries only fire post 10:00)
      FLAT        -> on |z| > entry_z: queue entry for next bar close
                     (e.g. signal 10:07 -> fill 10:08); flat slippage cost on PnL
      IN_POSITION -> mark-to-market P&L from entry prices (NOT an
                     accumulator of per-bar residuals). Exit on:
                       1) pnl_pct >= tp_pct        ("TP")
                       2) elapsed >= hold_minutes  ("Time stop")
                       3) end of session           ("EOD", safety net)
                     No stop-loss by design (time-box is the only cap).

Usage::

    python kalman_pairs_backtest.py --start 20240101 --end 20251231
    python kalman_pairs_backtest.py --entry-z 2.0 --hold-minutes 15 --tp-pct 1.0
"""

from __future__ import annotations

import argparse
import json
import os
import webbrowser
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from silver_backtesting.config import Paths
from silver_backtesting.pair_quick import load_pair_panel

BAR_MINUTES = 1
OBS_END_TIME = "10:00"      # end of observation-only window (no trading)
SESSION_START_TIME = "09:30"


# --------------------------------------------------------------------------- #
# Layer 0 — Offline prior
# --------------------------------------------------------------------------- #

@dataclass
class Prior:
    beta0: float
    P0: float
    R0: float


def fit_prior(
    hist: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    lookback_days: int,
    p0_inflation: float = 5.0,
) -> Prior:
    """
    OLS y = beta*x + eps over the trailing `lookback_days` sessions of
    minute returns (pooled across days), evaluated as-of the day the
    caller has already sliced `hist` down to (i.e. hist should already
    exclude the current/live session).

    p0_inflation: OLS SE^2 on beta understates day-to-day beta
    instability (it's *within-sample* sampling error, not the variance
    of beta drifting session to session). We scale it up so the filter
    isn't overconfident on day 1. Tune this against your own data by
    comparing OLS SE^2 to the empirical variance of rolling daily betas.
    """
    days = sorted(hist["Trading date"].unique())[-lookback_days:]
    sub = hist[hist["Trading date"].isin(days)].sort_values("DT")

    x = sub[ticker_b].pct_change().dropna()
    y = sub[ticker_a].pct_change().dropna()
    idx = x.index.intersection(y.index)
    x, y = x.loc[idx].values, y.loc[idx].values

    if len(x) < 30 or np.var(x) == 0:
        raise ValueError("Not enough historical data to fit a prior.")

    beta0 = float(np.dot(x, y) / np.dot(x, x))
    resid = y - beta0 * x
    n = len(x)
    R0 = float(np.sum(resid ** 2) / max(n - 1, 1))
    se2_beta = R0 / np.dot(x, x)          # OLS variance of beta estimate
    P0 = float(se2_beta * p0_inflation)

    return Prior(beta0=beta0, P0=P0, R0=R0)


# --------------------------------------------------------------------------- #
# Layer 1 — Online filter
# --------------------------------------------------------------------------- #

@dataclass
class KalmanState:
    beta: float
    P: float
    R: float          # frozen for the session
    mu: float          # forgetting factor, 0 < mu <= 1

    # last-step diagnostics, useful for logging/plots
    z: float = field(default=0.0)
    e: float = field(default=0.0)
    S: float = field(default=0.0)

    def step(self, x: float, y: float) -> None:
        """One filter update. Runs every bar, unconditionally."""
        if x == 0.0:
            # no information this bar; skip gracefully (K would be 0 anyway)
            self.z = 0.0
            self.e = y
            self.S = self.R
            return

        P_minus = self.P / self.mu
        y_hat = self.beta * x
        e = y - y_hat
        K = (P_minus * x) / (x * x * P_minus + self.R)

        self.beta = self.beta + K * e
        self.P = (1.0 - K * x) * P_minus
        S = x * x * self.P + self.R
        self.z = e / np.sqrt(S) if S > 0 else 0.0
        self.e = e
        self.S = S


# --------------------------------------------------------------------------- #
# Layer 2 — Trade state machine
# --------------------------------------------------------------------------- #

@dataclass
class TradeParams:
    entry_z: float = 2.0
    hold_minutes: int = 15
    tp_pct: float = 1.0            # percent, e.g. 1.0 = 1%
    notional: float = 10_000.0
    max_session_loss: float = 0.0  # 0 = disabled; set >0 (in $) to enable
    entry_delay_minutes: int = 1   # signal bar close -> fill N minutes later
    slippage_pct: float = 0.3      # flat entry cost = notional × slippage_pct / 100


@dataclass
class PendingEntry:
    signal_time: pd.Timestamp
    direction: int
    beta_entry: float
    entry_z: float


@dataclass
class OpenPosition:
    direction: int          # +1 = long PAAS/short SLV, -1 = short PAAS/long SLV
    signal_time: pd.Timestamp
    entry_time: pd.Timestamp
    beta_entry: float
    price_a_entry: float
    price_b_entry: float
    shares_a: float
    shares_b: float
    entry_z: float
    entry_slippage_usd: float = 0.0


def _entry_slippage_usd(notional: float, slippage_pct: float) -> float:
    if slippage_pct <= 0:
        return 0.0
    return notional * slippage_pct / 100.0


def _mark_to_market(pos: OpenPosition, price_a: float, price_b: float) -> float:
    """Direct mark-to-market PnL in $ from entry, minus flat entry slippage."""
    d = pos.direction
    pnl_a = d * pos.shares_a * (price_a - pos.price_a_entry)
    pnl_b = -d * pos.shares_b * (price_b - pos.price_b_entry)
    return pnl_a + pnl_b - pos.entry_slippage_usd


def _try_fill_pending_entry(
    *,
    pending: PendingEntry,
    dt: pd.Timestamp,
    price_a: float,
    price_b: float,
    tp: TradeParams,
) -> OpenPosition | None:
    delay = pd.Timedelta(minutes=tp.entry_delay_minutes)
    if dt < pending.signal_time + delay:
        return None
    if price_a <= 0 or price_b <= 0:
        return None

    fill_a, fill_b = price_a, price_b
    shares_a = tp.notional / fill_a
    shares_b = tp.notional * pending.beta_entry / fill_b
    return OpenPosition(
        direction=pending.direction,
        signal_time=pending.signal_time,
        entry_time=dt,
        beta_entry=pending.beta_entry,
        price_a_entry=fill_a,
        price_b_entry=fill_b,
        shares_a=shares_a,
        shares_b=shares_b,
        entry_z=pending.entry_z,
        entry_slippage_usd=_entry_slippage_usd(tp.notional, tp.slippage_pct),
    )


def _trade_legs(
    pos: OpenPosition,
    ticker_a: str,
    ticker_b: str,
    exit_a: float,
    exit_b: float,
) -> dict:
    """Entry/exit leg snapshot for reporting."""
    d = pos.direction
    n_b = pos.shares_b * pos.price_b_entry
    if d == 1:
        entry_a_act, entry_b_act = "BUY", "SELL"
        exit_a_act, exit_b_act = "SELL", "BUY"
        pnl_a = pos.shares_a * (exit_a - pos.price_a_entry)
        pnl_b = pos.shares_b * (pos.price_b_entry - exit_b)
    else:
        entry_a_act, entry_b_act = "SELL", "BUY"
        exit_a_act, exit_b_act = "BUY", "SELL"
        pnl_a = pos.shares_a * (pos.price_a_entry - exit_a)
        pnl_b = pos.shares_b * (exit_b - pos.price_b_entry)

    return {
        "entry": [
            {
                "Ticker": ticker_a,
                "Action": entry_a_act,
                "Price": round(pos.price_a_entry, 4),
                "Notional_usd": round(pos.shares_a * pos.price_a_entry, 2),
                "Shares": round(pos.shares_a, 4),
            },
            {
                "Ticker": ticker_b,
                "Action": entry_b_act,
                "Price": round(pos.price_b_entry, 4),
                "Notional_usd": round(n_b, 2),
                "Shares": round(pos.shares_b, 4),
                "Beta": round(pos.beta_entry, 4),
            },
        ],
        "exit": [
            {
                "Ticker": ticker_a,
                "Action": exit_a_act,
                "Price": round(exit_a, 4),
                "Notional_usd": round(pos.shares_a * exit_a, 2),
                "Shares": round(pos.shares_a, 4),
            },
            {
                "Ticker": ticker_b,
                "Action": exit_b_act,
                "Price": round(exit_b, 4),
                "Notional_usd": round(pos.shares_b * exit_b, 2),
                "Shares": round(pos.shares_b, 4),
            },
        ],
        "PnL_A_usd": round(pnl_a, 2),
        "PnL_B_usd": round(pnl_b, 2),
    }


def simulate_session(
    day: pd.DataFrame,
    td: int,
    ticker_a: str,
    ticker_b: str,
    prior: Prior,
    mu: float,
    tp: TradeParams,
) -> list[dict]:
    """
    Runs Layer 1 (filter) every bar of the session, unconditionally.
    Runs Layer 2 (trade gate) only once the observation window has
    elapsed (bars at/after OBS_END_TIME).
    """
    kf = KalmanState(beta=prior.beta0, P=prior.P0, R=prior.R0, mu=mu)

    obs_cutoff = pd.Timestamp(f"{day['DT'].iloc[0].date()} {OBS_END_TIME}")
    fills: list[dict] = []
    pos: OpenPosition | None = None
    pending: PendingEntry | None = None
    session_pnl = 0.0
    session_halted = False

    day = day.sort_values("DT").reset_index(drop=True)
    last_dt = day["DT"].iloc[-1]

    prev_a, prev_b = None, None
    for _, row in day.iterrows():
        dt = row["DT"]
        price_a = float(row[ticker_a])
        price_b = float(row[ticker_b])
        trading_live = dt >= obs_cutoff and not session_halted

        # ---- Layer 2: fill queued entry at next bar close ----
        if pos is None and pending is not None and trading_live:
            filled = _try_fill_pending_entry(
                pending=pending,
                dt=dt,
                price_a=price_a,
                price_b=price_b,
                tp=tp,
            )
            if filled is not None:
                pos = filled
                pending = None

        if prev_a is None:
            prev_a, prev_b = price_a, price_b
            continue

        x = price_b / prev_b - 1.0
        y = price_a / prev_a - 1.0
        prev_a, prev_b = price_a, price_b

        # ---- Layer 1: filter runs every single bar, no exceptions ----
        kf.step(x, y)

        # ---- Layer 2: manage open position first ----
        if pos is not None:
            pnl = _mark_to_market(pos, price_a, price_b)
            pnl_pct = 100.0 * pnl / tp.notional
            elapsed_min = (dt - pos.entry_time).total_seconds() / 60.0

            reason = None
            if tp.max_session_loss > 0 and session_pnl + pnl <= -tp.max_session_loss:
                reason = "Session loss cap"
            elif pnl_pct >= tp.tp_pct:
                reason = f"TP {pnl_pct:.2f}%"
            elif elapsed_min >= tp.hold_minutes:
                reason = f"Time stop ({tp.hold_minutes}m)"
            elif dt >= last_dt:
                reason = "EOD"

            if reason:
                session_pnl += pnl
                legs = _trade_legs(pos, ticker_a, ticker_b, price_a, price_b)
                fills.append({
                    "Trading date": td,
                    "Direction": "Long PAAS/Short SLV" if pos.direction == 1 else "Short PAAS/Long SLV",
                    "Signal_time": pos.signal_time,
                    "Entry_time": pos.entry_time,
                    "Exit_time": dt,
                    f"Entry_{ticker_a}": legs["entry"][0]["Price"],
                    f"Entry_{ticker_b}": legs["entry"][1]["Price"],
                    f"Exit_{ticker_a}": legs["exit"][0]["Price"],
                    f"Exit_{ticker_b}": legs["exit"][1]["Price"],
                    "Entry_z": round(pos.entry_z, 3),
                    "Beta_entry": round(pos.beta_entry, 4),
                    "Exit_reason": reason,
                    "Slippage_$": round(pos.entry_slippage_usd, 2),
                    "PnL_A_$": legs["PnL_A_usd"],
                    "PnL_B_$": legs["PnL_B_usd"],
                    "PnL_$": round(pnl, 2),
                    "PnL_pct": round(pnl_pct, 4),
                    "_legs": legs,
                })
                pos = None
                if tp.max_session_loss > 0 and session_pnl <= -tp.max_session_loss:
                    session_halted = True

        # ---- Layer 2: queue entry on signal bar (fill arrives next bar) ----
        if pos is None and trading_live and abs(kf.z) > tp.entry_z:
            pending = PendingEntry(
                signal_time=dt,
                direction=1 if kf.z < 0 else -1,
                beta_entry=kf.beta,
                entry_z=kf.z,
            )

    return fills


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def run_backtest(
    panel: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    prior_lookback: int,
    mu: float,
    tp: TradeParams,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_fills: list[dict] = []
    dates = sorted(panel["Trading date"].unique())

    for idx, td in enumerate(dates):
        if idx < 5:
            continue  # not enough history yet for a prior
        hist = panel[panel["Trading date"] < td]
        try:
            prior = fit_prior(hist, ticker_a, ticker_b, prior_lookback)
        except ValueError:
            continue

        day = panel[panel["Trading date"] == td]
        fills = simulate_session(day, int(td), ticker_a, ticker_b, prior, mu, tp)
        all_fills.extend(fills)

    trades_df = pd.DataFrame(all_fills)
    if trades_df.empty:
        return trades_df, pd.DataFrame()

    trades_df.insert(0, "Trade_id", range(1, len(trades_df) + 1))
    for col in ("Signal_time", "Entry_time", "Exit_time"):
        trades_df[col] = pd.to_datetime(trades_df[col]).dt.strftime("%Y-%m-%d %H:%M")

    daily_df = (
        trades_df.groupby("Trading date", as_index=False)
        .agg(Trades=("PnL_$", "count"), Session_PnL=("PnL_$", "sum"))
        .rename(columns={"Session_PnL": "Session_PnL_$"})
        .sort_values("Trading date")
    )
    daily_df["Cum_PnL_$"] = daily_df["Session_PnL_$"].cumsum()
    return trades_df, daily_df


def summary_stats(trades_df: pd.DataFrame, daily_df: pd.DataFrame, tp: TradeParams, mu: float) -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame({"Metric": ["Trades"], "Value": [0]})

    n = len(trades_df)
    wins = (trades_df["PnL_$"] > 0).sum()
    rows = [
        ("Trades", n),
        ("Sessions traded", daily_df["Trading date"].nunique()),
        ("Total PnL ($)", round(trades_df["PnL_$"].sum(), 2)),
        ("Avg PnL/trade ($)", round(trades_df["PnL_$"].mean(), 2)),
        ("Win rate (%)", round(100 * wins / n, 1)),
        ("TP exits", int(trades_df["Exit_reason"].str.startswith("TP").sum())),
        ("Time-stop exits", int(trades_df["Exit_reason"].str.startswith("Time stop").sum())),
        ("EOD exits", int((trades_df["Exit_reason"] == "EOD").sum())),
        ("Entry z threshold", tp.entry_z),
        ("Hold (min)", tp.hold_minutes),
        ("Take profit (%)", tp.tp_pct),
        ("Entry delay (min)", tp.entry_delay_minutes),
        ("Entry slippage (%)", tp.slippage_pct),
        ("Entry slippage ($)", round(_entry_slippage_usd(tp.notional, tp.slippage_pct), 2)),
        ("Forgetting factor (mu)", mu),
        ("Notional/leg", tp.notional),
        ("Max session loss", tp.max_session_loss if tp.max_session_loss > 0 else "off"),
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
        "Direction",
        "Signal_time",
        "Entry_time",
        "Exit_time",
        "Entry_z",
        "Beta_entry",
        "Exit_reason",
        "Slippage_$",
        "PnL_$",
        "PnL_pct",
    ]
    parts = [
        "<table id='trades-table' class='data-table interactive-table'>",
        "<thead><tr>",
        *[f"<th>{c}</th>" for c in cols],
        "</tr></thead><tbody>",
    ]
    for _, row in trades_df.iterrows():
        tid = int(row["Trade_id"])
        td = int(row["Trading date"])
        parts.append(
            f"<tr class='trade-row' data-trade='{tid}' data-td='{td}' tabindex='0'>"
        )
        for c in cols:
            val = row[c]
            if c in ("PnL_$", "Slippage_$") and pd.notna(val):
                text = f"{float(val):,.2f}"
            elif c == "PnL_pct" and pd.notna(val):
                text = f"{float(val):,.4f}"
            elif c in ("Entry_z", "Beta_entry") and pd.notna(val):
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
            "Direction": row["Direction"],
            "Signal_time": row["Signal_time"],
            "Entry_time": row["Entry_time"],
            "Exit_time": row["Exit_time"],
            f"Entry_{ticker_a}": float(row[f"Entry_{ticker_a}"]),
            f"Entry_{ticker_b}": float(row[f"Entry_{ticker_b}"]),
            f"Exit_{ticker_a}": float(row[f"Exit_{ticker_a}"]),
            f"Exit_{ticker_b}": float(row[f"Exit_{ticker_b}"]),
            "Entry_z": float(row["Entry_z"]),
            "Beta_entry": float(row["Beta_entry"]),
            "Exit_reason": row["Exit_reason"],
            "Slippage_usd": float(row["Slippage_$"]),
            "PnL_A_usd": float(row["PnL_A_$"]),
            "PnL_B_usd": float(row["PnL_B_$"]),
            "PnL_usd": float(row["PnL_$"]),
            "PnL_pct": float(row["PnL_pct"]),
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


def build_report_html(summary: pd.DataFrame, trades_df: pd.DataFrame, daily_df: pd.DataFrame, ticker_a: str, ticker_b: str) -> str:
    fig = make_subplots(rows=2, cols=1, row_heights=[0.45, 0.55], vertical_spacing=0.08)
    if not daily_df.empty:
        x = _trading_dates_for_plot(daily_df)
        fig.add_trace(go.Bar(x=x, y=daily_df["Session_PnL_$"], name="Daily PnL"), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=daily_df["Cum_PnL_$"], mode="lines", name="Cum PnL"), row=2, col=1)
    fig.update_layout(template="plotly_white", height=700, title=f"Kalman hedge-ratio pairs — {ticker_a} vs {ticker_b}")
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
<html><head><meta charset="utf-8"><title>Kalman Pairs Backtest</title>
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
.interactive-table tbody tr.day-match {{ background: #f0f7ff; }}
#explainer {{ background: #f7f9fc; border-left: 4px solid #0066cc; padding: 0.75rem 1rem; margin: 0.5rem 0 1rem; font-size: 13px; }}
#day-trades-panel table {{ margin-top: 0.5rem; }}
a.reset-link {{ font-size: 13px; margin-left: 0.5rem; }}
</style></head><body>
<h1>Kalman hedge-ratio pairs — {ticker_a} vs {ticker_b}</h1>
<h2>Summary</h2>{summary.to_html(index=False, border=0)}
<div>{chart}</div>
<h2>Daily PnL</h2>
<p>Click a day row or chart bar to show all trades for that session. Click a trade for detail.</p>
{_daily_table_html(daily_df)}
<h2>Trades <a href="#" class="reset-link" id="show-all-trades">(show all)</a></h2>
{_trades_table_html(trades_df)}
<h2 id="detail-title">Trade detail</h2>
<div id="explainer">
  Kalman z-score pairs on 1-min bars. Signal bar → entry next bar close. Flat slippage deducted from PnL.
</div>
<p id="trade-meta">Select a trade or day.</p>
<div id="day-trades-panel"></div>
<div id="trade-detail-panel"></div>
<script>
const TRADES = {_trades_json(trades_df, ticker_a, ticker_b)};
const TRADES_BY_TD = {_trades_by_td_json(trades_df)};
const TD_LABELS = {json.dumps(td_dates)};
const TA = {json.dumps(ticker_a)};
const TB = {json.dumps(ticker_b)};

function formatMoney(v) {{
  return (v >= 0 ? '' : '-') + '$' + Math.abs(v).toFixed(2);
}}

function legTable(title, timeLabel, legs) {{
  let html = '<h3>' + title + ' (' + timeLabel + ')</h3>';
  html += '<table class="data-table"><thead><tr>'
    + '<th>Ticker</th><th>Action</th><th>Fill price</th><th>Notional $</th><th>Shares</th><th>β</th>'
    + '</tr></thead><tbody>';
  for (const leg of legs) {{
    html += '<tr><td>' + leg.Ticker + '</td><td>' + leg.Action + '</td><td>' + leg.Price
      + '</td><td>' + leg.Notional_usd + '</td><td>' + leg.Shares
      + '</td><td>' + (leg.Beta != null ? leg.Beta : '—') + '</td></tr>';
  }}
  return html + '</tbody></table>';
}}

function renderTradeDetail(tradeId) {{
  const t = TRADES[String(tradeId)];
  if (!t) return;
  document.getElementById('detail-title').textContent = 'Trade #' + t.Trade_id;
  document.getElementById('trade-meta').textContent =
    t.Direction + ' | signal ' + t.Signal_time + ' → entry ' + t.Entry_time
    + ' → exit ' + t.Exit_time + ' (' + t.Exit_reason + ')'
    + ' | z=' + t.Entry_z.toFixed(3) + ' β=' + t.Beta_entry.toFixed(4)
    + ' | slippage ' + formatMoney(-t.Slippage_usd) + ' | PnL ' + formatMoney(t.PnL_usd)
    + ' (' + t.PnL_pct.toFixed(3) + '%)';
  let html = legTable('Entry fill', t.Entry_time, t.legs.entry);
  html += legTable('Exit fill', t.Exit_time, t.legs.exit);
  html += '<table class="data-table"><thead><tr><th>Leg</th><th>PnL $</th></tr></thead><tbody>'
    + '<tr><td>' + TA + '</td><td>' + formatMoney(t.PnL_A_usd) + '</td></tr>'
    + '<tr><td>' + TB + '</td><td>' + formatMoney(t.PnL_B_usd) + '</td></tr>'
    + '<tr><td>Slippage</td><td>' + formatMoney(-t.Slippage_usd) + '</td></tr>'
    + '<tr><td><b>Total</b></td><td><b>' + formatMoney(t.PnL_usd) + '</b></td></tr>'
    + '</tbody></table>';
  document.getElementById('trade-detail-panel').innerHTML = html;
}}

function renderDayTrades(td, focusTradeId) {{
  const ids = TRADES_BY_TD[String(td)] || [];
  const label = TD_LABELS[String(td)] || td;
  document.getElementById('detail-title').textContent =
    ids.length ? ('Trades on ' + label + ' (' + ids.length + ')') : ('Trades on ' + label);
  let dayPnl = 0;
  let html = '<table class="data-table"><thead><tr>'
    + '<th>#</th><th>Direction</th><th>Entry</th><th>Exit</th><th>Reason</th><th>PnL $</th>'
    + '</tr></thead><tbody>';
  ids.forEach(id => {{
    const t = TRADES[String(id)];
    dayPnl += t.PnL_usd;
    const sel = String(id) === String(focusTradeId) ? ' class="selected"' : '';
    html += '<tr class="day-trade-row"' + sel + ' data-trade="' + id + '" style="cursor:pointer">'
      + '<td>#' + t.Trade_id + '</td>'
      + '<td>' + t.Direction + '</td>'
      + '<td>' + t.Entry_time + '</td>'
      + '<td>' + t.Exit_time + '</td>'
      + '<td>' + t.Exit_reason + '</td>'
      + '<td>' + formatMoney(t.PnL_usd) + '</td>'
      + '</tr>';
  }});
  html += '</tbody></table>';
  if (ids.length) {{
    html += '<p><b>Day total:</b> ' + formatMoney(dayPnl) + '</p>';
  }} else {{
    html = '<p>No trades for this date.</p>';
  }}
  document.getElementById('day-trades-panel').innerHTML = html;
  document.getElementById('trade-meta').textContent =
    ids.length ? (ids.length + ' trade(s) on ' + label) : 'No trades for this date.';
  document.querySelectorAll('.day-trade-row').forEach(row => {{
    row.addEventListener('click', () => selectTrade(row.dataset.trade, false));
  }});
  if (focusTradeId) renderTradeDetail(focusTradeId);
  else document.getElementById('trade-detail-panel').innerHTML = '';
}}

function setTradeFilter(td) {{
  document.querySelectorAll('.trade-row').forEach(tr => {{
    const match = !td || tr.dataset.td === String(td);
    tr.style.display = match ? '' : 'none';
    tr.classList.toggle('day-match', !!td && match);
    tr.classList.remove('selected');
  }});
}}

function setDailySelection(td) {{
  document.querySelectorAll('.daily-row').forEach(tr => {{
    tr.classList.toggle('selected', !!td && tr.dataset.td === String(td));
  }});
}}

function selectDaily(td, focusTradeId) {{
  setDailySelection(td);
  setTradeFilter(td);
  renderDayTrades(td, focusTradeId);
  if (focusTradeId) {{
    document.querySelectorAll('.trade-row').forEach(tr => {{
      tr.classList.toggle('selected', tr.dataset.trade === String(focusTradeId));
    }});
  }}
  document.getElementById('day-trades-panel').scrollIntoView({{ behavior: 'smooth', block: 'nearest' }});
}}

function selectTrade(tradeId, scroll) {{
  const t = TRADES[String(tradeId)];
  if (!t) return;
  selectDaily(t.Trading_date, tradeId);
  if (scroll !== false) {{
    document.getElementById('trade-detail-panel').scrollIntoView({{ behavior: 'smooth', block: 'nearest' }});
  }}
}}

function showAllTrades() {{
  setDailySelection(null);
  setTradeFilter(null);
  document.getElementById('detail-title').textContent = 'Trade detail';
  document.getElementById('trade-meta').textContent = 'Showing all trades.';
  document.getElementById('day-trades-panel').innerHTML = '';
  document.getElementById('trade-detail-panel').innerHTML = '';
}}

document.querySelectorAll('.trade-row').forEach(tr => {{
  tr.addEventListener('click', () => selectTrade(tr.dataset.trade));
}});
document.querySelectorAll('.daily-row').forEach(tr => {{
  tr.addEventListener('click', () => selectDaily(tr.dataset.td, null));
}});
document.getElementById('show-all-trades').addEventListener('click', e => {{
  e.preventDefault();
  showAllTrades();
}});

const chartEl = document.getElementById('pnl-chart');
if (chartEl) {{
  chartEl.on('plotly_click', ev => {{
    if (!ev.points || !ev.points.length) return;
    const d = new Date(ev.points[0].x);
    const td = String(d.getFullYear())
      + String(d.getMonth() + 1).padStart(2, '0')
      + String(d.getDate()).padStart(2, '0');
    selectDaily(td, null);
  }});
}}

if (Object.keys(TRADES).length) {{
  const lastId = Object.keys(TRADES).sort((a, b) => Number(a) - Number(b)).pop();
  selectTrade(lastId);
}}
</script>
</body></html>"""


def _parse_args() -> argparse.Namespace:
    today = datetime.now() - timedelta(days=3)
    p = argparse.ArgumentParser(description="Kalman hedge-ratio pairs backtest.")
    p.add_argument("--ticker-a", default="PAAS")
    p.add_argument("--ticker-b", default="SLV")
    p.add_argument("--start", type=int, default=int((today - timedelta(days=180)).strftime("%Y%m%d")))
    p.add_argument("--end", type=int, default=int(today.strftime("%Y%m%d")))
    p.add_argument("--entry-z", type=float, default=2.0)
    p.add_argument("--hold-minutes", type=int, default=15)
    p.add_argument("--tp-pct", type=float, default=1.0)
    p.add_argument("--mu", type=float, default=0.99, help="Forgetting factor (0<mu<=1); effective memory ~ 1/(1-mu) minutes")
    p.add_argument("--prior-lookback", type=int, default=60, help="Trading days used to fit beta0/P0/R0")
    p.add_argument("--notional", type=float, default=10_000.0)
    p.add_argument(
        "--entry-delay-minutes",
        type=int,
        default=1,
        help="Minutes after signal bar close before entry fill (1 = next bar close)",
    )
    p.add_argument(
        "--slippage-pct",
        type=float,
        default=0.3,
        help="Flat entry cost as %% of notional (e.g. 0.3 -> $30 on $10k), deducted from PnL",
    )
    p.add_argument("--max-session-loss", type=float, default=0.0, help="0 = disabled")
    p.add_argument("--data-folder", default=None)
    p.add_argument("--output", default="kalman_pairs_backtest.html")
    p.add_argument("--no-browser", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    tp = TradeParams(
        entry_z=args.entry_z,
        hold_minutes=args.hold_minutes,
        tp_pct=args.tp_pct,
        notional=args.notional,
        max_session_loss=args.max_session_loss,
        entry_delay_minutes=args.entry_delay_minutes,
        slippage_pct=args.slippage_pct,
    )
    folder = args.data_folder or Paths().min_folder

    print(f"Loading {args.ticker_a}/{args.ticker_b} …")
    panel = load_pair_panel(args.ticker_a, args.ticker_b, folder, args.start, args.end)

    trades_df, daily_df = run_backtest(
        panel, args.ticker_a, args.ticker_b, args.prior_lookback, args.mu, tp
    )
    summary = summary_stats(trades_df, daily_df, tp, args.mu)
    print("\n=== Summary ===")
    print(summary.to_string(index=False))

    out = args.output
    with open(out, "w", encoding="utf-8") as f:
        f.write(build_report_html(summary, trades_df, daily_df, args.ticker_a, args.ticker_b))
    print(f"\nReport: {out}")

    if not args.no_browser:
        webbrowser.open(f"file://{os.path.abspath(out)}")


if __name__ == "__main__":
    main()