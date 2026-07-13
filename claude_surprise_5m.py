#!/usr/bin/env python3
"""
Kalman-filter hedge-ratio pairs trade on 1-min bars (SLV / PAAS).

Three layers, kept strictly separate:

  Layer 0 — Offline prior (once/day, before open)
      OLS  y = beta*x + eps  over last N sessions of **intraday-only**
      1-min returns (no overnight jumps in the sample) -> beta0, P0, R0

  Layer 1 — Online filter (every minute from 9:30; observation until 10:00)
      First step: optional prior_close -> 9:30 open gap (once)
      Then minute returns; R0 fixed or slow EWMA update (--adaptive-r)

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
      FLAT        -> on |z| > entry_z at bar *t* close (return t−1→t, β from t−1):
                     ``--same-bar-entry``: fill at bar *t* close (minimal delay)
                     else queue fill ``--entry-delay-minutes`` later (default next bar)
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


def _pool_intraday_returns(
    sub: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Within-session 1-min returns only; no overnight bars in the sample."""
    xs: list[float] = []
    ys: list[float] = []

    for _, group in sub.groupby("Trading date"):
        g = group.sort_values("DT")
        pa = g[ticker_a].to_numpy(dtype=float)
        pb = g[ticker_b].to_numpy(dtype=float)
        if len(pa) < 2:
            continue
        mask = (pa[:-1] > 0) & (pb[:-1] > 0) & (pa[1:] > 0) & (pb[1:] > 0)
        if not mask.any():
            continue
        ys.extend((pa[1:][mask] / pa[:-1][mask]) - 1.0)
        xs.extend((pb[1:][mask] / pb[:-1][mask]) - 1.0)

    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def fit_prior(
    hist: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    lookback_days: int,
    p0_inflation: float = 5.0,
) -> Prior:
    """
    OLS y = beta*x + eps over trailing ``lookback_days`` sessions of
    **intraday-only** minute returns. ``hist`` must exclude the live session.
    """
    days = sorted(hist["Trading date"].unique())[-lookback_days:]
    sub = hist[hist["Trading date"].isin(days)].sort_values("DT")
    x, y = _pool_intraday_returns(sub, ticker_a, ticker_b)

    if len(x) < 30 or np.var(x) == 0:
        raise ValueError("Not enough historical data to fit a prior.")

    beta0 = float(np.dot(x, y) / np.dot(x, x))
    resid = y - beta0 * x
    n = len(x)
    R0 = float(np.sum(resid ** 2) / max(n - 1, 1))
    se2_beta = R0 / np.dot(x, x)
    P0 = float(se2_beta * p0_inflation)

    return Prior(beta0=beta0, P0=P0, R0=R0)


def _prior_close_by_date(
    panel: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
) -> dict[int, tuple[float, float]]:
    """Map session *t* -> (prior_close_A, prior_close_B) from session *t-1*."""
    closes = (
        panel.sort_values("DT")
        .groupby("Trading date")[[ticker_a, ticker_b]]
        .last()
    )
    dates = [int(d) for d in closes.index]
    out: dict[int, tuple[float, float]] = {}
    for i in range(1, len(dates)):
        prev_td, cur_td = dates[i - 1], dates[i]
        pa = float(closes.loc[prev_td, ticker_a])
        pb = float(closes.loc[prev_td, ticker_b])
        if pa > 0 and pb > 0:
            out[cur_td] = (pa, pb)
    return out


# --------------------------------------------------------------------------- #
# Layer 1 — Online filter
# --------------------------------------------------------------------------- #

@dataclass
class KalmanState:
    beta: float
    P: float
    R: float
    mu: float
    adaptive_r: bool = False
    r_lambda: float = 0.98

    z: float = field(default=0.0)
    e: float = field(default=0.0)
    S: float = field(default=0.0)

    def step(self, x: float, y: float) -> None:
        """One filter update. Runs every bar, unconditionally."""
        if x == 0.0:
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

        if self.adaptive_r:
            lam = self.r_lambda
            self.R = lam * self.R + (1.0 - lam) * (e * e)


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
    entry_delay_minutes: int = 1   # delayed mode: minutes after signal bar close
    slippage_pct: float = 0.3      # flat entry cost = notional × slippage_pct / 100
    same_bar_entry: bool = False   # True = signal and fill at same bar close
    include_open_gap: bool = True  # inject prior_close -> 9:30 open once
    adaptive_r: bool = True        # slow EWMA update of R from innovations
    r_lambda: float = 0.98         # R_t = λ R_{t-1} + (1-λ) e_t²
    momentum: bool = False           # True = ride z; False = fade z (mean-revert)


def signal_direction(z: float, momentum: bool) -> int:
    """+1 = long PAAS/short SLV; −1 = short PAAS/long SLV."""
    if momentum:
        return -1 if z < 0 else 1
    return 1 if z < 0 else -1


def signal_mode_label(momentum: bool) -> str:
    return "momentum" if momentum else "mean_revert"


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


def _open_position_from_signal(
    *,
    signal_time: pd.Timestamp,
    direction: int,
    beta_entry: float,
    entry_z: float,
    price_a: float,
    price_b: float,
    tp: TradeParams,
) -> OpenPosition | None:
    if price_a <= 0 or price_b <= 0:
        return None
    shares_a = tp.notional / price_a
    shares_b = tp.notional * beta_entry / price_b
    return OpenPosition(
        direction=direction,
        signal_time=signal_time,
        entry_time=signal_time,
        beta_entry=beta_entry,
        price_a_entry=price_a,
        price_b_entry=price_b,
        shares_a=shares_a,
        shares_b=shares_b,
        entry_z=entry_z,
        entry_slippage_usd=_entry_slippage_usd(tp.notional, tp.slippage_pct),
    )


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


def _maybe_enter(
    *,
    pos: OpenPosition | None,
    pending: PendingEntry | None,
    dt: pd.Timestamp,
    price_a: float,
    price_b: float,
    kf: KalmanState,
    trading_live: bool,
    tp: TradeParams,
) -> tuple[OpenPosition | None, PendingEntry | None]:
    if pos is not None or not trading_live or abs(kf.z) <= tp.entry_z:
        return pos, pending

    direction = signal_direction(kf.z, tp.momentum)
    if tp.same_bar_entry:
        opened = _open_position_from_signal(
            signal_time=dt,
            direction=direction,
            beta_entry=kf.beta,
            entry_z=kf.z,
            price_a=price_a,
            price_b=price_b,
            tp=tp,
        )
        return (opened, None) if opened is not None else (pos, pending)

    pending = PendingEntry(
        signal_time=dt,
        direction=direction,
        beta_entry=kf.beta,
        entry_z=kf.z,
    )
    return pos, pending


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
    prior_close: tuple[float, float] | None = None,
) -> list[dict]:
    """
    Runs Layer 1 (filter) every bar of the session, unconditionally.
    Runs Layer 2 (trade gate) only once the observation window has
    elapsed (bars at/after OBS_END_TIME).
    """
    kf = KalmanState(
        beta=prior.beta0,
        P=prior.P0,
        R=prior.R0,
        mu=mu,
        adaptive_r=tp.adaptive_r,
        r_lambda=tp.r_lambda,
    )

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

        # ---- Layer 2: delayed entry fill (skipped in same-bar mode) ----
        if not tp.same_bar_entry and pos is None and pending is not None and trading_live:
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
            if (
                tp.include_open_gap
                and prior_close is not None
                and prior_close[0] > 0
                and prior_close[1] > 0
            ):
                x_gap = price_b / prior_close[1] - 1.0
                y_gap = price_a / prior_close[0] - 1.0
                kf.step(x_gap, y_gap)
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
                    "Signal_mode": signal_mode_label(tp.momentum),
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

        # ---- Layer 2: entry signal at bar t close (after kf.step on return t-1→t) ----
        pos, pending = _maybe_enter(
            pos=pos,
            pending=pending,
            dt=dt,
            price_a=price_a,
            price_b=price_b,
            kf=kf,
            trading_live=trading_live,
            tp=tp,
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
    prior_closes = _prior_close_by_date(panel, ticker_a, ticker_b)

    for idx, td in enumerate(dates):
        if idx < 5:
            continue  # not enough history yet for a prior
        hist = panel[panel["Trading date"] < td]
        try:
            prior = fit_prior(hist, ticker_a, ticker_b, prior_lookback)
        except ValueError:
            continue

        day = panel[panel["Trading date"] == td]
        fills = simulate_session(
            day,
            int(td),
            ticker_a,
            ticker_b,
            prior,
            mu,
            tp,
            prior_close=prior_closes.get(int(td)),
        )
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
        ("Entry delay (min)", "same bar" if tp.same_bar_entry else tp.entry_delay_minutes),
        ("Same-bar entry", "yes" if tp.same_bar_entry else "no"),
        ("Signal mode", signal_mode_label(tp.momentum)),
        ("Entry slippage (%)", tp.slippage_pct),
        ("Entry slippage ($)", round(_entry_slippage_usd(tp.notional, tp.slippage_pct), 2)),
        ("Forgetting factor (mu)", mu),
        ("Prior", "intraday-only minute returns"),
        ("Open gap at 9:30", "yes" if tp.include_open_gap else "no"),
        ("Adaptive R", "yes" if tp.adaptive_r else "no"),
        ("R EWMA lambda", tp.r_lambda if tp.adaptive_r else "n/a"),
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


def build_report_html(
    summary: pd.DataFrame,
    trades_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    tp: TradeParams,
) -> str:
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
  Prior: intraday-only 1-min returns (no overnight in OLS).<br>
  9:30: one prior_close→open gap into Kalman; then minute returns until 10:00 warm-up, then trading.<br>
  R: {"EWMA λ=" + str(tp.r_lambda) if tp.adaptive_r else "fixed R₀ all day"}.<br>
  Signal mode: {"momentum (ride z)" if tp.momentum else "mean_revert (fade z)"}.<br>
  Entry: {"same bar close" if tp.same_bar_entry else f"signal → fill {tp.entry_delay_minutes} min later"}.
  Exit: same bar when TP/time/EOD hits.
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
        "--same-bar-entry",
        action="store_true",
        help="Signal and fill at same bar close (return t-1→t); minimal execution delay",
    )
    p.add_argument(
        "--entry-delay-minutes",
        type=int,
        default=1,
        help="Delayed mode only: minutes after signal bar close before entry fill",
    )
    p.add_argument(
        "--slippage-pct",
        type=float,
        default=0.3,
        help="Flat entry cost as %% of notional (e.g. 0.3 -> $30 on $10k), deducted from PnL",
    )
    p.add_argument(
        "--no-open-gap",
        action="store_true",
        help="Disable prior_close -> 9:30 open gap injection",
    )
    p.add_argument(
        "--no-adaptive-r",
        action="store_true",
        help="Keep R fixed at R₀ all session (disable EWMA innovation update)",
    )
    p.add_argument(
        "--r-lambda",
        type=float,
        default=0.98,
        help="EWMA decay for adaptive R (higher = slower update)",
    )
    p.add_argument(
        "--momentum",
        action="store_true",
        help="Ride z (momentum) instead of fading z (mean-revert)",
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
        same_bar_entry=args.same_bar_entry,
        include_open_gap=not args.no_open_gap,
        adaptive_r=not args.no_adaptive_r,
        r_lambda=args.r_lambda,
        momentum=args.momentum,
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
        f.write(
            build_report_html(
                summary, trades_df, daily_df, args.ticker_a, args.ticker_b, tp
            )
        )
    print(f"\nReport: {out}")

    if not args.no_browser:
        webbrowser.open(f"file://{os.path.abspath(out)}")


if __name__ == "__main__":
    main()