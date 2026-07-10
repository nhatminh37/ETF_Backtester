#!/usr/bin/env python3
"""
Daily overreaction backtests.

**surprise** (default) — day −1 actual vs expected:
  expected = β_signal × r_SLV,  actual = r_PAAS  (prior session daily returns).
  surprise = actual − expected.
  Trade next day if |surprise| > threshold (mean-revert on PAAS):
    surprise > +threshold → short PAAS;  surprise < −threshold → long PAAS.

**quadrant** — spread size × PAAS direction → hedged spread (``--hedge-style``):
  ``beta`` (default): SELL/BUY A plus opposite β_trade×B ($10k / $10k×β).
  ``equal``: dollar-neutral — $10k on **both** legs (no β sizing).

Enter 9:30; exit EOD or session loss cap (``--max-session-loss``; does **not**
change how many signals fire — only when the position is closed).

Usage::

    python backtest_daily_overreact.py --trade-mode surprise --threshold 3.0
    python backtest_daily_overreact.py --trade-mode quadrant --hedge-style equal
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
TRADE_BETA_OFFSET_START = 20  # day -20
TRADE_BETA_OFFSET_END = 1  # day -1 (inclusive)


@dataclass
class DailyOverreactParams:
    threshold_pct: float = 3.0
    notional: float = 10_000.0
    max_session_loss: float = 100.0
    trade_mode: str = "surprise"  # "surprise" | "quadrant"
    hedge_style: str = "beta"  # "beta" | "equal" (quadrant spread legs only)


def _b_leg_notional(notional: float, beta: float, hedge_style: str) -> float:
    return notional if hedge_style == "equal" else notional * beta


def _spread_leg_breakdown(
    side: str,
    notional: float,
    beta: float,
    ticker_a: str,
    ticker_b: str,
    price_a_entry: float,
    price_b_entry: float,
    price_a_exit: float,
    price_b_exit: float,
    *,
    hedge_style: str = "beta",
) -> dict:
    """Spread legs: long = BUY A + SELL B; short = SELL A + BUY B (B sized by hedge_style)."""
    n_a = notional
    n_b = _b_leg_notional(notional, beta, hedge_style)
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
                "Leg": ticker_a,
                "Action": entry_a_action,
                "Price": round(price_a_entry, 4),
                "Notional_usd": round(n_a, 2),
                "Shares": round(sh_a, 4),
            },
            {
                "Leg": ticker_b,
                "Action": entry_b_action,
                "Price": round(price_b_entry, 4),
                "Hedge": hedge_style,
                "Beta": round(beta, 4) if hedge_style == "beta" else None,
                "Notional_usd": round(n_b, 2),
                "Shares": round(sh_b, 4),
            },
        ],
        "exit": [
            {
                "Leg": ticker_a,
                "Action": exit_a_action,
                "Price": round(price_a_exit, 4),
                "Notional_usd": round(sh_a * price_a_exit, 2),
                "Shares": round(sh_a, 4),
            },
            {
                "Leg": ticker_b,
                "Action": exit_b_action,
                "Price": round(price_b_exit, 4),
                "Notional_usd": round(sh_b * price_b_exit, 2),
                "Shares": round(sh_b, 4),
            },
        ],
        "PnL_A_usd": round(pnl_a, 2),
        "PnL_B_usd": round(pnl_b, 2),
        "PnL_usd": round(pnl_a + pnl_b, 2),
    }


def _spread_position_pnl_usd(
    side: str,
    notional: float,
    beta: float,
    ticker_a: str,
    ticker_b: str,
    entry_pa: float,
    entry_pb: float,
    mark_pa: float,
    mark_pb: float,
    *,
    hedge_style: str = "beta",
) -> float:
    return _spread_leg_breakdown(
        side,
        notional,
        beta,
        ticker_a,
        ticker_b,
        entry_pa,
        entry_pb,
        mark_pa,
        mark_pb,
        hedge_style=hedge_style,
    )["PnL_usd"]


def _scale_spread_leg_pnl(legs: dict, capped_total: float) -> dict:
    raw = legs["PnL_usd"]
    if raw == 0 or abs(raw - capped_total) < 1e-9:
        out = dict(legs)
        out["PnL_usd"] = round(capped_total, 2)
        return out
    scale = capped_total / raw
    out = dict(legs)
    out["PnL_A_usd"] = round(legs["PnL_A_usd"] * scale, 2)
    out["PnL_B_usd"] = round(legs["PnL_B_usd"] * scale, 2)
    out["PnL_usd"] = round(capped_total, 2)
    for leg_list in ("entry", "exit"):
        scaled = []
        for leg in legs[leg_list]:
            item = dict(leg)
            item["Notional_usd"] = round(item["Notional_usd"] * scale, 2)
            item["Shares"] = round(item["Shares"] * scale, 4)
            scaled.append(item)
        out[leg_list] = scaled
    return out


def _leg_breakdown(
    side: str,
    notional: float,
    ticker: str,
    price_entry: float,
    price_exit: float,
) -> dict:
    """Single-leg book: $notional long or short one ticker."""
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


def _paas_direction_label(r_a: float) -> str | None:
    if r_a > 0:
        return "up"
    if r_a < 0:
        return "down"
    return None


def _spread_leg_label(ticker_b: str, hedge_style: str) -> str:
    return ticker_b if hedge_style == "equal" else f"β×{ticker_b}"


def resolve_quadrant_trade(
    diff_ret: float,
    r_a: float,
    threshold: float,
    ticker_a: str,
    ticker_b: str,
    *,
    hedge_style: str = "beta",
) -> dict | None:
    """
    Map (spread, PAAS direction) → β-hedged spread side and leg labels.
    """
    paas_dir = _paas_direction_label(r_a)
    if paas_dir is None:
        return None

    b_leg = _spread_leg_label(ticker_b, hedge_style)

    if diff_ret > threshold:
        spread_regime = "large_spread"
        side = "short"
        leg_txt = f"SELL {ticker_a}, BUY {b_leg}"
        if paas_dir == "up":
            return {
                "side": side,
                "hedge_type": "spread",
                "trade_ticker": f"{ticker_a}-{ticker_b}",
                "signal_flag": "large_spread_paas_up",
                "trade_rule": f"PAAS overbought → short spread: {leg_txt}",
                "spread_regime": spread_regime,
                "paas_dir": paas_dir,
            }
        return {
            "side": side,
            "hedge_type": "spread",
            "trade_ticker": f"{ticker_a}-{ticker_b}",
            "signal_flag": "large_spread_paas_down",
            "trade_rule": f"SLV oversold → short spread: {leg_txt}",
            "spread_regime": spread_regime,
            "paas_dir": paas_dir,
        }

    if diff_ret < -threshold:
        spread_regime = "small_spread"
        side = "long"
        leg_txt = f"BUY {ticker_a}, SELL {b_leg}"
        if paas_dir == "up":
            return {
                "side": side,
                "hedge_type": "spread",
                "trade_ticker": f"{ticker_a}-{ticker_b}",
                "signal_flag": "small_spread_paas_up",
                "trade_rule": f"SLV overbought → long spread: {leg_txt}",
                "spread_regime": spread_regime,
                "paas_dir": paas_dir,
            }
        return {
            "side": side,
            "hedge_type": "spread",
            "trade_ticker": f"{ticker_a}-{ticker_b}",
            "signal_flag": "small_spread_paas_down",
            "trade_rule": f"PAAS oversold → long spread: {leg_txt}",
            "spread_regime": spread_regime,
            "paas_dir": paas_dir,
        }

    return None


def resolve_surprise_trade(
    surprise: float,
    threshold: float,
    ticker_a: str,
) -> dict | None:
    """
    Day −1 actual minus expected (β×r_SLV) → mean-revert PAAS next session.
    """
    if surprise > threshold:
        return {
            "side": "short",
            "hedge_type": "single",
            "trade_ticker": ticker_a,
            "signal_flag": "surprise_positive",
            "spread_regime": "actual_above_expected",
            "paas_dir": None,
            "trade_rule": (
                f"actual > expected by >{threshold * 100:.2f}% → short {ticker_a}"
            ),
        }
    if surprise < -threshold:
        return {
            "side": "long",
            "hedge_type": "single",
            "trade_ticker": ticker_a,
            "signal_flag": "surprise_negative",
            "spread_regime": "actual_below_expected",
            "paas_dir": None,
            "trade_rule": (
                f"actual < expected by >{threshold * 100:.2f}% → long {ticker_a}"
            ),
        }
    return None


def resolve_trade_plan(
    mode: str,
    surprise: float,
    r_a: float,
    threshold: float,
    ticker_a: str,
    ticker_b: str,
    *,
    hedge_style: str = "beta",
) -> dict | None:
    if mode == "quadrant":
        return resolve_quadrant_trade(
            surprise, r_a, threshold, ticker_a, ticker_b, hedge_style=hedge_style
        )
    if mode == "surprise":
        return resolve_surprise_trade(surprise, threshold, ticker_a)
    raise ValueError(f"trade_mode must be 'surprise' or 'quadrant', got {mode!r}")


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
        trade_beta_dates = dates[i - TRADE_BETA_OFFSET_START : i - TRADE_BETA_OFFSET_END + 1]
        signal_date = dates[i - 1]

        if len(signal_beta_dates) != SIGNAL_BETA_OFFSET_START - SIGNAL_BETA_OFFSET_END + 1:
            continue
        if len(trade_beta_dates) != TRADE_BETA_OFFSET_START - TRADE_BETA_OFFSET_END + 1:
            continue

        win_signal = daily_ret.loc[signal_beta_dates]
        win_trade = daily_ret.loc[trade_beta_dates]
        beta_signal = ols_beta(win_signal[ticker_a], win_signal[ticker_b])
        beta_trade = ols_beta(win_trade[ticker_a], win_trade[ticker_b])
        if pd.isna(beta_signal) or pd.isna(beta_trade):
            continue

        r_a = float(daily_ret.loc[signal_date, ticker_a])
        r_b = float(daily_ret.loc[signal_date, ticker_b])
        expected = beta_signal * r_b
        surprise = r_a - expected

        rows.append(
            {
                "Trading date": td,
                "Signal_date": signal_date,
                "beta_signal": float(beta_signal),
                "beta_trade": float(beta_trade),
                "Signal_actual_pct": round(r_a * 100, 4),
                "Signal_expected_pct": round(expected * 100, 4),
                "Signal_surprise_pct": round(surprise * 100, 4),
                "Signal_r_a_pct": round(r_a * 100, 4),
                "Signal_r_b_pct": round(r_b * 100, 4),
                "Signal_diff_ret_pct": round(surprise * 100, 4),
                "Signal_paas_dir": _paas_direction_label(r_a),
            }
        )

    return pd.DataFrame(rows).sort_values("Trading date").reset_index(drop=True)


def _session_entry_spread(
    panel: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    td: int,
) -> dict | None:
    day = panel.loc[panel["Trading date"] == td].sort_values("DT")
    if day.empty:
        return None

    open_mask = (day["DT"].dt.hour == OPEN_HOUR) & (day["DT"].dt.minute == OPEN_MINUTE)
    open_row = day.loc[open_mask].iloc[0] if open_mask.any() else day.iloc[0]
    return {
        "entry_dt": open_row["DT"],
        "entry_pa": float(open_row[ticker_a]),
        "entry_pb": float(open_row[ticker_b]),
    }


def _resolve_intraday_exit_spread(
    panel: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    td: int,
    *,
    side: str,
    beta: float,
    notional: float,
    max_session_loss: float,
    entry_dt: pd.Timestamp,
    entry_pa: float,
    entry_pb: float,
    hedge_style: str = "beta",
) -> dict | None:
    day = panel.loc[panel["Trading date"] == td].sort_values("DT")
    if day.empty:
        return None

    session_pnl = 0.0
    after_entry = day.loc[day["DT"] >= entry_dt]
    if after_entry.empty:
        return None

    for _, bar in after_entry.iterrows():
        pa = float(bar[ticker_a])
        pb = float(bar[ticker_b])
        unrealized = _spread_position_pnl_usd(
            side, notional, beta, ticker_a, ticker_b, entry_pa, entry_pb, pa, pb,
            hedge_style=hedge_style,
        )
        if max_session_loss > 0 and session_pnl + unrealized <= -max_session_loss:
            raw_pnl = unrealized
            capped_pnl = _cap_session_pnl(session_pnl, raw_pnl, max_session_loss)
            legs = _spread_leg_breakdown(
                side, notional, beta, ticker_a, ticker_b, entry_pa, entry_pb, pa, pb,
                hedge_style=hedge_style,
            )
            reason = "Session loss cap"
            if abs(capped_pnl - raw_pnl) > 1e-9:
                legs = _scale_spread_leg_pnl(legs, capped_pnl)
                reason = f"{reason} (${max_session_loss:.0f})"
            return {
                "exit_dt": bar["DT"],
                "exit_pa": pa,
                "exit_pb": pb,
                "exit_reason": reason,
                "legs": legs,
                "pnl_usd": capped_pnl,
            }

    close_row = day.iloc[-1]
    pa = float(close_row[ticker_a])
    pb = float(close_row[ticker_b])
    legs = _spread_leg_breakdown(
        side, notional, beta, ticker_a, ticker_b, entry_pa, entry_pb, pa, pb,
        hedge_style=hedge_style,
    )
    return {
        "exit_dt": close_row["DT"],
        "exit_pa": pa,
        "exit_pb": pb,
        "exit_reason": "EOD",
        "legs": legs,
        "pnl_usd": legs["PnL_usd"],
    }


def _session_entry(
    panel: pd.DataFrame,
    ticker: str,
    td: int,
) -> dict | None:
    day = panel.loc[panel["Trading date"] == td].sort_values("DT")
    if day.empty or ticker not in day.columns:
        return None

    open_mask = (day["DT"].dt.hour == OPEN_HOUR) & (day["DT"].dt.minute == OPEN_MINUTE)
    open_row = day.loc[open_mask].iloc[0] if open_mask.any() else day.iloc[0]
    return {
        "entry_dt": open_row["DT"],
        "entry_price": float(open_row[ticker]),
    }


def _resolve_intraday_exit(
    panel: pd.DataFrame,
    trade_ticker: str,
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
    if day.empty or trade_ticker not in day.columns:
        return None

    session_pnl = 0.0
    after_entry = day.loc[day["DT"] >= entry_dt]
    if after_entry.empty:
        return None

    for _, bar in after_entry.iterrows():
        mark = float(bar[trade_ticker])
        unrealized = _position_pnl_usd(
            side, notional, trade_ticker, entry_price, mark
        )
        if max_session_loss > 0 and session_pnl + unrealized <= -max_session_loss:
            raw_pnl = unrealized
            capped_pnl = _cap_session_pnl(session_pnl, raw_pnl, max_session_loss)
            legs = _leg_breakdown(side, notional, trade_ticker, entry_price, mark)
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
    mark = float(close_row[trade_ticker])
    legs = _leg_breakdown(side, notional, trade_ticker, entry_price, mark)
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
        surprise = float(row["Signal_surprise_pct"]) / 100.0
        r_a = float(row["Signal_r_a_pct"]) / 100.0
        plan = resolve_trade_plan(
            p.trade_mode, surprise, r_a, thresh, ticker_a, ticker_b,
            hedge_style=p.hedge_style,
        )
        if plan is None:
            continue

        td = int(row["Trading date"])
        hedge_type = plan.get("hedge_type", "single")

        if hedge_type == "spread":
            beta_trade = float(row["beta_trade"])
            entry = _session_entry_spread(panel, ticker_a, ticker_b, td)
            if entry is None:
                continue
            exit_px = _resolve_intraday_exit_spread(
                panel,
                ticker_a,
                ticker_b,
                td,
                side=plan["side"],
                beta=beta_trade,
                notional=p.notional,
                max_session_loss=p.max_session_loss,
                entry_dt=entry["entry_dt"],
                entry_pa=entry["entry_pa"],
                entry_pb=entry["entry_pb"],
                hedge_style=p.hedge_style,
            )
            entry_price = entry["entry_pa"]
            exit_price = exit_px["exit_pa"] if exit_px else float("nan")
            beta_entry = beta_trade
        else:
            trade_ticker = plan["trade_ticker"]
            entry = _session_entry(panel, trade_ticker, td)
            if entry is None:
                continue
            exit_px = _resolve_intraday_exit(
                panel,
                trade_ticker,
                td,
                side=plan["side"],
                notional=p.notional,
                max_session_loss=p.max_session_loss,
                entry_dt=entry["entry_dt"],
                entry_price=entry["entry_price"],
            )
            entry_price = entry["entry_price"]
            exit_price = exit_px["exit_price"] if exit_px else float("nan")
            beta_entry = float(row["beta_signal"])

        if exit_px is None:
            continue

        legs = exit_px["legs"]
        pnl_usd = exit_px["pnl_usd"]
        return_pct = 100.0 * pnl_usd / p.notional if p.notional else 0.0

        trade_rec: dict = {
            "Trade_id": trade_id + 1,
            "Signal_date": int(row["Signal_date"]),
            "Signal_flag": plan["signal_flag"],
            "Spread_regime": plan["spread_regime"],
            "Signal_paas_dir": plan["paas_dir"] or row["Signal_paas_dir"],
            "Trade_rule": plan["trade_rule"],
            "Signal_actual_pct": row["Signal_actual_pct"],
            "Signal_expected_pct": row["Signal_expected_pct"],
            "Signal_surprise_pct": row["Signal_surprise_pct"],
            "Signal_diff_ret_pct": row["Signal_diff_ret_pct"],
            "Beta_signal": round(float(row["beta_signal"]), 4),
            "Beta_trade": round(beta_entry, 4),
            "Side": plan["side"],
            "Hedge_type": hedge_type,
            "Hedge_style": p.hedge_style if hedge_type == "spread" else "single",
            "Trade_ticker": plan["trade_ticker"],
            "Entry_date": td,
            "Entry_time": pd.Timestamp(entry["entry_dt"]).strftime("%Y-%m-%d %H:%M"),
            "Exit_date": td,
            "Exit_time": pd.Timestamp(exit_px["exit_dt"]).strftime("%Y-%m-%d %H:%M"),
            "Exit_reason": exit_px["exit_reason"],
            "Return_pct": round(return_pct, 4),
            "PnL_$": round(pnl_usd, 2),
            "_legs": legs,
        }
        if hedge_type == "spread":
            trade_rec[f"Entry_{ticker_a}"] = round(entry["entry_pa"], 4)
            trade_rec[f"Entry_{ticker_b}"] = round(entry["entry_pb"], 4)
            trade_rec[f"Exit_{ticker_a}"] = round(exit_px["exit_pa"], 4)
            trade_rec[f"Exit_{ticker_b}"] = round(exit_px["exit_pb"], 4)
            trade_rec["PnL_A_$"] = legs["PnL_A_usd"]
            trade_rec["PnL_B_$"] = legs["PnL_B_usd"]
        else:
            trade_rec["Entry_price"] = round(entry_price, 4)
            trade_rec["Exit_price"] = round(exit_price, 4)

        trade_id += 1
        trades.append(trade_rec)

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

    ta = trades_df["Trade_ticker"]
    side = trades_df["Side"]
    hedge = trades_df.get("Hedge_type", pd.Series(["single"] * len(trades_df)))

    rules_row = (
        (
            "Rules",
            f"surprise > +{p.threshold_pct}% → short {ticker_a}; "
            f"< −{p.threshold_pct}% → long {ticker_a}",
        )
        if p.trade_mode == "surprise"
        else (
            "Rules",
            "large spread → short spread (SELL A, BUY β×B); "
            "small spread → long spread (BUY A, SELL β×B); "
            "quadrant picks side from PAAS direction",
        )
    )

    position_row = (
        f"Equal-dollar spread: ${p.notional:,.0f} on {ticker_a} and ${p.notional:,.0f} on {ticker_b}"
        if p.trade_mode == "quadrant" and p.hedge_style == "equal"
        else (
            f"β-hedged spread: ${p.notional:,.0f} on {ticker_a}, "
            f"${p.notional:,.0f}×β_trade on {ticker_b}"
            if p.trade_mode == "quadrant"
            else f"Single leg ${p.notional:,.0f} on {ticker_a}"
        )
    )

    rows = [
        ("Trades", n),
        ("Trade mode", p.trade_mode),
        ("Hedge style", p.hedge_style if p.trade_mode == "quadrant" else "n/a"),
        ("Long spread", int(((hedge == "spread") & (side == "long")).sum())),
        ("Short spread", int(((hedge == "spread") & (side == "short")).sum())),
        ("Long PAAS (single)", int(((ta == ticker_a) & (side == "long") & (hedge != "spread")).sum())),
        ("Short PAAS (single)", int(((ta == ticker_a) & (side == "short") & (hedge != "spread")).sum())),
        ("Total PnL ($)", round(total, 2)),
        ("Avg return per trade (%)", round(trades_df["Return_pct"].mean(), 3)),
        ("Avg PnL per trade ($)", round(trades_df["PnL_$"].mean(), 2)),
        ("Win rate (%)", round(100 * wins / n, 1)),
        ("Best trade ($)", round(trades_df["PnL_$"].max(), 2)),
        ("Worst trade ($)", round(trades_df["PnL_$"].min(), 2)),
        ("Signal β window", f"day −{SIGNAL_BETA_OFFSET_START} … −{SIGNAL_BETA_OFFSET_END}"),
        (
            "Signal (day −1)",
            f"actual r_{ticker_a}, expected β×r_{ticker_b}, surprise = actual − expected",
        ),
        rules_row,
        ("β trade window", f"day −{TRADE_BETA_OFFSET_START} … −{TRADE_BETA_OFFSET_END} (quadrant hedge)"),
        ("Position", position_row),
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


def _trades_table_html(trades_df: pd.DataFrame, trade_mode: str) -> str:
    if trades_df.empty:
        return "<p>No trades</p>"
    if trade_mode == "quadrant":
        cols = [
            "Trade_id",
            "Signal_date",
            "Spread_regime",
            "Signal_paas_dir",
            "Side",
            "Beta_trade",
            "Signal_surprise_pct",
            "Trade_rule",
            "Entry_date",
            "Exit_reason",
            "PnL_A_$",
            "PnL_B_$",
            "PnL_$",
        ]
    else:
        cols = [
            "Trade_id",
            "Signal_date",
            "Signal_actual_pct",
            "Signal_expected_pct",
            "Signal_surprise_pct",
            "Trade_ticker",
            "Side",
            "Trade_rule",
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
            if c in ("PnL_$", "PnL_A_$", "PnL_B_$") and pd.notna(val):
                text = f"{float(val):,.2f}"
            elif c in ("Signal_diff_ret_pct", "Signal_surprise_pct", "Signal_actual_pct", "Signal_expected_pct", "Return_pct") and pd.notna(val):
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
            "Spread_regime": row["Spread_regime"],
            "Signal_paas_dir": row["Signal_paas_dir"],
            "Trade_rule": row["Trade_rule"],
            "Signal_actual_pct": float(row["Signal_actual_pct"]),
            "Signal_expected_pct": float(row["Signal_expected_pct"]),
            "Signal_surprise_pct": float(row["Signal_surprise_pct"]),
            "Signal_diff_ret_pct": float(row["Signal_diff_ret_pct"]),
            "Beta_signal": float(row["Beta_signal"]),
            "Beta_trade": float(row["Beta_trade"]) if "Beta_trade" in row.index else float(row["Beta_signal"]),
            "Side": row["Side"],
            "Hedge_type": row["Hedge_type"] if "Hedge_type" in row.index else "single",
            "Trade_ticker": row["Trade_ticker"],
            "Entry_date": int(row["Entry_date"]),
            "Entry_time": row["Entry_time"],
            "Exit_date": int(row["Exit_date"]),
            "Exit_time": row["Exit_time"],
            "Exit_reason": row["Exit_reason"],
            "Return_pct": float(row["Return_pct"]),
            "PnL_usd": float(row["PnL_$"]),
            "PnL_A_usd": float(row["PnL_A_$"]) if "PnL_A_$" in row and pd.notna(row["PnL_A_$"]) else None,
            "PnL_B_usd": float(row["PnL_B_$"]) if "PnL_B_$" in row and pd.notna(row["PnL_B_$"]) else None,
            "Entry_price": float(row["Entry_price"]) if "Entry_price" in row and pd.notna(row.get("Entry_price")) else None,
            "Exit_price": float(row["Exit_price"]) if "Exit_price" in row and pd.notna(row.get("Exit_price")) else None,
            f"Entry_{ticker_a}": float(row[f"Entry_{ticker_a}"]) if f"Entry_{ticker_a}" in row else None,
            f"Entry_{ticker_b}": float(row[f"Entry_{ticker_b}"]) if f"Entry_{ticker_b}" in row else None,
            f"Exit_{ticker_a}": float(row[f"Exit_{ticker_a}"]) if f"Exit_{ticker_a}" in row else None,
            f"Exit_{ticker_b}": float(row[f"Exit_{ticker_b}"]) if f"Exit_{ticker_b}" in row else None,
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

    explainer = (
        f"<strong>Signal</strong> (day −1, trade next open):<br>"
        f"actual = r_{ticker_a}, expected = β_signal×r_{ticker_b}, "
        f"surprise = actual − expected.<br>"
        f"surprise &gt; +{p.threshold_pct}% → <strong>short {ticker_a}</strong>; "
        f"surprise &lt; −{p.threshold_pct}% → <strong>long {ticker_a}</strong> "
        f"(mean-revert).<br>"
        f"${p.notional:,.0f} single leg; exit EOD or −${p.max_session_loss:.0f} cap."
        if p.trade_mode == "surprise"
        else (
            f"<strong>Large spread</strong> (&gt; +{p.threshold_pct}%): <strong>short spread</strong><br>"
            f"<strong>Small spread</strong> (&lt; −{p.threshold_pct}%): <strong>long spread</strong><br>"
            + (
                f"Equal hedge: ${p.notional:,.0f} on {ticker_a} and ${p.notional:,.0f} on {ticker_b} (no β).<br>"
                if p.hedge_style == "equal"
                else (
                    f"β hedge: ${p.notional:,.0f} on {ticker_a}, "
                    f"${p.notional:,.0f}×β_trade on {ticker_b} "
                    f"(β_trade day −{TRADE_BETA_OFFSET_START}…−{TRADE_BETA_OFFSET_END}).<br>"
                )
            )
            + f"Exit EOD or −${p.max_session_loss:.0f} cap."
        )
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
{_trades_table_html(trades_df, p.trade_mode)}
<h2 id="legs-title">Trade legs</h2>
<div id="explainer">
  {explainer}
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
  title.textContent = 'Trade #' + t.Trade_id + ' — ' + t.Side.toUpperCase()
    + (t.Hedge_type === 'spread' ? ' spread' : ' ' + t.Trade_ticker);
  meta.textContent =
    t.Trade_rule + ' | β_trade ' + t.Beta_trade
    + ' | signal ' + t.Signal_date
    + ' surprise ' + t.Signal_surprise_pct.toFixed(3) + '%'
    + (t.Signal_paas_dir ? ' PAAS ' + t.Signal_paas_dir : '')
    + ' | Entry ' + t.Entry_time + ' Exit ' + t.Exit_time
    + ' (' + t.Exit_reason + ')'
    + ' | PnL $' + pnlTotal.toFixed(2);

  let html = legTable('Entry', t.Entry_time, t.legs.entry);
  html += legTable('Exit', t.Exit_time, t.legs.exit);
  html += '<table class="data-table"><thead><tr><th>PnL</th><th>$</th></tr></thead><tbody>';
  if (t.PnL_A_usd != null && t.PnL_B_usd != null) {{
    html += '<tr><td>' + t.ticker_a + '</td><td>' + t.PnL_A_usd.toFixed(2) + '</td></tr>';
    html += '<tr><td>' + t.ticker_b + ' (β hedge)</td><td>' + t.PnL_B_usd.toFixed(2) + '</td></tr>';
  }}
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
        description="Daily overreaction: quadrant spread×PAAS-dir rules, open-to-EOD."
    )
    p.add_argument("--ticker-a", default="PAAS")
    p.add_argument("--ticker-b", default="SLV")
    p.add_argument("--start", type=int, default=default_start)
    p.add_argument("--end", type=int, default=default_end)
    p.add_argument(
        "--trade-mode",
        choices=("surprise", "quadrant"),
        default="surprise",
        help="surprise = actual vs expected on day−1, trade PAAS; "
        "quadrant = spread×PAAS-dir rules",
    )
    p.add_argument(
        "--hedge-style",
        choices=("beta", "equal"),
        default="beta",
        help="Quadrant spread hedge: 'beta' = $10k×β on B leg; 'equal' = $10k on both legs",
    )
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
        trade_mode=args.trade_mode,
        hedge_style=args.hedge_style,
    )

    print(
        f"Loading {args.ticker_a}/{args.ticker_b} ({args.start}–{args.end}), "
        f"mode={params.trade_mode}"
        + (f", hedge={params.hedge_style}" if params.trade_mode == "quadrant" else "")
        + " …"
    )
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
