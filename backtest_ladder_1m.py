#!/usr/bin/env python3
"""
1-minute β_D spread ladder backtest.

Spread: diff_ret = r_A − β_D×r_B, cum_diff = session cumsum from open.

Entry (actual cum at 1-min bar when threshold first touched):
  Long  when cum >= +entry  (default +0.5%%)
  Short when cum <= −entry

Take profit (4%% of **initial** size each, fire when cum reaches or passes level):
  Long:  +0.6%%, +0.7%%, … +3.0%% session cum
  Short: mirror negative levels

Stop loss (50%% + 50%%, fire when cum reaches or passes level):
  Long:  cum <= +0.2%% → cut half; cum <= 0%% → flat rest
  Short: cum >= −0.2%% → cut half; cum >= 0%% → flat rest

Long and short are independent (both can be open). Re-entry allowed after flat.
EOD closes any remainder at last 1-min bar (15:59).

Usage::

    python backtest_ladder_1m.py --start 20250101 --end 20251231 --notional 10000
"""

from __future__ import annotations

import argparse
import json
import os
import webbrowser
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from silver_backtesting.config import Paths
from silver_backtesting.pair_quick import attach_diff_ret_1m, load_pair_panel


def _pct(x: float) -> float:
    """Percentage points → decimal (0.5 → 0.005)."""
    return x / 100.0


@dataclass
class LadderParams:
    entry_pct: float = 0.5
    tp_start_pct: float = 0.6
    tp_end_pct: float = 3.0
    tp_step_pct: float = 0.1
    tp_frac: float = 0.04
    stop_partial_pct: float = 0.2
    stop_partial_cut: float = 0.50
    stop_full_pct: float = 0.0
    notional: float = 10_000.0


@dataclass
class OpenPosition:
    side: str
    entry_cum: float
    entry_dt: pd.Timestamp
    remaining: float = 1.0
    tp_hit: set[float] = field(default_factory=set)
    stop_partial_done: bool = False


def _tp_levels(side: str, p: LadderParams) -> list[float]:
    step = _pct(p.tp_step_pct)
    start = _pct(p.tp_start_pct)
    end = _pct(p.tp_end_pct)
    levels: list[float] = []
    cur = start
    while cur <= end + 1e-12:
        levels.append(cur)
        cur += step
    if side == "short":
        levels = [-x for x in levels]
    return levels


def _leg_pnl(side: str, entry_cum: float, exit_cum: float, frac: float) -> float:
    if side == "long":
        return frac * (exit_cum - entry_cum)
    return frac * (entry_cum - exit_cum)


def _record_fill(
    fills: list[dict],
    *,
    td: int,
    side: str,
    reason: str,
    dt: pd.Timestamp,
    frac: float,
    entry_cum: float,
    exit_cum: float,
    pnl_dec: float,
    remaining: float,
    notional: float,
) -> None:
    fills.append(
        {
            "Trading date": td,
            "Side": side,
            "Reason": reason,
            "DT": dt,
            "Frac": frac,
            "Entry_cum_pct": entry_cum * 100,
            "Exit_cum_pct": exit_cum * 100,
            "PnL_pct": pnl_dec * 100,
            "PnL_$": pnl_dec * notional,
            "Remaining": remaining,
        }
    )


def _manage_position(
    pos: OpenPosition,
    cum: float,
    dt: pd.Timestamp,
    td: int,
    p: LadderParams,
    fills: list[dict],
) -> None:
    if pos.remaining <= 1e-9:
        pos.remaining = 0.0
        return

    stop_partial = (
        _pct(p.stop_partial_pct) if pos.side == "long" else -_pct(p.stop_partial_pct)
    )
    stop_full = _pct(p.stop_full_pct)

    for level in _tp_levels(pos.side, p):
        if level in pos.tp_hit or pos.remaining <= 0:
            continue
        hit = cum >= level if pos.side == "long" else cum <= level
        if not hit:
            continue
        frac = min(p.tp_frac, pos.remaining)
        pnl = _leg_pnl(pos.side, pos.entry_cum, cum, frac)
        pos.remaining -= frac
        pos.tp_hit.add(level)
        _record_fill(
            fills,
            td=td,
            side=pos.side,
            reason=f"TP@{level * 100:.2f}%",
            dt=dt,
            frac=frac,
            entry_cum=pos.entry_cum,
            exit_cum=cum,
            pnl_dec=pnl,
            remaining=pos.remaining,
            notional=p.notional,
        )

    if pos.remaining > 0 and not pos.stop_partial_done:
        hit = cum <= stop_partial if pos.side == "long" else cum >= stop_partial
        if hit:
            frac = pos.remaining * p.stop_partial_cut
            if frac > 0:
                pnl = _leg_pnl(pos.side, pos.entry_cum, cum, frac)
                pos.remaining -= frac
                pos.stop_partial_done = True
                _record_fill(
                    fills,
                    td=td,
                    side=pos.side,
                    reason=f"Stop@{p.stop_partial_pct}% (cut {p.stop_partial_cut:.0%})",
                    dt=dt,
                    frac=frac,
                    entry_cum=pos.entry_cum,
                    exit_cum=cum,
                    pnl_dec=pnl,
                    remaining=pos.remaining,
                    notional=p.notional,
                )

    if pos.remaining > 0:
        hit = cum <= stop_full if pos.side == "long" else cum >= stop_full
        if hit:
            frac = pos.remaining
            pnl = _leg_pnl(pos.side, pos.entry_cum, cum, frac)
            pos.remaining = 0.0
            _record_fill(
                fills,
                td=td,
                side=pos.side,
                reason=f"Stop@{p.stop_full_pct}% (flat)",
                dt=dt,
                frac=frac,
                entry_cum=pos.entry_cum,
                exit_cum=cum,
                pnl_dec=pnl,
                remaining=0.0,
                notional=p.notional,
            )


def simulate_session(
    day: pd.DataFrame,
    td: int,
    p: LadderParams,
) -> tuple[list[dict], float]:
    fills: list[dict] = []
    long_pos: OpenPosition | None = None
    short_pos: OpenPosition | None = None

    entry_long = _pct(p.entry_pct)
    entry_short = -entry_long

    for row in day.itertuples(index=False):
        if pd.isna(row.diff_ret) or pd.isna(row.cum_diff_ret) or pd.isna(row.beta_D):
            continue
        cum = float(row.cum_diff_ret)
        dt = row.DT

        if long_pos is None and cum >= entry_long:
            long_pos = OpenPosition("long", cum, dt)
            _record_fill(
                fills,
                td=td,
                side="long",
                reason=f"Entry (β_D={row.beta_D:.3f}, cum={cum * 100:.3f}%)",
                dt=dt,
                frac=0.0,
                entry_cum=cum,
                exit_cum=cum,
                pnl_dec=0.0,
                remaining=1.0,
                notional=p.notional,
            )

        if short_pos is None and cum <= entry_short:
            short_pos = OpenPosition("short", cum, dt)
            _record_fill(
                fills,
                td=td,
                side="short",
                reason=f"Entry (β_D={row.beta_D:.3f}, cum={cum * 100:.3f}%)",
                dt=dt,
                frac=0.0,
                entry_cum=cum,
                exit_cum=cum,
                pnl_dec=0.0,
                remaining=1.0,
                notional=p.notional,
            )

        if long_pos is not None and long_pos.remaining > 0:
            _manage_position(long_pos, cum, dt, td, p, fills)
            if long_pos.remaining <= 0:
                long_pos = None

        if short_pos is not None and short_pos.remaining > 0:
            _manage_position(short_pos, cum, dt, td, p, fills)
            if short_pos.remaining <= 0:
                short_pos = None

    for pos, side in ((long_pos, "long"), (short_pos, "short")):
        if pos is not None and pos.remaining > 0:
            exit_cum = float(day["cum_diff_ret"].iloc[-1])
            exit_dt = day["DT"].iloc[-1]
            frac = pos.remaining
            pnl = _leg_pnl(pos.side, pos.entry_cum, exit_cum, frac)
            pos.remaining = 0.0
            _record_fill(
                fills,
                td=td,
                side=side,
                reason="EOD",
                dt=exit_dt,
                frac=frac,
                entry_cum=pos.entry_cum,
                exit_cum=exit_cum,
                pnl_dec=pnl,
                remaining=0.0,
                notional=p.notional,
            )

    session_pnl = sum(f["PnL_$"] for f in fills if not str(f["Reason"]).startswith("Entry"))
    return fills, session_pnl


def run_backtest(df: pd.DataFrame, p: LadderParams) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_fills: list[dict] = []
    daily_rows: list[dict] = []

    for td, day in df.groupby("Trading date"):
        day = day.sort_values("DT")
        if day["beta_D"].isna().all():
            continue
        fills, session_pnl = simulate_session(day, int(td), p)
        if not fills:
            continue
        all_fills.extend(fills)
        entries = [f for f in fills if str(f["Reason"]).startswith("Entry")]
        daily_rows.append(
            {
                "Trading date": int(td),
                "Entries": len(entries),
                "Long": sum(1 for f in entries if f["Side"] == "long"),
                "Short": sum(1 for f in entries if f["Side"] == "short"),
                "Session_PnL_$": session_pnl,
            }
        )

    fills_df = pd.DataFrame(all_fills)
    daily_df = pd.DataFrame(daily_rows)
    if not daily_df.empty:
        daily_df = daily_df.sort_values("Trading date")
        daily_df["Cum_PnL_$"] = daily_df["Session_PnL_$"].cumsum()
    return fills_df, daily_df


def summary_stats(
    daily_df: pd.DataFrame,
    fills_df: pd.DataFrame,
    p: LadderParams,
) -> pd.DataFrame:
    if daily_df.empty:
        return pd.DataFrame({"Metric": ["Sessions"], "Value": [0]})

    exits = fills_df[~fills_df["Reason"].astype(str).str.startswith("Entry")]
    n_tp = exits["Reason"].astype(str).str.startswith("TP").sum()
    n_stop = exits["Reason"].astype(str).str.startswith("Stop").sum()
    n_eod = (exits["Reason"] == "EOD").sum()
    total = daily_df["Session_PnL_$"].sum()
    win_days = (daily_df["Session_PnL_$"] > 0).sum()
    n_days = len(daily_df)
    n_levels = len(_tp_levels("long", p))

    rows = [
        ("Sessions with activity", n_days),
        ("Total PnL ($)", round(total, 2)),
        ("Avg daily PnL ($)", round(daily_df["Session_PnL_$"].mean(), 2)),
        ("Win days (%)", round(100 * win_days / n_days, 1)),
        ("Exit fills", len(exits)),
        ("TP fills", int(n_tp)),
        ("Stop fills", int(n_stop)),
        ("EOD fills", int(n_eod)),
        ("Bar size", "1 min"),
        ("Spread", "r_A − β_D×r_B"),
        ("Entry (session cum %)", f"±{p.entry_pct}"),
        (
            "TP ladder",
            f"{p.tp_start_pct}…{p.tp_end_pct}% step {p.tp_step_pct}% × {p.tp_frac:.0%} ({n_levels} levels)",
        ),
        (
            "Stop loss",
            f"{p.stop_partial_pct}% cut {p.stop_partial_cut:.0%}; {p.stop_full_pct}% flat",
        ),
        ("Notional per leg", p.notional),
        ("Adapted β filter", "off"),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value"])


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


def _fills_json(fills_df: pd.DataFrame) -> str:
    if fills_df.empty:
        return "{}"
    grouped: dict[str, list[dict]] = {}
    for td, grp in fills_df.groupby("Trading date"):
        rows: list[dict] = []
        for rec in grp.sort_values("DT").to_dict(orient="records"):
            rows.append(
                {
                    "DT": pd.Timestamp(rec["DT"]).strftime("%Y-%m-%d %H:%M"),
                    "Side": rec["Side"],
                    "Reason": rec["Reason"],
                    "Frac": round(float(rec["Frac"]), 4),
                    "Entry_cum_pct": round(float(rec["Entry_cum_pct"]), 3),
                    "Exit_cum_pct": round(float(rec["Exit_cum_pct"]), 3),
                    "PnL_pct": round(float(rec["PnL_pct"]), 3),
                    "PnL_$": round(float(rec["PnL_$"]), 2),
                    "Remaining": round(float(rec["Remaining"]), 4),
                }
            )
        grouped[str(int(td))] = rows
    return json.dumps(grouped)


def build_report_html(
    summary: pd.DataFrame,
    daily_df: pd.DataFrame,
    fills_df: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
) -> str:
    fig = make_subplots(rows=2, cols=1, row_heights=[0.45, 0.55], vertical_spacing=0.08)
    if not daily_df.empty:
        x_dates = pd.to_datetime(daily_df["Trading date"].astype(str), format="%Y%m%d")
        fig.add_trace(go.Bar(x=x_dates, y=daily_df["Session_PnL_$"], name="Daily PnL"), row=1, col=1)
        fig.add_trace(
            go.Scatter(x=x_dates, y=daily_df["Cum_PnL_$"], mode="lines", name="Cumulative PnL"),
            row=2,
            col=1,
        )
    fig.update_layout(template="plotly_white", height=700, title="1m β_D ladder backtest")
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
<html><head><meta charset="utf-8"><title>1m ladder backtest</title>
<style>
body {{ font-family: sans-serif; margin: 2rem; }}
h1,h2 {{ color: #0066cc; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
th {{ background: #0066cc; color: white; padding: 6px; text-align: left; }}
td {{ border: 1px solid #ddd; padding: 6px; }}
.interactive-table tbody tr.daily-row {{ cursor: pointer; }}
.interactive-table tbody tr.daily-row:hover {{ background: #eef6ff; }}
.interactive-table tbody tr.daily-row.selected {{ background: #cce0ff; }}
</style></head><body>
<h1>1m β_D ladder — {ticker_a} vs {ticker_b}</h1>
<h2>Summary</h2>{summary.to_html(index=False, classes="data-table", border=0)}
<div>{chart}</div>
<h2>Daily PnL</h2>
<p>Click a row to see 1-min fills.</p>
{_daily_table_html(daily_df)}
<h2 id="fills-title">Session fills</h2>
<p id="session-meta">No session selected.</p>
<div id="fills-panel"></div>
<script>
const FILLS_BY_TD = {_fills_json(fills_df)};
const TD_LABELS = {json.dumps(td_dates)};
function renderFills(td) {{
  const rows = FILLS_BY_TD[String(td)] || [];
  const label = TD_LABELS[String(td)] || td;
  document.getElementById('fills-title').textContent = 'Session fills — ' + label;
  const panel = document.getElementById('fills-panel');
  if (!rows.length) {{ panel.innerHTML = '<p>No fills.</p>'; return; }}
  const exits = rows.filter(r => !String(r.Reason).startsWith('Entry'));
  const sessionPnl = exits.reduce((s,r) => s + r['PnL_$'], 0);
  document.getElementById('session-meta').textContent =
    'Session PnL: $' + sessionPnl.toFixed(2) + ' | fills: ' + rows.length;
  const cols = ['DT','Side','Reason','Frac','Entry_cum_pct','Exit_cum_pct','PnL_$'];
  let html = '<table class="data-table"><thead><tr>';
  cols.forEach(c => html += '<th>' + c + '</th>');
  html += '</tr></thead><tbody>';
  rows.forEach(r => {{
    html += '<tr>' + cols.map(c => '<td>' + r[c] + '</td>').join('') + '</tr>';
  }});
  html += '</tbody></table>';
  panel.innerHTML = html;
}}
function selectDailyRow(td) {{
  document.querySelectorAll('.daily-row').forEach(tr => {{
    tr.classList.toggle('selected', tr.dataset.td === String(td));
  }});
  renderFills(td);
}}
document.querySelectorAll('.daily-row').forEach(tr => {{
  tr.addEventListener('click', () => selectDailyRow(tr.dataset.td));
}});
if (Object.keys(TD_LABELS).length) selectDailyRow(Object.keys(TD_LABELS).sort().pop());
</script>
</body></html>"""


def _parse_args() -> argparse.Namespace:
    today = datetime.now() - timedelta(days=3)
    default_end = int(today.strftime("%Y%m%d"))
    default_start = int((today - timedelta(days=45)).strftime("%Y%m%d"))

    p = argparse.ArgumentParser(description="1-min β_D spread ladder backtest.")
    p.add_argument("--ticker-a", default="PAAS")
    p.add_argument("--ticker-b", default="SLV")
    p.add_argument("--start", type=int, default=default_start)
    p.add_argument("--end", type=int, default=default_end)
    p.add_argument("--entry", type=float, default=0.5, help="Entry threshold %% (0.5 = 0.005)")
    p.add_argument("--tp-start", type=float, default=0.6)
    p.add_argument("--tp-end", type=float, default=3.0)
    p.add_argument("--tp-step", type=float, default=0.1)
    p.add_argument("--tp-frac", type=float, default=0.04, help="Fraction per TP (0.04 = 4%%)")
    p.add_argument("--stop-partial", type=float, default=0.2)
    p.add_argument("--stop-full", type=float, default=0.0)
    p.add_argument("--notional", type=float, default=10_000.0)
    p.add_argument("--data-folder", default=None)
    p.add_argument("--output", default="ladder_1m_backtest.html")
    p.add_argument("--no-browser", action="store_true")
    p.add_argument("--export-csv", default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    folder = args.data_folder or Paths().min_folder
    params = LadderParams(
        entry_pct=args.entry,
        tp_start_pct=args.tp_start,
        tp_end_pct=args.tp_end,
        tp_step_pct=args.tp_step,
        tp_frac=args.tp_frac,
        stop_partial_pct=args.stop_partial,
        stop_full_pct=args.stop_full,
        notional=args.notional,
    )

    print(f"Loading {args.ticker_a}/{args.ticker_b} …")
    panel = load_pair_panel(args.ticker_a, args.ticker_b, folder, args.start, args.end)
    df = attach_diff_ret_1m(panel, args.ticker_a, args.ticker_b)
    print(f"  {df['Trading date'].nunique()} sessions, {len(df):,} 1-min bars")

    fills_df, daily_df = run_backtest(df, params)
    summary = summary_stats(daily_df, fills_df, params)
    print("\n=== Summary ===")
    print(summary.to_string(index=False))

    out = args.output
    with open(out, "w", encoding="utf-8") as f:
        f.write(build_report_html(summary, daily_df, fills_df, args.ticker_a, args.ticker_b))
    print(f"\nReport: {out}")

    if args.export_csv:
        os.makedirs(args.export_csv, exist_ok=True)
        daily_df.to_csv(os.path.join(args.export_csv, "daily_pnl.csv"), index=False)
        fills_df.to_csv(os.path.join(args.export_csv, "fills.csv"), index=False)

    if not args.no_browser:
        webbrowser.open(f"file://{os.path.abspath(out)}")


if __name__ == "__main__":
    main()
