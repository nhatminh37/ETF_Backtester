#!/usr/bin/env python3
"""
Backtest: cumulative β_D hedged return spread with scaled exits.

Signal (session cumulative diff_ret from open):
  diff_ret = r_A − β_D×r_B  (5-min bar returns by default)
  cum_diff = Σ diff_ret within session

Long when cum_diff (β_D) >= +entry_pct AND cum_diff_adapted (β_adapted) >= +adapted_pct.
Short when both <= −thresholds (mirror).

β_D = prior session 15m OLS. β_adapted = rolling 30m OLS (available after 30 min RTH).
The adapted leg acts as a confirmation filter: yesterday's overreaction must still
show up under today's hedge ratio, not just silver drift or stale β.

Take profit (each 20%% of **initial** size, 5 levels):
  Long:  entry+0.6, +0.7, +0.8, +0.9, +1.0 pct pts above entry (default 2.1…2.5)
  Short: mirror below entry

At most one entry per session (first long or short signal only).

Any size left at 15:59 → closed at session cum_diff.

Usage (Windows one line)::

    python backtest_diff_ret.py --start 20250101 --end 20251231 --notional 10000 --output diff_ret_report.html
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
from silver_backtesting.pair_quick import attach_diff_ret, load_pair_panel


def _pct(x: float) -> float:
    """Convert percentage points (1.5 = 1.5%%) to decimal return."""
    return x / 100.0


@dataclass
class StrategyParams:
    entry_pct: float = 1.0
    adapted_pct: float = 1.0
    require_adapted_confirm: bool = True
    beta_window: int = 30
    beta_warmup: int = 30
    tp_start_pct: float = 2.1
    tp_step_pct: float = 0.1
    tp_slices: int = 5
    tp_frac: float = 0.20
    notional: float = 10_000.0
    return_minutes: int = 5
    one_trade_per_day: bool = True


@dataclass
class OpenPosition:
    side: str
    entry_cum: float
    entry_dt: pd.Timestamp
    remaining: float = 1.0
    tp_hit: set[float] = field(default_factory=set)


def _tp_levels(side: str, p: StrategyParams) -> list[float]:
    step = _pct(p.tp_step_pct)
    start = _pct(p.tp_start_pct)
    levels = [start + i * step for i in range(p.tp_slices)]
    if side == "short":
        levels = [-x for x in levels]
    return levels


def _long_pnl(side: str, entry_cum: float, exit_cum: float, frac: float) -> float:
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
    p: StrategyParams,
    fills: list[dict],
) -> None:
    if pos.remaining <= 1e-9:
        pos.remaining = 0.0
        return

    for level in _tp_levels(pos.side, p):
        if level in pos.tp_hit or pos.remaining <= 0:
            continue
        hit = cum >= level if pos.side == "long" else cum <= level
        if not hit:
            continue
        frac = min(p.tp_frac, pos.remaining)
        pnl = _long_pnl(pos.side, pos.entry_cum, cum, frac)
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


def simulate_session(
    day: pd.DataFrame,
    td: int,
    p: StrategyParams,
) -> tuple[list[dict], float]:
    """Return fill log and session total PnL (decimal on unit notional)."""
    fills: list[dict] = []
    long_pos: OpenPosition | None = None
    short_pos: OpenPosition | None = None
    session_used = False

    entry_long = _pct(p.entry_pct)
    entry_short = -entry_long
    adapted_long = _pct(p.adapted_pct)
    adapted_short = -adapted_long

    for row in day.itertuples(index=False):
        dr = row.diff_ret
        if pd.isna(dr) or pd.isna(row.cum_diff_ret):
            continue
        cum = float(row.cum_diff_ret)
        cum_ad = float(row.cum_diff_ret_adapted) if hasattr(row, "cum_diff_ret_adapted") and pd.notna(row.cum_diff_ret_adapted) else float("nan")
        dt = row.DT

        can_enter = (
            not session_used
            and long_pos is None
            and short_pos is None
            and pd.notna(row.beta_D)
        )
        if can_enter:
            adapted_ok_long = (
                not p.require_adapted_confirm
                or (pd.notna(cum_ad) and cum_ad >= adapted_long)
            )
            adapted_ok_short = (
                not p.require_adapted_confirm
                or (pd.notna(cum_ad) and cum_ad <= adapted_short)
            )
            beta_ad = (
                float(row.beta_adapted)
                if hasattr(row, "beta_adapted") and pd.notna(row.beta_adapted)
                else float("nan")
            )
            if cum >= entry_long and adapted_ok_long:
                session_used = True
                long_pos = OpenPosition("long", cum, dt)
                _record_fill(
                    fills,
                    td=td,
                    side="long",
                    reason=(
                        f"Entry (β_D={row.beta_D:.3f}, β_ad={beta_ad:.3f}, "
                        f"cum_D={cum * 100:.2f}%, cum_ad={cum_ad * 100:.2f}%)"
                    ),
                    dt=dt,
                    frac=0.0,
                    entry_cum=cum,
                    exit_cum=cum,
                    pnl_dec=0.0,
                    remaining=1.0,
                    notional=p.notional,
                )
            elif cum <= entry_short and adapted_ok_short:
                session_used = True
                short_pos = OpenPosition("short", cum, dt)
                _record_fill(
                    fills,
                    td=td,
                    side="short",
                    reason=(
                        f"Entry (β_D={row.beta_D:.3f}, β_ad={beta_ad:.3f}, "
                        f"cum_D={cum * 100:.2f}%, cum_ad={cum_ad * 100:.2f}%)"
                    ),
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

    active_pos = long_pos or short_pos
    if active_pos is not None and active_pos.remaining > 0:
        exit_cum = float(day["cum_diff_ret"].iloc[-1])
        exit_dt = day["DT"].iloc[-1]
        frac = active_pos.remaining
        pnl = _long_pnl(active_pos.side, active_pos.entry_cum, exit_cum, frac)
        active_pos.remaining = 0.0
        _record_fill(
            fills,
            td=td,
            side=active_pos.side,
            reason="EOD",
            dt=exit_dt,
            frac=frac,
            entry_cum=active_pos.entry_cum,
            exit_cum=exit_cum,
            pnl_dec=pnl,
            remaining=0.0,
            notional=p.notional,
        )

    session_pnl = sum(f["PnL_$"] for f in fills if not str(f["Reason"]).startswith("Entry"))
    return fills, session_pnl


def run_backtest(df: pd.DataFrame, p: StrategyParams) -> tuple[pd.DataFrame, pd.DataFrame]:
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


def summary_stats(daily_df: pd.DataFrame, fills_df: pd.DataFrame, p: StrategyParams) -> pd.DataFrame:
    if daily_df.empty:
        return pd.DataFrame({"Metric": ["Trades"], "Value": [0]})

    exits = fills_df[~fills_df["Reason"].astype(str).str.startswith("Entry")]
    total = daily_df["Session_PnL_$"].sum()
    win_days = (daily_df["Session_PnL_$"] > 0).sum()
    n_days = len(daily_df)

    rows = [
        ("Sessions with activity", n_days),
        ("Total PnL ($)", round(total, 2)),
        ("Avg daily PnL ($)", round(daily_df["Session_PnL_$"].mean(), 2)),
        ("Win days (%)", round(100 * win_days / n_days, 1)),
        ("Exit fills", len(exits)),
        ("Notional per full leg", p.notional),
        ("Entry threshold β_D (%)", p.entry_pct),
        (
            "Adapted confirm",
            f"±{p.adapted_pct}% cum (roll {p.beta_window}m β)"
            if p.require_adapted_confirm
            else "off",
        ),
        ("TP levels (%)", f"{p.tp_start_pct}…{p.tp_start_pct + (p.tp_slices - 1) * p.tp_step_pct} step {p.tp_step_pct}"),
        ("Return bar (min)", p.return_minutes),
        ("Signal / PnL hedge", "β_D (prior day 15m)"),
        ("Stop loss", "none"),
        ("Max entries / session", 1 if p.one_trade_per_day else "unlimited"),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value"])


def _trading_dates_for_plot(daily_df: pd.DataFrame) -> pd.Series:
    """Convert YYYYMMDD ints to datetimes so Plotly shows readable dates."""
    return pd.to_datetime(daily_df["Trading date"].astype(str), format="%Y%m%d")


def _daily_table_html(daily_df: pd.DataFrame) -> str:
    """Clickable daily PnL table; rows carry ``data-td`` for fill drill-down."""
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
    """Serialize fills grouped by trading date for client-side drill-down."""
    if fills_df.empty:
        return "{}"

    grouped: dict[str, list[dict]] = {}
    for td, grp in fills_df.groupby("Trading date"):
        rows: list[dict] = []
        for rec in grp.sort_values("DT").to_dict(orient="records"):
            item = {
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
            rows.append(item)
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
        title="diff_ret ladder backtest (β_D hedged cum spread)",
    )
    fig.update_xaxes(title_text="Trading date", row=1, col=1)
    fig.update_xaxes(title_text="Trading date", row=2, col=1)
    fig.update_yaxes(title_text="Daily PnL ($)", row=1, col=1)
    fig.update_yaxes(title_text="Cumulative PnL ($)", row=2, col=1)
    chart = fig.to_html(full_html=False, include_plotlyjs="cdn", div_id="pnl-chart")

    summary_html = summary.to_html(index=False, classes="data-table", border=0)
    daily_html = _daily_table_html(daily_df)
    fills_data = _fills_json(fills_df)
    td_dates = (
        {
            str(int(td)): pd.to_datetime(str(int(td)), format="%Y%m%d").strftime("%Y-%m-%d")
            for td in daily_df["Trading date"].unique()
        }
        if not daily_df.empty
        else {}
    )
    td_dates_json = json.dumps(td_dates)

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>diff_ret backtest</title>
<style>
body {{ font-family: sans-serif; margin: 2rem; }}
h1,h2,h3 {{ color: #0066cc; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
th {{ background: #0066cc; color: white; padding: 6px; text-align: left; }}
td {{ border: 1px solid #ddd; padding: 6px; }}
.interactive-table tbody tr.daily-row {{ cursor: pointer; }}
.interactive-table tbody tr.daily-row:hover {{ background: #eef6ff; }}
.interactive-table tbody tr.daily-row.selected {{ background: #cce0ff; font-weight: 600; }}
#fills-panel {{ margin-top: 1rem; }}
#fills-hint {{ color: #666; font-style: italic; }}
#session-meta {{ margin: 0.5rem 0 1rem; color: #333; }}
</style></head><body>
<h1>β_D diff_ret ladder — {ticker_a} vs {ticker_b}</h1>
<h2>Summary</h2>{summary_html}
<div>{chart}</div>
<h2>Daily PnL</h2>
<p id="fills-hint">Click a daily row (or a daily PnL bar) to show fills with fill time.</p>
{daily_html}
<h2 id="fills-title">Session fills</h2>
<p id="session-meta">No session selected.</p>
<div id="fills-panel"><p>Select a date above.</p></div>
<script>
const FILLS_BY_TD = {fills_data};
const TD_LABELS = {td_dates_json};

function renderFills(td) {{
  const title = document.getElementById('fills-title');
  const meta = document.getElementById('session-meta');
  const panel = document.getElementById('fills-panel');
  const rows = FILLS_BY_TD[String(td)] || [];
  const label = TD_LABELS[String(td)] || td;

  title.textContent = 'Session fills — ' + label;
  if (!rows.length) {{
    meta.textContent = 'No fills for ' + label + '.';
    panel.innerHTML = '<p>No fills recorded.</p>';
    return;
  }}

  const exits = rows.filter(r => !String(r.Reason).startsWith('Entry'));
  const sessionPnl = exits.reduce((s, r) => s + r['PnL_$'], 0);
  meta.textContent = 'Trading date ' + td + ' | fills: ' + rows.length
    + ' | session PnL: $' + sessionPnl.toFixed(2) + ' (β_D cum)';

  const cols = ['DT', 'Side', 'Reason', 'Frac', 'Entry_cum_pct', 'Exit_cum_pct', 'PnL_pct', 'PnL_$', 'Remaining'];
  const labels = {{
    DT: 'Fill time (ET)',
    Side: 'Side',
    Reason: 'Reason',
    Frac: 'Frac',
    Entry_cum_pct: 'Entry cum %',
    Exit_cum_pct: 'Exit cum %',
    PnL_pct: 'PnL %',
    'PnL_$': 'PnL $',
    Remaining: 'Remaining',
  }};

  let html = '<table class="data-table"><thead><tr>';
  for (const c of cols) html += '<th>' + labels[c] + '</th>';
  html += '</tr></thead><tbody>';
  for (const r of rows) {{
    html += '<tr>';
    for (const c of cols) {{
      let v = r[c];
      if (c === 'PnL_$' && typeof v === 'number') v = v.toFixed(2);
      html += '<td>' + v + '</td>';
    }}
    html += '</tr>';
  }}
  html += '</tbody></table>';
  panel.innerHTML = html;
}}

function selectDailyRow(td) {{
  document.querySelectorAll('.daily-row').forEach(tr => {{
    tr.classList.toggle('selected', tr.dataset.td === String(td));
  }});
  renderFills(td);
  document.getElementById('fills-panel').scrollIntoView({{ behavior: 'smooth', block: 'nearest' }});
}}

document.querySelectorAll('.daily-row').forEach(tr => {{
  tr.addEventListener('click', () => selectDailyRow(tr.dataset.td));
  tr.addEventListener('keydown', e => {{
    if (e.key === 'Enter' || e.key === ' ') {{
      e.preventDefault();
      selectDailyRow(tr.dataset.td);
    }}
  }});
}});

const chartEl = document.getElementById('pnl-chart');
if (chartEl) {{
  chartEl.on('plotly_click', ev => {{
    if (!ev.points || !ev.points.length) return;
    const x = ev.points[0].x;
    if (!x) return;
    const d = new Date(x);
    const y = d.getFullYear();
    const m = String(d.getMonth() + 1).padStart(2, '0');
    const day = String(d.getDate()).padStart(2, '0');
    const td = String(y) + m + day;
    selectDailyRow(td);
  }});
}}

if (Object.keys(TD_LABELS).length) {{
  const firstTd = Object.keys(TD_LABELS).sort().pop();
  selectDailyRow(firstTd);
}}
</script>
</body></html>"""


def _parse_args() -> argparse.Namespace:
    today = datetime.now() - timedelta(days=3)
    default_end = int(today.strftime("%Y%m%d"))
    default_start = int((today - timedelta(days=45)).strftime("%Y%m%d"))

    p = argparse.ArgumentParser(description="Backtest cum diff_ret ladder with β_D hedge.")
    p.add_argument("--ticker-a", default="PAAS")
    p.add_argument("--ticker-b", default="SLV")
    p.add_argument("--start", type=int, default=default_start)
    p.add_argument("--end", type=int, default=default_end)
    p.add_argument("--entry", type=float, default=1.0, help="β_D cum_diff entry threshold (%%)")
    p.add_argument(
        "--adapted-pct",
        type=float,
        default=1.0,
        help="β_adapted cum_diff confirmation threshold (%%); default same as --entry",
    )
    p.add_argument(
        "--no-adapted-filter",
        action="store_true",
        help="Disable β_adapted confirmation (β_D signal only)",
    )
    p.add_argument("--beta-window", type=int, default=30, help="Rolling minutes for β_adapted")
    p.add_argument("--beta-warmup", type=int, default=30, help="Warmup minutes before β_adapted")
    p.add_argument("--tp-start", type=float, default=2.1, help="First TP level for long (%% cum)")
    p.add_argument("--notional", type=float, default=10_000.0)
    p.add_argument(
        "--return-minutes",
        type=int,
        default=5,
        help="Bar size for r_A, r_B returns (default: 5)",
    )
    p.add_argument("--data-folder", default=None)
    p.add_argument("--output", default="diff_ret_backtest.html")
    p.add_argument("--no-browser", action="store_true")
    p.add_argument("--export-csv", default=None, help="Optional folder for daily + fills CSV")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    folder = args.data_folder or Paths().min_folder
    params = StrategyParams(
        entry_pct=args.entry,
        adapted_pct=args.adapted_pct,
        require_adapted_confirm=not args.no_adapted_filter,
        beta_window=args.beta_window,
        beta_warmup=args.beta_warmup,
        tp_start_pct=args.tp_start,
        notional=args.notional,
        return_minutes=args.return_minutes,
    )

    print(f"Loading {args.ticker_a}/{args.ticker_b} …")
    df = load_pair_panel(args.ticker_a, args.ticker_b, folder, args.start, args.end)
    n_1m = len(df)
    df = attach_diff_ret(
        df,
        args.ticker_a,
        args.ticker_b,
        return_minutes=params.return_minutes,
        beta_window=params.beta_window,
        beta_warmup=params.beta_warmup,
        include_adapted=params.require_adapted_confirm,
    )
    print(
        f"  {df['Trading date'].nunique()} sessions, "
        f"{len(df):,} valid {params.return_minutes}m bars "
        f"(from {n_1m:,} 1-min rows; gaps skipped)"
    )

    fills_df, daily_df = run_backtest(df, params)
    summary = summary_stats(daily_df, fills_df, params)

    print("\n=== Summary ===")
    print(summary.to_string(index=False))

    html = build_report_html(summary, daily_df, fills_df, args.ticker_a, args.ticker_b)
    out = args.output
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\nReport: {out}")

    if args.export_csv:
        os.makedirs(args.export_csv, exist_ok=True)
        daily_df.to_csv(os.path.join(args.export_csv, "daily_pnl.csv"), index=False)
        fills_df.to_csv(os.path.join(args.export_csv, "fills.csv"), index=False)
        print(f"CSV exported to {args.export_csv}")

    if not args.no_browser:
        webbrowser.open(f"file://{os.path.abspath(out)}")


if __name__ == "__main__":
    main()
