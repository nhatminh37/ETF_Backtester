"""Silver backtesting pipeline orchestrating the notebook workflow."""

from __future__ import annotations

import os
import warnings
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import pytz
import yfinance as yf

from .backtest import run_spread_backtest
from .config import MONTH_MAPPING, Paths, SilverBacktestState, StrategyConfig
from .loaders import (
    active_table,
    convert_ticker,
    market_calendar,
    modify_contract,
    offshore_ETL,
    onshore_ETL,
)
from .research import (
    run_correlation_nyse_rth,
    run_opengap_longshort_research,
    run_other_correlation_tests,
)
from .spread import (
    attach_lookback_beta,
    compute_ind_trend,
    convert_datetime_cols_to_string,
    forward_beta_hedged_return,
    run_stats_spreads,
    select_minutes,
)
from .ui import create_dash_app

try:
    from IPython.display import display
except ImportError:  # pragma: no cover - script mode
    def display(obj: Any) -> None:
        print(obj)

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 500)


class SilverBacktestPipeline:
    """Orchestrates the silver backtesting workflow from the notebook."""

    def __init__(
        self,
        config: StrategyConfig | None = None,
        paths: Paths | None = None,
        state: SilverBacktestState | None = None,
    ):
        self.config = config or StrategyConfig()
        self.paths = paths or Paths()
        self.state = state or SilverBacktestState()
        self.state.df_mapping = pd.read_excel(self.paths.product_mapping)

    def audit_minute_files(self) -> None:
        """Audit minute CSV coverage per ticker."""
        try:
            self.state.file_list_og[0]
        except (IndexError, KeyError):
            self.state.file_list_og = os.listdir(self.paths.min_folder)

        startdate = self.config.startdate
        enddate = self.state.enddate or int(
            str(datetime.now() - timedelta(days=3))[:10].replace("-", "")
        )
        file_list = [
            x
            for x in self.state.file_list_og
            if (int(x[:8]) >= startdate) and (int(x[:8]) <= enddate)
        ]

        for ticker in self.config.ticker_list + ["SLV"]:
            ticker_clean = ticker.replace("-US", "")
            ticker_files = [
                x.split(ticker_clean)[0]
                for x in file_list
                if (f"{ticker_clean}.CSV" in x)
                and (len(x[8:].upper().replace(".CSV", "")) == len(ticker_clean))
            ]
            print(
                f"{ticker_clean} minutes file \tcount: {len(ticker_files)} "
                f"\tmin: {min(ticker_files)} \tmax: {max(ticker_files)}"
            )

    def prepare_minute_data(self) -> None:
        """Build the 1-minute panel ``df_all_copy2``."""
        startdate = self.config.startdate
        enddate = int(str(datetime.now() - timedelta(days=3))[:10].replace("-", ""))
        self.state.enddate = enddate
        ticker_list = self.config.ticker_list
        ind_include_cn = self.config.ind_include_cn

        self.state.df_calendar = market_calendar()

        df_active_og = None
        if ind_include_cn == 1:
            pd.read_csv(self.paths.ranking_filepath)[["Date", "AG_main", "SI_main"]]
            df_active_og = active_table(self.state.df_mapping, self.paths.active_table_folder)

        all_days = pd.date_range(start=str(startdate), end=str(enddate), freq="B")
        df_all = pd.DataFrame({"Trading date": all_days})
        df_all["Trading date"] = df_all["Trading date"].dt.strftime("%Y%m%d").astype(int)

        times_24h = pd.date_range("00:00", "23:59", freq="T").time
        times_24h = list(times_24h)

        df_all_expanded = df_all.loc[df_all.index.repeat(len(times_24h))].copy()
        df_all_expanded["time"] = times_24h * len(df_all)

        df_all_expanded["DT"] = pd.to_datetime(
            df_all_expanded["Trading date"].astype(str)
            + " "
            + df_all_expanded["time"].astype(str),
            format="%Y%m%d %H:%M:%S",
        )
        df_all_expanded = df_all_expanded.drop(columns="time").reset_index(drop=True)

        nyse_days = self.state.df_calendar["Date_cal"].astype(int).tolist()
        df_all_expanded["ind_nyse_rth"] = (
            df_all_expanded["Trading date"].isin(nyse_days)
            & df_all_expanded["DT"].dt.hour.between(9, 15)
            & (
                (df_all_expanded["DT"].dt.hour != 9)
                | (df_all_expanded["DT"].dt.minute >= 30)
            )
        )
        df_all = df_all_expanded

        hour = df_all["DT"].dt.hour
        minute = df_all["DT"].dt.minute
        df_all["ind_30m"] = pd.NA
        mask_rth = df_all["ind_nyse_rth"]
        df_all.loc[mask_rth, "ind_30m"] = (
            (hour[mask_rth] * 60 + minute[mask_rth]) - 570
        ) // 30 + 1

        et = pytz.timezone("US/Eastern")
        cst = pytz.timezone("Asia/Shanghai")
        df_all["DT_cst"] = df_all["DT"].dt.tz_localize(et).dt.tz_convert(cst)
        hour_cst = df_all["DT_cst"].dt.hour
        minute_cst = df_all["DT_cst"].dt.minute
        df_all["ind_xshg_rth"] = (
            (
                (hour_cst == 9) & (minute_cst >= 30)
                | (hour_cst == 10)
                | (hour_cst == 11)
            )
            | (
                (hour_cst == 13)
                | (hour_cst == 14)
                | ((hour_cst == 15) & (minute_cst == 0))
            )
        )

        df_all = df_all.sort_values("Trading date")

        if ind_include_cn == 1 and df_active_og is not None:
            for ticker in ["AG", "SI"]:
                df_active = (
                    df_active_og[df_active_og["Product"] == ticker.replace("AG", "SAI").lower()][
                        ["Contract", "Startdate", "Enddate"]
                    ]
                    .copy()
                    .rename(columns={"Contract": f"{ticker.upper()}_main"})
                )
                df_active = df_active.sort_values("Startdate")
                df_active[f"{ticker.upper()}_main"] = (
                    df_active[f"{ticker.upper()}_main"]
                    .str.replace("SAI", "AG")
                    .str.replace("sai", "AG")
                )
                df_all = pd.merge_asof(
                    df_all,
                    df_active[["Startdate", "Enddate", f"{ticker.upper()}_main"]],
                    left_on="Trading date",
                    right_on="Startdate",
                    direction="backward",
                )
                df_all[f"{ticker.upper()}_main"] = df_all[f"{ticker.upper()}_main"].where(
                    df_all["Trading date"] <= df_all["Enddate"]
                )
                df_all[f"{ticker.upper()}_main"] = (
                    df_all[f"{ticker.upper()}_main"]
                    .str.replace("^2", "")
                    .apply(modify_contract)
                )
                df_all.drop(columns=["Startdate", "Enddate"], inplace=True)
                df_all = convert_ticker(df_all, ticker, MONTH_MAPPING)

            for col in ["AG_main", "SI_main"]:
                df_all[col] = df_all[col].ffill()

        df_all["Date_str"] = df_all["DT"].astype(str)
        self.state.df_all_copy = df_all.copy()

        self.state.error_dict = {}

        if ind_include_cn == 1:
            product_dict: dict[str, str] = {
                "AG": self.paths.ag_minfolder,
                "SI": self.paths.si_minfolder,
            }
        else:
            product_dict = {}

        for ticker in ticker_list:
            if ticker == "AG":
                product_dict["AG-US"] = self.paths.min_folder
            else:
                product_dict[ticker] = self.paths.min_folder

        df_all = df_all.sort_values(by="DT")
        for ticker, folder in product_dict.items():
            df_concat = pd.DataFrame()
            if ticker == "AG":
                for key in (
                    df_all["Trading date"].astype(str) + "-" + df_all[f"{ticker}_main"]
                ).unique():
                    df_concat = onshore_ETL(
                        df_concat, folder, key, self.state.error_dict
                    )
                self.state.minute_product_frames["AG"] = df_concat.copy()
            elif ticker == "SI":
                for mc in df_all[f"{ticker}_main"].unique():
                    td_list = df_all[df_all[f"{ticker}_main"] == mc]["Trading date"].unique().tolist()
                    df_concat = offshore_ETL(
                        df_concat, folder, mc, td_list, self.state.error_dict
                    )
                self.state.minute_product_frames["SI"] = df_concat.copy()
            elif "finn" in folder.lower():
                for key in (df_all["Trading date"].astype(str) + ticker).unique():
                    try:
                        df_temp = pd.read_csv(
                            rf"{self.paths.min_folder}\\{key.replace('-US', '')}.csv"
                        )
                        if not df_temp.empty:
                            df_temp.rename(
                                columns={"Close": "close", "Open": "open"}, inplace=True
                            )
                            df_temp = df_temp[
                                ["Time ET Rounded", "open", "close"]
                            ].set_index("Time ET Rounded")
                            df_concat = pd.concat([df_concat, df_temp])
                    except Exception as e:
                        self.state.error_dict[key] = e

            df_concat.sort_index(inplace=True)
            df_concat.rename(
                columns={"close": ticker, "open": f"{ticker}_open"},
                inplace=True,
            )
            df_all = df_all.merge(
                df_concat, left_on="Date_str", right_index=True, how="left"
            )
            df_all = df_all.sort_values(by="DT")

            valid = df_all["DT"].diff().dt.total_seconds().between(50, 70)
            safe = valid & valid.shift()
            df_all.loc[safe, f"{ticker}_pct"] = (
                df_all[ticker] / df_all[ticker].shift() - 1
            )
            df_all.loc[safe, f"{ticker}_log"] = (
                np.log(df_all[ticker]) - np.log(df_all[ticker].shift())
            )

            df_all[f"{ticker}_pct_test"] = df_all[f"{ticker}"].pct_change()
            df_all[f"{ticker}_log_test"] = np.log(df_all[f"{ticker}"]).diff()

        s = df_all["DT"].diff().dt.total_seconds()
        valid = s.between(50, 70)
        safe = valid & valid.shift()

        if ind_include_cn == 1:
            df_all.loc[safe, "AG_log_shifted"] = (
                np.log(df_all["AG"].shift(-1)) - np.log(df_all["AG"])
            )
            df_all.loc[safe, "AG_pct_shifted"] = (
                df_all["AG"].shift(-1) / df_all["AG"] - 1
            )

        self.state.df_all = df_all
        self.state.df_all_copy2 = df_all.copy()
        print("Done")

    def run_correlation_nyse_rth(self, method: str = "pct") -> None:
        run_correlation_nyse_rth(self.config, self.state, method=method)

    def run_other_correlation_tests(
        self,
        td_range: list[int] | None = None,
        period: str = "",
    ) -> None:
        run_other_correlation_tests(
            self.config, self.state, self.paths, td_range=td_range, period=period
        )

    def download_daily_ticker_data(self) -> None:
        """Download daily OHLCV via yfinance into ``state.ticker_daily``."""
        self.state.df_calendar = market_calendar()
        ticker_list = self.config.ticker_list

        for ticker in ticker_list:
            self.state.ticker_daily[ticker] = pd.DataFrame()
            try:
                ticker_original = ticker
                ticker_clean = ticker.replace("-US", "")
                df = yf.download(
                    ticker_clean,
                    start="2020-09-01",
                    end=datetime.today().strftime("%Y-%m-%d"),
                    interval="1d",
                )
                df.columns = ["{}_{}".format(col[1], col[0]) for col in df.columns]
                df.reset_index(inplace=True)
                df["Trading date"] = (
                    df["Date"].astype(str).str.replace("-", "").astype(int)
                )
                df = df.merge(
                    self.state.df_calendar[["Date_cal", "Prevdate"]].set_index(
                        "Date_cal"
                    ),
                    left_on="Trading date",
                    right_index=True,
                    how="left",
                )
                df = df.merge(
                    df[["Trading date", f"{ticker_clean}_Open", f"{ticker_clean}_Close"]]
                    .set_index("Trading date")
                    .add_prefix("Prev"),
                    left_on="Prevdate",
                    right_index=True,
                    how="left",
                )
                df.rename(
                    columns={
                        f"Prev{ticker_clean}_Close": f"{ticker_clean}_PrevClose",
                        f"Prev{ticker_clean}_Open": f"{ticker_clean}_PrevOpen",
                    },
                    inplace=True,
                )
                df[f"{ticker_clean}_Prevret"] = (
                    df[f"{ticker_clean}_PrevClose"] / df[f"{ticker_clean}_PrevOpen"] - 1
                )
                df[f"{ticker_clean}_Gapret"] = (
                    df[f"{ticker_clean}_Open"] / df[f"{ticker_clean}_PrevClose"] - 1
                )
                df[f"{ticker_clean}_coc"] = (
                    df[f"{ticker_clean}_Close"] / df[f"{ticker_clean}_PrevClose"] - 1
                )

                df["Log_ret"] = np.log(
                    df[f"{ticker_clean}_Close"] / df[f"{ticker_clean}_Close"].shift(1)
                )
                df["RtRt"] = df["Log_ret"] * df["Log_ret"]
                df["Rtsum"] = (df["RtRt"].rolling(60).sum()).astype(float)
                df["rVola_60"] = np.sqrt(252 / 60 * df["Rtsum"])
                df = df.merge(
                    df[["Trading date", "rVola_60"]]
                    .set_index("Trading date")
                    .add_prefix(f"{ticker_clean}_"),
                    left_on="Prevdate",
                    right_index=True,
                    how="left",
                )
                df.columns = [x.replace(ticker_clean, ticker_original) for x in df.columns]
                self.state.ticker_daily[ticker_original] = df.copy()
            except Exception as e:
                print(f"{ticker}: {e}")

    def build_df_data(self) -> None:
        """Strip log columns and add ``Time`` for daily snapshots."""
        df_data = self.state.df_all_copy2.copy()
        df_data = df_data[[x for x in df_data.columns if "log" not in x]]
        df_data["Time"] = (
            df_data["Date_str"].str.split(" ").str[1].str.replace(":00", "")
        )
        self.state.df_data = df_data
        self.state.df_data_copy = df_data.copy()

    def build_si_daily_stats(self) -> None:
        """SI open/close daily stats (requires ``ind_include_cn``)."""
        if self.config.ind_include_cn != 1:
            return

        df_data = self.state.df_data
        df_pc = df_data[
            (
                df_data["Date_str"]
                .str.slice(start=-8)
                .isin(["09:30:00", "15:56:00", "15:57:00", "15:58:00", "15:59:00"])
            )
            & (df_data["ind_nyse_rth"] == True)
        ].copy()
        df_pc.loc[df_pc["Date_str"].str.slice(start=-8) == "09:30:00", "ind_openclose"] = 1
        df_pc.loc[df_pc["Date_str"].str.slice(start=-8) == "15:59:00", "ind_openclose"] = 0

        df_pcsi = (
            df_pc[["Trading date"]]
            .drop_duplicates()
            .reset_index(drop=True)
            .sort_values(by="Trading date")
        )
        df_pcsi = df_pcsi.merge(
            self.state.df_calendar[["Date_cal", "Prevdate"]].set_index("Date_cal"),
            left_on="Trading date",
            right_index=True,
            how="left",
        )
        for index, row in df_pcsi.iterrows():
            df_pcsi.loc[index, "SI_open"] = df_pc.loc[
                (df_pc["Trading date"] == row["Trading date"])
                & (df_pc["ind_openclose"] == 1)
            ]["SI_open"].iloc[0]
            close_price = df_pc.loc[
                (df_pc["Trading date"] == row["Trading date"])
                & (df_pc["ind_openclose"] == 0)
            ]["SI"].iloc[0]
            if close_price is None or pd.isna(close_price):
                try:
                    close_price = df_pc.loc[
                        (df_pc["Trading date"] == row["Trading date"])
                        & (df_pc["ind_openclose"] != 1)
                        & (df_pc["SI"].notna())
                    ]["SI"].iloc[-1]
                except Exception:
                    pass
            df_pcsi.loc[index, "SI_close"] = close_price

        df_pcsi = df_pcsi.merge(
            df_pcsi[["Trading date", "SI_open", "SI_close"]]
            .set_index("Trading date")
            .add_suffix("_prev"),
            left_on="Prevdate",
            right_index=True,
            how="left",
        )
        df_pcsi = df_pcsi.iloc[1:].reset_index(drop=True)
        df_pcsi["SI_Prevret"] = df_pcsi["SI_close_prev"] / df_pcsi["SI_open_prev"] - 1
        df_pcsi["SI_Gapret"] = df_pcsi["SI_open"] / df_pcsi["SI_close_prev"] - 1
        df_pcsi["SI_Dayret"] = df_pcsi["SI_close"] / df_pcsi["SI_open"] - 1
        df_pcsi["SI_coc"] = df_pcsi["SI_close"] / df_pcsi["SI_close_prev"] - 1
        self.state.df_pcsi = df_pcsi

    def run_opengap_longshort_research(self) -> None:
        run_opengap_longshort_research(self.config, self.state)

    def build_spread_features(self) -> None:
        """Z-scores, forward returns, and trend flags for spreads."""
        df_data = self.state.df_data_copy.copy()
        starting_min = self.config.starting_min
        ticker_list = self.config.ticker_list
        ind_include_cn = self.config.ind_include_cn

        market_open = df_data["DT"].dt.normalize() + pd.Timedelta(hours=9, minutes=30)
        cutoff = market_open + pd.to_timedelta(starting_min - 1, unit="m")
        df_data["ind_start_spread"] = (df_data["DT"] >= cutoff).astype(int)
        if "XAG_close" in df_data:
            df_data.rename(columns={"XAG_close": "XAG"}, inplace=True)

        if ind_include_cn == 1:
            run_list = ["SI", "XAG", "SLV"]
        else:
            run_list = ["SLV"]
        self.state.run_list = run_list

        for cmdty in run_list:
            for ticker in ticker_list:
                if ticker == cmdty:
                    continue
                spread_col = f"{ticker}-{cmdty}"
                beta_col = f"{spread_col}_beta20"

                df_data = attach_lookback_beta(
                    df_data,
                    ticker,
                    cmdty,
                    lookback_days=self.config.backtest_beta_lookback,
                    lag_sessions=1,
                )

                beta = df_data[beta_col]
                spread_level = df_data[ticker] - beta * df_data[cmdty]
                df_data[spread_col] = spread_level
                df_data[f"{spread_col}_pctchg"] = (
                    spread_level.diff() / spread_level.abs()
                )
                valid_mask = df_data["DT"].diff().eq(
                    pd.Timedelta(minutes=1)
                ) & df_data["ind_nyse_rth"]
                df_data[f"{spread_col}_chg"] = spread_level.diff()
                df_data.loc[~valid_mask, f"{spread_col}_chg"] = np.nan
                df_data.loc[~valid_mask, f"{spread_col}_pctchg"] = np.nan
                df_data[f"{spread_col}_sqrd"] = (
                    df_data[f"{spread_col}_chg"]
                    * df_data[f"{spread_col}_chg"]
                )

                col_sqrd = f"{spread_col}_sqrd"
                col_rvol = f"{spread_col}_rvol"

                df_data[col_rvol] = (
                    df_data.groupby(df_data["DT"].dt.date)[col_sqrd]
                    .rolling(window=starting_min - 1, min_periods=starting_min - 1)
                    .mean()
                    .shift(1)
                    .reset_index(level=0, drop=True)
                    .pow(0.5)
                )

                df_data[f"{spread_col}_zscore"] = (
                    df_data[f"{spread_col}_chg"] / df_data[col_rvol]
                )
                df_data[f"{spread_col}_prevzscore"] = df_data[
                    f"{spread_col}_zscore"
                ].shift(1)
                df_data.loc[~valid_mask, f"{spread_col}_prevzscore"] = np.nan

                minutes_to_keep = select_minutes(starting_min)
                df_data = forward_beta_hedged_return(
                    df_data,
                    ticker,
                    cmdty,
                    minutes_to_keep,
                    beta_col=beta_col,
                )

                df_data = compute_ind_trend(
                    df_data,
                    spread_col=spread_col,
                    rvol_col=col_rvol,
                    trend_window=starting_min,
                    trend_mult=1.15,
                    freq_thresh=0.55,
                )

        self.state.df_data = df_data
        print("Done")

    def run_spread_stats(self) -> None:
        """Sweep z-score thresholds and append spread stat tables."""
        df_dict = run_stats_spreads(
            self.state.df_data,
            self.config.stats_zs_list,
            self.config.stats_rp_list,
            self.config.stats_date_ranges,
            self.state.run_list,
            self.config.ticker_list,
        )

        df_appended = pd.DataFrame()
        for df_name, df in df_dict.items():
            df["Key"] = df_name
            df = df[["Key"] + [x for x in df.columns if x != "Key"]]
            df_appended = pd.concat([df_appended, df])

        self.state.df_appended = df_appended

    def run_backtest(self) -> None:
        """Preliminary spread backtest with PnL chart."""
        (
            self.state.backtest_results,
            self.state.backtest_pregroup,
            self.state.backtest_group,
            self.state.backtest_group2,
            self.state.backtest_fig,
            self.state.backtest_summary,
        ) = run_spread_backtest(self.state.df_data, self.config)
        display(self.state.backtest_summary)

    def export_strategy1_results(self, output_dir: str | None = None) -> dict[str, str]:
        """Save Strategy 1 artifacts to local CSV/HTML files for inspection."""
        out_dir = output_dir or self.config.local_output_dir
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        paths: dict[str, str] = {}

        if self.state.df_appended is not None and not self.state.df_appended.empty:
            spread_stats = self.state.df_appended[
                self.state.df_appended["Ticker-Cmdty"].isin(self.config.backtest_spreads)
            ]
            p = os.path.join(out_dir, f"strategy1_spread_stats_{ts}.csv")
            spread_stats.to_csv(p, index=False)
            paths["spread_stats"] = p

            p_all = os.path.join(out_dir, f"strategy1_spread_stats_all_{ts}.csv")
            self.state.df_appended.to_csv(p_all, index=False)
            paths["spread_stats_all"] = p_all

        if self.state.backtest_summary is not None:
            p = os.path.join(out_dir, f"strategy1_backtest_summary_{ts}.csv")
            self.state.backtest_summary.to_csv(p)
            paths["backtest_summary"] = p

        for spread in self.config.backtest_spreads:
            ticker_key = spread.split("-")[0]
            if ticker_key in self.state.backtest_group2:
                p = os.path.join(out_dir, f"strategy1_{spread.replace('-', '_')}_daily_pnl_{ts}.csv")
                self.state.backtest_group2[ticker_key].to_csv(p, index=False)
                paths[f"daily_pnl_{spread}"] = p
            if ticker_key in self.state.backtest_pregroup:
                p = os.path.join(out_dir, f"strategy1_{spread.replace('-', '_')}_trades_{ts}.csv")
                self.state.backtest_pregroup[ticker_key].to_csv(p, index=False)
                paths[f"trades_{spread}"] = p

        if self.state.backtest_fig is not None:
            p = os.path.join(out_dir, f"strategy1_pnl_chart_{ts}.html")
            self.state.backtest_fig.write_html(p)
            paths["pnl_chart"] = p

        if paths:
            print("Strategy 1 exports written:")
            for label, path in paths.items():
                print(f"  {label}: {path}")
        else:
            print("Strategy 1 export: nothing to write (pipeline may not have completed).")

        return paths

    def save_results_excel(self, output_file: str | None = None) -> str:
        """Write open-gap and spread stats to Excel."""
        if output_file is None:
            output_file = (
                f"{self.paths.output_folder}"
                f"{str(datetime.now())[:19].replace('-', '').replace(':', '-').replace(' ', '_')}.xlsx"
            )
        pd.DataFrame().to_excel(output_file, sheet_name="Opengap strat")
        with pd.ExcelWriter(
            output_file, mode="a", engine="openpyxl", if_sheet_exists="overlay"
        ) as writer:
            convert_datetime_cols_to_string(self.state.df_all).to_excel(
                writer, index=False, sheet_name="Opengap strat"
            )
            self.state.df_appended.to_excel(
                writer, index=False, sheet_name="Spread results"
            )
        return output_file

    def create_dash_app(
        self,
        df_test: pd.DataFrame | None = None,
        port: int = 8065,
    ) -> Any:
        return create_dash_app(self.state, df_test=df_test, port=port)
