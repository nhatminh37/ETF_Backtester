"""Correlation research and open-gap long/short analysis."""

from __future__ import annotations

import itertools
import math
import os
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from .config import Paths, SilverBacktestState, StrategyConfig
from .loaders import load_XAG
from .spread import get_dynamic_value

try:
    from IPython.display import display
except ImportError:  # pragma: no cover - script mode
    def display(obj: Any) -> None:
        print(obj)


def run_correlation_nyse_rth(
    config: StrategyConfig,
    state: SilverBacktestState,
    method: str = "pct",
) -> None:
    """NYSE RTH correlation tables stored in ``state.corr_tables``."""
    df_test = state.df_all_copy2.copy()
    df_test = df_test[
        (df_test["ind_nyse_rth"] == True)
        & (df_test["Trading date"].between(20251101, 20260201))
    ]
    ticker_list = config.ticker_list
    ticker_cols = [f"{t}_{method}" for t in ticker_list] + [f"XAG_{method}"]
    products = [f"AG_{method}_shifted", f"SI_{method}"]

    corr_matrix = df_test[ticker_cols + products].corr()
    corr_df = corr_matrix.loc[ticker_cols, products]
    corr_df.index = ticker_list + ["XAG"]

    for prod in products:
        base = prod.replace(f"_{method}", "")
        corr_df_prod = corr_df[[prod]].copy()

        cnt_series = pd.Series(
            {
                t: df_test[[f"{t}_{method}", prod]].dropna().shape[0]
                for t in ticker_list + ["XAG"]
            },
            name=f"{prod}_count",
        )
        corr_df_prod = corr_df_prod.join(cnt_series)
        state.corr_tables[base] = corr_df_prod

        for i in sorted(df_test["ind_30m"].unique()):
            df_sub = df_test[df_test["ind_30m"] == i]

            df_corr_temp = df_sub[ticker_cols + [prod]].corr()
            df_corr_temp = df_corr_temp.loc[ticker_cols, [prod]]
            df_corr_temp.index = ticker_list + ["XAG"]
            df_corr_temp.rename(columns={prod: f"{prod}_{i}"}, inplace=True)

            df_cnt_temp = pd.DataFrame(
                {
                    f"{prod}_count_{i}": [
                        df_sub[[col, prod]].dropna().shape[0]
                        for col in ticker_cols
                    ]
                },
                index=ticker_list + ["XAG"],
            )

            state.corr_tables[base] = (
                state.corr_tables[base]
                .merge(df_corr_temp, left_index=True, right_index=True, how="left")
                .merge(df_cnt_temp, left_index=True, right_index=True, how="left")
            )

        corr_cols = [f"{prod}_{i}" for i in range(2, 7)]
        state.corr_tables[base][f"{base}_mean_2_to_6"] = (
            state.corr_tables[base][corr_cols].mean(axis=1)
        )
        state.corr_tables[base] = state.corr_tables[base].dropna(
            axis=1, how="all"
        )
        display(state.corr_tables[base])


def run_other_correlation_tests(
    config: StrategyConfig,
    state: SilverBacktestState,
    paths: Paths,
    td_range: list[int] | None = None,
    period: str = "",
) -> None:
    """Assorted correlation diagnostics."""
    if td_range is None:
        td_range = [20260101, 20260201]

    products = ["XAG", "AG", "SI"]
    methods = ["pct", "log"]

    df_all = state.df_all_copy2.copy()

    if period == "NYSE":
        df_all = df_all[df_all["ind_nyse_rth"] == True]
    elif period == "XHSG":
        df_all = df_all[df_all["ind_xshg_rth"] == True]

    if len(td_range) == 2:
        td_range = sorted(td_range)
        df_all = df_all.loc[
            df_all["Trading date"].between(td_range[0], td_range[1], inclusive="both")
        ]
        print(
            f"Correlations for dates {df_all['Trading date'].min()}-"
            f"{df_all['Trading date'].max()}"
        )
    for method in methods:
        print(f"\n=== {method.upper()} correlations ===")
        for p1, p2 in itertools.combinations(products, 2):
            s1 = df_all[f"{p1}_{method}"]
            s2 = df_all[f"{p2}_{method}"]

            if "AG" in (p1, p2):
                s1 = s1.shift(-1) if p1 == "AG" else s1
                s2 = s2.shift(-1) if p2 == "AG" else s2

            corr = s1.corr(s2)
            print(f"{p1} vs {p2}: {corr:.5f}")

    df_test = state.df_all_copy2.copy()
    df_test = df_test[df_test["Trading date"] > 20260101]

    print(
        f"Date range: {df_test['Trading date'].min()}-{df_test['Trading date'].max()}"
    )
    print(
        f"Correlation using log. Original: "
        f"{round(df_test['AG_log'].corr(df_test['SI_log']), 3)}. "
        f"Shifted: {round(df_test['AG_log_shifted'].corr(df_test['SI_log']), 5)}"
    )
    print(
        f"Correlation using pct. Original: "
        f"{round(df_test['AG_pct'].corr(df_test['SI_pct']), 3)}. "
        f"Shifted: {round(df_test['AG_pct_shifted'].corr(df_test['SI_pct']), 5)}"
    )

    if state.df_XAG is None:
        state.df_XAG = load_XAG(paths.xag_file)

    df_correlation = state.df_XAG.copy()
    s = pd.to_datetime(df_correlation["DT ET"])
    df_correlation["ind_nyse_rth"] = (
        (s.dt.weekday < 5)
        & ((s.dt.hour > 9) | ((s.dt.hour == 9) & (s.dt.minute >= 30)))
        & (s.dt.hour < 16)
    ).astype(int)

    method = "log"
    print(f"Correlation test using XAG file using {method}")

    for prod in ["AG", "SI"]:
        prod_df = state.minute_product_frames.get(prod)
        if prod_df is None:
            continue
        df_correlation = df_correlation.merge(
            prod_df.rename(
                columns={"close": f"{prod}_close", "open": f"{prod}_open"}
            ),
            left_on="DT ET",
            right_index=True,
            how="left",
        )
        df_correlation[f"{prod}_{method}"] = np.log(
            df_correlation[f"{prod}_close"]
        ).diff()
        if prod == "AG":
            df_correlation[f"{prod}_{method}"] = df_correlation[
                f"{prod}_{method}"
            ].shift(-1)
        count = df_correlation.loc[
            (df_correlation[f"{prod}_{method}"].notna())
            & (df_correlation[f"XAG_{method}"].notna())
        ].shape[0]
        crr = round(
            df_correlation[f"{prod}_{method}"].corr(
                df_correlation[f"XAG_{method}"]
            ),
            4,
        )

        df_temp_0 = df_correlation[df_correlation["ind_nyse_rth"] == 0].copy()
        count_0 = df_temp_0.loc[
            (df_temp_0[f"{prod}_{method}"].notna())
            & (df_temp_0[f"XAG_{method}"].notna())
        ].shape[0]
        crr_0 = round(
            df_temp_0[f"{prod}_{method}"].corr(df_temp_0[f"XAG_{method}"]), 4
        )

        df_temp_1 = df_correlation[df_correlation["ind_nyse_rth"] == 1].copy()
        count_1 = df_temp_1.loc[
            (df_temp_1[f"{prod}_{method}"].notna())
            & (df_temp_1[f"XAG_{method}"].notna())
        ].shape[0]
        crr_1 = round(
            df_temp_1[f"{prod}_{method}"].corr(df_temp_1[f"XAG_{method}"]), 4
        )
        mindate = df_correlation["Trading date_utc"].min()
        maxdate = df_correlation["Trading date_utc"].max()
        print(
            f"{mindate}-{maxdate} | Correlation between XAG and {prod}: {crr} "
            f"(count: {count}). During NYSE rth: {crr_1} ({count_1}). "
            f"Outside rth: {crr_0} ({count_0})"
        )

    df_test = state.df_all_copy2.copy()
    print(f"{df_test['Trading date'].min()}-{df_test['Trading date'].max()}")
    r_lag = df_test["PAAS_log"]
    r_lead = df_test["SI_log"]
    for k in range(0, 16):
        corr = r_lag.shift(-k).corr(r_lead)
        print(f"correlation for shift -{k}: {corr}")

    df_test = state.df_all_copy2.copy()
    for td in df_test["Trading date"].unique():
        df_temp = state.df_all_copy2[
            state.df_all_copy2["Trading date"] == td
        ].copy()
        corr_reg = round(df_temp["AG_log"].corr(df_temp["SI_log"]), 2)
        corr_shift = round(
            df_temp["AG_log_shifted"].corr(df_temp["SI_log"]), 2
        )
        wd = datetime.strptime(str(td), "%Y%m%d").strftime("%A")
        print(f"{td} {wd} | regular: {corr_reg}, shifted: {corr_shift}")

    if state.df_test_combined is None:
        df_test_SI = pd.read_csv(
            rf"\\192.168.88.250\BigUSB\LAResearch\Interns\databento\Futures\ohlcv-1m\SI\SI2603\SI2603.csv"
        )
        df_test_SI["time_hk"] = (
            pd.to_datetime(df_test_SI["time_utc"])
            .dt.tz_convert("Asia/Hong_Kong")
            .astype(str)
            .str.split("+")
            .str[0]
        )
        df_test_SI.rename(columns={"close": "SI_close"}, inplace=True)
        df_test_SI = df_test_SI.sort_values(by="time_utc")
        df_test_SI["SI_pct"] = df_test_SI["SI_close"].pct_change()
        df_test_SI["SI_log"] = np.log(df_test_SI["SI_close"]).diff()
        jq_silver_folder = (
            rf"\\192.168.89.201\rsfs\research\Refinitiv\Data\JQData\Futures\1m\XSGE\ag\ag2603\\"
        )
        df_test_AG = pd.DataFrame()
        for file in os.listdir(jq_silver_folder):
            df_temp = pd.read_csv(jq_silver_folder + file)
            df_test_AG = pd.concat([df_test_AG, df_temp])

        df_test_AG.rename(
            columns={"Unnamed: 0": "Datetime HK", "close": "AG_close"},
            inplace=True,
        )
        df_test_AG = df_test_AG.sort_values(by="Datetime HK")
        df_test_AG["AG_close"] = df_test_AG["AG_close"]
        df_test_AG["AG_pct"] = df_test_AG["AG_close"].pct_change()
        df_test_AG["AG_log"] = np.log(df_test_AG["AG_close"]).diff()
        state.df_test_combined = df_test_AG.merge(
            df_test_SI[["time_hk", "SI_close", "SI_pct", "SI_log"]].set_index(
                "time_hk"
            ),
            left_on="Datetime HK",
            right_index=True,
            how="left",
        )
        state.df_test_combined["Date"] = (
            state.df_test_combined["Datetime HK"]
            .str.split(" ")
            .str[0]
            .str.replace("-", "")
            .astype(int)
        )

    products = ["AG", "SI"]
    methods = ["pct", "log"]
    td_range = [20260101, 20260201]

    df_all = state.df_test_combined.copy()

    if len(td_range) == 2:
        td_range = sorted(td_range)
        df_all = df_all.loc[
            df_all["Date"].between(td_range[0], td_range[1], inclusive="both")
        ]
        print(
            f"Correlations for dates {df_all['Date'].min()}-{df_all['Date'].max()}"
        )
    for method in methods:
        print(f"\n=== {method.upper()} correlations ===")
        for p1, p2 in itertools.combinations(products, 2):
            s1 = df_all[f"{p1}_{method}"]
            s2 = df_all[f"{p2}_{method}"]

            if "AG" in (p1, p2):
                s1 = s1.shift(-1) if p1 == "AG" else s1
                s2 = s2.shift(-1) if p2 == "AG" else s2

            corr = s1.corr(s2)
            print(f"{p1} vs {p2}: {corr:.4f}")


def run_opengap_longshort_research(
    config: StrategyConfig,
    state: SilverBacktestState,
) -> None:
    """Cross-sectional open-gap long/short research on ``df_all``."""
    ticker_list = config.ticker_list
    ind_include_cn = config.ind_include_cn
    df_data = state.df_data
    sd = config.opengap_stats_start

    time_dict = {
        "09:30": "1m",
        "09:34": "5m",
        "09:39": "10m",
        "09:59": "30m",
        "10:29": "1h",
        "12:29": "3h",
        "15:59": "eod",
    }
    df_all = (
        state.df_all_copy2[["Trading date"]]
        .drop_duplicates()
        .reset_index(drop=True)
        .copy()
    )

    for timeslot in time_dict.keys():
        if timeslot == "09:30":
            df_open = df_data[df_data["Time"] == timeslot][
                ["Trading date"] + [x for x in df_data.columns if "_open" in x]
            ].copy().set_index("Trading date")
            df_all = df_all.merge(df_open, left_on="Trading date", right_index=True, how="left")
        df_temp = df_data[df_data["Time"] == timeslot][
            ["Trading date", "AG", "SI"] + ticker_list
        ].set_index("Trading date")
        df_temp.columns = [
            x + f"_{timeslot}" for x in df_temp.columns if "date" not in x
        ]
        df_all = df_all.merge(df_temp, left_on="Trading date", right_index=True, how="left")

    state.df_prices = df_all.copy()

    for ticker in ticker_list:
        ticker_df = state.ticker_daily.get(ticker)
        if ticker_df is not None and not ticker_df.empty:
            df_all = df_all.merge(
                ticker_df[
                    [
                        "Trading date",
                        f"{ticker}_Prevret",
                        f"{ticker}_Gapret",
                        f"{ticker}_rVola_60",
                        f"{ticker}_coc",
                    ]
                ].set_index("Trading date"),
                left_on="Trading date",
                right_index=True,
                how="left",
            )

    if state.df_pcsi is not None:
        df_all = df_all.merge(
            state.df_pcsi[
                ["Trading date", "SI_Prevret", "SI_Gapret", "SI_Dayret", "SI_coc"]
            ].set_index("Trading date"),
            left_on="Trading date",
            right_index=True,
            how="left",
        )

    run_list = list(ticker_list)
    if ind_include_cn == 1:
        run_list.append(["AG", "SI"])

    for ticker in run_list:
        df_all[f"{ticker}_Return_1m"] = (
            df_all[f"{ticker}_09:30"] / df_all[f"{ticker}_open"] - 1
        )
        df_all[f"{ticker}_Return_5m"] = (
            df_all[f"{ticker}_09:34"] / df_all[f"{ticker}_open"] - 1
        )
        df_all[f"{ticker}_Return_10m"] = (
            df_all[f"{ticker}_09:39"] / df_all[f"{ticker}_open"] - 1
        )
        if not ticker == "AG":
            df_all[f"{ticker}_Prevgap1m"] = (
                df_all[f"{ticker}_Prevret"]
                + df_all[f"{ticker}_Gapret"]
                + df_all[f"{ticker}_Return_1m"]
            )
        df_all[f"{ticker}_1m-5m"] = (
            df_all[f"{ticker}_09:34"] / df_all[f"{ticker}_09:30"] - 1
        )
        df_all[f"{ticker}_1m-10m"] = (
            df_all[f"{ticker}_09:39"] / df_all[f"{ticker}_09:30"] - 1
        )
        df_all[f"{ticker}_5m-10m"] = (
            df_all[f"{ticker}_09:39"] / df_all[f"{ticker}_09:34"] - 1
        )
        df_all[f"{ticker}_5m-30m"] = (
            df_all[f"{ticker}_09:59"] / df_all[f"{ticker}_09:34"] - 1
        )
        df_all[f"{ticker}_10m-30m"] = (
            df_all[f"{ticker}_09:59"] / df_all[f"{ticker}_09:39"] - 1
        )
        df_all[f"{ticker}_10m-1h"] = (
            df_all[f"{ticker}_10:29"] / df_all[f"{ticker}_09:39"] - 1
        )

    ls_dict = {
        "1m": "_Return_1m",
        "5m": "_Return_5m",
        "10m": "_Return_10m",
        "Prevgap1m": "_Prevgap1m",
        "Prevret": "_Prevret",
    }
    for key, value in ls_dict.items():
        cols = [
            x
            for x in df_all.columns
            if (value in x)
            & ("XAG_" not in x)
            & ("AG_" not in x)
            & ("SI_" not in x)
            & ("SLV_" not in x)
        ]
        max_col = df_all[cols].idxmax(axis=1)
        min_col = df_all[cols].idxmin(axis=1)
        df_all[f"{key}_lowest"] = min_col.str.replace(value, "", regex=False)
        df_all[f"{key}_highest"] = max_col.str.replace(value, "", regex=False)

    lsret_list = [
        "Prevgap1m:1m-5m",
        "1m:1m-5m",
        "1m:1m-10m",
        "5m:5m-10m",
        "5m:5m-30m",
        "10m:10m-30m",
        "10m:10m-1h",
        "Prevret:Gapret",
    ]

    print(f"Starting from {sd}")
    for i in lsret_list:
        key, rt = i.split(":")[0], i.split(":")[1]
        df_all[f"{key}_long_{rt}"] = df_all.apply(
            lambda row: get_dynamic_value(
                row, f"{key}_lowest", "{ticker}_{ret_col}", ret_col=rt
            ),
            axis=1,
        )
        df_all[f"{key}_short_{rt}"] = df_all.apply(
            lambda row: get_dynamic_value(
                row, f"{key}_highest", "{ticker}_{ret_col}", ret_col=rt
            ),
            axis=1,
        )
        df_all[f"{key}_LS_{rt}"] = (
            df_all[f"{key}_long_{rt}"] - df_all[f"{key}_short_{rt}"]
        )
        df_all_stats = df_all[df_all["Trading date"] > sd].copy()
        si_movement = round((df_all_stats[f"SI_{rt}"].mean() * 100), 2)
        long_ret = round((df_all_stats[f"{key}_long_{rt}"].mean() * 100), 2)
        short_ret = round((df_all_stats[f"{key}_short_{rt}"].mean() * 100), 2)
        ls_ret = round((df_all_stats[f"{key}_LS_{rt}"].mean() * 100), 2)
        sharpe = round(
            (
                df_all_stats[f"{key}_LS_{rt}"].mean()
                / df_all_stats[f"{key}_LS_{rt}"].std()
                * math.sqrt(252)
            ),
            2,
        )
        print(
            f"Printing stats for: {key} | {rt}: \tSI_{rt}: {si_movement}% "
            f"\tLong: {long_ret}% \tShort: {short_ret}% \tLS: {ls_ret}% "
            f"\tLS Annualized Sharpe: {sharpe}"
        )

    df_all["Prevret_LS-SLV_Gapret"] = (
        df_all["Prevret_long_Gapret"] - df_all["SLV_Gapret"]
    )
    df_all["1m_LS-SLV_1m-5m"] = (
        df_all["1m_long_1m-5m"] - df_all["SLV_1m-5m"]
    )
    df_all["1m_LS-SLV_1m-5m_60"] = (
        df_all["1m_long_1m-5m"] - df_all["SLV_1m-5m"] * 0.6
    )
    state.df_all = df_all
