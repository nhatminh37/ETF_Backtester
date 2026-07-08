"""Configuration dataclasses and constants."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

MONTH_MAPPING = {
    "F": "01", "G": "02", "H": "03", "J": "04", "K": "05", "M": "06",
    "N": "07", "Q": "08", "U": "09", "V": "10", "X": "11", "Z": "12",
}


@dataclass
class Paths:
    """Network and local data paths used by the notebook."""

    si_minfolder: str = r"\\192.168.88.250\BigUSB\LAResearch\Interns\databento\Futures\ohlcv-1m\SI\\"
    ag_minfolder: str = r"\\192.168.89.201\rsfs\research\Refinitiv\Data\JQData\Futures\1m\XSGE\ag\\"
    ranking_filepath: str = r"\\192.168.88.250\BigUSB\LAResearch\Interns\jqdata\rank.csv"
    min_folder: str = r"\\192.168.88.250\BigUSB\LAResearch\RavenPack\Output\Finnhub\min data\\"
    product_mapping: str = r"\\192.168.88.250\BigUSB\LAResearch\Interns\TD\Mappings\product mapping.xlsx"
    active_table_folder: str = r"\\192.168.88.250\BigUSB\LAResearch\Interns\Active table\\"
    xag_file: str = r"\\192.168.88.250\BigUSB\LAResearch\Interns\CCXT\Binance\Minutes\XAG_consolidated\XAG_2026.csv"
    output_folder: str = r"\\192.168.88.250\BigUSB\LAResearch\Silver stocks strategy\Output\\"
    validation_folder: str = r"\\192.168.88.250\BigUSB\LAResearch\Silver stocks strategy\Validation\\"


@dataclass
class StrategyConfig:
    """Primary strategy and data parameters."""

    startdate: int = 20250101
    ticker_list: list[str] = field(
        default_factory=lambda: [
            "HL", "AG-US", "PAAS", "CDE", "WPM", "FSM", "EXK", "ASM", "SLV"
        ]
    )
    ind_include_cn: int = 0
    starting_min: int = 30

    stats_date_ranges: list[str] = field(
        default_factory=lambda: ["20250101-20251031", "20251101-20260131"]
    )
    stats_zs_list: list[float] = field(default_factory=lambda: [1.5, 2, 3, 4])
    stats_rp_list: list[str] = field(
        default_factory=lambda: ["fwd1m", "fwd5m", "fwd10m", "fwd30m"]
    )

    backtest_thres: float = 3
    backtest_base_notional: float = 10000
    backtest_slippage: float = 0.003
    backtest_spreads: list[str] = field(default_factory=lambda: ["PAAS-SLV"])
    backtest_rt_col: str = "fwd1m"
    backtest_start: int = 20260101
    backtest_thres_max: float = 50
    backtest_side: str = "All"
    backtest_beta_lookback: int = 20

    opengap_stats_start: int = 20251101

    export_results: bool = False
    local_output_dir: str = "output"


@dataclass
class SilverBacktestState:
    """Runtime state mirroring notebook globals."""

    df_calendar: pd.DataFrame | None = None
    df_mapping: pd.DataFrame | None = None
    df_all: pd.DataFrame | None = None
    df_all_copy: pd.DataFrame | None = None
    df_all_copy2: pd.DataFrame | None = None
    df_data: pd.DataFrame | None = None
    df_data_copy: pd.DataFrame | None = None
    df_appended: pd.DataFrame | None = None
    df_prices: pd.DataFrame | None = None
    df_pcsi: pd.DataFrame | None = None
    run_list: list[str] = field(default_factory=list)
    enddate: int = 0
    file_list_og: list[str] = field(default_factory=list)
    error_dict: dict[str, Any] = field(default_factory=dict)
    ticker_daily: dict[str, pd.DataFrame] = field(default_factory=dict)
    corr_tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    backtest_results: dict[str, pd.DataFrame] = field(default_factory=dict)
    backtest_pregroup: dict[str, pd.DataFrame] = field(default_factory=dict)
    backtest_group: dict[str, pd.DataFrame] = field(default_factory=dict)
    backtest_group2: dict[str, pd.DataFrame] = field(default_factory=dict)
    backtest_fig: Any = None
    backtest_summary: pd.DataFrame | None = None
    df_test_combined: pd.DataFrame | None = None
    df_XAG: pd.DataFrame | None = None
    minute_product_frames: dict[str, pd.DataFrame] = field(default_factory=dict)
