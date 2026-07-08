"""Entry point: ``python -m silver_backtesting``."""

from __future__ import annotations

from .config import StrategyConfig
from .pipeline import SilverBacktestPipeline
from .report import show_backtest_report


def main() -> None:
    """Run the default pipeline and open the HTML backtest report."""
    config = StrategyConfig()
    pipeline = SilverBacktestPipeline(config)

    pipeline.prepare_minute_data()
    pipeline.download_daily_ticker_data()
    pipeline.build_df_data()

    if config.ind_include_cn == 1:
        pipeline.build_si_daily_stats()
        pipeline.run_opengap_longshort_research()

    pipeline.build_spread_features()
    pipeline.run_spread_stats()
    pipeline.run_backtest()

    show_backtest_report(config, pipeline.state)

    if config.export_results:
        pipeline.export_strategy1_results()


if __name__ == "__main__":
    main()
