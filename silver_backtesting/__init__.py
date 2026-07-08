"""Silver backtesting package."""

from .config import Paths, SilverBacktestState, StrategyConfig
from .pipeline import SilverBacktestPipeline
from .report import build_backtest_html, show_backtest_report

__all__ = [
    "Paths",
    "SilverBacktestPipeline",
    "SilverBacktestState",
    "StrategyConfig",
    "build_backtest_html",
    "show_backtest_report",
]
