"""Data access layer for market data providers.

Expose convenient fetch_ohlcv for modules that import from agent.data.
"""

from .ccxt_client import fetch_ohlcv as fetch_ohlcv  # re-export for convenience
from .alpaca_client import fetch_ohlcv as fetch_alpaca_ohlcv  # named export

__all__ = [
	"ccxt_client",
	"alpaca_client",
	"fetch_ohlcv",
	"fetch_alpaca_ohlcv",
]


