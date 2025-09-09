from __future__ import annotations

from typing import List, Optional

import pandas as pd
import os

from ..logging_config import get_logger

try:
	import ccxt  # type: ignore
	_ccxt_ok = True
except Exception:
	_ccxt_ok = False

logger = get_logger(__name__)

def fetch_ohlcv_stub(symbol: str, timeframe: str, limit: int = None) -> pd.DataFrame:
	"""Stub for CCXT OHLCV. Replace with live CCXT calls later.

	Returns minimal DataFrame with columns: time, open, high, low, close, volume.
	"""
	# Get default limit from configuration if not provided
	if limit is None:
		try:
			from ..config import get_config
			config = get_config()
			limit = config.technical_analysis.data_fetching_default_limit
		except:
			limit = 200  # Fallback to hardcoded default
	
	logger.debug(f"Using stub OHLCV for {symbol} @ {timeframe}, limit={limit}")
	index = pd.date_range(end=pd.Timestamp.utcnow(), periods=limit, freq="h")
	base = pd.Series(range(limit), dtype="float64")
	ma = base.rolling(5, min_periods=1).mean()
	data = pd.DataFrame({
		"open": 100 + ma,
		"high": 101 + ma,
		"low": 99 + ma,
		"close": 100 + ma,
		"volume": 10_000.0,
	}, index=index)
	data.index.name = "time"
	logger.debug(f"Generated stub data: {len(data)} bars, close range: {data['close'].min():.2f}-{data['close'].max():.2f}")
	return data


def fetch_ohlcv(symbol: str, timeframe: str, limit: int = None) -> pd.DataFrame:
	"""Fetch OHLCV via CCXT if available else fallback to stub.

	Env overrides:
	- CCXT_EXCHANGE (default: binance)
	- CCXT_API_KEY, CCXT_API_SECRET (optional)
	"""
	# Get default limit from configuration if not provided
	if limit is None:
		try:
			from ..config import get_config
			config = get_config()
			limit = config.technical_analysis.data_fetching_default_limit
		except:
			limit = 200  # Fallback to hardcoded default
	
	logger.info(f"Fetching OHLCV: {symbol} @ {timeframe}, limit={limit}")
	
	if not _ccxt_ok:
		logger.error("CCXT not available - cannot fetch real crypto data")
		raise RuntimeError("CCXT library not available. Please install ccxt to fetch real crypto data.")

	# Try to get exchange from config first, then environment variable
	try:
		from ..config import get_config
		config = get_config()
		exchange_name = config.exchanges.ccxt_default_exchange.lower()
	except:
		exchange_name = os.getenv("CCXT_EXCHANGE", "binance").lower()
	logger.debug(f"Using CCXT exchange: {exchange_name}")
	
	exchange_class = getattr(ccxt, exchange_name, None)
	if exchange_class is None:
		logger.error(f"Exchange {exchange_name} not found in CCXT")
		raise RuntimeError(f"Exchange '{exchange_name}' not found in CCXT. Please check your exchange configuration.")
	
	# Setup exchange with optional API keys
	api_key = os.getenv("CCXT_API_KEY")
	api_secret = os.getenv("CCXT_API_SECRET")
	
	if api_key and api_secret:
		logger.debug(f"Using API keys for {exchange_name}")
	else:
		logger.debug(f"No API keys provided for {exchange_name}, using public endpoints")
	
	exchange = exchange_class({
		"apiKey": api_key,
		"secret": api_secret,
		"enableRateLimit": True,
	})
	
	try:
		logger.debug(f"Requesting {limit} bars from {exchange_name}")
		bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
		
		if not bars:
			logger.error(f"No data returned from {exchange_name} for {symbol}")
			raise RuntimeError(f"No market data available for {symbol} from {exchange_name}. Please check the symbol and try again.")
		
		logger.info(f"Received {len(bars)} bars from {exchange_name}")
		
		df = pd.DataFrame(bars, columns=["time","open","high","low","close","volume"]) 
		df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
		df.set_index("time", inplace=True)
		
		# Log data summary
		logger.debug(f"Data summary: close range {df['close'].min():.2f}-{df['close'].max():.2f}, "
					f"volume range {df['volume'].min():.0f}-{df['volume'].max():.0f}")
		
		return df
		
	except Exception as e:
		logger.error(f"Failed to fetch data from {exchange_name}: {e}")
		raise RuntimeError(f"Failed to fetch market data for {symbol} from {exchange_name}: {str(e)}")


