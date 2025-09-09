from __future__ import annotations

import os
import pandas as pd

from ..logging_config import get_logger

try:
	from alpaca_trade_api.rest import REST  # type: ignore
	_alpaca_ok = True
except Exception:
	_alpaca_ok = False

logger = get_logger(__name__)

def fetch_ohlcv_stub(symbol: str, timeframe: str, limit: int = None) -> pd.DataFrame:
	"""Stub for Alpaca OHLCV. Replace with Alpaca SDK calls later.

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
	ma = base.rolling(7, min_periods=1).mean()
	data = pd.DataFrame({
		"open": 400 + ma,
		"high": 401 + ma,
		"low": 399 + ma,
		"close": 400 + ma,
		"volume": 1_000_000.0,
	}, index=index)
	data.index.name = "time"
	logger.debug(f"Generated stub data: {len(data)} bars, close range: {data['close'].min():.2f}-{data['close'].max():.2f}")
	return data


def fetch_ohlcv(symbol: str, timeframe: str, limit: int = None) -> pd.DataFrame:
	"""Fetch OHLCV via Alpaca if available else fallback to stub.

	Env: ALPACA_KEY_ID, ALPACA_SECRET_KEY, ALPACA_BASE_URL
	Timeframe: map '1h' to '1Hour', '15m' to '15Min', etc.
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
	
	if not _alpaca_ok:
		logger.error("Alpaca SDK not available - cannot fetch real stock data")
		raise RuntimeError("Alpaca SDK not available. Please install alpaca-trade-api to fetch real stock data.")

	key = os.getenv("ALPACA_KEY_ID")
	secret = os.getenv("ALPACA_SECRET_KEY")
	# Try to get base URL from config first, then environment variable
	try:
		from ..config import get_config
		config = get_config()
		base_url = config.exchanges.alpaca_base_url
	except:
		base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
	
	if not key or not secret:
		logger.error("Alpaca API keys not configured")
		raise RuntimeError("Alpaca API keys not configured. Please set ALPACA_KEY_ID and ALPACA_SECRET_KEY environment variables.")

	logger.debug(f"Using Alpaca base URL: {base_url}")

	# crude mapping
	map_tf = {"1h": "1Hour", "15m": "15Min", "5m": "5Min", "1d": "1Day"}
	alp_tf = map_tf.get(timeframe, "1Hour")
	logger.debug(f"Mapped timeframe {timeframe} -> {alp_tf}")
	
	try:
		logger.debug(f"Initializing Alpaca REST client")
		rest = REST(key_id=key, secret_key=secret, base_url=base_url)
		
		logger.debug(f"Requesting {limit} bars from Alpaca for {symbol}")
		bars = rest.get_bars(symbol, timeframe=alp_tf, limit=limit).df
		
		if bars is None or bars.empty:
			logger.error(f"No data returned from Alpaca for {symbol}")
			raise RuntimeError(f"No market data available for {symbol} from Alpaca. Please check the symbol and try again.")
		
		logger.info(f"Received {len(bars)} bars from Alpaca")
		
		df = bars[["open","high","low","close","volume"]].copy()
		df.index.name = "time"
		
		# Log data summary
		logger.debug(f"Data summary: close range {df['close'].min():.2f}-{df['close'].max():.2f}, "
					f"volume range {df['volume'].min():.0f}-{df['volume'].max():.0f}")
		
		return df
		
	except Exception as e:
		logger.error(f"Failed to fetch data from Alpaca: {e}")
		raise RuntimeError(f"Failed to fetch market data for {symbol} from Alpaca: {str(e)}")


