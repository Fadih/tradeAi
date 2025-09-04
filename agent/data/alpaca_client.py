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

def fetch_ohlcv_stub(symbol: str, timeframe: str, limit: int = 200) -> pd.DataFrame:
	"""Stub for Alpaca OHLCV. Replace with Alpaca SDK calls later.

	Returns minimal DataFrame with columns: time, open, high, low, close, volume.
	"""
	logger.debug(f"Using stub OHLCV for {symbol} @ {timeframe}, limit={limit}")
	
	# Generate realistic stock-like data for SPY
	if symbol == "SPY":
		base_price = 450.0  # Realistic SPY price
		volatility = 0.02   # 2% daily volatility
	else:
		base_price = 100.0  # Generic price
		volatility = 0.01   # 1% daily volatility
	
	import numpy as np
	np.random.seed(42)  # For reproducible data
	
	index = pd.date_range(end=pd.Timestamp.utcnow(), periods=limit, freq="h")
	
	# Generate realistic price movements
	price_changes = np.random.normal(0, volatility/24, limit)  # Hourly volatility
	prices = [base_price]
	
	for i in range(1, limit):
		new_price = prices[-1] * (1 + price_changes[i])
		prices.append(max(new_price, base_price * 0.5))  # Prevent negative prices
	
	# Create OHLCV data
	data = pd.DataFrame({
		"open": prices,
		"high": [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
		"low": [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
		"close": prices,
		"volume": np.random.uniform(1000000, 5000000, limit),
	}, index=index)
	
	# Ensure high >= low and all values are valid
	data["high"] = data[["open", "close", "high"]].max(axis=1)
	data["low"] = data[["open", "close", "low"]].min(axis=1)
	
	# Validate all values are finite and positive
	data = data.replace([np.inf, -np.inf], np.nan)
	data = data.ffill().bfill()
	
	data.index.name = "time"
	logger.debug(f"Generated stub data: {len(data)} bars, close range: {data['close'].min():.2f}-{data['close'].max():.2f}")
	return data


def fetch_ohlcv(symbol: str, timeframe: str, limit: int = 200) -> pd.DataFrame:
	"""Fetch OHLCV via Alpaca if available else fallback to stub.

	Env: ALPACA_KEY_ID, ALPACA_SECRET_KEY, ALPACA_BASE_URL
	Timeframe: map '1h' to '1Hour', '15m' to '15Min', etc.
	"""
	logger.info(f"Fetching OHLCV: {symbol} @ {timeframe}, limit={limit}")
	
	if not _alpaca_ok:
		logger.warning("Alpaca SDK not available, using stub data")
		return fetch_ohlcv_stub(symbol, timeframe, limit)

	key = os.getenv("ALPACA_KEY_ID")
	secret = os.getenv("ALPACA_SECRET_KEY")
	base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
	
	if not key or not secret:
		logger.warning("Alpaca API keys not configured, using stub data")
		return fetch_ohlcv_stub(symbol, timeframe, limit)

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
			logger.warning(f"No data returned from Alpaca for {symbol}, using stub")
			return fetch_ohlcv_stub(symbol, timeframe, limit)
		
		logger.info(f"Received {len(bars)} bars from Alpaca")
		
		df = bars[["open","high","low","close","volume"]].copy()
		df.index.name = "time"
		
		# Log data summary
		logger.debug(f"Data summary: close range {df['close'].min():.2f}-{df['close'].max():.2f}, "
					f"volume range {df['volume'].min():.0f}-{df['volume'].max():.0f}")
		
		return df
		
	except Exception as e:
		logger.error(f"Failed to fetch data from Alpaca: {e}")
		logger.debug(f"Falling back to stub data for {symbol}")
		return fetch_ohlcv_stub(symbol, timeframe, limit)


