from __future__ import annotations

import os
import pandas as pd

try:
	from alpaca_trade_api.rest import REST  # type: ignore
	_alpaca_ok = True
except Exception:
	_alpaca_ok = False


def fetch_ohlcv_stub(symbol: str, timeframe: str, limit: int = 200) -> pd.DataFrame:
	"""Stub for Alpaca OHLCV. Replace with Alpaca SDK calls later.

	Returns minimal DataFrame with columns: time, open, high, low, close, volume.
	"""
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
	return data


def fetch_ohlcv(symbol: str, timeframe: str, limit: int = 200) -> pd.DataFrame:
	"""Fetch OHLCV via Alpaca if available else fallback to stub.

	Env: ALPACA_KEY_ID, ALPACA_SECRET_KEY, ALPACA_BASE_URL
	Timeframe: map '1h' to '1Hour', '15m' to '15Min', etc.
	"""
	if not _alpaca_ok:
		return fetch_ohlcv_stub(symbol, timeframe, limit)

	key = os.getenv("ALPACA_KEY_ID")
	secret = os.getenv("ALPACA_SECRET_KEY")
	base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
	if not key or not secret:
		return fetch_ohlcv_stub(symbol, timeframe, limit)

	# crude mapping
	map_tf = {"1h": "1Hour", "15m": "15Min", "5m": "5Min", "1d": "1Day"}
	alp_tf = map_tf.get(timeframe, "1Hour")
	try:
		rest = REST(key_id=key, secret_key=secret, base_url=base_url)
		bars = rest.get_bars(symbol, timeframe=alp_tf, limit=limit).df
		if bars is None or bars.empty:
			return fetch_ohlcv_stub(symbol, timeframe, limit)
		df = bars[["open","high","low","close","volume"]].copy()
		df.index.name = "time"
		return df
	except Exception:
		return fetch_ohlcv_stub(symbol, timeframe, limit)


