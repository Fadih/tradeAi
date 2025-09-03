from __future__ import annotations

from typing import List, Optional

import pandas as pd
import os

try:
	import ccxt  # type: ignore
	_ccxt_ok = True
except Exception:
	_ccxt_ok = False


def fetch_ohlcv_stub(symbol: str, timeframe: str, limit: int = 200) -> pd.DataFrame:
	"""Stub for CCXT OHLCV. Replace with live CCXT calls later.

	Returns minimal DataFrame with columns: time, open, high, low, close, volume.
	"""
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
	return data


def fetch_ohlcv(symbol: str, timeframe: str, limit: int = 200) -> pd.DataFrame:
	"""Fetch OHLCV via CCXT if available else fallback to stub.

	Env overrides:
	- CCXT_EXCHANGE (default: binance)
	- CCXT_API_KEY, CCXT_API_SECRET (optional)
	"""
	if not _ccxt_ok:
		return fetch_ohlcv_stub(symbol, timeframe, limit)

	exchange_name = os.getenv("CCXT_EXCHANGE", "binance").lower()
	exchange_class = getattr(ccxt, exchange_name, None)
	if exchange_class is None:
		return fetch_ohlcv_stub(symbol, timeframe, limit)
	exchange = exchange_class({
		"apiKey": os.getenv("CCXT_API_KEY"),
		"secret": os.getenv("CCXT_API_SECRET"),
		"enableRateLimit": True,
	})
	try:
		bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
		if not bars:
			return fetch_ohlcv_stub(symbol, timeframe, limit)
		df = pd.DataFrame(bars, columns=["time","open","high","low","close","volume"]) 
		df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
		df.set_index("time", inplace=True)
		return df
	except Exception:
		return fetch_ohlcv_stub(symbol, timeframe, limit)


