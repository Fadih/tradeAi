from __future__ import annotations

import pandas as pd


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
	"""Compute RSI using Wilder's smoothing.

	Returns a Series aligned to input index.
	"""
	delta = close.diff()
	gain = delta.clip(lower=0)
	loss = -delta.clip(upper=0)
	avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
	avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
	rs = avg_gain / avg_loss.replace(0, pd.NA)
	rsi = 100 - (100 / (1 + rs))
	return rsi.fillna(50.0)


def compute_ema(series: pd.Series, span: int) -> pd.Series:
	return series.ewm(span=span, adjust=False).mean()


def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
	fast_ema = compute_ema(close, fast)
	slow_ema = compute_ema(close, slow)
	macd = fast_ema - slow_ema
	signal_line = compute_ema(macd, signal)
	hist = macd - signal_line
	return pd.DataFrame({"macd": macd, "signal": signal_line, "hist": hist})


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
	prev_close = close.shift(1)
	tr = pd.concat([
		(high - low),
		(high - prev_close).abs(),
		(low - prev_close).abs(),
	], axis=1).max(axis=1)
	atr = tr.ewm(alpha=1 / period, adjust=False).mean()
	return atr.bfill().fillna(0.0)


def score_technical(ohlcv: pd.DataFrame) -> float:
	"""Simple tech score in [-1, 1] combining RSI and MACD histogram.

	- RSI: map 30->+1, 70->-1 linearly, clamp.
	- MACD hist: normalize by rolling std to [-1,1] via tanh-like squash.
	"""
	rsi = compute_rsi(ohlcv["close"]).iloc[-1]
	# Map RSI to [-1, 1]
	if rsi <= 30:
		rsi_score = 1.0
	elif rsi >= 70:
		rsi_score = -1.0
	else:
		# linear mapping 30..70 -> 1..-1
		rsi_score = 1 - (rsi - 30) * (2 / 40)

	macd_df = compute_macd(ohlcv["close"]) 
	hist = macd_df["hist"].iloc[-50:]
	std = float(hist.std() or 1.0)
	macd_score = float((hist.iloc[-1] / (std * 2)))  # scale
	macd_score = max(-1.0, min(1.0, macd_score))

	# Average for now
	return float(max(-1.0, min(1.0, (rsi_score + macd_score) / 2)))


def compute_tech_score_series(ohlcv: pd.DataFrame) -> pd.Series:
	"""Vectorized technical score per bar in [-1, 1] using RSI and MACD hist.

	This mirrors score_technical but returns a Series aligned with ohlcv index.
	"""
	rsi_series = compute_rsi(ohlcv["close"]).clip(0, 100)
	# Map RSI to [-1, 1] piecewise-linear around 30/70
	rsi_score = pd.Series(index=rsi_series.index, dtype="float64")
	rsi_score = 1 - (rsi_series - 30) * (2 / 40)
	rsi_score = rsi_score.where(rsi_series.between(30, 70), rsi_score)
	rsi_score = rsi_score.mask(rsi_series < 30, 1.0)
	rsi_score = rsi_score.mask(rsi_series > 70, -1.0)

	macd_df = compute_macd(ohlcv["close"]).fillna(0.0)
	hist = macd_df["hist"]
	std = hist.rolling(50, min_periods=10).std().replace(0, pd.NA).bfill().fillna(1.0)
	macd_score = (hist / (std * 2)).clip(-1, 1)

	return ((rsi_score + macd_score) / 2).clip(-1, 1)


