from __future__ import annotations

import pandas as pd

def get_config_value(config, path, default=None):
	"""Helper function to get configuration values from either dict or object"""
	if not config:
		return default
	
	keys = path.split('.')
	current = config
	
	for key in keys:
		if isinstance(current, dict):
			if key in current:
				current = current[key]
			else:
				return default
		elif hasattr(current, key):
			current = getattr(current, key)
		else:
			return default
	
	return current


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


def score_technical(ohlcv: pd.DataFrame, rsi_oversold: float = 25, rsi_overbought: float = 75) -> float:
	"""Enhanced tech score in [-1, 1] combining RSI, MACD histogram, and volume analysis.

	- RSI: map oversold->+1, overbought->-1 linearly, clamp.
	- MACD hist: normalize by rolling std to [-1,1] via tanh-like squash.
	- Volume: confirm signals with volume spikes (crypto-specific).
	"""
	rsi = compute_rsi(ohlcv["close"]).iloc[-1]
	# Map RSI to [-1, 1] using configurable thresholds
	if rsi <= rsi_oversold:
		rsi_score = 1.0
	elif rsi >= rsi_overbought:
		rsi_score = -1.0
	else:
		# linear mapping oversold..overbought -> 1..-1
		rsi_score = 1 - (rsi - rsi_oversold) * (2 / (rsi_overbought - rsi_oversold))

	macd_df = compute_macd(ohlcv["close"]) 
	hist = macd_df["hist"].iloc[-50:]
	std = float(hist.std() or 1.0)
	macd_score = float((hist.iloc[-1] / (std * 2)))  # scale
	macd_score = max(-1.0, min(1.0, macd_score))

	# Volume analysis for crypto markets
	volume = ohlcv["volume"].iloc[-20:]  # Last 20 periods
	volume_ma = volume.rolling(10).mean().iloc[-1]
	current_volume = volume.iloc[-1]
	
	# Volume confirmation: boost signal if volume is above average
	volume_ratio = current_volume / volume_ma if volume_ma > 0 else 1.0
	volume_boost = min(0.3, max(-0.3, (volume_ratio - 1.0) * 0.5))  # ±30% boost max
	
	# Combine scores with volume confirmation
	base_score = (rsi_score + macd_score) / 2
	volume_confirmed_score = base_score + volume_boost
	
	return float(max(-1.0, min(1.0, volume_confirmed_score)))


def score_multi_timeframe(symbol: str, primary_timeframe: str = None, config=None) -> float:
	"""Multi-timeframe analysis for all asset types.
	
	Analyzes multiple timeframes to confirm signals:
	- Ultra short: 5m (precise entries)
	- Short: 15m (trend confirmation)  
	- Primary: 1h (overall direction)
	"""
	from .logging_config import get_logger
	logger = get_logger(__name__)
	
	# Use default timeframe from config if not provided
	if primary_timeframe is None:
		from agent.config import load_config_from_env
		if config is None:
			config = load_config_from_env()
		primary_timeframe = config.universe.timeframe
	
	logger.info("=== MULTI-TIMEFRAME ANALYSIS START ===")
	logger.info(f"  - Symbol: {symbol}")
	logger.info(f"  - Primary timeframe: {primary_timeframe}")
	
	try:
		# Import appropriate data fetcher based on symbol type
		if "/" in symbol:
			# Crypto or Forex pairs
			from agent.data.ccxt_client import fetch_ohlcv
			fetch_func = fetch_ohlcv
			logger.info("  - Using CCXT client for crypto data")
		else:
			# Stocks/ETFs
			from agent.data.alpaca_client import fetch_ohlcv
			fetch_func = fetch_ohlcv
			logger.info("  - Using Alpaca client for stock data")
		
		# Get multi-timeframe configuration from config or use defaults
		timeframes_list = get_config_value(config, 'signals.multi_timeframe.timeframes', ["5m", "15m", "1h"])
		weights = get_config_value(config, 'signals.multi_timeframe.weights', {"5m": 0.3, "15m": 0.4, "1h": 0.3})
		data_points = get_config_value(config, 'signals.multi_timeframe.data_points', 50)
		
		logger.info(f"  - Timeframes to analyze: {timeframes_list}")
		logger.info(f"  - Weights: {weights}")
		logger.info(f"  - Data points per timeframe: {data_points}")
		
		# Fetch data for multiple timeframes
		timeframes = {}
		for tf in timeframes_list:
			logger.info(f"  - Fetching data for {tf}...")
			try:
				timeframes[tf] = fetch_func(symbol, tf, data_points)
				logger.info(f"    - {tf}: {len(timeframes[tf])} data points fetched")
			except Exception as fetch_error:
				logger.error(f"    - {tf}: Failed to fetch data - {fetch_error}")
				raise fetch_error
		
		# Calculate scores for each timeframe
		scores = {}
		for tf, df in timeframes.items():
			logger.info(f"  - Calculating technical score for {tf}...")
			try:
				scores[tf] = score_technical(df)
				logger.info(f"    - {tf} technical score: {scores[tf]:.3f}")
			except Exception as score_error:
				logger.error(f"    - {tf}: Failed to calculate score - {score_error}")
				raise score_error
		
		# Calculate weighted average
		logger.info("  - Calculating weighted average...")
		multi_tf_score = sum(scores[tf] * weights[tf] for tf in timeframes)
		logger.info(f"  - Individual scores: {scores}")
		logger.info(f"  - Weighted contributions: {[(tf, scores[tf] * weights[tf]) for tf in timeframes]}")
		logger.info(f"  - Raw multi-timeframe score: {multi_tf_score:.3f}")
		
		final_score = float(max(-1.0, min(1.0, multi_tf_score)))
		logger.info(f"  - Final multi-timeframe score: {final_score:.3f}")
		logger.info("=== MULTI-TIMEFRAME ANALYSIS COMPLETED ===")
		
		return final_score
		
	except Exception as e:
		logger.error(f"=== MULTI-TIMEFRAME ANALYSIS FAILED ===")
		logger.error(f"  - Error: {e}")
		logger.error(f"  - Error type: {type(e)}")
		import traceback
		logger.error(f"  - Traceback: {traceback.format_exc()}")
		logger.warning("  - Falling back to 0.0 score")
		# Fallback to single timeframe if multi-timeframe fails
		return 0.0


def detect_market_regime(ohlcv: pd.DataFrame) -> str:
	"""Detect market regime using EMA(50) vs EMA(200).
	
	Returns:
	- 'BULL': EMA(50) > EMA(200) - bullish trend
	- 'BEAR': EMA(50) < EMA(200) - bearish trend  
	- 'SIDEWAYS': EMA(50) ≈ EMA(200) - sideways market
	"""
	try:
		# Calculate EMAs
		ema_50 = ohlcv["close"].ewm(span=50).mean().iloc[-1]
		ema_200 = ohlcv["close"].ewm(span=200).mean().iloc[-1]
		
		# Calculate percentage difference
		diff_pct = (ema_50 - ema_200) / ema_200 * 100
		
		# Determine regime
		if diff_pct > 2.0:  # EMA50 is 2%+ above EMA200
			return "BULL"
		elif diff_pct < -2.0:  # EMA50 is 2%+ below EMA200
			return "BEAR"
		else:  # Within 2% range
			return "SIDEWAYS"
			
	except Exception as e:
		return "UNKNOWN"


def apply_regime_filter(tech_score: float, market_regime: str, suggestion: str) -> str:
	"""Apply regime filter to trading signals.
	
	- In BULL markets: Only allow BUY signals, convert SELL to NEUTRAL
	- In BEAR markets: Only allow SELL signals, convert BUY to NEUTRAL
	- In SIDEWAYS: Allow both but reduce confidence
	"""
	if market_regime == "BULL":
		if suggestion == "SELL":
			return "NEUTRAL"  # Don't short in bull markets
		elif suggestion == "BUY":
			return "BUY"  # Allow longs in bull markets
	elif market_regime == "BEAR":
		if suggestion == "BUY":
			return "NEUTRAL"  # Don't long in bear markets
		elif suggestion == "SELL":
			return "SELL"  # Allow shorts in bear markets
	elif market_regime == "SIDEWAYS":
		# In sideways markets, be more conservative
		if abs(tech_score) < 0.3:  # Require stronger signals
			return "NEUTRAL"
	
	return suggestion  # Default: keep original suggestion


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


