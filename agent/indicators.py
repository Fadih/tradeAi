from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, Any

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


def compute_rsi(
    close: pd.Series, 
    period: int = 14, 
    method: str = "wilder",
    min_periods: int = None,
    fill_na: bool = False
) -> pd.Series:
    """
    Enhanced RSI with selectable averaging methods for short-term crypto trading.
    
    Args:
        close: Price series
        period: RSI period (7-9 recommended for short-term crypto)
        method: "wilder" (RMA) or "cutler" (simple rolling means)
        min_periods: Minimum periods for calculation
        fill_na: Whether to fill NaN values with 50 (not recommended)
    
    Returns:
        RSI series (0-100) with proper NaN handling for warm-up period
    """
    close = close.astype("float64")
    delta = close.diff()
    
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    
    if min_periods is None:
        min_periods = period
    
    if method.lower() == "wilder":
        # Wilder's RMA via EWM (original method)
        avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=min_periods).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=min_periods).mean()
    elif method.lower() == "cutler":
        # Cutler's simple rolling means (sometimes steadier for short periods)
        avg_gain = gain.rolling(period, min_periods=min_periods).mean()
        avg_loss = loss.rolling(period, min_periods=min_periods).mean()
    else:
        raise ValueError("method must be 'wilder' or 'cutler'")
    
    # Avoid division by zero; map to 0/100 as appropriate
    rs_den = avg_loss.replace(0.0, np.nan)
    rs = avg_gain / rs_den
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    # Handle edge cases properly
    # Where avg_loss == 0 and avg_gain > 0, RSI should be 100
    # Where avg_gain == 0 and avg_loss > 0, RSI should be 0
    rsi = rsi.where(~(avg_loss == 0.0), 100.0).where(~(avg_gain == 0.0), 0.0)
    
    # Clip to valid range
    rsi = rsi.clip(0, 100)
    
    # Only fill NaN if explicitly requested (not recommended for trading)
    if fill_na:
        rsi = rsi.fillna(50.0)
    
    return rsi


def rsi_signal(rsi: pd.Series, signal_len: int = 4) -> pd.Series:
    """
    EMA-smoothed signal line on RSI for cross signals.
    Reduces whipsaws in short-term crypto trading.
    
    Args:
        rsi: RSI series
        signal_len: EMA period for signal line (3-5 recommended)
    
    Returns:
        Smoothed RSI signal line
    """
    return rsi.ewm(span=signal_len, adjust=False, min_periods=signal_len).mean()


def stoch_rsi(rsi: pd.Series, k: int = 14, d: int = 3) -> pd.DataFrame:
    """
    Stochastic RSI: %K and %D (0..100)
    Great for overbought/oversold timing on lower timeframes.
    
    Args:
        rsi: RSI series
        k: Period for %K calculation
        d: Period for %D smoothing
    
    Returns:
        DataFrame with 'stoch_rsi_k' and 'stoch_rsi_d' columns
    """
    rsi_min = rsi.rolling(k).min()
    rsi_max = rsi.rolling(k).max()
    
    # Avoid division by zero
    denominator = rsi_max - rsi_min
    k_line = (rsi - rsi_min) / denominator.replace(0, np.nan)
    k_line = (k_line * 100).clip(0, 100)
    
    d_line = k_line.rolling(d).mean()
    
    return pd.DataFrame({
        "stoch_rsi_k": k_line, 
        "stoch_rsi_d": d_line
    })


def compute_rsi_enhanced(close: pd.Series, period: int = 7, method: str = "wilder") -> dict:
    """
    Enhanced RSI calculation with multiple variants for short-term crypto trading.
    
    Args:
        close: Price series
        period: RSI period (7-9 recommended for short-term)
        method: "wilder" or "cutler"
    
    Returns:
        Dictionary with RSI variants and signal lines
    """
    # Main RSI
    rsi = compute_rsi(close, period=period, method=method, fill_na=False)
    
    # Signal line (EMA of RSI)
    rsi_sig = rsi_signal(rsi, signal_len=4)
    
    # Stochastic RSI
    stoch = stoch_rsi(rsi, k=14, d=3)
    
    return {
        'rsi': rsi,
        'rsi_signal': rsi_sig,
        'stoch_rsi_k': stoch['stoch_rsi_k'],
        'stoch_rsi_d': stoch['stoch_rsi_d']
    }


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


def compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> Dict[str, pd.Series]:
    """
    Calculate Average Directional Index (ADX) for trend strength
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ADX period
        
    Returns:
        Dictionary with ADX, +DI, and -DI values
    """
    # Calculate True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate Directional Movement
    dm_plus = high - high.shift(1)
    dm_minus = low.shift(1) - low
    
    # Filter directional movement
    dm_plus = np.where((dm_plus > dm_minus) & (dm_plus > 0), dm_plus, 0)
    dm_minus = np.where((dm_minus > dm_plus) & (dm_minus > 0), dm_minus, 0)
    
    dm_plus = pd.Series(dm_plus, index=high.index)
    dm_minus = pd.Series(dm_minus, index=high.index)
    
    # Calculate smoothed values using Wilder's smoothing
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    di_plus = 100 * (dm_plus.ewm(alpha=1/period, adjust=False).mean() / atr)
    di_minus = 100 * (dm_minus.ewm(alpha=1/period, adjust=False).mean() / atr)
    
    # Calculate DX
    dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
    dx = dx.fillna(0)
    
    # Calculate ADX
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    
    return {
        'adx': adx,
        'di_plus': di_plus,
        'di_minus': di_minus,
        'atr': atr
    }


def compute_volatility_regime(close: pd.Series, period: int = 20) -> Dict[str, Any]:
    """
    Calculate volatility regime classification
    
    Args:
        close: Close prices
        period: Period for volatility calculation
        
    Returns:
        Dictionary with volatility regime information
    """
    # Calculate returns
    returns = close.pct_change().dropna()
    
    # Calculate rolling volatility
    rolling_vol = returns.rolling(window=period).std()
    
    # Calculate volatility percentiles
    vol_percentile = rolling_vol.rolling(window=period*2).rank(pct=True) * 100
    
    # Classify volatility regime
    def classify_volatility(vol_pct):
        if vol_pct < 25:
            return "low"
        elif vol_pct < 75:
            return "medium"
        else:
            return "high"
    
    vol_regime = vol_percentile.apply(classify_volatility)
    
    # Calculate volatility trend
    vol_trend = rolling_vol.diff(period).apply(lambda x: "increasing" if x > 0 else "decreasing" if x < 0 else "stable")
    
    return {
        'volatility': rolling_vol,
        'volatility_percentile': vol_percentile,
        'volatility_regime': vol_regime,
        'volatility_trend': vol_trend,
        'current_volatility': rolling_vol.iloc[-1] if len(rolling_vol) > 0 else 0.0,
        'current_regime': vol_regime.iloc[-1] if len(vol_regime) > 0 else "medium"
    }


def compute_market_regime(high: pd.Series, low: pd.Series, close: pd.Series, 
                         adx_period: int = 14, vol_period: int = 20) -> Dict[str, Any]:
    """
    Comprehensive market regime detection
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        adx_period: ADX calculation period
        vol_period: Volatility calculation period
        
    Returns:
        Dictionary with comprehensive regime information
    """
    # Calculate ADX for trend strength
    adx_data = compute_adx(high, low, close, adx_period)
    adx = adx_data['adx']
    di_plus = adx_data['di_plus']
    di_minus = adx_data['di_minus']
    
    # Calculate volatility regime
    vol_data = compute_volatility_regime(close, vol_period)
    
    # Determine trend direction
    trend_direction = np.where(di_plus > di_minus, "uptrend", 
                              np.where(di_minus > di_plus, "downtrend", "sideways"))
    trend_direction = pd.Series(trend_direction, index=close.index)
    
    # Classify trend strength
    def classify_trend_strength(adx_val):
        if adx_val > 50:
            return "very_strong"
        elif adx_val > 25:
            return "strong"
        elif adx_val > 15:
            return "moderate"
        else:
            return "weak"
    
    trend_strength = adx.apply(classify_trend_strength)
    
    # Overall market regime classification
    def classify_market_regime(row):
        trend_str = row['trend_strength']
        vol_regime = row['volatility_regime']
        
        if trend_str in ["strong", "very_strong"]:
            return "trending"
        elif vol_regime == "low" and trend_str == "weak":
            return "consolidation"
        elif vol_regime == "high":
            return "volatile"
        else:
            return "ranging"
    
    regime_df = pd.DataFrame({
        'trend_strength': trend_strength,
        'volatility_regime': vol_data['volatility_regime']
    })
    
    market_regime = regime_df.apply(classify_market_regime, axis=1)
    
    return {
        'adx': adx,
        'di_plus': di_plus,
        'di_minus': di_minus,
        'trend_direction': trend_direction,
        'trend_strength': trend_strength,
        'market_regime': market_regime,
        'volatility_data': vol_data,
        'current_adx': adx.iloc[-1] if len(adx) > 0 else 0.0,
        'current_trend_direction': trend_direction.iloc[-1] if len(trend_direction) > 0 else "sideways",
        'current_trend_strength': trend_strength.iloc[-1] if len(trend_strength) > 0 else "weak",
        'current_market_regime': market_regime.iloc[-1] if len(market_regime) > 0 else "ranging"
    }


def compute_advanced_rsi_variants(close: pd.Series, periods: list = [7, 9, 14]) -> Dict[str, Any]:
    """
    Calculate multiple RSI variants with signal lines and crossovers
    
    Args:
        close: Close prices
        periods: List of RSI periods to calculate
        
    Returns:
        Dictionary with multiple RSI variants and signals
    """
    rsi_variants = {}
    
    for period in periods:
        # Calculate RSI
        rsi = compute_rsi(close, period=period)
        
        # Calculate RSI signal line (EMA of RSI)
        rsi_signal = rsi.ewm(span=3).mean()
        
        # Calculate Stochastic RSI
        stoch_rsi_data = stoch_rsi(rsi, k_period=14, d_period=3)
        
        # Detect crossovers
        rsi_cross_up = (rsi > rsi_signal) & (rsi.shift(1) <= rsi_signal.shift(1))
        rsi_cross_down = (rsi < rsi_signal) & (rsi.shift(1) >= rsi_signal.shift(1))
        
        rsi_variants[f'rsi_{period}'] = {
            'rsi': rsi,
            'signal_line': rsi_signal,
            'stoch_k': stoch_rsi_data['stoch_k'],
            'stoch_d': stoch_rsi_data['stoch_d'],
            'cross_up': rsi_cross_up,
            'cross_down': rsi_cross_down,
            'current_rsi': rsi.iloc[-1] if len(rsi) > 0 else 50.0,
            'current_signal': rsi_signal.iloc[-1] if len(rsi_signal) > 0 else 50.0,
            'current_stoch_k': stoch_rsi_data['stoch_k'].iloc[-1] if len(stoch_rsi_data['stoch_k']) > 0 else 50.0,
            'current_stoch_d': stoch_rsi_data['stoch_d'].iloc[-1] if len(stoch_rsi_data['stoch_d']) > 0 else 50.0
        }
    
    return rsi_variants


def compute_dynamic_position_sizing(close: pd.Series, volatility: pd.Series, 
                                  account_balance: float = 10000.0, 
                                  risk_per_trade: float = 0.02) -> Dict[str, Any]:
    """
    Calculate dynamic position sizing based on volatility and risk parameters
    
    Args:
        close: Close prices
        volatility: Volatility series (ATR or rolling std)
        account_balance: Account balance
        risk_per_trade: Risk per trade as percentage (0.02 = 2%)
        
    Returns:
        Dictionary with position sizing information
    """
    # Calculate risk amount
    risk_amount = account_balance * risk_per_trade
    
    # Calculate position size based on volatility
    position_size = risk_amount / volatility
    
    # Calculate position value
    position_value = position_size * close
    
    # Calculate position as percentage of account
    position_percentage = (position_value / account_balance) * 100
    
    # Apply maximum position size limits
    max_position_pct = 10.0  # Maximum 10% of account
    position_percentage = np.minimum(position_percentage, max_position_pct)
    
    # Recalculate position size with limits
    limited_position_value = (position_percentage / 100) * account_balance
    limited_position_size = limited_position_value / close
    
    return {
        'position_size': limited_position_size,
        'position_value': limited_position_value,
        'position_percentage': position_percentage,
        'risk_amount': risk_amount,
        'current_position_size': limited_position_size.iloc[-1] if len(limited_position_size) > 0 else 0.0,
        'current_position_value': limited_position_value.iloc[-1] if len(limited_position_value) > 0 else 0.0,
        'current_position_percentage': position_percentage.iloc[-1] if len(position_percentage) > 0 else 0.0
    }


def compute_volatility_adjusted_stops(close: pd.Series, high: pd.Series, low: pd.Series,
                                    atr: pd.Series, volatility_multiplier: float = 2.0) -> Dict[str, Any]:
    """
    Calculate volatility-adjusted stop losses and take profits
    
    Args:
        close: Close prices
        high: High prices
        low: Low prices
        atr: Average True Range
        volatility_multiplier: Multiplier for ATR-based stops
        
    Returns:
        Dictionary with stop loss and take profit levels
    """
    # Calculate ATR-based stops
    atr_stop_long = close - (atr * volatility_multiplier)
    atr_stop_short = close + (atr * volatility_multiplier)
    
    # Calculate percentage-based stops (backup)
    pct_stop_long = close * (1 - 0.02)  # 2% stop
    pct_stop_short = close * (1 + 0.02)  # 2% stop
    
    # Use the more conservative stop (further from price)
    stop_loss_long = np.maximum(atr_stop_long, pct_stop_long)
    stop_loss_short = np.minimum(atr_stop_short, pct_stop_short)
    
    # Calculate take profits (2:1 risk-reward ratio)
    take_profit_long = close + (2 * (close - stop_loss_long))
    take_profit_short = close - (2 * (stop_loss_short - close))
    
    return {
        'stop_loss_long': stop_loss_long,
        'stop_loss_short': stop_loss_short,
        'take_profit_long': take_profit_long,
        'take_profit_short': take_profit_short,
        'current_stop_long': stop_loss_long.iloc[-1] if len(stop_loss_long) > 0 else close.iloc[-1],
        'current_stop_short': stop_loss_short.iloc[-1] if len(stop_loss_short) > 0 else close.iloc[-1],
        'current_tp_long': take_profit_long.iloc[-1] if len(take_profit_long) > 0 else close.iloc[-1],
        'current_tp_short': take_profit_short.iloc[-1] if len(take_profit_short) > 0 else close.iloc[-1]
    }


def compute_bollinger_bands(close: pd.Series, period: int = 20, std_dev: float = 2.0) -> Dict[str, pd.Series]:
    """
    Calculate Bollinger Bands with squeeze detection
    
    Args:
        close: Price series
        period: Period for moving average
        std_dev: Standard deviation multiplier
        
    Returns:
        Dictionary with upper, middle, lower bands, width, percent, and squeeze detection
    """
    # Calculate middle band (SMA)
    middle = close.rolling(window=period).mean()
    
    # Calculate standard deviation
    std = close.rolling(window=period).std()
    
    # Calculate upper and lower bands
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    
    # Calculate band width
    width = (upper - lower) / middle
    
    # Calculate %B (position within bands)
    percent = (close - lower) / (upper - lower)
    
    # Detect squeeze (when bands are narrow)
    width_ma = width.rolling(window=20).mean()
    squeeze = width < width_ma * 0.8  # Squeeze when width is 20% below average
    
    return {
        'upper': upper,
        'middle': middle,
        'lower': lower,
        'width': width,
        'percent': percent,
        'squeeze': squeeze,
        'squeeze_strength': (width_ma - width) / width_ma  # How strong the squeeze is
    }


def compute_vwap_anchored(ohlcv: pd.DataFrame, anchor_time: str = None) -> Dict[str, pd.Series]:
    """
    Calculate anchored VWAP (Volume Weighted Average Price)
    
    Args:
        ohlcv: DataFrame with OHLCV data
        anchor_time: Time to anchor VWAP (e.g., '09:30' for session start)
        
    Returns:
        Dictionary with VWAP, deviation, and session data
    """
    high = ohlcv['high']
    low = ohlcv['low']
    close = ohlcv['close']
    volume = ohlcv['volume']
    
    # Calculate typical price
    typical_price = (high + low + close) / 3
    
    # Calculate VWAP
    vwap = (typical_price * volume).cumsum() / volume.cumsum()
    
    # Calculate deviation from VWAP
    deviation = ((close - vwap) / vwap) * 10000  # In basis points
    
    # Session-anchored VWAP (if anchor_time provided)
    session_vwap = None
    if anchor_time and 'timestamp' in ohlcv.columns:
        # This would require timestamp-based session detection
        # For now, return daily VWAP
        session_vwap = vwap
    
    return {
        'vwap': vwap,
        'deviation': deviation,
        'session_vwap': session_vwap,
        'typical_price': typical_price
    }


def compute_accumulation_distribution(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Calculate Accumulation/Distribution Line
    
    Args:
        high: High prices
        low: Low prices  
        close: Close prices
        volume: Volume data
        
    Returns:
        Accumulation/Distribution line
    """
    # Calculate Money Flow Multiplier
    mfm = ((close - low) - (high - close)) / (high - low)
    mfm = mfm.fillna(0)  # Handle division by zero
    
    # Calculate Money Flow Volume
    mfv = mfm * volume
    
    # Calculate Accumulation/Distribution Line
    ad_line = mfv.cumsum()
    
    return ad_line


def compute_ma_crossovers_and_slopes(close: pd.Series, periods: list = [5, 9, 12, 21, 26, 50, 200]) -> Dict[str, Any]:
    """
    Calculate moving average crossovers and slopes
    
    Args:
        close: Price series
        periods: List of MA periods
        
    Returns:
        Dictionary with MAs, crossovers, and slopes
    """
    mas = {}
    crossovers = {}
    slopes = {}
    
    # Calculate all moving averages
    for period in periods:
        mas[f'ema_{period}'] = compute_ema(close, period)
        mas[f'sma_{period}'] = close.rolling(window=period).mean()
        
        # Calculate slopes (rate of change)
        slopes[f'ema_{period}_slope'] = mas[f'ema_{period}'].diff()
        slopes[f'sma_{period}_slope'] = mas[f'sma_{period}'].diff()
    
    # Detect crossovers
    if len(periods) >= 2:
        # Fast vs Slow MA crossovers
        fast_period = min(periods)
        slow_period = max(periods)
        
        fast_ma = mas[f'ema_{fast_period}']
        slow_ma = mas[f'ema_{slow_period}']
        
        # Bullish crossover (fast crosses above slow)
        bullish_cross = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
        
        # Bearish crossover (fast crosses below slow)
        bearish_cross = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))
        
        crossovers['bullish'] = bullish_cross
        crossovers['bearish'] = bearish_cross
        crossovers['last_bullish'] = bullish_cross.iloc[-1] if len(bullish_cross) > 0 else False
        crossovers['last_bearish'] = bearish_cross.iloc[-1] if len(bearish_cross) > 0 else False
    
    return {
        'moving_averages': mas,
        'crossovers': crossovers,
        'slopes': slopes
    }


def compute_keltner_channels(high: pd.Series, low: pd.Series, close: pd.Series, 
                           ema_period: int = 20, atr_period: int = 14, 
                           atr_multiplier: float = 2.0) -> Dict[str, pd.Series]:
    """
    Calculate Keltner Channels
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        ema_period: Period for EMA
        atr_period: Period for ATR
        atr_multiplier: ATR multiplier for bands
        
    Returns:
        Dictionary with upper, middle, lower bands and breakout detection
    """
    # Calculate middle band (EMA)
    middle = compute_ema(close, ema_period)
    
    # Calculate ATR
    atr = compute_atr(high, low, close, atr_period)
    
    # Calculate upper and lower bands
    upper = middle + (atr * atr_multiplier)
    lower = middle - (atr * atr_multiplier)
    
    # Detect breakouts
    breakout_up = close > upper.shift(1)
    breakout_down = close < lower.shift(1)
    
    # Channel position (0 = lower band, 1 = upper band)
    channel_position = (close - lower) / (upper - lower)
    
    return {
        'upper': upper,
        'middle': middle,
        'lower': lower,
        'atr': atr,
        'breakout_up': breakout_up,
        'breakout_down': breakout_down,
        'channel_position': channel_position,
        'last_breakout_up': breakout_up.iloc[-1] if len(breakout_up) > 0 else False,
        'last_breakout_down': breakout_down.iloc[-1] if len(breakout_down) > 0 else False
    }


def compute_multi_timeframe_analysis(symbol: str, timeframes: list = ['5m', '15m', '1h', '4h']) -> Dict[str, Any]:
    """
    Perform multi-timeframe analysis
    
    Args:
        symbol: Trading symbol
        timeframes: List of timeframes to analyze
        
    Returns:
        Dictionary with multi-timeframe data
    """
    from agent.data import fetch_ohlcv
    
    mtf_data = {}
    
    for tf in timeframes:
        try:
            # Fetch data for each timeframe
            ohlcv = fetch_ohlcv(symbol, tf, limit=100)
            
            if len(ohlcv) > 0:
                # Calculate basic indicators for each timeframe
                close = ohlcv['close']
                high = ohlcv['high']
                low = ohlcv['low']
                
                # RSI
                rsi = compute_rsi(close, period=14)
                
                # Moving averages
                ema_20 = compute_ema(close, 20)
                ema_50 = compute_ema(close, 50)
                
                # Trend direction
                trend = 'up' if ema_20.iloc[-1] > ema_50.iloc[-1] else 'down'
                
                # Volatility
                atr = compute_atr(high, low, close, 14)
                
                mtf_data[tf] = {
                    'ohlcv': ohlcv,
                    'rsi': rsi.iloc[-1] if len(rsi) > 0 else 50,
                    'ema_20': ema_20.iloc[-1] if len(ema_20) > 0 else close.iloc[-1],
                    'ema_50': ema_50.iloc[-1] if len(ema_50) > 0 else close.iloc[-1],
                    'trend': trend,
                    'atr': atr.iloc[-1] if len(atr) > 0 else 0,
                    'current_price': close.iloc[-1]
                }
                
        except Exception as e:
            mtf_data[tf] = {'error': str(e)}
    
    # Determine overall trend across timeframes
    trends = [data.get('trend', 'neutral') for data in mtf_data.values() if 'trend' in data]
    overall_trend = 'up' if trends.count('up') > trends.count('down') else 'down' if trends.count('down') > trends.count('up') else 'neutral'
    
    mtf_data['overall_trend'] = overall_trend
    mtf_data['trend_consensus'] = trends.count('up') / len(trends) if trends else 0.5
    
    return mtf_data


def compute_cross_asset_correlation(symbols: list = ['BTC/USDT', 'ETH/USDT'], 
                                  timeframe: str = '1h', period: int = 24) -> Dict[str, Any]:
    """
    Calculate cross-asset correlation analysis
    
    Args:
        symbols: List of symbols to correlate
        timeframe: Timeframe for analysis
        period: Number of periods for correlation
        
    Returns:
        Dictionary with correlation data
    """
    from agent.data import fetch_ohlcv
    
    correlation_data = {}
    prices = {}
    
    # Fetch data for all symbols
    for symbol in symbols:
        try:
            ohlcv = fetch_ohlcv(symbol, timeframe, limit=period)
            if len(ohlcv) > 0:
                prices[symbol] = ohlcv['close'].pct_change().dropna()
                correlation_data[symbol] = {
                    'current_price': ohlcv['close'].iloc[-1],
                    'price_change_24h': ((ohlcv['close'].iloc[-1] - ohlcv['close'].iloc[-period]) / ohlcv['close'].iloc[-period]) * 100 if len(ohlcv) >= period else 0
                }
        except Exception as e:
            correlation_data[symbol] = {'error': str(e)}
    
    # Calculate correlations
    if len(prices) >= 2:
        price_df = pd.DataFrame(prices)
        correlation_matrix = price_df.corr()
        
        # Get BTC correlation with other assets
        btc_correlations = {}
        if 'BTC/USDT' in correlation_matrix.columns:
            for symbol in correlation_matrix.columns:
                if symbol != 'BTC/USDT':
                    btc_correlations[symbol] = correlation_matrix.loc['BTC/USDT', symbol]
        
        correlation_data['correlation_matrix'] = correlation_matrix
        correlation_data['btc_correlations'] = btc_correlations
        correlation_data['avg_correlation'] = correlation_matrix.mean().mean()
    
    return correlation_data


def compute_btc_dominance() -> Dict[str, Any]:
    """
    Calculate BTC dominance and market sentiment
    
    Returns:
        Dictionary with BTC dominance data
    """
    try:
        from agent.data import fetch_ohlcv
        
        # Fetch BTC and ETH data for dominance calculation
        btc_data = fetch_ohlcv('BTC/USDT', '1h', limit=24)
        eth_data = fetch_ohlcv('ETH/USDT', '1h', limit=24)
        
        if len(btc_data) > 0 and len(eth_data) > 0:
            btc_market_cap = btc_data['close'].iloc[-1] * 21000000  # Approximate BTC supply
            eth_market_cap = eth_data['close'].iloc[-1] * 120000000  # Approximate ETH supply
            
            total_crypto_cap = btc_market_cap + eth_market_cap
            btc_dominance = (btc_market_cap / total_crypto_cap) * 100
            
            # Calculate dominance trend
            btc_dominance_24h_ago = 50  # Placeholder - would need historical data
            dominance_change = btc_dominance - btc_dominance_24h_ago
            
            return {
                'btc_dominance': btc_dominance,
                'dominance_change': dominance_change,
                'btc_market_cap': btc_market_cap,
                'eth_market_cap': eth_market_cap,
                'total_crypto_cap': total_crypto_cap,
                'dominance_trend': 'increasing' if dominance_change > 0 else 'decreasing'
            }
    except Exception as e:
        return {'error': str(e), 'btc_dominance': 50, 'dominance_change': 0}


def compute_market_wide_sentiment() -> Dict[str, Any]:
    """
    Calculate market-wide sentiment indicators
    
    Returns:
        Dictionary with market sentiment data
    """
    try:
        from agent.data import fetch_ohlcv
        
        # Fetch data for major cryptocurrencies
        symbols = ['BTC/USDT', 'ETH/USDT', 'ADA/USDT', 'DOT/USDT', 'LINK/USDT']
        market_data = {}
        
        for symbol in symbols:
            try:
                ohlcv = fetch_ohlcv(symbol, '1h', limit=24)
                if len(ohlcv) > 0:
                    close = ohlcv['close']
                    price_change_24h = ((close.iloc[-1] - close.iloc[-24]) / close.iloc[-24]) * 100 if len(close) >= 24 else 0
                    
                    market_data[symbol] = {
                        'price_change_24h': price_change_24h,
                        'current_price': close.iloc[-1],
                        'sentiment': 'bullish' if price_change_24h > 0 else 'bearish'
                    }
            except Exception as e:
                market_data[symbol] = {'error': str(e)}
        
        # Calculate market-wide sentiment
        sentiments = [data.get('sentiment', 'neutral') for data in market_data.values() if 'sentiment' in data]
        bullish_count = sentiments.count('bullish')
        bearish_count = sentiments.count('bearish')
        total_count = len(sentiments)
        
        market_sentiment_score = (bullish_count - bearish_count) / total_count if total_count > 0 else 0
        
        return {
            'market_data': market_data,
            'market_sentiment_score': market_sentiment_score,
            'bullish_count': bullish_count,
            'bearish_count': bearish_count,
            'total_count': total_count,
            'overall_sentiment': 'bullish' if market_sentiment_score > 0.2 else 'bearish' if market_sentiment_score < -0.2 else 'neutral'
        }
        
    except Exception as e:
        return {'error': str(e), 'market_sentiment_score': 0, 'overall_sentiment': 'neutral'}


