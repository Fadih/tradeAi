from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import pandas as pd

from .indicators import compute_rsi, score_technical, compute_atr
from .logging_config import get_logger
import math

logger = get_logger(__name__)

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


@dataclass
class Tip:
	symbol: str
	timeframe: str
	indicator: str
	value: float
	suggestion: str  # BUY | SELL | NEUTRAL
	meta: Optional[Dict] = None


def make_rsi_tip(symbol: str, timeframe: str, ohlcv: pd.DataFrame, config=None) -> Tip:
	# Use configuration values if available, otherwise use defaults
	if config is None:
		from .config import get_config
		config = get_config()
	
	rsi_period = config.technical_analysis.rsi_period
	rsi_oversold = config.technical_analysis.rsi_oversold
	rsi_overbought = config.technical_analysis.rsi_overbought
	
	rsi = compute_rsi(ohlcv["close"], period=rsi_period).iloc[-1]
	if rsi <= rsi_oversold:
		suggestion = "BUY"
	elif rsi >= rsi_overbought:
		suggestion = "SELL"
	else:
		suggestion = "NEUTRAL"
	
	atr_period = config.technical_analysis.atr_period
	atr_stop_multiplier = config.risk_management.atr_stop_multiplier
	atr_tp_multiplier = config.risk_management.atr_take_profit_multiplier
	
	atr = float(compute_atr(ohlcv["high"], ohlcv["low"], ohlcv["close"], period=atr_period).iloc[-1])
	close_val = ohlcv["close"].ffill().iloc[-1]
	close = float(close_val) if math.isfinite(float(close_val)) else 1.0
	if not math.isfinite(atr) or atr <= 0:
		atr = max(1e-6, 0.01 * close)
	stop = close - atr_stop_multiplier * atr if suggestion == "BUY" else close + atr_stop_multiplier * atr
	tp = close + atr_tp_multiplier * atr if suggestion == "BUY" else close - atr_tp_multiplier * atr
	return Tip(
		symbol=symbol,
		timeframe=timeframe,
		indicator=f"RSI({rsi_period})",
		value=float(rsi),
		suggestion=suggestion,
		meta={"atr": float(atr), "close": close, "stop": stop, "tp": tp},
	)


def make_fused_tip(symbol: str, timeframe: str, ohlcv: pd.DataFrame, sentiment_score: float = 0.0, w_tech: float = 0.5, w_sent: float = 0.5, buy_th: float = 0.2, sell_th: float = -0.2, rsi_oversold: float = 25, rsi_overbought: float = 75, use_multi_timeframe: bool = True, config=None) -> Tip:
	logger.info("=== STARTING FUSED TIP GENERATION ===")
	logger.info(f"Input parameters:")
	logger.info(f"  - Symbol: {symbol}")
	logger.info(f"  - Timeframe: {timeframe}")
	logger.info(f"  - Sentiment Score: {sentiment_score:.3f}")
	logger.info(f"  - Technical Weight: {w_tech:.3f}")
	logger.info(f"  - Sentiment Weight: {w_sent:.3f}")
	logger.info(f"  - Buy Threshold: {buy_th:.3f}")
	logger.info(f"  - Sell Threshold: {sell_th:.3f}")
	logger.info(f"  - RSI Oversold: {rsi_oversold}")
	logger.info(f"  - RSI Overbought: {rsi_overbought}")
	logger.info(f"  - Multi-timeframe: {use_multi_timeframe}")
	logger.info(f"  - OHLCV Data Points: {len(ohlcv)}")
	logger.info(f"  - Latest Close Price: {ohlcv['close'].iloc[-1]:.2f}")
	
	# Detect asset type for appropriate analysis
	is_crypto = "/" in symbol  # Crypto pairs have "/" (e.g., BTC/USDT, ETH/USDT)
	logger.info(f"Asset type detection:")
	logger.info(f"  - Contains '/': {is_crypto}")
	
	# Get asset type configuration from universe
	crypto_symbols = get_config_value(config, 'universe.crypto_symbols',
		["BTC/USDT", "ETH/USDT", "BNB/USDT", "ADA/USDT", "SOL/USDT"])
	stock_symbols = get_config_value(config, 'universe.stock_symbols',
		["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "NVDA"])
	
	logger.info(f"  - Crypto symbols configured: {len(crypto_symbols)} symbols")
	logger.info(f"  - Stock symbols configured: {len(stock_symbols)} symbols")
	
	is_crypto_only = symbol.upper() in [s.upper() for s in crypto_symbols]
	is_stock = symbol.upper() in [s.upper() for s in stock_symbols]
	
	logger.info(f"  - Is crypto only: {is_crypto_only}")
	logger.info(f"  - Is stock: {is_stock}")
	logger.info(f"  - Final asset type: {'crypto' if is_crypto_only else 'stock' if is_stock else 'unknown'}")
	
	# Use multi-timeframe analysis for all symbols when enabled
	logger.info("=== TECHNICAL ANALYSIS ===")
	if use_multi_timeframe:
		logger.info("Using multi-timeframe analysis...")
		try:
			from .indicators import score_multi_timeframe
			tech = score_multi_timeframe(symbol, timeframe, config)
			logger.info(f"  - Multi-timeframe technical score: {tech:.3f}")
		except Exception as e:
			logger.warning(f"  - Multi-timeframe analysis failed: {e}")
			logger.info("  - Falling back to single timeframe analysis")
			# Fallback to single timeframe if multi-timeframe fails
			tech = score_technical(ohlcv, rsi_oversold=rsi_oversold, rsi_overbought=rsi_overbought)
			logger.info(f"  - Single timeframe technical score: {tech:.3f}")
	else:
		logger.info("Using single timeframe analysis...")
		tech = score_technical(ohlcv, rsi_oversold=rsi_oversold, rsi_overbought=rsi_overbought)
		logger.info(f"  - Single timeframe technical score: {tech:.3f}")
	
	# Enhance sentiment based on asset type
	logger.info("=== SENTIMENT ENHANCEMENT ===")
	logger.info(f"  - Original sentiment score: {sentiment_score:.3f}")
	
	if is_crypto_only:
		amplification = get_config_value(config, 'signals.sentiment_enhancement.crypto_amplification', 1.2)
		enhanced_sentiment = sentiment_score * amplification
		logger.info(f"  - Crypto amplification: {amplification:.2f}")
		logger.info(f"  - Enhanced sentiment: {enhanced_sentiment:.3f}")
	elif is_stock:
		amplification = get_config_value(config, 'signals.sentiment_enhancement.stock_amplification', 1.0)
		enhanced_sentiment = sentiment_score * amplification
		logger.info(f"  - Stock amplification: {amplification:.2f}")
		logger.info(f"  - Enhanced sentiment: {enhanced_sentiment:.3f}")
	else:
		# Default to stock amplification for unknown symbols
		amplification = get_config_value(config, 'signals.sentiment_enhancement.stock_amplification', 1.0)
		enhanced_sentiment = sentiment_score * amplification
		logger.info(f"  - Default (stock) amplification: {amplification:.2f}")
		logger.info(f"  - Enhanced sentiment: {enhanced_sentiment:.3f}")
	
	enhanced_sentiment = max(-1.0, min(1.0, enhanced_sentiment))  # Clamp to [-1, 1]
	logger.info(f"  - Clamped enhanced sentiment: {enhanced_sentiment:.3f}")
	
	# Calculate fused score
	logger.info("=== SCORE FUSION ===")
	tech_contribution = w_tech * tech
	sent_contribution = w_sent * enhanced_sentiment
	score = tech_contribution + sent_contribution
	
	logger.info(f"  - Technical contribution: {tech_contribution:.3f} ({w_tech:.2f} × {tech:.3f})")
	logger.info(f"  - Sentiment contribution: {sent_contribution:.3f} ({w_sent:.2f} × {enhanced_sentiment:.3f})")
	logger.info(f"  - Fused score: {score:.3f}")
	
	# Initial signal based on score
	logger.info("=== SIGNAL GENERATION ===")
	logger.info(f"  - Buy threshold: {buy_th:.3f}")
	logger.info(f"  - Sell threshold: {sell_th:.3f}")
	
	if score >= buy_th:
		suggestion = "BUY"
		logger.info(f"  - Signal: BUY (score {score:.3f} ≥ {buy_th:.3f})")
	elif score <= sell_th:
		suggestion = "SELL"
		logger.info(f"  - Signal: SELL (score {score:.3f} ≤ {sell_th:.3f})")
	else:
		suggestion = "NEUTRAL"
		logger.info(f"  - Signal: NEUTRAL ({sell_th:.3f} < score {score:.3f} < {buy_th:.3f})")
	
	# Apply regime filter for all symbols
	logger.info("=== MARKET REGIME ANALYSIS ===")
	market_regime = "N/A"
	if use_multi_timeframe:
		try:
			from .indicators import detect_market_regime, apply_regime_filter
			market_regime = detect_market_regime(ohlcv)
			logger.info(f"  - Detected market regime: {market_regime}")
			original_suggestion = suggestion
			suggestion = apply_regime_filter(tech, market_regime, suggestion)
			if suggestion != original_suggestion:
				logger.info(f"  - Regime filter changed signal: {original_suggestion} → {suggestion}")
			else:
				logger.info(f"  - Regime filter kept signal: {suggestion}")
		except Exception as e:
			logger.warning(f"  - Regime detection failed: {e}")
			# Fallback if regime detection fails
			market_regime = "UNKNOWN"
			logger.info(f"  - Using fallback regime: {market_regime}")
	else:
		logger.info("  - Regime analysis skipped (multi-timeframe disabled)")
	# Enhanced risk management for all asset types
	logger.info("=== RISK MANAGEMENT CALCULATION ===")
	atr = float(compute_atr(ohlcv["high"], ohlcv["low"], ohlcv["close"]).iloc[-1])
	close_val = ohlcv["close"].ffill().iloc[-1]
	close = float(close_val) if math.isfinite(float(close_val)) else 1.0
	if not math.isfinite(atr) or atr <= 0:
		atr = max(1e-6, 0.01 * close)
		logger.warning(f"  - ATR was invalid, using fallback: {atr:.6f}")
	
	logger.info(f"  - ATR: {atr:.4f}")
	logger.info(f"  - Close price: {close:.2f}")
	
	# Calculate volatility-adjusted risk parameters
	volatility = ohlcv["close"].pct_change().std() * 100  # Daily volatility %
	logger.info(f"  - Volatility: {volatility:.2f}%")
	
	# Base multipliers by asset type
	if is_crypto_only:
		asset_type = 'crypto'
		base_stop_multiplier = get_config_value(config, f'signals.risk_management_by_asset.{asset_type}.base_stop_multiplier', 2.0)
		base_tp_multiplier = get_config_value(config, f'signals.risk_management_by_asset.{asset_type}.base_tp_multiplier', 2.5)
		volatility_threshold_high = get_config_value(config, f'signals.risk_management_by_asset.{asset_type}.volatility_threshold_high', 3.0)
		volatility_threshold_low = get_config_value(config, f'signals.risk_management_by_asset.{asset_type}.volatility_threshold_low', 1.0)
		logger.info(f"  - Asset type: {asset_type}")
		logger.info(f"  - Base stop multiplier: {base_stop_multiplier:.2f}")
		logger.info(f"  - Base TP multiplier: {base_tp_multiplier:.2f}")
		logger.info(f"  - Volatility thresholds: {volatility_threshold_low:.1f}% - {volatility_threshold_high:.1f}%")
	else:
		# Default to stock parameters for all non-crypto symbols
		asset_type = 'stock'
		base_stop_multiplier = get_config_value(config, f'signals.risk_management_by_asset.{asset_type}.base_stop_multiplier', 1.5)
		base_tp_multiplier = get_config_value(config, f'signals.risk_management_by_asset.{asset_type}.base_tp_multiplier', 2.0)
		volatility_threshold_high = get_config_value(config, f'signals.risk_management_by_asset.{asset_type}.volatility_threshold_high', 2.0)
		volatility_threshold_low = get_config_value(config, f'signals.risk_management_by_asset.{asset_type}.volatility_threshold_low', 0.5)
		logger.info(f"  - Asset type: {asset_type}")
		logger.info(f"  - Base stop multiplier: {base_stop_multiplier:.2f}")
		logger.info(f"  - Base TP multiplier: {base_tp_multiplier:.2f}")
		logger.info(f"  - Volatility thresholds: {volatility_threshold_low:.1f}% - {volatility_threshold_high:.1f}%")
	
	# Adjust ATR multipliers based on volatility and market regime
	logger.info(f"  - Market regime: {market_regime}")
	if market_regime == "BULL":
		stop_adj = get_config_value(config, f'signals.regime_adjustments.bull_market.{"buy" if suggestion == "BUY" else "sell"}_stop_adjustment', 0.8 if suggestion == "BUY" else 1.3)
		tp_adj = get_config_value(config, f'signals.regime_adjustments.bull_market.{"buy" if suggestion == "BUY" else "sell"}_tp_adjustment', 1.2 if suggestion == "BUY" else 0.8)
		stop_multiplier = base_stop_multiplier * stop_adj
		tp_multiplier = base_tp_multiplier * tp_adj
		logger.info(f"  - Bull market adjustments: stop={stop_adj:.2f}, tp={tp_adj:.2f}")
	elif market_regime == "BEAR":
		stop_adj = get_config_value(config, f'signals.regime_adjustments.bear_market.{"buy" if suggestion == "BUY" else "sell"}_stop_adjustment', 1.3 if suggestion == "BUY" else 0.8)
		tp_adj = get_config_value(config, f'signals.regime_adjustments.bear_market.{"buy" if suggestion == "BUY" else "sell"}_tp_adjustment', 0.8 if suggestion == "BUY" else 1.2)
		stop_multiplier = base_stop_multiplier * stop_adj
		tp_multiplier = base_tp_multiplier * tp_adj
		logger.info(f"  - Bear market adjustments: stop={stop_adj:.2f}, tp={tp_adj:.2f}")
	else:  # SIDEWAYS
		stop_multiplier = base_stop_multiplier
		tp_multiplier = base_tp_multiplier
		logger.info(f"  - Sideways market: no regime adjustments")
	
	logger.info(f"  - Regime-adjusted stop multiplier: {stop_multiplier:.2f}")
	logger.info(f"  - Regime-adjusted TP multiplier: {tp_multiplier:.2f}")
	
	# Adjust for high volatility periods
	if volatility > volatility_threshold_high:  # High volatility
		high_vol_mult = get_config_value(config, 'signals.volatility_adjustments.high_volatility_multiplier', 1.5)
		high_vol_tp_adj = get_config_value(config, 'signals.volatility_adjustments.high_volatility_tp_adjustment', 1.2)
		stop_multiplier *= high_vol_mult
		tp_multiplier *= high_vol_tp_adj
		logger.info(f"  - High volatility detected: {volatility:.2f}% > {volatility_threshold_high:.1f}%")
		logger.info(f"  - High vol adjustments: stop={high_vol_mult:.2f}, tp={high_vol_tp_adj:.2f}")
	elif volatility < volatility_threshold_low:  # Low volatility
		low_vol_mult = get_config_value(config, 'signals.volatility_adjustments.low_volatility_multiplier', 0.8)
		stop_multiplier *= low_vol_mult
		tp_multiplier *= low_vol_mult
		logger.info(f"  - Low volatility detected: {volatility:.2f}% < {volatility_threshold_low:.1f}%")
		logger.info(f"  - Low vol adjustments: stop={low_vol_mult:.2f}, tp={low_vol_mult:.2f}")
	else:
		logger.info(f"  - Normal volatility: {volatility:.2f}% (no volatility adjustments)")
	
	logger.info(f"  - Final stop multiplier: {stop_multiplier:.2f}")
	logger.info(f"  - Final TP multiplier: {tp_multiplier:.2f}")
	
	# Calculate stop loss and take profit
	logger.info("=== STOP LOSS & TAKE PROFIT CALCULATION ===")
	if suggestion == "BUY":
		stop = close - (stop_multiplier * atr)
		tp = close + (tp_multiplier * atr)
		logger.info(f"  - BUY signal: stop below close")
		logger.info(f"  - Stop loss: {stop:.2f} (close {close:.2f} - {stop_multiplier:.2f} × {atr:.4f})")
		logger.info(f"  - Take profit: {tp:.2f} (close {close:.2f} + {tp_multiplier:.2f} × {atr:.4f})")
	elif suggestion == "SELL":
		stop = close + (stop_multiplier * atr)
		tp = close - (tp_multiplier * atr)
		logger.info(f"  - SELL signal: stop above close")
		logger.info(f"  - Stop loss: {stop:.2f} (close {close:.2f} + {stop_multiplier:.2f} × {atr:.4f})")
		logger.info(f"  - Take profit: {tp:.2f} (close {close:.2f} - {tp_multiplier:.2f} × {atr:.4f})")
	else:
		stop = close
		tp = close
		logger.info(f"  - NEUTRAL signal: no stop/tp levels")
		logger.info(f"  - Stop loss: {stop:.2f} (same as close)")
		logger.info(f"  - Take profit: {tp:.2f} (same as close)")
	
	# Calculate risk/reward ratio
	if suggestion == "BUY" and stop < close:
		risk_reward = (tp - close) / (close - stop)
		logger.info(f"  - Risk/Reward ratio: {risk_reward:.2f}")
	elif suggestion == "SELL" and stop > close:
		risk_reward = (close - tp) / (stop - close)
		logger.info(f"  - Risk/Reward ratio: {risk_reward:.2f}")
	else:
		risk_reward = 0.0
		logger.info(f"  - Risk/Reward ratio: {risk_reward:.2f} (neutral signal)")
	# Create final tip with comprehensive metadata
	logger.info("=== CREATING FINAL TIP ===")
	logger.info(f"  - Symbol: {symbol}")
	logger.info(f"  - Timeframe: {timeframe}")
	logger.info(f"  - Indicator: FUSED(tech+sent){'+MTF' if use_multi_timeframe else ''}")
	logger.info(f"  - Final score: {score:.3f}")
	logger.info(f"  - Final suggestion: {suggestion}")
	logger.info(f"  - Technical score: {tech:.3f}")
	logger.info(f"  - Sentiment score: {sentiment_score:.3f}")
	logger.info(f"  - Enhanced sentiment: {enhanced_sentiment:.3f}")
	logger.info(f"  - Market regime: {market_regime}")
	logger.info(f"  - Volatility: {volatility:.2f}%")
	logger.info(f"  - Risk/Reward ratio: {risk_reward:.2f}")
	
	# Determine position size recommendation
	if volatility > volatility_threshold_high:
		position_size = "conservative"
	elif volatility > volatility_threshold_low:
		position_size = "moderate"
	else:
		position_size = "aggressive"
	logger.info(f"  - Position size recommendation: {position_size}")
	
	logger.info("=== FUSED TIP GENERATION COMPLETED ===")
	
	return Tip(
		symbol=symbol,
		timeframe=timeframe,
		indicator="FUSED(tech+sent)" + ("+MTF" if use_multi_timeframe else ""),
		value=float(score),
		suggestion=suggestion,
		meta={
			"tech": float(tech),
			"sent": float(sentiment_score),
			"enhanced_sentiment": float(enhanced_sentiment),
			"atr": float(atr),
			"close": close,
			"stop": stop,
			"tp": tp,
			"multi_timeframe": use_multi_timeframe,
			"volume_boost": "included" if use_multi_timeframe else "single_tf",
			"market_regime": market_regime,
			"regime_filtered": use_multi_timeframe,
			"volatility": float(volatility),
			"stop_multiplier": float(stop_multiplier),
			"tp_multiplier": float(tp_multiplier),
			"risk_reward_ratio": float(risk_reward),
			"position_size_recommendation": position_size,
			"sentiment_enhanced": is_crypto_only,
			"tech_weight": float(w_tech),
			"sentiment_weight": float(w_sent),
			"asset_type": "crypto" if is_crypto_only else "stock",
			"is_crypto": is_crypto_only,
			"is_stock": is_stock,
			"base_stop_multiplier": float(base_stop_multiplier),
			"base_tp_multiplier": float(base_tp_multiplier),
		},
	)


