from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import pandas as pd

from .indicators import compute_rsi, score_technical, compute_atr
import math


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


def make_fused_tip(symbol: str, timeframe: str, ohlcv: pd.DataFrame, sentiment_score: float = 0.0, w_tech: float = 0.6, w_sent: float = 0.4, buy_th: float = 0.5, sell_th: float = -0.5) -> Tip:
	tech = score_technical(ohlcv)
	score = w_tech * tech + w_sent * sentiment_score
	if score >= buy_th:
		suggestion = "BUY"
	elif score <= sell_th:
		suggestion = "SELL"
	else:
		suggestion = "NEUTRAL"
	atr = float(compute_atr(ohlcv["high"], ohlcv["low"], ohlcv["close"]).iloc[-1])
	close_val = ohlcv["close"].ffill().iloc[-1]
	close = float(close_val) if math.isfinite(float(close_val)) else 1.0
	if not math.isfinite(atr) or atr <= 0:
		atr = max(1e-6, 0.01 * close)
	stop = close - 2 * atr if suggestion == "BUY" else close + 2 * atr
	tp = close + 3 * atr if suggestion == "BUY" else close - 3 * atr
	return Tip(
		symbol=symbol,
		timeframe=timeframe,
		indicator="FUSED(tech+sent)",
		value=float(score),
		suggestion=suggestion,
		meta={
			"tech": float(tech),
			"sent": float(sentiment_score),
			"atr": float(atr),
			"close": close,
			"stop": stop,
			"tp": tp,
		},
	)


