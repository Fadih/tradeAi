from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from .indicators import compute_tech_score_series, compute_atr, compute_ema


@dataclass
class BTResult:
	sharpe: float
	max_drawdown: float
	win_rate: float
	trades: int
	metrics: Dict[str, float]


def simulate_fused_strategy(ohlcv: pd.DataFrame, sentiment_bias: float = 0.0, buy_th: float = 0.5, sell_th: float = -0.5, w_tech: float = 0.7, w_sent: float = 0.3, use_regime_filter: bool = True) -> BTResult:
	close = ohlcv["close"].astype(float)
	ret = close.pct_change().fillna(0.0)
	tech_series = compute_tech_score_series(ohlcv)
	score = (w_tech * tech_series + w_sent * sentiment_bias).clip(-1, 1)

	# Positions: 1 for long, -1 for short, 0 for flat
	pos = pd.Series(0, index=close.index, dtype="int8")
	pos = pos.mask(score >= buy_th, 1)
	pos = pos.mask(score <= sell_th, -1)
	pos = pos.ffill().fillna(0)

	# Optional regime filter: allow longs in bull (EMA50>EMA200) and shorts in bear
	if use_regime_filter:
		ema50 = compute_ema(close, 50)
		ema200 = compute_ema(close, 200)
		bull = (ema50 > ema200)
		bear = (ema50 < ema200)
		pos = pos.mask(~bull & (pos > 0), 0)
		pos = pos.mask(~bear & (pos < 0), 0)

	# PnL with simple next-bar return
	strategy_ret = (pos.shift(1).fillna(0) * ret)
	# Basic stats
	sharpe = float((strategy_ret.mean() / (strategy_ret.std() + 1e-12)) * np.sqrt(252*24))
	cum = (1 + strategy_ret).cumprod()
	peak = cum.cummax()
	dd = (cum / peak) - 1
	max_dd = float(dd.min())
	# Trades and win-rate
	changes = pos.diff().fillna(0).abs()
	trades = int((changes > 0).sum())
	non_zero = strategy_ret[strategy_ret != 0]
	win_rate = float((non_zero[non_zero > 0].count()) / max(1, len(non_zero)))

	return BTResult(
		sharpe=sharpe,
		max_drawdown=max_dd,
		win_rate=win_rate,
		trades=trades,
		metrics={"avg_ret": float(strategy_ret.mean()), "std_ret": float(strategy_ret.std()), "bars": int(len(strategy_ret))},
	)


