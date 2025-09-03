from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
import pandas as pd

from .backtest import simulate_fused_strategy, BTResult


@dataclass
class TuneResult:
	best_params: Tuple[float, float, float, float]
	best: BTResult


def grid_search(
	ohlcv: pd.DataFrame,
	sentiment_bias: float,
	buy_th_values: Iterable[float] = (0.3, 0.4, 0.5, 0.6),
	sell_th_values: Iterable[float] = (-0.3, -0.4, -0.5, -0.6),
	w_tech_values: Iterable[float] = (0.5, 0.6, 0.7, 0.8),
	w_sent_values: Iterable[float] = (0.5, 0.4, 0.3, 0.2),
) -> TuneResult:
	best: BTResult | None = None
	best_params = (0.5, -0.5, 0.7, 0.3)
	for bt in buy_th_values:
		for st in sell_th_values:
			for wt in w_tech_values:
				for ws in w_sent_values:
					res = simulate_fused_strategy(ohlcv, sentiment_bias, bt, st, wt, ws, use_regime_filter=True)
					if best is None or res.sharpe > best.sharpe:
						best = res
						best_params = (bt, st, wt, ws)
	assert best is not None
	return TuneResult(best_params=best_params, best=best)


