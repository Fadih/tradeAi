from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

from .config import load_config_from_env
from .data.ccxt_client import fetch_ohlcv as ccxt_fetch
from .data.alpaca_client import fetch_ohlcv as alpaca_fetch
from .engine import make_rsi_tip, make_fused_tip
from .notifier import send_notifications
from .models.sentiment import SentimentAnalyzer
from .scheduler import start_scheduler
from .news.rss import fetch_headlines
from .backtest import simulate_fused_strategy
from .tune import grid_search


def _print_banner(compliance_text: str) -> None:
	print("=" * 80)
	print(compliance_text)
	print("=" * 80)


def cmd_show_config() -> None:
	config = load_config_from_env()
	_print_banner(config.guardrails.compliance_banner)
	print(json.dumps({
		"tickers": config.universe.tickers,
		"timeframe": config.universe.timeframe,
		"notifier": config.notifier.mode,
		"sentiment_model": config.models.sentiment_model,
		"timesfm_model": config.models.timesfm_model,
		"thresholds": {
			"buy": config.thresholds.buy_threshold,
			"sell": config.thresholds.sell_threshold,
		},
	}, indent=2))


def cmd_notify_test() -> None:
	config = load_config_from_env()
	_print_banner(config.guardrails.compliance_banner)
	print(f"[{datetime.utcnow().isoformat()}Z] Notification test for {config.universe.tickers} @ {config.universe.timeframe}")
	print("Mode: notifications-only; no trading will be executed.")


def main(argv: list[str] | None = None) -> int:
	parser = argparse.ArgumentParser(description="Trading Agent CLI (notifications-only)")
	sub = parser.add_subparsers(dest="command")

	_ = sub.add_parser("show-config", help="Print the effective configuration")
	_ = sub.add_parser("notify-test", help="Send a console test notification")
	_ = sub.add_parser("tip", help="Compute simple RSI tip for configured tickers")
	_ = sub.add_parser("run-once", help="Compute fused tips (tech+sentiment) and notify")
	sched_p = sub.add_parser("schedule", help="Run fused tips on a schedule (minute granularity)")
	sched_p.add_argument("--cron", default="*/15 * * * *", help="Cron string, e.g. '*/15 * * * *'")
	bt = sub.add_parser("backtest", help="Run a simple backtest on the first ticker")
	bt.add_argument("--bars", type=int, default=500, help="Number of bars to backtest")
	opt = sub.add_parser("tune", help="Grid search to maximize Sharpe on first ticker")
	opt.add_argument("--bars", type=int, default=500, help="Number of bars to use")

	args = parser.parse_args(argv)
	if args.command == "show-config":
		cmd_show_config()
		return 0
	if args.command == "notify-test":
		cmd_notify_test()
		return 0
	if args.command == "tip":
		config = load_config_from_env()
		_print_banner(config.guardrails.compliance_banner)
		tips = []
		for symbol in config.universe.tickers:
			if "/" in symbol:
				data = ccxt_fetch(symbol, config.universe.timeframe)
			else:
				data = alpaca_fetch(symbol, config.universe.timeframe)
			tips.append(make_rsi_tip(symbol, config.universe.timeframe, data))
		send_notifications(config.notifier, tips)
		return 0

	if args.command == "run-once":
		config = load_config_from_env()
		_print_banner(config.guardrails.compliance_banner)
		sent = SentimentAnalyzer(config.models.sentiment_model)
		news_texts = fetch_headlines([
			"https://news.google.com/rss/search?q=stock+market&hl=en-US&gl=US&ceid=US:en",
			"https://news.google.com/rss/search?q=crypto+btc&hl=en-US&gl=US&ceid=US:en",
		])
		sent_score = sent.score(news_texts)
		tips = []
		for symbol in config.universe.tickers:
			data = ccxt_fetch(symbol, config.universe.timeframe) if "/" in symbol else alpaca_fetch(symbol, config.universe.timeframe)
			tips.append(
				make_fused_tip(
					symbol,
					config.universe.timeframe,
					data,
					sentiment_score=sent_score,
					w_tech=0.7,
					w_sent=0.3,
					buy_th=config.thresholds.buy_threshold,
					sell_th=config.thresholds.sell_threshold,
				)
			)
		send_notifications(config.notifier, tips)
		return 0

	if args.command == "schedule":
		config = load_config_from_env()
		_print_banner(config.guardrails.compliance_banner)
		sent = SentimentAnalyzer(config.models.sentiment_model)

		def job():
			news_texts = [
				"Market update: mixed risk sentiment.",
			]
			sent_score = sent.score(news_texts)
			tips = []
			for symbol in config.universe.tickers:
				data = ccxt_fetch(symbol, config.universe.timeframe) if "/" in symbol else alpaca_fetch(symbol, config.universe.timeframe)
				tips.append(make_fused_tip(symbol, config.universe.timeframe, data, sent_score, 0.7, 0.3, config.thresholds.buy_threshold, config.thresholds.sell_threshold))
			send_notifications(config.notifier, tips)

		scheduler = start_scheduler(job, cron=args.cron)
		print("Scheduler started; press Ctrl+C to exit.")
		try:
			import time
			while True:
				time.sleep(1)
		except KeyboardInterrupt:
			print("Exiting scheduler.")
			return 0

	if args.command == "tune":
		config = load_config_from_env()
		_print_banner(config.guardrails.compliance_banner)
		sent = SentimentAnalyzer(config.models.sentiment_model)
		texts = fetch_headlines([
			"https://news.google.com/rss/search?q=stock+market&hl=en-US&gl=US&ceid=US:en",
			"https://news.google.com/rss/search?q=crypto+btc&hl=en-US&gl=US&ceid=US:en",
		])
		sent_score = sent.score(texts)
		symbol = config.universe.tickers[0]
		data = ccxt_fetch(symbol, config.universe.timeframe) if "/" in symbol else alpaca_fetch(symbol, config.universe.timeframe)
		data = data.tail(args.bars)
		res = grid_search(data, sentiment_bias=sent_score)
		bt, st, wt, ws = res.best_params
		print({
			"symbol": symbol,
			"best_params": {"buy_th": bt, "sell_th": st, "w_tech": wt, "w_sent": ws},
			"sharpe": round(res.best.sharpe, 3),
			"max_drawdown": round(res.best.max_drawdown, 3),
			"win_rate": round(res.best.win_rate, 3),
			"trades": res.best.trades,
		})
		return 0

	if args.command == "backtest":
		config = load_config_from_env()
		_print_banner(config.guardrails.compliance_banner)
		sent = SentimentAnalyzer(config.models.sentiment_model)
		texts = fetch_headlines([
			"https://news.google.com/rss/search?q=stock+market&hl=en-US&gl=US&ceid=US:en",
			"https://news.google.com/rss/search?q=crypto+btc&hl=en-US&gl=US&ceid=US:en",
		])
		sent_score = sent.score(texts)
		symbol = config.universe.tickers[0]
		data = ccxt_fetch(symbol, config.universe.timeframe) if "/" in symbol else alpaca_fetch(symbol, config.universe.timeframe)
		data = data.tail(args.bars)
		res = simulate_fused_strategy(
			data,
			sentiment_bias=sent_score,
			buy_th=config.thresholds.buy_threshold,
			sell_th=config.thresholds.sell_threshold,
			w_tech=0.7,
			w_sent=0.3,
		)
		print({
			"symbol": symbol,
			"bars": res.metrics.get("bars"),
			"sharpe": round(res.sharpe, 3),
			"max_drawdown": round(res.max_drawdown, 3),
			"win_rate": round(res.win_rate, 3),
			"trades": res.trades,
		})
		return 0

	parser.print_help()
	return 1


if __name__ == "__main__":
	import sys
	sys.exit(main())


