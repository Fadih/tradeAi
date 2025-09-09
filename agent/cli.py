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
from .logging_config import setup_logging, get_logger

# Setup logging first
logger = get_logger(__name__)

def _print_banner(compliance_text: str) -> None:
	print("=" * 80)
	print(compliance_text)
	print("=" * 80)


def cmd_show_config() -> None:
	logger.info("Executing show-config command")
	config = load_config_from_env()
	_print_banner(config.guardrails.compliance_banner)
	
	config_data = {
		"tickers": config.universe.tickers,
		"timeframe": config.universe.timeframe,
		"notifier": config.notifier.mode,
		"sentiment_model": config.models.sentiment_model,
		"timesfm_model": config.models.timesfm_model,
		"thresholds": {
			"buy": config.thresholds.buy_threshold,
			"sell": config.thresholds.sell_threshold,
		},
	}
	
	print(json.dumps(config_data, indent=2))
	logger.info("Configuration displayed successfully")


def cmd_notify_test() -> None:
	logger.info("Executing notify-test command")
	config = load_config_from_env()
	_print_banner(config.guardrails.compliance_banner)
	print(f"[{datetime.utcnow().isoformat()}Z] Notification test for {config.universe.tickers} @ {config.universe.timeframe}")
	print("Mode: notifications-only; no trading will be executed.")
	logger.info("Test notification sent")


def cmd_tip() -> None:
	logger.info("Executing tip command")
	config = load_config_from_env()
	_print_banner(config.guardrails.compliance_banner)
	
	tips = []
	for symbol in config.universe.tickers:
		logger.debug(f"Processing symbol: {symbol}")
		if "/" in symbol:
			data = ccxt_fetch(symbol, config.universe.timeframe)
		else:
			data = alpaca_fetch(symbol, config.universe.timeframe)
		tips.append(make_rsi_tip(symbol, config.universe.timeframe, data))
	
	send_notifications(config.notifier, tips)
	logger.info(f"Generated {len(tips)} RSI tips")


def cmd_run_once() -> None:
	logger.info("Executing run-once command")
	config = load_config_from_env()
	_print_banner(config.guardrails.compliance_banner)
	
	logger.info("Initializing sentiment analyzer")
	sent = SentimentAnalyzer(config.models.sentiment_model)
	
	logger.info("Fetching news headlines for sentiment analysis")
	# Use RSS feeds from configuration
	rss_feeds = config.sentiment_analysis.rss_feeds if config.sentiment_analysis.rss_enabled else []
	news_texts = fetch_headlines(rss_feeds, limit_per_feed=config.sentiment_analysis.rss_max_headlines_per_feed)
	
	logger.info(f"Scoring sentiment for {len(news_texts)} news items")
	sent_score = sent.score(news_texts)
	logger.info(f"Sentiment score: {sent_score:.3f}")
	
	tips = []
	for symbol in config.universe.tickers:
		logger.debug(f"Processing symbol: {symbol}")
		data = ccxt_fetch(symbol, config.universe.timeframe) if "/" in symbol else alpaca_fetch(symbol, config.universe.timeframe)
		tips.append(
			make_fused_tip(
				symbol,
				config.universe.timeframe,
				data,
				sentiment_score=sent_score,
				w_tech=config.signals.weights["technical_weight"],
				w_sent=config.signals.weights["sentiment_weight"],
				buy_th=config.thresholds.buy_threshold,
				sell_th=config.thresholds.sell_threshold,
				config=config,
			)
		)
	
	send_notifications(config.notifier, tips)
	logger.info(f"Generated {len(tips)} fused tips")


def cmd_backtest() -> None:
	logger.info("Executing backtest command")
	config = load_config_from_env()
	_print_banner(config.guardrails.compliance_banner)
	
	logger.info("Initializing sentiment analyzer")
	sent = SentimentAnalyzer(config.models.sentiment_model)
	
	logger.info("Fetching news headlines for sentiment analysis")
	# Use RSS feeds from configuration
	rss_feeds = config.sentiment_analysis.rss_feeds if config.sentiment_analysis.rss_enabled else []
	texts = fetch_headlines(rss_feeds, limit_per_feed=config.sentiment_analysis.rss_max_headlines_per_feed)
	
	logger.info(f"Scoring sentiment for {len(texts)} news items")
	sent_score = sent.score(texts)
	logger.info(f"Sentiment score: {sent_score:.3f}")
	
	symbol = config.universe.tickers[0]
	logger.info(f"Running backtest on {symbol}")
	
	data = ccxt_fetch(symbol, config.universe.timeframe) if "/" in symbol else alpaca_fetch(symbol, config.universe.timeframe)
	data = data.tail(500)  # Use default if not specified
	
	logger.info(f"Backtesting on {len(data)} bars")
	res = simulate_fused_strategy(
		data,
		sentiment_bias=sent_score,
		buy_th=config.thresholds.buy_threshold,
		sell_th=config.thresholds.sell_threshold,
		w_tech=0.7,
		w_sent=0.3,
	)
	
	result = {
		"symbol": symbol,
		"bars": res.metrics.get("bars"),
		"sharpe": round(res.sharpe, 3),
		"max_drawdown": round(res.max_drawdown, 3),
		"win_rate": round(res.win_rate, 3),
		"trades": res.trades,
	}
	
	print(result)
	logger.info(f"Backtest completed: Sharpe={res.sharpe:.3f}, DD={res.max_drawdown:.3f}, Win Rate={res.win_rate:.3f}")


def cmd_tune() -> None:
	logger.info("Executing tune command")
	config = load_config_from_env()
	_print_banner(config.guardrails.compliance_banner)
	
	logger.info("Initializing sentiment analyzer")
	sent = SentimentAnalyzer(config.models.sentiment_model)
	
	logger.info("Fetching news headlines for sentiment analysis")
	# Use RSS feeds from configuration
	rss_feeds = config.sentiment_analysis.rss_feeds if config.sentiment_analysis.rss_enabled else []
	texts = fetch_headlines(rss_feeds, limit_per_feed=config.sentiment_analysis.rss_max_headlines_per_feed)
	
	logger.info(f"Scoring sentiment for {len(texts)} news items")
	sent_score = sent.score(texts)
	logger.info(f"Sentiment score: {sent_score:.3f}")
	
	symbol = config.universe.tickers[0]
	logger.info(f"Running parameter tuning on {symbol}")
	
	data = ccxt_fetch(symbol, config.universe.timeframe) if "/" in symbol else alpaca_fetch(symbol, config.universe.timeframe)
	data = data.tail(500)  # Use default if not specified
	
	logger.info(f"Tuning on {len(data)} bars")
	logger.info("Starting grid search for optimal parameters...")
	
	res = grid_search(data, sentiment_bias=sent_score)
	bt, st, wt, ws = res.best_params
	
	result = {
		"symbol": symbol,
		"best_params": {"buy_th": bt, "sell_th": st, "w_tech": wt, "w_sent": ws},
		"sharpe": round(res.best.sharpe, 3),
		"max_drawdown": round(res.best.max_drawdown, 3),
		"win_rate": round(res.best.win_rate, 3),
		"trades": res.best.trades,
	}
	
	print(result)
	logger.info(f"Tuning completed: Best Sharpe={res.best.sharpe:.3f}, Params: buy_th={bt}, sell_th={st}, w_tech={wt}, w_sent={ws}")


def cmd_schedule() -> None:
	logger.info("Executing schedule command")
	config = load_config_from_env()
	_print_banner(config.guardrails.compliance_banner)
	
	logger.info("Initializing sentiment analyzer")
	sent = SentimentAnalyzer(config.models.sentiment_model)
	
	logger.info("Setting up scheduled job")
	
	def job():
		logger.info("Executing scheduled sentiment analysis and signal generation")
		news_texts = fetch_headlines([
			"https://news.google.com/rss/search?q=stock+market&hl=en-US&gl=US&ceid=US:en",
		])
		sent_score = sent.score(news_texts)
		logger.info(f"Scheduled sentiment score: {sent_score:.3f}")
		
		tips = []
		for symbol in config.universe.tickers:
			logger.debug(f"Processing scheduled symbol: {symbol}")
			data = ccxt_fetch(symbol, config.universe.timeframe) if "/" in symbol else alpaca_fetch(symbol, config.universe.timeframe)
			tips.append(make_fused_tip(symbol, config.universe.timeframe, data, sent_score, 0.7, 0.3, config.thresholds.buy_threshold, config.thresholds.sell_threshold))
		
		send_notifications(config.notifier, tips)
		logger.info(f"Scheduled job completed: {len(tips)} tips generated")
	
	scheduler = start_scheduler(job, cron="*/15 * * * *")
	logger.info("Scheduler started successfully")
	print("Scheduler started; press Ctrl+C to exit.")
	
	try:
		import time
		while True:
			time.sleep(1)
	except KeyboardInterrupt:
		logger.info("Scheduler interrupted by user")
		print("Exiting scheduler.")
		return 0


def main(argv: list[str] | None = None) -> int:
	# Initialize logging
	setup_logging()
	logger.info("Trading Agent CLI starting")
	
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
		cmd_tip()
		return 0
	if args.command == "run-once":
		cmd_run_once()
		return 0
	if args.command == "backtest":
		cmd_backtest()
		return 0
	if args.command == "tune":
		cmd_tune()
		return 0
	if args.command == "schedule":
		cmd_schedule()
		return 0

	parser.print_help()
	return 1


if __name__ == "__main__":
	import sys
	sys.exit(main())


