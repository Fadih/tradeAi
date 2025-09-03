from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class UniverseConfig:
	tickers: List[str] = field(default_factory=lambda: ["BTC/USDT", "ETH/USDT", "SPY"])
	timeframe: str = "1h"  # e.g., "15m", "1h"


@dataclass
class SignalThresholds:
	# Trigger levels for fused score in [-1, 1]
	buy_threshold: float = 0.5
	sell_threshold: float = -0.5


@dataclass
class NotifierConfig:
	mode: str = "console"  # console|telegram|slack
	telegram_token: Optional[str] = None
	telegram_chat_id: Optional[str] = None
	slack_webhook_url: Optional[str] = None


@dataclass
class ModelConfig:
	sentiment_model: str = os.getenv("HF_FIN_SENT_MODEL", "ProsusAI/finbert")
	timesfm_model: Optional[str] = os.getenv("HF_TIMESFM_MODEL", None)


@dataclass
class Guardrails:
	mode: str = "notifications-only"  # enforce: no trading
	compliance_banner: str = (
		"Compliance: Informational alerts only. Not financial advice. Paper trade first."
	)


@dataclass
class AgentConfig:
	universe: UniverseConfig = field(default_factory=UniverseConfig)
	thresholds: SignalThresholds = field(default_factory=SignalThresholds)
	notifier: NotifierConfig = field(default_factory=NotifierConfig)
	models: ModelConfig = field(default_factory=ModelConfig)
	guardrails: Guardrails = field(default_factory=Guardrails)


def load_config_from_env() -> AgentConfig:
	config = AgentConfig()
	# Universe overrides
	if tickers := os.getenv("AGENT_TICKERS"):
		config.universe.tickers = [t.strip() for t in tickers.split(",") if t.strip()]
	if timeframe := os.getenv("AGENT_TIMEFRAME"):
		config.universe.timeframe = timeframe

	# Notifier overrides
	config.notifier.mode = os.getenv("AGENT_NOTIFIER", config.notifier.mode)
	config.notifier.telegram_token = os.getenv("TELEGRAM_BOT_TOKEN", config.notifier.telegram_token)
	config.notifier.telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID", config.notifier.telegram_chat_id)
	config.notifier.slack_webhook_url = os.getenv("SLACK_WEBHOOK_URL", config.notifier.slack_webhook_url)

	# Thresholds overrides
	if buy := os.getenv("AGENT_BUY_THRESHOLD"):
		config.thresholds.buy_threshold = float(buy)
	if sell := os.getenv("AGENT_SELL_THRESHOLD"):
		config.thresholds.sell_threshold = float(sell)

	# Models overrides
	if sent := os.getenv("HF_FIN_SENT_MODEL"):
		config.models.sentiment_model = sent
	if tfm := os.getenv("HF_TIMESFM_MODEL"):
		config.models.timesfm_model = tfm

	return config


