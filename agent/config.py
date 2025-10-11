from __future__ import annotations

import os
import yaml
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path


@dataclass
class UniverseConfig:
	timeframe: str = "5m"  # Default timeframe optimized for short-term crypto trading
	timeframes: List[str] = field(default_factory=lambda: ["1m", "5m", "15m", "1h"])  # Available timeframes
	# Separate lists for asset type detection
	crypto_symbols: List[str] =field(default_factory=list)
	stock_symbols: List[str] = field(default_factory=list)



@dataclass
class SignalThresholds:
	# Trigger levels for fused score in [-1, 1]
	buy_threshold: float = 0.5
	sell_threshold: float = -0.5
	# Weight defaults for signal fusion
	technical_weight: float = 0.6
	sentiment_weight: float = 0.4


@dataclass
class NotifierConfig:
	mode: str = "console"  # console|telegram|slack
	telegram_token: Optional[str] = None
	telegram_chat_id: Optional[str] = None
	slack_webhook_url: Optional[str] = None
	# New notification configuration from app.yaml
	enabled: bool = True
	channels: List[str] = field(default_factory=lambda: ["console"])
	webhook_timeout_seconds: int = 10
	webhook_retry_attempts: int = 3
	webhook_retry_delay_seconds: int = 5


@dataclass
class SafetyConfig:
	# Incomplete candle handling
	allow_incomplete_candles: bool = False
	
	# Data validation
	min_data_points_for_indicators: int = 50
	safety_margin: int = 10
	
	# Candle freshness validation (in minutes)
	max_candle_staleness: Dict[str, int] = field(default_factory=lambda: {
		"1m": 2, "5m": 8, "15m": 20, "1h": 70
	})
	
	# Threshold validation
	validate_thresholds: bool = True
	normalize_weights: bool = True
	
	# Sentiment analysis
	deterministic_sampling: bool = True
	sentiment_timeout_seconds: int = 30
	
	# External data timeouts
	market_data_timeout_seconds: int = 15
	rss_timeout_seconds: int = 10
	reddit_timeout_seconds: int = 15
	
	# Circuit breaker settings
	circuit_breaker_enabled: bool = True
	circuit_breaker_failure_threshold: int = 3
	circuit_breaker_cooldown_minutes: int = 5


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
class AppConfig:
	name: str = "Trading AI Tips"
	version: str = "2.0.0"
	description: str = "AI-Powered Trading Signals & Analysis Platform"
	author: str = "Fadi Hussein"
	license: str = "MIT"
	website: str = "https://github.com/fadi-hussein/tradeAi"
	maintenance_message: str = ""

@dataclass
class ServerConfig:
	host: str = "0.0.0.0"
	port: int = 8000
	debug: bool = False
	reload: bool = False
	workers: int = 1

@dataclass
class DatabaseConfig:
	redis_host: str = "redis"
	redis_port: int = 6379
	redis_db: int = 0
	redis_password: str = ""
	max_connections: int = 10
	socket_timeout: int = 5
	socket_connect_timeout: int = 5
	retry_on_timeout: bool = True

@dataclass
class SecurityConfig:
	jwt_secret_key: str = "your-secret-key-change-in-production"
	jwt_algorithm: str = "HS256"
	access_token_expire_minutes: int = 1440  # 24 hours
	password_min_length: int = 8
	require_special_chars: bool = True
	require_numbers: bool = True
	require_uppercase: bool = True
	session_timeout_hours: int = 24
	max_login_attempts: int = 5
	lockout_duration_minutes: int = 30
	# Default admin user configuration
	default_admin_username: str = "admin"
	default_admin_password: str = "admin123"

@dataclass
class APIConfig:
	rate_limit_requests_per_minute: int = 100
	rate_limit_burst_limit: int = 200
	cors_allowed_origins: List[str] = field(default_factory=lambda: ["*"])
	cors_allowed_methods: List[str] = field(default_factory=lambda: ["GET", "POST", "PUT", "DELETE", "OPTIONS"])
	cors_allowed_headers: List[str] = field(default_factory=lambda: ["*"])
	cors_allow_credentials: bool = True

@dataclass
class TelegramNotificationConfig:
	signal_created: bool = True
	signal_changed: bool = True
	signal_deleted: bool = True
	maintenance: bool = True
	system_errors: bool = False
	max_notifications_per_minute: int = 10
	cooldown_seconds: int = 5

@dataclass
class TelegramConfig:
	bot_token: str = ""
	chat_id: str = ""
	enabled: bool = False
	api_url: str = "https://api.telegram.org/bot"
	timeout: int = 10
	max_retries: int = 3
	notifications: TelegramNotificationConfig = field(default_factory=TelegramNotificationConfig)

@dataclass
class MonitoringConfig:
	signal_check_interval_minutes: int = 3
	health_check_interval_seconds: int = 30
	max_concurrent_monitors: int = 10
	history_retention_days: int = 30

@dataclass
class LoggingConfig:
	level: str = ""
	format: str = ""
	console_enabled: bool = False
	file_enabled: bool = False
	file_path: str = ""
	max_bytes: int = 0
	backup_count: int = 0
	encoding: str = ""
	colored: bool = False
	timestamp: bool = False

@dataclass
class FeaturesConfig:
	signal_generation: bool = True
	signal_monitoring: bool = True
	position_tracking: bool = True
	user_management: bool = True
	admin_dashboard: bool = True
	api_documentation: bool = True
	real_time_updates: bool = True
	sentiment_analysis: bool = True
	technical_analysis: bool = True
	reddit_integration: bool = True
	news_integration: bool = True

@dataclass
class DevelopmentConfig:
	debug_mode: bool = False
	verbose_logging: bool = False
	mock_data: bool = False
	hot_reload: bool = False
	profiling: bool = False

@dataclass
class RSIConfig:
	"""Enhanced RSI configuration for short-term crypto trading"""
	# Basic RSI settings
	period: int = 7  # Optimized for short-term crypto trading
	method: str = "wilder"  # "wilder" or "cutler"
	signal_period: int = 4  # EMA smoothing for RSI signal line
	
	# Stochastic RSI settings
	stoch_rsi: dict = field(default_factory=lambda: {
		"k_period": 14,
		"d_period": 3
	})
	
	# Dynamic thresholds based on market regime
	thresholds: dict = field(default_factory=lambda: {
		"overbought": 70,  # Standard thresholds (for ranging markets)
		"oversold": 30,
		"overbought_tight": 55,  # Tight thresholds (for trending markets)
		"oversold_tight": 45,
		"overbought_extreme": 80,  # Extreme thresholds (for mean reversion)
		"oversold_extreme": 20
	})
	
	# Cross signal thresholds (for RSI signal line crosses)
	cross_thresholds: dict = field(default_factory=lambda: {
		"buy_cross": 45,  # Buy when RSI crosses above signal below this level
		"sell_cross": 55  # Sell when RSI crosses below signal above this level
	})

@dataclass
class MACDConfig:
	"""MACD configuration for short-term crypto trading"""
	fast_period: int = 4  # Ultra-fast for short-term crypto scalping
	slow_period: int = 9  # Ultra-fast for short-term crypto scalping
	signal_period: int = 3  # Ultra-fast for short-term crypto scalping

@dataclass
class ATRConfig:
	"""ATR configuration for risk management"""
	period: int = 7  # Optimized for short-term crypto trading
	multiplier: float = 2.0  # ATR multiplier for stop loss/take profit

@dataclass
class TechnicalAnalysisConfig:
	# Data fetching configuration
	data_fetching_default_limit: int = 200
	data_fetching_max_limit: int = 1000
	data_fetching_min_limit: int = 50
	
	# Enhanced indicator configurations
	rsi: RSIConfig = field(default_factory=RSIConfig)
	macd: MACDConfig = field(default_factory=MACDConfig)
	atr: ATRConfig = field(default_factory=ATRConfig)
	
	# Legacy fields for backward compatibility (deprecated)
	rsi_period: int = 7  # Now uses rsi.period
	rsi_overbought: int = 70  # Now uses rsi.thresholds.overbought
	rsi_oversold: int = 30  # Now uses rsi.thresholds.oversold
	macd_fast: int = 4  # Now uses macd.fast_period
	macd_slow: int = 9  # Now uses macd.slow_period
	macd_signal: int = 3  # Now uses macd.signal_period
	atr_period: int = 7  # Now uses atr.period
	atr_multiplier: float = 2.0  # Now uses atr.multiplier
	
	# Moving averages
	sma_periods: List[int] = field(default_factory=lambda: [20, 50, 200])
	ema_periods: List[int] = field(default_factory=lambda: [12, 26, 50])

@dataclass
class SentimentAnalysisConfig:
	model_name: str = "ProsusAI/finbert"
	max_length: int = 512
	batch_size: int = 32
	rss_enabled: bool = True
	rss_feeds: List[str] = field(default_factory=lambda: [
		"https://news.google.com/rss/search?q=stock+market&hl=en-US&gl=US&ceid=US:en",
		"https://news.google.com/rss/search?q=crypto+btc&hl=en-US&gl=US&ceid=US:en"
	])
	rss_max_headlines_per_feed: int = 15
	rss_sample_size: int = 15
	reddit_enabled: bool = True
	reddit_subreddits: List[str] = field(default_factory=lambda: [
		"CryptoCurrency", "Bitcoin", "ethereum", "investing", "stocks"
	])
	reddit_max_posts_per_subreddit: int = 10
	reddit_sample_size: int = 20
	hours_back: int = 3  # Time filtering for short-term trading sentiment
	positive_threshold: float = 0.1
	negative_threshold: float = -0.1
	neutral_range: List[float] = field(default_factory=lambda: [-0.1, 0.1])

@dataclass
class ExchangeConfig:
	ccxt_default_exchange: str = "binance"
	ccxt_rate_limit: bool = True
	ccxt_timeout: int = 30000
	ccxt_retries: int = 3
	alpaca_base_url: str = "https://paper-api.alpaca.markets"
	alpaca_data_url: str = "https://data.alpaca.markets"
	alpaca_timeout: int = 30
	alpaca_retries: int = 3

@dataclass
class RiskManagementConfig:
	stop_loss_percentage: float = 0.02  # 2%
	take_profit_percentage: float = 0.04  # 4%
	max_position_size: float = 0.1  # 10% of portfolio
	max_daily_signals: int = 10
	atr_stop_multiplier: float = 1.5
	atr_take_profit_multiplier: float = 2.5

@dataclass
class PositionTrackingConfig:
	enabled: bool = True
	update_interval_minutes: int = 5
	max_positions_per_user: int = 50
	auto_close_on_signal: bool = False
	max_holding_days: int = 30
	min_holding_minutes: int = 15

@dataclass
class BacktestingConfig:
	start_date: str = "2024-01-01"
	end_date: str = "2024-12-31"
	initial_capital: float = 10000
	commission: float = 0.001  # 0.1%
	metrics: List[str] = field(default_factory=lambda: [
		"sharpe_ratio", "max_drawdown", "win_rate", "total_return", "volatility"
	])

@dataclass
class SignalsConfig:
	# Default thresholds for signal generation
	thresholds: Dict[str, float] = field(default_factory=lambda: {
		"buy_threshold": 0.2,
		"sell_threshold": -0.2
	})
	
	# Weight configuration for signal fusion
	weights: Dict[str, float] = field(default_factory=lambda: {
		"technical_weight": 0.5,
		"sentiment_weight": 0.5
	})
	
	
	# Sentiment enhancement by asset type
	sentiment_enhancement: Dict[str, float] = field(default_factory=lambda: {
		"crypto_amplification": 1.2,
		"stock_amplification": 1.0
	})
	
	# Risk management by asset type
	risk_management_by_asset: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
		"crypto": {
			"base_stop_multiplier": 2.0,
			"base_tp_multiplier": 2.5,
			"volatility_threshold_high": 3.0,
			"volatility_threshold_low": 1.0
		},
		"stock": {
			"base_stop_multiplier": 1.5,
			"base_tp_multiplier": 2.0,
			"volatility_threshold_high": 2.0,
			"volatility_threshold_low": 0.5
		}
	})
	
	# Market regime adjustments
	regime_adjustments: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
		"bull_market": {
			"buy_stop_adjustment": 0.8,
			"sell_stop_adjustment": 1.3,
			"buy_tp_adjustment": 1.2,
			"sell_tp_adjustment": 0.8
		},
		"bear_market": {
			"buy_stop_adjustment": 1.3,
			"sell_stop_adjustment": 0.8,
			"buy_tp_adjustment": 0.8,
			"sell_tp_adjustment": 1.2
		}
	})
	
	# Volatility adjustments
	volatility_adjustments: Dict[str, float] = field(default_factory=lambda: {
		"high_volatility_multiplier": 1.5,
		"low_volatility_multiplier": 0.8,
		"high_volatility_tp_adjustment": 1.2
	})
	
	# Multi-timeframe analysis configuration
	multi_timeframe: Dict[str, Any] = field(default_factory=lambda: {
		"enabled": True,
		"timeframes": ["15m", "1h", "4h"],
		"weights": {"15m": 0.2, "1h": 0.5, "4h": 0.3},
		"data_points": 50
	})


@dataclass
class AgentConfig:
	# Application configuration
	app: AppConfig = field(default_factory=AppConfig)
	server: ServerConfig = field(default_factory=ServerConfig)
	database: DatabaseConfig = field(default_factory=DatabaseConfig)
	security: SecurityConfig = field(default_factory=SecurityConfig)
	api: APIConfig = field(default_factory=APIConfig)
	telegram: TelegramConfig = field(default_factory=TelegramConfig)
	monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
	logging: LoggingConfig = field(default_factory=LoggingConfig)
	features: FeaturesConfig = field(default_factory=FeaturesConfig)
	development: DevelopmentConfig = field(default_factory=DevelopmentConfig)
	
	# Trading configuration
	universe: UniverseConfig = field(default_factory=UniverseConfig)
	signals: SignalsConfig = field(default_factory=SignalsConfig)
	notifier: NotifierConfig = field(default_factory=NotifierConfig)
	models: ModelConfig = field(default_factory=ModelConfig)
	guardrails: Guardrails = field(default_factory=Guardrails)
	safety: SafetyConfig = field(default_factory=SafetyConfig)
	
	# New comprehensive trading configuration
	technical_analysis: TechnicalAnalysisConfig = field(default_factory=TechnicalAnalysisConfig)
	sentiment_analysis: SentimentAnalysisConfig = field(default_factory=SentimentAnalysisConfig)
	exchanges: ExchangeConfig = field(default_factory=ExchangeConfig)
	risk_management: RiskManagementConfig = field(default_factory=RiskManagementConfig)
	position_tracking: PositionTrackingConfig = field(default_factory=PositionTrackingConfig)
	backtesting: BacktestingConfig = field(default_factory=BacktestingConfig)

	# Optional Phase 4 (derivatives & microstructure) - stored as dict for flexible access
	phase4: Dict[str, Any] = field(default_factory=dict)


def load_yaml_config(config_path: str) -> Dict[str, Any]:
	"""Load configuration from YAML file"""
	try:
		with open(config_path, 'r') as file:
			return yaml.safe_load(file)
	except FileNotFoundError:
		return {}
	except yaml.YAMLError as e:
		print(f"Error parsing YAML config {config_path}: {e}")
		return {}

def load_config_from_files() -> AgentConfig:
	"""Load configuration from YAML files"""
	config = AgentConfig()
	
	# Get config directory path
	config_dir = Path(__file__).parent.parent / "config"
	
	# Load app configuration
	app_config = load_yaml_config(config_dir / "app.yaml")
	if app_config:
		# Update app config
		if "app" in app_config:
			app_data = app_config["app"]
			config.app = AppConfig(**app_data)
		
		# Update server config
		if "server" in app_config:
			server_data = app_config["server"]
			config.server = ServerConfig(**server_data)
		
		# Update database config
		if "database" in app_config:
			db_data = app_config["database"]
			if "redis" in db_data:
				redis_data = db_data["redis"]
				# Map YAML structure to dataclass fields
				config.database.redis_host = redis_data.get("host", "redis")
				config.database.redis_port = redis_data.get("port", 6379)
				config.database.redis_db = redis_data.get("db", 0)
				config.database.redis_password = redis_data.get("password", "")
				config.database.max_connections = redis_data.get("max_connections", 10)
				config.database.socket_timeout = redis_data.get("socket_timeout", 5)
				config.database.socket_connect_timeout = redis_data.get("socket_connect_timeout", 5)
				config.database.retry_on_timeout = redis_data.get("retry_on_timeout", True)
		
		# Update security config
		if "security" in app_config:
			security_data = app_config["security"]
			# Handle nested configurations
			if "jwt" in security_data:
				jwt_data = security_data["jwt"]
				config.security.jwt_secret_key = jwt_data.get("secret_key", "your-secret-key-change-in-production")
				config.security.jwt_algorithm = jwt_data.get("algorithm", "HS256")
				config.security.access_token_expire_minutes = jwt_data.get("access_token_expire_minutes", 1440)
			
			if "password" in security_data:
				password_data = security_data["password"]
				config.security.password_min_length = password_data.get("min_length", 8)
				config.security.require_special_chars = password_data.get("require_special_chars", True)
				config.security.require_numbers = password_data.get("require_numbers", True)
				config.security.require_uppercase = password_data.get("require_uppercase", True)
			
			if "session" in security_data:
				session_data = security_data["session"]
				config.security.session_timeout_hours = session_data.get("timeout_hours", 24)
				config.security.max_login_attempts = session_data.get("max_login_attempts", 5)
				config.security.lockout_duration_minutes = session_data.get("lockout_duration_minutes", 30)
			
			# Handle nested default_admin configuration
			if "default_admin" in security_data:
				admin_data = security_data["default_admin"]
				config.security.default_admin_username = admin_data.get("username", "admin")
				config.security.default_admin_password = admin_data.get("password", "admin123")
		
		# Update API config
		if "api" in app_config:
			api_data = app_config["api"]
			if "rate_limit" in api_data:
				rate_data = api_data["rate_limit"]
				config.api.rate_limit_requests_per_minute = rate_data.get("requests_per_minute", 100)
				config.api.rate_limit_burst_limit = rate_data.get("burst_limit", 200)
			
			if "cors" in api_data:
				cors_data = api_data["cors"]
				config.api.cors_allowed_origins = cors_data.get("allowed_origins", ["*"])
				config.api.cors_allowed_methods = cors_data.get("allowed_methods", ["GET", "POST", "PUT", "DELETE", "OPTIONS"])
				config.api.cors_allowed_headers = cors_data.get("allowed_headers", ["*"])
				config.api.cors_allow_credentials = cors_data.get("allow_credentials", True)
		
		# Update Telegram config
		if "telegram" in app_config:
			telegram_data = app_config["telegram"]
			config.telegram.bot_token = telegram_data.get("bot_token", "")
			config.telegram.enabled = telegram_data.get("enabled", False)
			config.telegram.api_url = telegram_data.get("api_url", "https://api.telegram.org/bot")
			config.telegram.timeout = telegram_data.get("timeout", 10)
			config.telegram.max_retries = telegram_data.get("max_retries", 3)
			
			if "notifications" in telegram_data:
				notifications_data = telegram_data["notifications"]
				config.telegram.notifications.signal_created = notifications_data.get("signal_created", True)
				config.telegram.notifications.signal_changed = notifications_data.get("signal_changed", True)
				config.telegram.notifications.signal_deleted = notifications_data.get("signal_deleted", True)
				config.telegram.notifications.maintenance = notifications_data.get("maintenance", True)
				config.telegram.notifications.system_errors = notifications_data.get("system_errors", False)
				config.telegram.notifications.max_notifications_per_minute = notifications_data.get("max_notifications_per_minute", 10)
				config.telegram.notifications.cooldown_seconds = notifications_data.get("cooldown_seconds", 5)
		
		# Update monitoring config
		if "monitoring" in app_config:
			monitoring_data = app_config["monitoring"]
			config.monitoring = MonitoringConfig(**monitoring_data)
		
		# Update features config
		if "features" in app_config:
			features_data = app_config["features"]
			config.features = FeaturesConfig(**features_data)
		
		# Update notifications config
		if "notifications" in app_config:
			notifications_data = app_config["notifications"]
			config.notifier.enabled = notifications_data.get("enabled", True)
			config.notifier.channels = notifications_data.get("channels", ["console"])
			if "webhook" in notifications_data:
				webhook_data = notifications_data["webhook"]
				config.notifier.webhook_timeout_seconds = webhook_data.get("timeout_seconds", 10)
				config.notifier.webhook_retry_attempts = webhook_data.get("retry_attempts", 3)
				config.notifier.webhook_retry_delay_seconds = webhook_data.get("retry_delay_seconds", 5)
		
		# Update development config
		if "development" in app_config:
			development_data = app_config["development"]
			config.development = DevelopmentConfig(**development_data)
	
	# Load trading configuration
	trading_config = load_yaml_config(config_dir / "trading.yaml")
	if trading_config:
		# Phase 4 (optional) - store raw dict for flexible access patterns
		if "phase4" in trading_config and isinstance(trading_config["phase4"], dict):
			config.phase4 = trading_config["phase4"]
		# Update universe config
		if "universe" in trading_config:
			universe_data = trading_config["universe"]
			if "symbols" in universe_data:
				# Flatten symbols from crypto and stocks
				symbols = []
				if "crypto" in universe_data["symbols"] and universe_data["symbols"]["crypto"]:
					config.universe.crypto_symbols = universe_data["symbols"]["crypto"]
					symbols.extend(universe_data["symbols"]["crypto"])
				if "stocks" in universe_data["symbols"] and universe_data["symbols"]["stocks"]:
					config.universe.stock_symbols = universe_data["symbols"]["stocks"]
					symbols.extend(universe_data["symbols"]["stocks"])
				config.universe.tickers = symbols
			if "default_timeframe" in universe_data:
				config.universe.timeframe = universe_data["default_timeframe"]
			if "timeframes" in universe_data:
				config.universe.timeframes = universe_data["timeframes"]
		
		# Update signal thresholds (legacy)
		if "signals" in trading_config:
			signals_data = trading_config["signals"]
			if "thresholds" in signals_data:
				thresholds_data = signals_data["thresholds"]
				config.thresholds = SignalThresholds(**thresholds_data)
			if "weights" in signals_data:
				weights_data = signals_data["weights"]
				config.thresholds.technical_weight = weights_data.get("technical_weight", 0.6)
				config.thresholds.sentiment_weight = weights_data.get("sentiment_weight", 0.4)
			
			# Update new signals configuration
			config.signals = SignalsConfig()
			if "thresholds" in signals_data:
				config.signals.thresholds.update(signals_data["thresholds"])
			if "weights" in signals_data:
				config.signals.weights.update(signals_data["weights"])
			if "asset_types" in signals_data:
				config.signals.asset_types.update(signals_data["asset_types"])
			if "sentiment_enhancement" in signals_data:
				config.signals.sentiment_enhancement.update(signals_data["sentiment_enhancement"])
			if "risk_management_by_asset" in signals_data:
				config.signals.risk_management_by_asset.update(signals_data["risk_management_by_asset"])
			if "regime_adjustments" in signals_data:
				config.signals.regime_adjustments.update(signals_data["regime_adjustments"])
			if "volatility_adjustments" in signals_data:
				config.signals.volatility_adjustments.update(signals_data["volatility_adjustments"])
			if "multi_timeframe" in signals_data:
				config.signals.multi_timeframe.update(signals_data["multi_timeframe"])
		
		# Update technical analysis config
		if "technical_analysis" in trading_config:
			tech_data = trading_config["technical_analysis"]
			
			# Data fetching configuration
			if "data_fetching" in tech_data:
				data_fetching_data = tech_data["data_fetching"]
				config.technical_analysis.data_fetching_default_limit = data_fetching_data.get("default_limit", 200)
				config.technical_analysis.data_fetching_max_limit = data_fetching_data.get("max_limit", 1000)
				config.technical_analysis.data_fetching_min_limit = data_fetching_data.get("min_limit", 50)
			
			# Enhanced RSI configuration
			if "rsi" in tech_data:
				rsi_data = tech_data["rsi"]
				# Update the new RSI config object
				config.technical_analysis.rsi.period = rsi_data.get("period", 7)
				config.technical_analysis.rsi.method = rsi_data.get("method", "wilder")
				config.technical_analysis.rsi.signal_period = rsi_data.get("signal_period", 4)
				
				# Stochastic RSI settings
				if "stoch_rsi" in rsi_data:
					stoch_data = rsi_data["stoch_rsi"]
					config.technical_analysis.rsi.stoch_rsi.update(stoch_data)
				
				# Dynamic thresholds
				if "thresholds" in rsi_data:
					config.technical_analysis.rsi.thresholds.update(rsi_data["thresholds"])
				
				# Cross signal thresholds
				if "cross_thresholds" in rsi_data:
					config.technical_analysis.rsi.cross_thresholds.update(rsi_data["cross_thresholds"])
				
				# Legacy fields for backward compatibility
				config.technical_analysis.rsi_period = rsi_data.get("period", 7)
				config.technical_analysis.rsi_overbought = rsi_data.get("thresholds", {}).get("overbought", 70)
				config.technical_analysis.rsi_oversold = rsi_data.get("thresholds", {}).get("oversold", 30)
			
			# Enhanced MACD configuration
			if "macd" in tech_data:
				macd_data = tech_data["macd"]
				# Update the new MACD config object
				config.technical_analysis.macd.fast_period = macd_data.get("fast_period", 4)
				config.technical_analysis.macd.slow_period = macd_data.get("slow_period", 9)
				config.technical_analysis.macd.signal_period = macd_data.get("signal_period", 3)
				
				# Legacy fields for backward compatibility
				config.technical_analysis.macd_fast = macd_data.get("fast_period", 4)
				config.technical_analysis.macd_slow = macd_data.get("slow_period", 9)
				config.technical_analysis.macd_signal = macd_data.get("signal_period", 3)
			
			# Enhanced ATR configuration
			if "atr" in tech_data:
				atr_data = tech_data["atr"]
				# Update the new ATR config object
				config.technical_analysis.atr.period = atr_data.get("period", 7)
				config.technical_analysis.atr.multiplier = atr_data.get("multiplier", 2.0)
				
				# Legacy fields for backward compatibility
				config.technical_analysis.atr_period = atr_data.get("period", 7)
				config.technical_analysis.atr_multiplier = atr_data.get("multiplier", 2.0)
			
			# Moving averages configuration
			if "moving_averages" in tech_data:
				ma_data = tech_data["moving_averages"]
				config.technical_analysis.sma_periods = ma_data.get("sma_periods", [20, 50, 200])
				config.technical_analysis.ema_periods = ma_data.get("ema_periods", [12, 26, 50])
		
		# Update sentiment analysis config
		if "sentiment_analysis" in trading_config:
			sentiment_data = trading_config["sentiment_analysis"]
			if "model" in sentiment_data:
				model_data = sentiment_data["model"]
				config.sentiment_analysis.model_name = model_data.get("name", "ProsusAI/finbert")
				config.sentiment_analysis.max_length = model_data.get("max_length", 512)
				config.sentiment_analysis.batch_size = model_data.get("batch_size", 32)
			
			# RSS configuration (flat structure)
			config.sentiment_analysis.rss_enabled = sentiment_data.get("rss_enabled", True)
			config.sentiment_analysis.rss_feeds = sentiment_data.get("rss_feeds", [])
			config.sentiment_analysis.rss_max_headlines_per_feed = sentiment_data.get("rss_max_headlines_per_feed", 15)
			
			# Reddit configuration (flat structure)
			config.sentiment_analysis.reddit_enabled = sentiment_data.get("reddit_enabled", True)
			config.sentiment_analysis.reddit_subreddits = sentiment_data.get("reddit_subreddits", [])
			config.sentiment_analysis.reddit_max_posts_per_subreddit = sentiment_data.get("reddit_max_posts_per_subreddit", 10)
			config.sentiment_analysis.reddit_sample_size = sentiment_data.get("reddit_sample_size", 20)
			
			# Scoring configuration
			if "scoring" in sentiment_data:
				scoring_data = sentiment_data["scoring"]
				config.sentiment_analysis.positive_threshold = scoring_data.get("positive_threshold", 0.1)
				config.sentiment_analysis.negative_threshold = scoring_data.get("negative_threshold", -0.1)
				config.sentiment_analysis.neutral_range = scoring_data.get("neutral_range", [-0.1, 0.1])
		
		# Update exchange config
		if "exchanges" in trading_config:
			exchange_data = trading_config["exchanges"]
			if "ccxt" in exchange_data:
				ccxt_data = exchange_data["ccxt"]
				config.exchanges.ccxt_default_exchange = ccxt_data.get("default_exchange", "binance")
				config.exchanges.ccxt_rate_limit = ccxt_data.get("rate_limit", True)
				config.exchanges.ccxt_timeout = ccxt_data.get("timeout", 30000)
				config.exchanges.ccxt_retries = ccxt_data.get("retries", 3)
			if "alpaca" in exchange_data:
				alpaca_data = exchange_data["alpaca"]
				config.exchanges.alpaca_base_url = alpaca_data.get("base_url", "https://paper-api.alpaca.markets")
				config.exchanges.alpaca_data_url = alpaca_data.get("data_url", "https://data.alpaca.markets")
				config.exchanges.alpaca_timeout = alpaca_data.get("timeout", 30)
				config.exchanges.alpaca_retries = alpaca_data.get("retries", 3)
		
		# Update risk management config
		if "signals" in trading_config and "risk_management" in trading_config["signals"]:
			risk_data = trading_config["signals"]["risk_management"]
			config.risk_management.stop_loss_percentage = risk_data.get("stop_loss_percentage", 0.02)
			config.risk_management.take_profit_percentage = risk_data.get("take_profit_percentage", 0.04)
			config.risk_management.max_position_size = risk_data.get("max_position_size", 0.1)
			config.risk_management.max_daily_signals = risk_data.get("max_daily_signals", 10)
		
		# Update position tracking config
		if "positions" in trading_config:
			position_data = trading_config["positions"]
			if "tracking" in position_data:
				tracking_data = position_data["tracking"]
				config.position_tracking.enabled = tracking_data.get("enabled", True)
				config.position_tracking.update_interval_minutes = tracking_data.get("update_interval_minutes", 5)
				config.position_tracking.max_positions_per_user = tracking_data.get("max_positions_per_user", 50)
			if "lifecycle" in position_data:
				lifecycle_data = position_data["lifecycle"]
				config.position_tracking.auto_close_on_signal = lifecycle_data.get("auto_close_on_signal", False)
				config.position_tracking.max_holding_days = lifecycle_data.get("max_holding_days", 30)
				config.position_tracking.min_holding_minutes = lifecycle_data.get("min_holding_minutes", 15)
		
		# Update backtesting config
		if "backtesting" in trading_config:
			backtest_data = trading_config["backtesting"]
			if "default" in backtest_data:
				default_data = backtest_data["default"]
				config.backtesting.start_date = default_data.get("start_date", "2024-01-01")
				config.backtesting.end_date = default_data.get("end_date", "2024-12-31")
				config.backtesting.initial_capital = default_data.get("initial_capital", 10000)
				config.backtesting.commission = default_data.get("commission", 0.001)
			if "metrics" in backtest_data:
				config.backtesting.metrics = backtest_data.get("metrics", [
					"sharpe_ratio", "max_drawdown", "win_rate", "total_return", "volatility"
				])
		
		# Update safety configuration
		if "safety" in trading_config:
			safety_data = trading_config["safety"]
			config.safety.allow_incomplete_candles = safety_data.get("allow_incomplete_candles", False)
			config.safety.min_data_points_for_indicators = safety_data.get("min_data_points_for_indicators", 50)
			config.safety.safety_margin = safety_data.get("safety_margin", 10)
			config.safety.max_candle_staleness = safety_data.get("max_candle_staleness", {
				"1m": 2, "5m": 8, "15m": 20, "1h": 70
			})
			config.safety.validate_thresholds = safety_data.get("validate_thresholds", True)
			config.safety.normalize_weights = safety_data.get("normalize_weights", True)
			config.safety.deterministic_sampling = safety_data.get("deterministic_sampling", True)
			config.safety.sentiment_timeout_seconds = safety_data.get("sentiment_timeout_seconds", 30)
			config.safety.market_data_timeout_seconds = safety_data.get("market_data_timeout_seconds", 15)
			config.safety.rss_timeout_seconds = safety_data.get("rss_timeout_seconds", 10)
			config.safety.reddit_timeout_seconds = safety_data.get("reddit_timeout_seconds", 15)
			
			# Circuit breaker settings
			if "circuit_breaker" in safety_data:
				cb_data = safety_data["circuit_breaker"]
				config.safety.circuit_breaker_enabled = cb_data.get("enabled", True)
				config.safety.circuit_breaker_failure_threshold = cb_data.get("failure_threshold", 3)
				config.safety.circuit_breaker_cooldown_minutes = cb_data.get("cooldown_minutes", 5)
	
	# Load logging configuration
	logging_config = load_yaml_config(config_dir / "logging.yaml")
	if logging_config and "logging" in logging_config:
		logging_data = logging_config["logging"]
		# All defaults come from YAML file - no hardcoded fallbacks
		config.logging.level = logging_data.get("level", "")
		config.logging.format = logging_data.get("format", "")
		
		if "destinations" in logging_data:
			destinations = logging_data["destinations"]
			config.logging.console_enabled = destinations.get("console", False)
			config.logging.file_enabled = destinations.get("file", False)
		
		if "file" in logging_data:
			file_data = logging_data["file"]
			config.logging.file_path = file_data.get("path", "")
			config.logging.max_bytes = file_data.get("max_bytes", 0)
			config.logging.backup_count = file_data.get("backup_count", 0)
			config.logging.encoding = file_data.get("encoding", "")
		
		if "console" in logging_data:
			console_data = logging_data["console"]
			config.logging.colored = console_data.get("colored", False)
			config.logging.timestamp = console_data.get("timestamp", False)
	
	return config

def load_config_from_env() -> AgentConfig:
	"""Load configuration from environment variables (legacy support)"""
	config = load_config_from_files()  # Start with file-based config
	
	# Environment variable overrides (ONLY for sensitive data)
	# Database configuration (sensitive)
	if redis_password := os.getenv("REDIS_PASSWORD"):
		config.database.redis_password = redis_password
	
	# Redis host and port from environment (for Kubernetes)
	if redis_host := os.getenv("REDIS_HOST"):
		config.database.redis_host = redis_host
	if redis_port := os.getenv("REDIS_PORT"):
		config.database.redis_port = int(redis_port)
	
	# Security configuration (sensitive)
	if jwt_secret := os.getenv("JWT_SECRET_KEY"):
		config.security.jwt_secret_key = jwt_secret
	if admin_password := os.getenv("DEFAULT_ADMIN_PASSWORD"):
		config.security.default_admin_password = admin_password
	
	# Logging configuration (can be overridden for debugging)
	if log_level := os.getenv("AGENT_LOG_LEVEL"):
		config.logging.level = log_level
	if log_format := os.getenv("AGENT_LOG_FORMAT"):
		config.logging.format = log_format
	if log_file := os.getenv("AGENT_LOG_FILE"):
		config.logging.file_path = log_file
	if max_bytes := os.getenv("AGENT_LOG_MAX_BYTES"):
		config.logging.max_bytes = int(max_bytes)
	if backup_count := os.getenv("AGENT_LOG_BACKUP_COUNT"):
		config.logging.backup_count = int(backup_count)
	
	# Notifier overrides
	config.notifier.mode = os.getenv("AGENT_NOTIFIER", config.notifier.mode)
	config.notifier.telegram_token = os.getenv("TELEGRAM_BOT_TOKEN", config.notifier.telegram_token)
	config.notifier.telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID", config.notifier.telegram_chat_id)
	config.notifier.slack_webhook_url = os.getenv("SLACK_WEBHOOK_URL", config.notifier.slack_webhook_url)
	
	# Telegram config overrides (new structure)
	# Note: chat_id is now per-user and stored in Redis, not in environment variables
	# Only override if environment variable is not empty
	if telegram_token := os.getenv("TELEGRAM_BOT_TOKEN"):
		config.telegram.bot_token = telegram_token
	if telegram_enabled := os.getenv("TELEGRAM_ENABLED"):
		config.telegram.enabled = telegram_enabled.lower() == "true"
	
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

def get_config() -> AgentConfig:
	"""Get the current configuration (file-based with env overrides)"""
	return load_config_from_env()


