"""
Real-time Signal Monitoring System
Monitors existing signals and updates their status based on changing market conditions
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pytz
import requests

from .cache.redis_client import TradingAgentRedis
from .data.ccxt_client import fetch_ohlcv as fetch_ccxt_ohlcv
from .data.alpaca_client import fetch_ohlcv as fetch_alpaca_ohlcv
from .models.sentiment import SentimentAnalyzer
from .news.rss import fetch_headlines_async
from .news.reddit import fetch_reddit_posts_async
from .config import get_config

logger = logging.getLogger(__name__)

class SignalMonitor:
    """Monitors and updates trading signals in real-time"""
    
    def __init__(self, redis_client: TradingAgentRedis, sentiment_analyzer: Optional[SentimentAnalyzer] = None):
        self.redis_client = redis_client
        self.sentiment_analyzer = sentiment_analyzer
        self.israel_tz = pytz.timezone('Asia/Jerusalem')
        
        # Load monitoring configuration
        try:
            config = get_config()
            self.signal_check_interval_minutes = config.monitoring.signal_check_interval_minutes
            self.health_check_interval_seconds = config.monitoring.health_check_interval_seconds
            self.max_concurrent_monitors = config.monitoring.max_concurrent_monitors
            self.history_retention_days = config.monitoring.history_retention_days
            logger.info(f"🔧 [MONITOR-CONFIG] Loaded monitoring config: check_interval={self.signal_check_interval_minutes}min, "
                       f"health_interval={self.health_check_interval_seconds}s, "
                       f"max_concurrent={self.max_concurrent_monitors}, "
                       f"retention={self.history_retention_days}days")
        except Exception as e:
            logger.warning(f"Failed to load monitoring config, using defaults: {e}")
            self.signal_check_interval_minutes = 2
            self.health_check_interval_seconds = 30
            self.max_concurrent_monitors = 10
            self.history_retention_days = 30
        
    async def monitor_all_signals(self) -> Dict[str, Any]:
        """Monitor all active signals and update their status"""
        try:
            # Get all signals from Redis
            all_signals = await self.redis_client.get_signals(limit=1000)
            
            logger.info(f"Retrieved {len(all_signals) if all_signals else 0} signals from Redis")
            
            if not all_signals:
                logger.info("No signals to monitor")
                return {"monitored": 0, "updated": 0, "errors": 0}
            
            logger.info(f"🔍 [MONITOR] Monitoring {len(all_signals)} signals")
            
            updated_count = 0
            error_count = 0
            
            # Group signals by symbol for efficient data fetching
            signals_by_symbol = {}
            for signal in all_signals:
                symbol = signal.get('symbol')
                if symbol:
                    if symbol not in signals_by_symbol:
                        signals_by_symbol[symbol] = []
                    signals_by_symbol[symbol].append(signal)
            
            # Limit concurrent processing based on configuration
            max_concurrent = min(self.max_concurrent_monitors, len(signals_by_symbol))
            logger.info(f"Processing {len(signals_by_symbol)} symbols with max {max_concurrent} concurrent monitors")
            
            # Process each symbol
            for symbol, signals in signals_by_symbol.items():
                try:
                    updated = await self._monitor_symbol_signals(symbol, signals)
                    updated_count += updated
                except Exception as e:
                    logger.error(f"Error monitoring signals for {symbol}: {e}")
                    error_count += 1
            
            result = {
                "monitored": len(all_signals),
                "updated": updated_count,
                "errors": error_count,
                "timestamp": datetime.now(self.israel_tz).isoformat()
            }
            
            logger.info(f"✅ [MONITOR] Monitoring completed: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error in monitor_all_signals: {e}")
            return {"monitored": 0, "updated": 0, "errors": 1, "error": str(e)}
    
    async def cleanup_old_history(self) -> Dict[str, Any]:
        """Clean up old signal history based on retention configuration"""
        try:
            cutoff_date = datetime.now(self.israel_tz) - timedelta(days=self.history_retention_days)
            cutoff_timestamp = cutoff_date.isoformat()
            
            logger.info(f"Cleaning up signal history older than {cutoff_timestamp} (retention: {self.history_retention_days} days)")
            
            # Get all signals
            all_signals = await self.redis_client.get_signals(limit=10000)
            if not all_signals:
                return {"cleaned": 0, "errors": 0}
            
            cleaned_count = 0
            error_count = 0
            
            for signal in all_signals:
                try:
                    signal_timestamp = signal.get('timestamp')
                    if signal_timestamp and signal_timestamp < cutoff_timestamp:
                        # Clean up old history for this signal
                        await self.redis_client.cleanup_signal_history(signal_timestamp)
                        cleaned_count += 1
                except Exception as e:
                    logger.error(f"Error cleaning history for signal {signal.get('timestamp', 'unknown')}: {e}")
                    error_count += 1
            
            logger.info(f"History cleanup completed: {cleaned_count} signals cleaned, {error_count} errors")
            return {"cleaned": cleaned_count, "errors": error_count}
            
        except Exception as e:
            logger.error(f"Error during history cleanup: {e}")
            return {"cleaned": 0, "errors": 1}
    
    async def _monitor_symbol_signals(self, symbol: str, signals: List[Dict]) -> int:
        """Monitor signals for a specific symbol"""
        try:
            # Fetch fresh market data
            from .config import load_config_from_env
            config = load_config_from_env()
            timeframe = signals[0].get('timeframe', config.universe.timeframe)
            
            if "/" in symbol:  # Crypto
                ohlcv_data = fetch_ccxt_ohlcv(symbol, timeframe, limit=200)
            else:  # Stock/ETF
                ohlcv_data = fetch_alpaca_ohlcv(symbol, timeframe, limit=200)
            
            if ohlcv_data is None or ohlcv_data.empty:
                logger.warning(f"No market data available for {symbol}")
                return 0
            
            # Get fresh sentiment data
            sentiment_score = await self._get_fresh_sentiment()
            
            updated_count = 0
            
            # Update each signal for this symbol (skip admin signals)
            for signal in signals:
                # Skip admin signals - only monitor user signals
                if signal.get('username') == 'admin':
                    logger.debug(f"Skipping admin signal: {signal.get('timestamp')}")
                    continue
                    
                try:
                    updated = await self._update_signal_status(signal, ohlcv_data, sentiment_score)
                    if updated:
                        updated_count += 1
                except Exception as e:
                    logger.error(f"Error updating signal {signal.get('timestamp')}: {e}")
            
            return updated_count
            
        except Exception as e:
            logger.error(f"Error monitoring symbol {symbol}: {e}")
            return 0
    
    async def _get_fresh_sentiment(self) -> float:
        """Get fresh sentiment score from multiple sources"""
        try:
            if not self.sentiment_analyzer:
                return 0.0
            
            # Load configuration
            from .config import load_config_from_env
            config = load_config_from_env()
            
            all_texts = []
            
            # Fetch RSS headlines (async for better performance)
            logger.info("📰 [MONITOR-DATA] Fetching RSS headlines...")
            # Use RSS feeds from configuration
            rss_feeds = config.sentiment_analysis.rss_feeds if config.sentiment_analysis.rss_enabled else []
            headlines = await fetch_headlines_async(rss_feeds, limit_per_feed=config.sentiment_analysis.rss_max_headlines_per_feed)
            if headlines:
                all_texts.extend(headlines)
                logger.info(f"📰 [MONITOR-DATA] RSS collected {len(headlines)} headlines")
            
            # Fetch Reddit posts using configuration (async for better performance)
            if config.sentiment_analysis.reddit_enabled:
                logger.info("🔴 [MONITOR-DATA] Fetching Reddit posts...")
                try:
                    reddit_posts = await fetch_reddit_posts_async(
                        subreddits=config.sentiment_analysis.reddit_subreddits,
                        limit_per_subreddit=config.sentiment_analysis.reddit_max_posts_per_subreddit
                    )
                    
                    if reddit_posts:
                        all_texts.extend(reddit_posts)
                        logger.info(f"🔴 [MONITOR-DATA] Reddit collected {len(reddit_posts)} posts from {len(config.sentiment_analysis.reddit_subreddits)} subreddits")
                        
                except Exception as e:
                    logger.warning(f"⚠️ [MONITOR-DATA] Could not fetch Reddit posts: {e}")
            else:
                logger.info("🔴 [MONITOR-DATA] Reddit integration disabled in configuration")
            
            if all_texts:
                # Use sample size from configuration
                import random
                sample_size = min(config.sentiment_analysis.reddit_sample_size, len(all_texts))
                shuffled_texts = all_texts.copy()
                random.shuffle(shuffled_texts)
                text_sample = shuffled_texts[:sample_size]
                logger.info(f"🧠 [MONITOR-SENTIMENT] Analyzing {len(text_sample)} texts (RSS + Reddit) for sentiment")
                sentiment_score = self.sentiment_analyzer.score(text_sample)
                logger.info(f"🧠 [MONITOR-SENTIMENT] Sentiment analysis result: {sentiment_score:.3f}")
                return sentiment_score
            
            return 0.0
            
        except Exception as e:
            logger.warning(f"Could not fetch fresh sentiment: {e}")
            return 0.0
    
    async def _update_signal_status(self, signal: Dict, ohlcv_data: Any, sentiment_score: float) -> bool:
        """Update a single signal's status based on fresh data using Phase3 signal generation"""
        try:
            # Check if this is a Phase3 signal
            is_phase3 = (signal.get('phase') == 'phase3_complete' or 
                        'regime_detection' in signal or 
                        'advanced_rsi' in signal or
                        'position_sizing' in signal)
            
            if is_phase3:
                # Use Phase3 signal generation for monitoring
                return await self._update_phase3_signal_status(signal, ohlcv_data, sentiment_score)
            else:
                # Use legacy monitoring for Phase1/Phase2 signals
                return await self._update_legacy_signal_status(signal, ohlcv_data, sentiment_score)
                
        except Exception as e:
            logger.error(f"Error updating signal status: {e}")
            return False
    
    async def _update_phase3_signal_status(self, signal: Dict, ohlcv_data: Any, sentiment_score: float) -> bool:
        """Update Phase3 signal status by refreshing its data with original parameters"""
        try:
            from web.phase3_signal_generator import phase3_signal_generator
            
            symbol = signal.get('symbol')
            timeframe = signal.get('timeframe', '1h')
            username = signal.get('username', 'monitoring')
            
            # Extract original parameters from the signal to maintain consistency
            original_buy_threshold = signal.get('applied_buy_threshold')
            original_sell_threshold = signal.get('applied_sell_threshold')
            original_tech_weight = signal.get('applied_tech_weight')
            original_sentiment_weight = signal.get('applied_sentiment_weight')
            
            logger.info(f"🔍 [MONITOR] {symbol}: Using original parameters - buy_thresh={original_buy_threshold}, sell_thresh={original_sell_threshold}, tech_weight={original_tech_weight}, sent_weight={original_sentiment_weight}")
            
            # Generate fresh Phase3 signal data using the same parameters as the original
            # But don't store it - we'll use the data to update the existing signal
            # Disable caching during monitoring to avoid event loop conflicts
            fresh_signal = await phase3_signal_generator.generate_phase3_signal_shared(
                symbol=symbol,
                timeframe=timeframe,
                username=username,
                output_level="monitoring",  # Use monitoring level for performance
                buy_threshold=original_buy_threshold,
                sell_threshold=original_sell_threshold,
                technical_weight=original_tech_weight,
                sentiment_weight=original_sentiment_weight,
                skip_storage=True,  # Don't store the new signal, just use the data
                disable_caching=True  # Disable caching to avoid Redis event loop conflicts
            )
            
            # Compare with original signal
            original_signal_type = signal.get('signal_type', 'HOLD')
            new_signal_type = fresh_signal.signal_type
            
            # Check if status actually changed
            status_changed = new_signal_type != original_signal_type
            
            if status_changed:
                logger.info(f"🔄 [MONITOR-STATUS] Phase3 signal status changed: {symbol} {original_signal_type} → {new_signal_type}")
                
                # Update dynamic fields with fresh data while preserving original metadata
                signal['signal_type'] = new_signal_type
                signal['fused_score'] = fresh_signal.fused_score
                signal['technical_score'] = fresh_signal.technical_score
                signal['sentiment_score'] = fresh_signal.sentiment_score
                signal['confidence'] = fresh_signal.confidence
                signal['stop_loss'] = fresh_signal.stop_loss
                signal['take_profit'] = fresh_signal.take_profit
                signal['last_updated'] = datetime.now(self.israel_tz).isoformat()
                
                # Preserve applied threshold values (they should remain the same)
                signal['applied_buy_threshold'] = fresh_signal.applied_buy_threshold
                signal['applied_sell_threshold'] = fresh_signal.applied_sell_threshold
                signal['applied_tech_weight'] = fresh_signal.applied_tech_weight
                signal['applied_sentiment_weight'] = fresh_signal.applied_sentiment_weight
                
                # Update Phase3 specific fields with fresh data
                signal['regime_detection'] = fresh_signal.regime_detection
                signal['advanced_rsi'] = fresh_signal.advanced_rsi
                signal['position_sizing'] = fresh_signal.position_sizing
                signal['risk_metrics'] = fresh_signal.risk_metrics
                signal['technical_indicators'] = fresh_signal.technical_indicators
                signal['market_microstructure'] = fresh_signal.market_microstructure
                signal['multi_timeframe'] = fresh_signal.multi_timeframe
                signal['btc_dominance'] = fresh_signal.btc_dominance
                signal['market_wide_sentiment'] = fresh_signal.market_wide_sentiment
                
                # Store updated signal (this updates the existing signal, doesn't create a new one)
                await self.redis_client.store_signal(signal)
                
                # Add comprehensive history event for status change
                await self.redis_client.add_signal_history_event(
                    signal_timestamp=signal.get('timestamp'),
                    event_type="status_changed",
                    description=f"Phase3 status changed: {original_signal_type} → {new_signal_type}",
                    metadata={
                        # Status change info
                        "old_status": original_signal_type,
                        "new_status": new_signal_type,
                        "reason": "Phase3 monitoring - market conditions changed",
                        
                        # Core scores
                        "fused_score": fresh_signal.fused_score,
                        "technical_score": fresh_signal.technical_score,
                        "sentiment_score": fresh_signal.sentiment_score,
                        "confidence": fresh_signal.confidence,
                        
                        # Applied thresholds
                        "applied_buy_threshold": fresh_signal.applied_buy_threshold,
                        "applied_sell_threshold": fresh_signal.applied_sell_threshold,
                        "applied_tech_weight": fresh_signal.applied_tech_weight,
                        "applied_sentiment_weight": fresh_signal.applied_sentiment_weight,
                        
                        # Price and risk management
                        "current_price": fresh_signal.technical_indicators.get('current_price', 0),
                        "stop_loss": fresh_signal.stop_loss,
                        "take_profit": fresh_signal.take_profit,
                        "risk_reward_ratio": fresh_signal.meta.get('risk_reward_ratio', 1.0),
                        
                        # Technical indicators
                        "rsi_14": fresh_signal.technical_indicators.get('rsi_14', 'N/A'),
                        "macd": fresh_signal.technical_indicators.get('macd', 'N/A'),
                        "atr": fresh_signal.technical_indicators.get('atr', 'N/A'),
                        "adx": fresh_signal.technical_indicators.get('adx', 'N/A'),
                        "bb_percent": fresh_signal.technical_indicators.get('bb_percent', 'N/A'),
                        "ema_20": fresh_signal.technical_indicators.get('ema_20', 'N/A'),
                        "ema_50": fresh_signal.technical_indicators.get('ema_50', 'N/A'),
                        
                        # Market analysis
                        "regime_classification": fresh_signal.regime_detection.get('regime_classification', 'unknown'),
                        "trend_strength": fresh_signal.regime_detection.get('trend_strength', 0),
                        "volatility_state": fresh_signal.regime_detection.get('volatility_state', 'medium'),
                        "rsi_alignment": fresh_signal.advanced_rsi.get('alignment', 'mixed'),
                        
                        # Multi-timeframe analysis
                        "overall_trend": fresh_signal.multi_timeframe.get('overall_trend', 'neutral'),
                        "trend_consensus": fresh_signal.multi_timeframe.get('trend_consensus', 0.5),
                        
                        # Market-wide sentiment
                        "btc_dominance": fresh_signal.btc_dominance.get('btc_dominance', 50),
                        "market_sentiment": fresh_signal.market_wide_sentiment.get('overall_sentiment', 'neutral'),
                        
                        # Position sizing and risk
                        "position_size": fresh_signal.position_sizing.get('recommended', 0),
                        "risk_level": fresh_signal.risk_metrics.get('risk_level', 'medium'),
                        "volatility": fresh_signal.risk_metrics.get('volatility', 0),
                        "var_95": fresh_signal.risk_metrics.get('var_95', 0),
                        "sharpe_ratio": fresh_signal.risk_metrics.get('sharpe_ratio', 1.0),
                        
                        # Market microstructure
                        "bollinger_squeeze": fresh_signal.bollinger_bands.get('squeeze', False),
                        "vwap_deviation": fresh_signal.vwap_analysis.get('deviation', 0),
                        "volume_trend": fresh_signal.volume_indicators.get('obv_trend', 'neutral'),
                        "ma_crossovers": fresh_signal.moving_averages.get('crossovers', {}).get('last_bullish', False),
                        
                        # Timestamp for comparison
                        "monitoring_timestamp": datetime.now(self.israel_tz).isoformat()
                    }
                )
                
                # Log comprehensive activity
                await self.redis_client.log_activity(
                    event_type="signal_status_changed",
                    description=f"Phase3 signal {symbol} changed from {original_signal_type} to {new_signal_type} | Price: ${fresh_signal.technical_indicators.get('current_price', 0):.2f} | Fused: {fresh_signal.fused_score:.4f} | Confidence: {fresh_signal.confidence:.2%}",
                    user="monitoring",
                    metadata={
                        "symbol": symbol,
                        "timeframe": fresh_signal.timeframe,
                        "old_type": original_signal_type,
                        "new_type": new_signal_type,
                        "fused_score": fresh_signal.fused_score,
                        "technical_score": fresh_signal.technical_score,
                        "sentiment_score": fresh_signal.sentiment_score,
                        "confidence": fresh_signal.confidence,
                        "current_price": fresh_signal.technical_indicators.get('current_price', 0),
                        "stop_loss": fresh_signal.stop_loss,
                        "take_profit": fresh_signal.take_profit,
                        "rsi_14": fresh_signal.technical_indicators.get('rsi_14', 'N/A'),
                        "regime": fresh_signal.regime_detection.get('regime_classification', 'unknown'),
                        "trend_strength": fresh_signal.regime_detection.get('trend_strength', 0),
                        "risk_level": fresh_signal.risk_metrics.get('risk_level', 'medium'),
                        "position_size": fresh_signal.position_sizing.get('recommended', 0)
                    }
                )
                
                # Send Telegram notification for status change
                await self.send_telegram_notification(
                    signal, original_signal_type, new_signal_type, 
                    fresh_signal.fused_score, fresh_signal.technical_score, fresh_signal.sentiment_score,
                    fresh_signal.technical_indicators.get('current_price', 0)
                )
                
                return True
            else:
                # Status didn't change, but add history event with dynamic data to show market changes
                logger.info(f"⏸️ [MONITOR-STATUS] Phase3 signal status unchanged for {signal.get('timestamp')}: {original_signal_type}")
                
                # Add comprehensive history event for regular monitoring cycle with dynamic data
                await self.redis_client.add_signal_history_event(
                    signal_timestamp=signal.get('timestamp'),
                    event_type="monitoring_cycle",
                    description=f"monitoring cycle - status remains {original_signal_type} | FUSED SCORE: {fresh_signal.fused_score:.4f} | TECHNICAL SCORE: {fresh_signal.technical_score:.4f} | SENTIMENT SCORE: {fresh_signal.sentiment_score:.4f}",
                    metadata={
                        # Status info
                        "current_status": original_signal_type,
                        "reason": "regular monitoring cycle - market data updated",
                        
                        # Core scores with current values
                        "fused_score": fresh_signal.fused_score,
                        "technical_score": fresh_signal.technical_score,
                        "sentiment_score": fresh_signal.sentiment_score,
                        "confidence": fresh_signal.confidence,
                        
                        # Applied thresholds
                        "applied_buy_threshold": fresh_signal.applied_buy_threshold,
                        "applied_sell_threshold": fresh_signal.applied_sell_threshold,
                        "applied_tech_weight": fresh_signal.applied_tech_weight,
                        "applied_sentiment_weight": fresh_signal.applied_sentiment_weight,
                        
                        # Price and risk management
                        "current_price": fresh_signal.technical_indicators.get('current_price', 0),
                        "stop_loss": fresh_signal.stop_loss,
                        "take_profit": fresh_signal.take_profit,
                        "risk_reward_ratio": fresh_signal.meta.get('risk_reward_ratio', 1.0),
                        
                        # Technical indicators
                        "rsi_14": fresh_signal.technical_indicators.get('rsi_14', 'N/A'),
                        "macd": fresh_signal.technical_indicators.get('macd', 'N/A'),
                        "atr": fresh_signal.technical_indicators.get('atr', 'N/A'),
                        "adx": fresh_signal.technical_indicators.get('adx', 'N/A'),
                        "bb_percent": fresh_signal.technical_indicators.get('bb_percent', 'N/A'),
                        "ema_20": fresh_signal.technical_indicators.get('ema_20', 'N/A'),
                        "ema_50": fresh_signal.technical_indicators.get('ema_50', 'N/A'),
                        
                        # Market analysis
                        "regime_classification": fresh_signal.regime_detection.get('regime_classification', 'unknown'),
                        "trend_strength": fresh_signal.regime_detection.get('trend_strength', 0),
                        "volatility_state": fresh_signal.regime_detection.get('volatility_state', 'medium'),
                        "rsi_alignment": fresh_signal.advanced_rsi.get('alignment', 'mixed'),
                        
                        # Multi-timeframe analysis
                        "overall_trend": fresh_signal.multi_timeframe.get('overall_trend', 'neutral'),
                        "trend_consensus": fresh_signal.multi_timeframe.get('trend_consensus', 0.5),
                        
                        # Market-wide sentiment
                        "btc_dominance": fresh_signal.btc_dominance.get('btc_dominance', 50),
                        "market_sentiment": fresh_signal.market_wide_sentiment.get('overall_sentiment', 'neutral'),
                        
                        # Position sizing and risk
                        "position_size": fresh_signal.position_sizing.get('recommended', 0),
                        "risk_level": fresh_signal.risk_metrics.get('risk_level', 'medium'),
                        "volatility": fresh_signal.risk_metrics.get('volatility', 0),
                        "var_95": fresh_signal.risk_metrics.get('var_95', 0),
                        "sharpe_ratio": fresh_signal.risk_metrics.get('sharpe_ratio', 1.0),
                        
                        # Market microstructure
                        "bollinger_squeeze": fresh_signal.bollinger_bands.get('squeeze', False),
                        "vwap_deviation": fresh_signal.vwap_analysis.get('deviation', 0),
                        "volume_trend": fresh_signal.volume_indicators.get('obv_trend', 'neutral'),
                        "ma_crossovers": fresh_signal.moving_averages.get('crossovers', {}).get('last_bullish', False),
                        
                        # Timestamp for comparison
                        "monitoring_timestamp": datetime.now(self.israel_tz).isoformat()
                    }
                )
                
                return False
            
        except Exception as e:
            logger.error(f"Error updating Phase3 signal status: {e}")
            return False
    
    async def _update_legacy_signal_status(self, signal: Dict, ohlcv_data: Any, sentiment_score: float) -> bool:
        """Update legacy Phase1/Phase2 signal status using original monitoring logic"""
        try:
            # Calculate fresh technical indicators
            from .indicators import compute_rsi, compute_macd, compute_atr
            
            close = ohlcv_data['close']
            high = ohlcv_data['high']
            low = ohlcv_data['low']
            
            # Calculate technical indicators
            rsi = compute_rsi(close)
            macd_df = compute_macd(close)
            macd_hist = macd_df['hist']
            atr = compute_atr(high, low, close)
            
            # Calculate fresh technical score
            tech_score = 0.0
            if not close.empty:
                current_close = close.iloc[-1]
                current_rsi = rsi.iloc[-1] if not rsi.empty else 50
                current_macd_hist = macd_hist.iloc[-1] if not macd_hist.empty else 0
                
                # Normalize RSI to [-1, 1]
                rsi_score = (current_rsi - 50) / 50
                
                # Normalize MACD histogram
                macd_score = max(min(current_macd_hist / 1000, 1), -1)
                
                tech_score = (rsi_score + macd_score) / 2
            
            # Get original weights and thresholds
            tech_weight = signal.get('applied_tech_weight', 0.6)
            sentiment_weight = signal.get('applied_sentiment_weight', 0.4)
            buy_threshold = signal.get('applied_buy_threshold', 0.7)
            sell_threshold = signal.get('applied_sell_threshold', -0.7)
            
            # Calculate fresh fused score
            fresh_fused_score = tech_weight * tech_score + sentiment_weight * sentiment_score
            
            # Determine new signal type
            if fresh_fused_score >= buy_threshold:
                new_signal_type = "BUY"
            elif fresh_fused_score <= sell_threshold:
                new_signal_type = "SELL"
            else:
                new_signal_type = "HOLD"
            
            # Get original signal type
            original_signal_type = signal.get('signal_type', 'HOLD')
            
            # Check if signal status changed
            if new_signal_type != original_signal_type:
                logger.info(f"🔄 [MONITOR-STATUS] Legacy signal status changed: {signal.get('symbol')} {original_signal_type} → {new_signal_type}")
                
                # Update signal in Redis
                signal['signal_type'] = new_signal_type
                signal['fused_score'] = fresh_fused_score
                signal['technical_score'] = tech_score
                signal['sentiment_score'] = sentiment_score
                signal['confidence'] = abs(fresh_fused_score)
                signal['last_updated'] = datetime.now(self.israel_tz).isoformat()
                signal['status_change_reason'] = f"Market conditions changed: Technical={tech_score:.3f}, Sentiment={sentiment_score:.3f}, Fused={fresh_fused_score:.3f}"
                
                # Update stop loss and take profit
                current_close = close.iloc[-1] if not close.empty else 0
                current_atr = atr.iloc[-1] if not atr.empty else 0
                
                if current_atr > 0:
                    if new_signal_type == "BUY":
                        signal['stop_loss'] = current_close - (2 * current_atr)
                        signal['take_profit'] = current_close + (3 * current_atr)
                    elif new_signal_type == "SELL":
                        signal['stop_loss'] = current_close + (2 * current_atr)
                        signal['take_profit'] = current_close - (3 * current_atr)
                
                # Store updated signal
                await self.redis_client.store_signal(signal)
                
                # Add history event for status change
                await self.redis_client.add_signal_history_event(
                    signal_timestamp=signal.get('timestamp'),
                    event_type="status_changed",
                    description=f"Status changed: {original_signal_type} → {new_signal_type}",
                    metadata={
                        "old_status": original_signal_type,
                        "new_status": new_signal_type,
                        "fused_score": fresh_fused_score,
                        "technical_score": tech_score,
                        "sentiment_score": sentiment_score,
                        "reason": "Market conditions changed"
                    }
                )
                
                # Log the change
                await self.redis_client.log_activity(
                    "signal_status_changed",
                    f"Signal status changed: {signal.get('symbol')} {original_signal_type} → {new_signal_type}",
                    signal.get('username', 'system'),
                    {
                        "symbol": signal.get('symbol'),
                        "old_status": original_signal_type,
                        "new_status": new_signal_type,
                        "fused_score": fresh_fused_score,
                        "technical_score": tech_score,
                        "sentiment_score": sentiment_score
                    }
                )
                
                # Send Telegram notification for status change
                await self.send_telegram_notification(
                    signal, original_signal_type, new_signal_type, 
                    fresh_fused_score, tech_score, sentiment_score,
                    current_close
                )
                
                return True
            
            # Status didn't change, but add history event for monitoring activity
            logger.info(f"Signal status unchanged for {signal.get('timestamp')}: {original_signal_type}")
            
            # Add history event for regular monitoring cycle
            await self.redis_client.add_signal_history_event(
                signal_timestamp=signal.get('timestamp'),
                event_type="monitoring_cycle",
                description=f"Status remains {original_signal_type}",
                metadata={
                    "current_status": original_signal_type,
                    "fused_score": fresh_fused_score,
                    "technical_score": tech_score,
                    "sentiment_score": sentiment_score,
                    "buy_threshold": buy_threshold,
                    "sell_threshold": sell_threshold,
                    "reason": "Regular monitoring cycle"
                }
            )
            
            return False
            
        except Exception as e:
            logger.error(f"Error updating signal status: {e}")
            return False
    
    async def send_telegram_notification(self, signal: Dict[str, Any], old_status: str, new_status: str, 
                                       fused_score: float, technical_score: float, sentiment_score: float, 
                                       current_price: float = None) -> None:
        """Send Telegram notification when signal status changes"""
        try:
            # Get configuration
            config = get_config()
            
            # Check if Telegram is enabled and bot token is configured
            if not config.telegram.enabled or not config.telegram.bot_token:
                logger.debug("Telegram not enabled or bot token not configured, skipping notification")
                return
            
            # Get user's Telegram chat_id from Redis
            username = signal.get('username', 'system')
            telegram_connection = await self.redis_client.get_telegram_connection(username)
            if not telegram_connection:
                logger.debug(f"No Telegram connection found for user {username}, skipping notification")
                return
            
            chat_id = telegram_connection.get('chat_id')
            if not chat_id:
                logger.debug(f"No chat_id found for user {username}, skipping notification")
                return
            
            # Create comprehensive notification message (escape special characters for Markdown)
            def escape_markdown(text):
                """Escape special characters for Telegram Markdown"""
                if text is None:
                    return "N/A"
                return str(text).replace('*', '\\*').replace('_', '\\_').replace('[', '\\[').replace('`', '\\`')
            
            symbol = signal.get('symbol', 'Unknown')
            timeframe = signal.get('timeframe', 'Unknown')
            timestamp = signal.get('timestamp', 'Unknown')
            username = signal.get('username', 'system')
            
            # Format the message with emojis and clear structure
            status_emoji = {
                'BUY': '🟢',
                'SELL': '🔴', 
                'HOLD': '🟡'
            }
            
            old_emoji = status_emoji.get(old_status, '⚪')
            new_emoji = status_emoji.get(new_status, '⚪')
            
            # Parse timestamp for better formatting
            try:
                if timestamp != 'Unknown':
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    formatted_time = dt.strftime('%Y-%m-%d %H:%M:%S')
                else:
                    formatted_time = 'Unknown'
            except:
                formatted_time = timestamp
            
            message = f"""🚨 *Signal Status Change Alert* 🚨

{old_emoji} *{escape_markdown(old_status)}* → {new_emoji} *{escape_markdown(new_status)}*

📊 *Symbol:* {escape_markdown(symbol)}
⏰ *Timeframe:* {escape_markdown(timeframe)}
👤 *User:* {escape_markdown(username)}
🕐 *Time:* {escape_markdown(formatted_time)}

📈 *Updated Scores:*
• Fused Score: {fused_score:.4f}
• Technical Score: {technical_score:.4f}
• Sentiment Score: {sentiment_score:.4f}

🎛️ *Applied Thresholds:*
• Buy Threshold: {signal.get('applied_buy_threshold', 'N/A')}
• Sell Threshold: {signal.get('applied_sell_threshold', 'N/A')}
• Technical Weight: {signal.get('applied_tech_weight', 'N/A')}
• Sentiment Weight: {signal.get('applied_sentiment_weight', 'N/A')}

🛡️ *Risk Management:*
• Stop Loss: ${signal.get('stop_loss', 'N/A')}
• Take Profit: ${signal.get('take_profit', 'N/A')}
• Current Price: ${current_price if current_price is not None else signal.get('technical_indicators', {}).get('current_price', 'N/A')}

📊 *Technical Indicators:*
• RSI: {signal.get('technical_indicators', {}).get('rsi_14', 'N/A')}
• MACD: {signal.get('technical_indicators', {}).get('macd', 'N/A')}
• ATR: {signal.get('technical_indicators', {}).get('atr', 'N/A')}

📊 *Market Analysis:*
• Regime: {escape_markdown(signal.get('regime_detection', {}).get('regime_classification', 'unknown'))}
• Trend: {escape_markdown(signal.get('multi_timeframe', {}).get('overall_trend', 'neutral'))}
• Volatility: {escape_markdown(signal.get('regime_detection', {}).get('volatility_state', 'medium'))}

🔍 *Change Reason:* Market conditions updated based on real\\-time analysis

⚡ *Monitoring System:* Automated status update"""
            
            # Send to Telegram
            url = f"https://api.telegram.org/bot{config.telegram.bot_token}/sendMessage"
            payload = {
                "chat_id": chat_id,
                "text": message,
                "parse_mode": "Markdown"
            }
            
            response = requests.post(url, json=payload, timeout=10)
            
            if response.status_code == 200:
                logger.info(f"Telegram notification sent successfully for {symbol} status change")
            else:
                logger.error(f"Failed to send Telegram notification: {response.status_code} - {response.text}")
                
        except Exception as e:
            logger.error(f"Error sending Telegram notification: {e}")
    
    async def start_monitoring(self, interval_minutes: int = 5):
        """Start continuous monitoring of signals"""
        logger.info(f"🚀 [MONITOR-START] Starting signal monitoring with {interval_minutes} minute intervals")
        
        while True:
            try:
                result = await self.monitor_all_signals()
                logger.info(f"✅ [MONITOR-CYCLE] Monitoring cycle completed: {result}")
                
                # Wait for next cycle
                await asyncio.sleep(interval_minutes * 60)
                
            except Exception as e:
                logger.error(f"Error in monitoring cycle: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying
