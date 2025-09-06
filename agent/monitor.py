"""
Real-time Signal Monitoring System
Monitors existing signals and updates their status based on changing market conditions
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pytz

from .cache.redis_client import TradingAgentRedis
from .data.ccxt_client import fetch_ohlcv as fetch_ccxt_ohlcv
from .data.alpaca_client import fetch_ohlcv as fetch_alpaca_ohlcv
from .models.sentiment import SentimentAnalyzer
from .news.rss import fetch_headlines
from .news.reddit import fetch_crypto_reddit_posts, fetch_stock_reddit_posts

logger = logging.getLogger(__name__)

class SignalMonitor:
    """Monitors and updates trading signals in real-time"""
    
    def __init__(self, redis_client: TradingAgentRedis, sentiment_analyzer: Optional[SentimentAnalyzer] = None):
        self.redis_client = redis_client
        self.sentiment_analyzer = sentiment_analyzer
        self.israel_tz = pytz.timezone('Asia/Jerusalem')
        
    async def monitor_all_signals(self) -> Dict[str, Any]:
        """Monitor all active signals and update their status"""
        try:
            # Get all signals from Redis
            all_signals = await self.redis_client.get_signals(limit=1000)
            
            logger.info(f"Retrieved {len(all_signals) if all_signals else 0} signals from Redis")
            
            if not all_signals:
                logger.info("No signals to monitor")
                return {"monitored": 0, "updated": 0, "errors": 0}
            
            logger.info(f"Monitoring {len(all_signals)} signals")
            
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
            
            logger.info(f"Monitoring completed: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error in monitor_all_signals: {e}")
            return {"monitored": 0, "updated": 0, "errors": 1, "error": str(e)}
    
    async def _monitor_symbol_signals(self, symbol: str, signals: List[Dict]) -> int:
        """Monitor signals for a specific symbol"""
        try:
            # Fetch fresh market data
            timeframe = signals[0].get('timeframe', '1h')
            
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
            
            all_texts = []
            
            # Fetch RSS headlines
            logger.info("Monitoring: Fetching RSS headlines...")
            headlines = fetch_headlines([
                "https://news.google.com/rss/search?q=stock+market&hl=en-US&gl=US&ceid=US:en",
                "https://news.google.com/rss/search?q=crypto+btc&hl=en-US&gl=US&ceid=US:en",
            ], limit_per_feed=10)  # Reduced to make room for Reddit
            if headlines:
                all_texts.extend(headlines)
                logger.info(f"Monitoring: RSS collected {len(headlines)} headlines")
            
            # Fetch Reddit posts (mix of crypto and stock for monitoring)
            logger.info("Monitoring: Fetching Reddit posts...")
            try:
                crypto_posts = fetch_crypto_reddit_posts(limit_per_subreddit=3)
                stock_posts = fetch_stock_reddit_posts(limit_per_subreddit=3)
                
                if crypto_posts:
                    all_texts.extend(crypto_posts)
                    logger.info(f"Monitoring: Reddit collected {len(crypto_posts)} crypto posts")
                if stock_posts:
                    all_texts.extend(stock_posts)
                    logger.info(f"Monitoring: Reddit collected {len(stock_posts)} stock posts")
                    
            except Exception as e:
                logger.warning(f"Monitoring: Could not fetch Reddit posts: {e}")
            
            if all_texts:
                # Use optimal sample size with mixed sources
                import random
                sample_size = min(20, len(all_texts))  # Increased to 20 for mixed sources
                shuffled_texts = all_texts.copy()
                random.shuffle(shuffled_texts)
                text_sample = shuffled_texts[:sample_size]
                logger.info(f"Monitoring: Analyzing {len(text_sample)} texts (RSS + Reddit) for sentiment")
                sentiment_score = self.sentiment_analyzer.score(text_sample)
                logger.info(f"Monitoring: Sentiment analysis result: {sentiment_score:.3f}")
                return sentiment_score
            
            return 0.0
            
        except Exception as e:
            logger.warning(f"Could not fetch fresh sentiment: {e}")
            return 0.0
    
    async def _update_signal_status(self, signal: Dict, ohlcv_data: Any, sentiment_score: float) -> bool:
        """Update a single signal's status based on fresh data"""
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
                logger.info(f"Signal status changed: {signal.get('symbol')} {original_signal_type} → {new_signal_type}")
                
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
                
                return True
            
            # Status didn't change, but add history event for monitoring activity
            logger.info(f"Signal status unchanged for {signal.get('timestamp')}: {original_signal_type}")
            
            # Add history event for regular monitoring cycle
            await self.redis_client.add_signal_history_event(
                signal_timestamp=signal.get('timestamp'),
                event_type="monitoring_cycle",
                description=f"Monitoring cycle completed - Status remains {original_signal_type}",
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
    
    async def start_monitoring(self, interval_minutes: int = 5):
        """Start continuous monitoring of signals"""
        logger.info(f"Starting signal monitoring with {interval_minutes} minute intervals")
        
        while True:
            try:
                result = await self.monitor_all_signals()
                logger.info(f"Monitoring cycle completed: {result}")
                
                # Wait for next cycle
                await asyncio.sleep(interval_minutes * 60)
                
            except Exception as e:
                logger.error(f"Error in monitoring cycle: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying
