#!/usr/bin/env python3
"""
Signal Generation Module
Handles trading signal generation with technical analysis, sentiment analysis, and risk management
"""

import logging
import random
from typing import Dict, Any, Optional, List
from datetime import datetime
import pytz
import requests

from pydantic import BaseModel
from fastapi import HTTPException

# Import trading agent modules
from agent.config import load_config_from_env
from agent.data.ccxt_client import fetch_ohlcv
from agent.data.alpaca_client import fetch_ohlcv as fetch_alpaca_ohlcv
from agent.indicators import compute_rsi, compute_ema, compute_macd, compute_atr
from agent.models.sentiment import SentimentAnalyzer
from agent.news.rss import fetch_headlines
from agent.news.reddit import fetch_crypto_reddit_posts, fetch_stock_reddit_posts
from agent.cache.redis_client import get_redis_client

# Configure logging
logger = logging.getLogger(__name__)

# Israel timezone
ISRAEL_TZ = pytz.timezone('Asia/Jerusalem')

def get_israel_time():
    """Get current time in Israel timezone with microsecond precision"""
    return datetime.now(ISRAEL_TZ)


class SignalRequest(BaseModel):
    """Request model for signal generation"""
    symbol: str
    timeframe: str = "1h"
    buy_threshold: Optional[float] = None
    sell_threshold: Optional[float] = None
    technical_weight: Optional[float] = None
    sentiment_weight: Optional[float] = None


class TradingSignal(BaseModel):
    """Trading signal model"""
    symbol: str
    timeframe: str = "1h"
    timestamp: str
    signal_type: str  # "BUY", "SELL", "HOLD"
    confidence: float
    technical_score: float
    sentiment_score: float
    fused_score: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    reasoning: str
    # Store the thresholds that were actually used for this signal
    applied_buy_threshold: Optional[float] = None
    applied_sell_threshold: Optional[float] = None
    applied_tech_weight: Optional[float] = None
    applied_sentiment_weight: Optional[float] = None


class SignalGenerator:
    """Main signal generation class"""
    
    def __init__(self):
        self.sentiment_analyzer = None
        self._initialize_sentiment_analyzer()
    
    def _initialize_sentiment_analyzer(self):
        """Initialize sentiment analyzer"""
        try:
            config = load_config_from_env()
            self.sentiment_analyzer = SentimentAnalyzer(config.models.sentiment_model)
            logger.info("Sentiment analyzer initialized successfully")
        except Exception as e:
            logger.warning(f"Could not initialize sentiment analyzer: {e}")
            self.sentiment_analyzer = None
    
    async def fetch_market_data(self, symbol: str, timeframe: str):
        """Fetch market data for the given symbol and timeframe"""
        logger.info(f"Fetching market data for {symbol} @ {timeframe}")
        
        if "/" in symbol:  # Crypto
            logger.info(f"Fetching crypto data for {symbol}")
            ohlcv = fetch_ohlcv(symbol, timeframe)
        else:  # Stock/ETF
            logger.info(f"Fetching stock data for {symbol}")
            ohlcv = fetch_alpaca_ohlcv(symbol, timeframe)
        
        logger.info(f"OHLCV type: {type(ohlcv)}, value: {ohlcv if isinstance(ohlcv, str) else 'DataFrame'}")
        logger.info(f"OHLCV DataFrame shape: {ohlcv.shape}")
        logger.info(f"OHLCV DataFrame columns: {list(ohlcv.columns)}")
        logger.info(f"OHLCV DataFrame head (first 3 rows):\n{ohlcv.head(3).to_string()}")
        logger.info(f"OHLCV DataFrame tail (last 3 rows):\n{ohlcv.tail(3).to_string()}")
        
        if ohlcv is None or ohlcv.empty:
            raise HTTPException(status_code=400, detail="Could not fetch market data")
        
        return ohlcv
    
    def calculate_technical_indicators(self, ohlcv):
        """Calculate technical indicators from OHLCV data"""
        logger.info("Starting indicators calculation")
        
        close = ohlcv['close']
        high = ohlcv['high']
        low = ohlcv['low']
        
        # Calculate indicators
        rsi = compute_rsi(close)
        ema_12 = compute_ema(close, 12)
        ema_26 = compute_ema(close, 26)
        macd_df = compute_macd(close)
        macd = macd_df['macd']
        macd_signal = macd_df['signal']
        macd_hist = macd_df['hist']
        atr = compute_atr(high, low, close)
        
        logger.info("Indicators calculation completed")
        
        return {
            'close': close,
            'high': high,
            'low': low,
            'rsi': rsi,
            'ema_12': ema_12,
            'ema_26': ema_26,
            'macd': macd,
            'macd_signal': macd_signal,
            'macd_hist': macd_hist,
            'atr': atr
        }
    
    async def analyze_sentiment(self, symbol: str) -> float:
        """Analyze sentiment from multiple sources (RSS + Reddit)"""
        logger.info("Starting sentiment analysis from multiple sources")
        sentiment_score = 0.0
        
        if not self.sentiment_analyzer:
            logger.warning("Sentiment analyzer not available")
            return sentiment_score
        
        try:
            all_texts = []
            
            # Fetch RSS headlines
            logger.info("Fetching RSS headlines...")
            headlines = fetch_headlines([
                "https://news.google.com/rss/search?q=stock+market&hl=en-US&gl=US&ceid=US:en",
                "https://news.google.com/rss/search?q=crypto+btc&hl=en-US&gl=US&ceid=US:en",
            ], limit_per_feed=10)  # Reduced to 10 to make room for Reddit
            
            if headlines:
                all_texts.extend(headlines)
                logger.info(f"RSS: Collected {len(headlines)} headlines")
            
            # Fetch Reddit posts based on symbol type
            logger.info("Fetching Reddit posts...")
            if "BTC" in symbol or "ETH" in symbol or "crypto" in symbol.lower():
                # Crypto symbol - fetch crypto Reddit posts
                reddit_posts = fetch_crypto_reddit_posts(limit_per_subreddit=5)
                if reddit_posts:
                    all_texts.extend(reddit_posts)
                logger.info(f"Reddit: Collected {len(reddit_posts)} crypto posts")
            else:
                # Stock symbol - fetch stock Reddit posts
                reddit_posts = fetch_stock_reddit_posts(limit_per_subreddit=5)
                if reddit_posts:
                    all_texts.extend(reddit_posts)
                    logger.info(f"Reddit: Collected {len(reddit_posts)} stock posts")
            
            if all_texts:
                # Use optimal sample size with mixed sources
                sample_size = min(20, len(all_texts))  # Increased to 20 for mixed sources
                shuffled_texts = all_texts.copy()
                random.shuffle(shuffled_texts)
                text_sample = shuffled_texts[:sample_size]
                logger.info(f"Analyzing {len(text_sample)} texts (RSS + Reddit) for sentiment")
                sentiment_score = self.sentiment_analyzer.score(text_sample)
                logger.info(f"Sentiment analysis result: {sentiment_score:.3f}")
            else:
                logger.warning("No texts collected from any source")
                
        except Exception as e:
            logger.warning(f"Could not fetch sentiment: {e}")
        
        logger.info("Sentiment analysis completed")
        return sentiment_score
    
    def calculate_technical_score(self, indicators: Dict[str, Any]) -> float:
        """Calculate technical score from indicators"""
        logger.info("Calculating technical score")
        
        close = indicators['close']
        rsi = indicators['rsi']
        macd_hist = indicators['macd_hist']
        
        logger.info(f"Close type: {type(close)}, value: {close if isinstance(close, str) else 'Series'}")
        logger.info(f"RSI type: {type(rsi)}, empty: {rsi.empty if hasattr(rsi, 'empty') else 'no empty attr'}")
        logger.info(f"MACD hist type: {type(macd_hist)}, empty: {macd_hist.empty if hasattr(macd_hist, 'empty') else 'no empty attr'}")
        
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
        
        logger.info(f"Technical score calculated: {tech_score:.3f}")
        return tech_score
    
    def determine_signal_type(self, fused_score: float, buy_threshold: float, sell_threshold: float) -> str:
        """Determine signal type based on fused score and thresholds"""
        if fused_score >= buy_threshold:
            return "BUY"
        elif fused_score <= sell_threshold:
            return "SELL"
        else:
            return "HOLD"
    
    def calculate_risk_management(self, indicators: Dict[str, Any], signal_type: str) -> tuple[Optional[float], Optional[float]]:
        """Calculate stop loss and take profit levels"""
        close = indicators['close']
        atr = indicators['atr']
        
        stop_loss = None
        take_profit = None
        
        if not close.empty and not atr.empty:
            current_close = close.iloc[-1]
            current_atr = atr.iloc[-1]
            
            if current_atr > 0:
                if signal_type == "BUY":
                    stop_loss = current_close - (2 * current_atr)
                    take_profit = current_close + (3 * current_atr)
                elif signal_type == "SELL":
                    stop_loss = current_close + (2 * current_atr)
                    take_profit = current_close - (3 * current_atr)
        
        return stop_loss, take_profit
    
    async def store_signal(self, signal: TradingSignal, username: str) -> bool:
        """Store the generated signal in Redis"""
        try:
            redis_client = await get_redis_client()
            if not redis_client:
                logger.warning("Redis not available - signal not persisted")
                return False
            
            # Add user information to signal
            signal_dict = signal.dict()
            signal_dict['username'] = username
            
            # Store the signal
            await redis_client.store_signal(signal_dict)
            
            # Add initial history event
            await redis_client.add_signal_history_event(
                signal_timestamp=signal.timestamp,
                event_type="signal_created",
                description=f"Signal created: {signal.signal_type} for {signal.symbol}",
                metadata={
                    "signal_type": signal.signal_type,
                    "confidence": signal.confidence,
                    "fused_score": signal.fused_score,
                    "technical_score": signal.technical_score,
                    "sentiment_score": signal.sentiment_score,
                    "applied_thresholds": {
                        "buy_threshold": signal.applied_buy_threshold,
                        "sell_threshold": signal.applied_sell_threshold,
                        "tech_weight": signal.applied_tech_weight,
                        "sentiment_weight": signal.applied_sentiment_weight
                    },
                    "created_by": username
                }
            )
            
            # Log activity
            await redis_client.log_activity(
                "signal_generated",
                f"Generated {signal.signal_type} trading tip for {signal.symbol} (confidence: {(signal.confidence * 100):.1f}%)",
                username,
                {
                    "symbol": signal.symbol,
                    "signal_type": signal.signal_type,
                    "confidence": signal.confidence,
                    "technical_score": signal.technical_score,
                    "sentiment_score": signal.sentiment_score,
                    "fused_score": signal.fused_score
                }
            )
            
            logger.info(f"Signal stored in Redis for user {username}: {signal.symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store signal in Redis: {e}")
            return False
    
    async def send_telegram_notification(self, signal: TradingSignal, username: str) -> None:
        """Send Telegram notification for new signal generation"""
        try:
            # Get configuration
            config = load_config_from_env()
            
            # Check if Telegram is enabled and bot token is configured
            if not config.telegram.enabled or not config.telegram.bot_token:
                logger.debug("Telegram not enabled or bot token not configured, skipping notification")
                return
            
            # Get user's Telegram chat_id from Redis
            from agent.cache.redis_client import get_redis_client
            redis_client = await get_redis_client()
            if not redis_client:
                logger.debug("Redis not available, skipping Telegram notification")
                return
            
            telegram_connection = await redis_client.get_telegram_connection(username)
            if not telegram_connection:
                logger.debug(f"No Telegram connection found for user {username}, skipping notification")
                return
            
            chat_id = telegram_connection.get('chat_id')
            if not chat_id:
                logger.debug(f"No chat_id found for user {username}, skipping notification")
                return
            
            # Create notification message
            status_emoji = {
                'BUY': '🟢',
                'SELL': '🔴', 
                'HOLD': '🟡'
            }
            
            emoji = status_emoji.get(signal.signal_type, '⚪')
            
            message = f"""
🎯 **New Trading Signal Generated** 🎯

{emoji} **{signal.signal_type}** Signal

📊 **Symbol**: {signal.symbol}
⏰ **Timeframe**: {signal.timeframe}
👤 **User**: {username}
🕐 **Time**: {signal.timestamp}

📈 **Scores**:
• Fused Score: {signal.fused_score:.3f}
• Technical Score: {signal.technical_score:.3f}
• Sentiment Score: {signal.sentiment_score:.3f}
• Confidence: {signal.confidence:.3f}

💡 **Risk Management**:
• Stop Loss: {signal.stop_loss or 'N/A'}
• Take Profit: {signal.take_profit or 'N/A'}

🎛️ **Parameters Used**:
• Buy Threshold: {signal.applied_buy_threshold or 'Default'}
• Sell Threshold: {signal.applied_sell_threshold or 'Default'}
• Tech Weight: {signal.applied_tech_weight or 'Default'}
• Sentiment Weight: {signal.applied_sentiment_weight or 'Default'}

💭 **Reasoning**: {signal.reasoning}
            """.strip()
            
            # Send to Telegram
            url = f"https://api.telegram.org/bot{config.telegram.bot_token}/sendMessage"
            payload = {
                "chat_id": chat_id,
                "text": message,
                "parse_mode": "Markdown"
            }
            
            response = requests.post(url, json=payload, timeout=10)
            
            if response.status_code == 200:
                logger.info(f"Telegram notification sent successfully for new {signal.symbol} signal")
            else:
                logger.error(f"Failed to send Telegram notification: {response.status_code} - {response.text}")
                
        except Exception as e:
            logger.error(f"Error sending Telegram notification: {e}")
    
    async def generate_signal(self, request: SignalRequest, username: str) -> TradingSignal:
        """Generate a new trading signal for a symbol"""
        try:
            logger.info(f"Starting signal generation for {request.symbol}")
            
            # Step 1: Fetch market data
            ohlcv = await self.fetch_market_data(request.symbol, request.timeframe)
            
            # Step 2: Calculate technical indicators
            indicators = self.calculate_technical_indicators(ohlcv)
            
            # Step 3: Analyze sentiment
            sentiment_score = await self.analyze_sentiment(request.symbol)
            
            # Step 4: Calculate technical score
            tech_score = self.calculate_technical_score(indicators)
            
            # Step 5: Get configuration and apply custom parameters
            config = load_config_from_env()
            tech_weight = request.technical_weight if request.technical_weight is not None else config.thresholds.technical_weight
            sentiment_weight = request.sentiment_weight if request.sentiment_weight is not None else config.thresholds.sentiment_weight
            
            # Calculate fused score
            fused_score = tech_weight * tech_score + sentiment_weight * sentiment_score
            
            # Define thresholds - use config defaults if not provided
            buy_threshold = request.buy_threshold if request.buy_threshold is not None else config.thresholds.buy_threshold
            # If custom sell_threshold is provided, use it; otherwise use negative of buy_threshold
            if request.sell_threshold is not None:
                sell_threshold = request.sell_threshold
            else:
                sell_threshold = -buy_threshold
            
            # Debug logging for applied thresholds
            logger.info(f"Applied threshold values: buy_threshold={buy_threshold}, sell_threshold={sell_threshold}, technical_weight={tech_weight}, sentiment_weight={sentiment_weight}")
            
            # Step 6: Determine signal type
            signal_type = self.determine_signal_type(fused_score, buy_threshold, sell_threshold)
            
            # Step 7: Calculate risk management
            stop_loss, take_profit = self.calculate_risk_management(indicators, signal_type)
            
            # Step 8: Create signal object
            signal = TradingSignal(
                symbol=request.symbol,
                timeframe=request.timeframe,
                timestamp=get_israel_time().isoformat(),
                signal_type=signal_type,
                confidence=abs(fused_score),
                technical_score=tech_score,
                sentiment_score=sentiment_score,
                fused_score=fused_score,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reasoning=f"Technical: {tech_score:.2f}, Sentiment: {sentiment_score:.2f}, Fused: {fused_score:.2f}",
                applied_buy_threshold=buy_threshold,
                applied_sell_threshold=sell_threshold,
                applied_tech_weight=tech_weight,
                applied_sentiment_weight=sentiment_weight
            )
            
            # Step 9: Store signal
            await self.store_signal(signal, username)
            
            # Step 10: Send Telegram notification for new signal
            await self.send_telegram_notification(signal, username)
            
            logger.info(f"Signal generation completed for {request.symbol}: {signal_type}")
            return signal
            
        except Exception as e:
            logger.error(f"Error generating signal: {e}")
            raise HTTPException(status_code=500, detail=str(e))


# Global signal generator instance
signal_generator = SignalGenerator()


async def generate_trading_signal(request: SignalRequest, username: str) -> TradingSignal:
    """Main function to generate a trading signal"""
    return await signal_generator.generate_signal(request, username)
