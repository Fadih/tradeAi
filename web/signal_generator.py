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
import pandas as pd

from pydantic import BaseModel
from fastapi import HTTPException

# Import trading agent modules
from agent.config import load_config_from_env
from agent.data.ccxt_client import fetch_ohlcv
from agent.data.alpaca_client import fetch_ohlcv as fetch_alpaca_ohlcv
from agent.indicators import compute_rsi, compute_ema, compute_macd, compute_atr, compute_rsi_enhanced, rsi_signal, stoch_rsi
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
        """Calculate technical indicators from OHLCV data with enhanced RSI"""
        logger.info("🔧 STARTING ENHANCED INDICATORS CALCULATION")
        
        close = ohlcv['close']
        high = ohlcv['high']
        low = ohlcv['low']
        
        # Load configuration for RSI settings
        config = load_config_from_env()
        rsi_config = config.technical_analysis.rsi
        
        logger.info(f"📊 RSI Configuration:")
        logger.info(f"   • Period: {rsi_config.period}")
        logger.info(f"   • Method: {rsi_config.method}")
        logger.info(f"   • Signal Period: {rsi_config.signal_period}")
        
        # Calculate enhanced RSI with multiple variants
        rsi_enhanced = compute_rsi_enhanced(
            close, 
            period=rsi_config.period, 
            method=rsi_config.method
        )
        
        # Extract RSI variants
        rsi = rsi_enhanced['rsi']
        rsi_signal_line = rsi_enhanced['rsi_signal']
        stoch_rsi_k = rsi_enhanced['stoch_rsi_k']
        stoch_rsi_d = rsi_enhanced['stoch_rsi_d']
        
        # Calculate other indicators
        ema_12 = compute_ema(close, 12)
        ema_26 = compute_ema(close, 26)
        macd_df = compute_macd(close)
        macd = macd_df['macd']
        macd_signal = macd_df['signal']
        macd_hist = macd_df['hist']
        atr = compute_atr(high, low, close)
        
        logger.info("✅ Enhanced indicators calculation completed")
        logger.info(f"📈 RSI Variants calculated:")
        logger.info(f"   • Main RSI: {len(rsi)} values")
        logger.info(f"   • RSI Signal: {len(rsi_signal_line)} values")
        logger.info(f"   • Stochastic RSI K: {len(stoch_rsi_k)} values")
        logger.info(f"   • Stochastic RSI D: {len(stoch_rsi_d)} values")
        
        return {
            'close': close,
            'high': high,
            'low': low,
            'rsi': rsi,
            'rsi_signal': rsi_signal_line,
            'stoch_rsi_k': stoch_rsi_k,
            'stoch_rsi_d': stoch_rsi_d,
            'ema_12': ema_12,
            'ema_26': ema_26,
            'macd': macd,
            'macd_signal': macd_signal,
            'macd_hist': macd_hist,
            'atr': atr
        }
    
    async def analyze_sentiment(self, symbol: str) -> float:
        """Analyze sentiment from multiple sources (RSS + Reddit)"""
        logger.info("💭 STARTING SENTIMENT ANALYSIS")
        sentiment_score = 0.0
        
        if not self.sentiment_analyzer:
            logger.warning("⚠️ Sentiment analyzer not available, returning 0.0")
            return sentiment_score
        
        try:
            all_texts = []
            
            # Fetch RSS headlines
            logger.info("📰 FETCHING RSS HEADLINES")
            headlines = fetch_headlines([
                "https://news.google.com/rss/search?q=stock+market&hl=en-US&gl=US&ceid=US:en",
                "https://news.google.com/rss/search?q=crypto+btc&hl=en-US&gl=US&ceid=US:en",
            ], limit_per_feed=10)  # Reduced to 10 to make room for Reddit
            
            if headlines:
                all_texts.extend(headlines)
                logger.info(f"✅ RSS: Collected {len(headlines)} headlines")
            else:
                logger.warning("⚠️ No RSS headlines collected")
            
            # Fetch Reddit posts based on symbol type
            logger.info("🔴 FETCHING REDDIT POSTS")
            symbol_type = "crypto" if ("BTC" in symbol or "ETH" in symbol or "crypto" in symbol.lower()) else "stock"
            logger.info(f"📊 Symbol type detected: {symbol_type}")
            
            if symbol_type == "crypto":
                # Crypto symbol - fetch crypto Reddit posts
                reddit_posts = fetch_crypto_reddit_posts(limit_per_subreddit=5)
                if reddit_posts:
                    all_texts.extend(reddit_posts)
                logger.info(f"✅ Reddit: Collected {len(reddit_posts) if reddit_posts else 0} crypto posts")
            else:
                # Stock symbol - fetch stock Reddit posts
                reddit_posts = fetch_stock_reddit_posts(limit_per_subreddit=5)
                if reddit_posts:
                    all_texts.extend(reddit_posts)
                logger.info(f"✅ Reddit: Collected {len(reddit_posts) if reddit_posts else 0} stock posts")
            
            logger.info(f"📊 TOTAL TEXTS COLLECTED: {len(all_texts)}")
            logger.info(f"   • RSS Headlines: {len(headlines) if headlines else 0}")
            logger.info(f"   • Reddit Posts: {len(all_texts) - (len(headlines) if headlines else 0)}")
            
            if all_texts:
                # Use optimal sample size with mixed sources
                sample_size = min(20, len(all_texts))  # Increased to 20 for mixed sources
                shuffled_texts = all_texts.copy()
                random.shuffle(shuffled_texts)
                text_sample = shuffled_texts[:sample_size]
                logger.info(f"🎯 ANALYZING {len(text_sample)} texts for sentiment (sample size: {sample_size})")
                logger.info(f"   • Sample breakdown: {len([t for t in text_sample if t in headlines]) if headlines else 0} RSS, {len([t for t in text_sample if t not in headlines]) if headlines else len(text_sample)} Reddit")
                
                sentiment_score = self.sentiment_analyzer.score(text_sample)
                logger.info(f"✅ Sentiment analysis result: {sentiment_score:.4f}")
                
                # Log some sample texts for debugging
                logger.info("📝 SAMPLE TEXTS FOR SENTIMENT ANALYSIS:")
                for i, text in enumerate(text_sample[:3]):  # Show first 3 texts
                    logger.info(f"   {i+1}. {text[:100]}{'...' if len(text) > 100 else ''}")
            else:
                logger.warning("⚠️ No texts collected from any source, using default sentiment score of 0.0")
                
        except Exception as e:
            logger.error(f"❌ Could not fetch sentiment: {e}")
        
        logger.info(f"✅ Sentiment analysis completed - Final score: {sentiment_score:.4f}")
        return sentiment_score
    
    def calculate_technical_score(self, indicators: Dict[str, Any]) -> float:
        """Calculate enhanced technical score using multiple RSI variants"""
        logger.info("🔧 CALCULATING ENHANCED TECHNICAL SCORE")
        
        close = indicators['close']
        rsi = indicators['rsi']
        rsi_signal = indicators['rsi_signal']
        stoch_rsi_k = indicators['stoch_rsi_k']
        stoch_rsi_d = indicators['stoch_rsi_d']
        macd_hist = indicators['macd_hist']
        
        logger.info(f"📊 Data validation:")
        logger.info(f"   • Close type: {type(close)}, empty: {close.empty if hasattr(close, 'empty') else 'no empty attr'}")
        logger.info(f"   • RSI type: {type(rsi)}, empty: {rsi.empty if hasattr(rsi, 'empty') else 'no empty attr'}")
        logger.info(f"   • RSI Signal type: {type(rsi_signal)}, empty: {rsi_signal.empty if hasattr(rsi_signal, 'empty') else 'no empty attr'}")
        logger.info(f"   • Stoch RSI K type: {type(stoch_rsi_k)}, empty: {stoch_rsi_k.empty if hasattr(stoch_rsi_k, 'empty') else 'no empty attr'}")
        logger.info(f"   • MACD hist type: {type(macd_hist)}, empty: {macd_hist.empty if hasattr(macd_hist, 'empty') else 'no empty attr'}")
        
        tech_score = 0.0
        if not close.empty:
            current_close = close.iloc[-1]
            current_rsi = rsi.iloc[-1] if not rsi.empty and not pd.isna(rsi.iloc[-1]) else 50
            current_rsi_signal = rsi_signal.iloc[-1] if not rsi_signal.empty and not pd.isna(rsi_signal.iloc[-1]) else 50
            current_stoch_k = stoch_rsi_k.iloc[-1] if not stoch_rsi_k.empty and not pd.isna(stoch_rsi_k.iloc[-1]) else 50
            current_stoch_d = stoch_rsi_d.iloc[-1] if not stoch_rsi_d.empty and not pd.isna(stoch_rsi_d.iloc[-1]) else 50
            current_macd_hist = macd_hist.iloc[-1] if not macd_hist.empty else 0
            
            logger.info(f"📈 Current indicator values:")
            logger.info(f"   • Close Price: ${current_close:.2f}")
            logger.info(f"   • RSI: {current_rsi:.2f}")
            logger.info(f"   • RSI Signal: {current_rsi_signal:.2f}")
            logger.info(f"   • Stochastic RSI K: {current_stoch_k:.2f}")
            logger.info(f"   • Stochastic RSI D: {current_stoch_d:.2f}")
            logger.info(f"   • MACD Histogram: {current_macd_hist:.4f}")
            
            # Enhanced RSI scoring with multiple components
            scores = []
            
            # 1. Main RSI score (normalized to [-1, 1])
            rsi_score = (current_rsi - 50) / 50
            scores.append(("RSI", rsi_score, 0.3))  # 30% weight
            logger.info(f"   • RSI Score: {rsi_score:.4f} = ({current_rsi:.2f} - 50) / 50")
            
            # 2. RSI Signal cross score (momentum)
            rsi_cross_score = (current_rsi - current_rsi_signal) / 10  # Normalize cross difference
            rsi_cross_score = max(min(rsi_cross_score, 1), -1)  # Clip to [-1, 1]
            scores.append(("RSI Cross", rsi_cross_score, 0.2))  # 20% weight
            logger.info(f"   • RSI Cross Score: {rsi_cross_score:.4f} = ({current_rsi:.2f} - {current_rsi_signal:.2f}) / 10")
            
            # 3. Stochastic RSI score (overbought/oversold timing)
            stoch_score = (current_stoch_k - 50) / 50
            scores.append(("Stoch RSI", stoch_score, 0.2))  # 20% weight
            logger.info(f"   • Stoch RSI Score: {stoch_score:.4f} = ({current_stoch_k:.2f} - 50) / 50")
            
            # 4. Stochastic RSI cross score (momentum confirmation)
            stoch_cross_score = (current_stoch_k - current_stoch_d) / 10
            stoch_cross_score = max(min(stoch_cross_score, 1), -1)
            scores.append(("Stoch Cross", stoch_cross_score, 0.1))  # 10% weight
            logger.info(f"   • Stoch Cross Score: {stoch_cross_score:.4f} = ({current_stoch_k:.2f} - {current_stoch_d:.2f}) / 10")
            
            # 5. MACD histogram score (trend momentum)
            macd_score = max(min(current_macd_hist / 1000, 1), -1)
            scores.append(("MACD", macd_score, 0.2))  # 20% weight
            logger.info(f"   • MACD Score: {macd_score:.4f} = max(min({current_macd_hist:.4f} / 1000, 1), -1)")
            
            # Calculate weighted technical score
            tech_score = sum(score * weight for _, score, weight in scores)
            
            logger.info(f"🧮 ENHANCED TECHNICAL SCORE CALCULATION:")
            for name, score, weight in scores:
                contribution = score * weight
                logger.info(f"   • {name}: {score:.4f} × {weight:.1f} = {contribution:.4f}")
            logger.info(f"   • Total Technical Score: {tech_score:.4f}")
            
        else:
            logger.warning("⚠️ Close price data is empty, using default technical score of 0.0")
        
        logger.info(f"✅ Enhanced technical score calculated: {tech_score:.4f}")
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
        logger.info("🛡️ CALCULATING RISK MANAGEMENT")
        
        close = indicators['close']
        atr = indicators['atr']
        
        stop_loss = None
        take_profit = None
        
        logger.info(f"📊 Risk management data validation:")
        logger.info(f"   • Close data empty: {close.empty if hasattr(close, 'empty') else 'no empty attr'}")
        logger.info(f"   • ATR data empty: {atr.empty if hasattr(atr, 'empty') else 'no empty attr'}")
        
        if not close.empty and not atr.empty:
            current_close = close.iloc[-1]
            current_atr = atr.iloc[-1]
            
            logger.info(f"📈 Current values for risk calculation:")
            logger.info(f"   • Current Close: ${current_close:.2f}")
            logger.info(f"   • Current ATR: {current_atr:.4f}")
            logger.info(f"   • Signal Type: {signal_type}")
            
            if current_atr > 0:
                if signal_type == "BUY":
                    stop_loss = current_close - (2 * current_atr)
                    take_profit = current_close + (3 * current_atr)
                    logger.info(f"🟢 BUY signal risk calculation:")
                    logger.info(f"   • Stop Loss: ${current_close:.2f} - (2 × {current_atr:.4f}) = ${stop_loss:.2f}")
                    logger.info(f"   • Take Profit: ${current_close:.2f} + (3 × {current_atr:.4f}) = ${take_profit:.2f}")
                elif signal_type == "SELL":
                    stop_loss = current_close + (2 * current_atr)
                    take_profit = current_close - (3 * current_atr)
                    logger.info(f"🔴 SELL signal risk calculation:")
                    logger.info(f"   • Stop Loss: ${current_close:.2f} + (2 × {current_atr:.4f}) = ${stop_loss:.2f}")
                    logger.info(f"   • Take Profit: ${current_close:.2f} - (3 × {current_atr:.4f}) = ${take_profit:.2f}")
                else:
                    logger.info(f"🟡 HOLD signal - no risk management levels calculated")
                
                if stop_loss and take_profit:
                    risk_amount = abs(current_close - stop_loss)
                    reward_amount = abs(take_profit - current_close)
                    risk_reward_ratio = reward_amount / risk_amount if risk_amount > 0 else 0
                    logger.info(f"📊 Risk/Reward Analysis:")
                    logger.info(f"   • Risk Amount: ${risk_amount:.2f}")
                    logger.info(f"   • Reward Amount: ${reward_amount:.2f}")
                    logger.info(f"   • Risk/Reward Ratio: 1:{risk_reward_ratio:.2f}")
            else:
                logger.warning("⚠️ ATR is zero or negative, cannot calculate risk management levels")
        else:
            logger.warning("⚠️ Missing close or ATR data, cannot calculate risk management levels")
        
        logger.info(f"✅ Risk management calculation completed:")
        logger.info(f"   • Stop Loss: ${stop_loss:.2f}" if stop_loss else "   • Stop Loss: N/A")
        logger.info(f"   • Take Profit: ${take_profit:.2f}" if take_profit else "   • Take Profit: N/A")
        
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
            logger.info("=" * 80)
            logger.info(f"🚀 STARTING SIGNAL GENERATION")
            logger.info(f"📊 Symbol: {request.symbol}")
            logger.info(f"⏰ Timeframe: {request.timeframe}")
            logger.info(f"👤 Username: {username}")
            logger.info(f"🕐 Timestamp: {get_israel_time().isoformat()}")
            logger.info("=" * 80)
            
            # Log input parameters
            logger.info("📋 INPUT PARAMETERS:")
            logger.info(f"   • Buy Threshold: {request.buy_threshold}")
            logger.info(f"   • Sell Threshold: {request.sell_threshold}")
            logger.info(f"   • Technical Weight: {request.technical_weight}")
            logger.info(f"   • Sentiment Weight: {request.sentiment_weight}")
            
            # Step 1: Fetch market data
            logger.info("📈 STEP 1: FETCHING MARKET DATA")
            ohlcv = await self.fetch_market_data(request.symbol, request.timeframe)
            logger.info(f"✅ Market data fetched successfully - Shape: {ohlcv.shape}")
            
            # Step 2: Calculate technical indicators
            logger.info("🔧 STEP 2: CALCULATING TECHNICAL INDICATORS")
            indicators = self.calculate_technical_indicators(ohlcv)
            logger.info("✅ Technical indicators calculated successfully")
            
            # Log current market values
            current_close = indicators['close'].iloc[-1]
            current_high = indicators['high'].iloc[-1]
            current_low = indicators['low'].iloc[-1]
            current_rsi = indicators['rsi'].iloc[-1] if not indicators['rsi'].empty else 50
            current_macd = indicators['macd'].iloc[-1] if not indicators['macd'].empty else 0
            current_macd_signal = indicators['macd_signal'].iloc[-1] if not indicators['macd_signal'].empty else 0
            current_macd_hist = indicators['macd_hist'].iloc[-1] if not indicators['macd_hist'].empty else 0
            current_atr = indicators['atr'].iloc[-1] if not indicators['atr'].empty else 0
            
            logger.info("📊 CURRENT MARKET VALUES:")
            logger.info(f"   • Close Price: ${current_close:.2f}")
            logger.info(f"   • High: ${current_high:.2f}")
            logger.info(f"   • Low: ${current_low:.2f}")
            logger.info(f"   • RSI: {current_rsi:.2f}")
            logger.info(f"   • MACD: {current_macd:.4f}")
            logger.info(f"   • MACD Signal: {current_macd_signal:.4f}")
            logger.info(f"   • MACD Histogram: {current_macd_hist:.4f}")
            logger.info(f"   • ATR: {current_atr:.4f}")
            
            # Step 3: Analyze sentiment
            logger.info("💭 STEP 3: ANALYZING SENTIMENT")
            sentiment_score = await self.analyze_sentiment(request.symbol)
            logger.info(f"✅ Sentiment analysis completed - Score: {sentiment_score:.4f}")
            
            # Step 4: Calculate technical score
            logger.info("📊 STEP 4: CALCULATING TECHNICAL SCORE")
            tech_score = self.calculate_technical_score(indicators)
            logger.info(f"✅ Technical score calculated - Score: {tech_score:.4f}")
            
            # Step 5: Get configuration and apply custom parameters
            logger.info("⚙️ STEP 5: LOADING CONFIGURATION AND APPLYING PARAMETERS")
            config = load_config_from_env()
            
            # Log default config values
            logger.info("🔧 DEFAULT CONFIG VALUES:")
            logger.info(f"   • Default Buy Threshold: {config.thresholds.buy_threshold}")
            logger.info(f"   • Default Sell Threshold: {config.thresholds.sell_threshold}")
            logger.info(f"   • Default Technical Weight: {config.thresholds.technical_weight}")
            logger.info(f"   • Default Sentiment Weight: {config.thresholds.sentiment_weight}")
            
            # Apply custom parameters or use defaults
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
            
            logger.info("🎯 APPLIED PARAMETERS:")
            logger.info(f"   • Buy Threshold: {buy_threshold}")
            logger.info(f"   • Sell Threshold: {sell_threshold}")
            logger.info(f"   • Technical Weight: {tech_weight}")
            logger.info(f"   • Sentiment Weight: {sentiment_weight}")
            
            logger.info("🧮 SCORE CALCULATIONS:")
            logger.info(f"   • Technical Score: {tech_score:.4f}")
            logger.info(f"   • Sentiment Score: {sentiment_score:.4f}")
            logger.info(f"   • Technical Contribution: {tech_weight} × {tech_score:.4f} = {tech_weight * tech_score:.4f}")
            logger.info(f"   • Sentiment Contribution: {sentiment_weight} × {sentiment_score:.4f} = {sentiment_weight * sentiment_score:.4f}")
            logger.info(f"   • Fused Score: {fused_score:.4f}")
            
            # Step 6: Determine signal type
            logger.info("🎯 STEP 6: DETERMINING SIGNAL TYPE")
            signal_type = self.determine_signal_type(fused_score, buy_threshold, sell_threshold)
            logger.info(f"✅ Signal type determined: {signal_type}")
            logger.info(f"   • Fused Score: {fused_score:.4f}")
            logger.info(f"   • Buy Threshold: {buy_threshold}")
            logger.info(f"   • Sell Threshold: {sell_threshold}")
            logger.info(f"   • Decision Logic: {'BUY' if fused_score >= buy_threshold else 'SELL' if fused_score <= sell_threshold else 'HOLD'}")
            
            # Step 7: Calculate risk management
            logger.info("🛡️ STEP 7: CALCULATING RISK MANAGEMENT")
            stop_loss, take_profit = self.calculate_risk_management(indicators, signal_type)
            logger.info(f"✅ Risk management calculated:")
            logger.info(f"   • Stop Loss: ${stop_loss:.2f}" if stop_loss else "   • Stop Loss: N/A")
            logger.info(f"   • Take Profit: ${take_profit:.2f}" if take_profit else "   • Take Profit: N/A")
            if stop_loss and take_profit:
                risk_reward_ratio = abs(take_profit - current_close) / abs(current_close - stop_loss)
                logger.info(f"   • Risk/Reward Ratio: 1:{risk_reward_ratio:.2f}")
            
            # Step 8: Create signal object
            logger.info("📝 STEP 8: CREATING SIGNAL OBJECT")
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
            logger.info("✅ Signal object created successfully")
            
            # Step 9: Store signal
            logger.info("💾 STEP 9: STORING SIGNAL")
            storage_success = await self.store_signal(signal, username)
            logger.info(f"✅ Signal storage: {'SUCCESS' if storage_success else 'FAILED'}")
            
            # Step 10: Send Telegram notification for new signal
            logger.info("📱 STEP 10: SENDING TELEGRAM NOTIFICATION")
            await self.send_telegram_notification(signal, username)
            logger.info("✅ Telegram notification sent")
            
            logger.info("=" * 80)
            logger.info(f"🎉 SIGNAL GENERATION COMPLETED SUCCESSFULLY")
            logger.info(f"📊 Symbol: {request.symbol}")
            logger.info(f"🎯 Signal Type: {signal_type}")
            logger.info(f"📈 Confidence: {signal.confidence:.4f}")
            logger.info(f"🧮 Fused Score: {fused_score:.4f}")
            logger.info(f"👤 Username: {username}")
            logger.info("=" * 80)
            
            return signal
            
        except Exception as e:
            logger.error("=" * 80)
            logger.error(f"❌ SIGNAL GENERATION FAILED")
            logger.error(f"📊 Symbol: {request.symbol}")
            logger.error(f"👤 Username: {username}")
            logger.error(f"🚨 Error: {str(e)}")
            logger.error("=" * 80)
            raise HTTPException(status_code=500, detail=str(e))


# Global signal generator instance
signal_generator = SignalGenerator()


async def generate_trading_signal(request: SignalRequest, username: str) -> TradingSignal:
    """Main function to generate a trading signal"""
    return await signal_generator.generate_signal(request, username)
