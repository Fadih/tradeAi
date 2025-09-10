#!/usr/bin/env python3
"""
Enhanced Signal Generation Module - Phase 1
Advanced trading signal generation with comprehensive technical analysis, sentiment analysis, and risk management
Based on ChatGPT recommendations for professional trading systems
"""

import logging
import random
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
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


class EnhancedTradingSignal(BaseModel):
    """Enhanced trading signal model with Phase 1 improvements"""
    symbol: str
    timeframe: str
    timestamp: str
    signal_type: str  # BUY, SELL, HOLD
    confidence: float
    technical_score: float
    sentiment_score: float
    fused_score: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reasoning: str
    applied_buy_threshold: float
    applied_sell_threshold: float
    applied_tech_weight: float
    applied_sentiment_weight: float
    
    # Enhanced features
    bollinger_position: Optional[float] = None  # %B position in Bollinger Bands
    vwap_distance: Optional[float] = None  # Distance from VWAP in basis points
    volume_profile: Optional[Dict[str, float]] = None  # Volume indicators
    regime_detection: Optional[str] = None  # Market regime (trending/ranging)
    volatility_state: Optional[str] = None  # High/Medium/Low volatility


class EnhancedSignalGenerator:
    """Enhanced signal generator with Phase 1 improvements"""
    
    def __init__(self):
        self.redis_client = None  # Will be initialized when needed
        self.sentiment_analyzer = None  # Will be initialized when needed
        logger.info("Enhanced Signal Generator initialized")
    
    # ==================== PHASE 1: ENHANCED TECHNICAL INDICATORS ====================
    
    def compute_bollinger_bands(self, close: pd.Series, period: int = 20, std_dev: float = 2.0) -> Dict[str, pd.Series]:
        """
        Calculate Bollinger Bands with width, %B, and squeeze detection
        
        Args:
            close: Price series
            period: Moving average period (default 20)
            std_dev: Standard deviation multiplier (default 2.0)
            
        Returns:
            Dict with 'upper', 'middle', 'lower', 'width', 'percent_b', 'squeeze'
        """
        logger.info(f"🔧 Computing Bollinger Bands: period={period}, std_dev={std_dev}")
        
        # Calculate middle band (SMA)
        middle = close.rolling(window=period).mean()
        
        # Calculate standard deviation
        std = close.rolling(window=period).std()
        
        # Calculate upper and lower bands
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        # Calculate band width (normalized)
        width = (upper - lower) / middle
        
        # Calculate %B (position within bands)
        percent_b = (close - lower) / (upper - lower)
        
        # Detect squeeze (low volatility periods)
        # Squeeze occurs when band width is in lowest 20th percentile
        width_percentile = width.rolling(window=period * 2).rank(pct=True)
        squeeze = width_percentile < 0.2
        
        logger.info(f"✅ Bollinger Bands calculated - Current %B: {percent_b.iloc[-1]:.3f}, Width: {width.iloc[-1]:.3f}")
        
        return {
            'upper': upper,
            'middle': middle,
            'lower': lower,
            'width': width,
            'percent_b': percent_b,
            'squeeze': squeeze
        }
    
    def compute_vwap(self, ohlcv: pd.DataFrame, anchor_time: Optional[str] = None) -> pd.Series:
        """
        Calculate Volume Weighted Average Price (VWAP)
        
        Args:
            ohlcv: DataFrame with OHLCV data
            anchor_time: Anchor time for VWAP calculation (e.g., '00:00' for daily)
            
        Returns:
            VWAP series
        """
        logger.info(f"🔧 Computing VWAP with anchor: {anchor_time}")
        
        # Calculate typical price (HLC/3)
        typical_price = (ohlcv['high'] + ohlcv['low'] + ohlcv['close']) / 3
        
        # Calculate volume-weighted price
        vwap = (typical_price * ohlcv['volume']).cumsum() / ohlcv['volume'].cumsum()
        
        # If anchor time is specified, reset VWAP at that time
        if anchor_time:
            # This would require more complex logic for time-based anchoring
            # For now, we'll use session-based VWAP
            pass
        
        logger.info(f"✅ VWAP calculated - Current VWAP: {vwap.iloc[-1]:.2f}")
        
        return vwap
    
    def compute_volume_indicators(self, ohlcv: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Calculate volume-based indicators: OBV, MFI, Accumulation/Distribution
        
        Args:
            ohlcv: DataFrame with OHLCV data
            
        Returns:
            Dict with volume indicators
        """
        logger.info("🔧 Computing volume indicators")
        
        close = ohlcv['close']
        volume = ohlcv['volume']
        high = ohlcv['high']
        low = ohlcv['low']
        
        # On-Balance Volume (OBV)
        price_change = close.diff()
        obv = (volume * price_change.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))).cumsum()
        
        # Money Flow Index (MFI)
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
        
        positive_flow_sum = positive_flow.rolling(window=14).sum()
        negative_flow_sum = negative_flow.rolling(window=14).sum()
        
        mfi = 100 - (100 / (1 + (positive_flow_sum / negative_flow_sum)))
        
        # Accumulation/Distribution Line
        clv = ((close - low) - (high - close)) / (high - low)
        clv = clv.fillna(0)  # Handle division by zero
        ad_line = (clv * volume).cumsum()
        
        logger.info(f"✅ Volume indicators calculated - OBV: {obv.iloc[-1]:.0f}, MFI: {mfi.iloc[-1]:.1f}")
        
        return {
            'obv': obv,
            'mfi': mfi,
            'ad_line': ad_line
        }
    
    def compute_enhanced_moving_averages(self, close: pd.Series) -> Dict[str, pd.Series]:
        """
        Calculate enhanced moving averages with multiple periods and crossovers
        
        Args:
            close: Price series
            
        Returns:
            Dict with various moving averages and their slopes
        """
        logger.info("🔧 Computing enhanced moving averages")
        
        periods = [5, 9, 21, 50, 200]
        ma_dict = {}
        
        # Calculate EMAs and SMAs
        for period in periods:
            ma_dict[f'ema_{period}'] = compute_ema(close, period)
            ma_dict[f'sma_{period}'] = close.rolling(window=period).mean()
            
            # Calculate slopes (rate of change)
            ma_dict[f'ema_{period}_slope'] = ma_dict[f'ema_{period}'].diff()
            ma_dict[f'sma_{period}_slope'] = ma_dict[f'sma_{period}'].diff()
        
        # Calculate crossover signals
        ma_dict['ema_5_21_cross'] = (ma_dict['ema_5'] > ma_dict['ema_21']).astype(int) * 2 - 1
        ma_dict['ema_9_21_cross'] = (ma_dict['ema_9'] > ma_dict['ema_21']).astype(int) * 2 - 1
        ma_dict['ema_21_50_cross'] = (ma_dict['ema_21'] > ma_dict['ema_50']).astype(int) * 2 - 1
        
        logger.info("✅ Enhanced moving averages calculated")
        
        return ma_dict
    
    def compute_keltner_channels(self, ohlcv: pd.DataFrame, ema_period: int = 20, atr_multiplier: float = 2.0) -> Dict[str, pd.Series]:
        """
        Calculate Keltner Channels (EMA ± m·ATR)
        
        Args:
            ohlcv: DataFrame with OHLCV data
            ema_period: EMA period (default 20)
            atr_multiplier: ATR multiplier (default 2.0)
            
        Returns:
            Dict with upper, middle, lower channels
        """
        logger.info(f"🔧 Computing Keltner Channels: EMA={ema_period}, ATR_mult={atr_multiplier}")
        
        close = ohlcv['close']
        
        # Calculate middle line (EMA)
        middle = compute_ema(close, ema_period)
        
        # Calculate ATR
        atr = compute_atr(ohlcv['high'], ohlcv['low'], ohlcv['close'], period=14)
        
        # Calculate upper and lower channels
        upper = middle + (atr * atr_multiplier)
        lower = middle - (atr * atr_multiplier)
        
        logger.info(f"✅ Keltner Channels calculated - Current position: {((close.iloc[-1] - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1])):.3f}")
        
        return {
            'upper': upper,
            'middle': middle,
            'lower': lower,
            'atr': atr
        }
    
    def detect_market_regime(self, indicators: Dict[str, Any]) -> Tuple[str, str]:
        """
        Detect market regime (trending/ranging) and volatility state
        
        Args:
            indicators: Dictionary of calculated indicators
            
        Returns:
            Tuple of (regime, volatility_state)
        """
        logger.info("🔧 Detecting market regime")
        
        try:
            # Get current values
            current_close = indicators['close'].iloc[-1]
            current_bb_width = indicators['bollinger']['width'].iloc[-1]
            current_atr = indicators['keltner']['atr'].iloc[-1]
            
            # Calculate EMA slopes for trend detection
            ema_21_slope = indicators['moving_averages']['ema_21_slope'].iloc[-1]
            ema_50_slope = indicators['moving_averages']['ema_50_slope'].iloc[-1]
            
            # Regime detection
            if abs(ema_21_slope) > 0.1 and abs(ema_50_slope) > 0.05:
                regime = "trending"
            else:
                regime = "ranging"
            
            # Volatility state detection
            bb_width_median = indicators['bollinger']['width'].rolling(window=50).median().iloc[-1]
            if current_bb_width > bb_width_median * 1.5:
                volatility_state = "high"
            elif current_bb_width < bb_width_median * 0.7:
                volatility_state = "low"
            else:
                volatility_state = "medium"
            
            logger.info(f"✅ Market regime detected - Regime: {regime}, Volatility: {volatility_state}")
            
            return regime, volatility_state
            
        except Exception as e:
            logger.error(f"❌ Error in detect_market_regime: {str(e)}")
            logger.error(f"Available indicators keys: {list(indicators.keys())}")
            if 'bollinger' in indicators:
                logger.error(f"Bollinger keys: {list(indicators['bollinger'].keys())}")
            if 'keltner' in indicators:
                logger.error(f"Keltner keys: {list(indicators['keltner'].keys())}")
            if 'moving_averages' in indicators:
                logger.error(f"Moving averages keys: {list(indicators['moving_averages'].keys())}")
            # Return default values
            return "trending", "medium"
    
    # ==================== ENHANCED TECHNICAL SCORE CALCULATION ====================
    
    def calculate_enhanced_technical_score(self, indicators: Dict[str, Any], regime: str) -> float:
        """
        Calculate enhanced technical score using multiple indicators and regime awareness
        
        Args:
            indicators: Dictionary of calculated indicators
            regime: Market regime (trending/ranging)
            
        Returns:
            Enhanced technical score (-1 to 1)
        """
        logger.info(f"🔧 Calculating enhanced technical score for {regime} regime")
        
        scores = []
        weights = []
        
        # Get current values
        current_close = indicators['close'].iloc[-1]
        current_rsi = indicators['rsi'].iloc[-1] if not indicators['rsi'].empty else 50
        current_bb_percent_b = indicators['bollinger']['percent_b'].iloc[-1]
        current_vwap = indicators['vwap'].iloc[-1]
        current_mfi = indicators['volume']['mfi'].iloc[-1]
        current_ema_21 = indicators['moving_averages']['ema_21'].iloc[-1]
        current_ema_50 = indicators['moving_averages']['ema_50'].iloc[-1]
        
        # 1. RSI Score (30% weight)
        rsi_score = (current_rsi - 50) / 50
        scores.append(rsi_score)
        weights.append(0.3)
        
        # 2. Bollinger Bands Score (20% weight)
        if regime == "trending":
            # In trending markets, use momentum-based scoring
            bb_score = (current_bb_percent_b - 0.5) * 2  # -1 to 1
        else:
            # In ranging markets, use mean reversion
            if current_bb_percent_b > 0.8:
                bb_score = -0.5  # Overbought, expect reversal
            elif current_bb_percent_b < 0.2:
                bb_score = 0.5   # Oversold, expect reversal
            else:
                bb_score = 0
        scores.append(bb_score)
        weights.append(0.2)
        
        # 3. VWAP Distance Score (15% weight)
        vwap_distance_bps = ((current_close - current_vwap) / current_vwap) * 10000
        vwap_score = max(min(vwap_distance_bps / 100, 1), -1)  # Normalize to -1 to 1
        scores.append(vwap_score)
        weights.append(0.15)
        
        # 4. Volume Profile Score (15% weight)
        volume_score = (current_mfi - 50) / 50
        scores.append(volume_score)
        weights.append(0.15)
        
        # 5. Moving Average Trend Score (20% weight)
        ma_trend_score = (current_ema_21 - current_ema_50) / current_ema_50
        ma_trend_score = max(min(ma_trend_score * 100, 1), -1)  # Normalize
        scores.append(ma_trend_score)
        weights.append(0.2)
        
        # Calculate weighted technical score
        tech_score = sum(score * weight for score, weight in zip(scores, weights))
        
        logger.info(f"🧮 ENHANCED TECHNICAL SCORE BREAKDOWN:")
        logger.info(f"   • RSI Score: {rsi_score:.4f} (30%)")
        logger.info(f"   • Bollinger Score: {bb_score:.4f} (20%)")
        logger.info(f"   • VWAP Score: {vwap_score:.4f} (15%)")
        logger.info(f"   • Volume Score: {volume_score:.4f} (15%)")
        logger.info(f"   • MA Trend Score: {ma_trend_score:.4f} (20%)")
        logger.info(f"   • Final Technical Score: {tech_score:.4f}")
        
        return tech_score
    
    # ==================== MAIN SIGNAL GENERATION ====================
    
    async def generate_enhanced_signal(self, request: SignalRequest, username: str) -> EnhancedTradingSignal:
        """
        Generate enhanced trading signal with Phase 1 improvements
        
        Args:
            request: Signal generation request
            username: Username for signal storage
            
        Returns:
            Enhanced trading signal
        """
        try:
            logger.info("=" * 80)
            logger.info(f"🚀 STARTING ENHANCED SIGNAL GENERATION (PHASE 1)")
            logger.info(f"📊 Symbol: {request.symbol}")
            logger.info(f"⏰ Timeframe: {request.timeframe}")
            logger.info(f"👤 Username: {username}")
            logger.info(f"🕐 Timestamp: {get_israel_time().isoformat()}")
            logger.info("=" * 80)
            
            # Step 1: Fetch market data
            logger.info("📈 STEP 1: FETCHING MARKET DATA")
            ohlcv = self.fetch_market_data(request.symbol, request.timeframe)
            logger.info(f"✅ Market data fetched successfully - Shape: {ohlcv.shape}")
            
            # Step 2: Calculate enhanced technical indicators
            logger.info("🔧 STEP 2: CALCULATING ENHANCED TECHNICAL INDICATORS")
            indicators = self.calculate_enhanced_technical_indicators(ohlcv)
            logger.info("✅ Enhanced technical indicators calculated successfully")
            
            # Step 3: Detect market regime
            logger.info("🎯 STEP 3: DETECTING MARKET REGIME")
            regime, volatility_state = self.detect_market_regime(indicators)
            logger.info(f"✅ Market regime detected - Regime: {regime}, Volatility: {volatility_state}")
            
            # Step 4: Analyze sentiment
            logger.info("💭 STEP 4: ANALYZING SENTIMENT")
            sentiment_score = await self.analyze_sentiment(request.symbol)
            logger.info(f"✅ Sentiment analysis completed - Score: {sentiment_score:.4f}")
            
            # Step 5: Calculate enhanced technical score
            logger.info("📊 STEP 5: CALCULATING ENHANCED TECHNICAL SCORE")
            tech_score = self.calculate_enhanced_technical_score(indicators, regime)
            logger.info(f"✅ Enhanced technical score calculated - Score: {tech_score:.4f}")
            
            # Step 6: Get configuration and apply parameters
            logger.info("⚙️ STEP 6: LOADING CONFIGURATION AND APPLYING PARAMETERS")
            config = load_config_from_env()
            
            # Apply custom parameters or use defaults
            tech_weight = request.technical_weight if request.technical_weight is not None else config.thresholds.technical_weight
            sentiment_weight = request.sentiment_weight if request.sentiment_weight is not None else config.thresholds.sentiment_weight
            
            # Calculate fused score
            fused_score = tech_weight * tech_score + sentiment_weight * sentiment_score
            
            # Define thresholds
            buy_threshold = request.buy_threshold if request.buy_threshold is not None else config.thresholds.buy_threshold
            if request.sell_threshold is not None:
                sell_threshold = request.sell_threshold
            else:
                sell_threshold = -buy_threshold
            
            logger.info("🧮 ENHANCED SCORE CALCULATIONS:")
            logger.info(f"   • Technical Score: {tech_score:.4f}")
            logger.info(f"   • Sentiment Score: {sentiment_score:.4f}")
            logger.info(f"   • Technical Contribution: {tech_weight} × {tech_score:.4f} = {tech_weight * tech_score:.4f}")
            logger.info(f"   • Sentiment Contribution: {sentiment_weight} × {sentiment_score:.4f} = {sentiment_weight * sentiment_score:.4f}")
            logger.info(f"   • Fused Score: {fused_score:.4f}")
            
            # Step 7: Determine signal type
            logger.info("🎯 STEP 7: DETERMINING SIGNAL TYPE")
            signal_type = self.determine_signal_type(fused_score, buy_threshold, sell_threshold)
            logger.info(f"✅ Signal type determined: {signal_type}")
            
            # Step 8: Calculate risk management
            logger.info("🛡️ STEP 8: CALCULATING RISK MANAGEMENT")
            stop_loss, take_profit = self.calculate_enhanced_risk_management(indicators, signal_type, regime)
            logger.info(f"✅ Enhanced risk management calculated")
            
            # Step 9: Create enhanced signal object
            logger.info("📝 STEP 9: CREATING ENHANCED SIGNAL OBJECT")
            signal = EnhancedTradingSignal(
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
                reasoning=f"Enhanced: Technical: {tech_score:.2f}, Sentiment: {sentiment_score:.2f}, Regime: {regime}, Volatility: {volatility_state}",
                applied_buy_threshold=buy_threshold,
                applied_sell_threshold=sell_threshold,
                applied_tech_weight=tech_weight,
                applied_sentiment_weight=sentiment_weight,
                bollinger_position=indicators['bollinger']['percent_b'].iloc[-1],
                vwap_distance=((indicators['close'].iloc[-1] - indicators['vwap'].iloc[-1]) / indicators['vwap'].iloc[-1]) * 10000,
                volume_profile={
                    'obv': indicators['volume']['obv'].iloc[-1],
                    'mfi': indicators['volume']['mfi'].iloc[-1],
                    'ad_line': indicators['volume']['ad_line'].iloc[-1]
                },
                regime_detection=regime,
                volatility_state=volatility_state
            )
            logger.info("✅ Enhanced signal object created successfully")
            
            # Step 10: Store signal
            logger.info("💾 STEP 10: STORING ENHANCED SIGNAL")
            storage_success = await self.store_signal(signal, username)
            logger.info(f"✅ Enhanced signal storage: {'SUCCESS' if storage_success else 'FAILED'}")
            
            # Step 11: Send Telegram notification
            logger.info("📱 STEP 11: SENDING TELEGRAM NOTIFICATION")
            await self.send_telegram_notification(signal, username)
            logger.info("✅ Telegram notification sent")
            
            logger.info("=" * 80)
            logger.info(f"🎉 ENHANCED SIGNAL GENERATION COMPLETED SUCCESSFULLY")
            logger.info(f"📊 Symbol: {request.symbol}")
            logger.info(f"🎯 Signal Type: {signal_type}")
            logger.info(f"📈 Confidence: {signal.confidence:.4f}")
            logger.info(f"🧮 Fused Score: {fused_score:.4f}")
            logger.info(f"🎭 Regime: {regime}")
            logger.info(f"📊 Volatility: {volatility_state}")
            logger.info(f"👤 Username: {username}")
            logger.info("=" * 80)
            
            return signal
            
        except Exception as e:
            logger.error("=" * 80)
            logger.error(f"❌ ENHANCED SIGNAL GENERATION FAILED")
            logger.error(f"📊 Symbol: {request.symbol}")
            logger.error(f"👤 Username: {username}")
            logger.error(f"🚨 Error: {str(e)}")
            logger.error("=" * 80)
            raise HTTPException(status_code=500, detail=str(e))
    
    def calculate_enhanced_technical_indicators(self, ohlcv: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate all enhanced technical indicators
        
        Args:
            ohlcv: OHLCV data
            
        Returns:
            Dictionary of calculated indicators
        """
        logger.info("🔧 CALCULATING ENHANCED TECHNICAL INDICATORS")
        
        indicators = {
            'close': ohlcv['close'],
            'high': ohlcv['high'],
            'low': ohlcv['low'],
            'volume': ohlcv['volume']
        }
        
        # Enhanced RSI (using existing enhanced function)
        rsi_data = compute_rsi_enhanced(ohlcv['close'], period=7)
        indicators.update(rsi_data)
        
        # Bollinger Bands
        indicators['bollinger'] = self.compute_bollinger_bands(ohlcv['close'])
        
        # VWAP
        indicators['vwap'] = self.compute_vwap(ohlcv)
        
        # Volume indicators
        indicators['volume'] = self.compute_volume_indicators(ohlcv)
        
        # Enhanced moving averages
        indicators['moving_averages'] = self.compute_enhanced_moving_averages(ohlcv['close'])
        
        # Keltner Channels
        indicators['keltner'] = self.compute_keltner_channels(ohlcv)
        
        # MACD (using existing function)
        macd_data = compute_macd(ohlcv['close'])
        indicators.update(macd_data)
        
        # ATR (using existing function)
        indicators['atr'] = compute_atr(ohlcv['high'], ohlcv['low'], ohlcv['close'])
        
        logger.info("✅ All enhanced technical indicators calculated")
        
        return indicators
    
    def calculate_enhanced_risk_management(self, indicators: Dict[str, Any], signal_type: str, regime: str) -> Tuple[Optional[float], Optional[float]]:
        """
        Calculate enhanced risk management with regime awareness
        
        Args:
            indicators: Dictionary of calculated indicators
            signal_type: Signal type (BUY/SELL/HOLD)
            regime: Market regime (trending/ranging)
            
        Returns:
            Tuple of (stop_loss, take_profit)
        """
        logger.info(f"🛡️ CALCULATING ENHANCED RISK MANAGEMENT for {signal_type} in {regime} regime")
        
        if signal_type == "HOLD":
            return None, None
        
        current_price = indicators['close'].iloc[-1]
        current_atr = indicators['atr'].iloc[-1]
        
        # Regime-based risk management
        if regime == "trending":
            # In trending markets, use wider stops and targets
            stop_multiplier = 2.5
            target_multiplier = 3.0
        else:
            # In ranging markets, use tighter stops and targets
            stop_multiplier = 1.5
            target_multiplier = 2.0
        
        if signal_type == "BUY":
            stop_loss = current_price - (current_atr * stop_multiplier)
            take_profit = current_price + (current_atr * target_multiplier)
        else:  # SELL
            stop_loss = current_price + (current_atr * stop_multiplier)
            take_profit = current_price - (current_atr * target_multiplier)
        
        logger.info(f"✅ Enhanced risk management calculated:")
        logger.info(f"   • Stop Loss: ${stop_loss:.2f}" if stop_loss else "   • Stop Loss: N/A")
        logger.info(f"   • Take Profit: ${take_profit:.2f}" if take_profit else "   • Take Profit: N/A")
        if stop_loss and take_profit:
            risk_reward_ratio = abs(take_profit - current_price) / abs(current_price - stop_loss)
            logger.info(f"   • Risk/Reward Ratio: 1:{risk_reward_ratio:.2f}")
        
        return stop_loss, take_profit
    
    # ==================== HELPER METHODS (Reuse from original) ====================
    
    def fetch_market_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Fetch market data for the given symbol and timeframe"""
        try:
            # Determine if it's a crypto or stock symbol
            if "/" in symbol and any(crypto in symbol.upper() for crypto in ["BTC", "ETH", "USDT", "USDC"]):
                # Crypto symbol
                ohlcv = fetch_ohlcv(symbol, timeframe, limit=200)
            else:
                # Stock symbol
                ohlcv = fetch_alpaca_ohlcv(symbol, timeframe, limit=200)
            
            if ohlcv.empty:
                raise HTTPException(status_code=400, detail=f"No market data available for {symbol}")
            
            return ohlcv
            
        except Exception as e:
            logger.error(f"Failed to fetch market data for {symbol}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to fetch market data for {symbol}: {str(e)}")
    
    async def analyze_sentiment(self, symbol: str) -> float:
        """Analyze sentiment for the given symbol"""
        try:
            # Get symbol-specific sentiment
            base_symbol = symbol.split('/')[0].lower()
            
            # Fetch headlines and posts
            headlines = await fetch_headlines([])  # Pass empty feeds list
            crypto_posts = await fetch_crypto_reddit_posts()
            stock_posts = await fetch_stock_reddit_posts()
            
            # Combine all texts
            all_texts = headlines + crypto_posts + stock_posts
            
            # Filter for symbol-specific content
            symbol_texts = [text for text in all_texts if base_symbol in text.lower()]
            
            if not symbol_texts:
                # Fallback to general sentiment
                symbol_texts = all_texts[:50]  # Use first 50 texts
            
            # Initialize sentiment analyzer if needed
            if self.sentiment_analyzer is None:
                self.sentiment_analyzer = SentimentAnalyzer("ProsusAI/finbert")
            
            # Analyze sentiment
            sentiment_score = self.sentiment_analyzer.analyze_sentiment(symbol_texts)
            
            return sentiment_score
            
        except Exception as e:
            logger.error(f"Failed to analyze sentiment for {symbol}: {str(e)}")
            return 0.0
    
    def determine_signal_type(self, fused_score: float, buy_threshold: float, sell_threshold: float) -> str:
        """Determine signal type based on fused score and thresholds"""
        if fused_score >= buy_threshold:
            return "BUY"
        elif fused_score <= sell_threshold:
            return "SELL"
        else:
            return "HOLD"
    
    async def store_signal(self, signal: EnhancedTradingSignal, username: str) -> bool:
        """Store signal in Redis"""
        try:
            redis_client = await get_redis_client()
            if not redis_client:
                logger.error("Redis client not available")
                return False
                
            signal_key = f"signal:{username}:{signal.symbol}:{signal.timestamp}"
            signal_data = signal.dict()
            
            await redis_client.set(signal_key, signal_data, ex=86400)  # 24 hours
            
            # Also store in user's signal list
            user_signals_key = f"user_signals:{username}"
            await redis_client.lpush(user_signals_key, signal_key)
            await redis_client.ltrim(user_signals_key, 0, 99)  # Keep last 100 signals
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store signal: {str(e)}")
            return False
    
    async def send_telegram_notification(self, signal: EnhancedTradingSignal, username: str):
        """Send Telegram notification for new signal"""
        try:
            config = load_config_from_env()
            
            if not config.telegram.enabled or not config.telegram.bot_token:
                logger.info("Telegram notifications disabled")
                return
            
            # Get user's Telegram connection
            redis_client = await get_redis_client()
            if not redis_client:
                logger.error("Redis client not available for Telegram notification")
                return
                
            telegram_connection = await redis_client.get_telegram_connection(username)
            if not telegram_connection or not telegram_connection.get('chat_id'):
                logger.info(f"No Telegram connection found for user {username}")
                return
            
            chat_id = telegram_connection['chat_id']
            
            # Create notification message
            message = f"""
🚀 *New Enhanced Trading Signal*

📊 *Symbol:* {signal.symbol}
⏰ *Timeframe:* {signal.timeframe}
🎯 *Signal:* {signal.signal_type}
📈 *Confidence:* {signal.confidence:.2%}
🧮 *Fused Score:* {signal.fused_score:.4f}

📊 *Technical Score:* {signal.technical_score:.4f}
💭 *Sentiment Score:* {signal.sentiment_score:.4f}

🎭 *Market Regime:* {signal.regime_detection}
📊 *Volatility:* {signal.volatility_state}

💰 *Risk Management:*
• Stop Loss: ${signal.stop_loss:.2f} (if signal.stop_loss else 'N/A')
• Take Profit: ${signal.take_profit:.2f} (if signal.take_profit else 'N/A')

📈 *Enhanced Features:*
• Bollinger Position: {signal.bollinger_position:.3f} (if signal.bollinger_position else 'N/A')
• VWAP Distance: {signal.vwap_distance:.1f} bps (if signal.vwap_distance else 'N/A')

🕐 *Generated:* {signal.timestamp}
👤 *User:* {username}
            """.strip()
            
            # Send notification
            url = f"{config.telegram.api_url}{config.telegram.bot_token}/sendMessage"
            payload = {
                'chat_id': chat_id,
                'text': message,
                'parse_mode': 'Markdown'
            }
            
            response = requests.post(url, json=payload, timeout=config.telegram.timeout)
            response.raise_for_status()
            
            logger.info(f"Telegram notification sent successfully for {signal.symbol}")
            
        except Exception as e:
            logger.error(f"Failed to send Telegram notification: {str(e)}")


# Create global instance
enhanced_signal_generator = EnhancedSignalGenerator()
