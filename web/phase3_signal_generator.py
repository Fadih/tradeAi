"""
Phase 3 Advanced Signal Generator

This module provides the most advanced signal generation capabilities with:
- Independent technical analysis (no inheritance from other phases)
- Advanced regime detection with ADX and volatility analysis
- Multiple RSI variants and crossovers
- Enhanced sentiment analysis
- Dynamic position sizing and risk management
- Comprehensive market microstructure analysis
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np

from agent.config import load_config_from_env
from agent.indicators import (
    compute_rsi, compute_ema, compute_macd, compute_atr,
    compute_adx, compute_volatility_regime, compute_market_regime,
    compute_advanced_rsi_variants, compute_dynamic_position_sizing,
    compute_volatility_adjusted_stops, compute_bollinger_bands,
    compute_vwap, compute_obv, compute_mfi, compute_vwap_anchored,
    compute_accumulation_distribution, compute_ma_crossovers_and_slopes,
    compute_keltner_channels, compute_multi_timeframe_analysis,
    compute_cross_asset_correlation, compute_btc_dominance,
    compute_market_wide_sentiment
)
from agent.sentiment import SentimentAnalyzer
from agent.data import fetch_ohlcv, fetch_alpaca_ohlcv
from agent.cache.redis_client import get_redis_client
from web.main import get_israel_time

logger = logging.getLogger(__name__)

class Phase3TradingSignal:
    """Phase 3 complete trading signal model with all Phase 1, 2, and 3 features"""
    
    def __init__(
        self,
        symbol: str,
        timeframe: str,
        timestamp: datetime,
        signal_type: str,
        technical_score: float,
        sentiment_score: float,
        fused_score: float,
        confidence: float,
        stop_loss: float,
        take_profit: float,
        # Phase 3 Advanced Features
        regime_detection: Dict[str, Any],
        advanced_rsi: Dict[str, Any],
        position_sizing: Dict[str, Any],
        volatility_adjusted_stops: Dict[str, Any],
        risk_metrics: Dict[str, Any],
        technical_indicators: Dict[str, Any],
        market_microstructure: Dict[str, Any],
        # Phase 1 Enhanced Technical Analysis
        bollinger_bands: Dict[str, Any],
        vwap_analysis: Dict[str, Any],
        volume_indicators: Dict[str, Any],
        moving_averages: Dict[str, Any],
        keltner_channels: Dict[str, Any],
        # Phase 2 Multi-Timeframe Analysis
        multi_timeframe: Dict[str, Any],
        cross_asset_correlation: Dict[str, Any],
        btc_dominance: Dict[str, Any],
        market_wide_sentiment: Dict[str, Any],
        username: str
    ):
        self.symbol = symbol
        self.timeframe = timeframe
        self.timestamp = timestamp
        self.signal_type = signal_type
        self.technical_score = technical_score
        self.sentiment_score = sentiment_score
        self.fused_score = fused_score
        self.confidence = confidence
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        
        # Phase 3 Advanced Features
        self.regime_detection = regime_detection
        self.advanced_rsi = advanced_rsi
        self.position_sizing = position_sizing
        self.volatility_adjusted_stops = volatility_adjusted_stops
        self.risk_metrics = risk_metrics
        self.technical_indicators = technical_indicators
        self.market_microstructure = market_microstructure
        
        # Phase 1 Enhanced Technical Analysis
        self.bollinger_bands = bollinger_bands
        self.vwap_analysis = vwap_analysis
        self.volume_indicators = volume_indicators
        self.moving_averages = moving_averages
        self.keltner_channels = keltner_channels
        
        # Phase 2 Multi-Timeframe Analysis
        self.multi_timeframe = multi_timeframe
        self.cross_asset_correlation = cross_asset_correlation
        self.btc_dominance = btc_dominance
        self.market_wide_sentiment = market_wide_sentiment
        
        self.username = username

class Phase3SignalGenerator:
    """Phase 3 Advanced Signal Generator - Completely Independent"""
    
    def __init__(self):
        self.sentiment_analyzer = None
        logger.info("🚀 Phase 3 Signal Generator initialized - Independent mode")
    
    def fetch_market_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Fetch market data for Phase 3 analysis"""
        logger.info(f"📊 Fetching market data for {symbol} @ {timeframe}")
        
        try:
            # Try crypto first
            if "/" in symbol and any(crypto in symbol.upper() for crypto in ["BTC", "ETH", "ADA", "DOT", "LINK", "UNI", "AAVE", "SOL", "MATIC", "AVAX"]):
                ohlcv = fetch_ohlcv(symbol, timeframe, limit=200)
                logger.info(f"✅ Crypto data fetched: {len(ohlcv)} candles")
                return ohlcv
            else:
                # Try stocks
                ohlcv = fetch_alpaca_ohlcv(symbol, timeframe, limit=200)
                logger.info(f"✅ Stock data fetched: {len(ohlcv)} candles")
                return ohlcv
                
        except Exception as e:
            logger.error(f"❌ Failed to fetch market data: {e}")
            raise
    
    def calculate_phase3_technical_indicators(self, ohlcv: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive technical indicators for Phase 3 with all Phase 1, 2, and 3 features"""
        logger.info("🔧 Calculating Phase 3 complete technical indicators")
        
        try:
            high = ohlcv['high']
            low = ohlcv['low']
            close = ohlcv['close']
            volume = ohlcv['volume']
            
            indicators = {}
            
            # Basic indicators
            indicators['rsi_14'] = compute_rsi(close, period=14)
            indicators['rsi_21'] = compute_rsi(close, period=21)
            
            # Enhanced Moving Averages (Phase 1) - All periods
            indicators['ema_5'] = compute_ema(close, period=5)
            indicators['ema_9'] = compute_ema(close, period=9)
            indicators['ema_12'] = compute_ema(close, period=12)
            indicators['ema_21'] = compute_ema(close, period=21)
            indicators['ema_26'] = compute_ema(close, period=26)
            indicators['ema_50'] = compute_ema(close, period=50)
            indicators['ema_200'] = compute_ema(close, period=200)
            
            # MACD
            macd_data = compute_macd(close)
            indicators['macd'] = macd_data['macd']
            indicators['macd_signal'] = macd_data['signal']
            indicators['macd_histogram'] = macd_data['histogram']
            
            # ATR
            indicators['atr'] = compute_atr(high, low, close, period=14)
            
            # ADX for trend strength
            adx_data = compute_adx(high, low, close, period=14)
            indicators['adx'] = adx_data['adx']
            indicators['di_plus'] = adx_data['di_plus']
            indicators['di_minus'] = adx_data['di_minus']
            
            logger.info("✅ Phase 3 technical indicators calculated")
            return indicators
            
        except Exception as e:
            logger.error(f"❌ Error calculating Phase 3 technical indicators: {e}")
            raise
    
    def analyze_advanced_regime_detection(self, ohlcv: pd.DataFrame) -> Dict[str, Any]:
        """Advanced market regime detection for Phase 3"""
        logger.info("🔍 Analyzing advanced regime detection")
        
        try:
            high = ohlcv['high']
            low = ohlcv['low']
            close = ohlcv['close']
            
            # Get ADX-based regime detection
            adx_data = compute_adx(high, low, close, adx_period=14)
            
            # Get volatility regime
            vol_data = compute_volatility_regime(close, period=20)
            
            # Get comprehensive market regime
            regime_data = compute_market_regime(high, low, close, adx_period=14, vol_period=20)
            
            # Combine all regime information
            combined_regime = {
                'adx': adx_data,
                'volatility': vol_data,
                'market_regime': regime_data,
                'trend_strength': adx_data.get('adx', pd.Series([0])).iloc[-1] if isinstance(adx_data.get('adx'), pd.Series) else 0,
                'volatility_state': vol_data.get('regime', 'medium'),
                'regime_classification': regime_data.get('regime', 'trending')
            }
            
            logger.info(f"✅ Advanced regime detection completed: {combined_regime['regime_classification']}")
            return combined_regime
            
        except Exception as e:
            logger.error(f"❌ Error in advanced regime detection: {e}")
            # Return default regime data
            return {
                'adx': {'adx': 0, 'di_plus': 0, 'di_minus': 0},
                'volatility': {'regime': 'medium', 'volatility': 0.02, 'trend': 'stable'},
                'market_regime': {'regime': 'trending', 'strength': 0.5, 'volatility': 'medium', 'trend': 'up'},
                'trend_strength': 0.5,
                'volatility_state': 'medium',
                'regime_classification': 'trending'
            }
    
    def analyze_advanced_rsi_variants(self, ohlcv: pd.DataFrame) -> Dict[str, Any]:
        """Advanced RSI variants analysis for Phase 3"""
        logger.info("📊 Analyzing advanced RSI variants")
        
        try:
            close = ohlcv['close']
            
            # Get advanced RSI variants
            rsi_data = compute_advanced_rsi_variants(close, periods=[7, 9, 14, 21])
            
            # Calculate RSI crossovers and signals
            rsi_7 = rsi_data.get('rsi_7', pd.Series([50]))
            rsi_14 = rsi_data.get('rsi_14', pd.Series([50]))
            rsi_21 = rsi_data.get('rsi_21', pd.Series([50]))
            
            # RSI alignment analysis
            current_rsi_7 = rsi_7.iloc[-1] if len(rsi_7) > 0 else 50
            current_rsi_14 = rsi_14.iloc[-1] if len(rsi_14) > 0 else 50
            current_rsi_21 = rsi_21.iloc[-1] if len(rsi_21) > 0 else 50
            
            # Determine RSI alignment
            if current_rsi_7 > current_rsi_14 > current_rsi_21:
                rsi_alignment = "bullish"
            elif current_rsi_7 < current_rsi_14 < current_rsi_21:
                rsi_alignment = "bearish"
            else:
                rsi_alignment = "mixed"
            
            # Add alignment to RSI data
            rsi_data['alignment'] = rsi_alignment
            rsi_data['current_values'] = {
                'rsi_7': current_rsi_7,
                'rsi_14': current_rsi_14,
                'rsi_21': current_rsi_21
            }
            
            logger.info(f"✅ Advanced RSI variants analysis completed: {rsi_alignment}")
            return rsi_data
            
        except Exception as e:
            logger.error(f"❌ Error in advanced RSI variants analysis: {e}")
            return {
                'rsi_7': pd.Series([50]),
                'rsi_9': pd.Series([50]),
                'rsi_14': pd.Series([50]),
                'rsi_21': pd.Series([50]),
                'alignment': 'mixed',
                'current_values': {'rsi_7': 50, 'rsi_14': 50, 'rsi_21': 50}
            }
    
    async def analyze_phase3_sentiment(self, symbol: str) -> float:
        """Phase 3 enhanced sentiment analysis"""
        logger.info("💭 Analyzing Phase 3 sentiment")
        
        try:
            # Initialize sentiment analyzer if needed
            if self.sentiment_analyzer is None:
                self.sentiment_analyzer = SentimentAnalyzer("ProsusAI/finbert")
            
            # Get configuration
            config = load_config_from_env()
            
            # Fetch sentiment data
            sentiment_score = await self.sentiment_analyzer.analyze_sentiment(
                symbol, 
                config.sentiment_analysis.rss_feeds,
                config.sentiment_analysis.reddit_subreddits,
                config.sentiment_analysis.rss_max_headlines_per_feed,
                config.sentiment_analysis.reddit_max_posts_per_subreddit
            )
            
            logger.info(f"✅ Phase 3 sentiment analysis completed: {sentiment_score:.4f}")
            return sentiment_score
            
        except Exception as e:
            logger.error(f"❌ Error in Phase 3 sentiment analysis: {e}")
            return 0.0
    
    def calculate_phase3_technical_score(self, indicators: Dict[str, Any], regime_data: Dict[str, Any], rsi_data: Dict[str, Any]) -> float:
        """Calculate Phase 3 technical score"""
        logger.info("🧮 Calculating Phase 3 technical score")
        
        try:
            score = 0.0
            
            # RSI analysis (40% weight)
            rsi_14 = indicators.get('rsi_14', pd.Series([50]))
            current_rsi = rsi_14.iloc[-1] if len(rsi_14) > 0 else 50
            
            if current_rsi < 30:
                rsi_score = 0.8  # Oversold - bullish
            elif current_rsi > 70:
                rsi_score = -0.8  # Overbought - bearish
            elif 40 <= current_rsi <= 60:
                rsi_score = 0.0  # Neutral
            else:
                rsi_score = (50 - current_rsi) / 50  # Linear interpolation
            
            # RSI alignment bonus
            rsi_alignment = rsi_data.get('alignment', 'mixed')
            if rsi_alignment == 'bullish':
                rsi_score += 0.2
            elif rsi_alignment == 'bearish':
                rsi_score -= 0.2
            
            score += rsi_score * 0.4
            
            # MACD analysis (25% weight)
            macd = indicators.get('macd', pd.Series([0]))
            macd_signal = indicators.get('macd_signal', pd.Series([0]))
            
            if len(macd) > 0 and len(macd_signal) > 0:
                current_macd = macd.iloc[-1]
                current_signal = macd_signal.iloc[-1]
                
                if current_macd > current_signal:
                    macd_score = 0.5
                else:
                    macd_score = -0.5
                
                score += macd_score * 0.25
            
            # Bollinger Bands analysis (20% weight)
            bb_percent = indicators.get('bb_percent', pd.Series([0.5]))
            if len(bb_percent) > 0:
                current_bb = bb_percent.iloc[-1]
                
                if current_bb < 0.2:
                    bb_score = 0.6  # Near lower band - bullish
                elif current_bb > 0.8:
                    bb_score = -0.6  # Near upper band - bearish
                else:
                    bb_score = (0.5 - current_bb) * 2  # Linear interpolation
                
                score += bb_score * 0.2
            
            # ADX trend strength (15% weight)
            trend_strength = regime_data.get('trend_strength', 0.5)
            regime_classification = regime_data.get('regime_classification', 'trending')
            
            if regime_classification == 'trending' and trend_strength > 25:
                adx_score = 0.3
            elif regime_classification == 'consolidation':
                adx_score = -0.1
            else:
                adx_score = 0.0
            
            score += adx_score * 0.15
            
            # Normalize score to [-1, 1]
            final_score = max(-1.0, min(1.0, score))
            
            logger.info(f"✅ Phase 3 technical score calculated: {final_score:.4f}")
            return final_score
            
        except Exception as e:
            logger.error(f"❌ Error calculating Phase 3 technical score: {e}")
            return 0.0
    
    def determine_phase3_signal_type(self, technical_score: float, sentiment_score: float, regime_data: Dict[str, Any]) -> str:
        """Determine Phase 3 signal type"""
        logger.info("🎯 Determining Phase 3 signal type")
        
        try:
            # Get configuration
            config = load_config_from_env()
            
            # Calculate fused score
            tech_weight = config.thresholds.technical_weight
            sentiment_weight = config.thresholds.sentiment_weight
            fused_score = tech_weight * technical_score + sentiment_weight * sentiment_score
            
            # Get thresholds
            buy_threshold = config.thresholds.buy_threshold
            sell_threshold = -buy_threshold
            
            # Determine signal based on fused score and regime
            regime_classification = regime_data.get('regime_classification', 'trending')
            
            if fused_score > buy_threshold:
                signal_type = "BUY"
            elif fused_score < sell_threshold:
                signal_type = "SELL"
            elif regime_classification == 'trending' and technical_score > 0.3:
                signal_type = "BUY"
            elif regime_classification == 'trending' and technical_score < -0.3:
                signal_type = "SELL"
            else:
                signal_type = "HOLD"
            
            logger.info(f"✅ Phase 3 signal type determined: {signal_type}")
            return signal_type
            
        except Exception as e:
            logger.error(f"❌ Error determining Phase 3 signal type: {e}")
            return "HOLD"
    
    def calculate_enhanced_risk_management(self, ohlcv: pd.DataFrame, signal_type: str, regime_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate enhanced risk management for Phase 3"""
        logger.info("🛡️ Calculating enhanced risk management")
        
        try:
            high = ohlcv['high']
            low = ohlcv['low']
            close = ohlcv['close']
            
            current_price = close.iloc[-1]
            
            # Calculate ATR for volatility-based stops
            atr = compute_atr(high, low, close, period=14)
            current_atr = atr.iloc[-1] if len(atr) > 0 else current_price * 0.02
            
            # Dynamic position sizing
            volatility = regime_data.get('volatility', {}).get('volatility', 0.02)
            position_data = compute_dynamic_position_sizing(
                close, 
                pd.Series([volatility] * len(close)), 
                account_balance=10000.0, 
                risk_per_trade=0.02
            )
            
            # Volatility-adjusted stops
            stops_data = compute_volatility_adjusted_stops(
                close, high, low, atr, volatility_multiplier=2.0
            )
            
            # Calculate stop loss and take profit based on signal type
            if signal_type == "BUY":
                stop_loss = current_price - (current_atr * 2.0)
                take_profit = current_price + (current_atr * 3.0)
            elif signal_type == "SELL":
                stop_loss = current_price + (current_atr * 2.0)
                take_profit = current_price - (current_atr * 3.0)
            else:  # HOLD
                stop_loss = current_price - (current_atr * 1.5)
                take_profit = current_price + (current_atr * 1.5)
            
            risk_data = {
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'position_sizing': position_data,
                'volatility_adjusted_stops': stops_data,
                'risk_metrics': {
                    'atr': current_atr,
                    'volatility': volatility,
                    'risk_reward_ratio': abs(take_profit - current_price) / abs(stop_loss - current_price) if stop_loss != current_price else 1.0,
                    'max_loss_percent': abs(stop_loss - current_price) / current_price * 100
                }
            }
            
            logger.info(f"✅ Enhanced risk management calculated")
            return risk_data
            
        except Exception as e:
            logger.error(f"❌ Error calculating enhanced risk management: {e}")
            current_price = ohlcv['close'].iloc[-1]
            return {
                'stop_loss': current_price * 0.98,
                'take_profit': current_price * 1.02,
                'position_sizing': {'position_size': 1000, 'risk_amount': 200, 'leverage': 1.0},
                'volatility_adjusted_stops': {'atr_stop': current_price * 0.02, 'percentage_stop': 0.02},
                'risk_metrics': {'atr': current_price * 0.02, 'volatility': 0.02, 'risk_reward_ratio': 1.0, 'max_loss_percent': 2.0}
            }
    
    def analyze_market_microstructure(self, ohlcv: pd.DataFrame) -> Dict[str, Any]:
        """Analyze market microstructure for Phase 3"""
        logger.info("🔬 Analyzing market microstructure")
        
        try:
            high = ohlcv['high']
            low = ohlcv['low']
            close = ohlcv['close']
            volume = ohlcv['volume']
            
            # Calculate price gaps
            price_changes = close.diff()
            gaps = price_changes[abs(price_changes) > close.shift(1) * 0.01]  # Gaps > 1%
            
            # Volume analysis
            avg_volume = volume.rolling(window=20).mean()
            volume_spikes = volume > avg_volume * 2
            
            # Price momentum
            momentum_5 = close.pct_change(5)
            momentum_10 = close.pct_change(10)
            
            microstructure = {
                'gaps': {
                    'count': len(gaps),
                    'largest_gap': gaps.max() if len(gaps) > 0 else 0,
                    'gap_direction': 'up' if len(gaps) > 0 and gaps.iloc[-1] > 0 else 'down'
                },
                'volume': {
                    'avg_volume': avg_volume.iloc[-1] if len(avg_volume) > 0 else 0,
                    'current_volume': volume.iloc[-1] if len(volume) > 0 else 0,
                    'volume_spikes': volume_spikes.sum(),
                    'volume_trend': 'increasing' if volume.iloc[-1] > avg_volume.iloc[-1] else 'decreasing'
                },
                'momentum': {
                    'momentum_5': momentum_5.iloc[-1] if len(momentum_5) > 0 else 0,
                    'momentum_10': momentum_10.iloc[-1] if len(momentum_10) > 0 else 0,
                    'momentum_trend': 'accelerating' if abs(momentum_5.iloc[-1]) > abs(momentum_10.iloc[-1]) else 'decelerating'
                }
            }
            
            logger.info("✅ Market microstructure analysis completed")
            return microstructure
            
        except Exception as e:
            logger.error(f"❌ Error analyzing market microstructure: {e}")
            return {
                'gaps': {'count': 0, 'largest_gap': 0, 'gap_direction': 'none'},
                'volume': {'avg_volume': 0, 'current_volume': 0, 'volume_spikes': 0, 'volume_trend': 'stable'},
                'momentum': {'momentum_5': 0, 'momentum_10': 0, 'momentum_trend': 'stable'}
            }
    
    # ===== PHASE 1 ENHANCED TECHNICAL ANALYSIS METHODS =====
    
    def analyze_bollinger_bands(self, ohlcv: pd.DataFrame) -> Dict[str, Any]:
        """Analyze Bollinger Bands with squeeze detection (Phase 1)"""
        logger.info("📊 Analyzing Bollinger Bands with squeeze detection")
        
        try:
            close = ohlcv['close']
            bb_data = compute_bollinger_bands(close, period=20, std_dev=2.0)
            
            # Get current values
            current_bb = {
                'upper': bb_data['upper'].iloc[-1] if len(bb_data['upper']) > 0 else close.iloc[-1],
                'middle': bb_data['middle'].iloc[-1] if len(bb_data['middle']) > 0 else close.iloc[-1],
                'lower': bb_data['lower'].iloc[-1] if len(bb_data['lower']) > 0 else close.iloc[-1],
                'width': bb_data['width'].iloc[-1] if len(bb_data['width']) > 0 else 0,
                'percent': bb_data['percent'].iloc[-1] if len(bb_data['percent']) > 0 else 0.5,
                'squeeze': bb_data['squeeze'].iloc[-1] if len(bb_data['squeeze']) > 0 else False,
                'squeeze_strength': bb_data['squeeze_strength'].iloc[-1] if len(bb_data['squeeze_strength']) > 0 else 0
            }
            
            logger.info(f"✅ Bollinger Bands analysis completed - Squeeze: {current_bb['squeeze']}")
            return current_bb
            
        except Exception as e:
            logger.error(f"❌ Error analyzing Bollinger Bands: {e}")
            return {'squeeze': False, 'squeeze_strength': 0, 'percent': 0.5}
    
    def analyze_vwap_enhanced(self, ohlcv: pd.DataFrame) -> Dict[str, Any]:
        """Analyze enhanced VWAP with anchoring (Phase 1)"""
        logger.info("📊 Analyzing enhanced VWAP with anchoring")
        
        try:
            # Standard VWAP
            vwap_data = compute_vwap(ohlcv)
            
            # Anchored VWAP
            vwap_anchored_data = compute_vwap_anchored(ohlcv)
            
            vwap_analysis = {
                'vwap': vwap_data['vwap'].iloc[-1] if len(vwap_data['vwap']) > 0 else ohlcv['close'].iloc[-1],
                'deviation': vwap_data['deviation'].iloc[-1] if len(vwap_data['deviation']) > 0 else 0,
                'anchored_vwap': vwap_anchored_data['vwap'].iloc[-1] if len(vwap_anchored_data['vwap']) > 0 else ohlcv['close'].iloc[-1],
                'anchored_deviation': vwap_anchored_data['deviation'].iloc[-1] if len(vwap_anchored_data['deviation']) > 0 else 0,
                'session_vwap': vwap_anchored_data.get('session_vwap', {}).iloc[-1] if vwap_anchored_data.get('session_vwap') is not None else None
            }
            
            logger.info("✅ Enhanced VWAP analysis completed")
            return vwap_analysis
            
        except Exception as e:
            logger.error(f"❌ Error analyzing enhanced VWAP: {e}")
            return {'vwap': ohlcv['close'].iloc[-1], 'deviation': 0}
    
    def analyze_volume_indicators(self, ohlcv: pd.DataFrame) -> Dict[str, Any]:
        """Analyze comprehensive volume indicators (Phase 1)"""
        logger.info("📊 Analyzing comprehensive volume indicators")
        
        try:
            high = ohlcv['high']
            low = ohlcv['low']
            close = ohlcv['close']
            volume = ohlcv['volume']
            
            # OBV
            obv = compute_obv(close, volume)
            
            # MFI
            mfi = compute_mfi(high, low, close, volume, period=14)
            
            # Accumulation/Distribution
            ad_line = compute_accumulation_distribution(high, low, close, volume)
            
            volume_indicators = {
                'obv': obv.iloc[-1] if len(obv) > 0 else 0,
                'obv_trend': 'up' if obv.iloc[-1] > obv.iloc[-5] else 'down' if len(obv) > 5 else 'neutral',
                'mfi': mfi.iloc[-1] if len(mfi) > 0 else 50,
                'mfi_signal': 'overbought' if mfi.iloc[-1] > 80 else 'oversold' if mfi.iloc[-1] < 20 else 'neutral',
                'ad_line': ad_line.iloc[-1] if len(ad_line) > 0 else 0,
                'ad_trend': 'accumulation' if ad_line.iloc[-1] > ad_line.iloc[-5] else 'distribution' if len(ad_line) > 5 else 'neutral'
            }
            
            logger.info("✅ Volume indicators analysis completed")
            return volume_indicators
            
        except Exception as e:
            logger.error(f"❌ Error analyzing volume indicators: {e}")
            return {'obv': 0, 'mfi': 50, 'ad_line': 0}
    
    def analyze_moving_averages_enhanced(self, ohlcv: pd.DataFrame) -> Dict[str, Any]:
        """Analyze enhanced moving averages with crossovers and slopes (Phase 1)"""
        logger.info("📊 Analyzing enhanced moving averages")
        
        try:
            close = ohlcv['close']
            ma_data = compute_ma_crossovers_and_slopes(close, periods=[5, 9, 12, 21, 26, 50, 200])
            
            # Get current values
            current_mas = {}
            for period in [5, 9, 12, 21, 26, 50, 200]:
                ema_key = f'ema_{period}'
                if ema_key in ma_data['moving_averages']:
                    current_mas[ema_key] = ma_data['moving_averages'][ema_key].iloc[-1]
                    current_mas[f'{ema_key}_slope'] = ma_data['slopes'][f'{ema_key}_slope'].iloc[-1]
            
            moving_averages = {
                'current_values': current_mas,
                'crossovers': {
                    'last_bullish': ma_data['crossovers'].get('last_bullish', False),
                    'last_bearish': ma_data['crossovers'].get('last_bearish', False)
                },
                'slopes': {k: v.iloc[-1] if len(v) > 0 else 0 for k, v in ma_data['slopes'].items()}
            }
            
            logger.info("✅ Enhanced moving averages analysis completed")
            return moving_averages
            
        except Exception as e:
            logger.error(f"❌ Error analyzing enhanced moving averages: {e}")
            return {'current_values': {}, 'crossovers': {'last_bullish': False, 'last_bearish': False}}
    
    def analyze_keltner_channels(self, ohlcv: pd.DataFrame) -> Dict[str, Any]:
        """Analyze Keltner Channels with breakout detection (Phase 1)"""
        logger.info("📊 Analyzing Keltner Channels")
        
        try:
            high = ohlcv['high']
            low = ohlcv['low']
            close = ohlcv['close']
            
            kc_data = compute_keltner_channels(high, low, close, ema_period=20, atr_period=14, atr_multiplier=2.0)
            
            keltner_channels = {
                'upper': kc_data['upper'].iloc[-1] if len(kc_data['upper']) > 0 else close.iloc[-1],
                'middle': kc_data['middle'].iloc[-1] if len(kc_data['middle']) > 0 else close.iloc[-1],
                'lower': kc_data['lower'].iloc[-1] if len(kc_data['lower']) > 0 else close.iloc[-1],
                'channel_position': kc_data['channel_position'].iloc[-1] if len(kc_data['channel_position']) > 0 else 0.5,
                'last_breakout_up': kc_data.get('last_breakout_up', False),
                'last_breakout_down': kc_data.get('last_breakout_down', False),
                'atr': kc_data['atr'].iloc[-1] if len(kc_data['atr']) > 0 else 0
            }
            
            logger.info("✅ Keltner Channels analysis completed")
            return keltner_channels
            
        except Exception as e:
            logger.error(f"❌ Error analyzing Keltner Channels: {e}")
            return {'channel_position': 0.5, 'last_breakout_up': False, 'last_breakout_down': False}
    
    # ===== PHASE 2 MULTI-TIMEFRAME ANALYSIS METHODS =====
    
    def analyze_multi_timeframe(self, symbol: str) -> Dict[str, Any]:
        """Analyze multi-timeframe data (Phase 2)"""
        logger.info("📊 Analyzing multi-timeframe data")
        
        try:
            mtf_data = compute_multi_timeframe_analysis(symbol, timeframes=['5m', '15m', '1h', '4h'])
            
            logger.info(f"✅ Multi-timeframe analysis completed - Overall trend: {mtf_data.get('overall_trend', 'neutral')}")
            return mtf_data
            
        except Exception as e:
            logger.error(f"❌ Error analyzing multi-timeframe: {e}")
            return {'overall_trend': 'neutral', 'trend_consensus': 0.5}
    
    def analyze_cross_asset_correlation(self, symbol: str) -> Dict[str, Any]:
        """Analyze cross-asset correlation (Phase 2)"""
        logger.info("📊 Analyzing cross-asset correlation")
        
        try:
            # Determine symbols to correlate based on input
            if 'BTC' in symbol.upper():
                symbols = ['BTC/USDT', 'ETH/USDT', 'ADA/USDT']
            elif 'ETH' in symbol.upper():
                symbols = ['ETH/USDT', 'BTC/USDT', 'ADA/USDT']
            else:
                symbols = ['BTC/USDT', 'ETH/USDT', symbol]
            
            correlation_data = compute_cross_asset_correlation(symbols, timeframe='1h', period=24)
            
            logger.info("✅ Cross-asset correlation analysis completed")
            return correlation_data
            
        except Exception as e:
            logger.error(f"❌ Error analyzing cross-asset correlation: {e}")
            return {'avg_correlation': 0, 'btc_correlations': {}}
    
    def analyze_btc_dominance(self) -> Dict[str, Any]:
        """Analyze BTC dominance (Phase 2)"""
        logger.info("📊 Analyzing BTC dominance")
        
        try:
            dominance_data = compute_btc_dominance()
            
            logger.info(f"✅ BTC dominance analysis completed - Dominance: {dominance_data.get('btc_dominance', 50):.1f}%")
            return dominance_data
            
        except Exception as e:
            logger.error(f"❌ Error analyzing BTC dominance: {e}")
            return {'btc_dominance': 50, 'dominance_change': 0}
    
    def analyze_market_wide_sentiment(self) -> Dict[str, Any]:
        """Analyze market-wide sentiment (Phase 2)"""
        logger.info("📊 Analyzing market-wide sentiment")
        
        try:
            sentiment_data = compute_market_wide_sentiment()
            
            logger.info(f"✅ Market-wide sentiment analysis completed - Sentiment: {sentiment_data.get('overall_sentiment', 'neutral')}")
            return sentiment_data
            
        except Exception as e:
            logger.error(f"❌ Error analyzing market-wide sentiment: {e}")
            return {'market_sentiment_score': 0, 'overall_sentiment': 'neutral'}
    
    async def store_signal(self, signal: Phase3TradingSignal, username: str):
        """Store Phase 3 signal in Redis"""
        logger.info("💾 Storing Phase 3 signal")
        
        try:
            redis_client = await get_redis_client()
            if redis_client:
                signal_data = {
                    'symbol': signal.symbol,
                    'timeframe': signal.timeframe,
                    'timestamp': signal.timestamp.isoformat(),
                    'signal_type': signal.signal_type,
                    'technical_score': signal.technical_score,
                    'sentiment_score': signal.sentiment_score,
                    'fused_score': signal.fused_score,
                    'confidence': signal.confidence,
                    'stop_loss': signal.stop_loss,
                    'take_profit': signal.take_profit,
                    # Phase 3 Advanced Features
                    'regime_detection': signal.regime_detection,
                    'advanced_rsi': signal.advanced_rsi,
                    'position_sizing': signal.position_sizing,
                    'volatility_adjusted_stops': signal.volatility_adjusted_stops,
                    'risk_metrics': signal.risk_metrics,
                    'technical_indicators': signal.technical_indicators,
                    'market_microstructure': signal.market_microstructure,
                    # Phase 1 Enhanced Technical Analysis
                    'bollinger_bands': signal.bollinger_bands,
                    'vwap_analysis': signal.vwap_analysis,
                    'volume_indicators': signal.volume_indicators,
                    'moving_averages': signal.moving_averages,
                    'keltner_channels': signal.keltner_channels,
                    # Phase 2 Multi-Timeframe Analysis
                    'multi_timeframe': signal.multi_timeframe,
                    'cross_asset_correlation': signal.cross_asset_correlation,
                    'btc_dominance': signal.btc_dominance,
                    'market_wide_sentiment': signal.market_wide_sentiment,
                    'username': username,
                    'phase': 'phase3_complete'
                }
                
                await redis_client.store_signal(signal_data)
                logger.info("✅ Phase 3 signal stored successfully")
            else:
                logger.warning("⚠️ Redis not available - signal not stored")
                
        except Exception as e:
            logger.error(f"❌ Error storing Phase 3 signal: {e}")
    
    async def send_telegram_notification(self, signal: Phase3TradingSignal, username: str):
        """Send Telegram notification for Phase 3 signal"""
        logger.info("📱 Sending Phase 3 Telegram notification")
        
        try:
            # Get configuration
            config = load_config_from_env()
            
            if not config.telegram.enabled or not config.telegram.bot_token:
                logger.info("📱 Telegram notifications disabled")
                return
            
            # Get user's chat ID from Redis
            redis_client = await get_redis_client()
            if not redis_client:
                logger.warning("⚠️ Redis not available - cannot send Telegram notification")
                return
            
            chat_id = await redis_client.get_telegram_connection(username)
            if not chat_id:
                logger.info(f"📱 No Telegram connection found for user {username}")
                return
            
            # Create comprehensive notification message
            message = f"""
🚀 *Complete Phase 3 Trading Signal*

📊 *Symbol:* {signal.symbol}
⏰ *Timeframe:* {signal.timeframe}
🎯 *Signal:* {signal.signal_type}
📈 *Technical Score:* {signal.technical_score:.4f}
💭 *Sentiment Score:* {signal.sentiment_score:.4f}
🧮 *Fused Score:* {signal.fused_score:.4f}
🎲 *Confidence:* {signal.confidence:.2%}

🛡️ *Risk Management:*
• Stop Loss: ${signal.stop_loss:.2f}
• Take Profit: ${signal.take_profit:.2f}
• Risk/Reward: {signal.risk_metrics.get('risk_reward_ratio', 1.0):.2f}

📊 *Phase 1 Features:*
• Bollinger Squeeze: {signal.bollinger_bands.get('squeeze', False)}
• VWAP Deviation: {signal.vwap_analysis.get('deviation', 0):.2f} bps
• Volume Trend: {signal.volume_indicators.get('obv_trend', 'neutral')}
• MA Crossovers: {signal.moving_averages.get('crossovers', {}).get('last_bullish', False)}

📊 *Phase 2 Features:*
• Multi-TF Trend: {signal.multi_timeframe.get('overall_trend', 'neutral')}
• Trend Consensus: {signal.multi_timeframe.get('trend_consensus', 0.5):.2%}
• BTC Dominance: {signal.btc_dominance.get('btc_dominance', 50):.1f}%
• Market Sentiment: {signal.market_wide_sentiment.get('overall_sentiment', 'neutral')}

📊 *Phase 3 Features:*
• Regime: {signal.regime_detection.get('regime_classification', 'unknown')}
• RSI Alignment: {signal.advanced_rsi.get('alignment', 'mixed')}
• Volatility: {signal.regime_detection.get('volatility_state', 'medium')}

🕐 *Generated:* {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S')} (Israel Time)
            """
            
            # Send notification
            import requests
            url = f"https://api.telegram.org/bot{config.telegram.bot_token}/sendMessage"
            data = {
                'chat_id': chat_id,
                'text': message,
                'parse_mode': 'Markdown'
            }
            
            response = requests.post(url, data=data, timeout=10)
            if response.status_code == 200:
                logger.info("✅ Phase 3 Telegram notification sent successfully")
            else:
                logger.error(f"❌ Failed to send Phase 3 Telegram notification: {response.status_code}")
                
        except Exception as e:
            logger.error(f"❌ Error sending Phase 3 Telegram notification: {e}")
    
    async def generate_phase3_signal(self, request, username: str) -> Phase3TradingSignal:
        """
        Generate a Phase 3 advanced trading signal - Completely Independent
        
        Args:
            request: Signal generation request
            username: Username of the user requesting the signal
            
        Returns:
            Phase 3 advanced trading signal
        """
        try:
            logger.info("=" * 80)
            logger.info(f"🚀 STARTING PHASE 3 ADVANCED SIGNAL GENERATION (INDEPENDENT)")
            logger.info(f"📊 Symbol: {request.symbol}")
            logger.info(f"⏰ Timeframe: {request.timeframe}")
            logger.info(f"👤 Username: {username}")
            logger.info(f"🕐 Timestamp: {get_israel_time().isoformat()}")
            logger.info("=" * 80)
            
            # Step 1: Fetch market data
            logger.info("📊 STEP 1: FETCHING MARKET DATA")
            ohlcv = self.fetch_market_data(request.symbol, request.timeframe)
            logger.info(f"✅ Market data fetched: {len(ohlcv)} candles")
            
            # Step 2: Calculate Phase 3 technical indicators
            logger.info("🔧 STEP 2: CALCULATING PHASE 3 TECHNICAL INDICATORS")
            technical_indicators = self.calculate_phase3_technical_indicators(ohlcv)
            logger.info(f"✅ Phase 3 technical indicators calculated")
            
            # Step 3: Advanced regime detection
            logger.info("🔍 STEP 3: ADVANCED REGIME DETECTION")
            regime_data = self.analyze_advanced_regime_detection(ohlcv)
            logger.info(f"✅ Advanced regime detection completed")
            
            # Step 4: Advanced RSI variants analysis
            logger.info("📊 STEP 4: ADVANCED RSI VARIANTS ANALYSIS")
            rsi_data = self.analyze_advanced_rsi_variants(ohlcv)
            logger.info(f"✅ Advanced RSI variants analysis completed")
            
            # Step 5: Phase 3 sentiment analysis
            logger.info("💭 STEP 5: PHASE 3 SENTIMENT ANALYSIS")
            sentiment_score = await self.analyze_phase3_sentiment(request.symbol)
            logger.info(f"✅ Phase 3 sentiment analysis completed: {sentiment_score:.4f}")
            
            # Step 6: Calculate Phase 3 technical score
            logger.info("🧮 STEP 6: PHASE 3 TECHNICAL SCORE CALCULATION")
            phase3_tech_score = self.calculate_phase3_technical_score(
                technical_indicators, regime_data, rsi_data
            )
            logger.info(f"✅ Phase 3 technical score calculated: {phase3_tech_score:.4f}")
            
            # Step 7: Determine signal type
            logger.info("🎯 STEP 7: DETERMINING SIGNAL TYPE")
            signal_type = self.determine_phase3_signal_type(
                phase3_tech_score, sentiment_score, regime_data
            )
            logger.info(f"✅ Signal type determined: {signal_type}")
            
            # Step 8: Enhanced risk management
            logger.info("🛡️ STEP 8: ENHANCED RISK MANAGEMENT")
            risk_data = self.calculate_enhanced_risk_management(ohlcv, signal_type, regime_data)
            stop_loss = risk_data.get('stop_loss')
            take_profit = risk_data.get('take_profit')
            logger.info(f"✅ Enhanced risk management calculated")
            
            # Step 9: Market microstructure analysis
            logger.info("🔬 STEP 9: MARKET MICROSTRUCTURE ANALYSIS")
            microstructure = self.analyze_market_microstructure(ohlcv)
            logger.info(f"✅ Market microstructure analysis completed")
            
            # Step 10: Phase 1 Enhanced Technical Analysis
            logger.info("📊 STEP 10: PHASE 1 ENHANCED TECHNICAL ANALYSIS")
            bollinger_bands = self.analyze_bollinger_bands(ohlcv)
            vwap_analysis = self.analyze_vwap_enhanced(ohlcv)
            volume_indicators = self.analyze_volume_indicators(ohlcv)
            moving_averages = self.analyze_moving_averages_enhanced(ohlcv)
            keltner_channels = self.analyze_keltner_channels(ohlcv)
            logger.info(f"✅ Phase 1 enhanced technical analysis completed")
            
            # Step 11: Phase 2 Multi-Timeframe Analysis
            logger.info("📊 STEP 11: PHASE 2 MULTI-TIMEFRAME ANALYSIS")
            multi_timeframe = self.analyze_multi_timeframe(request.symbol)
            cross_asset_correlation = self.analyze_cross_asset_correlation(request.symbol)
            btc_dominance = self.analyze_btc_dominance()
            market_wide_sentiment = self.analyze_market_wide_sentiment()
            logger.info(f"✅ Phase 2 multi-timeframe analysis completed")
            
            # Step 12: Calculate fused score and confidence
            logger.info("🧮 STEP 12: CALCULATING FUSED SCORE AND CONFIDENCE")
            config = load_config_from_env()
            tech_weight = config.thresholds.technical_weight
            sentiment_weight = config.thresholds.sentiment_weight
            fused_score = tech_weight * phase3_tech_score + sentiment_weight * sentiment_score
            
            # Calculate confidence based on multiple factors
            confidence = 0.5  # Base confidence
            
            # Regime confidence
            if regime_data.get('regime_classification') == 'trending':
                confidence += 0.15
            
            # RSI alignment confidence
            if rsi_data.get('alignment') in ['bullish', 'bearish']:
                confidence += 0.15
            
            # Technical score confidence
            if abs(phase3_tech_score) > 0.5:
                confidence += 0.1
            
            # Multi-timeframe consensus confidence
            trend_consensus = multi_timeframe.get('trend_consensus', 0.5)
            if trend_consensus > 0.7 or trend_consensus < 0.3:
                confidence += 0.1
            
            # Market-wide sentiment confidence
            market_sentiment = market_wide_sentiment.get('market_sentiment_score', 0)
            if abs(market_sentiment) > 0.3:
                confidence += 0.05
            
            # Bollinger squeeze confidence
            if bollinger_bands.get('squeeze', False):
                confidence += 0.05
            
            confidence = min(1.0, confidence)
            logger.info(f"✅ Fused score: {fused_score:.4f}, Confidence: {confidence:.2%}")
            
            # Step 13: Create Complete Phase 3 Signal
            logger.info("🎯 STEP 13: CREATING COMPLETE PHASE 3 SIGNAL")
            phase3_signal = Phase3TradingSignal(
                symbol=request.symbol,
                timeframe=request.timeframe,
                timestamp=get_israel_time(),
                signal_type=signal_type,
                technical_score=phase3_tech_score,
                sentiment_score=sentiment_score,
                fused_score=fused_score,
                confidence=confidence,
                stop_loss=stop_loss,
                take_profit=take_profit,
                # Phase 3 Advanced Features
                regime_detection=regime_data,
                advanced_rsi=rsi_data,
                position_sizing=risk_data.get('position_sizing', {}),
                volatility_adjusted_stops=risk_data.get('volatility_adjusted_stops', {}),
                risk_metrics=risk_data.get('risk_metrics', {}),
                technical_indicators=technical_indicators,
                market_microstructure=microstructure,
                # Phase 1 Enhanced Technical Analysis
                bollinger_bands=bollinger_bands,
                vwap_analysis=vwap_analysis,
                volume_indicators=volume_indicators,
                moving_averages=moving_averages,
                keltner_channels=keltner_channels,
                # Phase 2 Multi-Timeframe Analysis
                multi_timeframe=multi_timeframe,
                cross_asset_correlation=cross_asset_correlation,
                btc_dominance=btc_dominance,
                market_wide_sentiment=market_wide_sentiment,
                username=username
            )
            logger.info(f"✅ Complete Phase 3 signal created successfully")
            
            # Step 14: Store signal
            logger.info("💾 STEP 14: STORING SIGNAL")
            await self.store_signal(phase3_signal, username)
            logger.info(f"✅ Signal stored successfully")
            
            # Step 15: Send Telegram notification
            logger.info("📱 STEP 15: SENDING TELEGRAM NOTIFICATION")
            await self.send_telegram_notification(phase3_signal, username)
            logger.info(f"✅ Telegram notification sent")
            
            logger.info("=" * 80)
            logger.info(f"🎉 COMPLETE PHASE 3 SIGNAL GENERATION COMPLETED SUCCESSFULLY")
            logger.info(f"📊 Symbol: {request.symbol}")
            logger.info(f"🎯 Signal: {signal_type}")
            logger.info(f"📈 Technical Score: {phase3_tech_score:.4f}")
            logger.info(f"💭 Sentiment Score: {sentiment_score:.4f}")
            logger.info(f"🧮 Fused Score: {fused_score:.4f}")
            logger.info(f"🎲 Confidence: {confidence:.2%}")
            logger.info(f"🛡️ Stop Loss: ${stop_loss:.2f}")
            logger.info(f"🛡️ Take Profit: ${take_profit:.2f}")
            logger.info("=" * 50)
            logger.info("📊 PHASE 1 FEATURES:")
            logger.info(f"   • Bollinger Squeeze: {bollinger_bands.get('squeeze', False)}")
            logger.info(f"   • VWAP Deviation: {vwap_analysis.get('deviation', 0):.2f} bps")
            logger.info(f"   • Volume Trend: {volume_indicators.get('obv_trend', 'neutral')}")
            logger.info(f"   • MA Crossovers: {moving_averages.get('crossovers', {}).get('last_bullish', False)}")
            logger.info(f"   • Keltner Breakout: {keltner_channels.get('last_breakout_up', False)}")
            logger.info("=" * 50)
            logger.info("📊 PHASE 2 FEATURES:")
            logger.info(f"   • Multi-TF Trend: {multi_timeframe.get('overall_trend', 'neutral')}")
            logger.info(f"   • Trend Consensus: {multi_timeframe.get('trend_consensus', 0.5):.2%}")
            logger.info(f"   • BTC Dominance: {btc_dominance.get('btc_dominance', 50):.1f}%")
            logger.info(f"   • Market Sentiment: {market_wide_sentiment.get('overall_sentiment', 'neutral')}")
            logger.info("=" * 50)
            logger.info("📊 PHASE 3 FEATURES:")
            logger.info(f"   • Regime: {regime_data.get('regime_classification', 'unknown')}")
            logger.info(f"   • RSI Alignment: {rsi_data.get('alignment', 'mixed')}")
            logger.info(f"   • Risk/Reward: {risk_data.get('risk_metrics', {}).get('risk_reward_ratio', 1.0):.2f}")
            logger.info("=" * 80)
            
            return phase3_signal
            
        except Exception as e:
            logger.error("=" * 80)
            logger.error(f"❌ PHASE 3 SIGNAL GENERATION FAILED")
            logger.error(f"📊 Symbol: {request.symbol}")
            logger.error(f"👤 Username: {username}")
            logger.error(f"🚨 Error: {str(e)}")
            logger.error("=" * 80)
            raise

# Create global instance
phase3_signal_generator = Phase3SignalGenerator()