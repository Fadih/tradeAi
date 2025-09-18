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
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np
from pydantic import BaseModel
import aiohttp
import hashlib
import json
from functools import lru_cache
import time

from agent.config import load_config_from_env
from agent.indicators import (
    compute_rsi, compute_ema, compute_macd, compute_atr,
    compute_adx, compute_volatility_regime, compute_market_regime,
    compute_advanced_rsi_variants, compute_dynamic_position_sizing,
    compute_volatility_adjusted_stops, compute_bollinger_bands,
    compute_obv, compute_mfi, compute_vwap_anchored,
    compute_accumulation_distribution, compute_ma_crossovers_and_slopes,
    compute_keltner_channels, compute_multi_timeframe_analysis,
    compute_cross_asset_correlation, compute_btc_dominance,
    compute_market_wide_sentiment
)
from agent.models.sentiment import SentimentAnalyzer
from agent.news.rss import fetch_headlines
from agent.news.reddit import fetch_reddit_posts
from agent.data.ccxt_client import fetch_ohlcv
from agent.data.alpaca_client import fetch_ohlcv as fetch_alpaca_ohlcv
from agent.cache.redis_client import get_redis_client

# Local time helper to avoid circular import with web.main
try:
    import pytz  # type: ignore
    def get_israel_time() -> datetime:
        return datetime.now(pytz.timezone("Asia/Jerusalem"))
except Exception:
    # Fallback without pytz
    def get_israel_time() -> datetime:
        return datetime.now()

logger = logging.getLogger(__name__)

def timing_decorator(phase_name: str):
    """Decorator to time function execution"""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            result = await func(*args, **kwargs)
            end_time = time.time()
            duration = end_time - start_time
            logger.info(f"⏱️  {phase_name}: {duration:.3f}s")
            return result
        
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            duration = end_time - start_time
            logger.info(f"⏱️  {phase_name}: {duration:.3f}s")
            return result
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    return decorator

def to_serializable(obj: Any) -> Any:
    """Recursively convert pandas and numpy types to JSON-serializable Python types."""
    try:
        import numpy as _np  # local alias to avoid shadowing
        import pandas as _pd
    except Exception:
        _np = None
        _pd = None

    # Handle scalar values first
    if _np is not None and isinstance(obj, (_np.bool_,)):
        return bool(obj)
    if _np is not None and isinstance(obj, (_np.integer,)):
        return int(obj)
    if _np is not None and isinstance(obj, (_np.floating,)):
        val = float(obj)
        # Handle NaN, inf, -inf
        if _np.isnan(val) or _np.isinf(val):
            return 0.0
        return val
    
    # Handle Python built-in types
    if isinstance(obj, (int, float)):
        if _np is not None and (_np.isnan(obj) or _np.isinf(obj)):
            return 0.0
        return obj
    if isinstance(obj, bool):
        return obj
    if obj is None:
        return None

    if _pd is not None and isinstance(obj, _pd.Series):
        # Convert element-wise with best-effort numeric casting
        result_list = []
        for v in obj.tolist():
            try:
                if v is None:
                    result_list.append(None)
                elif _np is not None and isinstance(v, (_np.bool_,)):
                    result_list.append(bool(v))
                elif _np is not None and isinstance(v, (_np.integer, _np.floating)):
                    val = float(v)
                    if _np.isnan(val) or _np.isinf(val):
                        result_list.append(0.0)
                    else:
                        result_list.append(val)
                else:
                    # Try numeric conversion first
                    val = float(v)
                    if _np is not None and (_np.isnan(val) or _np.isinf(val)):
                        result_list.append(0.0)
                    else:
                        result_list.append(val)
            except Exception:
                # Fallback to string representation
                result_list.append(str(v))
        return result_list
    if _pd is not None and isinstance(obj, _pd.DataFrame):
        out: Dict[str, Any] = {}
        for col in obj.columns:
            try:
                series = obj[col]
                out[col] = to_serializable(series)
            except Exception:
                out[str(col)] = [str(v) for v in obj[col].tolist()]
        return out
    if _np is not None and isinstance(obj, _np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_serializable(v) for v in obj]
    return obj

class Phase3SignalRequest(BaseModel):
    """Phase 3 signal generation request with custom parameters"""
    symbol: str
    timeframe: str
    buy_threshold: Optional[float] = None
    sell_threshold: Optional[float] = None
    technical_weight: Optional[float] = None
    sentiment_weight: Optional[float] = None

class Phase3TradingSignal(BaseModel):
    """Phase 3 complete trading signal model with all Phase 1, 2, and 3 features"""
    
    # Core fields
    symbol: str
    timeframe: str
    timestamp: datetime
    signal_type: str
    technical_score: float
    sentiment_score: float
    fused_score: float
    confidence: float
    stop_loss: float
    take_profit: float
    output_level: str = "full"  # Performance optimization: "minimal", "standard", "full"
        
        # Phase 3 Advanced Features
    regime_detection: Dict[str, Any]
    advanced_rsi: Dict[str, Any]
    position_sizing: Dict[str, Any]
    volatility_adjusted_stops: Dict[str, Any]
    risk_metrics: Dict[str, Any]
    technical_indicators: Dict[str, Any]
    market_microstructure: Dict[str, Any]
        
        # Phase 1 Enhanced Technical Analysis
    bollinger_bands: Dict[str, Any]
    vwap_analysis: Dict[str, Any]
    volume_indicators: Dict[str, Any]
    moving_averages: Dict[str, Any]
    keltner_channels: Dict[str, Any]
        
        # Phase 2 Multi-Timeframe Analysis
    multi_timeframe: Dict[str, Any]
    cross_asset_correlation: Dict[str, Any]
    btc_dominance: Dict[str, Any]
    market_wide_sentiment: Dict[str, Any]

    # Applied Parameters (for UI display)
    applied_buy_threshold: Optional[float] = None
    applied_sell_threshold: Optional[float] = None
    applied_tech_weight: Optional[float] = None
    applied_sentiment_weight: Optional[float] = None

    # Metadata
    username: str

class Phase3SignalGenerator:
    """Phase 3 Advanced Signal Generator with Performance Optimizations"""
    
    def __init__(self):
        self.sentiment_analyzer = None
        self.session = None  # aiohttp session for async requests
        self.cache_ttl = 300  # 5 minutes cache TTL
        logger.info("🚀 Phase 3 Signal Generator initialized - Performance Optimized mode")
    
    
    def fetch_market_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Fetch market data for Phase 3 analysis"""
        logger.info(f"📊 Fetching market data for {symbol} @ {timeframe}")
        
        try:
            # Try crypto first
            if "/" in symbol and any(crypto in symbol.upper() for crypto in ["BTC", "ETH", "ADA", "DOT", "LINK", "UNI", "AAVE", "SOL", "MATIC", "AVAX"]):
                ohlcv = fetch_ohlcv(symbol, timeframe)
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
    
    def _get_last_valid_value(self, series: pd.Series) -> float:
        """Get the last valid (non-NaN) value from a pandas Series"""
        try:
            # Drop NaN values and get the last value
            valid_values = series.dropna()
            if len(valid_values) > 0:
                return float(valid_values.iloc[-1])
            else:
                return 0.0
        except Exception:
            return 0.0

    def calculate_phase3_technical_indicators(self, ohlcv: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive technical indicators for Phase 3 with all Phase 1, 2, and 3 features"""
        start_time = time.time()
        logger.info("🔧 Calculating Phase 3 complete technical indicators")
        
        try:
            high = ohlcv['high']
            low = ohlcv['low']
            close = ohlcv['close']
            volume = ohlcv['volume']
            
            indicators = {}
            
            # Basic indicators
            logger.info("📊 Computing RSI indicators (14, 21 periods)")
            rsi_14_series = compute_rsi(close, period=14)
            rsi_21_series = compute_rsi(close, period=21)
            indicators['rsi_14'] = rsi_14_series
            indicators['rsi_21'] = rsi_21_series
            # Add simple 'rsi' field for UI compatibility (scalar value)
            rsi_value = self._get_last_valid_value(rsi_14_series)
            indicators['rsi'] = rsi_value
            logger.info(f"📊 RSI: {rsi_value}")
            
            # Enhanced Moving Averages (Phase 1) - All periods
            logger.info("📊 Computing EMAs (5, 9, 12, 21, 26, 50, 200 periods)")
            indicators['ema_5'] = compute_ema(close, span=5)
            indicators['ema_9'] = compute_ema(close, span=9)
            indicators['ema_12'] = compute_ema(close, span=12)
            indicators['ema_21'] = compute_ema(close, span=21)
            indicators['ema_26'] = compute_ema(close, span=26)
            indicators['ema_50'] = compute_ema(close, span=50)
            indicators['ema_200'] = compute_ema(close, span=200)
            
            # MACD
            logger.info("📊 Computing MACD (12, 26, 9)")
            macd_data = compute_macd(close)
            macd_value = self._get_last_valid_value(macd_data['macd'])
            indicators['macd'] = macd_value
            indicators['macd_signal'] = macd_data['signal']
            # Indicators DataFrame uses 'hist' column name
            indicators['macd_histogram'] = macd_data['hist']
            logger.info(f"📈 MACD: {macd_value}")
            
            # ATR
            logger.info("📊 Computing ATR (14 period)")
            atr_series = compute_atr(high, low, close, period=14)
            atr_value = self._get_last_valid_value(atr_series)
            indicators['atr'] = atr_value
            logger.info(f"📉 ATR: {atr_value}")
            
            # ADX for trend strength
            logger.info("📊 Computing ADX and DI indicators (14 period)")
            adx_data = compute_adx(high, low, close, period=14)
            indicators['adx'] = adx_data['adx']
            indicators['di_plus'] = adx_data['di_plus']
            indicators['di_minus'] = adx_data['di_minus']
            
            duration = time.time() - start_time
            logger.info(f"✅ Phase 3 technical indicators calculated in {duration:.3f}s")
            
            # Log key values
            try:
                rsi_14_val = float(indicators['rsi_14'].iloc[-1]) if not indicators['rsi_14'].empty else None
                ema_21_val = float(indicators['ema_21'].iloc[-1]) if not indicators['ema_21'].empty else None
                ema_50_val = float(indicators['ema_50'].iloc[-1]) if not indicators['ema_50'].empty else None
                atr_val = float(indicators['atr'].iloc[-1]) if not indicators['atr'].empty else None
                adx_val = float(indicators['adx'].iloc[-1]) if not indicators['adx'].empty else None
                logger.info(f"📊 Key values: RSI14={rsi_14_val:.2f}, EMA21={ema_21_val:.2f}, EMA50={ema_50_val:.2f}, ATR={atr_val:.2f}, ADX={adx_val:.2f}")
            except Exception:
                pass
                
            return indicators
            
        except Exception as e:
            logger.error(f"❌ Error calculating Phase 3 technical indicators: {e}")
            raise
    
    def analyze_advanced_regime_detection(self, ohlcv: pd.DataFrame) -> Dict[str, Any]:
        """Advanced market regime detection for Phase 3"""
        start_time = time.time()
        logger.info("🔍 Analyzing advanced regime detection")
        
        try:
            high = ohlcv['high']
            low = ohlcv['low']
            close = ohlcv['close']
            
            # Get ADX-based regime detection
            adx_data = compute_adx(high, low, close, period=14)
            
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
            
            duration = time.time() - start_time
            logger.info(f"✅ Advanced regime detection completed in {duration:.3f}s: {combined_regime['regime_classification']}")
            logger.info(f"🔍 Regime details: trend_strength={combined_regime['trend_strength']:.3f}, volatility_state={combined_regime['volatility_state']}")
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
        start_time = time.time()
        logger.info("📊 Analyzing advanced RSI variants")
        
        try:
            close = ohlcv['close']
            
            # Get advanced RSI variants from indicators helper
            rsi_data = compute_advanced_rsi_variants(close, periods=[7, 9, 14, 21])
            
            # Normalize keys defensively (some helpers may not return all keys)
            rsi_7 = rsi_data.get('rsi_7', pd.Series([50]))
            rsi_14 = rsi_data.get('rsi_14', pd.Series([50]))
            rsi_21 = rsi_data.get('rsi_21', pd.Series([50]))
            
            # RSI alignment analysis - handle both Series and scalar values
            def get_rsi_value(rsi_data):
                if hasattr(rsi_data, 'iloc') and len(rsi_data) > 0:
                    return float(rsi_data.iloc[-1])
                elif isinstance(rsi_data, (int, float)):
                    return float(rsi_data)
                else:
                    return 50.0
            
            current_rsi_7 = get_rsi_value(rsi_7)
            current_rsi_14 = get_rsi_value(rsi_14)
            current_rsi_21 = get_rsi_value(rsi_21)
            
            if current_rsi_7 > current_rsi_14 > current_rsi_21:
                rsi_alignment = "bullish"
            elif current_rsi_7 < current_rsi_14 < current_rsi_21:
                rsi_alignment = "bearish"
            else:
                rsi_alignment = "mixed"
            
            rsi_data['alignment'] = rsi_alignment
            rsi_data['current_values'] = {
                'rsi_7': current_rsi_7,
                'rsi_14': current_rsi_14,
                'rsi_21': current_rsi_21
            }
            
            duration = time.time() - start_time
            logger.info(f"✅ Advanced RSI variants analysis completed in {duration:.3f}s: {rsi_alignment}")
            logger.info(f"📊 RSI values: RSI7={current_rsi_7:.2f}, RSI14={current_rsi_14:.2f}, RSI21={current_rsi_21:.2f}")
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
            
            # Collect texts from RSS and Reddit (recent, symbol-filtered)
            rss_texts = fetch_headlines(
                config.sentiment_analysis.rss_feeds,
                limit_per_feed=config.sentiment_analysis.rss_max_headlines_per_feed,
                symbol=symbol,
                hours_back=6,
            )
            reddit_texts = fetch_reddit_posts(
                config.sentiment_analysis.reddit_subreddits,
                limit_per_subreddit=config.sentiment_analysis.reddit_max_posts_per_subreddit,
                symbol=symbol,
                hours_back=6,
            )
            texts = (rss_texts or []) + (reddit_texts or [])
            if not texts:
                logger.info("No sentiment texts collected; returning neutral sentiment 0.0")
                return 0.0

            # Score sentiment using analyzer
            sentiment_score = self.sentiment_analyzer.score(texts)
            
            logger.info(f"✅ Phase 3 sentiment analysis completed: {sentiment_score:.4f}")
            return sentiment_score
            
        except Exception as e:
            logger.error(f"❌ Error in Phase 3 sentiment analysis: {e}")
            return 0.0
    
    def calculate_phase3_technical_score(self, indicators: Dict[str, Any], regime_data: Dict[str, Any], rsi_data: Dict[str, Any]) -> float:
        """Calculate Phase 3 technical score"""
        start_time = time.time()
        logger.info("🧮 Calculating Phase 3 technical score")
        
        try:
            score = 0.0
            
            # RSI analysis (40% weight)
            rsi_14 = indicators.get('rsi_14', pd.Series([50]))
            current_rsi = rsi_14.iloc[-1] if len(rsi_14) > 0 else 50
            
            logger.info(f"🔍 RSI Analysis: current_rsi={current_rsi:.2f}, type={type(current_rsi)}")
            
            if current_rsi < 30:
                rsi_score = 0.8  # Oversold - bullish
                logger.info(f"📉 RSI Oversold: {current_rsi:.2f} < 30 → score=0.8")
            elif current_rsi > 70:
                rsi_score = -0.8  # Overbought - bearish
                logger.info(f"📈 RSI Overbought: {current_rsi:.2f} > 70 → score=-0.8")
            elif 40 <= current_rsi <= 60:
                rsi_score = 0.0  # Neutral
                logger.info(f"⚖️ RSI Neutral: 40 <= {current_rsi:.2f} <= 60 → score=0.0")
            else:
                rsi_score = (50 - current_rsi) / 50  # Linear interpolation
                logger.info(f"📊 RSI Linear: (50 - {current_rsi:.2f}) / 50 = {rsi_score:.4f}")
            
            # RSI alignment bonus
            rsi_alignment = rsi_data.get('alignment', 'mixed')
            logger.info(f"🔄 RSI Alignment: {rsi_alignment}")
            if rsi_alignment == 'bullish':
                rsi_score += 0.2
                logger.info(f"📈 RSI Bullish alignment bonus: +0.2 → {rsi_score:.4f}")
            elif rsi_alignment == 'bearish':
                rsi_score -= 0.2
                logger.info(f"📉 RSI Bearish alignment penalty: -0.2 → {rsi_score:.4f}")
            
            rsi_contribution = rsi_score * 0.4
            score += rsi_contribution
            logger.info(f"🎯 RSI Final: score={rsi_score:.4f} * 0.4 = {rsi_contribution:.4f}")
            
            # MACD analysis (25% weight)
            macd = indicators.get('macd', pd.Series([0]))
            macd_signal = indicators.get('macd_signal', pd.Series([0]))
            
            logger.info(f"📊 MACD Analysis: macd_series_length={len(macd)}, signal_series_length={len(macd_signal)}")
            
            if len(macd) > 0 and len(macd_signal) > 0:
                current_macd = macd.iloc[-1]
                current_signal = macd_signal.iloc[-1]
                
                logger.info(f"📊 MACD Values: MACD={current_macd:.4f}, Signal={current_signal:.4f}")
                
                if current_macd > current_signal:
                    macd_score = 0.5
                    logger.info(f"📈 MACD Bullish: {current_macd:.4f} > {current_signal:.4f} → score=0.5")
                else:
                    macd_score = -0.5
                    logger.info(f"📉 MACD Bearish: {current_macd:.4f} <= {current_signal:.4f} → score=-0.5")
                
                macd_contribution = macd_score * 0.25
                score += macd_contribution
                logger.info(f"🎯 MACD Final: score={macd_score:.4f} * 0.25 = {macd_contribution:.4f}")
            else:
                logger.warning("⚠️ MACD data missing or empty")
            
            # Bollinger Bands analysis (20% weight)
            bb_percent = indicators.get('bb_percent', pd.Series([0.5]))
            logger.info(f"📊 Bollinger Bands Analysis: bb_percent_length={len(bb_percent)}")
            
            if len(bb_percent) > 0:
                current_bb = bb_percent.iloc[-1]
                logger.info(f"📊 BB Percent: {current_bb:.4f}")
                
                if current_bb < 0.2:
                    bb_score = 0.6  # Near lower band - bullish
                    logger.info(f"📈 BB Near Lower Band: {current_bb:.4f} < 0.2 → score=0.6")
                elif current_bb > 0.8:
                    bb_score = -0.6  # Near upper band - bearish
                    logger.info(f"📉 BB Near Upper Band: {current_bb:.4f} > 0.8 → score=-0.6")
                else:
                    bb_score = (0.5 - current_bb) * 2  # Linear interpolation
                    logger.info(f"📊 BB Linear: (0.5 - {current_bb:.4f}) * 2 = {bb_score:.4f}")
                
                bb_contribution = bb_score * 0.2
                score += bb_contribution
                logger.info(f"🎯 BB Final: score={bb_score:.4f} * 0.2 = {bb_contribution:.4f}")
            else:
                logger.warning("⚠️ Bollinger Bands data missing or empty")
            
            # ADX trend strength (15% weight)
            trend_strength = regime_data.get('trend_strength', 0.5)
            regime_classification = regime_data.get('regime_classification', 'trending')
            
            logger.info(f"📊 ADX Analysis: trend_strength={trend_strength:.2f}, regime={regime_classification}")
            
            if regime_classification == 'trending' and trend_strength > 25:
                adx_score = 0.3
                logger.info(f"📈 ADX Strong Trend: regime='trending' and strength={trend_strength:.2f} > 25 → score=0.3")
            elif regime_classification == 'consolidation':
                adx_score = -0.1
                logger.info(f"📉 ADX Consolidation: regime='consolidation' → score=-0.1")
            else:
                adx_score = 0.0
                logger.info(f"⚖️ ADX Neutral: regime='{regime_classification}', strength={trend_strength:.2f} → score=0.0")
            
            adx_contribution = adx_score * 0.15
            score += adx_contribution
            logger.info(f"🎯 ADX Final: score={adx_score:.4f} * 0.15 = {adx_contribution:.4f}")
            
            # Normalize score to [-1, 1]
            # Optional Phase 4 extensions: derivatives & microstructure (opt-in)
            try:
                config = load_config_from_env()
                phase4_enabled = getattr(config, 'phase4', {}).get('enabled', False)
            except Exception:
                phase4_enabled = False
            
            if phase4_enabled:
                logger.info("🧩 Phase 4 enabled: blending derivatives and microstructure scores")
                derivatives_score = getattr(self, '_last_derivatives_score', 0.0)
                microstructure_score = getattr(self, '_last_microstructure_score', 0.0)
                try:
                    derivatives_weight = config.phase4['weights']['derivatives_weight']
                    microstructure_weight = config.phase4['weights']['microstructure_weight']
                except Exception:
                    derivatives_weight = 0.15
                    microstructure_weight = 0.10
                # Blend with conservative weights
                score += derivatives_score * derivatives_weight
                score += microstructure_score * microstructure_weight
                logger.info(f"🧩 Phase 4 blend -> derivatives: {derivatives_score:.3f} (w={derivatives_weight}), micro: {microstructure_score:.3f} (w={microstructure_weight})")
            
            # Log total score before normalization
            logger.info(f"📊 Total Technical Score (before normalization): {score:.4f}")
            
            final_score = max(-1.0, min(1.0, score))
            
            duration = time.time() - start_time
            logger.info(f"✅ Phase 3 technical score calculated in {duration:.3f}s: {final_score:.4f}")
            logger.info(f"🧮 Score breakdown: RSI={rsi_score:.3f}*0.4, MACD={macd_score:.3f}*0.25, BB={bb_score:.3f}*0.2, ADX={adx_score:.3f}*0.15")
            logger.info(f"🎯 Final normalized score: {final_score:.4f} (clamped to [-1, 1])")
            return final_score
            
        except Exception as e:
            logger.error(f"❌ Error calculating Phase 3 technical score: {e}")
            return 0.0
    
    def calculate_phase3_fused_score(self, technical_score: float, sentiment_score: float, 
                                   technical_weight: Optional[float] = None, sentiment_weight: Optional[float] = None) -> float:
        """Calculate Phase 3 fused score"""
        start_time = time.time()
        logger.info("🔗 Calculating Phase 3 fused score")
        
        try:
            from agent.config import load_config_from_env
            config = load_config_from_env()
            
            # Use custom weights if provided, otherwise use config defaults
            tech_weight = technical_weight if technical_weight is not None else config.thresholds.technical_weight
            sent_weight = sentiment_weight if sentiment_weight is not None else config.thresholds.sentiment_weight
            
            fused_score = tech_weight * technical_score + sent_weight * sentiment_score
            
            duration = time.time() - start_time
            logger.info(f"✅ Phase 3 fused score calculated in {duration:.3f}s: {fused_score:.4f}")
            logger.info(f"🔗 Fused breakdown: tech={technical_score:.4f}*{tech_weight:.2f} + sent={sentiment_score:.4f}*{sent_weight:.2f}")
            return fused_score
            
        except Exception as e:
            logger.error(f"❌ Error calculating Phase 3 fused score: {e}")
            return 0.0
    
    def determine_signal_type_and_confidence(self, fused_score: float, buy_threshold: Optional[float] = None, sell_threshold: Optional[float] = None) -> tuple:
        """Determine signal type and confidence from fused score"""
        start_time = time.time()
        logger.info("🎯 Determining signal type and confidence")
        
        try:
            from agent.config import load_config_from_env
            config = load_config_from_env()
            
            # Use custom thresholds if provided, otherwise use config defaults
            buy_thresh = buy_threshold if buy_threshold is not None else config.thresholds.buy_threshold
            sell_thresh = sell_threshold if sell_threshold is not None else -buy_thresh
            
            logger.info(f"🎯 Thresholds: BUY>={buy_thresh:.3f}, SELL<={sell_thresh:.3f}, current={fused_score:.4f}")
            
            # Determine signal type
            if fused_score >= buy_thresh:
                signal_type = "BUY"
            elif fused_score <= sell_thresh:
                signal_type = "SELL"
            else:
                signal_type = "HOLD"
            
            # Calculate confidence based on how far from neutral
            confidence = min(abs(fused_score) * 2, 1.0)  # Scale to 0-1
            
            duration = time.time() - start_time
            logger.info(f"✅ Signal type and confidence determined in {duration:.3f}s: {signal_type} ({confidence:.2%})")
            return signal_type, confidence
            
        except Exception as e:
            logger.error(f"❌ Error determining signal type and confidence: {e}")
            return "HOLD", 0.5
    
    def calculate_enhanced_risk_management(self, ohlcv: pd.DataFrame, technical_indicators: Dict[str, Any], regime_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Calculate enhanced risk management metrics"""
        start_time = time.time()
        logger.info("🛡️ Calculating enhanced risk management")
        
        try:
            current_price = ohlcv['close'].iloc[-1]
            volatility = technical_indicators.get('atr', pd.Series([current_price * 0.02])).iloc[-1]
            
            risk_metrics = {
                'volatility': float(volatility / current_price),
                'var_95': float(volatility * 1.645 / current_price),  # 95% VaR
                'sharpe_ratio': 1.0,  # Placeholder
                'max_drawdown': 0.05  # Placeholder
            }
            
            duration = time.time() - start_time
            logger.info(f"✅ Enhanced risk management calculated in {duration:.3f}s")
            logger.info(f"🛡️ Risk metrics: vol={risk_metrics['volatility']:.4f}, VaR95={risk_metrics['var_95']:.4f}, Sharpe={risk_metrics['sharpe_ratio']:.2f}")
            return risk_metrics
            
        except Exception as e:
            logger.error(f"❌ Error calculating enhanced risk management: {e}")
            current_price = ohlcv['close'].iloc[-1]
            return {
                'volatility': 0.02,
                'var_95': 0.03,
                'sharpe_ratio': 1.0,
                'max_drawdown': 0.05
            }
    
    def calculate_volatility_adjusted_stops(self, ohlcv: pd.DataFrame, signal_type: str, risk_metrics: Dict[str, Any]) -> tuple:
        """Calculate volatility-adjusted stop loss and take profit"""
        try:
            current_price = ohlcv['close'].iloc[-1]
            volatility = risk_metrics.get('volatility', 0.02)
            
            if signal_type == "BUY":
                stop_loss = current_price * (1 - volatility * 2)
                take_profit = current_price * (1 + volatility * 3)
            elif signal_type == "SELL":
                stop_loss = current_price * (1 + volatility * 2)
                take_profit = current_price * (1 - volatility * 3)
            else:
                stop_loss = current_price * 0.98
                take_profit = current_price * 1.02
            
            return float(stop_loss), float(take_profit)
            
        except Exception as e:
            logger.error(f"❌ Error calculating volatility-adjusted stops: {e}")
            current_price = ohlcv['close'].iloc[-1]
            return float(current_price * 0.98), float(current_price * 1.02)
    
    def compute_dynamic_position_sizing(self, risk_metrics: Dict[str, Any], confidence: float, signal_type: str) -> Dict[str, Any]:
        """Compute dynamic position sizing based on risk and confidence"""
        start_time = time.time()
        logger.info("📏 Computing dynamic position sizing")
        
        try:
            volatility = risk_metrics.get('volatility', 0.02)
            
            # Base position size
            base_size = 0.1  # 10%
            
            # Adjust for confidence
            confidence_multiplier = confidence
            
            # Adjust for volatility (lower volatility = larger position)
            volatility_multiplier = max(0.5, 1.0 - volatility * 10)
            
            # Adjust for signal type
            signal_multiplier = 1.0 if signal_type != "HOLD" else 0.5
            
            recommended_size = base_size * confidence_multiplier * volatility_multiplier * signal_multiplier
            recommended_size = min(recommended_size, 0.2)  # Cap at 20%
            
            result = {
                'recommended_size': recommended_size,
                'risk_per_trade': recommended_size * 0.1,  # 10% of position as risk
                'max_position': 0.2  # 20% max position
            }
            
            duration = time.time() - start_time
            logger.info(f"✅ Dynamic position sizing computed in {duration:.3f}s")
            logger.info(f"📏 Position sizing: base={base_size:.1%}, conf_mult={confidence_multiplier:.2f}, vol_mult={volatility_multiplier:.2f}, signal_mult={signal_multiplier:.1f}")
            logger.info(f"📏 Final sizing: recommended={recommended_size:.1%}, risk_per_trade={result['risk_per_trade']:.1%}, max={result['max_position']:.1%}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error computing dynamic position sizing: {e}")
            return {
                'recommended_size': 0.1,
                'risk_per_trade': 0.01,
                'max_position': 0.2
            }
    
    def generate_phase3_reasoning(self, signal_type: str, technical_score: float, sentiment_score: float, fused_score: float, regime_data: Dict[str, Any], rsi_data: Dict[str, Any]) -> str:
        """Generate Phase 3 reasoning for the signal"""
        try:
            regime = regime_data.get('regime_classification', 'unknown')
            rsi_alignment = rsi_data.get('alignment', 'neutral')
            
            reasoning = f"Phase 3 Analysis: {signal_type} signal based on "
            reasoning += f"Technical Score: {technical_score:.3f}, "
            reasoning += f"Sentiment Score: {sentiment_score:.3f}, "
            reasoning += f"Fused Score: {fused_score:.3f}. "
            reasoning += f"Market Regime: {regime}, RSI Alignment: {rsi_alignment}."
            
            return reasoning
            
        except Exception as e:
            logger.error(f"❌ Error generating Phase 3 reasoning: {e}")
            return f"Phase 3 Analysis: {signal_type} signal based on technical and sentiment analysis."
    
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

    # ==============================
    # Phase 4 (Optional) Stub Scorers
    # ==============================
    def analyze_derivatives_stub(self, symbol: str) -> Dict[str, Any]:
        """Stub for derivatives analysis (funding, open interest, basis).
        Returns a dict and stores a normalized score in self._last_derivatives_score [-1,1]."""
        logger.info("📈 [Phase4] Derivatives stub analysis")
        try:
            config = load_config_from_env()
            if not getattr(config, 'phase4', {}).get('derivatives', {}).get('enabled', False):
                self._last_derivatives_score = 0.0
                return {"enabled": False, "score": 0.0}
            # Placeholder heuristic: neutral score
            self._last_derivatives_score = 0.0
            return {
                "enabled": True,
                "funding_rate_extreme": config.phase4['derivatives']['funding_rate_extreme'],
                "open_interest_lookback_hours": config.phase4['derivatives']['open_interest_lookback_hours'],
                "basis_lookback_hours": config.phase4['derivatives']['basis_lookback_hours'],
                "score": self._last_derivatives_score
            }
        except Exception as e:
            logger.warning(f"[Phase4] Derivatives stub failed: {e}")
            self._last_derivatives_score = 0.0
            return {"enabled": False, "score": 0.0}

    def analyze_microstructure_stub(self, symbol: str) -> Dict[str, Any]:
        """Stub for microstructure analysis (order book imbalance, spread, flow).
        Returns a dict and stores a normalized score in self._last_microstructure_score [-1,1]."""
        logger.info("📊 [Phase4] Microstructure stub analysis")
        try:
            config = load_config_from_env()
            if not getattr(config, 'phase4', {}).get('microstructure', {}).get('enabled', False):
                self._last_microstructure_score = 0.0
                return {"enabled": False, "score": 0.0}
            # Placeholder heuristic: neutral score
            self._last_microstructure_score = 0.0
            return {
                "enabled": True,
                "order_book_levels": config.phase4['microstructure']['order_book_levels'],
                "imbalance_threshold": config.phase4['microstructure']['imbalance_threshold'],
                "spread_widen_threshold_bps": config.phase4['microstructure']['spread_widen_threshold_bps'],
                "score": self._last_microstructure_score
            }
        except Exception as e:
            logger.warning(f"[Phase4] Microstructure stub failed: {e}")
            self._last_microstructure_score = 0.0
            return {"enabled": False, "score": 0.0}
    
    
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
            vwap_data = compute_vwap_anchored(ohlcv)
            
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
        """Analyze multi-timeframe data (Phase 2) - OPTIMIZED: Only for requested symbol"""
        logger.info("📊 Analyzing multi-timeframe data (optimized for single symbol)")
        
        try:
            # OPTIMIZATION: Only fetch data for the requested symbol across multiple timeframes
            # This is much more efficient than fetching multiple symbols
            logger.info(f"🚀 OPTIMIZATION: Multi-timeframe analysis for {symbol} only (no cross-symbol fetching)")
            
            # Use a simplified multi-timeframe analysis for the single symbol
            # We'll analyze the current timeframe and derive trend consensus
            from agent.data import fetch_ohlcv
            
            timeframes = ['1m', '5m']  # Optimized for ultra-short-term scalping
            trend_scores = []
            
            for tf in timeframes:
                try:
                    logger.info(f"📊 Fetching {symbol} data for {tf} timeframe")
                    ohlcv = fetch_ohlcv(symbol, tf, limit=50)  # Reduced limit for faster fetching
                    
                    if len(ohlcv) > 0:
                        close = ohlcv['close']
                        # Simple trend analysis using EMA
                        ema_20 = compute_ema(close, 20)
                        ema_50 = compute_ema(close, 50)
                        
                        if len(ema_20) > 0 and len(ema_50) > 0:
                            current_ema20 = ema_20.iloc[-1]
                            current_ema50 = ema_50.iloc[-1]
                            
                            if current_ema20 > current_ema50:
                                trend_scores.append(1.0)  # Bullish
                                logger.info(f"📈 {tf}: Bullish trend (EMA20 > EMA50)")
                            else:
                                trend_scores.append(-1.0)  # Bearish
                                logger.info(f"📉 {tf}: Bearish trend (EMA20 < EMA50)")
                        else:
                            trend_scores.append(0.0)  # Neutral
                    else:
                        trend_scores.append(0.0)  # Neutral
                        
                except Exception as e:
                    logger.warning(f"⚠️ Error fetching {tf} data for {symbol}: {e}")
                    trend_scores.append(0.0)  # Neutral on error
            
            # Calculate overall trend consensus
            if trend_scores:
                avg_trend = sum(trend_scores) / len(trend_scores)
                if avg_trend > 0.3:
                    overall_trend = 'bullish'
                elif avg_trend < -0.3:
                    overall_trend = 'bearish'
                else:
                    overall_trend = 'neutral'
                
                trend_consensus = (avg_trend + 1) / 2  # Convert to 0-1 scale
            else:
                overall_trend = 'neutral'
                trend_consensus = 0.5
            
            logger.info(f"✅ Multi-timeframe analysis completed - Overall trend: {overall_trend} (consensus: {trend_consensus:.2f})")
            
            return {
                'overall_trend': overall_trend,
                'trend_consensus': trend_consensus,
                'timeframes_analyzed': timeframes,
                'trend_scores': trend_scores,
                'optimization_note': 'Multi-timeframe analysis optimized for single symbol only'
            }
            
        except Exception as e:
            logger.error(f"❌ Error analyzing multi-timeframe: {e}")
            return {'overall_trend': 'neutral', 'trend_consensus': 0.5}
    
    def analyze_cross_asset_correlation(self, symbol: str) -> Dict[str, Any]:
        """Analyze cross-asset correlation (Phase 2) - OPTIMIZED: Skip cross-asset analysis"""
        logger.info("📊 Analyzing cross-asset correlation (optimized - disabled for performance)")
        
        try:
            # OPTIMIZATION: Skip cross-asset correlation to avoid fetching multiple symbols
            # Cross-asset correlation shows how your symbol moves with other cryptocurrencies
            # Example: Most altcoins have 70-90% correlation with Bitcoin
            logger.info(f"🚀 OPTIMIZATION: Skipping cross-asset correlation for {symbol} to improve performance")
            logger.info("💡 Cross-asset correlation disabled - focusing on single symbol analysis")
            logger.info("📊 Cross-asset correlation would show how {symbol} moves with BTC/ETH")
            
            # Return a simplified result
            return {
                'avg_correlation': 0.5,  # Neutral correlation
                'btc_correlations': {},
                'optimization_note': 'Cross-asset correlation disabled for performance',
                'explanation': 'Cross-asset correlation shows how your symbol moves with other cryptocurrencies (e.g., BTC, ETH)'
            }
            
        except Exception as e:
            logger.error(f"❌ Error analyzing cross-asset correlation: {e}")
            return {'avg_correlation': 0, 'btc_correlations': {}}
    
    def analyze_btc_dominance(self) -> Dict[str, Any]:
        """Analyze BTC dominance (Phase 2) - OPTIMIZED: Skip BTC dominance analysis"""
        logger.info("📊 Analyzing BTC dominance (optimized - disabled for performance)")
        
        try:
            # OPTIMIZATION: Skip BTC dominance analysis to avoid additional API calls
            # BTC dominance shows Bitcoin's market share vs all other cryptocurrencies
            # When BTC dominance rises, altcoins usually fall (and vice versa)
            logger.info("🚀 OPTIMIZATION: Skipping BTC dominance analysis to improve performance")
            logger.info("💡 BTC dominance analysis disabled - focusing on single symbol analysis")
            logger.info("📊 BTC dominance shows Bitcoin's market share vs all other cryptocurrencies")
            
            # Return a neutral default value
            return {
                'btc_dominance': 50.0,  # Neutral dominance
                'dominance_change': 0.0,
                'optimization_note': 'BTC dominance analysis disabled for performance',
                'explanation': 'BTC dominance shows Bitcoin market share vs all other cryptocurrencies'
            }
            
        except Exception as e:
            logger.error(f"❌ Error analyzing BTC dominance: {e}")
            return {'btc_dominance': 50, 'dominance_change': 0}
    
    def analyze_market_wide_sentiment(self) -> Dict[str, Any]:
        """Analyze market-wide sentiment (Phase 2) - OPTIMIZED: Use existing sentiment analysis"""
        logger.info("📊 Analyzing market-wide sentiment (optimized - using existing sentiment)")
        
        try:
            # OPTIMIZATION: Skip additional market-wide sentiment analysis since we already have sentiment
            # Market-wide sentiment analyzes sentiment across the entire crypto market (not just your symbol)
            # When whole market is bearish, even good technicals might fail
            logger.info("🚀 OPTIMIZATION: Skipping market-wide sentiment analysis to improve performance")
            logger.info("💡 Market-wide sentiment analysis disabled - using existing sentiment analysis")
            logger.info("📊 Market-wide sentiment analyzes sentiment across entire crypto market")
            
            # Return a neutral default value (the main sentiment analysis already provides this)
            return {
                'market_sentiment_score': 0.0,  # Neutral sentiment
                'overall_sentiment': 'neutral',
                'optimization_note': 'Market-wide sentiment analysis disabled for performance',
                'explanation': 'Market-wide sentiment analyzes sentiment across entire crypto market'
            }
            
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
                    # Applied Parameters (for UI display)
                    'applied_buy_threshold': signal.applied_buy_threshold,
                    'applied_sell_threshold': signal.applied_sell_threshold,
                    'applied_tech_weight': signal.applied_tech_weight,
                    'applied_sentiment_weight': signal.applied_sentiment_weight,
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
                
                # Log the applied threshold values before storage
                logger.info(f"🔍 Storing applied thresholds: buy={signal.applied_buy_threshold}, sell={signal.applied_sell_threshold}, tech={signal.applied_tech_weight}, sent={signal.applied_sentiment_weight}")
                
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
    
    # Performance optimization methods
    def _get_cache_key(self, prefix: str, *args) -> str:
        """Generate cache key from arguments"""
        key_data = f"{prefix}:{':'.join(str(arg) for arg in args)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def _get_cached_data(self, cache_key: str) -> Optional[Any]:
        """Get data from Redis cache"""
        try:
            from agent.cache.redis_client import get_redis_client
            redis_client = await get_redis_client()
            cached = await redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
        except Exception as e:
            logger.warning(f"Cache get error: {e}")
        return None
    
    async def _set_cached_data(self, cache_key: str, data: Any) -> None:
        """Set data in Redis cache"""
        try:
            from agent.cache.redis_client import get_redis_client
            redis_client = await get_redis_client()
            await redis_client.setex(cache_key, self.cache_ttl, json.dumps(data, default=str))
        except Exception as e:
            logger.warning(f"Cache set error: {e}")
    
    async def fetch_ohlcv_data_cached(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Fetch OHLCV data with caching"""
        start_time = time.time()
        cache_key = self._get_cache_key("ohlcv", symbol, timeframe)
        cached_data = await self._get_cached_data(cache_key)
        
        if cached_data:
            duration = time.time() - start_time
            logger.info(f"📦 Using cached OHLCV data for {symbol} @ {timeframe} ({duration:.3f}s)")
            return pd.DataFrame(cached_data)
        
        # Fetch fresh data
        ohlcv = self.fetch_market_data(symbol, timeframe)
        if not ohlcv.empty:
            await self._set_cached_data(cache_key, ohlcv.to_dict('records'))
        duration = time.time() - start_time
        logger.info(f"🔄 Fresh OHLCV data fetched for {symbol} @ {timeframe} ({duration:.3f}s)")
        return ohlcv
    
    async def calculate_phase3_technical_indicators_cached(self, ohlcv: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Calculate technical indicators with caching"""
        cache_key = self._get_cache_key("tech_indicators", symbol, str(ohlcv.index[-1]))
        cached_data = await self._get_cached_data(cache_key)
        
        if cached_data:
            logger.info(f"📦 Using cached technical indicators for {symbol}")
            return cached_data
        
        # Calculate fresh indicators
        indicators = self.calculate_phase3_technical_indicators(ohlcv)
        await self._set_cached_data(cache_key, indicators)
        return indicators
    
    async def analyze_advanced_regime_detection_cached(self, ohlcv: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Analyze regime detection with caching"""
        cache_key = self._get_cache_key("regime", symbol, str(ohlcv.index[-1]))
        cached_data = await self._get_cached_data(cache_key)
        
        if cached_data:
            logger.info(f"📦 Using cached regime detection for {symbol}")
            return cached_data
        
        # Calculate fresh regime data
        regime_data = self.analyze_advanced_regime_detection(ohlcv)
        await self._set_cached_data(cache_key, regime_data)
        return regime_data
    
    async def analyze_advanced_rsi_variants_cached(self, ohlcv: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Analyze RSI variants with caching"""
        cache_key = self._get_cache_key("rsi_variants", symbol, str(ohlcv.index[-1]))
        cached_data = await self._get_cached_data(cache_key)
        
        if cached_data:
            logger.info(f"📦 Using cached RSI variants for {symbol}")
            return cached_data
        
        # Calculate fresh RSI data
        rsi_data = self.analyze_advanced_rsi_variants(ohlcv)
        await self._set_cached_data(cache_key, rsi_data)
        return rsi_data
    
    async def analyze_phase3_sentiment_async(self, symbol: str) -> float:
        """Optimized async sentiment analysis with parallel data fetching and caching"""
        start_time = time.time()
        logger.info(f"🚀 Starting optimized sentiment analysis for {symbol}")
        
        try:
            # Initialize sentiment analyzer if needed
            if self.sentiment_analyzer is None:
                self.sentiment_analyzer = SentimentAnalyzer("ProsusAI/finbert")
            
            # Get configuration
            config = load_config_from_env()
            
            # Use the new optimized async sentiment analysis
            sentiment_score = await self.sentiment_analyzer.analyze_sentiment_async(
                symbol=symbol,
                rss_feeds=config.sentiment_analysis.rss_feeds,
                reddit_subreddits=config.sentiment_analysis.reddit_subreddits,
                rss_max_headlines=config.sentiment_analysis.rss_max_headlines_per_feed,
                reddit_max_posts=config.sentiment_analysis.reddit_max_posts_per_subreddit,
                rss_hours_back=6,
                reddit_hours_back=6
            )
            
            duration = time.time() - start_time
            logger.info(f"✅ Optimized sentiment analysis completed for {symbol} in {duration:.3f}s (score: {sentiment_score:.4f})")
            return sentiment_score
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"❌ Error in optimized sentiment analysis for {symbol} after {duration:.3f}s: {e}")
            return 0.0
    
    # Add cached versions for all other analysis methods
    async def analyze_market_microstructure_cached(self, ohlcv: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        cache_key = self._get_cache_key("microstructure", symbol, str(ohlcv.index[-1]))
        cached_data = await self._get_cached_data(cache_key)
        if cached_data:
            return cached_data
        data = self.analyze_market_microstructure(ohlcv)
        await self._set_cached_data(cache_key, data)
        return data
    
    async def analyze_bollinger_bands_cached(self, ohlcv: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        cache_key = self._get_cache_key("bollinger", symbol, str(ohlcv.index[-1]))
        cached_data = await self._get_cached_data(cache_key)
        if cached_data:
            return cached_data
        data = self.analyze_bollinger_bands(ohlcv)
        await self._set_cached_data(cache_key, data)
        return data
    
    async def analyze_vwap_enhanced_cached(self, ohlcv: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        cache_key = self._get_cache_key("vwap", symbol, str(ohlcv.index[-1]))
        cached_data = await self._get_cached_data(cache_key)
        if cached_data:
            return cached_data
        data = self.analyze_vwap_enhanced(ohlcv)
        await self._set_cached_data(cache_key, data)
        return data
    
    async def analyze_volume_indicators_cached(self, ohlcv: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        cache_key = self._get_cache_key("volume", symbol, str(ohlcv.index[-1]))
        cached_data = await self._get_cached_data(cache_key)
        if cached_data:
            return cached_data
        data = self.analyze_volume_indicators(ohlcv)
        await self._set_cached_data(cache_key, data)
        return data
    
    async def analyze_moving_averages_enhanced_cached(self, ohlcv: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        cache_key = self._get_cache_key("ma", symbol, str(ohlcv.index[-1]))
        cached_data = await self._get_cached_data(cache_key)
        if cached_data:
            return cached_data
        data = self.analyze_moving_averages_enhanced(ohlcv)
        await self._set_cached_data(cache_key, data)
        return data
    
    async def analyze_keltner_channels_cached(self, ohlcv: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        cache_key = self._get_cache_key("keltner", symbol, str(ohlcv.index[-1]))
        cached_data = await self._get_cached_data(cache_key)
        if cached_data:
            return cached_data
        data = self.analyze_keltner_channels(ohlcv)
        await self._set_cached_data(cache_key, data)
        return data
    
    async def analyze_multi_timeframe_cached(self, symbol: str) -> Dict[str, Any]:
        cache_key = self._get_cache_key("multi_timeframe", symbol)
        cached_data = await self._get_cached_data(cache_key)
        if cached_data:
            return cached_data
        data = self.analyze_multi_timeframe(symbol)
        await self._set_cached_data(cache_key, data)
        return data
    
    async def analyze_cross_asset_correlation_cached(self, symbol: str) -> Dict[str, Any]:
        cache_key = self._get_cache_key("correlation", symbol)
        cached_data = await self._get_cached_data(cache_key)
        if cached_data:
            return cached_data
        data = self.analyze_cross_asset_correlation(symbol)
        await self._set_cached_data(cache_key, data)
        return data
    
    async def analyze_btc_dominance_cached(self) -> Dict[str, Any]:
        cache_key = self._get_cache_key("btc_dominance")
        cached_data = await self._get_cached_data(cache_key)
        if cached_data:
            return cached_data
        data = self.analyze_btc_dominance()
        await self._set_cached_data(cache_key, data)
        return data
    
    async def analyze_market_wide_sentiment_cached(self) -> Dict[str, Any]:
        cache_key = self._get_cache_key("market_sentiment")
        cached_data = await self._get_cached_data(cache_key)
        if cached_data:
            return cached_data
        data = self.analyze_market_wide_sentiment()
        await self._set_cached_data(cache_key, data)
        return data

    async def generate_phase3_signal_shared(self, symbol: str, timeframe: str, username: str, output_level: str = "full", 
                                           buy_threshold: Optional[float] = None, sell_threshold: Optional[float] = None,
                                           technical_weight: Optional[float] = None, sentiment_weight: Optional[float] = None) -> Phase3TradingSignal:
        """Shared Phase3 signal generation function that can be used by both manual generation and monitoring"""
        total_start_time = time.time()
        logger.info(f"🚀 Starting Phase 3 signal generation for {symbol} @ {timeframe} (user: {username}, output: {output_level})")
        
        # Log custom parameters if provided
        if any([buy_threshold, sell_threshold, technical_weight, sentiment_weight]):
            logger.info("🎛️ Custom parameters provided:")
            if buy_threshold is not None:
                logger.info(f"   📈 Buy Threshold: {buy_threshold:.4f}")
            if sell_threshold is not None:
                logger.info(f"   📉 Sell Threshold: {sell_threshold:.4f}")
            if technical_weight is not None:
                logger.info(f"   🔧 Technical Weight: {technical_weight:.4f}")
            if sentiment_weight is not None:
                logger.info(f"   💭 Sentiment Weight: {sentiment_weight:.4f}")
        else:
            logger.info("🎛️ Using default configuration parameters")
        
        try:
            # Step 1: Fetch OHLCV data (cached)
            step1_start = time.time()
            ohlcv = await self.fetch_ohlcv_data_cached(symbol, timeframe)
            step1_duration = time.time() - step1_start
            candles_count = len(ohlcv) if ohlcv is not None else 0
            first_ts = str(ohlcv.index[0]) if candles_count else 'n/a'
            last_ts = str(ohlcv.index[-1]) if candles_count else 'n/a'
            last_close = float(ohlcv['close'].iloc[-1]) if candles_count else float('nan')
            try:
                logger.info(
                    "⏱️  STEP 1 (Data Fetch): %.3fs | candles=%s, timeframe=%s, first=%s, last=%s, last_close=%.2f",
                    step1_duration, candles_count, timeframe, first_ts, last_ts, last_close
                )
            except Exception:
                logger.info(f"⏱️  STEP 1 (Data Fetch): {step1_duration:.3f}s")

            # Step 2-5: Parallel analysis (cached)
            step2_start = time.time()
            try:
                (
                    technical_indicators,
                    regime_data,
                    rsi_data,
                    sentiment_score,
                ) = await asyncio.gather(
                    self.calculate_phase3_technical_indicators_cached(ohlcv, symbol),
                    self.analyze_advanced_regime_detection_cached(ohlcv, symbol),
                    self.analyze_advanced_rsi_variants_cached(ohlcv, symbol),
                    self.analyze_phase3_sentiment_async(symbol),
                )
                try:
                    # Log sentiment score
                    logger.info("🔎 SENTIMENT_SCORE: %s", sentiment_score)
                    
                    # Log complete technical_indicators content
                    logger.info("🔧 TECHNICAL_INDICATORS COMPLETE CONTENT:")
                    if isinstance(technical_indicators, dict):
                        for key, value in technical_indicators.items():
                            if hasattr(value, 'iloc') and len(value) > 0:
                                # Handle pandas Series
                                logger.info("   %s: %s (last value: %s)", key, type(value).__name__, value.iloc[-1])
                            else:
                                logger.info("   %s: %s", key, value)
                    else:
                        logger.info("   Type: %s, Value: %s", type(technical_indicators), technical_indicators)
                    
                    # Log complete regime_data content
                    logger.info("🏳️ REGIME_DATA COMPLETE CONTENT:")
                    if isinstance(regime_data, dict):
                        for key, value in regime_data.items():
                            if hasattr(value, 'iloc') and len(value) > 0:
                                # Handle pandas Series
                                logger.info("   %s: %s (last value: %s)", key, type(value).__name__, value.iloc[-1])
                            else:
                                logger.info("   %s: %s", key, value)
                    else:
                        logger.info("   Type: %s, Value: %s", type(regime_data), regime_data)
                    
                    # Log complete rsi_data content
                    logger.info("📐 RSI_DATA COMPLETE CONTENT:")
                    if isinstance(rsi_data, dict):
                        for key, value in rsi_data.items():
                            if hasattr(value, 'iloc') and len(value) > 0:
                                # Handle pandas Series
                                logger.info("   %s: %s (last value: %s)", key, type(value).__name__, value.iloc[-1])
                            else:
                                logger.info("   %s: %s", key, value)
                    else:
                        logger.info("   Type: %s, Value: %s", type(rsi_data), rsi_data)
                        
                except Exception as e:
                    logger.error("❌ Error logging parameter content: %s", e)
            except Exception as e:
                logger.error(f"❌ Error in parallel analysis: {e}")
                # Fallback to default values
                technical_indicators = {}
                regime_data = {'regime_classification': 'unknown', 'confidence': 0.5, 'trend_strength': 0.0}
                rsi_data = {'alignment': 'neutral', 'momentum': 0.0}
                sentiment_score = 0.0
            step2_duration = time.time() - step2_start
            logger.info(f"⏱️  STEP 2-5 (Parallel Analysis): {step2_duration:.3f}s")

            # Add current price to technical indicators for better analysis
            if 'close' in ohlcv.columns and len(ohlcv) > 0:
                current_price = ohlcv['close'].iloc[-1]
                technical_indicators['current_price'] = current_price
                logger.info(f"💰 Current Price: {current_price:.2f}")
            
            # Add current values for UI compatibility (scalar values instead of arrays)
            try:
                if 'rsi_14' in technical_indicators and not technical_indicators['rsi_14'].empty:
                    # Store the current RSI value for UI display (overwrite the array with current value)
                    current_rsi = float(technical_indicators['rsi_14'].iloc[-1])
                    technical_indicators['rsi_14_series'] = technical_indicators['rsi_14']  # Keep original series
                    technical_indicators['rsi_14'] = current_rsi  # Overwrite with current value
                if 'macd' in technical_indicators and not technical_indicators['macd'].empty:
                    # Store the current MACD value for UI display (overwrite the array with current value)
                    current_macd = float(technical_indicators['macd'].iloc[-1])
                    technical_indicators['macd_series'] = technical_indicators['macd']  # Keep original series
                    technical_indicators['macd'] = current_macd  # Overwrite with current value
                if 'atr' in technical_indicators and not technical_indicators['atr'].empty:
                    # Store the current ATR value for UI display (overwrite the array with current value)
                    current_atr = float(technical_indicators['atr'].iloc[-1])
                    technical_indicators['atr_series'] = technical_indicators['atr']  # Keep original series
                    technical_indicators['atr'] = current_atr  # Overwrite with current value
                logger.info(f"📊 Current values: RSI14={technical_indicators.get('rsi_14', 'N/A')}, MACD={technical_indicators.get('macd', 'N/A')}, ATR={technical_indicators.get('atr', 'N/A')}")
            except Exception as e:
                logger.warning(f"⚠️ Error extracting current indicator values: {e}")

            # Step 6: Calculate technical score
            step6_start = time.time()
            technical_score = self.calculate_phase3_technical_score(technical_indicators, regime_data, rsi_data)
            step6_duration = time.time() - step6_start
            try:
                logger.info(
                    "⏱️  STEP 6 (Technical Score): %.3fs | score=%.4f, rsi=%s, ema20=%s, ema50=%s, atr=%s, regime=%s, rsi_align=%s",
                    step6_duration,
                    float(technical_score),
                    technical_indicators.get('rsi') if isinstance(technical_indicators, dict) else None,
                    technical_indicators.get('ema_20') if isinstance(technical_indicators, dict) else None,
                    technical_indicators.get('ema_50') if isinstance(technical_indicators, dict) else None,
                    technical_indicators.get('atr') if isinstance(technical_indicators, dict) else None,
                    regime_data.get('regime_classification') if isinstance(regime_data, dict) else None,
                    rsi_data.get('alignment') if isinstance(rsi_data, dict) else None,
                )
            except Exception:
                logger.info(f"⏱️  STEP 6 (Technical Score): {step6_duration:.3f}s")

            # Step 7-8: Phase 1 & 2 analysis (cached)
            step7_start = time.time()
            (
                microstructure,
                bollinger_bands,
                vwap_analysis,
                volume_indicators,
                moving_averages,
                keltner_channels,
                multi_timeframe,
                cross_asset_correlation,
                btc_dominance,
                market_wide_sentiment,
            ) = await asyncio.gather(
                self.analyze_market_microstructure_cached(ohlcv, symbol),
                self.analyze_bollinger_bands_cached(ohlcv, symbol),
                self.analyze_vwap_enhanced_cached(ohlcv, symbol),
                self.analyze_volume_indicators_cached(ohlcv, symbol),
                self.analyze_moving_averages_enhanced_cached(ohlcv, symbol),
                self.analyze_keltner_channels_cached(ohlcv, symbol),
                self.analyze_multi_timeframe_cached(symbol),
                self.analyze_cross_asset_correlation_cached(symbol),
                self.analyze_btc_dominance_cached(),
                self.analyze_market_wide_sentiment_cached(),
            )
            step7_duration = time.time() - step7_start
            try:
                logger.info(
                    "⏱️  STEP 7-8 (Phase 1/2): %.3fs | micro:last_tick=%s, boll:squeeze=%s, vwap:dev=%s, vol:obv=%s, ma:last_cross=%s, kelt:last_break=%s, mtf:trend=%s/cons=%.2f, btc_dom=%.2f, mkt_sent=%.3f",
                    step7_duration,
                    microstructure.get('last_tick') if isinstance(microstructure, dict) else None,
                    bollinger_bands.get('squeeze') if isinstance(bollinger_bands, dict) else None,
                    vwap_analysis.get('deviation') if isinstance(vwap_analysis, dict) else None,
                    (volume_indicators.get('obv_trend') if isinstance(volume_indicators, dict) else None),
                    (moving_averages.get('crossovers', {}).get('last') if isinstance(moving_averages, dict) else None),
                    (keltner_channels.get('last_breakout_up') if isinstance(keltner_channels, dict) else None),
                    (multi_timeframe.get('overall_trend') if isinstance(multi_timeframe, dict) else None),
                    float(multi_timeframe.get('trend_consensus', 0.0)) if isinstance(multi_timeframe, dict) else 0.0,
                    float(btc_dominance.get('btc_dominance', 0.0)) if isinstance(btc_dominance, dict) else 0.0,
                    float(market_wide_sentiment.get('market_sentiment_score', 0.0)) if isinstance(market_wide_sentiment, dict) else 0.0,
                )
            except Exception:
                logger.info(f"⏱️  STEP 7-8 (Phase 1/2 Analysis): {step7_duration:.3f}s")

            # Step 9: Calculate fused score
            step9_start = time.time()
            fused_score = self.calculate_phase3_fused_score(technical_score, sentiment_score, technical_weight, sentiment_weight)
            try:
                from agent.config import load_config_from_env as _load_cfg
                _cfg = _load_cfg()
                logger.info(
                    "🧮 Fused score details → tech_weight=%.2f, sentiment_weight=%.2f, technical=%.4f, sentiment=%.4f, fused=%.4f",
                    float(_cfg.thresholds.technical_weight),
                    float(_cfg.thresholds.sentiment_weight),
                    float(technical_score),
                    float(sentiment_score),
                    float(fused_score)
                )
            except Exception:
                pass
            step9_duration = time.time() - step9_start
            logger.info(f"⏱️  STEP 9 (Fused Score): {step9_duration:.3f}s")

            # Step 10: Determine signal type and confidence
            signal_type, confidence = self.determine_signal_type_and_confidence(fused_score, buy_threshold, sell_threshold)
            try:
                conf_components = []
                if regime_data.get('regime_classification') == 'trending':
                    conf_components.append('regime(+trend)')
                if rsi_data.get('alignment') in ['bullish', 'bearish']:
                    conf_components.append('rsi(alignment)')
                if abs(technical_score) > 0.5:
                    conf_components.append('technical(>0.5)')
                logger.info("🎛️ Confidence factors → %s | final=%.2f%%", ','.join(conf_components) if conf_components else 'none', float(confidence) * 100.0)
            except Exception:
                pass
            
            # Step 11: Calculate risk management
            risk_metrics = self.calculate_enhanced_risk_management(ohlcv, technical_indicators, regime_data)
            stop_loss, take_profit = self.calculate_volatility_adjusted_stops(ohlcv, signal_type, risk_metrics)
            try:
                logger.info(
                    "🛡️ Risk metrics → vol=%.4f, var95=%.4f, sharpe=%.3f, mdd=%.4f | SL=%.2f, TP=%.2f",
                    float(risk_metrics.get('volatility', 0.0)),
                    float(risk_metrics.get('var_95', 0.0)),
                    float(risk_metrics.get('sharpe_ratio', 0.0)),
                    float(risk_metrics.get('max_drawdown', 0.0)),
                    float(stop_loss),
                    float(take_profit)
                )
            except Exception:
                pass
            
            # Step 12: Calculate position sizing
            position_sizing = self.compute_dynamic_position_sizing(risk_metrics, confidence, signal_type)
            try:
                logger.info(
                    "📏 Position sizing → recommended=%.2f%%, risk_per_trade=%.2f%%, max=%.2f%%",
                    float(position_sizing.get('recommended_size', 0.0)) * 100.0,
                    float(position_sizing.get('risk_per_trade', 0.0)) * 100.0,
                    float(position_sizing.get('max_position', 0.0)) * 100.0,
                )
            except Exception:
                pass 

            # Final summary block
            try:
                total_duration = time.time() - total_start_time
                logger.info("=" * 80)
                logger.info(
                    "✅ Phase 3 summary → %s %s @ %s | fused=%.4f, tech=%.4f, sent=%.4f, conf=%.2f%% | SL=%.2f TP=%.2f | total=%.3fs",
                    signal_type,
                    symbol,
                    timeframe,
                    float(fused_score),
                    float(technical_score),
                    float(sentiment_score),
                    float(confidence) * 100.0,
                    float(stop_loss),
                    float(take_profit),
                    total_duration,
                )
                logger.info("=" * 80)
            except Exception:
                pass
            
            # Step 13: Create Phase3 signal (ensure all data is serializable)
            phase3_signal = Phase3TradingSignal(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=datetime.now(pytz.timezone('Asia/Jerusalem')),
                signal_type=signal_type,
                technical_score=float(technical_score),
                sentiment_score=float(sentiment_score),
                fused_score=float(fused_score),
                confidence=float(confidence),
                stop_loss=float(stop_loss),
                take_profit=float(take_profit),
                reasoning=self.generate_phase3_reasoning(signal_type, technical_score, sentiment_score, fused_score, regime_data, rsi_data),
                # Phase 3 Advanced Features
                regime_detection=to_serializable(regime_data),
                advanced_rsi=to_serializable(rsi_data),
                position_sizing=to_serializable(position_sizing),
                volatility_adjusted_stops={'stop_loss': float(stop_loss), 'take_profit': float(take_profit)},
                risk_metrics=to_serializable(risk_metrics),
                technical_indicators=to_serializable(technical_indicators),
                market_microstructure=to_serializable(microstructure),
                # Phase 1 Enhanced Technical Analysis
                bollinger_bands=to_serializable(bollinger_bands),
                vwap_analysis=to_serializable(vwap_analysis),
                volume_indicators=to_serializable(volume_indicators),
                moving_averages=to_serializable(moving_averages),
                keltner_channels=to_serializable(keltner_channels),
                # Phase 2 Multi-Timeframe Analysis
                multi_timeframe=to_serializable(multi_timeframe),
                cross_asset_correlation=to_serializable(cross_asset_correlation),
                btc_dominance=to_serializable(btc_dominance),
                market_wide_sentiment=to_serializable(market_wide_sentiment),
                # Applied Parameters (for UI display)
                applied_buy_threshold=buy_threshold,
                applied_sell_threshold=sell_threshold,
                applied_tech_weight=technical_weight,
                applied_sentiment_weight=sentiment_weight,
                # Configuration
                output_level=output_level,
                username=username
            )
            
            # Step 14: Store signal (await to ensure it's visible in recent tips immediately)
            logger.info("💾 Storing signal (await)")
            await self.store_signal(phase3_signal, username)
            # Step 15: Send notification in background (non-blocking)
            asyncio.create_task(self.send_telegram_notification(phase3_signal, username))

            total_duration = time.time() - total_start_time
            logger.info(f"⏱️  TOTAL EXECUTION TIME: {total_duration:.3f}s")
            logger.info(f"✅ Phase 3 signal generated successfully: {signal_type} (confidence: {confidence:.3f})")
            
            # Log comprehensive response details
            logger.info("=" * 80)
            logger.info("📊 PHASE 3 SIGNAL RESPONSE DETAILS")
            logger.info("=" * 80)
            logger.info(f"🎯 Signal Type: {signal_type}")
            logger.info(f"📈 Symbol: {phase3_signal.symbol}")
            logger.info(f"⏰ Timeframe: {phase3_signal.timeframe}")
            logger.info(f"📅 Timestamp: {phase3_signal.timestamp}")
            logger.info(f"👤 Username: {phase3_signal.username}")
            logger.info(f"🎲 Confidence: {phase3_signal.confidence:.4f} ({phase3_signal.confidence*100:.2f}%)")
            logger.info(f"🔗 Fused Score: {phase3_signal.fused_score:.4f}")
            logger.info(f"🔧 Technical Score: {phase3_signal.technical_score:.4f}")
            logger.info(f"💭 Sentiment Score: {phase3_signal.sentiment_score:.4f}")
            logger.info(f"🛡️ Stop Loss: {phase3_signal.stop_loss:.2f}")
            logger.info(f"🎯 Take Profit: {phase3_signal.take_profit:.2f}")
            logger.info(f"💰 Current Price: {phase3_signal.technical_indicators.get('current_price', 'N/A')}")
            # Safely extract technical indicator values
            try:
                rsi_data = phase3_signal.technical_indicators.get('rsi_14', pd.Series([50]))
                rsi_value = rsi_data.iloc[-1] if hasattr(rsi_data, 'iloc') and len(rsi_data) > 0 else (rsi_data[-1] if isinstance(rsi_data, list) and len(rsi_data) > 0 else 'N/A')
                logger.info(f"📊 RSI: {rsi_value}")
            except Exception:
                logger.info(f"📊 RSI: N/A")
            
            try:
                macd_data = phase3_signal.technical_indicators.get('macd', pd.Series([0]))
                macd_value = macd_data.iloc[-1] if hasattr(macd_data, 'iloc') and len(macd_data) > 0 else (macd_data[-1] if isinstance(macd_data, list) and len(macd_data) > 0 else 'N/A')
                logger.info(f"📈 MACD: {macd_value}")
            except Exception:
                logger.info(f"📈 MACD: N/A")
            
            try:
                atr_data = phase3_signal.technical_indicators.get('atr', pd.Series([0]))
                atr_value = atr_data.iloc[-1] if hasattr(atr_data, 'iloc') and len(atr_data) > 0 else (atr_data[-1] if isinstance(atr_data, list) and len(atr_data) > 0 else 'N/A')
                logger.info(f"📉 ATR: {atr_value}")
            except Exception:
                logger.info(f"📉 ATR: N/A")
            # Log applied parameters (these are stored in the signal generation process)
            logger.info(f"🎛️ Applied Buy Threshold: {buy_threshold if buy_threshold is not None else 'default'}")
            logger.info(f"🎛️ Applied Sell Threshold: {sell_threshold if sell_threshold is not None else 'default'}")
            logger.info(f"🎛️ Applied Technical Weight: {technical_weight if technical_weight is not None else 'default'}")
            logger.info(f"🎛️ Applied Sentiment Weight: {sentiment_weight if sentiment_weight is not None else 'default'}")
            logger.info(f"📊 Regime Classification: {phase3_signal.regime_detection.get('regime_classification', 'N/A')}")
            logger.info(f"📊 Trend Strength: {phase3_signal.regime_detection.get('trend_strength', 'N/A')}")
            logger.info(f"📊 RSI Alignment: {phase3_signal.advanced_rsi.get('alignment', 'N/A')}")
            logger.info(f"📊 Position Size: {phase3_signal.position_sizing.get('recommended_size', 'N/A')}")
            logger.info(f"📊 Risk Level: {phase3_signal.risk_metrics.get('risk_level', 'N/A')}")
            logger.info(f"📊 Volatility: {phase3_signal.risk_metrics.get('volatility', 'N/A')}")
            logger.info(f"📊 VaR 95%: {phase3_signal.risk_metrics.get('var_95', 'N/A')}")
            logger.info(f"📊 Sharpe Ratio: {phase3_signal.risk_metrics.get('sharpe_ratio', 'N/A')}")
            logger.info(f"📊 Multi-timeframe Trend: {phase3_signal.multi_timeframe.get('overall_trend', 'N/A')}")
            logger.info(f"📊 Trend Consensus: {phase3_signal.multi_timeframe.get('trend_consensus', 'N/A')}")
            # Log sentiment information from available fields
            logger.info(f"💭 Sentiment Score: {phase3_signal.sentiment_score:.4f}")
            logger.info(f"💭 Market-wide Sentiment: {phase3_signal.market_wide_sentiment.get('overall_sentiment', 'N/A')}")
            logger.info(f"💭 Market Sentiment Score: {phase3_signal.market_wide_sentiment.get('market_sentiment_score', 'N/A')}")
            # Note: Detailed sentiment analysis and reasoning are not stored in the model
            logger.info("=" * 80)
            logger.info("📊 END OF PHASE 3 SIGNAL RESPONSE")
            logger.info("=" * 80)
            
            return phase3_signal
            
        except Exception as e:
            logger.error(f"❌ Error generating Phase 3 signal: {e}")
            raise

    async def generate_phase3_signal(self, request: Phase3SignalRequest, username: str, output_level: str = "full") -> Phase3TradingSignal:
        """
        Generate a Phase 3 advanced trading signal - Completely Independent
        
        Args:
            request: Signal generation request
            username: Username of the user requesting the signal
            output_level: Level of detail in output ("full", "summary", "minimal")
        
        Returns:
            Phase3TradingSignal: Complete Phase 3 signal with all advanced features
        """
        return await self.generate_phase3_signal_shared(
            request.symbol, 
            request.timeframe, 
            username, 
            output_level,
            request.buy_threshold,
            request.sell_threshold,
            request.technical_weight,
            request.sentiment_weight
        )

# Create global instance
phase3_signal_generator = Phase3SignalGenerator()
