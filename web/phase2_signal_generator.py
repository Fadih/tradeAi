#!/usr/bin/env python3
"""
Phase 2 Enhanced Signal Generation Module
Multi-timeframe analysis, cross-asset correlation, and advanced market microstructure
Based on ChatGPT recommendations for professional trading systems
"""

import logging
import random
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import pytz
import requests
import pandas as pd
import asyncio

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

# Import Phase 1 enhanced generator
from .enhanced_signal_generator import EnhancedSignalGenerator, EnhancedTradingSignal, SignalRequest

# Configure logging
logger = logging.getLogger(__name__)

# Israel timezone
ISRAEL_TZ = pytz.timezone('Asia/Jerusalem')

def get_israel_time():
    """Get current time in Israel timezone with microsecond precision"""
    return datetime.now(ISRAEL_TZ)


class Phase2TradingSignal(EnhancedTradingSignal):
    """Phase 2 enhanced trading signal with multi-timeframe and cross-asset analysis"""
    
    # Multi-timeframe analysis
    multi_timeframe_scores: Optional[Dict[str, float]] = None  # Scores for each timeframe
    timeframe_consensus: Optional[str] = None  # Overall timeframe consensus
    timeframe_strength: Optional[float] = None  # Strength of consensus
    
    # Cross-asset correlation
    btc_correlation: Optional[float] = None  # Correlation with BTC
    eth_correlation: Optional[float] = None  # Correlation with ETH
    btc_dominance: Optional[float] = None  # BTC dominance at signal time
    
    # Returns analysis
    returns_analysis: Optional[Dict[str, float]] = None  # Various return metrics
    
    # Volatility metrics
    volatility_metrics: Optional[Dict[str, float]] = None  # Volatility analysis
    
    # Market microstructure
    gaps_micro_moves: Optional[Dict[str, Any]] = None  # Gaps and micro-moves detection


class Phase2SignalGenerator(EnhancedSignalGenerator):
    """Phase 2 enhanced signal generator with multi-timeframe and cross-asset analysis"""
    
    def __init__(self):
        super().__init__()
        self.timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
        self.timeframe_weights = {
            '1m': 0.05,   # Very short-term
            '5m': 0.15,   # Short-term
            '15m': 0.20,  # Short-term
            '1h': 0.25,   # Medium-term
            '4h': 0.20,   # Medium-term
            '1d': 0.15    # Long-term
        }
        logger.info("Phase 2 Signal Generator initialized")
    
    # ==================== PHASE 2: MULTI-TIMEFRAME ANALYSIS ====================
    
    async def analyze_multi_timeframe(self, symbol: str) -> Dict[str, float]:
        """
        Analyze signal across multiple timeframes
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary with scores for each timeframe
        """
        logger.info(f"🔧 PHASE 2: Analyzing multi-timeframe for {symbol}")
        
        timeframe_scores = {}
        
        # Analyze each timeframe
        for timeframe in self.timeframes:
            try:
                logger.info(f"📊 Analyzing {symbol} @ {timeframe}")
                
                # Fetch data for this timeframe
                ohlcv = self.fetch_market_data(symbol, timeframe)
                
                # Calculate technical indicators for this timeframe
                indicators = self.calculate_enhanced_technical_indicators(ohlcv)
                
                # Detect market regime
                regime, volatility_state = self.detect_market_regime(indicators)
                
                # Calculate technical score for this timeframe
                tech_score = self.calculate_enhanced_technical_score(indicators, regime)
                
                timeframe_scores[timeframe] = tech_score
                
                logger.info(f"✅ {timeframe}: Technical Score = {tech_score:.4f}, Regime = {regime}")
                
            except Exception as e:
                logger.error(f"Failed to analyze {timeframe} for {symbol}: {str(e)}")
                timeframe_scores[timeframe] = 0.0
        
        logger.info(f"✅ Multi-timeframe analysis completed for {symbol}")
        return timeframe_scores
    
    def calculate_timeframe_consensus(self, timeframe_scores: Dict[str, float]) -> Tuple[str, float]:
        """
        Calculate timeframe consensus and strength
        
        Args:
            timeframe_scores: Scores for each timeframe
            
        Returns:
            Tuple of (consensus, strength)
        """
        logger.info("🔧 Calculating timeframe consensus")
        
        # Calculate weighted average
        weighted_sum = 0.0
        total_weight = 0.0
        
        for timeframe, score in timeframe_scores.items():
            weight = self.timeframe_weights.get(timeframe, 0.0)
            weighted_sum += score * weight
            total_weight += weight
        
        if total_weight == 0:
            return "neutral", 0.0
        
        consensus_score = weighted_sum / total_weight
        
        # Determine consensus direction
        if consensus_score > 0.1:
            consensus = "bullish"
        elif consensus_score < -0.1:
            consensus = "bearish"
        else:
            consensus = "neutral"
        
        # Calculate strength (how consistent the signals are)
        positive_signals = sum(1 for score in timeframe_scores.values() if score > 0.1)
        negative_signals = sum(1 for score in timeframe_scores.values() if score < -0.1)
        total_signals = len(timeframe_scores)
        
        if total_signals == 0:
            strength = 0.0
        else:
            strength = max(positive_signals, negative_signals) / total_signals
        
        logger.info(f"✅ Timeframe consensus: {consensus} (strength: {strength:.2f})")
        
        return consensus, strength
    
    # ==================== PHASE 2: CROSS-ASSET CORRELATION ====================
    
    async def analyze_cross_asset_correlation(self, symbol: str) -> Dict[str, float]:
        """
        Analyze correlation with major assets (BTC, ETH)
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary with correlation metrics
        """
        logger.info(f"🔧 PHASE 2: Analyzing cross-asset correlation for {symbol}")
        
        correlation_metrics = {}
        
        try:
            # Get current symbol data
            current_ohlcv = self.fetch_market_data(symbol, '1h')
            current_returns = current_ohlcv['close'].pct_change().dropna()
            
            # Analyze BTC correlation
            try:
                btc_ohlcv = self.fetch_market_data('BTC/USDT', '1h')
                btc_returns = btc_ohlcv['close'].pct_change().dropna()
                
                # Align time series
                min_length = min(len(current_returns), len(btc_returns))
                if min_length > 10:
                    correlation_1h = current_returns.iloc[-min_length:].corr(btc_returns.iloc[-min_length:])
                    correlation_metrics['btc_correlation_1h'] = correlation_1h if not pd.isna(correlation_1h) else 0.0
                else:
                    correlation_metrics['btc_correlation_1h'] = 0.0
                    
            except Exception as e:
                logger.error(f"Failed to calculate BTC correlation: {str(e)}")
                correlation_metrics['btc_correlation_1h'] = 0.0
            
            # Analyze ETH correlation
            try:
                eth_ohlcv = self.fetch_market_data('ETH/USDT', '1h')
                eth_returns = eth_ohlcv['close'].pct_change().dropna()
                
                # Align time series
                min_length = min(len(current_returns), len(eth_returns))
                if min_length > 10:
                    correlation_4h = current_returns.iloc[-min_length:].corr(eth_returns.iloc[-min_length:])
                    correlation_metrics['eth_correlation_1h'] = correlation_4h if not pd.isna(correlation_4h) else 0.0
                else:
                    correlation_metrics['eth_correlation_1h'] = 0.0
                    
            except Exception as e:
                logger.error(f"Failed to calculate ETH correlation: {str(e)}")
                correlation_metrics['eth_correlation_1h'] = 0.0
            
            # Calculate BTC dominance (simplified - in real implementation, you'd fetch this from an API)
            try:
                btc_market_cap = btc_ohlcv['close'].iloc[-1] * 21000000  # Approximate BTC supply
                eth_market_cap = eth_ohlcv['close'].iloc[-1] * 120000000  # Approximate ETH supply
                total_crypto_market_cap = btc_market_cap + eth_market_cap
                btc_dominance = (btc_market_cap / total_crypto_market_cap) * 100
                correlation_metrics['btc_dominance'] = btc_dominance
            except:
                correlation_metrics['btc_dominance'] = 45.0  # Default value
            
            logger.info(f"✅ Cross-asset correlation analysis completed")
            
        except Exception as e:
            logger.error(f"Failed to analyze cross-asset correlation: {str(e)}")
            correlation_metrics = {
                'btc_correlation_1h': 0.0,
                'eth_correlation_1h': 0.0,
                'btc_dominance': 45.0
            }
        
        return correlation_metrics
    
    # ==================== PHASE 2: RETURNS ANALYSIS ====================
    
    def analyze_returns(self, ohlcv: pd.DataFrame) -> Dict[str, float]:
        """
        Analyze various return metrics
        
        Args:
            ohlcv: OHLCV data
            
        Returns:
            Dictionary with return metrics
        """
        logger.info("🔧 PHASE 2: Analyzing returns")
        
        close = ohlcv['close']
        returns_metrics = {}
        
        try:
            # Simple returns
            returns_metrics['simple_return_1'] = close.pct_change(1).iloc[-1] if len(close) > 1 else 0.0
            returns_metrics['simple_return_3'] = close.pct_change(3).iloc[-1] if len(close) > 3 else 0.0
            returns_metrics['simple_return_5'] = close.pct_change(5).iloc[-1] if len(close) > 5 else 0.0
            returns_metrics['simple_return_15'] = close.pct_change(15).iloc[-1] if len(close) > 15 else 0.0
            
            # Log returns
            log_returns = np.log(close / close.shift(1)).dropna()
            returns_metrics['log_return_1'] = log_returns.iloc[-1] if len(log_returns) > 0 else 0.0
            returns_metrics['log_return_3'] = np.log(close.iloc[-1] / close.iloc[-4]) if len(close) > 3 else 0.0
            returns_metrics['log_return_5'] = np.log(close.iloc[-1] / close.iloc[-6]) if len(close) > 5 else 0.0
            returns_metrics['log_return_15'] = np.log(close.iloc[-1] / close.iloc[-16]) if len(close) > 15 else 0.0
            
            # Cumulative returns over different windows
            returns_metrics['cumulative_return_5'] = (close.iloc[-1] / close.iloc[-6] - 1) if len(close) > 5 else 0.0
            returns_metrics['cumulative_return_15'] = (close.iloc[-1] / close.iloc[-16] - 1) if len(close) > 15 else 0.0
            
            logger.info("✅ Returns analysis completed")
            
        except Exception as e:
            logger.error(f"Failed to analyze returns: {str(e)}")
            returns_metrics = {
                'simple_return_1': 0.0, 'simple_return_3': 0.0, 'simple_return_5': 0.0, 'simple_return_15': 0.0,
                'log_return_1': 0.0, 'log_return_3': 0.0, 'log_return_5': 0.0, 'log_return_15': 0.0,
                'cumulative_return_5': 0.0, 'cumulative_return_15': 0.0
            }
        
        return returns_metrics
    
    # ==================== PHASE 2: VOLATILITY METRICS ====================
    
    def analyze_volatility_metrics(self, ohlcv: pd.DataFrame) -> Dict[str, float]:
        """
        Analyze volatility metrics
        
        Args:
            ohlcv: OHLCV data
            
        Returns:
            Dictionary with volatility metrics
        """
        logger.info("🔧 PHASE 2: Analyzing volatility metrics")
        
        close = ohlcv['close']
        high = ohlcv['high']
        low = ohlcv['low']
        volatility_metrics = {}
        
        try:
            # Rolling standard deviation of log returns
            log_returns = np.log(close / close.shift(1)).dropna()
            volatility_metrics['rolling_volatility_5'] = log_returns.rolling(5).std().iloc[-1] if len(log_returns) > 4 else 0.0
            volatility_metrics['rolling_volatility_20'] = log_returns.rolling(20).std().iloc[-1] if len(log_returns) > 19 else 0.0
            
            # Realized volatility (sum of squared returns)
            volatility_metrics['realized_volatility_5'] = np.sqrt((log_returns.iloc[-5:]**2).sum()) if len(log_returns) > 4 else 0.0
            volatility_metrics['realized_volatility_20'] = np.sqrt((log_returns.iloc[-20:]**2).sum()) if len(log_returns) > 19 else 0.0
            
            # True Range and ATR
            tr = np.maximum(high - low, 
                          np.maximum(abs(high - close.shift(1)), 
                                   abs(low - close.shift(1))))
            volatility_metrics['true_range'] = tr.iloc[-1] if len(tr) > 0 else 0.0
            volatility_metrics['atr_14'] = tr.rolling(14).mean().iloc[-1] if len(tr) > 13 else 0.0
            
            # Volatility percentiles
            vol_20 = log_returns.rolling(20).std()
            if len(vol_20) > 0:
                current_vol = vol_20.iloc[-1]
                vol_percentile = (vol_20 < current_vol).sum() / len(vol_20) * 100
                volatility_metrics['volatility_percentile'] = vol_percentile
            else:
                volatility_metrics['volatility_percentile'] = 50.0
            
            logger.info("✅ Volatility metrics analysis completed")
            
        except Exception as e:
            logger.error(f"Failed to analyze volatility metrics: {str(e)}")
            volatility_metrics = {
                'rolling_volatility_5': 0.0, 'rolling_volatility_20': 0.0,
                'realized_volatility_5': 0.0, 'realized_volatility_20': 0.0,
                'true_range': 0.0, 'atr_14': 0.0, 'volatility_percentile': 50.0
            }
        
        return volatility_metrics
    
    # ==================== PHASE 2: GAPS & MICRO-MOVES ====================
    
    def detect_gaps_micro_moves(self, ohlcv: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect gaps and micro-moves
        
        Args:
            ohlcv: OHLCV data
            
        Returns:
            Dictionary with gap and micro-move detection
        """
        logger.info("🔧 PHASE 2: Detecting gaps and micro-moves")
        
        close = ohlcv['close']
        high = ohlcv['high']
        low = ohlcv['low']
        volume = ohlcv['volume']
        
        gaps_micro_moves = {}
        
        try:
            # Gap detection
            prev_close = close.shift(1)
            gap_up = int((low > prev_close).sum())
            gap_down = int((high < prev_close).sum())
            gaps_micro_moves['gap_up_count'] = gap_up
            gaps_micro_moves['gap_down_count'] = gap_down
            
            # Recent gap detection (last 5 bars)
            recent_gap_up = int((low.iloc[-5:] > prev_close.iloc[-5:]).sum())
            recent_gap_down = int((high.iloc[-5:] < prev_close.iloc[-5:]).sum())
            gaps_micro_moves['recent_gap_up'] = recent_gap_up
            gaps_micro_moves['recent_gap_down'] = recent_gap_down
            
            # Micro-trend runs (consecutive up/down moves)
            price_changes = close.diff()
            up_runs = []
            down_runs = []
            current_run = 0
            current_direction = 0
            
            for change in price_changes:
                if change > 0:
                    if current_direction == 1:
                        current_run += 1
                    else:
                        if current_run > 0:
                            down_runs.append(current_run)
                        current_run = 1
                        current_direction = 1
                elif change < 0:
                    if current_direction == -1:
                        current_run += 1
                    else:
                        if current_run > 0:
                            up_runs.append(current_run)
                        current_run = 1
                        current_direction = -1
                else:
                    if current_run > 0:
                        if current_direction == 1:
                            up_runs.append(current_run)
                        else:
                            down_runs.append(current_run)
                    current_run = 0
                    current_direction = 0
            
            gaps_micro_moves['max_up_run'] = int(max(up_runs)) if up_runs else 0
            gaps_micro_moves['max_down_run'] = int(max(down_runs)) if down_runs else 0
            gaps_micro_moves['current_run_length'] = int(current_run)
            gaps_micro_moves['current_run_direction'] = int(current_direction)
            
            # Volume spikes
            avg_volume = volume.rolling(20).mean()
            volume_spike = float(volume.iloc[-1] / avg_volume.iloc[-1]) if avg_volume.iloc[-1] > 0 else 1.0
            gaps_micro_moves['volume_spike'] = volume_spike
            
            logger.info("✅ Gaps and micro-moves detection completed")
            
        except Exception as e:
            logger.error(f"Failed to detect gaps and micro-moves: {str(e)}")
            gaps_micro_moves = {
                'gap_up_count': 0, 'gap_down_count': 0,
                'recent_gap_up': 0, 'recent_gap_down': 0,
                'max_up_run': 0, 'max_down_run': 0,
                'current_run_length': 0, 'current_run_direction': 0,
                'volume_spike': 1.0
            }
        
        return gaps_micro_moves
    
    # ==================== PHASE 2: ENHANCED TECHNICAL SCORE ====================
    
    def calculate_phase2_technical_score(self, 
                                       timeframe_scores: Dict[str, float],
                                       correlation_metrics: Dict[str, float],
                                       returns_metrics: Dict[str, float],
                                       volatility_metrics: Dict[str, float]) -> float:
        """
        Calculate Phase 2 enhanced technical score
        
        Args:
            timeframe_scores: Multi-timeframe scores
            correlation_metrics: Cross-asset correlation metrics
            returns_metrics: Returns analysis
            volatility_metrics: Volatility metrics
            
        Returns:
            Enhanced technical score
        """
        logger.info("🔧 PHASE 2: Calculating enhanced technical score")
        
        # Multi-timeframe weighted score (60% weight)
        timeframe_score = 0.0
        total_weight = 0.0
        for timeframe, score in timeframe_scores.items():
            weight = self.timeframe_weights.get(timeframe, 0.0)
            timeframe_score += score * weight
            total_weight += weight
        
        if total_weight > 0:
            timeframe_score = timeframe_score / total_weight
        
        # Cross-asset correlation adjustment (20% weight)
        btc_corr = correlation_metrics.get('btc_correlation_1h', 0.0)
        eth_corr = correlation_metrics.get('eth_correlation_1h', 0.0)
        btc_dom = correlation_metrics.get('btc_dominance', 45.0)
        
        # If high correlation with BTC and BTC is dominant, amplify the signal
        correlation_adjustment = 0.0
        if btc_corr > 0.7 and btc_dom > 50:
            correlation_adjustment = timeframe_score * 0.2
        elif btc_corr < -0.7:
            correlation_adjustment = -timeframe_score * 0.2
        
        # Returns momentum (10% weight)
        returns_score = 0.0
        if returns_metrics.get('simple_return_5', 0) > 0.02:
            returns_score = 0.1
        elif returns_metrics.get('simple_return_5', 0) < -0.02:
            returns_score = -0.1
        
        # Volatility adjustment (10% weight)
        vol_percentile = volatility_metrics.get('volatility_percentile', 50.0)
        volatility_adjustment = 0.0
        if vol_percentile > 80:  # High volatility
            volatility_adjustment = -0.05  # Reduce confidence in high volatility
        elif vol_percentile < 20:  # Low volatility
            volatility_adjustment = 0.05   # Increase confidence in low volatility
        
        # Calculate final score
        final_score = (timeframe_score * 0.6 + 
                      correlation_adjustment * 0.2 + 
                      returns_score * 0.1 + 
                      volatility_adjustment * 0.1)
        
        logger.info(f"🧮 PHASE 2 TECHNICAL SCORE BREAKDOWN:")
        logger.info(f"   • Timeframe Score: {timeframe_score:.4f} (60%)")
        logger.info(f"   • Correlation Adjustment: {correlation_adjustment:.4f} (20%)")
        logger.info(f"   • Returns Score: {returns_score:.4f} (10%)")
        logger.info(f"   • Volatility Adjustment: {volatility_adjustment:.4f} (10%)")
        logger.info(f"   • Final Phase 2 Score: {final_score:.4f}")
        
        return final_score
    
    # ==================== MAIN PHASE 2 SIGNAL GENERATION ====================
    
    async def generate_phase2_signal(self, request: SignalRequest, username: str) -> Phase2TradingSignal:
        """
        Generate Phase 2 enhanced trading signal
        
        Args:
            request: Signal generation request
            username: Username for signal storage
            
        Returns:
            Phase 2 enhanced trading signal
        """
        try:
            logger.info("=" * 80)
            logger.info(f"🚀 STARTING PHASE 2 SIGNAL GENERATION")
            logger.info(f"📊 Symbol: {request.symbol}")
            logger.info(f"⏰ Timeframe: {request.timeframe}")
            logger.info(f"👤 Username: {username}")
            logger.info(f"🕐 Timestamp: {get_israel_time().isoformat()}")
            logger.info("=" * 80)
            
            # Step 1: Multi-timeframe analysis
            logger.info("📈 STEP 1: MULTI-TIMEFRAME ANALYSIS")
            timeframe_scores = await self.analyze_multi_timeframe(request.symbol)
            timeframe_consensus, timeframe_strength = self.calculate_timeframe_consensus(timeframe_scores)
            logger.info(f"✅ Multi-timeframe analysis completed - Consensus: {timeframe_consensus}")
            
            # Step 2: Cross-asset correlation analysis
            logger.info("🔗 STEP 2: CROSS-ASSET CORRELATION ANALYSIS")
            correlation_metrics = await self.analyze_cross_asset_correlation(request.symbol)
            logger.info(f"✅ Cross-asset correlation analysis completed")
            
            # Step 3: Get primary timeframe data for additional analysis
            logger.info("📊 STEP 3: PRIMARY TIMEFRAME ANALYSIS")
            ohlcv = self.fetch_market_data(request.symbol, request.timeframe)
            
            # Step 4: Returns analysis
            logger.info("📈 STEP 4: RETURNS ANALYSIS")
            returns_metrics = self.analyze_returns(ohlcv)
            logger.info(f"✅ Returns analysis completed")
            
            # Step 5: Volatility metrics
            logger.info("📊 STEP 5: VOLATILITY METRICS")
            volatility_metrics = self.analyze_volatility_metrics(ohlcv)
            logger.info(f"✅ Volatility metrics analysis completed")
            
            # Step 6: Gaps and micro-moves detection
            logger.info("🔍 STEP 6: GAPS & MICRO-MOVES DETECTION")
            gaps_micro_moves = self.detect_gaps_micro_moves(ohlcv)
            logger.info(f"✅ Gaps and micro-moves detection completed")
            
            # Step 7: Calculate Phase 2 enhanced technical score
            logger.info("🧮 STEP 7: PHASE 2 ENHANCED TECHNICAL SCORE")
            phase2_tech_score = self.calculate_phase2_technical_score(
                timeframe_scores, correlation_metrics, returns_metrics, volatility_metrics
            )
            logger.info(f"✅ Phase 2 technical score calculated: {phase2_tech_score:.4f}")
            
            # Step 8: Sentiment analysis (reuse from Phase 1)
            logger.info("💭 STEP 8: SENTIMENT ANALYSIS")
            sentiment_score = await self.analyze_sentiment(request.symbol)
            logger.info(f"✅ Sentiment analysis completed: {sentiment_score:.4f}")
            
            # Step 9: Get configuration and apply parameters
            logger.info("⚙️ STEP 9: LOADING CONFIGURATION AND APPLYING PARAMETERS")
            config = load_config_from_env()
            
            # Apply custom parameters or use defaults
            tech_weight = request.technical_weight if request.technical_weight is not None else config.thresholds.technical_weight
            sentiment_weight = request.sentiment_weight if request.sentiment_weight is not None else config.thresholds.sentiment_weight
            
            # Calculate fused score
            fused_score = tech_weight * phase2_tech_score + sentiment_weight * sentiment_score
            
            # Define thresholds
            buy_threshold = request.buy_threshold if request.buy_threshold is not None else config.thresholds.buy_threshold
            if request.sell_threshold is not None:
                sell_threshold = request.sell_threshold
            else:
                sell_threshold = -buy_threshold
            
            logger.info("🧮 PHASE 2 SCORE CALCULATIONS:")
            logger.info(f"   • Phase 2 Technical Score: {phase2_tech_score:.4f}")
            logger.info(f"   • Sentiment Score: {sentiment_score:.4f}")
            logger.info(f"   • Technical Contribution: {tech_weight} × {phase2_tech_score:.4f} = {tech_weight * phase2_tech_score:.4f}")
            logger.info(f"   • Sentiment Contribution: {sentiment_weight} × {sentiment_score:.4f} = {sentiment_weight * sentiment_score:.4f}")
            logger.info(f"   • Fused Score: {fused_score:.4f}")
            
            # Step 10: Determine signal type
            logger.info("🎯 STEP 10: DETERMINING SIGNAL TYPE")
            signal_type = self.determine_signal_type(fused_score, buy_threshold, sell_threshold)
            logger.info(f"✅ Signal type determined: {signal_type}")
            
            # Step 11: Calculate risk management (reuse from Phase 1)
            logger.info("🛡️ STEP 11: CALCULATING RISK MANAGEMENT")
            logger.info(f"🔍 DEBUG: About to calculate enhanced technical indicators")
            
            indicators = self.calculate_enhanced_technical_indicators(ohlcv)
            logger.info(f"🔍 DEBUG: Enhanced technical indicators calculated successfully")
            logger.info(f"🔍 DEBUG: indicators type: {type(indicators)}")
            logger.info(f"🔍 DEBUG: indicators keys: {list(indicators.keys())}")
            
            logger.info(f"🔍 DEBUG: About to call detect_market_regime")
            # Debug: Check what detect_market_regime returns
            regime_result = self.detect_market_regime(indicators)
            logger.info(f"🔍 DEBUG: detect_market_regime returned: {type(regime_result)} - {regime_result}")
            
            if isinstance(regime_result, tuple) and len(regime_result) == 2:
                regime, volatility_state = regime_result
            else:
                logger.error(f"❌ ERROR: detect_market_regime returned unexpected result: {regime_result}")
                regime, volatility_state = "trending", "medium"
            
            stop_loss, take_profit = self.calculate_enhanced_risk_management(indicators, signal_type, regime)
            logger.info(f"✅ Risk management calculated")
            
            # Step 12: Create Phase 2 enhanced signal object
            logger.info("📝 STEP 12: CREATING PHASE 2 ENHANCED SIGNAL OBJECT")
            signal = Phase2TradingSignal(
                symbol=request.symbol,
                timeframe=request.timeframe,
                timestamp=get_israel_time().isoformat(),
                signal_type=signal_type,
                confidence=abs(fused_score),
                technical_score=phase2_tech_score,
                sentiment_score=sentiment_score,
                fused_score=fused_score,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reasoning=f"Phase 2: Technical: {phase2_tech_score:.2f}, Sentiment: {sentiment_score:.2f}, Consensus: {timeframe_consensus}, Strength: {timeframe_strength:.2f}",
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
                volatility_state=volatility_state,
                # Phase 2 enhanced features
                multi_timeframe_scores=timeframe_scores,
                timeframe_consensus=timeframe_consensus,
                timeframe_strength=timeframe_strength,
                btc_correlation=correlation_metrics.get('btc_correlation_1h', 0.0),
                eth_correlation=correlation_metrics.get('eth_correlation_1h', 0.0),
                btc_dominance=correlation_metrics.get('btc_dominance', 45.0),
                returns_analysis=returns_metrics,
                volatility_metrics=volatility_metrics,
                gaps_micro_moves=gaps_micro_moves
            )
            logger.info("✅ Phase 2 enhanced signal object created successfully")
            
            # Step 13: Store signal
            logger.info("💾 STEP 13: STORING PHASE 2 SIGNAL")
            storage_success = await self.store_signal(signal, username)
            logger.info(f"✅ Phase 2 signal storage: {'SUCCESS' if storage_success else 'FAILED'}")
            
            # Step 14: Send Telegram notification
            logger.info("📱 STEP 14: SENDING TELEGRAM NOTIFICATION")
            await self.send_telegram_notification(signal, username)
            logger.info("✅ Telegram notification sent")
            
            logger.info("=" * 80)
            logger.info(f"🎉 PHASE 2 SIGNAL GENERATION COMPLETED SUCCESSFULLY")
            logger.info(f"📊 Symbol: {request.symbol}")
            logger.info(f"🎯 Signal Type: {signal_type}")
            logger.info(f"📈 Confidence: {signal.confidence:.4f}")
            logger.info(f"🧮 Fused Score: {fused_score:.4f}")
            logger.info(f"🎭 Timeframe Consensus: {timeframe_consensus}")
            logger.info(f"💪 Consensus Strength: {timeframe_strength:.2f}")
            logger.info(f"🔗 BTC Correlation: {correlation_metrics.get('btc_correlation_1h', 0.0):.3f}")
            logger.info(f"👤 Username: {username}")
            logger.info("=" * 80)
            
            return signal
            
        except Exception as e:
            logger.error("=" * 80)
            logger.error(f"❌ PHASE 2 SIGNAL GENERATION FAILED")
            logger.error(f"📊 Symbol: {request.symbol}")
            logger.error(f"👤 Username: {username}")
            logger.error(f"🚨 Error: {str(e)}")
            logger.error("=" * 80)
            raise HTTPException(status_code=500, detail=str(e))


# Create global instance
phase2_signal_generator = Phase2SignalGenerator()
