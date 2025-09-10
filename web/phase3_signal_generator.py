#!/usr/bin/env python3
"""
Phase 3 Advanced Signal Generation Module
Advanced regime detection, RSI variants, and enhanced risk management
Based on ChatGPT recommendations for institutional-grade trading systems
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
from agent.indicators import (
    compute_rsi, compute_ema, compute_macd, compute_atr, compute_rsi_enhanced, 
    rsi_signal, stoch_rsi, compute_adx, compute_volatility_regime, compute_market_regime,
    compute_advanced_rsi_variants, compute_dynamic_position_sizing, compute_volatility_adjusted_stops
)
from agent.models.sentiment import SentimentAnalyzer
from agent.news.rss import fetch_headlines
from agent.news.reddit import fetch_crypto_reddit_posts, fetch_stock_reddit_posts
from agent.cache.redis_client import get_redis_client

# Import Phase 2 enhanced generator
from .phase2_signal_generator import Phase2SignalGenerator, Phase2TradingSignal, SignalRequest

# Configure logging
logger = logging.getLogger(__name__)

# Israel timezone
ISRAEL_TZ = pytz.timezone('Asia/Jerusalem')

def get_israel_time():
    """Get current time in Israel timezone with microsecond precision"""
    return datetime.now(ISRAEL_TZ)


class Phase3TradingSignal(Phase2TradingSignal):
    """Phase 3 advanced trading signal with regime detection, advanced RSI, and enhanced risk management"""
    
    # Advanced regime detection
    adx_values: Optional[Dict[str, float]] = None  # ADX, +DI, -DI values
    trend_strength: Optional[str] = None  # weak, moderate, strong, very_strong
    volatility_regime: Optional[str] = None  # low, medium, high
    market_regime: Optional[str] = None  # trending, ranging, consolidation, volatile
    
    # Advanced RSI variants
    rsi_variants: Optional[Dict[str, Dict[str, float]]] = None  # Multiple RSI periods with signals
    rsi_consensus: Optional[str] = None  # bullish, bearish, neutral
    rsi_strength: Optional[float] = None  # Strength of RSI consensus
    
    # Enhanced risk management
    position_sizing: Optional[Dict[str, float]] = None  # Dynamic position sizing
    volatility_adjusted_stops: Optional[Dict[str, float]] = None  # ATR-based stops
    risk_metrics: Optional[Dict[str, float]] = None  # Risk per trade, max drawdown, etc.


class Phase3SignalGenerator(Phase2SignalGenerator):
    """Phase 3 advanced signal generator with regime detection, advanced RSI, and enhanced risk management"""
    
    def __init__(self):
        super().__init__()
        self.rsi_periods = [7, 9, 14]  # Multiple RSI periods
        self.account_balance = 10000.0  # Default account balance
        self.risk_per_trade = 0.02  # 2% risk per trade
        logger.info("Phase 3 Advanced Signal Generator initialized")
    
    # ==================== PHASE 3: ADVANCED REGIME DETECTION ====================
    
    def analyze_advanced_regime_detection(self, ohlcv: pd.DataFrame) -> Dict[str, Any]:
        """
        Advanced regime detection with ADX, volatility analysis, and market classification
        
        Args:
            ohlcv: OHLCV data
            
        Returns:
            Dictionary with comprehensive regime analysis
        """
        logger.info("🔧 PHASE 3: Advanced regime detection analysis")
        
        high = ohlcv['high']
        low = ohlcv['low']
        close = ohlcv['close']
        
        try:
            # Calculate comprehensive market regime
            logger.info(f"🔍 DEBUG: About to call compute_market_regime")
            regime_data = compute_market_regime(high, low, close, adx_period=14, vol_period=20)
            logger.info(f"🔍 DEBUG: compute_market_regime returned: {type(regime_data)} - {regime_data}")
            
            # Extract key metrics
            adx_values = {
                'adx': regime_data['current_adx'],
                'di_plus': regime_data['di_plus'].iloc[-1] if len(regime_data['di_plus']) > 0 else 0.0,
                'di_minus': regime_data['di_minus'].iloc[-1] if len(regime_data['di_minus']) > 0 else 0.0
            }
            
            trend_strength = regime_data['current_trend_strength']
            market_regime = regime_data['current_market_regime']
            volatility_regime = regime_data['volatility_data']['current_regime']
            
            logger.info(f"✅ Advanced regime detection completed:")
            logger.info(f"   • ADX: {adx_values['adx']:.2f}")
            logger.info(f"   • Trend Strength: {trend_strength}")
            logger.info(f"   • Market Regime: {market_regime}")
            logger.info(f"   • Volatility Regime: {volatility_regime}")
            
            return {
                'adx_values': adx_values,
                'trend_strength': trend_strength,
                'market_regime': market_regime,
                'volatility_regime': volatility_regime,
                'trend_direction': regime_data['current_trend_direction'],
                'volatility_data': regime_data['volatility_data']
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze advanced regime detection: {str(e)}")
            return {
                'adx_values': {'adx': 0.0, 'di_plus': 0.0, 'di_minus': 0.0},
                'trend_strength': 'weak',
                'market_regime': 'ranging',
                'volatility_regime': 'medium',
                'trend_direction': 'sideways',
                'volatility_data': {'current_volatility': 0.0, 'current_regime': 'medium'}
            }
    
    # ==================== PHASE 3: ADVANCED RSI VARIANTS ====================
    
    def analyze_advanced_rsi_variants(self, ohlcv: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze multiple RSI variants with signal lines and crossovers
        
        Args:
            ohlcv: OHLCV data
            
        Returns:
            Dictionary with RSI variants analysis
        """
        logger.info("🔧 PHASE 3: Advanced RSI variants analysis")
        
        close = ohlcv['close']
        
        try:
            # Calculate advanced RSI variants
            rsi_variants = compute_advanced_rsi_variants(close, periods=self.rsi_periods)
            
            # Calculate RSI consensus
            rsi_scores = []
            for period in self.rsi_periods:
                rsi_data = rsi_variants[f'rsi_{period}']
                current_rsi = rsi_data['current_rsi']
                
                # Convert RSI to score (-1 to 1)
                if current_rsi > 70:
                    score = -1.0  # Overbought
                elif current_rsi < 30:
                    score = 1.0   # Oversold
                elif current_rsi > 50:
                    score = -(current_rsi - 50) / 20  # Bearish
                else:
                    score = (50 - current_rsi) / 20   # Bullish
                
                rsi_scores.append(score)
            
            # Calculate consensus
            avg_rsi_score = np.mean(rsi_scores)
            
            if avg_rsi_score > 0.3:
                rsi_consensus = "bullish"
            elif avg_rsi_score < -0.3:
                rsi_consensus = "bearish"
            else:
                rsi_consensus = "neutral"
            
            # Calculate strength (how consistent the signals are)
            positive_signals = sum(1 for score in rsi_scores if score > 0.1)
            negative_signals = sum(1 for score in rsi_scores if score < -0.1)
            total_signals = len(rsi_scores)
            
            if total_signals == 0:
                rsi_strength = 0.0
            else:
                rsi_strength = max(positive_signals, negative_signals) / total_signals
            
            logger.info(f"✅ Advanced RSI variants analysis completed:")
            logger.info(f"   • RSI Consensus: {rsi_consensus}")
            logger.info(f"   • RSI Strength: {rsi_strength:.2f}")
            logger.info(f"   • Average RSI Score: {avg_rsi_score:.3f}")
            
            return {
                'rsi_variants': rsi_variants,
                'rsi_consensus': rsi_consensus,
                'rsi_strength': rsi_strength,
                'rsi_scores': rsi_scores,
                'avg_rsi_score': avg_rsi_score
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze advanced RSI variants: {str(e)}")
            return {
                'rsi_variants': {},
                'rsi_consensus': 'neutral',
                'rsi_strength': 0.0,
                'rsi_scores': [0.0, 0.0, 0.0],
                'avg_rsi_score': 0.0
            }
    
    # ==================== PHASE 3: ENHANCED RISK MANAGEMENT ====================
    
    def calculate_enhanced_risk_management(self, ohlcv: pd.DataFrame, signal_type: str, 
                                         regime_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate enhanced risk management with dynamic position sizing and volatility-adjusted stops
        
        Args:
            ohlcv: OHLCV data
            signal_type: BUY, SELL, or HOLD
            regime_data: Market regime data
            
        Returns:
            Dictionary with enhanced risk management metrics
        """
        logger.info("🔧 PHASE 3: Enhanced risk management calculation")
        logger.info(f"🔍 DEBUG: ohlcv type: {type(ohlcv)}")
        logger.info(f"🔍 DEBUG: signal_type: {signal_type} (type: {type(signal_type)})")
        logger.info(f"🔍 DEBUG: regime_data: {regime_data} (type: {type(regime_data)})")
        
        close = ohlcv['close']
        high = ohlcv['high']
        low = ohlcv['low']
        
        try:
            # Calculate ATR for volatility-adjusted stops
            atr = compute_atr(high, low, close, period=14)
            
            # Calculate volatility-adjusted stops
            stops_data = compute_volatility_adjusted_stops(close, high, low, atr, volatility_multiplier=2.0)
            
            # Calculate dynamic position sizing
            position_data = compute_dynamic_position_sizing(
                close, atr, 
                account_balance=self.account_balance, 
                risk_per_trade=self.risk_per_trade
            )
            
            # Adjust position sizing based on market regime
            regime_multiplier = 1.0
            market_regime = regime_data.get('market_regime', 'ranging')
            volatility_regime = regime_data.get('volatility_regime', 'medium')
            
            if market_regime == 'trending':
                regime_multiplier = 1.2  # Increase position in trending markets
            elif market_regime == 'volatile':
                regime_multiplier = 0.7  # Reduce position in volatile markets
            elif market_regime == 'consolidation':
                regime_multiplier = 0.8  # Reduce position in consolidation
            
            if volatility_regime == 'high':
                regime_multiplier *= 0.8  # Further reduce in high volatility
            elif volatility_regime == 'low':
                regime_multiplier *= 1.1  # Slightly increase in low volatility
            
            # Apply regime adjustments
            adjusted_position_size = position_data['current_position_size'] * regime_multiplier
            adjusted_position_value = adjusted_position_size * close.iloc[-1]
            adjusted_position_percentage = (adjusted_position_value / self.account_balance) * 100
            
            # Calculate risk metrics
            current_price = close.iloc[-1]
            current_atr = atr.iloc[-1] if len(atr) > 0 else current_price * 0.02
            
            if signal_type == "BUY":
                stop_loss = stops_data['current_stop_long']
                take_profit = stops_data['current_tp_long']
            elif signal_type == "SELL":
                stop_loss = stops_data['current_stop_short']
                take_profit = stops_data['current_tp_short']
            else:
                stop_loss = current_price
                take_profit = current_price
            
            # Calculate risk-reward ratio
            if signal_type in ["BUY", "SELL"] and stop_loss != current_price:
                risk_amount = abs(current_price - stop_loss)
                reward_amount = abs(take_profit - current_price)
                risk_reward_ratio = reward_amount / risk_amount if risk_amount > 0 else 0.0
            else:
                risk_reward_ratio = 0.0
            
            # Calculate maximum drawdown estimate
            max_drawdown_estimate = (current_atr * 3) / current_price  # 3x ATR as max drawdown estimate
            
            risk_metrics = {
                'risk_per_trade': self.risk_per_trade,
                'risk_amount': position_data['risk_amount'],
                'risk_reward_ratio': risk_reward_ratio,
                'max_drawdown_estimate': max_drawdown_estimate,
                'volatility_multiplier': 2.0,
                'regime_multiplier': regime_multiplier
            }
            
            position_sizing = {
                'position_size': adjusted_position_size,
                'position_value': adjusted_position_value,
                'position_percentage': adjusted_position_percentage,
                'account_balance': self.account_balance,
                'risk_per_trade': self.risk_per_trade
            }
            
            volatility_adjusted_stops = {
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'atr_value': current_atr,
                'volatility_multiplier': 2.0
            }
            
            logger.info(f"✅ Enhanced risk management calculated:")
            logger.info(f"   • Position Size: {adjusted_position_size:.4f}")
            logger.info(f"   • Position Value: ${adjusted_position_value:.2f}")
            logger.info(f"   • Position %: {adjusted_position_percentage:.2f}%")
            logger.info(f"   • Risk-Reward Ratio: {risk_reward_ratio:.2f}")
            logger.info(f"   • Regime Multiplier: {regime_multiplier:.2f}")
            
            return {
                'position_sizing': position_sizing,
                'volatility_adjusted_stops': volatility_adjusted_stops,
                'risk_metrics': risk_metrics,
                'stop_loss': stop_loss,
                'take_profit': take_profit
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate enhanced risk management: {str(e)}")
            current_price = close.iloc[-1]
            return {
                'position_sizing': {
                    'position_size': 0.0,
                    'position_value': 0.0,
                    'position_percentage': 0.0,
                    'account_balance': self.account_balance,
                    'risk_per_trade': self.risk_per_trade
                },
                'volatility_adjusted_stops': {
                    'stop_loss': current_price,
                    'take_profit': current_price,
                    'atr_value': current_price * 0.02,
                    'volatility_multiplier': 2.0
                },
                'risk_metrics': {
                    'risk_per_trade': self.risk_per_trade,
                    'risk_amount': self.account_balance * self.risk_per_trade,
                    'risk_reward_ratio': 0.0,
                    'max_drawdown_estimate': 0.05,
                    'volatility_multiplier': 2.0,
                    'regime_multiplier': 1.0
                },
                'stop_loss': current_price,
                'take_profit': current_price
            }
    
    # ==================== PHASE 3: ENHANCED TECHNICAL SCORE ====================
    
    def calculate_phase3_technical_score(self, 
                                       regime_data: Dict[str, Any],
                                       rsi_data: Dict[str, Any],
                                       phase2_tech_score: float) -> float:
        """
        Calculate Phase 3 enhanced technical score with regime and RSI analysis
        
        Args:
            regime_data: Advanced regime detection data
            rsi_data: Advanced RSI variants data
            phase2_tech_score: Base technical score from Phase 2
            
        Returns:
            Enhanced technical score
        """
        logger.info("🔧 PHASE 3: Calculating enhanced technical score")
        
        # Base score from Phase 2 (60% weight)
        base_score = phase2_tech_score * 0.6
        
        # Regime-based adjustment (25% weight)
        regime_score = 0.0
        market_regime = regime_data.get('market_regime', 'ranging')
        trend_strength = regime_data.get('trend_strength', 'weak')
        adx = regime_data.get('adx_values', {}).get('adx', 0.0)
        
        if market_regime == 'trending' and trend_strength in ['strong', 'very_strong']:
            regime_score = 0.2  # Boost in strong trending markets
        elif market_regime == 'volatile':
            regime_score = -0.1  # Reduce confidence in volatile markets
        elif market_regime == 'consolidation':
            regime_score = -0.05  # Slight reduction in consolidation
        
        # ADX-based adjustment
        if adx > 25:
            regime_score += 0.1  # Boost for strong trends
        elif adx < 15:
            regime_score -= 0.05  # Reduce for weak trends
        
        # RSI consensus adjustment (15% weight)
        rsi_score = 0.0
        rsi_consensus = rsi_data.get('rsi_consensus', 'neutral')
        rsi_strength = rsi_data.get('rsi_strength', 0.0)
        avg_rsi_score = rsi_data.get('avg_rsi_score', 0.0)
        
        if rsi_consensus == 'bullish' and rsi_strength > 0.6:
            rsi_score = 0.15
        elif rsi_consensus == 'bearish' and rsi_strength > 0.6:
            rsi_score = -0.15
        elif rsi_consensus == 'neutral':
            rsi_score = avg_rsi_score * 0.1  # Small adjustment based on average score
        
        # Calculate final score
        final_score = base_score + regime_score + rsi_score
        
        logger.info(f"🧮 PHASE 3 TECHNICAL SCORE BREAKDOWN:")
        logger.info(f"   • Base Score (60%): {base_score:.4f}")
        logger.info(f"   • Regime Adjustment (25%): {regime_score:.4f}")
        logger.info(f"   • RSI Adjustment (15%): {rsi_score:.4f}")
        logger.info(f"   • Final Phase 3 Score: {final_score:.4f}")
        
        return final_score
    
    # ==================== MAIN PHASE 3 SIGNAL GENERATION ====================
    
    async def generate_phase3_signal(self, request: SignalRequest, username: str) -> Phase3TradingSignal:
        """
        Generate Phase 3 advanced trading signal
        
        Args:
            request: Signal generation request
            username: Username for signal storage
            
        Returns:
            Phase 3 advanced trading signal
        """
        try:
            logger.info("=" * 80)
            logger.info(f"🚀 STARTING PHASE 3 ADVANCED SIGNAL GENERATION")
            logger.info(f"📊 Symbol: {request.symbol}")
            logger.info(f"⏰ Timeframe: {request.timeframe}")
            logger.info(f"👤 Username: {username}")
            logger.info(f"🕐 Timestamp: {get_israel_time().isoformat()}")
            logger.info("=" * 80)
            
            # Step 1: Get Phase 2 base analysis
            logger.info("📈 STEP 1: PHASE 2 BASE ANALYSIS")
            phase2_signal = await self.generate_phase2_signal(request, username)
            logger.info(f"✅ Phase 2 base analysis completed")
            
            # Step 2: Advanced regime detection
            logger.info("🔍 STEP 2: ADVANCED REGIME DETECTION")
            ohlcv = self.fetch_market_data(request.symbol, request.timeframe)
            logger.info(f"🔍 DEBUG: ohlcv type: {type(ohlcv)}")
            logger.info(f"🔍 DEBUG: About to call analyze_advanced_regime_detection")
            regime_data = self.analyze_advanced_regime_detection(ohlcv)
            logger.info(f"🔍 DEBUG: analyze_advanced_regime_detection returned: {type(regime_data)} - {regime_data}")
            logger.info(f"✅ Advanced regime detection completed")
            
            # Step 3: Advanced RSI variants analysis
            logger.info("📊 STEP 3: ADVANCED RSI VARIANTS ANALYSIS")
            rsi_data = self.analyze_advanced_rsi_variants(ohlcv)
            logger.info(f"✅ Advanced RSI variants analysis completed")
            
            # Step 4: Calculate Phase 3 enhanced technical score
            logger.info("🧮 STEP 4: PHASE 3 ENHANCED TECHNICAL SCORE")
            phase3_tech_score = self.calculate_phase3_technical_score(
                regime_data, rsi_data, phase2_signal.technical_score
            )
            logger.info(f"✅ Phase 3 technical score calculated: {phase3_tech_score:.4f}")
            
            # Step 5: Enhanced risk management
            logger.info("🛡️ STEP 5: ENHANCED RISK MANAGEMENT")
            logger.info(f"🔍 DEBUG: About to call calculate_enhanced_risk_management")
            logger.info(f"🔍 DEBUG: ohlcv type: {type(ohlcv)}")
            logger.info(f"🔍 DEBUG: phase2_signal.signal_type: {phase2_signal.signal_type} (type: {type(phase2_signal.signal_type)})")
            logger.info(f"🔍 DEBUG: regime_data: {regime_data} (type: {type(regime_data)})")
            
            # Use Phase 3 specific method with correct parameters
            risk_data = self.calculate_enhanced_risk_management(ohlcv, phase2_signal.signal_type, regime_data)
            # Extract stop_loss and take_profit from risk_data
            stop_loss = risk_data.get('stop_loss')
            take_profit = risk_data.get('take_profit')
            logger.info(f"✅ Enhanced risk management calculated")
            
            # Step 6: Get configuration and apply parameters
            logger.info("⚙️ STEP 6: LOADING CONFIGURATION AND APPLYING PARAMETERS")
            config = load_config_from_env()
            
            # Apply custom parameters or use defaults
            tech_weight = request.technical_weight if request.technical_weight is not None else config.thresholds.technical_weight
            sentiment_weight = request.sentiment_weight if request.sentiment_weight is not None else config.thresholds.sentiment_weight
            
            # Calculate fused score with Phase 3 technical score
            fused_score = tech_weight * phase3_tech_score + sentiment_weight * phase2_signal.sentiment_score
            
            # Define thresholds
            buy_threshold = request.buy_threshold if request.buy_threshold is not None else config.thresholds.buy_threshold
            if request.sell_threshold is not None:
                sell_threshold = request.sell_threshold
            else:
                sell_threshold = -buy_threshold
            
            logger.info("🧮 PHASE 3 SCORE CALCULATIONS:")
            logger.info(f"   • Phase 3 Technical Score: {phase3_tech_score:.4f}")
            logger.info(f"   • Sentiment Score: {phase2_signal.sentiment_score:.4f}")
            logger.info(f"   • Technical Contribution: {tech_weight} × {phase3_tech_score:.4f} = {tech_weight * phase3_tech_score:.4f}")
            logger.info(f"   • Sentiment Contribution: {sentiment_weight} × {phase2_signal.sentiment_score:.4f} = {sentiment_weight * phase2_signal.sentiment_score:.4f}")
            logger.info(f"   • Fused Score: {fused_score:.4f}")
            
            # Step 7: Determine signal type
            logger.info("🎯 STEP 7: DETERMINING SIGNAL TYPE")
            signal_type = self.determine_signal_type(fused_score, buy_threshold, sell_threshold)
            logger.info(f"✅ Signal type determined: {signal_type}")
            
            # Step 8: Create Phase 3 advanced signal object
            logger.info("📝 STEP 8: CREATING PHASE 3 ADVANCED SIGNAL OBJECT")
            signal = Phase3TradingSignal(
                # Inherit all Phase 2 fields
                symbol=phase2_signal.symbol,
                timeframe=phase2_signal.timeframe,
                timestamp=phase2_signal.timestamp,
                signal_type=signal_type,
                confidence=abs(fused_score),
                technical_score=phase3_tech_score,
                sentiment_score=phase2_signal.sentiment_score,
                fused_score=fused_score,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reasoning=f"Phase 3: Technical: {phase3_tech_score:.2f}, Sentiment: {phase2_signal.sentiment_score:.2f}, Regime: {regime_data['market_regime']}, RSI: {rsi_data['rsi_consensus']}",
                applied_buy_threshold=buy_threshold,
                applied_sell_threshold=sell_threshold,
                applied_tech_weight=tech_weight,
                applied_sentiment_weight=sentiment_weight,
                bollinger_position=phase2_signal.bollinger_position,
                vwap_distance=phase2_signal.vwap_distance,
                volume_profile=phase2_signal.volume_profile,
                regime_detection=phase2_signal.regime_detection,
                volatility_state=phase2_signal.volatility_state,
                multi_timeframe_scores=phase2_signal.multi_timeframe_scores,
                timeframe_consensus=phase2_signal.timeframe_consensus,
                timeframe_strength=phase2_signal.timeframe_strength,
                btc_correlation=phase2_signal.btc_correlation,
                eth_correlation=phase2_signal.eth_correlation,
                btc_dominance=phase2_signal.btc_dominance,
                returns_analysis=phase2_signal.returns_analysis,
                volatility_metrics=phase2_signal.volatility_metrics,
                gaps_micro_moves=phase2_signal.gaps_micro_moves,
                # Phase 3 advanced features
                adx_values=regime_data['adx_values'],
                trend_strength=regime_data['trend_strength'],
                volatility_regime=regime_data['volatility_regime'],
                market_regime=regime_data['market_regime'],
                rsi_variants=rsi_data['rsi_variants'],
                rsi_consensus=rsi_data['rsi_consensus'],
                rsi_strength=rsi_data['rsi_strength'],
                position_sizing=risk_data.get('position_sizing'),
                volatility_adjusted_stops=risk_data.get('volatility_adjusted_stops'),
                risk_metrics=risk_data.get('risk_metrics')
            )
            logger.info("✅ Phase 3 advanced signal object created successfully")
            
            # Step 9: Store signal
            logger.info("💾 STEP 9: STORING PHASE 3 SIGNAL")
            storage_success = await self.store_signal(signal, username)
            logger.info(f"✅ Phase 3 signal storage: {'SUCCESS' if storage_success else 'FAILED'}")
            
            # Step 10: Send Telegram notification
            logger.info("📱 STEP 10: SENDING TELEGRAM NOTIFICATION")
            await self.send_telegram_notification(signal, username)
            logger.info("✅ Telegram notification sent")
            
            logger.info("=" * 80)
            logger.info(f"🎉 PHASE 3 ADVANCED SIGNAL GENERATION COMPLETED SUCCESSFULLY")
            logger.info(f"📊 Symbol: {request.symbol}")
            logger.info(f"🎯 Signal Type: {signal_type}")
            logger.info(f"📈 Confidence: {signal.confidence:.4f}")
            logger.info(f"🧮 Fused Score: {fused_score:.4f}")
            logger.info(f"🔍 Market Regime: {regime_data['market_regime']}")
            logger.info(f"💪 Trend Strength: {regime_data['trend_strength']}")
            logger.info(f"📊 RSI Consensus: {rsi_data['rsi_consensus']}")
            logger.info(f"🛡️ Position Size: {stop_loss:.2f} (SL), {take_profit:.2f} (TP)")
            logger.info(f"👤 Username: {username}")
            logger.info("=" * 80)
            
            return signal
            
        except Exception as e:
            logger.error("=" * 80)
            logger.error(f"❌ PHASE 3 ADVANCED SIGNAL GENERATION FAILED")
            logger.error(f"📊 Symbol: {request.symbol}")
            logger.error(f"👤 Username: {username}")
            logger.error(f"🚨 Error: {str(e)}")
            logger.error("=" * 80)
            raise HTTPException(status_code=500, detail=str(e))


# Create global instance
phase3_signal_generator = Phase3SignalGenerator()
