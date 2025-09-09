"""
Trading Signal Validation Module

This module implements safety checks and validation for trading signals
based on ChatGPT's recommendations for trading hygiene and robustness.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timezone
import asyncio
import hashlib
import random

from .config import AgentConfig, SafetyConfig
from .logging_config import get_logger

logger = get_logger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass


class DataValidator:
    """Validates market data and trading parameters"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.safety = config.safety
    
    def validate_ohlcv_data(self, ohlcv: pd.DataFrame, symbol: str, timeframe: str, 
                           allow_incomplete: bool = None) -> pd.DataFrame:
        """
        Validate and clean OHLCV data according to safety rules.
        
        Args:
            ohlcv: Market data DataFrame
            symbol: Trading symbol
            timeframe: Timeframe string
            allow_incomplete: Override config setting for incomplete candles
            
        Returns:
            Cleaned and validated OHLCV data
            
        Raises:
            ValidationError: If data fails validation
        """
        if ohlcv is None or ohlcv.empty:
            raise ValidationError("OHLCV data is None or empty")
        
        # Check minimum data points for indicators
        min_required = self.safety.min_data_points_for_indicators + self.safety.safety_margin
        if len(ohlcv) < min_required:
            raise ValidationError(
                f"Insufficient data points: {len(ohlcv)} < {min_required} "
                f"(min_indicators: {self.safety.min_data_points_for_indicators} + "
                f"safety_margin: {self.safety.safety_margin})"
            )
        
        # Sort by index and ensure timezone alignment
        ohlcv = ohlcv.sort_index()
        
        # Check for incomplete candles (last row)
        allow_incomplete = allow_incomplete if allow_incomplete is not None else self.safety.allow_incomplete_candles
        if not allow_incomplete:
            # Drop the last row (incomplete candle)
            if len(ohlcv) > 1:
                ohlcv = ohlcv.iloc[:-1]
                logger.debug(f"Dropped incomplete candle for {symbol} @ {timeframe}")
        
        # Validate candle freshness
        self._validate_candle_freshness(ohlcv, timeframe)
        
        # Clean NaN and infinite values
        ohlcv = self._clean_nan_values(ohlcv)
        
        # Validate data types and ranges
        self._validate_data_types(ohlcv)
        
        return ohlcv
    
    def _validate_candle_freshness(self, ohlcv: pd.DataFrame, timeframe: str):
        """Validate that the last candle is not stale"""
        if timeframe not in self.safety.max_candle_staleness:
            logger.warning(f"No staleness threshold defined for timeframe {timeframe}")
            return
        
        max_staleness_minutes = self.safety.max_candle_staleness[timeframe]
        last_timestamp = ohlcv.index[-1]
        
        # Convert to timezone-aware if needed
        if last_timestamp.tzinfo is None:
            last_timestamp = last_timestamp.tz_localize('UTC')
        
        now = datetime.now(timezone.utc)
        staleness_minutes = (now - last_timestamp).total_seconds() / 60
        
        if staleness_minutes > max_staleness_minutes:
            raise ValidationError(
                f"Market data is stale: {staleness_minutes:.1f} minutes old "
                f"(max allowed: {max_staleness_minutes} minutes for {timeframe})"
            )
        
        logger.debug(f"Data freshness check passed: {staleness_minutes:.1f} minutes old")
    
    def _clean_nan_values(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Clean NaN and infinite values from OHLCV data"""
        original_len = len(ohlcv)
        
        # Replace infinite values with NaN
        ohlcv = ohlcv.replace([np.inf, -np.inf], np.nan)
        
        # Drop rows with any NaN values
        ohlcv = ohlcv.dropna()
        
        dropped_rows = original_len - len(ohlcv)
        if dropped_rows > 0:
            logger.warning(f"Dropped {dropped_rows} rows with NaN/infinite values")
            
            # Check if we still have enough data
            min_required = self.safety.min_data_points_for_indicators + self.safety.safety_margin
            if len(ohlcv) < min_required:
                raise ValidationError(
                    f"Too many NaN values dropped: {len(ohlcv)} < {min_required} required"
                )
        
        return ohlcv
    
    def _validate_data_types(self, ohlcv: pd.DataFrame):
        """Validate data types and ranges"""
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        for col in required_columns:
            if col not in ohlcv.columns:
                raise ValidationError(f"Missing required column: {col}")
            
            # Check for numeric data
            if not pd.api.types.is_numeric_dtype(ohlcv[col]):
                raise ValidationError(f"Column {col} is not numeric")
            
            # Check for reasonable price ranges (basic sanity check)
            if col in ['open', 'high', 'low', 'close']:
                if (ohlcv[col] <= 0).any():
                    raise ValidationError(f"Non-positive prices found in {col}")
        
        # Validate OHLC relationships
        if not (ohlcv['high'] >= ohlcv['low']).all():
            raise ValidationError("High prices are not always >= Low prices")
        
        if not ((ohlcv['high'] >= ohlcv['open']) & (ohlcv['high'] >= ohlcv['close'])).all():
            raise ValidationError("High prices are not always >= Open/Close prices")
        
        if not ((ohlcv['low'] <= ohlcv['open']) & (ohlcv['low'] <= ohlcv['close'])).all():
            raise ValidationError("Low prices are not always <= Open/Close prices")
    
    def validate_thresholds(self, buy_threshold: Optional[float], sell_threshold: Optional[float], 
                          technical_weight: Optional[float], sentiment_weight: Optional[float]) -> Tuple[float, float, float, float]:
        """
        Validate and normalize trading thresholds and weights.
        
        Returns:
            Tuple of (buy_threshold, sell_threshold, technical_weight, sentiment_weight)
        """
        # Use config defaults if not provided
        if buy_threshold is None:
            buy_threshold = self.config.thresholds.buy_threshold
        if sell_threshold is None:
            sell_threshold = self.config.thresholds.sell_threshold
        if technical_weight is None:
            technical_weight = self.config.thresholds.technical_weight
        if sentiment_weight is None:
            sentiment_weight = self.config.thresholds.sentiment_weight
        
        if not self.safety.validate_thresholds:
            return buy_threshold, sell_threshold, technical_weight, sentiment_weight
        
        # Validate thresholds
        if not (0 <= buy_threshold <= 1):
            raise ValidationError(f"Buy threshold must be between 0 and 1, got {buy_threshold}")
        
        if not (-1 <= sell_threshold <= 0):
            raise ValidationError(f"Sell threshold must be between -1 and 0, got {sell_threshold}")
        
        if buy_threshold <= abs(sell_threshold):
            raise ValidationError(f"Buy threshold ({buy_threshold}) must be > |sell_threshold| ({abs(sell_threshold)})")
        
        # Validate and normalize weights
        if not (0 <= technical_weight <= 1):
            raise ValidationError(f"Technical weight must be between 0 and 1, got {technical_weight}")
        
        if not (0 <= sentiment_weight <= 1):
            raise ValidationError(f"Sentiment weight must be between 0 and 1, got {sentiment_weight}")
        
        if self.safety.normalize_weights:
            total_weight = technical_weight + sentiment_weight
            if total_weight > 0:
                technical_weight = technical_weight / total_weight
                sentiment_weight = sentiment_weight / total_weight
                logger.debug(f"Normalized weights: tech={technical_weight:.3f}, sent={sentiment_weight:.3f}")
            else:
                raise ValidationError("Both weights cannot be zero")
        
        return buy_threshold, sell_threshold, technical_weight, sentiment_weight
    
    def validate_symbol_timeframe(self, symbol: str, timeframe: str):
        """Validate that symbol and timeframe are in allowed lists"""
        # Validate symbol
        all_symbols = self.config.universe.crypto_symbols + self.config.universe.stock_symbols
        if symbol not in all_symbols:
            raise ValidationError(f"Symbol {symbol} not in allowed list: {all_symbols}")
        
        # Validate timeframe
        if timeframe not in self.config.universe.timeframes:
            raise ValidationError(f"Timeframe {timeframe} not in allowed list: {self.config.universe.timeframes}")


class SentimentValidator:
    """Validates sentiment analysis data and implements deterministic sampling"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.safety = config.safety
    
    def deterministic_sample(self, texts: list, symbol: str, timeframe: str, 
                           latest_bar_time: datetime, sample_size: int = None) -> list:
        """
        Create deterministic sample of texts for reproducible sentiment analysis.
        
        Args:
            texts: List of text strings
            symbol: Trading symbol
            timeframe: Timeframe
            latest_bar_time: Timestamp of latest bar
            sample_size: Number of texts to sample (default: all)
            
        Returns:
            Deterministically sampled list of texts
        """
        if not self.safety.deterministic_sampling:
            # Use random sampling if deterministic sampling is disabled
            if sample_size and len(texts) > sample_size:
                return random.sample(texts, sample_size)
            return texts
        
        if not texts:
            return texts
        
        # Create deterministic seed from symbol, timeframe, and latest bar time
        seed_string = f"{symbol}_{timeframe}_{latest_bar_time.isoformat()}"
        seed = int(hashlib.md5(seed_string.encode()).hexdigest()[:8], 16)
        
        # Set random seed for reproducible sampling
        random.seed(seed)
        
        # Sample texts
        if sample_size and len(texts) > sample_size:
            sampled_texts = random.sample(texts, sample_size)
        else:
            sampled_texts = texts.copy()
        
        # Shuffle for deterministic order
        random.shuffle(sampled_texts)
        
        # Reset random seed to avoid affecting other parts of the system
        random.seed()
        
        logger.debug(f"Deterministic sampling: {len(sampled_texts)} texts from {len(texts)} total")
        return sampled_texts
    
    def validate_sentiment_score(self, score: float) -> float:
        """Validate sentiment score is in expected range"""
        if not isinstance(score, (int, float)):
            raise ValidationError(f"Sentiment score must be numeric, got {type(score)}")
        
        if np.isnan(score) or np.isinf(score):
            logger.warning(f"Invalid sentiment score: {score}, using 0.0")
            return 0.0
        
        # Clamp to reasonable range
        score = max(-1.0, min(1.0, score))
        return score


class TimeoutManager:
    """Manages timeouts for external I/O operations"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.safety = config.safety
    
    async def with_timeout(self, coro, timeout_seconds: int, operation_name: str):
        """Execute coroutine with timeout"""
        try:
            return await asyncio.wait_for(coro, timeout=timeout_seconds)
        except asyncio.TimeoutError:
            logger.error(f"Timeout after {timeout_seconds}s for {operation_name}")
            raise ValidationError(f"Timeout: {operation_name} took longer than {timeout_seconds} seconds")
        except Exception as e:
            logger.error(f"Error in {operation_name}: {e}")
            raise ValidationError(f"Error in {operation_name}: {str(e)}")


def safe_float(value: Any, default: float = 0.0) -> float:
    """
    Safely convert value to float with proper error handling.
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
        
    Returns:
        Float value or default
    """
    try:
        if value is None:
            return default
        
        # Handle pandas/numpy types
        if hasattr(value, 'item'):
            value = value.item()
        
        # Convert to float
        result = float(value)
        
        # Check for NaN or infinite
        if np.isnan(result) or np.isinf(result):
            logger.warning(f"Invalid numeric value: {value}, using default: {default}")
            return default
        
        return result
    except (ValueError, TypeError) as e:
        logger.warning(f"Failed to convert {value} to float: {e}, using default: {default}")
        return default


def redact_sensitive_data(text: str, max_length: int = 50) -> str:
    """
    Redact sensitive data from text for logging.
    
    Args:
        text: Text to redact
        max_length: Maximum length to show
        
    Returns:
        Redacted text safe for logging
    """
    if not text:
        return ""
    
    if len(text) <= max_length:
        return text
    
    # Show first and last few characters
    visible_length = max_length // 2
    return f"{text[:visible_length]}...{text[-visible_length:]}"
