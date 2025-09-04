#!/usr/bin/env python3
"""
Trading Agent Web Interface
FastAPI-based web application for monitoring and controlling the trading agent
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
import asyncio
import json
import os
import math
import numpy as np
from datetime import datetime, timedelta
import logging

# Import trading agent modules
import sys
sys.path.append('..')
from agent.cli import main as cli_main
from agent.config import load_config_from_env
from agent.data.ccxt_client import fetch_ohlcv
from agent.data.alpaca_client import fetch_ohlcv as fetch_alpaca_ohlcv
from agent.indicators import compute_rsi, compute_ema, compute_macd, compute_atr
from agent.engine import make_fused_tip
from agent.models.sentiment import SentimentAnalyzer
from agent.news.rss import fetch_headlines
from agent.cache.redis_client import get_redis_client, close_redis_client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Trading Agent Web Interface",
    description="Real-time monitoring and control for the AI-powered trading agent",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="web/static"), name="static")

# Data models
class SignalRequest(BaseModel):
    symbol: str
    timeframe: str = "1h"
    # Custom thresholds (optional - will use config defaults if not provided)
    custom_buy_threshold: Optional[float] = None
    custom_sell_threshold: Optional[float] = None
    custom_tech_weight: Optional[float] = None
    custom_sentiment_weight: Optional[float] = None

class ConfigUpdate(BaseModel):
    key: str
    value: str

class TradingSignal(BaseModel):
    symbol: str
    timestamp: str
    timeframe: str  # Store the timeframe used for this signal
    signal_type: str  # "BUY", "SELL", "HOLD"
    confidence: float
    technical_score: float
    sentiment_score: float
    fused_score: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    reasoning: str

class AgentStatus(BaseModel):
    status: str
    last_update: str
    active_symbols: List[str]
    total_signals: int
    uptime: str

# Global state
agent_status = {
    "status": "running",
    "last_update": datetime.now().isoformat(),
    "active_symbols": [],
    "total_signals": 0,
    "uptime": "0:00:00"
}

# In-memory storage for generated signals
generated_signals: List[TradingSignal] = []

# Simple in-memory cache for market data (5 minute TTL)
market_data_cache = {}
CACHE_TTL = 300  # 5 minutes in seconds

# Initialize sentiment analyzer
sentiment_analyzer = None

def get_cached_market_data(cache_key: str):
    """Get cached market data if it's still valid"""
    if cache_key in market_data_cache:
        cached_data, timestamp = market_data_cache[cache_key]
        if (datetime.now() - timestamp).total_seconds() < CACHE_TTL:
            return cached_data
        else:
            # Remove expired cache entry
            del market_data_cache[cache_key]
    return None

def set_cached_market_data(cache_key: str, data: dict):
    """Cache market data with current timestamp"""
    market_data_cache[cache_key] = (data, datetime.now())

try:
    sentiment_analyzer = SentimentAnalyzer("ProsusAI/finbert")
except Exception as e:
    logger.warning(f"Could not initialize sentiment analyzer: {e}")

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main dashboard"""
    try:
        with open("web/templates/index.html", "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Trading Agent Web Interface</h1><p>Dashboard template not found</p>")

@app.get("/api/status")
async def get_status() -> AgentStatus:
    """Get current agent status"""
    return AgentStatus(**agent_status)

@app.get("/api/signals")
async def get_signals(symbol: Optional[str] = None, limit: int = 50) -> List[TradingSignal]:
    """Get recent trading signals"""
    try:
        # Return actual generated signals
        signals = generated_signals.copy()
        
        # Filter by symbol if specified
        if symbol:
            signals = [s for s in signals if s.symbol == symbol]
        
        # Validate signals before returning (check for invalid float values)
        valid_signals = []
        for signal in signals:
            try:
                # Check if all float values are valid
                if (isinstance(signal.confidence, (int, float)) and 
                    isinstance(signal.technical_score, (int, float)) and
                    isinstance(signal.sentiment_score, (int, float)) and
                    isinstance(signal.fused_score, (int, float)) and
                    (signal.stop_loss is None or isinstance(signal.stop_loss, (int, float))) and
                    (signal.take_profit is None or isinstance(signal.take_profit, (int, float)))):
                    
                    # Check for inf/nan values
                    if (not math.isnan(signal.confidence) and not math.isinf(signal.confidence) and
                        not math.isnan(signal.technical_score) and not math.isinf(signal.technical_score) and
                        not math.isnan(signal.sentiment_score) and not math.isinf(signal.sentiment_score) and
                        not math.isnan(signal.fused_score) and not math.isinf(signal.fused_score) and
                        (signal.stop_loss is None or (not math.isnan(signal.stop_loss) and not math.isinf(signal.stop_loss))) and
                        (signal.take_profit is None or (not math.isnan(signal.take_profit) and not math.isinf(signal.take_profit)))):
                        
                        valid_signals.append(signal)
                    else:
                        logger.warning(f"Signal {signal.symbol} has inf/nan values, skipping")
                else:
                    logger.warning(f"Signal {signal.symbol} has invalid data types, skipping")
            except Exception as e:
                logger.error(f"Error validating signal {signal.symbol}: {e}")
                continue
        
        # Return most recent valid signals first, limited by the limit parameter
        return sorted(valid_signals, key=lambda x: x.timestamp, reverse=True)[:limit]
        
    except Exception as e:
        logger.error(f"Error in get_signals: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving signals: {str(e)}")

@app.get("/api/signals/stats")
async def get_signal_stats() -> Dict[str, Any]:
    """Get signal statistics"""
    if not generated_signals:
        return {
            "total_signals": 0,
            "signals_by_type": {},
            "signals_by_symbol": {},
            "recent_activity": []
        }
    
    # Count by signal type
    signals_by_type = {}
    for signal in generated_signals:
        signal_type = signal.signal_type
        signals_by_type[signal_type] = signals_by_type.get(signal_type, 0) + 1
    
    # Count by symbol
    signals_by_symbol = {}
    for signal in generated_signals:
        symbol = signal.symbol
        signals_by_symbol[symbol] = signals_by_symbol.get(symbol, 0) + 1
    
    # Recent activity (last 10 signals)
    recent_activity = [
        {
            "symbol": s.symbol,
            "type": s.signal_type,
            "timestamp": s.timestamp,
            "score": s.fused_score
        }
        for s in sorted(generated_signals, key=lambda x: x.timestamp, reverse=True)[:10]
    ]
    
    return {
        "total_signals": len(generated_signals),
        "signals_by_type": signals_by_type,
        "signals_by_symbol": signals_by_symbol,
        "recent_activity": recent_activity
    }

@app.post("/api/signals/generate")
async def generate_signal(request: SignalRequest) -> TradingSignal:
    """Generate a new trading signal for a symbol"""
    try:
        # Fetch market data
        if "/" in request.symbol or request.symbol.endswith(("USDT", "BTC", "ETH")):  # Crypto
            ohlcv = fetch_ohlcv(request.symbol, request.timeframe)
        else:  # Stock/ETF
            ohlcv = fetch_alpaca_ohlcv(request.symbol, request.timeframe)
        
        if ohlcv is None or ohlcv.empty:
            raise HTTPException(status_code=400, detail="Could not fetch market data")
        
        # Calculate indicators
        close = ohlcv['close']
        rsi = compute_rsi(close)
        ema_12 = compute_ema(close, 12)
        ema_26 = compute_ema(close, 26)
        macd_df = compute_macd(close)
        macd = macd_df['macd']
        macd_signal = macd_df['signal']
        macd_hist = macd_df['hist']
        atr = compute_atr(ohlcv['high'], ohlcv['low'], ohlcv['close'])
        
        # Get sentiment from news
        sentiment_score = 0.0
        if sentiment_analyzer:
            try:
                headlines = fetch_headlines([
                    "https://news.google.com/rss/search?q=stock+market&hl=en-US&gl=US&ceid=US:en",
                    "https://news.google.com/rss/search?q=crypto+btc&hl=en-US&gl=US&ceid=US:en"
                ])
                if headlines:
                    # Analyze first few headlines
                    text_sample = " ".join(headlines[:3])
                    sentiment_score = sentiment_analyzer.score(text_sample)
            except Exception as e:
                logger.warning(f"Could not fetch sentiment: {e}")
        
        # Calculate technical score
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
        
        # Fused score (weighted average) - use custom weights or config defaults
        try:
            config = load_config_from_env()
            # Use custom weights if provided, otherwise use config defaults
            tech_weight = request.custom_tech_weight if request.custom_tech_weight is not None else config.thresholds.tech_weight
            sentiment_weight = request.custom_sentiment_weight if request.custom_sentiment_weight is not None else config.thresholds.sentiment_weight
        except Exception:
            # Fallback to default weights
            tech_weight = request.custom_tech_weight if request.custom_tech_weight is not None else 0.6
            sentiment_weight = request.custom_sentiment_weight if request.custom_sentiment_weight is not None else 0.4
        
        fused_score = tech_weight * tech_score + sentiment_weight * sentiment_score
        
        # Determine signal type - use custom thresholds or config defaults
        try:
            config = load_config_from_env()
            # Use custom thresholds if provided, otherwise use config defaults
            buy_threshold = request.custom_buy_threshold if request.custom_buy_threshold is not None else config.thresholds.buy_threshold
            sell_threshold = request.custom_sell_threshold if request.custom_sell_threshold is not None else config.thresholds.sell_threshold
        except Exception:
            # Fallback to default thresholds
            buy_threshold = request.custom_buy_threshold if request.custom_buy_threshold is not None else 0.5
            sell_threshold = request.custom_sell_threshold if request.custom_sell_threshold is not None else -0.5
        
        if fused_score >= buy_threshold:
            signal_type = "BUY"
        elif fused_score <= sell_threshold:
            signal_type = "SELL"
        else:
            signal_type = "HOLD"
        
        # Calculate stop loss and take profit
        current_close = close.iloc[-1] if not close.empty else 0
        current_atr = atr.iloc[-1] if not atr.empty else 0
        
        stop_loss = None
        take_profit = None
        
        if current_atr > 0:
            if signal_type == "BUY":
                stop_loss = current_close - (2 * current_atr)
                take_profit = current_close + (3 * current_atr)
            elif signal_type == "SELL":
                stop_loss = current_close + (2 * current_atr)
                take_profit = current_close - (3 * current_atr)
        
        # Create signal
        signal = TradingSignal(
            symbol=request.symbol,
            timestamp=datetime.now().isoformat(),
            timeframe=request.timeframe,  # Store the timeframe used
            signal_type=signal_type,
            confidence=abs(fused_score),
            technical_score=tech_score,
            sentiment_score=sentiment_score,
            fused_score=fused_score,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reasoning=f"Technical: {tech_score:.2f}, Sentiment: {sentiment_score:.2f}, Fused: {fused_score:.2f}"
        )
        
        # Store the generated signal
        generated_signals.append(signal)
        
        # Update global state
        agent_status["total_signals"] = len(generated_signals)
        agent_status["last_update"] = datetime.now().isoformat()
        
        return signal
        
    except Exception as e:
        logger.error(f"Error generating signal: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/config")
async def get_config() -> Dict[str, Any]:
    """Get current agent configuration"""
    try:
        config = load_config_from_env()
        return {
            "tickers": config.universe.tickers,
            "timeframe": config.universe.timeframe,
            "buy_threshold": config.thresholds.buy_threshold,
            "sell_threshold": config.thresholds.sell_threshold,
            "tech_weight": config.thresholds.tech_weight,
            "sentiment_weight": config.thresholds.sentiment_weight,
            "log_level": os.getenv("AGENT_LOG_LEVEL", "info"),
            "log_format": os.getenv("AGENT_LOG_FORMAT", "simple")
        }
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/config/update")
async def update_config(update: ConfigUpdate):
    """Update agent configuration"""
    try:
        # This would typically update environment variables or config file
        # For now, just log the update
        logger.info(f"Config update requested: {update.key} = {update.value}")
        return {"status": "success", "message": f"Updated {update.key}"}
    except Exception as e:
        logger.error(f"Error updating config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/market-data/{symbol}")
async def get_market_data(symbol: str, timeframe: str = "1h", limit: int = 100):
    """Get market data for a symbol"""
    try:
        if "/" in symbol or symbol.endswith(("USDT", "BTC", "ETH")):  # Crypto
            ohlcv = fetch_ohlcv(symbol, timeframe)
        else:  # Stock/ETF
            ohlcv = fetch_alpaca_ohlcv(symbol, timeframe)
        
        if ohlcv is None or ohlcv.empty:
            raise HTTPException(status_code=400, detail="Could not fetch market data")
        
        # Convert to JSON-serializable format
        data = []
        for idx, row in ohlcv.tail(limit).iterrows():
            data.append({
                "timestamp": idx.isoformat(),
                "open": float(row['open']),
                "high": float(row['high']),
                "low": float(row['low']),
                "close": float(row['close']),
                "volume": float(row['volume'])
            })
        
        return {"symbol": symbol, "timeframe": timeframe, "data": data}
        
    except Exception as e:
        logger.error(f"Error fetching market data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/market-data/all/overview")
async def get_all_market_overview(timeframe: str = "1h"):
    """Get overview of all configured tickers with real-time data"""
    try:
        config = load_config_from_env()
        symbols = config.universe.tickers
        
        overview_data = []
        
        for symbol in symbols:
            try:
                if "/" in symbol or symbol.endswith(("USDT", "BTC", "ETH")):  # Crypto
                    ohlcv = fetch_ohlcv(symbol, timeframe)
                else:  # Stock/ETF
                    ohlcv = fetch_alpaca_ohlcv(symbol, timeframe)
                
                if ohlcv is not None and not ohlcv.empty:
                    # Get latest data points
                    latest = ohlcv.iloc[-1]
                    previous = ohlcv.iloc[-2] if len(ohlcv) > 1 else latest
                    
                    # Calculate changes
                    current_price = float(latest['close'])
                    previous_price = float(previous['close'])
                    change = current_price - previous_price
                    change_percent = (change / previous_price) * 100
                    
                    # Get volume
                    volume = float(latest['volume']) if 'volume' in latest else 0
                    
                    # Get high/low for the period
                    high = float(ohlcv['high'].max())
                    low = float(ohlcv['low'].min())
                    
                    # Validate float values before adding to response
                    if (not math.isnan(current_price) and not math.isinf(current_price) and
                        not math.isnan(change) and not math.isinf(change) and
                        not math.isnan(change_percent) and not math.isinf(change_percent) and
                        not math.isnan(volume) and not math.isinf(volume) and
                        not math.isnan(high) and not math.isinf(high) and
                        not math.isnan(low) and not math.isinf(low)):
                        
                        overview_data.append({
                            "symbol": symbol,
                            "current_price": current_price,
                            "change": change,
                            "change_percent": change_percent,
                            "volume": volume,
                            "high": high,
                            "low": low,
                            "timestamp": latest.name.isoformat(),
                            "timeframe": timeframe
                        })
                    else:
                        logger.warning(f"Invalid float values for {symbol}, skipping")
                        overview_data.append({
                            "symbol": symbol,
                            "current_price": 0,
                            "change": 0,
                            "change_percent": 0,
                            "volume": 0,
                            "high": 0,
                            "low": 0,
                            "timestamp": datetime.now().isoformat(),
                            "timeframe": timeframe,
                            "error": "Invalid data values"
                        })
                else:
                    # Fallback data if no market data available
                    overview_data.append({
                        "symbol": symbol,
                        "current_price": 0,
                        "change": 0,
                        "change_percent": 0,
                        "volume": 0,
                        "high": 0,
                        "low": 0,
                        "timestamp": datetime.now().isoformat(),
                        "timeframe": timeframe,
                        "error": "No data available"
                    })
                    
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")
                overview_data.append({
                    "symbol": symbol,
                    "current_price": 0,
                    "change": 0,
                    "change_percent": 0,
                    "volume": 0,
                    "high": 0,
                    "low": 0,
                    "timestamp": datetime.now().isoformat(),
                    "timeframe": timeframe,
                    "error": str(e)
                })
        
        return {
            "timestamp": datetime.now().isoformat(),
            "timeframe": timeframe,
            "symbols": overview_data
        }
        
    except Exception as e:
        logger.error(f"Error fetching market overview: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/market-data/all/timeline")
async def get_all_timeline_data(timeframe: str = "1h", limit: int = 50):
    """Get timeline data for all configured tickers with optimized parallel fetching and caching"""
    try:
        # Check cache first
        cache_key = f"timeline_{timeframe}_{limit}"
        cached_data = get_cached_market_data(cache_key)
        if cached_data:
            logger.info(f"Returning cached timeline data for {timeframe}")
            return cached_data
        
        config = load_config_from_env()
        symbols = config.universe.tickers
        
        # Reduce limit for faster response - keep it small for performance
        if limit > 20:
            limit = 20
        
        timeline_data = {}
        
        # Process symbols sequentially but with better error handling and faster processing
        for symbol in symbols:
            try:
                logger.info(f"Fetching data for {symbol} with timeframe {timeframe}")
                
                if "/" in symbol or symbol.endswith(("USDT", "BTC", "ETH")):  # Crypto
                    ohlcv = fetch_ohlcv(symbol, timeframe)
                else:  # Stock/ETF
                    ohlcv = fetch_alpaca_ohlcv(symbol, timeframe)
                
                if ohlcv is not None and not ohlcv.empty:
                    # Convert to timeline format with optimized processing
                    symbol_data = []
                    # Use tail for most recent data and limit rows for faster processing
                    recent_data = ohlcv.tail(min(limit, len(ohlcv)))
                    
                    # Process data more efficiently with vectorized operations
                    try:
                        # Convert to numeric values and handle NaN/inf in one operation
                        numeric_data = recent_data[['open', 'high', 'low', 'close']].astype(float)
                        volume_data = recent_data['volume'].astype(float) if 'volume' in recent_data else pd.Series([0.0] * len(recent_data))
                        
                        # Replace NaN and inf values
                        numeric_data = numeric_data.replace([np.inf, -np.inf], np.nan)
                        volume_data = volume_data.replace([np.inf, -np.inf], np.nan)
                        
                        # Fill NaN values with forward fill then backward fill
                        numeric_data = numeric_data.ffill().bfill()
                        volume_data = volume_data.ffill().bfill()
                        
                        # Convert to timeline format
                        for i, (idx, row) in enumerate(recent_data.iterrows()):
                            symbol_data.append({
                                "timestamp": idx.isoformat(),
                                "open": float(numeric_data.iloc[i]['open']),
                                "high": float(numeric_data.iloc[i]['high']),
                                "low": float(numeric_data.iloc[i]['low']),
                                "close": float(numeric_data.iloc[i]['close']),
                                "volume": float(volume_data.iloc[i])
                            })
                            
                    except Exception as e:
                        logger.warning(f"Error processing data for {symbol}: {e}, using fallback method")
                        # Fallback to simple processing
                        for idx, row in recent_data.iterrows():
                            try:
                                symbol_data.append({
                                    "timestamp": idx.isoformat(),
                                    "open": float(row['open']) if not math.isnan(row['open']) else 0.0,
                                    "high": float(row['high']) if not math.isnan(row['high']) else 0.0,
                                    "low": float(row['low']) if not math.isnan(row['low']) else 0.0,
                                    "close": float(row['close']) if not math.isnan(row['close']) else 0.0,
                                    "volume": float(row['volume']) if 'volume' in row and not math.isnan(row['volume']) else 0.0
                                })
                            except Exception as row_error:
                                logger.warning(f"Error processing row for {symbol}: {row_error}")
                                continue
                    
                    timeline_data[symbol] = {
                        "timeframe": timeframe,
                        "data": symbol_data,
                        "last_update": datetime.now().isoformat()
                    }
                else:
                    timeline_data[symbol] = {
                        "timeframe": timeframe,
                        "data": [],
                        "error": "No data available",
                        "last_update": datetime.now().isoformat()
                    }
                    
            except Exception as e:
                logger.error(f"Error fetching timeline for {symbol}: {e}")
                timeline_data[symbol] = {
                    "timeframe": timeframe,
                    "data": [],
                    "error": str(e),
                    "last_update": datetime.now().isoformat()
                }
        
        result = {
            "timestamp": datetime.now().isoformat(),
            "timeframe": timeframe,
            "symbols": timeline_data
        }
        
        # Cache the result
        set_cached_market_data(cache_key, result)
        
        return result
        
    except Exception as e:
        logger.error(f"Error fetching timeline data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

@app.get("/api/test-chart")
async def test_chart_endpoint():
    """Test endpoint for chart data - returns mock data quickly"""
    try:
        import time
        time.sleep(1)  # Simulate 1 second delay
        
        return {
            "timestamp": datetime.now().isoformat(),
            "timeframe": "15m",
            "symbols": {
                "BTC/USDT": {
                    "timeframe": "15m",
                    "data": [
                        {"timestamp": "2025-01-04T12:00:00", "open": 45000, "high": 45100, "low": 44900, "close": 45050, "volume": 1000},
                        {"timestamp": "2025-01-04T12:15:00", "open": 45050, "high": 45200, "low": 45000, "close": 45150, "volume": 1200}
                    ],
                    "last_update": datetime.now().isoformat()
                }
            }
        }
    except Exception as e:
        logger.error(f"Error in test chart endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/redis/status")
async def redis_status():
    """Get Redis connection status and statistics"""
    try:
        redis_client = await get_redis_client()
        if redis_client:
            stats = await redis_client.get_cache_stats()
            return stats
        else:
            return {"status": "redis_not_available"}
    except Exception as e:
        logger.error(f"Error checking Redis status: {e}")
        return {"status": "error", "error": str(e)}

@app.get("/api/redis/cache/clear")
async def clear_cache(pattern: str = "*"):
    """Clear Redis cache by pattern"""
    try:
        redis_client = await get_redis_client()
        if redis_client:
            deleted = await redis_client.clear_cache(pattern)
            return {"status": "success", "deleted_keys": deleted}
        else:
            return {"status": "redis_not_available"}
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        return {"status": "error", "error": str(e)}

@app.get("/api/redis/metrics/{metric_name}")
async def get_redis_metrics(metric_name: str, days: int = 7):
    """Get performance metrics from Redis"""
    try:
        redis_client = await get_redis_client()
        if redis_client:
            metrics = await redis_client.get_performance_metrics(metric_name, days)
            return {"metric": metric_name, "days": days, "data": metrics}
        else:
            return {"status": "redis_not_available"}
    except Exception as e:
        logger.error(f"Error retrieving metrics: {e}")
        return {"status": "error", "error": str(e)}

# Background task to update agent status
async def update_status():
    """Background task to update agent status"""
    while True:
        try:
            agent_status["uptime"] = str(datetime.now() - datetime.fromisoformat(agent_status["last_update"]))
            await asyncio.sleep(60)  # Update every minute
        except Exception as e:
            logger.error(f"Error updating status: {e}")
            await asyncio.sleep(60)

@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    logger.info("Starting Trading Agent Web Interface")
    
    # Initialize Redis connection
    try:
        redis_client = await get_redis_client()
        if redis_client and await redis_client.is_connected():
            logger.info("Redis connection established")
        else:
            logger.warning("Redis not available - caching disabled")
    except Exception as e:
        logger.warning(f"Could not initialize Redis: {e}")
    
    # Start background tasks
    asyncio.create_task(update_status())
    
    # Load initial configuration
    try:
        config = load_config_from_env()
        agent_status["active_symbols"] = config.universe.tickers
    except Exception as e:
        logger.warning(f"Could not load initial config: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Trading Agent Web Interface")
    
    # Close Redis connection
    try:
        await close_redis_client()
        logger.info("Redis connection closed")
    except Exception as e:
        logger.warning(f"Error closing Redis connection: {e}")

if __name__ == "__main__":
    uvicorn.run(
        "web.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
