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

class ConfigUpdate(BaseModel):
    key: str
    value: str

class TradingSignal(BaseModel):
    symbol: str
    timestamp: str
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

# Initialize sentiment analyzer
sentiment_analyzer = None
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
    # This would typically query a database
    # For now, return mock data
    mock_signals = [
        TradingSignal(
            symbol="BTC/USDT",
            timestamp=datetime.now().isoformat(),
            signal_type="BUY",
            confidence=0.85,
            technical_score=0.8,
            sentiment_score=0.7,
            fused_score=0.75,
            stop_loss=45000.0,
            take_profit=52000.0,
            reasoning="Strong technical momentum with positive sentiment"
        )
    ]
    
    if symbol:
        mock_signals = [s for s in mock_signals if s.symbol == symbol]
    
    return mock_signals[:limit]

@app.post("/api/signals/generate")
async def generate_signal(request: SignalRequest) -> TradingSignal:
    """Generate a new trading signal for a symbol"""
    try:
        # Fetch market data
        if "/" in request.symbol:  # Crypto
            ohlcv = await fetch_ohlcv(request.symbol, request.timeframe)
        else:  # Stock/ETF
            ohlcv = await fetch_alpaca_ohlcv(request.symbol, request.timeframe)
        
        if ohlcv is None or ohlcv.empty:
            raise HTTPException(status_code=400, detail="Could not fetch market data")
        
        # Calculate indicators
        close = ohlcv['close']
        rsi = compute_rsi(close)
        ema_12 = compute_ema(close, 12)
        ema_26 = compute_ema(close, 26)
        macd, macd_signal, macd_hist = compute_macd(close)
        atr = compute_atr(ohlcv)
        
        # Get sentiment from news
        sentiment_score = 0.0
        if sentiment_analyzer:
            try:
                headlines = await fetch_headlines()
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
        
        # Fused score (weighted average)
        tech_weight = 0.6
        sentiment_weight = 0.4
        fused_score = tech_weight * tech_score + sentiment_weight * sentiment_score
        
        # Determine signal type
        if fused_score >= 0.7:
            signal_type = "BUY"
        elif fused_score <= -0.7:
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
            signal_type=signal_type,
            confidence=abs(fused_score),
            technical_score=tech_score,
            sentiment_score=sentiment_score,
            fused_score=fused_score,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reasoning=f"Technical: {tech_score:.2f}, Sentiment: {sentiment_score:.2f}, Fused: {fused_score:.2f}"
        )
        
        # Update global state
        agent_status["total_signals"] += 1
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
        if "/" in symbol:  # Crypto
            ohlcv = await fetch_ohlcv(symbol, timeframe)
        else:  # Stock/ETF
            ohlcv = await fetch_alpaca_ohlcv(symbol, timeframe)
        
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

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

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
