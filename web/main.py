#!/usr/bin/env python3
"""
Trading Agent Web Interface
FastAPI-based web application for monitoring and controlling the trading agent
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
import asyncio
import json
import os
from datetime import datetime, timedelta
import logging
import hashlib
import secrets

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

@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    await initialize_default_admin()

# Security
security = HTTPBearer()

# Data models
class UserLogin(BaseModel):
    username: str
    password: str

class UserCreate(BaseModel):
    username: str
    password: str
    role: str = "user"

class User(BaseModel):
    username: str
    role: str
    status: str
    created_at: str
    last_login: Optional[str] = None

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
    # Store the thresholds that were actually used for this signal
    applied_buy_threshold: Optional[float] = None
    applied_sell_threshold: Optional[float] = None
    applied_tech_weight: Optional[float] = None
    applied_sentiment_weight: Optional[float] = None

class AgentStatus(BaseModel):
    status: str
    last_update: str
    active_symbols: List[str]
    total_signals: int
    uptime: str

# JWT tokens (in production, use proper JWT library)
ACTIVE_TOKENS = {}

# Initialize default admin user in Redis on startup
async def initialize_default_admin():
    """Initialize default admin user in Redis if it doesn't exist"""
    try:
        redis_client = await get_redis_client()
        if redis_client:
            # Check if admin user exists
            admin_user = await redis_client.get_user("admin")
            if not admin_user:
                # Create default admin user
                admin_data = {
                    "username": "admin",
                    "password_hash": hashlib.sha256("admin123".encode()).hexdigest(),
                    "role": "admin",
                    "status": "active",
                    "created_at": datetime.now().isoformat(),
                    "last_login": None
                }
                await redis_client.store_user("admin", admin_data)
                logger.info("Default admin user created in Redis")
            else:
                logger.info("Admin user already exists in Redis")
    except Exception as e:
        logger.error(f"Failed to initialize default admin user: {e}")

# User management functions
async def get_user_from_redis(username: str) -> Optional[Dict[str, Any]]:
    """Get user from Redis"""
    try:
        redis_client = await get_redis_client()
        if redis_client:
            return await redis_client.get_user(username)
        return None
    except Exception as e:
        logger.error(f"Failed to get user {username} from Redis: {e}")
        return None

async def get_all_users_from_redis() -> List[Dict[str, Any]]:
    """Get all users from Redis"""
    try:
        redis_client = await get_redis_client()
        if redis_client:
            return await redis_client.get_all_users()
        return []
    except Exception as e:
        logger.error(f"Failed to get all users from Redis: {e}")
        return []

async def store_user_in_redis(username: str, user_data: Dict[str, Any]) -> bool:
    """Store user in Redis"""
    try:
        redis_client = await get_redis_client()
        if redis_client:
            return await redis_client.store_user(username, user_data)
        return False
    except Exception as e:
        logger.error(f"Failed to store user {username} in Redis: {e}")
        return False

def create_access_token(username: str, role: str) -> str:
    """Create a simple access token (in production, use proper JWT)"""
    token = secrets.token_urlsafe(32)
    ACTIVE_TOKENS[token] = {
        "username": username,
        "role": role,
        "expires": datetime.now() + timedelta(hours=24)
    }
    return token

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """Verify the access token"""
    token = credentials.credentials
    if token not in ACTIVE_TOKENS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token_data = ACTIVE_TOKENS[token]
    if datetime.now() > token_data["expires"]:
        del ACTIVE_TOKENS[token]
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return token_data

# Routes
@app.get("/")
async def root():
    """Redirect to login page"""
    return HTMLResponse("""
    <html>
        <head>
            <meta http-equiv="refresh" content="0; url=/login">
        </head>
        <body>
            <p>Redirecting to <a href="/login">login</a>...</p>
        </body>
    </html>
    """)

@app.get("/login")
async def login_page():
    """Serve login page"""
    with open("web/templates/login.html", "r") as f:
        return HTMLResponse(f.read())

@app.get("/dashboard")
async def dashboard_page():
    """Serve dashboard page (requires authentication)"""
    with open("web/templates/index.html", "r") as f:
        return HTMLResponse(f.read())

@app.get("/admin")
async def admin_page():
    """Serve admin page (requires admin authentication)"""
    with open("web/templates/admin.html", "r") as f:
        return HTMLResponse(f.read())

# Authentication endpoints
@app.post("/api/auth/login")
async def login(user_data: UserLogin):
    """User login"""
    username = user_data.username
    password = user_data.password
    
    # Get user from Redis
    user = await get_user_from_redis(username)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    
    if user["password_hash"] != password_hash:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Update last login in Redis
    try:
        redis_client = await get_redis_client()
        if redis_client:
            await redis_client.update_user_last_login(username)
    except Exception as e:
        logger.warning(f"Failed to update last login for {username}: {e}")
    
    # Create access token
    access_token = create_access_token(username, user["role"])
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "username": username,
        "role": user["role"]
    }

@app.post("/api/auth/logout")
async def logout(current_user: Dict[str, Any] = Depends(verify_token)):
    """User logout"""
    # In production, you might want to blacklist the token
    return {"message": "Logged out successfully"}

# Admin endpoints
@app.get("/api/admin/dashboard")
async def admin_dashboard(current_user: Dict[str, Any] = Depends(verify_token)):
    """Get admin dashboard data"""
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    # Get user stats from Redis
    try:
        all_users = await get_all_users_from_redis()
        total_users = len(all_users)
        active_users = len([u for u in all_users if u["status"] == "active"])
    except Exception:
        total_users = 0
        active_users = 0
    
    # Get signal stats from Redis
    try:
        redis_client = await get_redis_client()
        if redis_client:
            total_signals = await redis_client.get_signal_count()
            # Get today's signals
            today = datetime.now().date()
            all_signals = await redis_client.get_signals(1000)
            today_signals = len([s for s in all_signals if datetime.fromisoformat(s['timestamp']).date() == today])
        else:
            total_signals = 0
            today_signals = 0
    except Exception:
        total_signals = 0
        today_signals = 0
    
    return {
        "total_users": total_users,
        "active_users": active_users,
        "total_signals": total_signals,
        "today_signals": today_signals
    }

@app.get("/api/user/dashboard")
async def user_dashboard(current_user: Dict[str, Any] = Depends(verify_token)):
    """Get user dashboard data (shows only user's own signals)"""
    
    # Get user's own signal stats from Redis
    try:
        redis_client = await get_redis_client()
        if redis_client:
            user_signals = await redis_client.get_signals(1000, username=current_user['username'])
            total_user_signals = len(user_signals)
            
            # Get today's signals for this user
            today = datetime.now().date()
            today_user_signals = len([s for s in user_signals if datetime.fromisoformat(s['timestamp']).date() == today])
        else:
            total_user_signals = 0
            today_user_signals = 0
    except Exception:
        total_user_signals = 0
        today_user_signals = 0
    
    return {
        "username": current_user["username"],
        "role": current_user["role"],
        "total_signals": total_user_signals,
        "today_signals": today_user_signals
    }

@app.get("/api/admin/users")
async def admin_get_users(current_user: Dict[str, Any] = Depends(verify_token)):
    """Get all users (admin only)"""
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        all_users = await get_all_users_from_redis()
        users = []
        for user_data in all_users:
            users.append({
                "username": user_data["username"],
                "role": user_data["role"],
                "status": user_data["status"],
                "created_at": user_data["created_at"],
                "last_login": user_data["last_login"]
            })
        return users
    except Exception as e:
        logger.error(f"Failed to get users: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve users")

@app.post("/api/admin/users")
async def admin_create_user(user_data: UserCreate, current_user: Dict[str, Any] = Depends(verify_token)):
    """Create new user (admin only)"""
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    username = user_data.username
    
    # Check if user already exists in Redis
    existing_user = await get_user_from_redis(username)
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already exists")
    
    # Create new user
    password_hash = hashlib.sha256(user_data.password.encode()).hexdigest()
    new_user = {
        "username": username,
        "password_hash": password_hash,
        "role": user_data.role,
        "status": "active",
        "created_at": datetime.now().isoformat(),
        "last_login": None
    }
    
    # Store user in Redis
    if await store_user_in_redis(username, new_user):
        return {"message": "User created successfully", "username": username}
    else:
        raise HTTPException(status_code=500, detail="Failed to create user")

@app.delete("/api/admin/users/{username}/signals")
async def admin_clear_user_signals(username: str, current_user: Dict[str, Any] = Depends(verify_token)):
    """Clear all signals for a specific user (admin only)"""
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    if username not in USERS_DB:
        raise HTTPException(status_code=404, detail="User not found")
    
    try:
        redis_client = await get_redis_client()
        if redis_client:
            # Clear user-specific signals
            user_key = f"signals:user:{username}"
            cleared_count = await redis_client.clear_user_signals(username)
            logger.info(f"Cleared {cleared_count} signals for user {username}")
            return {"message": f"Cleared {cleared_count} signals for user {username}", "cleared_count": cleared_count}
        else:
            raise HTTPException(status_code=500, detail="Redis not available")
    except Exception as e:
        logger.error(f"Failed to clear signals for user {username}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear signals: {str(e)}")

# Global state
agent_status = {
    "status": "running",
    "last_update": datetime.now().isoformat(),
    "active_symbols": [],
    "total_signals": 0,
    "uptime": "0:00:00"
}

# Simple in-memory cache for market data (5 minute TTL)
market_data_cache = {}
CACHE_TTL = 300  # 5 minutes in seconds


# Initialize sentiment analyzer
sentiment_analyzer = None
try:
    sentiment_analyzer = SentimentAnalyzer("ProsusAI/finbert")
except Exception as e:
    logger.warning(f"Could not initialize sentiment analyzer: {e}")



@app.get("/api/status")
async def get_status(request: Request) -> AgentStatus:
    """Get current agent status (user-specific if authenticated, global if not)"""
    # Create a copy of the global status to avoid modifying it
    status_data = agent_status.copy()
    
    try:
        # Check if user is authenticated
        current_user = None
        try:
            auth_header = request.headers.get("Authorization")
            if auth_header and auth_header.startswith("Bearer "):
                token = auth_header.split(" ")[1]
                # Verify token manually
                if token in ACTIVE_TOKENS:
                    token_data = ACTIVE_TOKENS[token]
                    if datetime.now() < token_data["expires"]:  # Fixed: use "expires" not "expires_at"
                        current_user = {"username": token_data["username"], "role": token_data["role"]}
        except Exception:
            # If token verification fails, treat as unauthenticated
            current_user = None
        
        # Update signal count from Redis if available
        redis_client = await get_redis_client()
        if redis_client:
            if current_user:
                # For authenticated users, show only their own signal count
                user_signals = await redis_client.get_signals(1000, username=current_user['username'])
                status_data["total_signals"] = len(user_signals)
                logger.debug(f"User {current_user['username']} sees {len(user_signals)} signals")
            else:
                # For unauthenticated requests, show global count
                status_data["total_signals"] = await redis_client.get_signal_count()
                logger.debug(f"Unauthenticated request sees {status_data['total_signals']} signals")
    except Exception as e:
        logger.warning(f"Failed to update signal count from Redis: {e}")
    
    return AgentStatus(**status_data)

@app.get("/api/signals")
async def get_signals(symbol: Optional[str] = None, limit: int = 50, current_user: Dict[str, Any] = Depends(verify_token)) -> List[TradingSignal]:
    """Get recent trading signals from Redis (user-specific)"""
    try:
        redis_client = await get_redis_client()
        if redis_client:
            # Get signals from Redis for current user
            signal_dicts = await redis_client.get_signals(limit, symbol, current_user['username'])
            
            # Convert dicts back to TradingSignal objects
            valid_signals = []
            for signal_dict in signal_dicts:
                try:
                    # Handle both old and new signal formats
                    if 'applied_buy_threshold' not in signal_dict:
                        signal_dict['applied_buy_threshold'] = None
                        signal_dict['applied_sell_threshold'] = None
                        signal_dict['applied_tech_weight'] = None
                        signal_dict['applied_sentiment_weight'] = None
                    
                    signal = TradingSignal(**signal_dict)
                    
                    # Validate signal data
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
                    logger.warning(f"Failed to convert signal dict: {e}")
                    continue
            
            return valid_signals
        else:
            logger.warning("Redis not available - returning empty signals list")
            return []
        
    except Exception as e:
        logger.error(f"Error getting signals: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving signals: {str(e)}")

@app.get("/api/signals/stats")
async def get_signal_stats() -> Dict[str, Any]:
    """Get signal statistics from Redis"""
    try:
        redis_client = await get_redis_client()
        if not redis_client:
            return {
                "total_signals": 0,
                "signals_by_type": {},
                "signals_by_symbol": {},
                "recent_activity": []
            }
        
        # Get all signals from Redis
        all_signals = await redis_client.get_signals(1000)  # Get up to 1000 signals for stats
        
        if not all_signals:
            return {
                "total_signals": 0,
                "signals_by_type": {},
                "signals_by_symbol": {},
                "recent_activity": []
            }
        
        # Count by signal type
        signals_by_type = {}
        for signal in all_signals:
            signal_type = signal['signal_type']
            signals_by_type[signal_type] = signals_by_type.get(signal_type, 0) + 1
        
        # Count by symbol
        signals_by_symbol = {}
        for signal in all_signals:
            symbol = signal['symbol']
            signals_by_symbol[symbol] = signals_by_symbol.get(symbol, 0) + 1
        
        # Recent activity (last 10 signals)
        recent_activity = [
            {
                "symbol": s['symbol'],
                "type": s['signal_type'],
                "timestamp": s['timestamp'],
                "score": s['fused_score']
            }
            for s in sorted(all_signals, key=lambda x: x['timestamp'], reverse=True)[:10]
        ]
        
        return {
            "total_signals": len(all_signals),
            "signals_by_type": signals_by_type,
            "signals_by_symbol": signals_by_symbol,
            "recent_activity": recent_activity
        }
    except Exception as e:
        logger.error(f"Error getting signal stats: {e}")
        return {
            "total_signals": 0,
            "signals_by_type": {},
            "signals_by_symbol": {},
            "recent_activity": []
        }


@app.post("/api/signals/generate")
async def generate_signal(request: SignalRequest, current_user: Dict[str, Any] = Depends(verify_token)) -> TradingSignal:
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
            reasoning=f"Technical: {tech_score:.2f}, Sentiment: {sentiment_score:.2f}, Fused: {fused_score:.2f}",
            applied_buy_threshold=buy_threshold,
            applied_sell_threshold=sell_threshold,
            applied_tech_weight=tech_weight,
            applied_sentiment_weight=sentiment_weight
        )
        
        # Add user information to signal
        signal_dict = signal.dict()
        signal_dict['username'] = current_user['username']
        
        # Store the generated signal in Redis
        try:
            redis_client = await get_redis_client()
            if redis_client:
                # Use the signal_dict that already has username added
                await redis_client.store_signal(signal_dict)
                logger.info(f"Signal stored in Redis for user {current_user['username']}: {signal.symbol}")
            else:
                logger.warning("Redis not available - signal not persisted")
        except Exception as e:
            logger.error(f"Failed to store signal in Redis: {e}")
        
        # Update global state
        try:
            redis_client = await get_redis_client()
            if redis_client:
                agent_status["total_signals"] = await redis_client.get_signal_count()
            else:
                agent_status["total_signals"] += 1
        except Exception as e:
            logger.error(f"Failed to update signal count: {e}")
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
