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
import math
from datetime import datetime, timedelta
import pytz

# Application version
APP_VERSION = "1.0.0"

# Application startup time
APP_START_TIME = datetime.now()

# Israel timezone
ISRAEL_TZ = pytz.timezone('Asia/Jerusalem')

def get_israel_time():
    """Get current time in Israel timezone"""
    return datetime.now(ISRAEL_TZ)
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
from agent.news.reddit import fetch_crypto_reddit_posts, fetch_stock_reddit_posts
from agent.cache.redis_client import get_redis_client, close_redis_client
from agent.monitor import SignalMonitor
from agent.positions import PositionTracker
from agent.scheduler import start_scheduler

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

# Startup event moved to end of file to avoid conflicts

# Security
security = HTTPBearer()

# Data models
class UserLogin(BaseModel):
    username: str
    password: str

class SystemConfig(BaseModel):
    # Trading Agent Settings
    active_markets: List[str] = ["BTC/USD", "ETH/USD", "AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "NVDA"]
    signal_generation_enabled: bool = True
    generation_frequency_minutes: int = 30
    default_timeframe: str = "1h"
    
    # Technical Analysis Settings
    buy_threshold: float = 0.7
    sell_threshold: float = -0.7
    technical_weight: float = 0.6
    sentiment_weight: float = 0.4
    
    # Data Sources
    data_provider: str = "alpaca"
    data_refresh_rate_minutes: int = 5
    historical_data_days: int = 30
    
    # News Sources
    news_weight: float = 0.3
    news_keywords: List[str] = ["bitcoin", "ethereum", "crypto", "stock", "market", "trading"]
    
    # Security Settings
    session_timeout_hours: int = 24
    max_login_attempts: int = 5
    password_min_length: int = 8
    api_rate_limit_per_minute: int = 100
    
    # UI Settings
    auto_refresh_enabled: bool = True
    refresh_interval_seconds: int = 30
    theme: str = "light"
    date_format: str = "YYYY-MM-DD"
    timezone: str = "UTC"
    currency: str = "USD"
    
    # Trading Configuration
    max_position_size: float = 10000.0
    default_stop_loss_percent: float = 5.0
    default_take_profit_percent: float = 10.0
    risk_per_trade_percent: float = 2.0
    
    # Notifications
    email_notifications_enabled: bool = False
    smtp_server: str = ""
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    alert_recipients: List[str] = []
    
    # System Settings
    log_level: str = "INFO"
    cache_ttl_hours: int = 24
    backup_enabled: bool = True
    backup_frequency_hours: int = 24

class ConfigUpdate(BaseModel):
    key: str
    value: Any
    category: str

class UserCreate(BaseModel):
    username: str
    password: str
    role: str = "user"
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    additional_info: Optional[str] = None
    activation_days: Optional[int] = 30  # Default 30 days activation period

class UserUpdate(BaseModel):
    username: str
    password: Optional[str] = None
    role: str = "user"
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    additional_info: Optional[str] = None
    activation_days: Optional[int] = None  # For extending activation period

class UserProfileUpdate(BaseModel):
    password: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    additional_info: Optional[str] = None

class User(BaseModel):
    username: str
    role: str
    status: str
    created_at: str
    last_login: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    additional_info: Optional[str] = None
    activation_expires_at: Optional[str] = None
    activation_days: Optional[int] = None

class UserActivationExtension(BaseModel):
    username: str
    additional_days: int
    reason: Optional[str] = None

class SignalRequest(BaseModel):
    symbol: str
    timeframe: str = "1h"
    buy_threshold: Optional[float] = None
    sell_threshold: Optional[float] = None
    technical_weight: Optional[float] = None
    sentiment_weight: Optional[float] = None

class ConfigUpdate(BaseModel):
    key: str
    value: str

class TradingSignal(BaseModel):
    symbol: str
    timeframe: str = "1h"  # Add timeframe field
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
    total_trading_tips: int
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
            
            # Log system startup
            await redis_client.log_activity(
                "system_startup",
                "Trading AI Tips system started successfully",
                "system",
                {"version": "1.0.0", "startup_time": datetime.now().isoformat()}
            )
    except Exception as e:
        logger.error(f"Failed to initialize default admin user: {e}")

async def start_signal_monitoring():
    """Start the signal monitoring scheduler"""
    try:
        logger.info("Starting signal monitoring scheduler...")
        
        # Create monitoring job function (synchronous wrapper for async function)
        def monitoring_job():
            try:
                import asyncio
                import concurrent.futures
                
                # Create a new event loop in a separate thread
                def run_async():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        return loop.run_until_complete(run_monitoring_cycle())
                    finally:
                        loop.close()
                
                # Run in a thread pool to avoid blocking
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(run_async)
                    future.result(timeout=60)  # 60 second timeout
                    
            except Exception as e:
                logger.error(f"Error starting monitoring job: {e}")
        
        # Start scheduler to run every 2 minutes
        scheduler = start_scheduler(monitoring_job, cron="*/2 * * * *")
        logger.info("Signal monitoring scheduler started - running every 2 minutes")
        
    except Exception as e:
        logger.error(f"Failed to start signal monitoring: {e}")

async def run_monitoring_cycle():
    """Run a single monitoring cycle"""
    try:
        redis_client = await get_redis_client()
        if not redis_client:
            logger.warning("Redis not available for monitoring")
            return
        
        logger.info(f"Redis client connected: {await redis_client.is_connected()}")
        
        # Test signal retrieval directly
        test_signals = await redis_client.get_signals(limit=10)
        logger.info(f"Direct signal retrieval test: {len(test_signals) if test_signals else 0} signals")
        
        # Get sentiment analyzer (lazy initialization)
        analyzer = get_sentiment_analyzer()
        
        # Create monitor and run monitoring
        monitor = SignalMonitor(redis_client, analyzer)
        result = await monitor.monitor_all_signals()
        
        logger.info(f"Monitoring cycle completed: {result}")
        
    except Exception as e:
        logger.error(f"Error in monitoring cycle: {e}")

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
    
    # Handle both old and new password field formats
    stored_password_hash = user.get("password_hash") or user.get("password")
    if not stored_password_hash or stored_password_hash != password_hash:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Check if user is active (considering activation period)
    user_status = get_user_status_with_activation(user)
    if user_status != "active":
        if not is_user_activation_valid(user.get("activation_expires_at"), user.get("activation_days")):
            if user.get("activation_days") == 0:
                raise HTTPException(status_code=403, detail="Account has been deactivated. Please contact administrator to reactivate your access.")
            else:
                raise HTTPException(status_code=403, detail="Account activation period has expired. Please contact administrator to extend your access.")
        else:
            raise HTTPException(status_code=403, detail="Account is inactive")
    
    # Update last login in Redis
    try:
        redis_client = await get_redis_client()
        if redis_client:
            await redis_client.update_user_last_login(username)
    except Exception as e:
        logger.warning(f"Failed to update last login for {username}: {e}")
    
    # Create access token
    access_token = create_access_token(username, user["role"])
    
    # Log activity
    redis_client = await get_redis_client()
    if redis_client:
        await redis_client.log_activity(
            "user_login",
            f"User '{username}' logged in successfully",
            username,
            {"username": username, "role": user["role"]}
        )
    
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
        inactive_users = len([u for u in all_users if u["status"] == "inactive"])
        expired_users = len([u for u in all_users if u["status"] == "expired"])
    except Exception:
        total_users = 0
        active_users = 0
        inactive_users = 0
        expired_users = 0
    
    # Get signal stats from Redis
    try:
        redis_client = await get_redis_client()
        if redis_client:
            total_signals = await redis_client.get_signal_count()
            logger.info(f"Admin dashboard: total_signals = {total_signals}")
            # Get today's signals
            today = datetime.now().date()
            all_signals = await redis_client.get_signals(1000)
            logger.info(f"Admin dashboard: all_signals count = {len(all_signals)}")
            today_signals = len([s for s in all_signals if datetime.fromisoformat(s['timestamp']).date() == today])
            logger.info(f"Admin dashboard: today_signals = {today_signals}")
        else:
            logger.warning("Admin dashboard: Redis client not available")
            total_signals = 0
            today_signals = 0
    except Exception as e:
        logger.error(f"Admin dashboard: Exception getting signal stats: {e}")
        total_signals = 0
        today_signals = 0
    
    # Get system information
    import psutil
    import platform
    
    try:
        # System uptime
        boot_time = datetime.fromtimestamp(psutil.boot_time())
        uptime = datetime.now() - boot_time
        uptime_str = str(uptime).split('.')[0]  # Remove microseconds
        
        # System resources
        cpu_percent = psutil.cpu_percent(interval=0.1)  # Reduced interval for faster response
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Application uptime (since last restart)
        app_uptime = datetime.now() - APP_START_TIME
        app_uptime_str = str(app_uptime).split('.')[0]
        
        system_info = {
            "system_uptime": uptime_str,
            "app_uptime": app_uptime_str,
            "cpu_usage": round(cpu_percent, 1),
            "memory_usage": round(memory.percent, 1),
            "memory_available": f"{memory.available // (1024**3)} GB",
            "disk_usage": round(disk.percent, 1),
            "disk_free": f"{disk.free // (1024**3)} GB",
            "platform": platform.system(),
            "python_version": platform.python_version(),
            "app_version": APP_VERSION,
            "server_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        logger.warning(f"Could not get system information: {e}")
        system_info = {
            "system_uptime": "Unknown",
            "app_uptime": "Unknown",
            "cpu_usage": 0,
            "memory_usage": 0,
            "memory_available": "Unknown",
            "disk_usage": 0,
            "disk_free": "Unknown",
            "platform": "Unknown",
            "python_version": "Unknown",
            "app_version": APP_VERSION,
            "server_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    # Get Redis connection status
    try:
        redis_client = await get_redis_client()
        redis_status = "Connected" if redis_client and await redis_client.is_connected() else "Disconnected"
    except Exception:
        redis_status = "Error"
    
    return {
        "total_users": total_users,
        "active_users": active_users,
        "inactive_users": inactive_users,
        "expired_users": expired_users,
        "total_trading_tips": total_signals,
        "today_trading_tips": today_signals,
        "system_info": system_info,
        "redis_status": redis_status,
        "last_updated": datetime.now().isoformat()
    }

@app.get("/api/admin/activities")
async def admin_get_activities(limit: int = 20, current_user: Dict[str, Any] = Depends(verify_token)):
    """Get recent activities (admin only)"""
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        redis_client = await get_redis_client()
        if not redis_client:
            raise HTTPException(status_code=500, detail="Redis not available")
        
        activities = await redis_client.get_recent_activities(limit)
        return {"activities": activities}
    except Exception as e:
        logger.error(f"Failed to get recent activities: {e}")
        raise HTTPException(status_code=500, detail="Failed to get activities")

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
        "total_trading_tips": total_user_signals,
        "today_trading_tips": today_user_signals
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
            # Get actual status considering activation period
            actual_status = get_user_status_with_activation(user_data)
            
            users.append({
                "username": user_data["username"],
                "role": user_data["role"],
                "status": actual_status,
                "created_at": user_data["created_at"],
                "last_login": user_data["last_login"],
                "first_name": user_data.get("first_name"),
                "last_name": user_data.get("last_name"),
                "email": user_data.get("email"),
                "phone": user_data.get("phone"),
                "additional_info": user_data.get("additional_info"),
                "activation_expires_at": user_data.get("activation_expires_at"),
                "activation_days": user_data.get("activation_days")
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
    
    # Calculate activation expiry (only for non-admin users)
    activation_expires_at = None
    activation_days = user_data.activation_days or 30
    
    if user_data.role != "admin":
        activation_expires_at = calculate_activation_expiry(activation_days)
        # If activation_days is 0, user will be immediately deactivated
    
    new_user = {
        "username": username,
        "password_hash": password_hash,
        "role": user_data.role,
        "status": "active",
        "created_at": datetime.now().isoformat(),
        "last_login": None,
        "first_name": user_data.first_name,
        "last_name": user_data.last_name,
        "email": user_data.email,
        "phone": user_data.phone,
        "additional_info": user_data.additional_info,
        "activation_expires_at": activation_expires_at,
        "activation_days": activation_days
    }
    
    # Debug logging
    logger.info(f"Creating user {username} with data: {new_user}")
    
    # Store user in Redis
    if await store_user_in_redis(username, new_user):
        # Log activity
        redis_client = await get_redis_client()
        if redis_client:
            await redis_client.log_activity(
                "user_created",
                f"Created new user '{username}' with role '{user_data.role}'",
                current_user["username"],
                {"username": username, "role": user_data.role}
            )
        
        return {"message": "User created successfully", "username": username}
    else:
        raise HTTPException(status_code=500, detail="Failed to create user")

@app.get("/api/admin/users/{username}/signals")
async def admin_get_user_signals(username: str, current_user: Dict[str, Any] = Depends(verify_token)):
    """Get all signals for a specific user with configurations (admin only)"""
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        # Check if user exists
        user_data = await get_user_from_redis(username)
        if not user_data:
            raise HTTPException(status_code=404, detail="User not found")
        
        redis_client = await get_redis_client()
        if not redis_client:
            raise HTTPException(status_code=500, detail="Redis not available")
        
        # Get all signals for the user
        signals = await redis_client.get_signals(1000, username=username)
        
        # Format signals with detailed configuration information
        formatted_signals = []
        for signal in signals:
            formatted_signal = {
                "id": signal.get('timestamp', 'unknown'),
                "market": signal.get('symbol', 'unknown'),
                "timeframe": signal.get('timeframe', '1h'),
                "recommendation": signal.get('signal_type', 'HOLD'),
                "confidence": signal.get('confidence', 0.0),
                "technical_score": signal.get('technical_score', 0.0),
                "sentiment_score": signal.get('sentiment_score', 0.0),
                "fused_score": signal.get('fused_score', 0.0),
                "reasoning": signal.get('reasoning', 'No reasoning provided'),
                "generated_at": signal.get('timestamp', ''),
                "configuration": {
                    "buy_threshold": signal.get('applied_buy_threshold', 0.5),
                    "sell_threshold": signal.get('applied_sell_threshold', -0.5),
                    "tech_weight": signal.get('applied_tech_weight', 0.6),
                    "sentiment_weight": signal.get('applied_sentiment_weight', 0.4)
                },
                "risk_management": {
                    "stop_loss": signal.get('stop_loss'),
                    "take_profit": signal.get('take_profit')
                }
            }
            formatted_signals.append(formatted_signal)
        
        return {
            "username": username,
            "total_signals": len(formatted_signals),
            "signals": formatted_signals
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get signals for user {username}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve signals: {str(e)}")

@app.delete("/api/admin/users/{username}/signals")
async def admin_clear_user_signals(username: str, current_user: Dict[str, Any] = Depends(verify_token)):
    """Clear all signals for a specific user (admin only)"""
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        # Check if user exists
        user_data = await get_user_from_redis(username)
        if not user_data:
            raise HTTPException(status_code=404, detail="User not found")
        
        redis_client = await get_redis_client()
        if redis_client:
            # Clear user-specific signals
            user_key = f"signals:user:{username}"
            cleared_count = await redis_client.clear_user_signals(username)
            logger.info(f"Cleared {cleared_count} signals for user {username}")
            return {"message": f"Cleared {cleared_count} signals for user {username}", "cleared_count": cleared_count}
        else:
            raise HTTPException(status_code=500, detail="Redis not available")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to clear signals for user {username}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear signals: {str(e)}")

@app.put("/api/admin/users/{username}")
async def admin_edit_user(username: str, user_data: UserUpdate, current_user: Dict[str, Any] = Depends(verify_token)):
    """Edit user details (admin only)"""
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        # Check if user exists
        existing_user = await get_user_from_redis(username)
        if not existing_user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Check if new username already exists (if username is being changed)
        if user_data.username != username:
            existing_new_user = await get_user_from_redis(user_data.username)
            if existing_new_user:
                raise HTTPException(status_code=400, detail="Username already exists")
        
        # Hash the password only if provided
        if user_data.password and user_data.password.strip():
            password_hash = hashlib.sha256(user_data.password.encode()).hexdigest()
        else:
            # Keep existing password
            password_hash = existing_user.get("password_hash")
        
        # Handle activation period extension
        activation_expires_at = existing_user.get("activation_expires_at")
        activation_days = existing_user.get("activation_days", 30)
        
        if user_data.activation_days is not None and user_data.role != "admin":
            # Update activation period (can be extension or deactivation)
            if user_data.activation_days <= 0:
                # Deactivate user by setting activation_days to 0
                activation_expires_at = None
                activation_days = 0
            else:
                # Extend activation period
                activation_expires_at = extend_user_activation(activation_expires_at, user_data.activation_days)
                activation_days = user_data.activation_days
        
        # Update user data
        updated_user_data = {
            "username": user_data.username,
            "password_hash": password_hash,
            "role": user_data.role,
            "created_at": existing_user.get("created_at", datetime.now().isoformat()),
            "last_login": existing_user.get("last_login"),
            "status": existing_user.get("status", "active"),
            "first_name": user_data.first_name,
            "last_name": user_data.last_name,
            "email": user_data.email,
            "phone": user_data.phone,
            "additional_info": user_data.additional_info,
            "activation_expires_at": activation_expires_at,
            "activation_days": activation_days
        }
        
        # Store updated user in Redis
        success = await store_user_in_redis(user_data.username, updated_user_data)
        
        if success:
            # If username changed, remove old user entry
            if user_data.username != username:
                redis_client = await get_redis_client()
                if redis_client:
                    await redis_client.delete(f"user:{username}")
            
            # Log activity
            redis_client = await get_redis_client()
            if redis_client:
                await redis_client.log_activity(
                    "user_updated",
                    f"Updated user '{user_data.username}' (role: {user_data.role})",
                    current_user["username"],
                    {"username": user_data.username, "role": user_data.role, "old_username": username if user_data.username != username else None}
                )
            
            logger.info(f"Updated user {user_data.username}")
            return {"message": f"User {user_data.username} updated successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to update user")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update user {username}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update user: {str(e)}")

@app.delete("/api/admin/users/{username}")
async def admin_delete_user(username: str, delete_signals: bool = False, current_user: Dict[str, Any] = Depends(verify_token)):
    """Delete user (admin only)"""
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    # Prevent admin from deleting themselves
    if username == current_user["username"]:
        raise HTTPException(status_code=400, detail="Cannot delete your own account")
    
    try:
        # Check if user exists
        user_data = await get_user_from_redis(username)
        if not user_data:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Check if this is the last admin user
        if user_data.get("role") == "admin":
            all_users = await get_all_users_from_redis()
            admin_count = sum(1 for user in all_users if user.get("role") == "admin")
            
            if admin_count <= 1:
                raise HTTPException(
                    status_code=400, 
                    detail="Cannot delete the last admin user. At least one admin must remain in the system."
                )
        
        redis_client = await get_redis_client()
        if not redis_client:
            raise HTTPException(status_code=500, detail="Redis not available")
        
        # Delete user signals if requested
        signals_deleted = 0
        if delete_signals:
            signals_deleted = await redis_client.clear_user_signals(username)
            logger.info(f"Deleted {signals_deleted} signals for user {username}")
        
        # Delete user from Redis using the proper method
        await redis_client.delete_user(username)
        
        # Log activity
        await redis_client.log_activity(
            "user_deleted",
            f"Deleted user '{username}'" + (f" and {signals_deleted} trading tips" if signals_deleted > 0 else ""),
            current_user["username"],
            {"username": username, "signals_deleted": signals_deleted}
        )
        
        logger.info(f"Deleted user {username}")
        return {
            "message": f"User {username} deleted successfully",
            "signals_deleted": signals_deleted
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete user {username}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete user: {str(e)}")

@app.post("/api/admin/users/{username}/extend-activation")
async def admin_extend_user_activation(username: str, extension_data: UserActivationExtension, current_user: Dict[str, Any] = Depends(verify_token)):
    """Extend user activation period (admin only)"""
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        # Get existing user
        user_data = await get_user_from_redis(username)
        if not user_data:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Check if user is admin (admins don't have activation periods)
        if user_data.get("role") == "admin":
            raise HTTPException(status_code=400, detail="Admin users do not have activation periods")
        
        # Extend or deactivate user
        current_expiry = user_data.get("activation_expires_at")
        current_days = user_data.get("activation_days", 0)
        
        if extension_data.additional_days <= 0:
            # Deactivate user
            user_data["activation_expires_at"] = None
            user_data["activation_days"] = 0
            new_expiry = None
        else:
            # Extend activation period
            new_expiry = extend_user_activation(current_expiry, extension_data.additional_days)
            user_data["activation_expires_at"] = new_expiry
            user_data["activation_days"] = extension_data.additional_days
        
        # Store updated user
        success = await store_user_in_redis(username, user_data)
        
        if success:
            # Log activity
            redis_client = await get_redis_client()
            if redis_client:
                await redis_client.log_activity(
                    "user_activation_extended",
                    f"Extended activation period for user '{username}' by {extension_data.additional_days} days",
                    current_user["username"],
                    {
                        "username": username,
                        "additional_days": extension_data.additional_days,
                        "new_expiry": new_expiry,
                        "reason": extension_data.reason
                    }
                )
            
            if extension_data.additional_days <= 0:
                return {
                    "message": f"User {username} has been deactivated",
                    "new_expiry": None,
                    "additional_days": 0
                }
            else:
                return {
                    "message": f"User {username} activation extended by {extension_data.additional_days} days",
                    "new_expiry": new_expiry,
                    "additional_days": extension_data.additional_days
                }
        else:
            raise HTTPException(status_code=500, detail="Failed to update user activation")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to extend user activation for {username}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to extend user activation: {str(e)}")

# User Profile API endpoints
@app.get("/api/user/profile")
async def get_user_profile(current_user: Dict[str, Any] = Depends(verify_token)):
    """Get current user's profile information"""
    try:
        redis_client = await get_redis_client()
        if not redis_client:
            raise HTTPException(status_code=500, detail="Redis not available")
        
        username = current_user["username"]
        user_data = await redis_client.get_user(username)
        
        if not user_data:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Get user status with activation info
        actual_status = get_user_status_with_activation(user_data)
        
        return {
            "username": user_data.get("username"),
            "role": user_data.get("role"),
            "status": actual_status,
            "created_at": user_data.get("created_at"),
            "last_login": user_data.get("last_login"),
            "first_name": user_data.get("first_name"),
            "last_name": user_data.get("last_name"),
            "email": user_data.get("email"),
            "phone": user_data.get("phone"),
            "additional_info": user_data.get("additional_info"),
            "activation_expires_at": user_data.get("activation_expires_at"),
            "activation_days": user_data.get("activation_days")
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get user profile for {username}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get user profile: {str(e)}")

@app.put("/api/user/profile")
async def update_user_profile(profile_data: UserProfileUpdate, current_user: Dict[str, Any] = Depends(verify_token)):
    """Update current user's profile information (excluding activation period)"""
    try:
        redis_client = await get_redis_client()
        if not redis_client:
            raise HTTPException(status_code=500, detail="Redis not available")
        
        username = current_user["username"]
        user_data = await redis_client.get_user(username)
        
        if not user_data:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Update user data (excluding activation period - only admin can manage that)
        updated_data = {
            "username": username,
            "password_hash": user_data.get("password_hash") or user_data.get("password"),  # Keep existing password hash
            "role": user_data.get("role"),  # Keep existing role
            "created_at": user_data.get("created_at"),  # Keep original creation date
            "last_login": user_data.get("last_login"),  # Keep last login
            "first_name": profile_data.first_name if profile_data.first_name is not None else user_data.get("first_name"),
            "last_name": profile_data.last_name if profile_data.last_name is not None else user_data.get("last_name"),
            "email": profile_data.email if profile_data.email is not None else user_data.get("email"),
            "phone": profile_data.phone if profile_data.phone is not None else user_data.get("phone"),
            "additional_info": profile_data.additional_info if profile_data.additional_info is not None else user_data.get("additional_info"),
            "activation_expires_at": user_data.get("activation_expires_at"),  # Keep activation info
            "activation_days": user_data.get("activation_days")  # Keep activation info
        }
        
        # Update password if provided
        if profile_data.password:
            updated_data["password_hash"] = hashlib.sha256(profile_data.password.encode()).hexdigest()
        
        # Store updated user data
        success = await redis_client.store_user(username, updated_data)
        
        if success:
            # Log the activity
            await redis_client.log_activity(
                "user_profile_updated",
                f"User '{username}' updated their profile",
                username,
                {
                    "updated_fields": [k for k, v in profile_data.dict().items() if v is not None]
                }
            )
            
            return {
                "message": "Profile updated successfully",
                "username": username
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to update profile")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update user profile for {username}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update profile: {str(e)}")

@app.put("/api/user/change-password")
async def change_user_password(password_data: dict, current_user: Dict[str, Any] = Depends(verify_token)):
    """Change current user's password"""
    try:
        redis_client = await get_redis_client()
        if not redis_client:
            raise HTTPException(status_code=500, detail="Redis not available")
        
        username = current_user["username"]
        user_data = await redis_client.get_user(username)
        
        if not user_data:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Verify current password
        current_password = password_data.get("current_password")
        new_password = password_data.get("new_password")
        
        if not current_password or not new_password:
            raise HTTPException(status_code=400, detail="Current password and new password are required")
        
        current_password_hash = hashlib.sha256(current_password.encode()).hexdigest()
        if user_data.get("password_hash") != current_password_hash:
            raise HTTPException(status_code=400, detail="Current password is incorrect")
        
        if len(new_password) < 6:
            raise HTTPException(status_code=400, detail="New password must be at least 6 characters long")
        
        # Update password
        updated_data = user_data.copy()
        updated_data["password_hash"] = hashlib.sha256(new_password.encode()).hexdigest()
        
        success = await redis_client.store_user(username, updated_data)
        
        if success:
            # Log the activity
            await redis_client.log_activity(
                "user_password_changed",
                f"User '{username}' changed their password",
                username,
                {}
            )
            
            return {
                "message": "Password changed successfully",
                "username": username
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to change password")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to change password for {username}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to change password: {str(e)}")

# Settings API endpoints
@app.get("/api/admin/settings")
async def admin_get_settings(current_user: Dict[str, Any] = Depends(verify_token)):
    """Get all system settings (admin only)"""
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        return {
            "config": system_config.dict(),
            "categories": {
                "trading": ["active_markets", "signal_generation_enabled", "generation_frequency_minutes", "default_timeframe"],
                "technical": ["buy_threshold", "sell_threshold", "technical_weight", "sentiment_weight"],
                "data": ["data_provider", "data_refresh_rate_minutes", "historical_data_days"],
                "news": ["news_weight", "news_keywords"],
                "security": ["session_timeout_hours", "max_login_attempts", "password_min_length", "api_rate_limit_per_minute"],
                "ui": ["auto_refresh_enabled", "refresh_interval_seconds", "theme", "date_format", "timezone", "currency"],
                "trading_config": ["max_position_size", "default_stop_loss_percent", "default_take_profit_percent", "risk_per_trade_percent"],
                "notifications": ["email_notifications_enabled", "smtp_server", "smtp_port", "smtp_username", "smtp_password", "alert_recipients"],
                "system": ["log_level", "cache_ttl_hours", "backup_enabled", "backup_frequency_hours"]
            }
        }
    except Exception as e:
        logger.error(f"Failed to get settings: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve settings")

@app.put("/api/admin/settings/{key}")
async def admin_update_setting(key: str, config_update: ConfigUpdate, current_user: Dict[str, Any] = Depends(verify_token)):
    """Update a specific setting (admin only)"""
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        success = await update_config(key, config_update.value, config_update.category)
        if success:
            return {"message": f"Setting {key} updated successfully", "value": config_update.value}
        else:
            raise HTTPException(status_code=400, detail=f"Failed to update setting {key}")
    except Exception as e:
        logger.error(f"Failed to update setting {key}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update setting: {str(e)}")

@app.post("/api/admin/settings/bulk")
async def admin_bulk_update_settings(updates: List[ConfigUpdate], current_user: Dict[str, Any] = Depends(verify_token)):
    """Update multiple settings at once (admin only)"""
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        results = []
        for update in updates:
            success = await update_config(update.key, update.value, update.category)
            results.append({"key": update.key, "success": success})
        
        # Log activity
        redis_client = await get_redis_client()
        if redis_client:
            await redis_client.log_activity(
                "settings_updated",
                f"Bulk updated {len(updates)} settings",
                current_user["username"],
                {"updated_keys": [u.key for u in updates]}
            )
        
        return {"message": f"Updated {len(updates)} settings", "results": results}
    except Exception as e:
        logger.error(f"Failed to bulk update settings: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update settings: {str(e)}")

@app.post("/api/admin/settings/reset")
async def admin_reset_settings(current_user: Dict[str, Any] = Depends(verify_token)):
    """Reset all settings to defaults (admin only)"""
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        global system_config
        system_config = SystemConfig()
        await save_config_to_redis()
        
        # Log activity
        redis_client = await get_redis_client()
        if redis_client:
            await redis_client.log_activity(
                "settings_reset",
                "Reset all settings to defaults",
                current_user["username"],
                {}
            )
        
        return {"message": "All settings reset to defaults"}
    except Exception as e:
        logger.error(f"Failed to reset settings: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reset settings: {str(e)}")

@app.get("/api/admin/settings/export")
async def admin_export_settings(current_user: Dict[str, Any] = Depends(verify_token)):
    """Export current settings as JSON (admin only)"""
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        return {
            "config": system_config.dict(),
            "exported_at": datetime.now().isoformat(),
            "version": "1.0"
        }
    except Exception as e:
        logger.error(f"Failed to export settings: {e}")
        raise HTTPException(status_code=500, detail="Failed to export settings")

@app.post("/api/admin/settings/import")
async def admin_import_settings(config_data: dict, current_user: Dict[str, Any] = Depends(verify_token)):
    """Import settings from JSON (admin only)"""
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        global system_config
        system_config = SystemConfig(**config_data)
        await save_config_to_redis()
        
        # Log activity
        redis_client = await get_redis_client()
        if redis_client:
            await redis_client.log_activity(
                "settings_imported",
                "Imported settings from file",
                current_user["username"],
                {"imported_keys": list(config_data.keys())}
            )
        
        return {"message": "Settings imported successfully"}
    except Exception as e:
        logger.error(f"Failed to import settings: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to import settings: {str(e)}")

# Global state
agent_status = {
    "status": "running",
    "last_update": datetime.now().isoformat(),
    "active_symbols": [],
    "total_signals": 0,
    "uptime": "0:00:00"
}

# Global system configuration
system_config = SystemConfig()

# Configuration change callbacks
config_change_callbacks = []

# Configuration management functions
async def load_config_from_redis():
    """Load configuration from Redis"""
    try:
        redis_client = await get_redis_client()
        if redis_client:
            config_data = await redis_client.client.get("system:config")
            if config_data:
                config_dict = json.loads(config_data)
                global system_config
                system_config = SystemConfig(**config_dict)
                logger.info("Configuration loaded from Redis")
                return True
    except Exception as e:
        logger.error(f"Failed to load config from Redis: {e}")
    return False

async def save_config_to_redis():
    """Save configuration to Redis"""
    try:
        redis_client = await get_redis_client()
        if redis_client:
            config_dict = system_config.dict()
            await redis_client.client.set("system:config", json.dumps(config_dict))
            logger.info("Configuration saved to Redis")
            return True
    except Exception as e:
        logger.error(f"Failed to save config to Redis: {e}")
    return False

async def update_config(key: str, value: Any, category: str = "general"):
    """Update a configuration value and trigger callbacks"""
    try:
        global system_config
        if hasattr(system_config, key):
            old_value = getattr(system_config, key)
            setattr(system_config, key, value)
            
            # Save to Redis
            await save_config_to_redis()
            
            # Trigger callbacks
            for callback in config_change_callbacks:
                try:
                    await callback(key, value, old_value, category)
                except Exception as e:
                    logger.error(f"Config callback error: {e}")
            
            logger.info(f"Configuration updated: {key} = {value}")
            return True
        else:
            logger.error(f"Configuration key not found: {key}")
            return False
    except Exception as e:
        logger.error(f"Failed to update config: {e}")
        return False

def register_config_callback(callback):
    """Register a callback for configuration changes"""
    config_change_callbacks.append(callback)

# User activation period management functions
def calculate_activation_expiry(activation_days: int) -> Optional[str]:
    """Calculate activation expiry date"""
    if activation_days <= 0:
        return None  # No expiry set for zero or negative days (immediate deactivation)
    expiry_date = datetime.now() + timedelta(days=activation_days)
    return expiry_date.isoformat()

def is_user_activation_valid(activation_expires_at: Optional[str], activation_days: Optional[int] = None) -> bool:
    """Check if user activation is still valid"""
    # If activation_days is 0 or negative, user is immediately deactivated
    if activation_days is not None and activation_days <= 0:
        return False
    
    if not activation_expires_at:
        return True  # No expiry set (admin users or legacy users)
    
    try:
        expiry_date = datetime.fromisoformat(activation_expires_at)
        return datetime.now() < expiry_date
    except (ValueError, TypeError):
        return True  # Invalid date format, assume valid

def extend_user_activation(current_expiry: Optional[str], additional_days: int) -> str:
    """Extend user activation period"""
    if not current_expiry:
        # No current expiry, set from now
        return calculate_activation_expiry(additional_days)
    
    try:
        current_date = datetime.fromisoformat(current_expiry)
        # If current expiry is in the past, extend from now
        if current_date < datetime.now():
            return calculate_activation_expiry(additional_days)
        else:
            # Extend from current expiry date
            new_expiry = current_date + timedelta(days=additional_days)
            return new_expiry.isoformat()
    except (ValueError, TypeError):
        # Invalid current expiry, set from now
        return calculate_activation_expiry(additional_days)

def get_user_status_with_activation(user_data: Dict[str, Any]) -> str:
    """Get user status considering activation period"""
    if user_data.get("role") == "admin":
        return user_data.get("status", "active")  # Admins are always active
    
    if not is_user_activation_valid(user_data.get("activation_expires_at"), user_data.get("activation_days")):
        return "inactive"
    
    return user_data.get("status", "active")

# Simple in-memory cache for market data (5 minute TTL)
market_data_cache = {}
CACHE_TTL = 300  # 5 minutes in seconds


# Initialize sentiment analyzer lazily (only when needed)
sentiment_analyzer = None

def get_sentiment_analyzer():
    """Get sentiment analyzer instance, initializing it lazily if needed"""
    global sentiment_analyzer
    if sentiment_analyzer is None:
        try:
            sentiment_analyzer = SentimentAnalyzer("ProsusAI/finbert")
            logger.info("Sentiment analyzer initialized")
        except Exception as e:
            logger.warning(f"Could not initialize sentiment analyzer: {e}")
    return sentiment_analyzer



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
                status_data["total_trading_tips"] = len(user_signals)
                logger.debug(f"User {current_user['username']} sees {len(user_signals)} signals")
            else:
                # For unauthenticated requests, show global count
                status_data["total_trading_tips"] = await redis_client.get_signal_count()
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
                    # Debug logging for signal data from Redis
                    logger.info(f"Signal from Redis: {signal_dict.get('symbol', 'unknown')} - applied_buy_threshold: {signal_dict.get('applied_buy_threshold', 'NOT_FOUND')}")
                    
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

@app.delete("/api/signals/{timestamp}")
async def delete_signal(timestamp: str, current_user: Dict[str, Any] = Depends(verify_token)):
    """Delete a specific signal by timestamp (user-specific)"""
    try:
        redis_client = await get_redis_client()
        if not redis_client:
            raise HTTPException(status_code=500, detail="Redis not available")
        
        # Get all signals for the user
        user_signals = await redis_client.get_signals(1000, None, current_user['username'])
        
        # Find the signal with matching timestamp
        signal_to_delete = None
        for signal in user_signals:
            if signal.get('timestamp') == timestamp:
                signal_to_delete = signal
                break
        
        if not signal_to_delete:
            raise HTTPException(status_code=404, detail="Signal not found")
        
        # Delete the signal from Redis
        deleted = await redis_client.delete_signal(timestamp, current_user['username'])
        
        if deleted:
            logger.info(f"Signal {timestamp} deleted by user {current_user['username']}")
            return {"message": "Signal deleted successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to delete signal")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting signal: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting signal: {str(e)}")

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

@app.get("/api/admin/signals/all")
async def admin_get_all_signals(limit: int = 100, offset: int = 0, current_user: Dict[str, Any] = Depends(verify_token)):
    """Get all trading signals with detailed information (admin only)"""
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        redis_client = await get_redis_client()
        if not redis_client:
            raise HTTPException(status_code=500, detail="Redis not available")
        
        # Get all signals
        all_signals = await redis_client.get_signals(limit + offset)
        
        # Apply pagination
        paginated_signals = all_signals[offset:offset + limit]
        
        # Format signals with detailed information
        formatted_signals = []
        for signal in paginated_signals:
            formatted_signal = {
                "id": signal.get('timestamp', 'unknown'),
                "market": signal.get('symbol', 'unknown'),
                "timeframe": signal.get('timeframe', '1h'),
                "recommendation": signal.get('signal_type', 'HOLD'),
                "confidence": signal.get('confidence', 0.0),
                "technical_score": signal.get('technical_score', 0.0),
                "sentiment_score": signal.get('sentiment_score', 0.0),
                "fused_score": signal.get('fused_score', 0.0),
                "reasoning": signal.get('reasoning', 'No reasoning provided'),
                "generated_at": signal.get('timestamp', ''),
                "username": signal.get('username', 'unknown'),
                "configuration": {
                    "buy_threshold": signal.get('applied_buy_threshold', 0.5),
                    "sell_threshold": signal.get('applied_sell_threshold', -0.5),
                    "tech_weight": signal.get('applied_tech_weight', 0.6),
                    "sentiment_weight": signal.get('applied_sentiment_weight', 0.4)
                },
                "risk_management": {
                    "stop_loss": signal.get('stop_loss'),
                    "take_profit": signal.get('take_profit')
                }
            }
            formatted_signals.append(formatted_signal)
        
        return {
            "signals": formatted_signals,
            "total_count": len(all_signals),
            "returned_count": len(formatted_signals),
            "offset": offset,
            "limit": limit
        }
        
    except Exception as e:
        logger.error(f"Failed to get all signals: {e}")
        raise HTTPException(status_code=500, detail="Failed to get signals")


@app.post("/api/signals/generate")
async def generate_signal(request: SignalRequest, current_user: Dict[str, Any] = Depends(verify_token)) -> TradingSignal:
    """Generate a new trading signal for a symbol"""
    try:
        logger.info(f"Starting signal generation for {request.symbol}")
        # Fetch market data
        if "/" in request.symbol:  # Crypto
            logger.info(f"Fetching crypto data for {request.symbol}")
            ohlcv = fetch_ohlcv(request.symbol, request.timeframe)
        else:  # Stock/ETF
            logger.info(f"Fetching stock data for {request.symbol}")
            ohlcv = fetch_alpaca_ohlcv(request.symbol, request.timeframe)
        
        logger.info(f"OHLCV type: {type(ohlcv)}, value: {ohlcv if isinstance(ohlcv, str) else 'DataFrame'}")
        logger.info(f"OHLCV DataFrame shape: {ohlcv.shape}")
        logger.info(f"OHLCV DataFrame columns: {list(ohlcv.columns)}")
        logger.info(f"OHLCV DataFrame head (first 3 rows):\n{ohlcv.head(3).to_string()}")
        logger.info(f"OHLCV DataFrame tail (last 3 rows):\n{ohlcv.tail(3).to_string()}")
        
        if ohlcv is None or ohlcv.empty:
            raise HTTPException(status_code=400, detail="Could not fetch market data")
        
        # Calculate indicators
        logger.info("Starting indicators calculation")
        close = ohlcv['close']
        high = ohlcv['high']
        low = ohlcv['low']
        rsi = compute_rsi(close)
        ema_12 = compute_ema(close, 12)
        ema_26 = compute_ema(close, 26)
        macd_df = compute_macd(close)
        macd = macd_df['macd']
        macd_signal = macd_df['signal']
        macd_hist = macd_df['hist']
        atr = compute_atr(high, low, close)
        logger.info("Indicators calculation completed")
        
        # Get sentiment from multiple sources (RSS + Reddit)
        logger.info("Starting sentiment analysis from multiple sources")
        sentiment_score = 0.0
        analyzer = get_sentiment_analyzer()
        if analyzer:
            try:
                all_texts = []
                
                # Fetch RSS headlines
                logger.info("Fetching RSS headlines...")
                headlines = fetch_headlines([
                    "https://news.google.com/rss/search?q=stock+market&hl=en-US&gl=US&ceid=US:en",
                    "https://news.google.com/rss/search?q=crypto+btc&hl=en-US&gl=US&ceid=US:en",
                ], limit_per_feed=10)  # Reduced to 10 to make room for Reddit
                if headlines:
                    all_texts.extend(headlines)
                    logger.info(f"RSS: Collected {len(headlines)} headlines")
                
                # Fetch Reddit posts based on symbol type
                logger.info("Fetching Reddit posts...")
                if "BTC" in request.symbol or "ETH" in request.symbol or "crypto" in request.symbol.lower():
                    # Crypto symbol - fetch crypto Reddit posts
                    reddit_posts = fetch_crypto_reddit_posts(limit_per_subreddit=5)
                    if reddit_posts:
                        all_texts.extend(reddit_posts)
                    logger.info(f"Reddit: Collected {len(reddit_posts)} crypto posts")
                else:
                    # Stock symbol - fetch stock Reddit posts
                    reddit_posts = fetch_stock_reddit_posts(limit_per_subreddit=5)
                    if reddit_posts:
                        all_texts.extend(reddit_posts)
                        logger.info(f"Reddit: Collected {len(reddit_posts)} stock posts")
                
                if all_texts:
                    # Use optimal sample size with mixed sources
                    import random
                    sample_size = min(20, len(all_texts))  # Increased to 20 for mixed sources
                    shuffled_texts = all_texts.copy()
                    random.shuffle(shuffled_texts)
                    text_sample = shuffled_texts[:sample_size]
                    logger.info(f"Analyzing {len(text_sample)} texts (RSS + Reddit) for sentiment")
                    sentiment_score = analyzer.score(text_sample)
                    logger.info(f"Sentiment analysis result: {sentiment_score:.3f}")
                else:
                    logger.warning("No texts collected from any source")
                    
            except Exception as e:
                logger.warning(f"Could not fetch sentiment: {e}")
        logger.info("Sentiment analysis completed")
        
        # Calculate technical score
        logger.info(f"Close type: {type(close)}, value: {close if isinstance(close, str) else 'Series'}")
        logger.info(f"RSI type: {type(rsi)}, empty: {rsi.empty if hasattr(rsi, 'empty') else 'no empty attr'}")
        logger.info(f"MACD hist type: {type(macd_hist)}, empty: {macd_hist.empty if hasattr(macd_hist, 'empty') else 'no empty attr'}")
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
        
        # Debug logging for custom thresholds
        logger.info(f"Custom threshold values received: buy_threshold={request.buy_threshold}, sell_threshold={request.sell_threshold}, technical_weight={request.technical_weight}, sentiment_weight={request.sentiment_weight}")
        
        # Use custom parameters if provided, otherwise use config defaults
        config = load_config_from_env()
        tech_weight = request.technical_weight if request.technical_weight is not None else config.thresholds.technical_weight
        sentiment_weight = request.sentiment_weight if request.sentiment_weight is not None else config.thresholds.sentiment_weight
        fused_score = tech_weight * tech_score + sentiment_weight * sentiment_score
        
        # Define thresholds - use config defaults if not provided
        buy_threshold = request.buy_threshold if request.buy_threshold is not None else config.thresholds.buy_threshold
        # If custom sell_threshold is provided, use it; otherwise use negative of buy_threshold
        if request.sell_threshold is not None:
            sell_threshold = request.sell_threshold
        else:
            sell_threshold = -buy_threshold
        
        # Debug logging for applied thresholds
        logger.info(f"Applied threshold values: buy_threshold={buy_threshold}, sell_threshold={sell_threshold}, technical_weight={tech_weight}, sentiment_weight={sentiment_weight}")
        
        # Determine signal type
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
        
        # Add user information to signal
        signal_dict = signal.dict()
        signal_dict['username'] = current_user['username']
        
        # Store the generated signal in Redis
        try:
            redis_client = await get_redis_client()
            if redis_client:
                # Use the signal_dict that already has username added
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
                        "created_by": current_user["username"]
                    }
                )
                
                # Log activity
                await redis_client.log_activity(
                    "signal_generated",
                    f"Generated {signal.signal_type} trading tip for {signal.symbol} (confidence: {(signal.confidence * 100):.1f}%)",
                    current_user["username"],
                    {
                        "symbol": signal.symbol,
                        "signal_type": signal.signal_type,
                        "confidence": signal.confidence,
                        "technical_score": signal.technical_score,
                        "sentiment_score": signal.sentiment_score,
                        "fused_score": signal.fused_score
                    }
                )
                
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




@app.get("/api/config/app")
async def get_app_config() -> Dict[str, Any]:
    """Get application configuration section"""
    try:
        from agent.config import get_config
        from dataclasses import asdict
        
        # Get the loaded configuration
        loaded_config = get_config()
        
        return {
            "message": "Application configuration section",
            "source": "runtime_config",
            "section": "app",
            "config": {
                "app": asdict(loaded_config.app),
                "server": asdict(loaded_config.server),
                "database": asdict(loaded_config.database),
                "security": asdict(loaded_config.security),
                "api": asdict(loaded_config.api),
                "telegram": asdict(loaded_config.telegram),
                "monitoring": asdict(loaded_config.monitoring),
                "features": asdict(loaded_config.features),
                "development": asdict(loaded_config.development)
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error loading app config: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load app config: {str(e)}")

# Redis-based configuration manager
from agent.redis_config_manager import get_redis_config_manager

def get_cached_config():
    """Get cached configuration from Redis"""
    # This will be replaced with async Redis calls in the endpoints
    from agent.config import get_config
    return get_config()  # Fallback to file-based config

@app.get("/api/config/symbols")
async def get_symbols_config() -> Dict[str, Any]:
    """Get symbols configuration (fast endpoint for UI dropdown)"""
    try:
        redis_config_manager = get_redis_config_manager()
        universe_config = await redis_config_manager.get_universe_config()
        
        if universe_config:
            return {
                "crypto_symbols": universe_config.get("crypto_symbols", []),
                "stock_symbols": universe_config.get("stock_symbols", []),
                "timeframes": universe_config.get("timeframes", []),
                "default_timeframe": universe_config.get("timeframe", "5m")
            }
        else:
            # Fallback to file-based config if Redis is not available
            logger.warning("Redis config not available, falling back to file-based config")
            cached_config = get_cached_config()
            return {
                "crypto_symbols": cached_config.universe.crypto_symbols,
                "stock_symbols": cached_config.universe.stock_symbols,
                "timeframes": cached_config.universe.timeframes,
                "default_timeframe": cached_config.universe.timeframe
            }
    except Exception as e:
        logger.error(f"Error loading symbols config: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load symbols config: {str(e)}")

@app.get("/api/config/trading")
async def get_trading_config() -> Dict[str, Any]:
    """Get trading configuration section"""
    try:
        from dataclasses import asdict
        
        # Get the cached configuration
        loaded_config = get_cached_config()
        
        return {
            "message": "Trading configuration section",
            "source": "runtime_config",
            "section": "trading",
            "config": {
                "universe": asdict(loaded_config.universe),
                "signals": asdict(loaded_config.signals),
                "guardrails": asdict(loaded_config.guardrails),
                "safety": asdict(loaded_config.safety),
                "technical_analysis": asdict(loaded_config.technical_analysis),
                "sentiment_analysis": asdict(loaded_config.sentiment_analysis),
                "exchanges": asdict(loaded_config.exchanges),
                "risk_management": asdict(loaded_config.risk_management),
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error loading trading config: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load trading config: {str(e)}")

@app.get("/api/config/logging")
async def get_logging_config() -> Dict[str, Any]:
    """Get logging configuration section"""
    try:
        from agent.config import get_config
        from dataclasses import asdict
        
        # Get the loaded configuration
        loaded_config = get_config()
        
        return {
            "message": "Logging configuration section",
            "source": "runtime_config",
            "section": "logging",
            "config": {
                "logging": asdict(loaded_config.logging)
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error loading logging config: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load logging config: {str(e)}")

@app.get("/api/telegram/connection")
async def get_telegram_connection(current_user: Dict[str, Any] = Depends(verify_token)):
    """Get Telegram connection status for the current user"""
    try:
        redis_client = await get_redis_client()
        if not redis_client:
            raise HTTPException(status_code=500, detail="Redis not available")
        
        # Get user's Telegram connection from Redis
        telegram_connection = await redis_client.get_telegram_connection(current_user['username'])
        
        # Check if Telegram is enabled in config
        from agent.config import get_config
        config = get_config()
        telegram_enabled = config.telegram.enabled and bool(config.telegram.bot_token)
        
        return {
            "connected": bool(telegram_connection),
            "chat_id": telegram_connection.get('chat_id') if telegram_connection else None,
            "telegram_enabled": telegram_enabled,
            "bot_token_configured": bool(config.telegram.bot_token)
        }
    except Exception as e:
        logger.error(f"Error getting Telegram connection status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get Telegram connection status: {str(e)}")

@app.post("/api/telegram/connect")
async def connect_telegram(
    chat_id: str,
    current_user: Dict[str, Any] = Depends(verify_token)
):
    """Connect user's Telegram account"""
    try:
        redis_client = await get_redis_client()
        if not redis_client:
            raise HTTPException(status_code=500, detail="Redis not available")
        
        # Validate chat ID format (should be numeric)
        if not chat_id.isdigit():
            raise HTTPException(status_code=400, detail="Invalid chat ID format")
        
        # Store chat ID in Redis using the proper method
        await redis_client.store_telegram_connection(current_user['username'], chat_id)
        
        # Log the connection
        await redis_client.log_activity(
            "telegram_connected",
            f"User {current_user['username']} connected Telegram chat ID {chat_id}",
            current_user['username'],
            {"chat_id": chat_id}
        )
        
        return {"status": "success", "message": "Telegram connected successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error connecting Telegram: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to connect Telegram: {str(e)}")

@app.post("/api/telegram/disconnect")
async def disconnect_telegram(current_user: Dict[str, Any] = Depends(verify_token)):
    """Disconnect user's Telegram account"""
    try:
        redis_client = await get_redis_client()
        if not redis_client:
            raise HTTPException(status_code=500, detail="Redis not available")
        
        # Remove Telegram connection from Redis
        await redis_client.remove_telegram_connection(current_user['username'])
        
        # Log the disconnection
        await redis_client.log_activity(
            "telegram_disconnected",
            f"User {current_user['username']} disconnected Telegram",
            current_user['username']
        )
        
        return {"status": "success", "message": "Telegram disconnected successfully"}
    except Exception as e:
        logger.error(f"Error disconnecting Telegram: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to disconnect Telegram: {str(e)}")

@app.post("/api/telegram/test")
async def test_telegram_connection(current_user: Dict[str, Any] = Depends(verify_token)):
    """Send a test message to user's Telegram"""
    try:
        redis_client = await get_redis_client()
        if not redis_client:
            raise HTTPException(status_code=500, detail="Redis not available")
        
        # Get user's Telegram connection
        telegram_connection = await redis_client.get_telegram_connection(current_user['username'])
        if not telegram_connection:
            raise HTTPException(status_code=400, detail="No Telegram account connected")
        
        chat_id = telegram_connection.get('chat_id')
        if not chat_id:
            raise HTTPException(status_code=400, detail="No chat ID found in connection")
        
        # Get Telegram configuration
        from agent.config import get_config
        config = get_config()
        
        if not config.telegram.bot_token:
            raise HTTPException(status_code=500, detail="Telegram bot token not configured")
        
        # Send test message
        import requests
        import json
        from datetime import datetime
        
        message = f"🧪 Test message from Trading AI Tips System\n\n" \
                 f"✅ Connection successful!\n" \
                 f"👤 User: {current_user['username']}\n" \
                 f"🕐 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n" \
                 f"📱 Chat ID: {chat_id}\n\n" \
                 f"You will now receive trading notifications here!"
        
        url = f"{config.telegram.api_url}{config.telegram.bot_token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "HTML"
        }
        
        response = requests.post(url, json=payload, timeout=config.telegram.timeout)
        
        if response.status_code == 200:
            result = response.json()
            if result.get('ok'):
                # Log the test message
                await redis_client.log_activity(
                    "telegram_test_sent",
                    f"Test message sent to Telegram for user {current_user['username']}",
                    current_user['username'],
                    {"chat_id": chat_id, "message_id": result.get('result', {}).get('message_id')}
                )
                
                return {
                    "status": "success", 
                    "message": "Test message sent successfully",
                    "chat_id": chat_id,
                    "message_id": result.get('result', {}).get('message_id')
                }
            else:
                raise HTTPException(status_code=500, detail=f"Telegram API error: {result.get('description', 'Unknown error')}")
        else:
            raise HTTPException(status_code=500, detail=f"Failed to send message: HTTP {response.status_code}")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error sending test message: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to send test message: {str(e)}")

# Message endpoints
@app.get("/api/messages")
async def get_user_messages(current_user: Dict[str, Any] = Depends(verify_token)):
    """Get user's messages (received and sent)"""
    try:
        redis_client = await get_redis_client()
        if not redis_client:
            raise HTTPException(status_code=500, detail="Redis not available")
        
        username = current_user['username']
        
        # Get received messages
        received_messages = await redis_client.get_user_messages(username, "received", 50)
        
        # Get sent messages
        sent_messages = await redis_client.get_user_messages(username, "sent", 50)
        
        # Count unread messages
        unread_count = len([msg for msg in received_messages if msg.get("status") == "unread"])
        
        return {
            "received_messages": received_messages,
            "sent_messages": sent_messages,
            "unread_count": unread_count,
            "total_received": len(received_messages),
            "total_sent": len(sent_messages)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting user messages: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get messages: {str(e)}")

@app.post("/api/messages/{message_id}/read")
async def mark_message_as_read(message_id: str, current_user: Dict[str, Any] = Depends(verify_token)):
    """Mark a message as read"""
    try:
        redis_client = await get_redis_client()
        if not redis_client:
            raise HTTPException(status_code=500, detail="Redis not available")
        
        username = current_user['username']
        success = await redis_client.mark_message_as_read(message_id, username)
        
        if success:
            return {"status": "success", "message": "Message marked as read"}
        else:
            raise HTTPException(status_code=404, detail="Message not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error marking message as read: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to mark message as read: {str(e)}")

@app.post("/api/messages/delete")
async def delete_messages(request: Dict[str, Any], current_user: Dict[str, Any] = Depends(verify_token)):
    """Delete messages"""
    try:
        redis_client = await get_redis_client()
        if not redis_client:
            raise HTTPException(status_code=500, detail="Redis not available")
        
        message_ids = request.get("message_ids", [])
        message_type = request.get("type", "received")  # received or sent
        
        if not message_ids:
            raise HTTPException(status_code=400, detail="No message IDs provided")
        
        username = current_user['username']
        deleted_count = await redis_client.delete_messages(message_ids, username)
        
        return {
            "status": "success", 
            "message": f"Deleted {deleted_count} messages",
            "deleted_count": deleted_count
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting messages: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete messages: {str(e)}")

@app.post("/api/messages/send")
async def send_message_to_admin(request: Dict[str, Any], current_user: Dict[str, Any] = Depends(verify_token)):
    """Send a message to admin"""
    try:
        redis_client = await get_redis_client()
        if not redis_client:
            raise HTTPException(status_code=500, detail="Redis not available")
        
        message = request.get("message", "").strip()
        subject = request.get("subject", "").strip()
        
        if not message:
            raise HTTPException(status_code=400, detail="Message content is required")
        
        username = current_user['username']
        success = await redis_client.send_message_to_admin(username, message, subject)
        
        if success:
            # Log the activity
            await redis_client.log_activity(
                "message_sent_to_admin",
                f"User {username} sent a message to admin",
                username,
                {"subject": subject, "message_length": len(message)}
            )
            
            return {"status": "success", "message": "Message sent to admin successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to send message")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error sending message to admin: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to send message: {str(e)}")

@app.post("/api/contact-admin")
async def contact_admin(request: Dict[str, Any], current_user: Dict[str, Any] = Depends(verify_token)):
    """Contact admin endpoint (alias for /api/messages/send)"""
    try:
        redis_client = await get_redis_client()
        if not redis_client:
            raise HTTPException(status_code=500, detail="Redis not available")
        
        message = request.get("message", "").strip()
        subject = request.get("subject", "").strip()
        
        if not message:
            raise HTTPException(status_code=400, detail="Message content is required")
        
        username = current_user['username']
        success = await redis_client.send_message_to_admin(username, message, subject)
        
        if success:
            # Log the activity
            await redis_client.log_activity(
                "message_sent_to_admin",
                f"User {username} sent a message to admin",
                username,
                {"subject": subject, "message_length": len(message)}
            )
            
            return {"status": "success", "message": "Message sent to admin successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to send message")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error sending message to admin: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to send message: {str(e)}")

# Admin message endpoints
@app.get("/api/admin/messages")
async def get_admin_messages(current_user: Dict[str, Any] = Depends(verify_token)):
    """Get admin messages (admin only)"""
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        redis_client = await get_redis_client()
        if not redis_client:
            raise HTTPException(status_code=500, detail="Redis not available")
        
        # Get messages sent to admin
        admin_messages = await redis_client.get_admin_messages(100)
        
        # Get messages sent by admin
        admin_sent_messages = await redis_client.get_admin_sent_messages(100)
        
        # Count unread messages
        unread_count = len([msg for msg in admin_messages if msg.get("status") == "unread"])
        
        logger.info(f"Admin messages API: {len(admin_messages)} received, {len(admin_sent_messages)} sent, {unread_count} unread")
        
        return {
            "received_messages": admin_messages,
            "sent_messages": admin_sent_messages,
            "unread_count": unread_count,
            "total_count": len(admin_messages),
            "total_sent": len(admin_sent_messages)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting admin messages: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get admin messages: {str(e)}")

@app.post("/api/admin/messages/{message_id}/read")
async def mark_admin_message_as_read(message_id: str, current_user: Dict[str, Any] = Depends(verify_token)):
    """Mark an admin message as read (admin only)"""
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        redis_client = await get_redis_client()
        if not redis_client:
            raise HTTPException(status_code=500, detail="Redis not available")
        
        success = await redis_client.mark_admin_message_as_read(message_id)
        
        if success:
            return {"status": "success", "message": "Message marked as read"}
        else:
            raise HTTPException(status_code=404, detail="Message not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error marking admin message as read: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to mark message as read: {str(e)}")

@app.get("/api/admin/messages/sent")
async def get_admin_sent_messages(current_user: Dict[str, Any] = Depends(verify_token)):
    """Get admin sent messages (admin only)"""
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        redis_client = await get_redis_client()
        if not redis_client:
            raise HTTPException(status_code=500, detail="Redis not available")
        
        # Get admin sent messages
        sent_messages = await redis_client.get_admin_sent_messages()
        
        return {
            "sent_messages": sent_messages,
            "total_sent": len(sent_messages)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting admin sent messages: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get admin sent messages: {str(e)}")

@app.get("/api/admin/users/unread-messages")
async def get_unread_message_counts(current_user: Dict[str, Any] = Depends(verify_token)):
    """Get unread message counts for all users (admin only)"""
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        redis_client = await get_redis_client()
        if not redis_client:
            raise HTTPException(status_code=500, detail="Redis not available")
        
        # Get admin messages to calculate unread counts per user
        admin_messages = await redis_client.get_admin_messages()
        unread_counts = {}
        
        # Count unread messages per user
        for message in admin_messages:
            if message.get("status") == "unread":
                from_username = message.get("from_username")
                if from_username and from_username != "admin":
                    unread_counts[from_username] = unread_counts.get(from_username, 0) + 1
        
        return unread_counts
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting unread message counts: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get unread message counts: {str(e)}")

@app.post("/api/admin/messages/bulk-delete")
async def bulk_delete_admin_messages(request: Dict[str, Any], current_user: Dict[str, Any] = Depends(verify_token)):
    """Bulk delete admin messages (admin only)"""
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        redis_client = await get_redis_client()
        if not redis_client:
            raise HTTPException(status_code=500, detail="Redis not available")
        
        message_ids = request.get("message_ids", [])
        message_type = request.get("message_type", "received")  # "received" or "sent"
        
        if not message_ids:
            raise HTTPException(status_code=400, detail="No message IDs provided")
        
        deleted_count = 0
        
        if message_type == "received":
            # Delete from admin messages
            deleted_count = await redis_client.bulk_delete_admin_messages(message_ids)
        elif message_type == "sent":
            # Delete from admin sent messages
            deleted_count = await redis_client.bulk_delete_admin_sent_messages(message_ids)
        else:
            raise HTTPException(status_code=400, detail="Invalid message type")
        
        return {
            "status": "success",
            "message": f"Successfully deleted {deleted_count} messages",
            "deleted_count": deleted_count
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error bulk deleting admin messages: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete messages: {str(e)}")

@app.post("/api/admin/messages/send")
async def send_message_to_user(request: Dict[str, Any], current_user: Dict[str, Any] = Depends(verify_token)):
    """Send a message from admin to user (admin only)"""
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        redis_client = await get_redis_client()
        if not redis_client:
            raise HTTPException(status_code=500, detail="Redis not available")
        
        to_username = request.get("to_username", "").strip()
        message = request.get("message", "").strip()
        subject = request.get("subject", "").strip()
        
        if not to_username or not message:
            raise HTTPException(status_code=400, detail="Username and message content are required")
        
        # Check if user exists
        user_data = await redis_client.get_user(to_username)
        if not user_data:
            raise HTTPException(status_code=404, detail="User not found")
        
        from_username = current_user['username']
        success = await redis_client.send_message_to_user(from_username, to_username, message, subject)
        
        if success:
            # Log the activity
            await redis_client.log_activity(
                "admin_message_sent",
                f"Admin {from_username} sent a message to user {to_username}",
                from_username,
                {"to_username": to_username, "subject": subject, "message_length": len(message)}
            )
            
            return {"status": "success", "message": f"Message sent to {to_username} successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to send message")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error sending message to user: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to send message: {str(e)}")

@app.get("/api/admin/users/messaging")
async def get_users_for_messaging(current_user: Dict[str, Any] = Depends(verify_token)):
    """Get list of users for messaging (admin only)"""
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        redis_client = await get_redis_client()
        if not redis_client:
            raise HTTPException(status_code=500, detail="Redis not available")
        
        # Get all users
        users = await redis_client.get_all_users()
        
        # Format users for messaging dropdown
        messaging_users = []
        for user in users:
            if user.get("username") != "admin":  # Don't include admin in the list
                messaging_users.append({
                    "username": user.get("username", ""),
                    "first_name": user.get("first_name", ""),
                    "last_name": user.get("last_name", ""),
                    "email": user.get("email", ""),
                    "role": user.get("role", "user")
                })
        
        return {
            "users": messaging_users,
            "total": len(messaging_users)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting users for messaging: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get users: {str(e)}")

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
async def get_market_overview(timeframe: str = "1h"):
    """Get market overview for all configured symbols"""
    try:
        # Get configured symbols
        config = load_config_from_env()
        symbols = config.universe.tickers
        
        # Fetch data for all symbols
        symbols_data = []
        for symbol in symbols:
            try:
                if "/" in symbol:  # Crypto
                    ohlcv = fetch_ohlcv(symbol, timeframe)
                else:  # Stock/ETF
                    ohlcv = fetch_alpaca_ohlcv(symbol, timeframe)
                
                if ohlcv is not None and not ohlcv.empty:
                    latest = ohlcv.iloc[-1]
                    prev = ohlcv.iloc[-2] if len(ohlcv) > 1 else latest
                    
                    change = latest['close'] - prev['close']
                    change_percent = (change / prev['close']) * 100 if prev['close'] != 0 else 0
                    
                    # Handle NaN and infinite values
                    def safe_float(value):
                        import math
                        if math.isnan(value) or math.isinf(value):
                            return 0.0
                        return float(value)
                    
                    # Get high/low for the period
                    high = safe_float(ohlcv['high'].max())
                    low = safe_float(ohlcv['low'].min())
                    
                    symbols_data.append({
                        "symbol": symbol,
                        "current_price": safe_float(latest['close']),
                        "price": safe_float(latest['close']),  # Keep both for compatibility
                        "change": safe_float(change),
                        "change_percent": safe_float(change_percent),
                        "volume": safe_float(latest['volume']),
                        "high": high,
                        "low": low,
                        "timestamp": latest.name.isoformat() if hasattr(latest.name, 'isoformat') else str(latest.name)
                    })
                else:
                    symbols_data.append({
                        "symbol": symbol,
                        "error": "No data available"
                    })
            except Exception as e:
                symbols_data.append({
                    "symbol": symbol,
                    "error": str(e)
                })
        
        return {
            "symbols": symbols_data,
            "timeframe": timeframe,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error fetching market overview: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """Health check endpoint with system information"""
    try:
        # Get system information
        import psutil
        import platform
        
        # System uptime
        boot_time = datetime.fromtimestamp(psutil.boot_time())
        uptime = datetime.now() - boot_time
        uptime_str = str(uptime).split('.')[0]
        
        # Application uptime
        app_uptime = datetime.now() - APP_START_TIME
        app_uptime_str = str(app_uptime).split('.')[0]
        
        # CPU and memory
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Redis status
        redis_client = await get_redis_client()
        redis_status = "Connected" if redis_client and await redis_client.is_connected() else "Disconnected"
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": APP_VERSION,
            "system_info": {
                "system_uptime": uptime_str,
                "app_uptime": app_uptime_str,
                "cpu_usage": round(cpu_percent, 1),
                "memory_usage": round(memory.percent, 1),
                "memory_available": f"{memory.available // (1024**3)} GB",
                "disk_usage": round(disk.percent, 1),
                "disk_free": f"{disk.free // (1024**3)} GB",
                "platform": platform.system(),
            "python_version": platform.python_version(),
            "redis_status": redis_status
        }
        }
    except Exception as e:
        logger.warning(f"Could not get system information for health check: {e}")
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": APP_VERSION,
            "system_info": {
                "system_uptime": "Unknown",
                "app_uptime": "Unknown",
                "cpu_usage": 0,
                "memory_usage": 0,
                "memory_available": "Unknown",
                "disk_usage": 0,
                "disk_free": "Unknown",
                "platform": "Unknown",
            "python_version": "Unknown",
            "redis_status": "Unknown"
            }
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
            
            # Start background configuration loader
            from agent.background_config_loader import start_config_loader, load_config_once
            await load_config_once()  # Load configuration immediately
            await start_config_loader(interval_minutes=1)  # Then reload every minute
            logger.info("Background configuration loader started")
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
            
    # Load system configuration from Redis
    try:
        await load_config_from_redis()
        logger.info("System configuration loaded")
    except Exception as e:
        logger.warning(f"Could not load system configuration: {e}")
    
    # Initialize default admin
    try:
        await initialize_default_admin()
        logger.info("Default admin initialized")
    except Exception as e:
        logger.warning(f"Could not initialize default admin: {e}")
    
    # Start the signal monitoring scheduler
    try:
        await start_signal_monitoring()
        logger.info("Signal monitoring started")
    except Exception as e:
        logger.warning(f"Could not start signal monitoring: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Trading Agent Web Interface")
    
    # Stop background configuration loader
    try:
        from agent.background_config_loader import stop_config_loader
        await stop_config_loader()
        logger.info("Background configuration loader stopped")
    except Exception as e:
        logger.warning(f"Error stopping configuration loader: {e}")
    
    # Close Redis connection
    try:
        await close_redis_client()
        logger.info("Redis connection closed")
    except Exception as e:
        logger.warning(f"Error closing Redis connection: {e}")

# Signal Monitoring Endpoints
@app.post("/api/signals/monitor")
async def monitor_signals(current_user: Dict[str, Any] = Depends(verify_token)):
    """Manually trigger signal monitoring (admin only)"""
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        # Run monitoring cycle
        await run_monitoring_cycle()
        
        return {
            "status": "success",
            "message": "Signal monitoring completed",
            "timestamp": datetime.now(ISRAEL_TZ).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to monitor signals: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to monitor signals: {str(e)}")

@app.get("/api/signals/monitor/status")
async def get_monitoring_status(current_user: Dict[str, Any] = Depends(verify_token)):
    """Get monitoring system status (admin only)"""
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        redis_client = await get_redis_client()
        if not redis_client:
            raise HTTPException(status_code=500, detail="Redis not available")
        
        # Get recent monitoring activities
        activities = await redis_client.get_recent_activities(limit=10)
        monitoring_activities = [a for a in activities if a.get('event_type') == 'signal_status_changed']
        
        # Get total signals count
        all_signals = await redis_client.get_signals(limit=1000)
        
        return {
            "monitoring_enabled": True,
            "schedule": "Every 2 minutes",
            "total_signals": len(all_signals),
            "recent_changes": len(monitoring_activities),
            "last_activity": monitoring_activities[0] if monitoring_activities else None,
            "timestamp": datetime.now(ISRAEL_TZ).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get monitoring status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get monitoring status: {str(e)}")

@app.get("/api/signals/{timestamp}/history")
async def get_signal_history(timestamp: str, current_user: Dict[str, Any] = Depends(verify_token)):
    """Get history events for a specific signal (users only, not admins)"""
    try:
        # Only allow regular users to access history (not admins)
        if current_user["role"] == "admin":
            raise HTTPException(status_code=403, detail="History access restricted to users only")
        
        redis_client = await get_redis_client()
        if not redis_client:
            raise HTTPException(status_code=500, detail="Redis not available")
        
        # Get the signal to verify it exists and user has access
        signal = await redis_client.get_signal_by_timestamp(timestamp)
        if not signal:
            raise HTTPException(status_code=404, detail="Signal not found")
        
        # Check if user has access to this signal (must be signal owner)
        if signal.get("username") != current_user["username"]:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Get signal history
        history = await redis_client.get_signal_history(timestamp)
        
        return {
            "signal_timestamp": timestamp,
            "symbol": signal.get("symbol"),
            "current_status": signal.get("signal_type"),
            "history": history,
            "total_events": len(history)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get signal history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get signal history: {str(e)}")

# Position Tracking Endpoints
class PositionRequest(BaseModel):
    signal_timestamp: str
    action: str  # "BUY", "SELL", "CLOSE"
    quantity: float
    price: float
    notes: str = ""

class PositionUpdateRequest(BaseModel):
    current_price: float
    action: Optional[str] = None
    quantity: Optional[float] = None
    notes: str = ""

@app.post("/api/positions")
async def create_position(request: PositionRequest, current_user: Dict[str, Any] = Depends(verify_token)):
    """Create a new trading position based on a signal"""
    try:
        redis_client = await get_redis_client()
        if not redis_client:
            raise HTTPException(status_code=500, detail="Redis not available")
        
        position_tracker = PositionTracker(redis_client)
        position = await position_tracker.create_position(
            username=current_user["username"],
            signal_timestamp=request.signal_timestamp,
            action=request.action,
            quantity=request.quantity,
            price=request.price,
            notes=request.notes
        )
        
        return position
        
    except Exception as e:
        logger.error(f"Failed to create position: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create position: {str(e)}")

@app.get("/api/positions")
async def get_user_positions(status: Optional[str] = None, current_user: Dict[str, Any] = Depends(verify_token)):
    """Get all positions for the current user"""
    try:
        redis_client = await get_redis_client()
        if not redis_client:
            raise HTTPException(status_code=500, detail="Redis not available")
        
        position_tracker = PositionTracker(redis_client)
        positions = await position_tracker.get_user_positions(current_user["username"], status)
        
        return {"positions": positions, "count": len(positions)}
        
    except Exception as e:
        logger.error(f"Failed to get positions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get positions: {str(e)}")

@app.put("/api/positions/{position_id}")
async def update_position(position_id: str, request: PositionUpdateRequest, current_user: Dict[str, Any] = Depends(verify_token)):
    """Update an existing position"""
    try:
        redis_client = await get_redis_client()
        if not redis_client:
            raise HTTPException(status_code=500, detail="Redis not available")
        
        position_tracker = PositionTracker(redis_client)
        position = await position_tracker.update_position(
            position_id=position_id,
            current_price=request.current_price,
            action=request.action,
            quantity=request.quantity,
            notes=request.notes
        )
        
        return position
        
    except Exception as e:
        logger.error(f"Failed to update position: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update position: {str(e)}")

@app.get("/api/positions/performance")
async def get_position_performance(current_user: Dict[str, Any] = Depends(verify_token)):
    """Get performance summary for the current user"""
    try:
        redis_client = await get_redis_client()
        if not redis_client:
            raise HTTPException(status_code=500, detail="Redis not available")
        
        position_tracker = PositionTracker(redis_client)
        performance = await position_tracker.get_position_performance(current_user["username"])
        
        return performance
        
    except Exception as e:
        logger.error(f"Failed to get position performance: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get position performance: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "web.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
