#!/usr/bin/env python3
"""
Redis Client for Trading Agent
Handles caching, real-time data storage, and performance optimization
"""

import json
import pickle
from typing import Any, Optional, Dict, List, Union
import asyncio
import logging
from datetime import datetime, timedelta
import pytz

# Israel timezone
ISRAEL_TZ = pytz.timezone('Asia/Jerusalem')

def get_israel_time():
    """Get current time in Israel timezone"""
    return datetime.now(ISRAEL_TZ)

try:
    import redis.asyncio as redis
    from redis.asyncio import Redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    Redis = None

from ..logging_config import get_logger

logger = get_logger(__name__)

class TradingAgentRedis:
    """Redis client for trading agent caching and data storage"""
    
    def __init__(self, 
                 host: str = "localhost", 
                 port: int = 6379, 
                 db: int = 0,
                 password: Optional[str] = None,
                 max_connections: int = 20):
        """Initialize Redis client"""
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.max_connections = max_connections
        self.client: Optional[Redis] = None
        self.connection_pool = None
        
        if not REDIS_AVAILABLE:
            logger.warning("Redis not available. Caching will be disabled.")
            return
            
        self._init_connection_pool()
    
    def _init_connection_pool(self):
        """Initialize Redis connection pool"""
        try:
            self.connection_pool = redis.ConnectionPool(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                max_connections=self.max_connections,
                decode_responses=False  # Keep binary for pickle
            )
            logger.info(f"Redis connection pool initialized: {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to initialize Redis connection pool: {e}")
            self.connection_pool = None
    
    async def connect(self) -> bool:
        """Connect to Redis"""
        if not REDIS_AVAILABLE or not self.connection_pool:
            return False
            
        try:
            self.client = redis.Redis(connection_pool=self.connection_pool)
            await self.client.ping()
            logger.info("Successfully connected to Redis")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.client = None
            return False
    
    async def disconnect(self):
        """Disconnect from Redis"""
        if self.client:
            try:
                await self.client.close()
                logger.info("Disconnected from Redis")
            except Exception as e:
                logger.error(f"Error disconnecting from Redis: {e}")
            finally:
                self.client = None
    
    async def is_connected(self) -> bool:
        """Check if connected to Redis and attempt reconnection if needed"""
        if not self.client:
            # Try to reconnect if client is None
            return await self.connect()
        try:
            await self.client.ping()
            return True
        except Exception as e:
            logger.warning(f"Redis connection lost, attempting to reconnect: {e}")
            # Close the existing client before reconnecting to avoid event loop conflicts
            try:
                await self.client.close()
            except:
                pass
            self.client = None
            # Try to reconnect
            return await self.connect()
    
    # Market Data Caching
    async def cache_market_data(self, 
                               symbol: str, 
                               timeframe: str, 
                               data: Any, 
                               ttl: int = 300) -> bool:
        """Cache market data with TTL"""
        if not await self.is_connected():
            return False
            
        try:
            key = f"market_data:{symbol}:{timeframe}"
            serialized_data = pickle.dumps(data)
            await self.client.setex(key, ttl, serialized_data)
            logger.debug(f"Cached market data: {key} (TTL: {ttl}s)")
            return True
        except Exception as e:
            logger.error(f"Failed to cache market data: {e}")
            return False
    
    async def get_cached_market_data(self, symbol: str, timeframe: str) -> Optional[Any]:
        """Retrieve cached market data"""
        if not await self.is_connected():
            return None
            
        try:
            key = f"market_data:{symbol}:{timeframe}"
            data = await self.client.get(key)
            if data:
                deserialized = pickle.loads(data)
                logger.debug(f"Retrieved cached market data: {key}")
                return deserialized
            return None
        except Exception as e:
            logger.error(f"Failed to retrieve cached market data: {e}")
            return None
    
    # Signal Storage
    
    async def get_recent_signals(self, symbol: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent signals for a symbol"""
        if not await self.is_connected():
            return []
            
        try:
            signals_key = f"signals:{symbol}"
            signal_ids = await self.client.lrange(signals_key, 0, limit - 1)
            
            signals = []
            for signal_id in signal_ids:
                try:
                    signal_data = await self.client.get(signal_id.decode())
                    if signal_data:
                        # Try JSON first (new format), fallback to pickle (old format)
                        if isinstance(signal_data, bytes):
                            try:
                                signal = json.loads(signal_data.decode('utf-8'))
                            except (json.JSONDecodeError, UnicodeDecodeError):
                                signal = pickle.loads(signal_data)
                        else:
                            signal = json.loads(signal_data)
                        signals.append(signal)
                except Exception as e:
                    logger.warning(f"Failed to retrieve signal {signal_id}: {e}")
                    continue
            
            logger.debug(f"Retrieved {len(signals)} signals for {symbol}")
            return signals
        except Exception as e:
            logger.error(f"Failed to retrieve recent signals: {e}")
            return []

    async def get_signals(self, limit: int = 100, symbol: Optional[str] = None, username: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get trading signals from Redis (optimized with pipeline)"""
        if not await self.is_connected():
            return []
            
        try:
            if username:
                # Get signals for specific user
                user_key = f"signals:user:{username}"
                signal_ids = await self.client.lrange(user_key, 0, limit - 1)
            elif symbol:
                # Get signals for specific symbol
                symbol_key = f"signals:symbol:{symbol}"
                signal_ids = await self.client.lrange(symbol_key, 0, limit - 1)
            else:
                # Get all signals
                signal_ids = await self.client.lrange("signals:list", 0, limit - 1)
            
            if not signal_ids:
                return []
            
            # Use pipeline for batch operations (much faster)
            pipe = self.client.pipeline()
            for signal_id in signal_ids:
                pipe.get(signal_id.decode())
            
            # Execute all gets in one batch
            signal_data_list = await pipe.execute()
            
            signals = []
            for signal_data in signal_data_list:
                if signal_data:
                    try:
                        # Try JSON first (new format), fallback to pickle (old format)
                        if isinstance(signal_data, bytes):
                            try:
                                signal = json.loads(signal_data.decode('utf-8'))
                            except (json.JSONDecodeError, UnicodeDecodeError):
                                # Fallback to pickle for backward compatibility
                                signal = pickle.loads(signal_data)
                        else:
                            signal = json.loads(signal_data)
                        signals.append(signal)
                    except Exception as e:
                        logger.warning(f"Failed to deserialize signal: {e}")
                        continue
            
            # Sort by timestamp (newest first)
            signals.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            return signals
        except Exception as e:
            logger.error(f"Failed to get signals: {e}")
            return []

    async def get_signal_count(self) -> int:
        """Get total number of signals in Redis"""
        if not await self.is_connected():
            logger.warning("Redis not connected for signal count")
            return 0
            
        try:
            # Count all signals by scanning all user indices
            all_users = await self.client.smembers("users:list")
            logger.info(f"Found users for signal count: {all_users}")
            total_count = 0
            
            for username in all_users:
                try:
                    user_key = f"signals:user:{username.decode()}"
                    user_signal_count = await self.client.llen(user_key)
                    logger.info(f"User {username.decode()} has {user_signal_count} signals")
                    total_count += user_signal_count
                except Exception as e:
                    logger.warning(f"Failed to count signals for user {username}: {e}")
                    continue
            
            logger.info(f"Total signal count: {total_count}")
            return total_count
        except Exception as e:
            logger.error(f"Failed to get signal count: {e}")
            return 0

    async def clear_user_signals(self, username: str) -> int:
        """Clear all signals for a specific user"""
        if not await self.is_connected():
            return 0
            
        try:
            user_key = f"signals:user:{username}"
            
            # Get all signal IDs for this user
            signal_ids = await self.client.lrange(user_key, 0, -1)
            cleared_count = len(signal_ids)
            
            if cleared_count > 0:
                # Remove signals from user index
                await self.client.delete(user_key)
                
                # Remove individual signal data and clean up all indices
                for signal_id in signal_ids:
                    try:
                        signal_id_str = signal_id.decode()
                        
                        # Get the signal data to find symbol
                        signal_data = await self.client.get(signal_id_str)
                        if signal_data:
                            # Try JSON first (new format), fallback to pickle (old format)
                            if isinstance(signal_data, bytes):
                                try:
                                    signal = json.loads(signal_data.decode('utf-8'))
                                except (json.JSONDecodeError, UnicodeDecodeError):
                                    signal = pickle.loads(signal_data)
                            else:
                                signal = json.loads(signal_data)
                            symbol = signal.get('symbol', 'unknown')
                            timestamp = signal.get('timestamp', '')
                            
                            # Remove from symbol-specific list (remove only 1 occurrence)
                            symbol_key = f"signals:symbol:{symbol}"
                            await self.client.lrem(symbol_key, 1, signal_id_str)
                            
                            # Remove from global list (remove only 1 occurrence)
                            await self.client.lrem("signals:list", 1, signal_id_str)
                            
                            # Delete the signal data itself
                            await self.client.delete(signal_id_str)
                            
                            # Delete the signal history if timestamp exists
                            if timestamp:
                                history_key = f"signal_history:{timestamp}"
                                await self.client.delete(history_key)
                            
                    except Exception as e:
                        logger.warning(f"Failed to delete signal {signal_id}: {e}")
                        continue
                
                logger.info(f"Cleared {cleared_count} signals and their history for user {username} from all indices")
            
            return cleared_count
        except Exception as e:
            logger.error(f"Failed to clear signals for user {username}: {e}")
            return 0

    async def delete_signal(self, timestamp: str, username: str) -> bool:
        """Delete a specific signal by timestamp for a user"""
        if not await self.is_connected():
            return False
            
        try:
            user_key = f"signals:user:{username}"
            
            # Get all signal IDs for this user
            signal_ids = await self.client.lrange(user_key, 0, -1)
            
            # Find the signal with matching timestamp
            signal_to_delete = None
            for signal_id in signal_ids:
                try:
                    signal_id_str = signal_id.decode()
                    signal_data = await self.client.get(signal_id_str)
                    if signal_data:
                        # Try JSON first (new format), fallback to pickle (old format)
                        if isinstance(signal_data, bytes):
                            try:
                                signal = json.loads(signal_data.decode('utf-8'))
                            except (json.JSONDecodeError, UnicodeDecodeError):
                                signal = pickle.loads(signal_data)
                        else:
                            signal = json.loads(signal_data)
                        if signal.get('timestamp') == timestamp:
                            signal_to_delete = signal_id_str
                            break
                except Exception as e:
                    logger.warning(f"Failed to process signal {signal_id}: {e}")
                    continue
            
            if not signal_to_delete:
                logger.warning(f"Signal {timestamp} not found for user {username}")
                return False
            
            # Get the signal data to find symbol
            signal_data = await self.client.get(signal_to_delete)
            if signal_data:
                # Try JSON first (new format), fallback to pickle (old format)
                if isinstance(signal_data, bytes):
                    try:
                        signal = json.loads(signal_data.decode('utf-8'))
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        signal = pickle.loads(signal_data)
                else:
                    signal = json.loads(signal_data)
                symbol = signal.get('symbol', 'unknown')
                
                # Remove from user's signal list (remove only 1 occurrence)
                await self.client.lrem(user_key, 1, signal_to_delete)
                
                # Remove from symbol-specific list (remove only 1 occurrence)
                symbol_key = f"signals:symbol:{symbol}"
                await self.client.lrem(symbol_key, 1, signal_to_delete)
                
                # Remove from global list (remove only 1 occurrence)
                await self.client.lrem("signals:list", 1, signal_to_delete)
                
                # Delete the signal data itself
                await self.client.delete(signal_to_delete)
                
                # Delete the signal history
                history_key = f"signal_history:{timestamp}"
                await self.client.delete(history_key)
                
                logger.info(f"Deleted signal {timestamp} and its history for user {username}")
                return True
            else:
                logger.warning(f"Signal data not found for {signal_to_delete}")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting signal {timestamp} for user {username}: {e}")
            return False

    # User Management
    async def store_user(self, username: str, user_data: Dict[str, Any]) -> bool:
        """Store user data in Redis"""
        if not await self.is_connected():
            return False
            
        try:
            user_key = f"user:{username}"
            serialized_user = pickle.dumps(user_data)
            await self.client.setex(user_key, 86400 * 365, serialized_user)  # 1 year TTL
            
            # Add to users list
            await self.client.sadd("users:list", username)
            
            logger.debug(f"Stored user: {username}")
            return True
        except Exception as e:
            logger.error(f"Failed to store user {username}: {e}")
            return False

    async def get_user(self, username: str) -> Optional[Dict[str, Any]]:
        """Get user data from Redis"""
        if not await self.is_connected():
            return None
            
        try:
            user_key = f"user:{username}"
            user_data = await self.client.get(user_key)
            if user_data:
                user = pickle.loads(user_data)
                logger.debug(f"Retrieved user: {username}")
                return user
            return None
        except Exception as e:
            logger.error(f"Failed to retrieve user {username}: {e}")
            return None

    async def get_all_users(self) -> List[Dict[str, Any]]:
        """Get all users from Redis"""
        if not await self.is_connected():
            return []
            
        try:
            usernames = await self.client.smembers("users:list")
            users = []
            
            for username in usernames:
                try:
                    user_data = await self.client.get(f"user:{username.decode()}")
                    if user_data:
                        user = pickle.loads(user_data)
                        users.append(user)
                except Exception as e:
                    logger.warning(f"Failed to load user {username}: {e}")
                    continue
            
            return users
        except Exception as e:
            logger.error(f"Failed to get all users: {e}")
            return []

    async def delete_user(self, username: str) -> bool:
        """Delete user from Redis"""
        if not await self.is_connected():
            return False
            
        try:
            user_key = f"user:{username}"
            await self.client.delete(user_key)
            await self.client.srem("users:list", username)
            
            logger.info(f"Deleted user: {username}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete user {username}: {e}")
            return False

    async def update_user_last_login(self, username: str) -> bool:
        """Update user's last login timestamp"""
        if not await self.is_connected():
            return False
            
        try:
            user = await self.get_user(username)
            if user:
                user["last_login"] = datetime.now().isoformat()
                await self.store_user(username, user)
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to update last login for user {username}: {e}")
            return False
    
    # Configuration Caching
    async def cache_config(self, config_key: str, config_value: Any, ttl: int = 3600) -> bool:
        """Cache configuration value"""
        if not await self.is_connected():
            logger.error(f"Redis not connected, cannot cache config: {config_key}")
            return False
            
        try:
            key = f"config:{config_key}"
            logger.debug(f"Attempting to cache config: {key}, value type: {type(config_value)}")
            serialized_value = pickle.dumps(config_value)
            logger.debug(f"Successfully serialized config: {key}, size: {len(serialized_value)} bytes")
            await self.client.setex(key, ttl, serialized_value)
            logger.debug(f"Cached config: {key}")
            return True
        except Exception as e:
            logger.error(f"Failed to cache config {config_key}: {e}")
            logger.error(f"Config value type: {type(config_value)}")
            logger.error(f"Config value: {config_value}")
            return False
    
    async def get(self, key: str) -> Optional[str]:
        """Generic get method for Redis keys"""
        if not await self.is_connected():
            return None
        try:
            return await self.client.get(key)
        except Exception as e:
            logger.error(f"Failed to get key {key}: {e}")
            return None
    
    async def setex(self, key: str, time: int, value: str) -> bool:
        """Set key with expiration time"""
        if not await self.is_connected():
            return False
        try:
            await self.client.setex(key, time, value)
            return True
        except Exception as e:
            logger.error(f"Failed to setex key {key}: {e}")
            return False

    async def get_cached_config(self, config_key: str) -> Optional[Any]:
        """Retrieve cached configuration"""
        if not await self.is_connected():
            return None
            
        try:
            key = f"config:{config_key}"
            data = await self.client.get(key)
            if data:
                deserialized = pickle.loads(data)
                logger.debug(f"Retrieved cached config: {key}")
                return deserialized
            return None
        except Exception as e:
            logger.error(f"Failed to retrieve cached config: {e}")
            return None
    
    # Performance Metrics
    async def store_performance_metric(self, metric_name: str, value: float, timestamp: Optional[datetime] = None) -> bool:
        """Store performance metric with timestamp"""
        if not await self.is_connected():
            return False
            
        try:
            ts = timestamp or datetime.now()
            key = f"metrics:{metric_name}:{ts.strftime('%Y%m%d')}"
            metric_data = {
                'value': value,
                'timestamp': ts.isoformat()
            }
            serialized = pickle.dumps(metric_data)
            await self.client.zadd(key, {serialized: ts.timestamp()})
            await self.client.expire(key, 86400 * 7)  # Keep for 7 days
            logger.debug(f"Stored metric: {metric_name} = {value}")
            return True
        except Exception as e:
            logger.error(f"Failed to store performance metric: {e}")
            return False
    
    async def get_performance_metrics(self, metric_name: str, days: int = 7) -> List[Dict[str, Any]]:
        """Get performance metrics for the last N days"""
        if not await self.is_connected():
            return []
            
        try:
            metrics = []
            for i in range(days):
                date = datetime.now() - timedelta(days=i)
                key = f"metrics:{metric_name}:{date.strftime('%Y%m%d')}"
                
                try:
                    data = await self.client.zrange(key, 0, -1, withscores=True)
                    for serialized, score in data:
                        metric = pickle.loads(serialized)
                        metrics.append(metric)
                except Exception as e:
                    logger.warning(f"Failed to retrieve metrics for {key}: {e}")
                    continue
            
            logger.debug(f"Retrieved {len(metrics)} performance metrics for {metric_name}")
            return metrics
        except Exception as e:
            logger.error(f"Failed to retrieve performance metrics: {e}")
            return []
    
    # Real-time Data
    async def publish_market_update(self, channel: str, data: Dict[str, Any]) -> bool:
        """Publish market update to Redis channel"""
        if not await self.is_connected():
            return False
            
        try:
            message = json.dumps(data)
            await self.client.publish(channel, message)
            logger.debug(f"Published to channel {channel}: {len(message)} bytes")
            return True
        except Exception as e:
            logger.error(f"Failed to publish market update: {e}")
            return False
    
    async def subscribe_to_channel(self, channel: str, callback) -> bool:
        """Subscribe to Redis channel"""
        if not await self.is_connected():
            return False
            
        try:
            pubsub = self.client.pubsub()
            await pubsub.subscribe(channel)
            
            async def listener():
                try:
                    async for message in pubsub.listen():
                        if message['type'] == 'message':
                            data = json.loads(message['data'])
                            await callback(data)
                except Exception as e:
                    logger.error(f"Error in channel listener: {e}")
                finally:
                    await pubsub.close()
            
            asyncio.create_task(listener())
            logger.info(f"Subscribed to channel: {channel}")
            return True
        except Exception as e:
            logger.error(f"Failed to subscribe to channel: {e}")
            return False
    
    # Utility Methods
    async def clear_cache(self, pattern: str = "*") -> int:
        """Clear cache by pattern"""
        if not await self.is_connected():
            return 0
            
        try:
            keys = await self.client.keys(pattern)
            if keys:
                deleted = await self.client.delete(*keys)
                logger.info(f"Cleared {deleted} cache keys matching pattern: {pattern}")
                return deleted
            return 0
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return 0
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if not await self.is_connected():
            return {"status": "disconnected"}
            
        try:
            info = await self.client.info()
            keys = await self.client.dbsize()
            
            stats = {
                "status": "connected",
                "host": self.host,
                "port": self.port,
                "db": self.db,
                "total_keys": keys,
                "memory_usage": info.get('used_memory_human', 'N/A'),
                "connected_clients": info.get('connected_clients', 0),
                "uptime": info.get('uptime_in_seconds', 0)
            }
            
            logger.debug(f"Cache stats: {stats}")
            return stats
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {"status": "error", "error": str(e)}

    # Signal Storage Methods
    async def store_signal(self, signal: Dict[str, Any]) -> bool:
        """Store a trading signal in Redis"""
        if not await self.is_connected():
            return False
            
        try:
            # Store signal with timestamp as key (using JSON for faster serialization)
            signal_key = f"signal:{signal['timestamp']}"
            
            # Check if signal already exists
            signal_exists = await self.client.exists(signal_key)
            
            await self.client.setex(
                signal_key, 
                86400 * 30,  # Expire after 30 days
                json.dumps(signal, default=str)  # Use JSON instead of pickle
            )
            
            # Ensure indices are updated even if the signal existed before
            # Global list: move to head (dedupe then push)
            await self.client.lrem("signals:list", 0, signal_key)
            await self.client.lpush("signals:list", signal_key)
            await self.client.ltrim("signals:list", 0, 999)  # Keep last 1000 signals

            # Symbol index: move to head (dedupe then push)
            symbol_key = f"signals:symbol:{signal['symbol']}"
            await self.client.lrem(symbol_key, 0, signal_key)
            await self.client.lpush(symbol_key, signal_key)
            await self.client.ltrim(symbol_key, 0, 99)  # Keep last 100 signals per symbol

            # User index: move to head (dedupe then push) if username present
            if 'username' in signal:
                user_key = f"signals:user:{signal['username']}"
                await self.client.lrem(user_key, 0, signal_key)
                await self.client.lpush(user_key, signal_key)
                await self.client.ltrim(user_key, 0, 99)  # Keep last 100 signals per user

            logger.debug(f"Indexed signal: {signal_key}")
            
            return True
        except Exception as e:
            logger.error(f"Failed to store signal: {e}")
            return False



    async def clear_signals(self, symbol: Optional[str] = None) -> int:
        """Clear signals from Redis"""
        if not await self.is_connected():
            return 0
            
        try:
            if symbol:
                # Clear signals for specific symbol
                symbol_key = f"signals:symbol:{symbol}"
                signal_keys = await self.client.lrange(symbol_key, 0, -1)
                if signal_keys:
                    await self.client.delete(*signal_keys)
                    await self.client.delete(symbol_key)
                    return len(signal_keys)
                return 0
            else:
                # Clear all signals
                signal_keys = await self.client.lrange("signals:list", 0, -1)
                if signal_keys:
                    await self.client.delete(*signal_keys)
                    await self.client.delete("signals:list")
                    
                    # Clear symbol indices
                    symbol_keys = await self.client.keys("signals:symbol:*")
                    if symbol_keys:
                        await self.client.delete(*symbol_keys)
                    
                    return len(signal_keys)
                return 0
        except Exception as e:
            logger.error(f"Failed to clear signals: {e}")
            return 0

    async def log_activity(self, event_type: str, description: str, user: str = "system", metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Log an activity event"""
        if not await self.is_connected():
            return False
            
        try:
            activity = {
                "timestamp": datetime.now().isoformat(),
                "event_type": event_type,
                "description": description,
                "user": user,
                "metadata": metadata or {}
            }
            
            # Store in recent activities list (keep last 100)
            await self.client.lpush("activity:recent", json.dumps(activity))
            await self.client.ltrim("activity:recent", 0, 99)  # Keep only last 100
            
            # Also store by date for historical queries
            date_key = datetime.now().strftime("%Y-%m-%d")
            await self.client.lpush(f"activity:date:{date_key}", json.dumps(activity))
            await self.client.ltrim(f"activity:date:{date_key}", 0, 999)  # Keep 1000 per day
            
            # Set expiration for daily logs (30 days)
            await self.client.expire(f"activity:date:{date_key}", 30 * 24 * 3600)
            
            logger.info(f"Activity logged: {event_type} - {description} by {user}")
            return True
        except Exception as e:
            logger.error(f"Failed to log activity: {e}")
            return False

    async def get_recent_activities(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent activities"""
        if not await self.is_connected():
            return []
            
        try:
            activities = await self.client.lrange("activity:recent", 0, limit - 1)
            return [json.loads(activity) for activity in activities]
        except Exception as e:
            logger.error(f"Failed to get recent activities: {e}")
            return []

    # Position Tracking Methods
    async def store_position(self, position: Dict[str, Any]) -> bool:
        """Store a trading position"""
        if not await self.is_connected():
            return False
            
        try:
            position_id = position['id']
            position_key = f"position:{position_id}"
            
            # Store position data
            await self.client.setex(
                position_key,
                86400 * 90,  # Expire after 90 days
                pickle.dumps(position)
            )
            
            # Add to user positions list
            user_key = f"positions:user:{position['username']}"
            await self.client.lpush(user_key, position_key)
            await self.client.ltrim(user_key, 0, 199)  # Keep last 200 positions per user
            
            # Add to all positions list
            await self.client.lpush("positions:all", position_key)
            await self.client.ltrim("positions:all", 0, 999)  # Keep last 1000 positions
            
            logger.debug(f"Stored position: {position_key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store position: {e}")
            return False
    
    async def get_position(self, position_id: str) -> Optional[Dict[str, Any]]:
        """Get a position by ID"""
        if not await self.is_connected():
            return None
            
        try:
            position_key = f"position:{position_id}"
            position_data = await self.client.get(position_key)
            
            if position_data:
                return pickle.loads(position_data)
            return None
            
        except Exception as e:
            logger.error(f"Failed to get position {position_id}: {e}")
            return None
    
    async def get_user_positions(self, username: str) -> List[Dict[str, Any]]:
        """Get all positions for a user"""
        if not await self.is_connected():
            return []
            
        try:
            user_key = f"positions:user:{username}"
            position_keys = await self.client.lrange(user_key, 0, 199)
            
            positions = []
            for position_key in position_keys:
                position_data = await self.client.get(position_key.decode())
                if position_data:
                    positions.append(pickle.loads(position_data))
            
            # Sort by creation date (newest first)
            positions.sort(key=lambda x: x.get('created_at', ''), reverse=True)
            return positions
            
        except Exception as e:
            logger.error(f"Failed to get user positions for {username}: {e}")
            return []
    
    async def get_signal_by_timestamp(self, timestamp: str) -> Optional[Dict[str, Any]]:
        """Get a signal by timestamp"""
        if not await self.is_connected():
            return None
            
        try:
            signal_key = f"signal:{timestamp}"
            signal_data = await self.client.get(signal_key)
            
            if signal_data:
                # Try JSON first (new format), fallback to pickle (old format)
                if isinstance(signal_data, bytes):
                    try:
                        return json.loads(signal_data.decode('utf-8'))
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        return pickle.loads(signal_data)
                else:
                    return json.loads(signal_data)
            return None
            
        except Exception as e:
            logger.error(f"Failed to get signal {timestamp}: {e}")
            return None
    
    # Signal History Methods
    async def add_signal_history_event(self, signal_timestamp: str, event_type: str, 
                                     description: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Add a history event to a signal"""
        if not await self.is_connected():
            return False
            
        try:
            history_key = f"signal_history:{signal_timestamp}"
            
            event = {
                "timestamp": get_israel_time().isoformat(),
                "event_type": event_type,
                "description": description,
                "metadata": metadata or {}
            }
            
            # Add event to history list
            await self.client.lpush(history_key, json.dumps(event))
            await self.client.ltrim(history_key, 0, 19)  # Keep only last 20 events
            await self.client.expire(history_key, 86400 * 90)  # Expire after 90 days
            
            logger.debug(f"Added history event to signal {signal_timestamp}: {event_type}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add signal history event: {e}")
            return False
    
    async def get_signal_history(self, signal_timestamp: str) -> List[Dict[str, Any]]:
        """Get history events for a signal"""
        if not await self.is_connected():
            return []
            
        try:
            history_key = f"signal_history:{signal_timestamp}"
            events = await self.client.lrange(history_key, 0, 19)
            
            history = []
            for event in events:
                try:
                    history.append(json.loads(event))
                except Exception as e:
                    logger.warning(f"Failed to parse history event: {e}")
                    continue
            
            # Sort by timestamp (newest first)
            history.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            return history
            
        except Exception as e:
            logger.error(f"Failed to get signal history for {signal_timestamp}: {e}")
            return []

    async def get_activities_by_date(self, date: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get activities for a specific date (YYYY-MM-DD)"""
        if not await self.is_connected():
            return []
            
        try:
            activities = await self.client.lrange(f"activity:date:{date}", 0, limit - 1)
            return [json.loads(activity) for activity in activities]
        except Exception as e:
            logger.error(f"Failed to get activities for date {date}: {e}")
            return []

    # Telegram connection management methods
    
    async def store_telegram_connection(self, username: str, chat_id: str, 
                                      first_name: str = None, last_name: str = None) -> bool:
        """Store user's Telegram connection"""
        if not await self.is_connected():
            return False
            
        try:
            # Store connection data
            connection_data = {
                "username": username,
                "chat_id": chat_id,
                "first_name": first_name,
                "last_name": last_name,
                "connected_at": datetime.now().isoformat(),
                "notifications_enabled": True
            }
            
            # Store in user:telegram:{username} key
            telegram_key = f"user:telegram:{username}"
            serialized_connection = pickle.dumps(connection_data)
            await self.client.setex(telegram_key, 86400 * 365, serialized_connection)  # 1 year TTL
            
            # Store reverse mapping: telegram:user:{chat_id} -> username
            reverse_key = f"telegram:user:{chat_id}"
            await self.client.setex(reverse_key, 86400 * 365, username)
            
            # Update user record with telegram_chat_id
            user_data = await self.get_user(username)
            if user_data:
                user_data["telegram_chat_id"] = chat_id
                await self.store_user(username, user_data)
            
            logger.info(f"Stored Telegram connection for user: {username} -> {chat_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store Telegram connection for user {username}: {e}")
            return False
    
    async def get_telegram_connection(self, username: str) -> Optional[Dict[str, Any]]:
        """Get user's Telegram connection"""
        if not await self.is_connected():
            return None
            
        try:
            telegram_key = f"user:telegram:{username}"
            connection_data = await self.client.get(telegram_key)
            if connection_data:
                connection = pickle.loads(connection_data)
                logger.debug(f"Retrieved Telegram connection for user: {username}")
                return connection
            return None
        except Exception as e:
            logger.error(f"Failed to retrieve Telegram connection for user {username}: {e}")
            return None
    
    async def get_username_by_telegram_chat_id(self, chat_id: str) -> Optional[str]:
        """Get username by Telegram chat ID"""
        if not await self.is_connected():
            return None
            
        try:
            reverse_key = f"telegram:user:{chat_id}"
            username = await self.client.get(reverse_key)
            if username:
                return username.decode('utf-8')
            return None
        except Exception as e:
            logger.error(f"Failed to get username for Telegram chat ID {chat_id}: {e}")
            return None
    
    async def remove_telegram_connection(self, username: str) -> bool:
        """Remove user's Telegram connection"""
        if not await self.is_connected():
            return False
            
        try:
            # Get connection data first to get chat_id
            connection = await self.get_telegram_connection(username)
            if not connection:
                logger.warning(f"No Telegram connection found for user: {username}")
                return False
            
            chat_id = connection.get("chat_id")
            
            # Remove connection data
            telegram_key = f"user:telegram:{username}"
            await self.client.delete(telegram_key)
            
            # Remove reverse mapping
            if chat_id:
                reverse_key = f"telegram:user:{chat_id}"
                await self.client.delete(reverse_key)
            
            # Update user record to remove telegram_chat_id
            user_data = await self.get_user(username)
            if user_data:
                user_data.pop("telegram_chat_id", None)
                await self.store_user(username, user_data)
            
            logger.info(f"Removed Telegram connection for user: {username}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove Telegram connection for user {username}: {e}")
            return False
    
    async def get_all_telegram_connections(self) -> List[Dict[str, Any]]:
        """Get all Telegram connections"""
        if not await self.is_connected():
            return []
            
        try:
            # Get all user:telegram:* keys
            pattern = "user:telegram:*"
            keys = await self.client.keys(pattern)
            connections = []
            
            for key in keys:
                connection_data = await self.client.get(key)
                if connection_data:
                    connection = pickle.loads(connection_data)
                    connections.append(connection)
            
            logger.debug(f"Retrieved {len(connections)} Telegram connections")
            return connections
            
        except Exception as e:
            logger.error(f"Failed to retrieve Telegram connections: {e}")
            return []

    # Contact Admin Methods
    
    async def send_message_to_admin(self, from_username: str, message: str, subject: str = None) -> bool:
        """Send a message from user to admin"""
        if not await self.is_connected():
            return False
            
        try:
            message_data = {
                "id": f"msg_{get_israel_time().strftime('%Y%m%d_%H%M%S_%f')}_{from_username}",
                "from_username": from_username,
                "to_username": "admin",
                "subject": subject or "Contact Admin",
                "message": message,
                "timestamp": get_israel_time().isoformat(),
                "status": "unread",
                "type": "user_to_admin"
            }
            
            # Store in admin messages list
            admin_messages_key = "admin:messages"
            await self.client.lpush(admin_messages_key, json.dumps(message_data))
            await self.client.ltrim(admin_messages_key, 0, 999)  # Keep last 1000 messages
            await self.client.expire(admin_messages_key, 86400 * 90)  # 90 days TTL
            
            # Also store in user's sent messages
            user_sent_key = f"user:messages:sent:{from_username}"
            await self.client.lpush(user_sent_key, json.dumps(message_data))
            await self.client.ltrim(user_sent_key, 0, 99)  # Keep last 100 sent messages
            await self.client.expire(user_sent_key, 86400 * 90)  # 90 days TTL
            
            logger.info(f"Message sent from {from_username} to admin")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send message from {from_username} to admin: {e}")
            return False
    
    async def send_message_to_user(self, from_username: str, to_username: str, message: str, subject: str = None) -> bool:
        """Send a message from admin to specific user"""
        if not await self.is_connected():
            return False
            
        try:
            message_data = {
                "id": f"msg_{get_israel_time().strftime('%Y%m%d_%H%M%S_%f')}_{from_username}",
                "from_username": from_username,
                "to_username": to_username,
                "subject": subject or "Message from Admin",
                "message": message,
                "timestamp": get_israel_time().isoformat(),
                "status": "unread",
                "type": "admin_to_user"
            }
            
            # Store in user's received messages
            user_received_key = f"user:messages:received:{to_username}"
            await self.client.lpush(user_received_key, json.dumps(message_data))
            await self.client.ltrim(user_received_key, 0, 99)  # Keep last 100 received messages
            await self.client.expire(user_received_key, 86400 * 90)  # 90 days TTL
            
            # Also store in admin's sent messages
            admin_sent_key = f"admin:messages:sent"
            await self.client.lpush(admin_sent_key, json.dumps(message_data))
            await self.client.ltrim(admin_sent_key, 0, 999)  # Keep last 1000 sent messages
            await self.client.expire(admin_sent_key, 86400 * 90)  # 90 days TTL
            
            logger.info(f"Message sent from {from_username} to {to_username}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send message from {from_username} to {to_username}: {e}")
            return False
    
    async def get_admin_messages(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get all messages sent to admin"""
        if not await self.is_connected():
            return []
            
        try:
            admin_messages_key = "admin:messages"
            messages = await self.client.lrange(admin_messages_key, 0, limit - 1)
            
            result = []
            for message in messages:
                try:
                    message_data = json.loads(message)
                    result.append(message_data)
                except Exception as e:
                    logger.warning(f"Failed to parse admin message: {e}")
                    continue
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get admin messages: {e}")
            return []
    
    async def get_admin_sent_messages(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get all messages sent by admin"""
        if not await self.is_connected():
            return []
            
        try:
            admin_sent_key = "admin:messages:sent"
            messages = await self.client.lrange(admin_sent_key, 0, limit - 1)
            
            result = []
            for message in messages:
                try:
                    message_data = json.loads(message)
                    result.append(message_data)
                except Exception as e:
                    logger.warning(f"Failed to parse admin sent message: {e}")
                    continue
            
            # No need to sort - Redis lpush already maintains newest first order
            return result
            
        except Exception as e:
            logger.error(f"Failed to get admin sent messages: {e}")
            return []

    async def bulk_delete_admin_messages(self, message_ids: List[str]) -> int:
        """Bulk delete admin received messages"""
        if not await self.is_connected():
            return 0
            
        try:
            admin_messages_key = "admin:messages"
            messages = await self.client.lrange(admin_messages_key, 0, -1)
            
            # Filter out messages to delete
            remaining_messages = []
            deleted_count = 0
            
            for message in messages:
                try:
                    message_data = json.loads(message)
                    if message_data.get('id') not in message_ids:
                        remaining_messages.append(message)
                    else:
                        deleted_count += 1
                except Exception as e:
                    logger.warning(f"Failed to parse message during bulk delete: {e}")
                    remaining_messages.append(message)  # Keep unparseable messages
            
            # Replace the entire list with remaining messages
            if remaining_messages:
                await self.client.delete(admin_messages_key)
                await self.client.lpush(admin_messages_key, *remaining_messages)
            else:
                await self.client.delete(admin_messages_key)
            
            logger.info(f"Bulk deleted {deleted_count} admin received messages")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to bulk delete admin messages: {e}")
            return 0
    
    async def bulk_delete_admin_sent_messages(self, message_ids: List[str]) -> int:
        """Bulk delete admin sent messages"""
        if not await self.is_connected():
            return 0
            
        try:
            admin_sent_key = "admin:messages:sent"
            messages = await self.client.lrange(admin_sent_key, 0, -1)
            
            # Filter out messages to delete
            remaining_messages = []
            deleted_count = 0
            
            for message in messages:
                try:
                    message_data = json.loads(message)
                    if message_data.get('id') not in message_ids:
                        remaining_messages.append(message)
                    else:
                        deleted_count += 1
                except Exception as e:
                    logger.warning(f"Failed to parse message during bulk delete: {e}")
                    remaining_messages.append(message)  # Keep unparseable messages
            
            # Replace the entire list with remaining messages
            if remaining_messages:
                await self.client.delete(admin_sent_key)
                await self.client.lpush(admin_sent_key, *remaining_messages)
            else:
                await self.client.delete(admin_sent_key)
            
            logger.info(f"Bulk deleted {deleted_count} admin sent messages")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to bulk delete admin sent messages: {e}")
            return 0

    async def get_user_messages(self, username: str, message_type: str = "received", limit: int = 50) -> List[Dict[str, Any]]:
        """Get messages for a specific user (received or sent)"""
        if not await self.is_connected():
            return []
            
        try:
            if message_type == "received":
                messages_key = f"user:messages:received:{username}"
            else:  # sent
                messages_key = f"user:messages:sent:{username}"
            
            messages = await self.client.lrange(messages_key, 0, limit - 1)
            
            result = []
            for message in messages:
                try:
                    message_data = json.loads(message)
                    result.append(message_data)
                except Exception as e:
                    logger.warning(f"Failed to parse user message: {e}")
                    continue
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get {message_type} messages for user {username}: {e}")
            return []
    
    async def mark_message_as_read(self, message_id: str, username: str) -> bool:
        """Mark a message as read"""
        if not await self.is_connected():
            return False
            
        try:
            # Update in admin messages (limit to recent messages for better performance)
            admin_messages_key = "admin:messages"
            messages = await self.client.lrange(admin_messages_key, 0, 199)  # Reduced from 999 to 200
            
            admin_updated_count = 0
            for i, message in enumerate(messages):
                try:
                    message_data = json.loads(message)
                    if message_data.get("id") == message_id and message_data.get("status") == "unread":
                        message_data["status"] = "read"
                        await self.client.lset(admin_messages_key, i, json.dumps(message_data))
                        admin_updated_count += 1
                        break  # Found and updated, no need to continue
                except Exception as e:
                    logger.warning(f"Failed to update admin message: {e}")
                    continue
            
            if admin_updated_count > 0:
                logger.debug(f"Updated admin message with ID {message_id}")
            
            # Update in user messages (limit to recent messages for better performance)
            user_messages_key = f"user:messages:received:{username}"
            messages = await self.client.lrange(user_messages_key, 0, 99)  # Increased to 100 to ensure we find the message
            
            updated_count = 0
            for i, message in enumerate(messages):
                try:
                    message_data = json.loads(message)
                    if message_data.get("id") == message_id and message_data.get("status") == "unread":
                        message_data["status"] = "read"
                        await self.client.lset(user_messages_key, i, json.dumps(message_data))
                        updated_count += 1
                        break  # Found and updated, no need to continue
                except Exception as e:
                    logger.warning(f"Failed to update user message: {e}")
                    continue
            
            if updated_count > 0:
                logger.debug(f"Updated user message with ID {message_id}")
                return True
            
            # If no messages were updated, return False
            logger.warning(f"No message found with ID {message_id} for user {username}")
            return False
            
        except Exception as e:
            logger.error(f"Failed to mark message {message_id} as read: {e}")
            return False
    
    async def get_unread_message_count(self, username: str) -> int:
        """Get count of unread messages for a user"""
        if not await self.is_connected():
            return 0
            
        try:
            unread_count = 0
            
            # Count unread messages received by the user
            user_messages_key = f"user:messages:received:{username}"
            messages = await self.client.lrange(user_messages_key, 0, 99)
            
            for message in messages:
                try:
                    message_data = json.loads(message)
                    if message_data.get("status") == "unread":
                        unread_count += 1
                except Exception as e:
                    logger.warning(f"Failed to parse message for unread count: {e}")
                    continue
            
            # If user is admin, also count unread messages sent TO admin
            if username == "admin":
                admin_messages_key = "admin:messages"
                admin_messages = await self.client.lrange(admin_messages_key, 0, 99)
                
                for message in admin_messages:
                    try:
                        message_data = json.loads(message)
                        if message_data.get("status") == "unread" and message_data.get("to_username") == "admin":
                            unread_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to parse admin message for unread count: {e}")
                        continue
            
            return unread_count
            
        except Exception as e:
            logger.error(f"Failed to get unread message count for user {username}: {e}")
            return 0
    
    async def mark_admin_message_as_read(self, message_id: str) -> bool:
        """Mark an admin message as read"""
        if not await self.is_connected():
            return False
            
        try:
            admin_messages_key = "admin:messages"
            messages = await self.client.lrange(admin_messages_key, 0, 99)
            
            for i, message in enumerate(messages):
                try:
                    message_data = json.loads(message)
                    if message_data.get("id") == message_id:
                        # Update the message status to read
                        message_data["status"] = "read"
                        updated_message = json.dumps(message_data)
                        
                        # Replace the message in the list
                        await self.client.lset(admin_messages_key, i, updated_message)
                        
                        logger.info(f"Marked admin message {message_id} as read")
                        return True
                        
                except Exception as e:
                    logger.warning(f"Failed to parse admin message for mark as read: {e}")
                    continue
            
            logger.warning(f"Admin message {message_id} not found")
            return False
            
        except Exception as e:
            logger.error(f"Failed to mark admin message {message_id} as read: {e}")
            return False
    
    async def get_unread_messages_from_user_to_admin(self, username: str) -> int:
        """Get count of unread messages from a specific user to admin"""
        if not await self.is_connected():
            return 0
            
        try:
            admin_messages_key = "admin:messages"
            messages = await self.client.lrange(admin_messages_key, 0, 99)
            
            unread_count = 0
            for message in messages:
                try:
                    message_data = json.loads(message)
                    if (message_data.get("from_username") == username and 
                        message_data.get("to_username") == "admin" and 
                        message_data.get("status") == "unread"):
                        unread_count += 1
                except Exception as e:
                    logger.warning(f"Failed to parse admin message for unread count: {e}")
                    continue
            
            return unread_count
            
        except Exception as e:
            logger.error(f"Failed to get unread messages from {username} to admin: {e}")
            return 0
    
    async def cleanup_duplicate_messages(self, username: str) -> int:
        """Clean up duplicate messages for a user"""
        if not await self.is_connected():
            return 0
            
        try:
            user_messages_key = f"user:messages:received:{username}"
            messages = await self.client.lrange(user_messages_key, 0, 99)
            
            # Group messages by ID
            message_groups = {}
            for message in messages:
                try:
                    message_data = json.loads(message)
                    message_id = message_data.get("id")
                    if message_id:
                        if message_id not in message_groups:
                            message_groups[message_id] = []
                        message_groups[message_id].append(message_data)
                except Exception as e:
                    logger.warning(f"Failed to parse message for cleanup: {e}")
                    continue
            
            # Remove duplicates, keeping only the first occurrence
            cleaned_messages = []
            removed_count = 0
            
            for message_id, message_list in message_groups.items():
                if len(message_list) > 1:
                    # Keep the first message, remove duplicates
                    cleaned_messages.append(json.dumps(message_list[0]))
                    removed_count += len(message_list) - 1
                    logger.info(f"Removed {len(message_list) - 1} duplicate messages with ID {message_id}")
                else:
                    # Keep single messages as is
                    cleaned_messages.append(json.dumps(message_list[0]))
            
            # Update the Redis list with cleaned messages
            if removed_count > 0:
                # Clear the existing list
                await self.client.delete(user_messages_key)
                
                # Add cleaned messages back
                if cleaned_messages:
                    await self.client.lpush(user_messages_key, *cleaned_messages)
                    await self.client.ltrim(user_messages_key, 0, 99)
                    await self.client.expire(user_messages_key, 86400 * 90)
                
                logger.info(f"Cleaned up {removed_count} duplicate messages for user {username}")
            
            return removed_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup duplicate messages for user {username}: {e}")
            return 0
    
    async def delete_messages(self, message_ids: list, username: str) -> int:
        """Delete selected messages for a user (both received and sent)"""
        if not await self.is_connected():
            return 0
            
        try:
            total_deleted_count = 0
            
            # Delete from received messages
            received_messages_key = f"user:messages:received:{username}"
            received_messages = await self.client.lrange(received_messages_key, 0, 99)
            
            remaining_received = []
            received_deleted_count = 0
            
            for message in received_messages:
                try:
                    message_data = json.loads(message)
                    message_id = message_data.get("id")
                    
                    if message_id in message_ids:
                        received_deleted_count += 1
                        logger.info(f"Deleting received message {message_id} for user {username}")
                    else:
                        remaining_received.append(message)
                        
                except Exception as e:
                    logger.warning(f"Failed to parse received message for deletion: {e}")
                    remaining_received.append(message)
                    continue
            
            # Update received messages list
            if received_deleted_count > 0:
                await self.client.delete(received_messages_key)
                if remaining_received:
                    await self.client.lpush(received_messages_key, *remaining_received)
                    await self.client.ltrim(received_messages_key, 0, 99)
                    await self.client.expire(received_messages_key, 86400 * 90)
            
            total_deleted_count += received_deleted_count
            
            # Delete from sent messages
            sent_messages_key = f"user:messages:sent:{username}"
            sent_messages = await self.client.lrange(sent_messages_key, 0, 99)
            
            remaining_sent = []
            sent_deleted_count = 0
            
            for message in sent_messages:
                try:
                    message_data = json.loads(message)
                    message_id = message_data.get("id")
                    
                    if message_id in message_ids:
                        sent_deleted_count += 1
                        logger.info(f"Deleting sent message {message_id} for user {username}")
                    else:
                        remaining_sent.append(message)
                        
                except Exception as e:
                    logger.warning(f"Failed to parse sent message for deletion: {e}")
                    remaining_sent.append(message)
                    continue
            
            # Update sent messages list
            if sent_deleted_count > 0:
                await self.client.delete(sent_messages_key)
                if remaining_sent:
                    await self.client.lpush(sent_messages_key, *remaining_sent)
                    await self.client.ltrim(sent_messages_key, 0, 99)
                    await self.client.expire(sent_messages_key, 86400 * 90)
            
            total_deleted_count += sent_deleted_count
            
            if total_deleted_count > 0:
                logger.info(f"Deleted {total_deleted_count} messages for user {username} (received: {received_deleted_count}, sent: {sent_deleted_count})")
            
            return total_deleted_count
            
        except Exception as e:
            logger.error(f"Failed to delete messages for user {username}: {e}")
            return 0
    
    async def delete_admin_messages(self, message_ids: list) -> int:
        """Delete selected messages from admin messages"""
        if not await self.is_connected():
            return 0
            
        try:
            admin_messages_key = "admin:messages"
            messages = await self.client.lrange(admin_messages_key, 0, 999)
            
            # Filter out messages to delete
            remaining_messages = []
            deleted_count = 0
            
            for message in messages:
                try:
                    message_data = json.loads(message)
                    message_id = message_data.get("id")
                    
                    if message_id in message_ids:
                        deleted_count += 1
                        logger.info(f"Deleting admin message {message_id}")
                    else:
                        remaining_messages.append(message)
                        
                except Exception as e:
                    logger.warning(f"Failed to parse admin message for deletion: {e}")
                    # Keep unparseable messages
                    remaining_messages.append(message)
                    continue
            
            # Update the Redis list with remaining messages
            if deleted_count > 0:
                # Clear the existing list
                await self.client.delete(admin_messages_key)
                
                # Add remaining messages back
                if remaining_messages:
                    await self.client.lpush(admin_messages_key, *remaining_messages)
                    await self.client.ltrim(admin_messages_key, 0, 999)
                    await self.client.expire(admin_messages_key, 86400 * 90)
                
                logger.info(f"Deleted {deleted_count} admin messages")
            
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to delete admin messages: {e}")
            return 0

# Global Redis instance
redis_client: Optional[TradingAgentRedis] = None

async def get_redis_client() -> Optional[TradingAgentRedis]:
    """Get global Redis client instance"""
    global redis_client
    
    if redis_client is None:
        # Initialize from configuration
        from ..config import get_config
        config = get_config()
        
        host = config.database.redis_host
        port = config.database.redis_port
        db = config.database.redis_db
        password = config.database.redis_password
        
        redis_client = TradingAgentRedis(host=host, port=port, db=db, password=password)
        await redis_client.connect()
    
    return redis_client

async def create_fresh_redis_client() -> Optional[TradingAgentRedis]:
    """Create a fresh Redis client instance (useful for separate event loops)"""
    try:
        from ..config import get_config
        config = get_config()
        
        host = config.database.redis_host
        port = config.database.redis_port
        db = config.database.redis_db
        password = config.database.redis_password
        
        client = TradingAgentRedis(host=host, port=port, db=db, password=password)
        connected = await client.connect()
        
        if connected:
            return client
        else:
            logger.error("Failed to connect fresh Redis client")
            return None
            
    except Exception as e:
        logger.error(f"Failed to create fresh Redis client: {e}")
        return None

async def close_redis_client():
    """Close global Redis client"""
    global redis_client
    if redis_client:
        await redis_client.disconnect()
        redis_client = None

