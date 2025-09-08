"""
Telegram Bot API Client for sending notifications to users
"""

import asyncio
import aiohttp
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import pytz

# Israel timezone
ISRAEL_TZ = pytz.timezone('Asia/Jerusalem')

def get_israel_time():
    """Get current time in Israel timezone"""
    return datetime.now(ISRAEL_TZ)

logger = logging.getLogger(__name__)

@dataclass
class TelegramMessage:
    """Represents a Telegram message to be sent"""
    chat_id: str
    text: str
    parse_mode: str = "HTML"
    disable_web_page_preview: bool = True
    disable_notification: bool = False

@dataclass
class TelegramUser:
    """Represents a Telegram user connection"""
    username: str
    chat_id: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    connected_at: Optional[datetime] = None
    notifications_enabled: bool = True

class TelegramClient:
    """Telegram Bot API client for sending notifications"""
    
    def __init__(self, config: Any = None, bot_token: str = None, api_url: str = "https://api.telegram.org/bot", 
                 timeout: int = 10, max_retries: int = 3):
        # Handle both config object and direct parameters
        if config and hasattr(config, 'telegram'):
            telegram_config = config.telegram
            self.bot_token = getattr(telegram_config, 'bot_token', '')
            self.enabled = getattr(telegram_config, 'enabled', False)
            self.api_url = getattr(telegram_config, 'api_url', api_url).rstrip('/')
            self.timeout = getattr(telegram_config, 'timeout', timeout)
            self.max_retries = getattr(telegram_config, 'max_retries', max_retries)
        else:
            self.bot_token = bot_token or ''
            self.enabled = bool(bot_token)
            self.api_url = api_url.rstrip('/')
            self.timeout = timeout
            self.max_retries = max_retries
            
        self.session: Optional[aiohttp.ClientSession] = None
        self.last_error = None  # Store the last error for better error reporting
        
        # Rate limiting tracking
        self.user_last_message: Dict[str, datetime] = {}
        self.user_message_count: Dict[str, int] = {}
        self.rate_limit_window = timedelta(minutes=1)
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    def _get_api_url(self, method: str) -> str:
        """Get full API URL for a method"""
        return f"{self.api_url}{self.bot_token}/{method}"
    
    async def _make_request(self, method: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make a request to Telegram Bot API with retries"""
        url = self._get_api_url(method)
        
        for attempt in range(self.max_retries):
            try:
                async with self.session.post(url, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        if result.get("ok"):
                            return result.get("result", {})
                        else:
                            error_msg = result.get("description", "Unknown error")
                            self.last_error = error_msg  # Store the error for better reporting
                            logger.error(f"Telegram API error: {error_msg}")
                            raise Exception(f"Telegram API error: {error_msg}")
                    else:
                        logger.warning(f"Telegram API HTTP {response.status} on attempt {attempt + 1}")
                        # Try to get error details from response
                        try:
                            error_data = await response.json()
                            if error_data.get("description"):
                                self.last_error = error_data.get("description")
                                logger.error(f"Telegram API error details: {error_data.get('description')}")
                        except:
                            self.last_error = f"HTTP {response.status} error"
                            logger.error(f"Telegram API HTTP {response.status} error")
                        
            except asyncio.TimeoutError:
                logger.warning(f"Telegram API timeout on attempt {attempt + 1}")
            except Exception as e:
                logger.warning(f"Telegram API request failed on attempt {attempt + 1}: {e}")
                
            if attempt < self.max_retries - 1:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        # Don't overwrite the specific error with generic message
        if not self.last_error:
            self.last_error = f"Failed to send Telegram message after {self.max_retries} attempts"
        raise Exception(f"Failed to send Telegram message after {self.max_retries} attempts")
    
    async def send_message(self, message: TelegramMessage) -> bool:
        """Send a message to a Telegram chat"""
        if not self.session:
            raise Exception("Telegram client not initialized. Use async context manager.")
        
        if not self.bot_token:
            logger.warning("Telegram bot token not configured")
            return False
        
        try:
            data = {
                "chat_id": message.chat_id,
                "text": message.text,
                "parse_mode": message.parse_mode,
                "disable_web_page_preview": message.disable_web_page_preview,
                "disable_notification": message.disable_notification
            }
            
            result = await self._make_request("sendMessage", data)
            logger.info(f"Telegram message sent to chat {message.chat_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False
    
    async def send_simple_message(self, chat_id: str, text: str, parse_mode: str = "HTML") -> bool:
        """Send a simple message to a Telegram chat (convenience method)"""
        message = TelegramMessage(
            chat_id=chat_id,
            text=text,
            parse_mode=parse_mode
        )
        return await self.send_message(message)
    
    async def get_bot_info(self) -> Optional[Dict[str, Any]]:
        """Get bot information"""
        if not self.session:
            raise Exception("Telegram client not initialized. Use async context manager.")
        
        if not self.bot_token:
            return None
        
        try:
            result = await self._make_request("getMe", {})
            return result
        except Exception as e:
            logger.error(f"Failed to get bot info: {e}")
            return None
    
    async def get_chat_info(self, chat_id: str) -> Optional[Dict[str, Any]]:
        """Get chat information"""
        if not self.session:
            raise Exception("Telegram client not initialized. Use async context manager.")
        
        if not self.bot_token:
            return None
        
        try:
            data = {"chat_id": chat_id}
            result = await self._make_request("getChat", data)
            return result
        except Exception as e:
            logger.error(f"Failed to get chat info for {chat_id}: {e}")
            return None
    
    def can_send_message(self, chat_id: str, cooldown_seconds: int = 5, 
                        max_per_minute: int = 10) -> bool:
        """Check if we can send a message to a user (rate limiting)"""
        now = get_israel_time()
        
        # Check cooldown
        if chat_id in self.user_last_message:
            time_since_last = now - self.user_last_message[chat_id]
            if time_since_last.total_seconds() < cooldown_seconds:
                return False
        
        # Check rate limit
        if chat_id in self.user_message_count:
            # Reset counter if window has passed
            if now - self.user_last_message.get(chat_id, now) > self.rate_limit_window:
                self.user_message_count[chat_id] = 0
            
            if self.user_message_count[chat_id] >= max_per_minute:
                return False
        
        return True
    
    def record_message_sent(self, chat_id: str):
        """Record that a message was sent to a user"""
        now = get_israel_time()
        self.user_last_message[chat_id] = now
        
        if chat_id in self.user_message_count:
            self.user_message_count[chat_id] += 1
        else:
            self.user_message_count[chat_id] = 1
    
    async def send_notification(self, chat_id: str, text: str, 
                              cooldown_seconds: int = 5, max_per_minute: int = 10) -> bool:
        """Send a notification with rate limiting"""
        if not self.can_send_message(chat_id, cooldown_seconds, max_per_minute):
            logger.warning(f"Rate limit exceeded for chat {chat_id}")
            return False
        
        message = TelegramMessage(chat_id=chat_id, text=text)
        success = await self.send_message(message)
        
        if success:
            self.record_message_sent(chat_id)
        
        return success

class TelegramNotificationManager:
    """Manages Telegram notifications for the trading system"""
    
    def __init__(self, config):
        self.config = config
        self.client: Optional[TelegramClient] = None
        self.user_connections: Dict[str, TelegramUser] = {}
        
    async def initialize(self):
        """Initialize the Telegram client"""
        if not self.config.telegram.enabled or not self.config.telegram.bot_token:
            logger.info("Telegram notifications disabled or bot token not configured")
            return False
        
        self.client = TelegramClient(
            bot_token=self.config.telegram.bot_token,
            api_url=self.config.telegram.api_url,
            timeout=self.config.telegram.timeout,
            max_retries=self.config.telegram.max_retries
        )
        
        # Test bot connection
        async with self.client:
            bot_info = await self.client.get_bot_info()
            if bot_info:
                logger.info(f"Telegram bot connected: @{bot_info.get('username', 'unknown')}")
                return True
            else:
                logger.error("Failed to connect to Telegram bot")
                return False
    
    async def send_signal_created_notification(self, username: str, symbol: str, 
                                             signal_type: str, confidence: float):
        """Send notification when a new signal is created"""
        if not self._should_send_notification(username, "signal_created"):
            return
        
        text = f"🎯 <b>New Trading Tip Created</b>\n\n"
        text += f"<b>Symbol:</b> {symbol}\n"
        text += f"<b>Signal:</b> {signal_type}\n"
        text += f"<b>Confidence:</b> {confidence:.2%}\n\n"
        text += f"<i>Generated at {get_israel_time().strftime('%Y-%m-%d %H:%M:%S')}</i>"
        
        await self._send_to_user(username, text)
    
    async def send_signal_changed_notification(self, username: str, symbol: str, 
                                             old_status: str, new_status: str, 
                                             confidence: float):
        """Send notification when a signal status changes"""
        if not self._should_send_notification(username, "signal_changed"):
            return
        
        text = f"🔄 <b>Signal Status Updated</b>\n\n"
        text += f"<b>Symbol:</b> {symbol}\n"
        text += f"<b>Changed:</b> {old_status} → {new_status}\n"
        text += f"<b>Confidence:</b> {confidence:.2%}\n\n"
        text += f"<i>Updated at {get_israel_time().strftime('%Y-%m-%d %H:%M:%S')}</i>"
        
        await self._send_to_user(username, text)
    
    async def send_signal_deleted_notification(self, username: str, symbol: str):
        """Send notification when a signal is deleted"""
        if not self._should_send_notification(username, "signal_deleted"):
            return
        
        text = f"🗑️ <b>Signal Deleted</b>\n\n"
        text += f"<b>Symbol:</b> {symbol}\n\n"
        text += f"<i>Deleted at {get_israel_time().strftime('%Y-%m-%d %H:%M:%S')}</i>"
        
        await self._send_to_user(username, text)
    
    async def send_maintenance_notification(self, username: str, message: str):
        """Send maintenance notification"""
        if not self._should_send_notification(username, "maintenance"):
            return
        
        text = f"🔧 <b>System Maintenance Notice</b>\n\n"
        text += f"{message}\n\n"
        text += f"<i>Sent at {get_israel_time().strftime('%Y-%m-%d %H:%M:%S')}</i>"
        
        await self._send_to_user(username, text)
    
    def _should_send_notification(self, username: str, notification_type: str) -> bool:
        """Check if we should send a notification"""
        if not self.client or not self.config.telegram.enabled:
            return False
        
        # Check if user has Telegram connected
        if username not in self.user_connections:
            return False
        
        user = self.user_connections[username]
        if not user.notifications_enabled:
            return False
        
        # Check notification type settings
        if notification_type == "signal_created" and not self.config.telegram.notifications.signal_created:
            return False
        elif notification_type == "signal_changed" and not self.config.telegram.notifications.signal_changed:
            return False
        elif notification_type == "signal_deleted" and not self.config.telegram.notifications.signal_deleted:
            return False
        elif notification_type == "maintenance" and not self.config.telegram.notifications.maintenance:
            return False
        
        return True
    
    async def _send_to_user(self, username: str, text: str):
        """Send a message to a specific user"""
        if username not in self.user_connections:
            logger.warning(f"No Telegram connection found for user {username}")
            return
        
        user = self.user_connections[username]
        
        async with self.client:
            success = await self.client.send_notification(
                chat_id=user.chat_id,
                text=text,
                cooldown_seconds=self.config.telegram.notifications.cooldown_seconds,
                max_per_minute=self.config.telegram.notifications.max_notifications_per_minute
            )
            
            if success:
                logger.info(f"Telegram notification sent to user {username}")
            else:
                logger.warning(f"Failed to send Telegram notification to user {username}")
    
    def add_user_connection(self, username: str, chat_id: str, 
                          first_name: str = None, last_name: str = None):
        """Add a user's Telegram connection"""
        user = TelegramUser(
            username=username,
            chat_id=chat_id,
            first_name=first_name,
            last_name=last_name,
            connected_at=get_israel_time(),
            notifications_enabled=True
        )
        self.user_connections[username] = user
        logger.info(f"Added Telegram connection for user {username}")
    
    def remove_user_connection(self, username: str):
        """Remove a user's Telegram connection"""
        if username in self.user_connections:
            del self.user_connections[username]
            logger.info(f"Removed Telegram connection for user {username}")
    
    def get_user_connection(self, username: str) -> Optional[TelegramUser]:
        """Get a user's Telegram connection"""
        return self.user_connections.get(username)
    
    def is_user_connected(self, username: str) -> bool:
        """Check if a user has Telegram connected"""
        return username in self.user_connections

# Global Telegram client instance
_telegram_client: Optional[TelegramClient] = None

def get_telegram_client() -> Optional[TelegramClient]:
    """Get global Telegram client instance"""
    global _telegram_client
    
    if _telegram_client is None:
        try:
            from agent.config import load_config_from_env
            config = load_config_from_env()
            _telegram_client = TelegramClient(config)
        except Exception as e:
            logger.error(f"Failed to create Telegram client: {e}")
            return None
    
    return _telegram_client
