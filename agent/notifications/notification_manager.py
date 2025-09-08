"""
Central notification manager for the trading system
"""

import asyncio
import logging
from typing import Optional, Dict, Any
from datetime import datetime
import pytz

from .telegram_client import TelegramNotificationManager, TelegramUser
from .message_templates import TelegramMessageTemplates

# Israel timezone
ISRAEL_TZ = pytz.timezone('Asia/Jerusalem')

def get_israel_time():
    """Get current time in Israel timezone"""
    return datetime.now(ISRAEL_TZ)

logger = logging.getLogger(__name__)

class NotificationManager:
    """Central manager for all notification systems"""
    
    def __init__(self, config):
        self.config = config
        self.telegram_manager: Optional[TelegramNotificationManager] = None
        self.initialized = False
    
    async def initialize(self):
        """Initialize all notification systems"""
        try:
            # Initialize Telegram notifications
            if self.config.telegram.enabled:
                self.telegram_manager = TelegramNotificationManager(self.config)
                telegram_success = await self.telegram_manager.initialize()
                if telegram_success:
                    logger.info("Telegram notifications initialized successfully")
                else:
                    logger.warning("Failed to initialize Telegram notifications")
                    self.telegram_manager = None
            
            self.initialized = True
            logger.info("Notification manager initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize notification manager: {e}")
            self.initialized = False
    
    async def send_signal_created_notification(self, username: str, symbol: str, 
                                             signal_type: str, confidence: float,
                                             technical_score: float = None, 
                                             sentiment_score: float = None,
                                             created_by: str = None):
        """Send notification when a new signal is created"""
        if not self.initialized:
            return
        
        try:
            # Send Telegram notification
            if self.telegram_manager:
                await self.telegram_manager.send_signal_created_notification(
                    username=username,
                    symbol=symbol,
                    signal_type=signal_type,
                    confidence=confidence
                )
            
            logger.info(f"Signal created notification sent to {username} for {symbol}")
            
        except Exception as e:
            logger.error(f"Failed to send signal created notification: {e}")
    
    async def send_signal_changed_notification(self, username: str, symbol: str,
                                             old_status: str, new_status: str,
                                             confidence: float,
                                             technical_score: float = None,
                                             sentiment_score: float = None):
        """Send notification when a signal status changes"""
        if not self.initialized:
            return
        
        try:
            # Send Telegram notification
            if self.telegram_manager:
                await self.telegram_manager.send_signal_changed_notification(
                    username=username,
                    symbol=symbol,
                    old_status=old_status,
                    new_status=new_status,
                    confidence=confidence
                )
            
            logger.info(f"Signal changed notification sent to {username} for {symbol}")
            
        except Exception as e:
            logger.error(f"Failed to send signal changed notification: {e}")
    
    async def send_signal_deleted_notification(self, username: str, symbol: str,
                                             deleted_by: str = None):
        """Send notification when a signal is deleted"""
        if not self.initialized:
            return
        
        try:
            # Send Telegram notification
            if self.telegram_manager:
                await self.telegram_manager.send_signal_deleted_notification(
                    username=username,
                    symbol=symbol
                )
            
            logger.info(f"Signal deleted notification sent to {username} for {symbol}")
            
        except Exception as e:
            logger.error(f"Failed to send signal deleted notification: {e}")
    
    async def send_maintenance_notification(self, username: str, message: str):
        """Send maintenance notification to a user"""
        if not self.initialized:
            return
        
        try:
            # Send Telegram notification
            if self.telegram_manager:
                await self.telegram_manager.send_maintenance_notification(
                    username=username,
                    message=message
                )
            
            logger.info(f"Maintenance notification sent to {username}")
            
        except Exception as e:
            logger.error(f"Failed to send maintenance notification: {e}")
    
    async def send_maintenance_notification_to_all(self, message: str):
        """Send maintenance notification to all connected users"""
        if not self.initialized or not self.telegram_manager:
            return
        
        try:
            # Get all connected users
            connected_users = list(self.telegram_manager.user_connections.keys())
            
            # Send to all users
            tasks = []
            for username in connected_users:
                task = self.send_maintenance_notification(username, message)
                tasks.append(task)
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
                logger.info(f"Maintenance notification sent to {len(connected_users)} users")
            
        except Exception as e:
            logger.error(f"Failed to send maintenance notification to all users: {e}")
    
    def add_telegram_connection(self, username: str, chat_id: str,
                              first_name: str = None, last_name: str = None):
        """Add a user's Telegram connection"""
        if self.telegram_manager:
            self.telegram_manager.add_user_connection(
                username=username,
                chat_id=chat_id,
                first_name=first_name,
                last_name=last_name
            )
    
    def remove_telegram_connection(self, username: str):
        """Remove a user's Telegram connection"""
        if self.telegram_manager:
            self.telegram_manager.remove_user_connection(username)
    
    def get_telegram_connection(self, username: str) -> Optional[TelegramUser]:
        """Get a user's Telegram connection"""
        if self.telegram_manager:
            return self.telegram_manager.get_user_connection(username)
        return None
    
    def is_telegram_connected(self, username: str) -> bool:
        """Check if a user has Telegram connected"""
        if self.telegram_manager:
            return self.telegram_manager.is_user_connected(username)
        return False
    
    def get_telegram_connection_count(self) -> int:
        """Get the number of users with Telegram connections"""
        if self.telegram_manager:
            return len(self.telegram_manager.user_connections)
        return 0
    
    def get_telegram_connected_users(self) -> list:
        """Get list of usernames with Telegram connections"""
        if self.telegram_manager:
            return list(self.telegram_manager.user_connections.keys())
        return []
    
    async def send_test_notification(self, username: str, message: str = None):
        """Send a test notification to verify connection"""
        if not self.initialized:
            return False
        
        if not message:
            message = f"🧪 <b>Test Notification</b>\n\nThis is a test message to verify your Telegram connection is working properly.\n\n<i>Sent at {get_israel_time().strftime('%Y-%m-%d %H:%M:%S')}</i>"
        
        try:
            if self.telegram_manager:
                await self.telegram_manager._send_to_user(username, message)
                return True
        except Exception as e:
            logger.error(f"Failed to send test notification: {e}")
        
        return False
