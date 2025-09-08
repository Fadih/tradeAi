"""
Message templates for Telegram notifications
"""

from typing import Dict, Any
from datetime import datetime
import pytz

# Israel timezone
ISRAEL_TZ = pytz.timezone('Asia/Jerusalem')

def get_israel_time():
    """Get current time in Israel timezone"""
    return datetime.now(ISRAEL_TZ)

class TelegramMessageTemplates:
    """Templates for Telegram notification messages"""
    
    @staticmethod
    def signal_created(symbol: str, signal_type: str, confidence: float, 
                      technical_score: float = None, sentiment_score: float = None,
                      created_by: str = None) -> str:
        """Template for new signal created notification"""
        text = f"🎯 <b>New Trading Tip Created</b>\n\n"
        text += f"<b>Symbol:</b> {symbol}\n"
        text += f"<b>Signal:</b> {signal_type}\n"
        text += f"<b>Confidence:</b> {confidence:.2%}\n"
        
        if technical_score is not None:
            text += f"<b>Technical Score:</b> {technical_score:.4f}\n"
        
        if sentiment_score is not None:
            text += f"<b>Sentiment Score:</b> {sentiment_score:.4f}\n"
        
        if created_by:
            text += f"<b>Created by:</b> {created_by}\n"
        
        text += f"\n<i>Generated at {get_israel_time().strftime('%Y-%m-%d %H:%M:%S')}</i>"
        
        return text
    
    @staticmethod
    def signal_changed(symbol: str, old_status: str, new_status: str, 
                      confidence: float, technical_score: float = None, 
                      sentiment_score: float = None) -> str:
        """Template for signal status change notification"""
        text = f"🔄 <b>Signal Status Updated</b>\n\n"
        text += f"<b>Symbol:</b> {symbol}\n"
        text += f"<b>Changed:</b> {old_status} → {new_status}\n"
        text += f"<b>Confidence:</b> {confidence:.2%}\n"
        
        if technical_score is not None:
            text += f"<b>Technical Score:</b> {technical_score:.4f}\n"
        
        if sentiment_score is not None:
            text += f"<b>Sentiment Score:</b> {sentiment_score:.4f}\n"
        
        text += f"\n<i>Updated at {get_israel_time().strftime('%Y-%m-%d %H:%M:%S')}</i>"
        
        return text
    
    @staticmethod
    def signal_deleted(symbol: str, deleted_by: str = None) -> str:
        """Template for signal deleted notification"""
        text = f"🗑️ <b>Signal Deleted</b>\n\n"
        text += f"<b>Symbol:</b> {symbol}\n"
        
        if deleted_by:
            text += f"<b>Deleted by:</b> {deleted_by}\n"
        
        text += f"\n<i>Deleted at {get_israel_time().strftime('%Y-%m-%d %H:%M:%S')}</i>"
        
        return text
    
    @staticmethod
    def maintenance(message: str) -> str:
        """Template for maintenance notification"""
        text = f"🔧 <b>System Maintenance Notice</b>\n\n"
        text += f"{message}\n\n"
        text += f"<i>Sent at {get_israel_time().strftime('%Y-%m-%d %H:%M:%S')}</i>"
        
        return text
    
    @staticmethod
    def system_error(error_type: str, error_message: str, 
                    affected_symbol: str = None) -> str:
        """Template for system error notification"""
        text = f"⚠️ <b>System Error Alert</b>\n\n"
        text += f"<b>Error Type:</b> {error_type}\n"
        text += f"<b>Message:</b> {error_message}\n"
        
        if affected_symbol:
            text += f"<b>Affected Symbol:</b> {affected_symbol}\n"
        
        text += f"\n<i>Occurred at {get_israel_time().strftime('%Y-%m-%d %H:%M:%S')}</i>"
        
        return text
    
    @staticmethod
    def connection_welcome(username: str, bot_username: str) -> str:
        """Template for welcome message when user connects"""
        text = f"🎉 <b>Welcome to Trading AI Tips!</b>\n\n"
        text += f"Hello {username}! 👋\n\n"
        text += f"You're now connected to receive real-time notifications about your trading signals.\n\n"
        text += f"<b>What you'll receive:</b>\n"
        text += f"• 🎯 New trading tips\n"
        text += f"• 🔄 Signal status updates\n"
        text += f"• 🗑️ Signal deletions\n"
        text += f"• 🔧 System maintenance notices\n\n"
        text += f"<b>Commands:</b>\n"
        text += f"/start - Show this welcome message\n"
        text += f"/stop - Disable notifications\n"
        text += f"/status - Check connection status\n\n"
        text += f"<i>Connected at {get_israel_time().strftime('%Y-%m-%d %H:%M:%S')}</i>"
        
        return text
    
    @staticmethod
    def connection_disconnected(username: str) -> str:
        """Template for disconnection message"""
        text = f"👋 <b>Disconnected from Trading AI Tips</b>\n\n"
        text += f"Goodbye {username}! You will no longer receive notifications.\n\n"
        text += f"To reconnect, visit the dashboard and link your Telegram account again.\n\n"
        text += f"<i>Disconnected at {get_israel_time().strftime('%Y-%m-%d %H:%M:%S')}</i>"
        
        return text
    
    @staticmethod
    def connection_status(username: str, is_connected: bool, 
                         notifications_enabled: bool = True) -> str:
        """Template for connection status message"""
        if is_connected:
            status_icon = "✅"
            status_text = "Connected"
        else:
            status_icon = "❌"
            status_text = "Not Connected"
        
        text = f"{status_icon} <b>Telegram Connection Status</b>\n\n"
        text += f"<b>User:</b> {username}\n"
        text += f"<b>Status:</b> {status_text}\n"
        
        if is_connected:
            text += f"<b>Notifications:</b> {'Enabled' if notifications_enabled else 'Disabled'}\n"
        
        text += f"\n<i>Checked at {get_israel_time().strftime('%Y-%m-%d %H:%M:%S')}</i>"
        
        return text
    
    @staticmethod
    def rate_limit_exceeded(username: str, cooldown_seconds: int) -> str:
        """Template for rate limit exceeded message"""
        text = f"⏰ <b>Rate Limit Exceeded</b>\n\n"
        text += f"Too many notifications sent. Please wait {cooldown_seconds} seconds before sending another message.\n\n"
        text += f"<i>Rate limited at {get_israel_time().strftime('%Y-%m-%d %H:%M:%S')}</i>"
        
        return text
    
    @staticmethod
    def invalid_command(command: str) -> str:
        """Template for invalid command message"""
        text = f"❓ <b>Unknown Command</b>\n\n"
        text += f"Command '{command}' not recognized.\n\n"
        text += f"<b>Available commands:</b>\n"
        text += f"/start - Show welcome message\n"
        text += f"/stop - Disable notifications\n"
        text += f"/status - Check connection status\n"
        text += f"/help - Show this help message\n\n"
        text += f"<i>Command received at {get_israel_time().strftime('%Y-%m-%d %H:%M:%S')}</i>"
        
        return text
    
    @staticmethod
    def test_message(username: str, timestamp: str) -> str:
        """Template for test message"""
        text = f"🧪 <b>Test Message</b>\n\n"
        text += f"Hello {username}! 👋\n\n"
        text += f"This is a test message to verify your Telegram connection is working properly.\n\n"
        text += f"✅ <b>Connection Status:</b> Active\n"
        text += f"📱 <b>Notifications:</b> Enabled\n\n"
        text += f"<i>Test sent at {timestamp}</i>"
        
        return text

# Legacy template for backward compatibility
TEST_MESSAGE_TEMPLATE = "🧪 <b>Test Message</b>\n\nHello {username}! 👋\n\nThis is a test message to verify your Telegram connection is working properly.\n\n✅ <b>Connection Status:</b> Active\n📱 <b>Notifications:</b> Enabled\n\n<i>Test sent at {timestamp}</i>"