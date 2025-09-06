"""
Position Tracking System
Tracks user positions and trading decisions based on signals
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
import pytz

from .cache.redis_client import TradingAgentRedis

logger = logging.getLogger(__name__)

class PositionTracker:
    """Tracks user positions and trading decisions"""
    
    def __init__(self, redis_client: TradingAgentRedis):
        self.redis_client = redis_client
        self.israel_tz = pytz.timezone('Asia/Jerusalem')
    
    async def create_position(self, username: str, signal_timestamp: str, action: str, 
                            quantity: float, price: float, notes: str = "") -> Dict[str, Any]:
        """Create a new position based on a signal"""
        try:
            # Get the original signal
            signal = await self.redis_client.get_signal_by_timestamp(signal_timestamp)
            if not signal:
                raise ValueError(f"Signal {signal_timestamp} not found")
            
            position = {
                "id": f"{username}_{signal_timestamp}_{action}",
                "username": username,
                "signal_timestamp": signal_timestamp,
                "symbol": signal.get('symbol'),
                "timeframe": signal.get('timeframe'),
                "action": action,  # "BUY", "SELL", "CLOSE"
                "quantity": quantity,
                "entry_price": price,
                "current_price": price,
                "status": "OPEN",  # "OPEN", "CLOSED"
                "created_at": datetime.now(self.israel_tz).isoformat(),
                "notes": notes,
                "signal_data": {
                    "signal_type": signal.get('signal_type'),
                    "confidence": signal.get('confidence'),
                    "fused_score": signal.get('fused_score'),
                    "technical_score": signal.get('technical_score'),
                    "sentiment_score": signal.get('sentiment_score'),
                    "applied_thresholds": {
                        "buy_threshold": signal.get('applied_buy_threshold'),
                        "sell_threshold": signal.get('applied_sell_threshold'),
                        "tech_weight": signal.get('applied_tech_weight'),
                        "sentiment_weight": signal.get('applied_sentiment_weight')
                    }
                }
            }
            
            # Store position
            await self.redis_client.store_position(position)
            
            # Log activity
            await self.redis_client.log_activity(
                "position_created",
                f"Position created: {action} {quantity} {signal.get('symbol')} at {price}",
                username,
                position
            )
            
            logger.info(f"Position created: {username} {action} {quantity} {signal.get('symbol')} at {price}")
            return position
            
        except Exception as e:
            logger.error(f"Error creating position: {e}")
            raise
    
    async def update_position(self, position_id: str, current_price: float, 
                            action: str = None, quantity: float = None, 
                            notes: str = "") -> Dict[str, Any]:
        """Update an existing position"""
        try:
            position = await self.redis_client.get_position(position_id)
            if not position:
                raise ValueError(f"Position {position_id} not found")
            
            # Update current price
            position['current_price'] = current_price
            position['last_updated'] = datetime.now(self.israel_tz).isoformat()
            
            # Calculate P&L
            if position['action'] == 'BUY':
                pnl = (current_price - position['entry_price']) * position['quantity']
            else:  # SELL
                pnl = (position['entry_price'] - current_price) * position['quantity']
            
            position['unrealized_pnl'] = pnl
            position['pnl_percentage'] = (pnl / (position['entry_price'] * position['quantity'])) * 100
            
            # Update action if provided
            if action:
                position['action'] = action
                if action == 'CLOSE':
                    position['status'] = 'CLOSED'
                    position['closed_at'] = datetime.now(self.israel_tz).isoformat()
                    position['realized_pnl'] = pnl
            
            # Update quantity if provided
            if quantity is not None:
                position['quantity'] = quantity
            
            # Update notes
            if notes:
                position['notes'] = notes
            
            # Store updated position
            await self.redis_client.store_position(position)
            
            # Log activity
            await self.redis_client.log_activity(
                "position_updated",
                f"Position updated: {position_id} - P&L: {pnl:.2f} ({position['pnl_percentage']:.2f}%)",
                position['username'],
                {
                    "position_id": position_id,
                    "current_price": current_price,
                    "unrealized_pnl": pnl,
                    "pnl_percentage": position['pnl_percentage']
                }
            )
            
            logger.info(f"Position updated: {position_id} - P&L: {pnl:.2f}")
            return position
            
        except Exception as e:
            logger.error(f"Error updating position: {e}")
            raise
    
    async def get_user_positions(self, username: str, status: str = None) -> List[Dict[str, Any]]:
        """Get all positions for a user"""
        try:
            positions = await self.redis_client.get_user_positions(username)
            
            if status:
                positions = [p for p in positions if p.get('status') == status]
            
            return positions
            
        except Exception as e:
            logger.error(f"Error getting user positions: {e}")
            return []
    
    async def get_position_performance(self, username: str) -> Dict[str, Any]:
        """Get performance summary for a user"""
        try:
            positions = await self.redis_client.get_user_positions(username)
            
            if not positions:
                return {
                    "total_positions": 0,
                    "open_positions": 0,
                    "closed_positions": 0,
                    "total_pnl": 0.0,
                    "win_rate": 0.0,
                    "avg_pnl": 0.0
                }
            
            open_positions = [p for p in positions if p.get('status') == 'OPEN']
            closed_positions = [p for p in positions if p.get('status') == 'CLOSED']
            
            # Calculate total P&L
            total_pnl = sum(p.get('unrealized_pnl', 0) for p in open_positions)
            total_pnl += sum(p.get('realized_pnl', 0) for p in closed_positions)
            
            # Calculate win rate
            winning_positions = [p for p in closed_positions if p.get('realized_pnl', 0) > 0]
            win_rate = (len(winning_positions) / len(closed_positions)) * 100 if closed_positions else 0
            
            # Calculate average P&L
            avg_pnl = total_pnl / len(positions) if positions else 0
            
            return {
                "total_positions": len(positions),
                "open_positions": len(open_positions),
                "closed_positions": len(closed_positions),
                "total_pnl": total_pnl,
                "win_rate": win_rate,
                "avg_pnl": avg_pnl,
                "winning_positions": len(winning_positions),
                "losing_positions": len(closed_positions) - len(winning_positions)
            }
            
        except Exception as e:
            logger.error(f"Error getting position performance: {e}")
            return {
                "total_positions": 0,
                "open_positions": 0,
                "closed_positions": 0,
                "total_pnl": 0.0,
                "win_rate": 0.0,
                "avg_pnl": 0.0,
                "error": str(e)
            }
