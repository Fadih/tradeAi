# Trading AI Tips - API Documentation

## Overview
This document provides comprehensive documentation for all API endpoints in the Trading AI Tips system. The API is built with FastAPI and provides endpoints for authentication, user management, signal generation, monitoring, and system administration.

## Base URL
```
http://localhost:8000
```

## Authentication
Most endpoints require authentication using Bearer tokens. Include the token in the Authorization header:
```
Authorization: Bearer <your_token>
```

---

## 🔐 Authentication Endpoints

### 1. Login
**POST** `/api/auth/login`

Authenticate a user and receive an access token.

**Request Body:**
```json
{
  "username": "string",
  "password": "string"
}
```

**Response (Success - 200):**
```json
{
  "access_token": "string",
  "token_type": "bearer",
  "username": "string",
  "role": "string"
}
```

**Response (Error - 401):**
```json
{
  "detail": "Invalid credentials"
}
```

**Response (Error - 403):**
```json
{
  "detail": "Account activation period has expired. Please contact administrator to extend your access."
}
```

**Response (Error - 403):**
```json
{
  "detail": "Account has been deactivated. Please contact administrator to reactivate your access."
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin123"}'
```

### 2. Logout
**POST** `/api/auth/logout`

Logout the current user.

**Headers:**
```
Authorization: Bearer <token>
```

**Response (200):**
```json
{
  "message": "Logged out successfully"
}
```

---

## 📊 Dashboard & Status Endpoints

### 3. Get System Status
**GET** `/api/status`

Get current system status and statistics.

**Headers:**
```
Authorization: Bearer <token> (optional - provides user-specific data)
```

**Response (200):**
```json
{
  "status": "running",
  "last_update": "2025-09-06T20:00:00+03:00",
  "active_symbols": ["BTC/USDT", "ETH/USDT", "SPY"],
  "total_signals": 15,
  "signals_today": 3,
  "uptime": "2:30:45"
}
```

### 4. Health Check
**GET** `/api/health`

Get system health information.

**Response (200):**
```json
{
  "status": "healthy",
  "timestamp": "2025-09-06T20:00:00+03:00",
  "version": "2.0",
  "uptime": "2:30:45",
  "services": {
    "redis": "connected",
    "monitoring": "active"
  }
}
```

### 5. User Dashboard
**GET** `/api/user/dashboard`

Get user-specific dashboard data.

**Headers:**
```
Authorization: Bearer <token>
```

**Response (200):**
```json
{
  "username": "user1",
  "role": "user",
  "total_trading_tips": 5,
  "today_trading_tips": 2
}
```

### 6. Admin Dashboard
**GET** `/api/admin/dashboard`

Get admin dashboard statistics and data.

**Headers:**
```
Authorization: Bearer <admin_token>
```

**Response (200):**
```json
{
  "total_users": 15,
  "active_users": 12,
  "inactive_users": 2,
  "expired_users": 1,
  "total_trading_tips": 150,
  "today_trading_tips": 8,
  "system_info": {
    "uptime": "2:30:45",
    "memory_usage": "45%",
    "cpu_usage": "12%"
  },
  "redis_status": "connected",
  "last_updated": "2025-09-06T20:00:00+03:00"
}
```

---

## 📈 Signal Management Endpoints

### 7. Generate Signal
**POST** `/api/signals/generate`

Generate a new trading signal for a symbol.

**Headers:**
```
Authorization: Bearer <token>
```

**Request Body:**
```json
{
  "symbol": "string",
  "timeframe": "string" (optional, default: "1h"),
  "buy_threshold": 0.5 (optional, default: 0.7),
  "sell_threshold": -0.5 (optional, default: -0.7),
  "technical_weight": 0.6 (optional, default: 0.6),
  "sentiment_weight": 0.4 (optional, default: 0.4)
}
```

**Response (200):**
```json
{
  "symbol": "BTC/USDT",
  "timeframe": "1h",
  "timestamp": "2025-09-06T20:00:00+03:00",
  "signal_type": "BUY",
  "confidence": 0.85,
  "technical_score": 0.8,
  "sentiment_score": 0.9,
  "fused_score": 0.85,
  "stop_loss": 45000.0,
  "take_profit": 50000.0,
  "reasoning": "Technical: 0.80, Sentiment: 0.90, Fused: 0.85",
  "applied_buy_threshold": 0.5,
  "applied_sell_threshold": -0.5,
  "applied_tech_weight": 0.6,
  "applied_sentiment_weight": 0.4
}
```

**Response (Error - 400):**
```json
{
  "detail": "Could not fetch market data"
}
```

### 8. Get User Signals
**GET** `/api/signals`

Get trading signals for the authenticated user.

**Headers:**
```
Authorization: Bearer <token>
```

**Query Parameters:**
- `symbol` (optional): Filter by symbol
- `limit` (optional): Number of signals to return (default: 50)

**Response (200):**
```json
[
  {
    "symbol": "BTC/USDT",
    "timeframe": "1h",
    "timestamp": "2025-09-06T20:00:00+03:00",
    "signal_type": "BUY",
    "confidence": 0.85,
    "technical_score": 0.8,
    "sentiment_score": 0.9,
    "fused_score": 0.85,
    "stop_loss": 45000.0,
    "take_profit": 50000.0,
    "reasoning": "Technical: 0.80, Sentiment: 0.90, Fused: 0.85",
    "applied_buy_threshold": 0.5,
    "applied_sell_threshold": -0.5,
    "applied_tech_weight": 0.6,
    "applied_sentiment_weight": 0.4
  }
]
```

### 9. Delete Signal
**DELETE** `/api/signals/{timestamp}`

Delete a specific signal by timestamp.

**Headers:**
```
Authorization: Bearer <token>
```

**Path Parameters:**
- `timestamp`: Signal timestamp (ISO format)

**Response (200):**
```json
{
  "message": "Signal deleted successfully"
}
```

### 10. Get Signal History
**GET** `/api/signals/{timestamp}/history`

Get history events for a specific signal.

**Headers:**
```
Authorization: Bearer <token>
```

**Path Parameters:**
- `timestamp`: Signal timestamp (ISO format)

**Response (200):**
```json
{
  "signal_timestamp": "2025-09-06T20:00:00+03:00",
  "symbol": "BTC/USDT",
  "current_status": "HOLD",
  "history": [
    {
      "timestamp": "2025-09-06T20:00:00+03:00",
      "event_type": "signal_created",
      "description": "Signal created: HOLD for BTC/USDT",
      "metadata": {
        "signal_type": "HOLD",
        "confidence": 0.106,
        "fused_score": -0.106,
        "technical_score": -0.144,
        "sentiment_score": -0.05,
        "applied_thresholds": {
          "buy_threshold": 0.5,
          "sell_threshold": -0.5,
          "tech_weight": 0.6,
          "sentiment_weight": 0.4
        },
        "created_by": "user1"
      }
    },
    {
      "timestamp": "2025-09-06T20:02:00+03:00",
      "event_type": "monitoring_cycle",
      "description": "Monitoring cycle completed - Status remains HOLD",
      "metadata": {
        "current_status": "HOLD",
        "fused_score": -0.106,
        "technical_score": -0.144,
        "sentiment_score": -0.05,
        "buy_threshold": 0.5,
        "sell_threshold": -0.5,
        "reason": "Regular monitoring cycle"
      }
    }
  ],
  "total_events": 2
}
```

### 11. Get Signal Statistics
**GET** `/api/signals/stats`

Get signal statistics from Redis.

**Response (200):**
```json
{
  "total_signals": 150,
  "signals_today": 8,
  "signals_by_type": {
    "BUY": 45,
    "SELL": 30,
    "HOLD": 75
  },
  "signals_by_symbol": {
    "BTC/USDT": 50,
    "ETH/USDT": 40,
    "SPY": 30,
    "AAPL": 30
  },
  "average_confidence": 0.75,
  "last_updated": "2025-09-06T20:00:00+03:00"
}
```

---

## 🔄 Monitoring Endpoints

### 12. Trigger Monitoring
**POST** `/api/signals/monitor`

Manually trigger signal monitoring (admin only).

**Headers:**
```
Authorization: Bearer <admin_token>
```

**Response (200):**
```json
{
  "status": "success",
  "message": "Signal monitoring completed",
  "timestamp": "2025-09-06T20:00:00+03:00"
}
```

### 13. Get Monitoring Status
**GET** `/api/signals/monitor/status`

Get monitoring system status (admin only).

**Headers:**
```
Authorization: Bearer <admin_token>
```

**Response (200):**
```json
{
  "monitoring_enabled": true,
  "last_run": "2025-09-06T20:00:00+03:00",
  "next_run": "2025-09-06T20:02:00+03:00",
  "interval_minutes": 2,
  "active_signals": 15,
  "last_results": {
    "monitored": 15,
    "updated": 2,
    "errors": 0
  }
}
```

---

## 📊 Market Data Endpoints

### 14. Get Market Data
**GET** `/api/market-data/{symbol}`

Get market data for a specific symbol.

**Path Parameters:**
- `symbol`: Trading symbol (e.g., BTC/USDT, AAPL)

**Query Parameters:**
- `timeframe` (optional): Timeframe (default: "1h")
- `limit` (optional): Number of data points (default: 100)

**Response (200):**
```json
{
  "symbol": "BTC/USDT",
  "timeframe": "1h",
  "data": [
    {
      "timestamp": "2025-09-06T19:00:00+03:00",
      "open": 45000.0,
      "high": 46000.0,
      "low": 44000.0,
      "close": 45500.0,
      "volume": 1000.0
    }
  ],
  "indicators": {
    "rsi": 45.2,
    "macd": 0.5,
    "atr": 500.0
  }
}
```

### 15. Get Market Overview
**GET** `/api/market-data/all/overview`

Get market overview for all configured symbols.

**Query Parameters:**
- `timeframe` (optional): Timeframe (default: "1h")

**Response (200):**
```json
{
  "overview": [
    {
      "symbol": "BTC/USDT",
      "current_price": 45500.0,
      "change_24h": 2.5,
      "volume_24h": 1000000.0,
      "rsi": 45.2,
      "macd": 0.5,
      "atr": 500.0
    }
  ],
  "last_updated": "2025-09-06T20:00:00+03:00"
}
```

---

## 💼 Position Management Endpoints

### 16. Create Position
**POST** `/api/positions`

Create a new trading position based on a signal.

**Headers:**
```
Authorization: Bearer <token>
```

**Request Body:**
```json
{
  "signal_timestamp": "2025-09-06T20:00:00+03:00",
  "position_size": 1000.0,
  "entry_price": 45500.0,
  "notes": "string" (optional)
}
```

**Response (200):**
```json
{
  "position_id": "pos_123456",
  "signal_timestamp": "2025-09-06T20:00:00+03:00",
  "symbol": "BTC/USDT",
  "position_size": 1000.0,
  "entry_price": 45500.0,
  "current_price": 45500.0,
  "status": "open",
  "pnl": 0.0,
  "pnl_percentage": 0.0,
  "created_at": "2025-09-06T20:00:00+03:00",
  "notes": "string"
}
```

### 17. Get User Positions
**GET** `/api/positions`

Get all positions for the current user.

**Headers:**
```
Authorization: Bearer <token>
```

**Query Parameters:**
- `status` (optional): Filter by status (open, closed)

**Response (200):**
```json
[
  {
    "position_id": "pos_123456",
    "signal_timestamp": "2025-09-06T20:00:00+03:00",
    "symbol": "BTC/USDT",
    "position_size": 1000.0,
    "entry_price": 45500.0,
    "current_price": 46000.0,
    "status": "open",
    "pnl": 500.0,
    "pnl_percentage": 1.1,
    "created_at": "2025-09-06T20:00:00+03:00",
    "notes": "string"
  }
]
```

### 18. Update Position
**PUT** `/api/positions/{position_id}`

Update an existing position.

**Headers:**
```
Authorization: Bearer <token>
```

**Path Parameters:**
- `position_id`: Position ID

**Request Body:**
```json
{
  "status": "closed",
  "exit_price": 46000.0,
  "notes": "Closed at target"
}
```

**Response (200):**
```json
{
  "message": "Position updated successfully",
  "position_id": "pos_123456"
}
```

### 19. Get Position Performance
**GET** `/api/positions/performance`

Get performance summary for the current user.

**Headers:**
```
Authorization: Bearer <token>
```

**Response (200):**
```json
{
  "total_positions": 10,
  "open_positions": 3,
  "closed_positions": 7,
  "total_pnl": 1500.0,
  "total_pnl_percentage": 15.0,
  "win_rate": 70.0,
  "average_win": 300.0,
  "average_loss": -100.0,
  "best_trade": 800.0,
  "worst_trade": -200.0
}
```

---

## 👥 User Management Endpoints (Admin Only)

### 20. Get All Users
**GET** `/api/admin/users`

Get list of all users in the system.

**Headers:**
```
Authorization: Bearer <admin_token>
```

**Response (200):**
```json
[
  {
    "username": "user1",
    "role": "user",
    "status": "active",
    "created_at": "2025-09-01T10:00:00+03:00",
    "last_login": "2025-09-06T11:30:00+03:00",
    "first_name": "John",
    "last_name": "Doe",
    "email": "john@example.com",
    "phone": "1234567890",
    "additional_info": "Regular user",
    "activation_expires_at": "2025-10-01T10:00:00+03:00",
    "activation_days": 30
  }
]
```

### 21. Create User
**POST** `/api/admin/users`

Create a new user account.

**Headers:**
```
Authorization: Bearer <admin_token>
```

**Request Body:**
```json
{
  "username": "string",
  "password": "string",
  "role": "user" | "admin",
  "first_name": "string" (optional),
  "last_name": "string" (optional),
  "email": "string" (optional),
  "phone": "string" (optional),
  "additional_info": "string" (optional),
  "activation_days": 30 (optional, default: 30)
}
```

**Response (201):**
```json
{
  "message": "User created successfully",
  "username": "newuser"
}
```

### 22. Edit User
**PUT** `/api/admin/users/{username}`

Update an existing user's information.

**Headers:**
```
Authorization: Bearer <admin_token>
```

**Path Parameters:**
- `username`: Username of the user to update

**Request Body:**
```json
{
  "username": "string",
  "password": "string" (optional),
  "role": "user" | "admin",
  "first_name": "string" (optional),
  "last_name": "string" (optional),
  "email": "string" (optional),
  "phone": "string" (optional),
  "additional_info": "string" (optional),
  "activation_days": 30 (optional, for extending activation)
}
```

**Response (200):**
```json
{
  "message": "User updated successfully",
  "username": "updateduser"
}
```

### 23. Delete User
**DELETE** `/api/admin/users/{username}`

Delete a user account.

**Headers:**
```
Authorization: Bearer <admin_token>
```

**Path Parameters:**
- `username`: Username of the user to delete

**Query Parameters:**
- `delete_signals` (optional): Whether to delete user's signals (default: false)

**Response (200):**
```json
{
  "message": "User deleted successfully",
  "signals_deleted": 5
}
```

### 24. Extend User Activation
**POST** `/api/admin/users/{username}/extend-activation`

Extend or modify a user's activation period.

**Headers:**
```
Authorization: Bearer <admin_token>
```

**Path Parameters:**
- `username`: Username of the user

**Request Body:**
```json
{
  "username": "string",
  "additional_days": 30 (0 to deactivate, positive to extend),
  "reason": "string" (optional)
}
```

**Response (200) - Extension:**
```json
{
  "message": "User john activation extended by 30 days",
  "new_expiry": "2025-10-05T12:00:00+03:00",
  "additional_days": 30
}
```

**Response (200) - Deactivation:**
```json
{
  "message": "User john has been deactivated",
  "new_expiry": null,
  "additional_days": 0
}
```

### 25. Get User Signals (Admin)
**GET** `/api/admin/users/{username}/signals`

Get all signals for a specific user with configurations (admin only).

**Headers:**
```
Authorization: Bearer <admin_token>
```

**Path Parameters:**
- `username`: Username of the user

**Response (200):**
```json
[
  {
    "symbol": "BTC/USDT",
    "timeframe": "1h",
    "timestamp": "2025-09-06T20:00:00+03:00",
    "signal_type": "BUY",
    "confidence": 0.85,
    "technical_score": 0.8,
    "sentiment_score": 0.9,
    "fused_score": 0.85,
    "stop_loss": 45000.0,
    "take_profit": 50000.0,
    "reasoning": "Technical: 0.80, Sentiment: 0.90, Fused: 0.85",
    "applied_buy_threshold": 0.5,
    "applied_sell_threshold": -0.5,
    "applied_tech_weight": 0.6,
    "applied_sentiment_weight": 0.4,
    "username": "user1"
  }
]
```

### 26. Clear User Signals
**DELETE** `/api/admin/users/{username}/signals`

Clear all trading signals for a specific user.

**Headers:**
```
Authorization: Bearer <admin_token>
```

**Path Parameters:**
- `username`: Username of the user

**Response (200):**
```json
{
  "message": "Signals cleared successfully",
  "signals_deleted": 10
}
```

### 27. Get All Signals (Admin)
**GET** `/api/admin/signals/all`

Get all trading signals with detailed information (admin only).

**Headers:**
```
Authorization: Bearer <admin_token>
```

**Query Parameters:**
- `limit` (optional): Number of signals to return (default: 100)
- `offset` (optional): Number of signals to skip (default: 0)

**Response (200):**
```json
{
  "signals": [
    {
      "symbol": "BTC/USDT",
      "timeframe": "1h",
      "timestamp": "2025-09-06T20:00:00+03:00",
      "signal_type": "BUY",
      "confidence": 0.85,
      "technical_score": 0.8,
      "sentiment_score": 0.9,
      "fused_score": 0.85,
      "stop_loss": 45000.0,
      "take_profit": 50000.0,
      "reasoning": "Technical: 0.80, Sentiment: 0.90, Fused: 0.85",
      "applied_buy_threshold": 0.5,
      "applied_sell_threshold": -0.5,
      "applied_tech_weight": 0.6,
      "applied_sentiment_weight": 0.4,
      "username": "user1"
    }
  ],
  "total": 150,
  "limit": 100,
  "offset": 0
}
```

### 28. Get Admin Activities
**GET** `/api/admin/activities`

Get recent activities (admin only).

**Headers:**
```
Authorization: Bearer <admin_token>
```

**Query Parameters:**
- `limit` (optional): Number of activities to return (default: 20)

**Response (200):**
```json
{
  "activities": [
    {
      "timestamp": "2025-09-06T20:00:00+03:00",
      "action": "user_created",
      "description": "Created new user 'newuser' with role 'user'",
      "user": "admin",
      "metadata": {
        "username": "newuser",
        "role": "user"
      }
    }
  ]
}
```

---

## 👤 User Profile Endpoints

### 29. Get User Profile
**GET** `/api/user/profile`

Get current user's profile information.

**Headers:**
```
Authorization: Bearer <token>
```

**Response (200):**
```json
{
  "username": "user1",
  "role": "user",
  "status": "active",
  "created_at": "2025-09-01T10:00:00+03:00",
  "last_login": "2025-09-06T11:30:00+03:00",
  "first_name": "John",
  "last_name": "Doe",
  "email": "john@example.com",
  "phone": "1234567890",
  "additional_info": "Regular user",
  "activation_expires_at": "2025-10-01T10:00:00+03:00",
  "activation_days": 30
}
```

### 30. Update User Profile
**PUT** `/api/user/profile`

Update current user's profile information (excluding activation period).

**Headers:**
```
Authorization: Bearer <token>
```

**Request Body:**
```json
{
  "first_name": "string" (optional),
  "last_name": "string" (optional),
  "email": "string" (optional),
  "phone": "string" (optional),
  "additional_info": "string" (optional)
}
```

**Response (200):**
```json
{
  "message": "Profile updated successfully"
}
```

### 31. Change Password
**PUT** `/api/user/change-password`

Change current user's password.

**Headers:**
```
Authorization: Bearer <token>
```

**Request Body:**
```json
{
  "current_password": "string",
  "new_password": "string"
}
```

**Response (200):**
```json
{
  "message": "Password changed successfully"
}
```

---

## ⚙️ Settings Endpoints (Admin Only)

### 32. Get System Settings
**GET** `/api/admin/settings`

Get all system configuration settings.

**Headers:**
```
Authorization: Bearer <admin_token>
```

**Response (200):**
```json
{
  "config": {
    "active_markets": ["BTC/USDT", "ETH/USDT", "SPY"],
    "signal_generation_enabled": true,
    "generation_frequency_minutes": 30,
    "buy_threshold": 0.7,
    "sell_threshold": -0.7,
    "technical_weight": 0.6,
    "sentiment_weight": 0.4,
    "session_timeout_hours": 24,
    "auto_refresh_enabled": true,
    "refresh_interval_seconds": 30
  },
  "categories": {
    "trading": ["active_markets", "signal_generation_enabled"],
    "technical": ["buy_threshold", "sell_threshold"],
    "security": ["session_timeout_hours"],
    "ui": ["auto_refresh_enabled", "refresh_interval_seconds"]
  }
}
```

### 33. Update Single Setting
**PUT** `/api/admin/settings/{key}`

Update a specific system setting.

**Headers:**
```
Authorization: Bearer <admin_token>
```

**Path Parameters:**
- `key`: Setting key to update

**Request Body:**
```json
{
  "key": "string",
  "value": "any",
  "category": "string"
}
```

**Response (200):**
```json
{
  "message": "Setting active_markets updated successfully",
  "value": ["BTC/USDT", "ETH/USDT", "SPY", "AAPL"]
}
```

### 34. Bulk Update Settings
**POST** `/api/admin/settings/bulk`

Update multiple settings at once.

**Headers:**
```
Authorization: Bearer <admin_token>
```

**Request Body:**
```json
[
  {
    "key": "active_markets",
    "value": ["BTC/USDT", "ETH/USDT", "SPY"],
    "category": "trading"
  },
  {
    "key": "buy_threshold",
    "value": 0.8,
    "category": "technical"
  }
]
```

**Response (200):**
```json
{
  "message": "Updated 2 settings",
  "results": [
    {"key": "active_markets", "success": true},
    {"key": "buy_threshold", "success": true}
  ]
}
```

### 35. Reset Settings
**POST** `/api/admin/settings/reset`

Reset all settings to default values.

**Headers:**
```
Authorization: Bearer <admin_token>
```

**Response (200):**
```json
{
  "message": "All settings reset to defaults"
}
```

### 36. Export Settings
**GET** `/api/admin/settings/export`

Export current settings as JSON.

**Headers:**
```
Authorization: Bearer <admin_token>
```

**Response (200):**
```json
{
  "config": {
    "active_markets": ["BTC/USDT", "ETH/USDT"],
    "signal_generation_enabled": true
  },
  "exported_at": "2025-09-06T20:00:00+03:00",
  "version": "2.0"
}
```

### 37. Import Settings
**POST** `/api/admin/settings/import`

Import settings from JSON.

**Headers:**
```
Authorization: Bearer <admin_token>
```

**Request Body:**
```json
{
  "active_markets": ["BTC/USDT", "ETH/USDT", "SPY"],
  "signal_generation_enabled": true,
  "buy_threshold": 0.7,
  "sell_threshold": -0.7
}
```

**Response (200):**
```json
{
  "message": "Settings imported successfully"
}
```

---

## 🔧 Configuration Endpoints

### 38. Get Configuration
**GET** `/api/config`

Get current agent configuration.

**Response (200):**
```json
{
  "active_markets": ["BTC/USDT", "ETH/USDT", "SPY"],
  "signal_generation_enabled": true,
  "generation_frequency_minutes": 30,
  "buy_threshold": 0.7,
  "sell_threshold": -0.7,
  "technical_weight": 0.6,
  "sentiment_weight": 0.4
}
```

### 39. Update Configuration
**POST** `/api/config/update`

Update agent configuration.

**Request Body:**
```json
{
  "key": "string",
  "value": "any"
}
```

**Response (200):**
```json
{
  "message": "Configuration updated successfully"
}
```

---

## 🗄️ Redis Management Endpoints

### 40. Get Redis Status
**GET** `/api/redis/status`

Get Redis connection status and statistics.

**Response (200):**
```json
{
  "status": "connected",
  "version": "7.0.0",
  "uptime": "2:30:45",
  "memory_usage": "45MB",
  "connected_clients": 5,
  "total_commands_processed": 10000,
  "keyspace": {
    "db0": {
      "keys": 150,
      "expires": 50
    }
  }
}
```

### 41. Clear Cache
**GET** `/api/redis/cache/clear`

Clear Redis cache by pattern.

**Query Parameters:**
- `pattern` (optional): Cache pattern to clear (default: "*")

**Response (200):**
```json
{
  "message": "Cache cleared successfully",
  "pattern": "*",
  "keys_deleted": 50
}
```

### 42. Get Redis Metrics
**GET** `/api/redis/metrics/{metric_name}`

Get performance metrics from Redis.

**Path Parameters:**
- `metric_name`: Metric name (e.g., "memory", "commands", "clients")

**Query Parameters:**
- `days` (optional): Number of days to retrieve (default: 7)

**Response (200):**
```json
{
  "metric": "memory",
  "data": [
    {
      "timestamp": "2025-09-06T20:00:00+03:00",
      "value": 45.2
    }
  ],
  "period_days": 7
}
```

---

## 📄 Static Pages

### 43. Root Page
**GET** `/`

Redirect to login page.

**Response (302):**
Redirect to `/login`

### 44. Login Page
**GET** `/login`

Serve the login page.

**Response (200):**
HTML login page

### 45. Dashboard Page
**GET** `/dashboard`

Serve the main dashboard page (requires authentication).

**Headers:**
```
Authorization: Bearer <token>
```

**Response (200):**
HTML dashboard page

### 46. Admin Page
**GET** `/admin`

Serve the admin dashboard page (requires admin authentication).

**Headers:**
```
Authorization: Bearer <admin_token>
```

**Response (200):**
HTML admin page

---

## 🔧 Configuration Settings Reference

### Trading Settings
- `active_markets`: List of active trading markets
- `signal_generation_enabled`: Enable/disable signal generation
- `generation_frequency_minutes`: How often to generate signals
- `default_timeframe`: Default timeframe for analysis
- `max_position_size`: Maximum position size in dollars
- `risk_per_trade_percent`: Risk percentage per trade

### Technical Analysis Settings
- `buy_threshold`: Minimum score for BUY signals (default: 0.7)
- `sell_threshold`: Maximum score for SELL signals (default: -0.7)
- `technical_weight`: Weight for technical indicators (0-1, default: 0.6)
- `sentiment_weight`: Weight for sentiment analysis (0-1, default: 0.4)

### Data Sources Settings
- `data_provider`: Data provider (alpaca, ccxt, yahoo)
- `data_refresh_rate_minutes`: Data refresh frequency
- `historical_data_days`: Days of historical data to keep
- `news_weight`: How much news affects signals (0-1)
- `news_keywords`: Keywords for news filtering

### Security Settings
- `session_timeout_hours`: Session timeout duration (default: 24)
- `max_login_attempts`: Maximum failed login attempts
- `password_min_length`: Minimum password length
- `api_rate_limit_per_minute`: API rate limit

### UI Settings
- `auto_refresh_enabled`: Enable auto-refresh (default: true)
- `refresh_interval_seconds`: Auto-refresh interval (default: 30)
- `theme`: UI theme (light, dark, auto)
- `date_format`: Date display format
- `timezone`: System timezone
- `currency`: Default currency

### Notifications Settings
- `email_notifications_enabled`: Enable email notifications
- `smtp_server`: SMTP server address
- `smtp_port`: SMTP server port
- `smtp_username`: SMTP username
- `smtp_password`: SMTP password
- `alert_recipients`: List of alert recipients

### System Settings
- `log_level`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `cache_ttl_hours`: Cache time-to-live
- `backup_enabled`: Enable automatic backups
- `backup_frequency_hours`: Backup frequency

---

## 📝 Error Codes

### HTTP Status Codes
- `200`: Success
- `201`: Created
- `400`: Bad Request
- `401`: Unauthorized
- `403`: Forbidden
- `404`: Not Found
- `500`: Internal Server Error

### Common Error Messages
- `"Invalid credentials"`: Wrong username/password
- `"Admin access required"`: Endpoint requires admin privileges
- `"User not found"`: Specified user doesn't exist
- `"Username already exists"`: Username is taken
- `"Account activation period has expired"`: User's activation period expired
- `"Account has been deactivated"`: User was manually deactivated
- `"Cannot delete the last admin user"`: System protection against deleting all admins
- `"Could not fetch market data"`: Market data unavailable
- `"Invalid symbol or timeframe"`: Invalid trading symbol or timeframe

---

## 🔐 Authentication Flow

1. **Login**: POST to `/api/auth/login` with username/password
2. **Get Token**: Receive access token in response
3. **Use Token**: Include token in Authorization header for protected endpoints
4. **Token Expiry**: Tokens expire based on session timeout settings

---

## 📊 Rate Limiting

- API calls are rate-limited based on `api_rate_limit_per_minute` setting
- Default: 100 requests per minute per user
- Rate limit headers included in responses

---

## 🚀 Quick Start Examples

### 1. Login and Get Token
```bash
TOKEN=$(curl -s -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin123"}' | \
  jq -r '.access_token')
```

### 2. Get System Status
```bash
curl -X GET http://localhost:8000/api/status \
  -H "Authorization: Bearer $TOKEN"
```

### 3. Create a User
```bash
curl -X POST http://localhost:8000/api/admin/users \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "username": "newuser",
    "password": "password123",
    "role": "user",
    "activation_days": 30
  }'
```

### 4. Generate a Signal
```bash
curl -X POST http://localhost:8000/api/signals/generate \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "symbol": "BTC/USDT",
    "timeframe": "1h",
    "buy_threshold": 0.5,
    "sell_threshold": -0.5,
    "technical_weight": 0.6,
    "sentiment_weight": 0.4
  }'
```

### 5. Get Signal History
```bash
curl -X GET "http://localhost:8000/api/signals/2025-09-06T20:00:00+03:00/history" \
  -H "Authorization: Bearer $TOKEN"
```

### 6. Trigger Monitoring
```bash
curl -X POST http://localhost:8000/api/signals/monitor \
  -H "Authorization: Bearer $TOKEN"
```

### 7. Create a Position
```bash
curl -X POST http://localhost:8000/api/positions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "signal_timestamp": "2025-09-06T20:00:00+03:00",
    "position_size": 1000.0,
    "entry_price": 45500.0,
    "notes": "BTC long position"
  }'
```

---

## 📞 Support

For API support and questions, please refer to the system logs or contact the system administrator.

**Last Updated**: September 6, 2025
**API Version**: 2.0
**System Version**: Trading AI Tips v2.0

---

## 🔄 Recent Updates

### Version 2.0 (September 6, 2025)
- ✅ **Enhanced Signal Generation**: Added custom threshold parameters
- ✅ **Monitoring System**: Added automatic signal monitoring with history tracking
- ✅ **Position Management**: Added position tracking and performance analytics
- ✅ **Signal History**: Added comprehensive signal history with monitoring events
- ✅ **User Profile Management**: Added user profile and password management
- ✅ **Redis Management**: Added Redis status and cache management endpoints
- ✅ **Market Data**: Enhanced market data endpoints with technical indicators
- ✅ **Admin Activities**: Added activity logging and monitoring
- ✅ **Bulk Operations**: Added bulk settings update and user management
- ✅ **Enhanced Security**: Improved authentication and authorization