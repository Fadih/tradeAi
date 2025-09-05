# Trading AI Tips - API Documentation

## Overview
This document provides comprehensive documentation for all API endpoints in the Trading AI Tips system. The API is built with FastAPI and provides endpoints for authentication, user management, signal generation, and system administration.

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

---

## 📊 Dashboard Endpoints

### 2. Get Dashboard Status
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
  "last_update": "2025-09-05T12:00:00",
  "active_symbols": 8,
  "total_signals": 15,
  "signals_today": 3,
  "uptime": "2:30:45"
}
```

**Example:**
```bash
curl -X GET http://localhost:8000/api/status \
  -H "Authorization: Bearer <token>"
```

### 3. Get Configuration
**GET** `/api/config`

Get system configuration including active markets.

**Response (200):**
```json
{
  "active_markets": ["BTC/USD", "ETH/USD", "AAPL", "GOOGL"],
  "signal_generation_enabled": true,
  "generation_frequency_minutes": 30
}
```

**Example:**
```bash
curl -X GET http://localhost:8000/api/config
```

### 4. Generate Signal
**POST** `/api/generate-signal`

Generate a trading signal for a specific market.

**Headers:**
```
Authorization: Bearer <token>
```

**Request Body:**
```json
{
  "symbol": "string",
  "timeframe": "string" (optional, default: "1h")
}
```

**Response (200):**
```json
{
  "symbol": "BTC/USD",
  "signal": "BUY",
  "confidence": 0.85,
  "timestamp": "2025-09-05T12:00:00",
  "technical_score": 0.8,
  "sentiment_score": 0.9,
  "reasoning": "Strong technical indicators combined with positive sentiment"
}
```

**Response (Error - 400):**
```json
{
  "detail": "Invalid symbol or timeframe"
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/api/generate-signal \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{"symbol": "BTC/USD", "timeframe": "1h"}'
```

### 5. Get User Signals
**GET** `/api/signals`

Get trading signals for the authenticated user.

**Headers:**
```
Authorization: Bearer <token>
```

**Query Parameters:**
- `limit` (optional): Number of signals to return (default: 10)
- `symbol` (optional): Filter by symbol

**Response (200):**
```json
[
  {
    "symbol": "BTC/USD",
    "signal": "BUY",
    "confidence": 0.85,
    "timestamp": "2025-09-05T12:00:00",
    "technical_score": 0.8,
    "sentiment_score": 0.9,
    "reasoning": "Strong technical indicators"
  }
]
```

**Example:**
```bash
curl -X GET "http://localhost:8000/api/signals?limit=5&symbol=BTC/USD" \
  -H "Authorization: Bearer <token>"
```

---

## 👥 User Management Endpoints (Admin Only)

### 6. Get All Users
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
    "created_at": "2025-09-01T10:00:00",
    "last_login": "2025-09-05T11:30:00",
    "first_name": "John",
    "last_name": "Doe",
    "email": "john@example.com",
    "phone": "1234567890",
    "additional_info": "Regular user",
    "activation_expires_at": "2025-10-01T10:00:00",
    "activation_days": 30
  }
]
```

**Response (Error - 403):**
```json
{
  "detail": "Admin access required"
}
```

**Example:**
```bash
curl -X GET http://localhost:8000/api/admin/users \
  -H "Authorization: Bearer <admin_token>"
```

### 7. Create User
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

**Response (Error - 400):**
```json
{
  "detail": "Username already exists"
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/api/admin/users \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <admin_token>" \
  -d '{
    "username": "newuser",
    "password": "password123",
    "role": "user",
    "first_name": "Jane",
    "last_name": "Smith",
    "email": "jane@example.com",
    "activation_days": 30
  }'
```

### 8. Edit User
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

**Response (Error - 404):**
```json
{
  "detail": "User not found"
}
```

**Example:**
```bash
curl -X PUT http://localhost:8000/api/admin/users/john \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <admin_token>" \
  -d '{
    "username": "john",
    "first_name": "John",
    "last_name": "Updated",
    "email": "john.updated@example.com",
    "activation_days": 15
  }'
```

### 9. Delete User
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

**Response (Error - 400):**
```json
{
  "detail": "Cannot delete the last admin user"
}
```

**Example:**
```bash
curl -X DELETE "http://localhost:8000/api/admin/users/john?delete_signals=true" \
  -H "Authorization: Bearer <admin_token>"
```

### 10. Extend User Activation
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
  "new_expiry": "2025-10-05T12:00:00",
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

**Response (Error - 400):**
```json
{
  "detail": "Admin users do not have activation periods"
}
```

**Example - Extend:**
```bash
curl -X POST http://localhost:8000/api/admin/users/john/extend-activation \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <admin_token>" \
  -d '{
    "username": "john",
    "additional_days": 30,
    "reason": "Monthly extension"
  }'
```

**Example - Deactivate:**
```bash
curl -X POST http://localhost:8000/api/admin/users/john/extend-activation \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <admin_token>" \
  -d '{
    "username": "john",
    "additional_days": 0,
    "reason": "Account suspension"
  }'
```

### 11. Clear User Signals
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

**Example:**
```bash
curl -X DELETE http://localhost:8000/api/admin/users/john/signals \
  -H "Authorization: Bearer <admin_token>"
```

### 12. Get Admin Dashboard
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
  "total_signals": 150,
  "signals_today": 8,
  "recent_activity": [
    {
      "action": "user_created",
      "description": "User 'newuser' created",
      "timestamp": "2025-09-05T12:00:00",
      "user": "admin"
    }
  ]
}
```

**Example:**
```bash
curl -X GET http://localhost:8000/api/admin/dashboard \
  -H "Authorization: Bearer <admin_token>"
```

### 13. Get All Signals (Admin)
**GET** `/api/admin/signals`

Get all trading signals across all users.

**Headers:**
```
Authorization: Bearer <admin_token>
```

**Query Parameters:**
- `limit` (optional): Number of signals to return (default: 50)
- `user` (optional): Filter by username
- `symbol` (optional): Filter by symbol
- `page` (optional): Page number for pagination (default: 1)
- `page_size` (optional): Items per page (default: 25)

**Response (200):**
```json
{
  "signals": [
    {
      "symbol": "BTC/USD",
      "signal": "BUY",
      "confidence": 0.85,
      "timestamp": "2025-09-05T12:00:00",
      "user": "john",
      "technical_score": 0.8,
      "sentiment_score": 0.9,
      "reasoning": "Strong technical indicators"
    }
  ],
  "total": 150,
  "page": 1,
  "page_size": 25,
  "total_pages": 6
}
```

**Example:**
```bash
curl -X GET "http://localhost:8000/api/admin/signals?limit=10&user=john&symbol=BTC/USD" \
  -H "Authorization: Bearer <admin_token>"
```

---

## ⚙️ Settings Endpoints (Admin Only)

### 14. Get System Settings
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
    "active_markets": ["BTC/USD", "ETH/USD", "AAPL"],
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

**Example:**
```bash
curl -X GET http://localhost:8000/api/admin/settings \
  -H "Authorization: Bearer <admin_token>"
```

### 15. Update Single Setting
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
  "value": ["BTC/USD", "ETH/USD", "AAPL", "GOOGL"]
}
```

**Example:**
```bash
curl -X PUT http://localhost:8000/api/admin/settings/active_markets \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <admin_token>" \
  -d '{
    "key": "active_markets",
    "value": ["BTC/USD", "ETH/USD", "AAPL", "GOOGL"],
    "category": "trading"
  }'
```

### 16. Bulk Update Settings
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
    "value": ["BTC/USD", "ETH/USD", "AAPL"],
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

**Example:**
```bash
curl -X POST http://localhost:8000/api/admin/settings/bulk \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <admin_token>" \
  -d '[
    {
      "key": "active_markets",
      "value": ["BTC/USD", "ETH/USD", "AAPL"],
      "category": "trading"
    },
    {
      "key": "buy_threshold",
      "value": 0.8,
      "category": "technical"
    }
  ]'
```

### 17. Reset Settings
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

**Example:**
```bash
curl -X POST http://localhost:8000/api/admin/settings/reset \
  -H "Authorization: Bearer <admin_token>"
```

### 18. Export Settings
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
    "active_markets": ["BTC/USD", "ETH/USD"],
    "signal_generation_enabled": true
  },
  "exported_at": "2025-09-05T12:00:00",
  "version": "1.0"
}
```

**Example:**
```bash
curl -X GET http://localhost:8000/api/admin/settings/export \
  -H "Authorization: Bearer <admin_token>"
```

### 19. Import Settings
**POST** `/api/admin/settings/import`

Import settings from JSON.

**Headers:**
```
Authorization: Bearer <admin_token>
```

**Request Body:**
```json
{
  "active_markets": ["BTC/USD", "ETH/USD", "AAPL"],
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

**Example:**
```bash
curl -X POST http://localhost:8000/api/admin/settings/import \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <admin_token>" \
  -d '{
    "active_markets": ["BTC/USD", "ETH/USD", "AAPL"],
    "signal_generation_enabled": true,
    "buy_threshold": 0.7
  }'
```

---

## 📄 Static Pages

### 20. Login Page
**GET** `/login`

Serve the login page.

**Response (200):**
HTML login page

**Example:**
```bash
curl -X GET http://localhost:8000/login
```

### 21. Dashboard Page
**GET** `/dashboard`

Serve the main dashboard page (requires authentication).

**Response (200):**
HTML dashboard page

**Example:**
```bash
curl -X GET http://localhost:8000/dashboard \
  -H "Authorization: Bearer <token>"
```

### 22. Admin Page
**GET** `/admin`

Serve the admin dashboard page (requires admin authentication).

**Response (200):**
HTML admin page

**Example:**
```bash
curl -X GET http://localhost:8000/admin \
  -H "Authorization: Bearer <admin_token>"
```

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
- `buy_threshold`: Minimum score for BUY signals
- `sell_threshold`: Maximum score for SELL signals
- `technical_weight`: Weight for technical indicators (0-1)
- `sentiment_weight`: Weight for sentiment analysis (0-1)

### Data Sources Settings
- `data_provider`: Data provider (alpaca, ccxt, yahoo)
- `data_refresh_rate_minutes`: Data refresh frequency
- `historical_data_days`: Days of historical data to keep
- `news_weight`: How much news affects signals (0-1)
- `news_keywords`: Keywords for news filtering

### Security Settings
- `session_timeout_hours`: Session timeout duration
- `max_login_attempts`: Maximum failed login attempts
- `password_min_length`: Minimum password length
- `api_rate_limit_per_minute`: API rate limit

### UI Settings
- `auto_refresh_enabled`: Enable auto-refresh
- `refresh_interval_seconds`: Auto-refresh interval
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
curl -X POST http://localhost:8000/api/generate-signal \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"symbol": "BTC/USD", "timeframe": "1h"}'
```

---

## 📞 Support

For API support and questions, please refer to the system logs or contact the system administrator.

**Last Updated**: September 5, 2025
**API Version**: 1.0
**System Version**: Trading AI Tips v2.0
