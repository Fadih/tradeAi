# Trading AI Tips - API Quick Reference

## 🔐 Authentication
```bash
# Login
curl -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin123"}'
```

## 📊 Dashboard
```bash
# Get Status
curl -X GET http://localhost:8000/api/status \
  -H "Authorization: Bearer <token>"

# Get Config
curl -X GET http://localhost:8000/api/config

# Generate Signal
curl -X POST http://localhost:8000/api/generate-signal \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{"symbol": "BTC/USD", "timeframe": "1h"}'

# Get User Signals
curl -X GET http://localhost:8000/api/signals \
  -H "Authorization: Bearer <token>"
```

## 👥 User Management (Admin)
```bash
# Get All Users
curl -X GET http://localhost:8000/api/admin/users \
  -H "Authorization: Bearer <admin_token>"

# Create User
curl -X POST http://localhost:8000/api/admin/users \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <admin_token>" \
  -d '{
    "username": "newuser",
    "password": "password123",
    "role": "user",
    "activation_days": 30
  }'

# Edit User
curl -X PUT http://localhost:8000/api/admin/users/username \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <admin_token>" \
  -d '{"username": "username", "first_name": "John", "activation_days": 15}'

# Delete User
curl -X DELETE http://localhost:8000/api/admin/users/username \
  -H "Authorization: Bearer <admin_token>"

# Extend/Deactivate User
curl -X POST http://localhost:8000/api/admin/users/username/extend-activation \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <admin_token>" \
  -d '{"username": "username", "additional_days": 30, "reason": "Extension"}'

# Clear User Signals
curl -X DELETE http://localhost:8000/api/admin/users/username/signals \
  -H "Authorization: Bearer <admin_token>"
```

## ⚙️ Settings (Admin)
```bash
# Get Settings
curl -X GET http://localhost:8000/api/admin/settings \
  -H "Authorization: Bearer <admin_token>"

# Update Single Setting
curl -X PUT http://localhost:8000/api/admin/settings/active_markets \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <admin_token>" \
  -d '{"key": "active_markets", "value": ["BTC/USD", "ETH/USD"], "category": "trading"}'

# Bulk Update Settings
curl -X POST http://localhost:8000/api/admin/settings/bulk \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <admin_token>" \
  -d '[{"key": "active_markets", "value": ["BTC/USD"], "category": "trading"}]'

# Reset Settings
curl -X POST http://localhost:8000/api/admin/settings/reset \
  -H "Authorization: Bearer <admin_token>"

# Export Settings
curl -X GET http://localhost:8000/api/admin/settings/export \
  -H "Authorization: Bearer <admin_token>"

# Import Settings
curl -X POST http://localhost:8000/api/admin/settings/import \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <admin_token>" \
  -d '{"active_markets": ["BTC/USD"], "buy_threshold": 0.7}'
```

## 📄 Pages
```bash
# Login Page
curl -X GET http://localhost:8000/login

# Dashboard Page
curl -X GET http://localhost:8000/dashboard \
  -H "Authorization: Bearer <token>"

# Admin Page
curl -X GET http://localhost:8000/admin \
  -H "Authorization: Bearer <admin_token>"
```

## 🔧 Common Settings Keys
- `active_markets`: Trading markets list
- `signal_generation_enabled`: Enable/disable signals
- `buy_threshold`: BUY signal threshold
- `sell_threshold`: SELL signal threshold
- `session_timeout_hours`: Session timeout
- `auto_refresh_enabled`: Auto-refresh toggle
- `refresh_interval_seconds`: Refresh interval

## 📝 Common Responses
```json
// Success
{"message": "Operation successful"}

// Error
{"detail": "Error message"}

// Login Success
{
  "access_token": "token",
  "token_type": "bearer",
  "username": "user",
  "role": "user"
}

// Signal Response
{
  "symbol": "BTC/USD",
  "signal": "BUY",
  "confidence": 0.85,
  "timestamp": "2025-09-05T12:00:00"
}
```

## 🚀 Quick Start
```bash
# 1. Get admin token
TOKEN=$(curl -s -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin123"}' | \
  jq -r '.access_token')

# 2. Check status
curl -X GET http://localhost:8000/api/status \
  -H "Authorization: Bearer $TOKEN"

# 3. Create user
curl -X POST http://localhost:8000/api/admin/users \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"username": "test", "password": "test123", "role": "user"}'
```
