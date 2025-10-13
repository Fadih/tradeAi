# API Documentation - v${VERSION}

## Base URL

- **Production:** `http://<your-domain>:32224`
- **Local:** `http://localhost:8000`
- **Docker:** `http://localhost:8000`

## Interactive Documentation

- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`

## Authentication

Most endpoints require JWT authentication.

### Login

```bash
curl -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "your-password"}'
```

Response:
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
  "token_type": "bearer",
  "user": {
    "username": "admin",
    "role": "admin"
  }
}
```

### Using Token

```bash
curl -X GET http://localhost:8000/api/signals \
  -H "Authorization: Bearer YOUR_TOKEN_HERE"
```

## Key Endpoints

### Health & Status

- `GET /api/health` - Health check
- `GET /api/status` - Application status
- `GET /metrics` - Prometheus metrics

### Trading Signals

- `GET /api/signals` - Get recent signals
- `POST /api/signals/generate` - Generate new signal
- `GET /api/signals/monitor/status` - Monitor status
- `GET /api/signals/stats` - Signal statistics

### Market Data

- `GET /api/market-data/{symbol}` - Get market data for symbol
- `GET /api/market-data/all/overview` - Multi-symbol overview

### Configuration

- `GET /api/config/app` - Application config
- `GET /api/config/trading` - Trading parameters
- `POST /api/config/update` - Update configuration

### User Management

- `GET /api/user/dashboard` - User dashboard
- `GET /api/user/profile` - User profile
- `PUT /api/user/profile` - Update profile

### Admin Endpoints

- `GET /api/admin/dashboard` - Admin dashboard
- `GET /api/admin/users` - List all users
- `POST /api/admin/users` - Create user
- `GET /api/admin/signals/all` - All signals

## WebSocket Support

Real-time updates via WebSocket (if implemented):

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/signals');
ws.onmessage = (event) => {
  console.log('New signal:', JSON.parse(event.data));
};
```

## Rate Limiting

- Default: 100 requests per minute per IP
- Authenticated: 1000 requests per minute

## Error Responses

```json
{
  "detail": "Error message",
  "status_code": 400
}
```

## Examples

See full examples in the `/examples` directory.

---

**For complete API documentation, visit:** `http://localhost:8000/docs`
