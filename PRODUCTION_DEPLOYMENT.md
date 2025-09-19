# Production Deployment Guide

This guide explains how to deploy the Trading Agent to production using the Docker Hub image.

## Prerequisites

1. Docker and Docker Compose installed
2. Access to the Docker Hub image: `fadihussien/tradingtips:v1.0.0`
3. Production environment variables configured

## Quick Start

1. **Create environment file:**
   ```bash
   cp .env.prod.example .env.prod
   # Edit .env.prod with your actual values
   ```

2. **Deploy to production:**
   ```bash
   make docker-prod-deploy
   ```

## Environment Variables

Create a `.env.prod` file with the following variables:

```bash
# Redis Configuration
REDIS_PASSWORD=your_secure_redis_password_here

# Security Configuration
JWT_SECRET_KEY=your_jwt_secret_key_here_minimum_32_characters
DEFAULT_ADMIN_PASSWORD=your_secure_admin_password_here

# Trading API Keys
CCXT_API_KEY=your_ccxt_api_key_here
CCXT_API_SECRET=your_ccxt_api_secret_here
ALPACA_KEY_ID=your_alpaca_key_id_here
ALPACA_SECRET_KEY=your_alpaca_secret_key_here

# Hugging Face Token (for sentiment analysis)
HF_TOKEN=your_huggingface_token_here

# Notification Services
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
TELEGRAM_CHAT_ID=your_telegram_chat_id_here
SLACK_WEBHOOK_URL=your_slack_webhook_url_here

# Logging Configuration (Production)
AGENT_LOG_LEVEL=INFO
AGENT_LOG_FORMAT=json
AGENT_LOG_FILE=/app/logs/trading_agent.log
AGENT_LOG_MAX_BYTES=10485760
AGENT_LOG_BACKUP_COUNT=5

# Monitoring
ENABLE_MONITORING=1
```

## Available Commands

### Production Deployment
```bash
# Deploy to production
make docker-prod-deploy

# Stop production services
make docker-prod-stop

# View production logs
make docker-prod-logs

# Check production service status
make docker-prod-status
```

### Manual Docker Compose Commands
```bash
# Start production services
docker-compose -f docker-compose.prod.yml --env-file .env.prod up -d

# Stop production services
docker-compose -f docker-compose.prod.yml down

# View logs
docker-compose -f docker-compose.prod.yml logs -f

# Check status
docker-compose -f docker-compose.prod.yml ps
```

## Production Features

### Security
- Redis password protection
- JWT secret key for authentication
- Secure admin password
- Read-only configuration mounting

### Monitoring
- Health checks for both Redis and Trading Agent
- Structured JSON logging
- Log rotation with size limits
- Automatic restart on failure

### Performance
- Resource limits and reservations
- Optimized Python settings
- Production-grade logging
- Efficient volume management

## Access Points

After deployment, the following endpoints will be available:

- **Web Interface:** http://localhost:8000
- **API Documentation:** http://localhost:8000/docs
- **Health Check:** http://localhost:8000/api/health
- **Status Endpoint:** http://localhost:8000/api/status

## Troubleshooting

### Check Service Status
```bash
make docker-prod-status
```

### View Logs
```bash
make docker-prod-logs
```

### Restart Services
```bash
make docker-prod-stop
make docker-prod-deploy
```

### Update to Latest Image
```bash
docker pull fadihussien/tradingtips:v1.0.0
make docker-prod-deploy
```

## Production Considerations

1. **Backup:** Ensure Redis data is backed up regularly
2. **Monitoring:** Set up external monitoring for the health endpoints
3. **Logs:** Configure log aggregation for production logs
4. **Updates:** Plan for rolling updates with zero downtime
5. **Security:** Regularly rotate API keys and passwords
6. **Resources:** Monitor resource usage and adjust limits as needed

## Support

For issues or questions:
1. Check the logs: `make docker-prod-logs`
2. Verify environment variables in `.env.prod`
3. Ensure all required API keys are configured
4. Check Docker and Docker Compose versions
