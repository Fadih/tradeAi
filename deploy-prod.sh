#!/bin/bash

# Production Deployment Script for Trading Agent
# This script deploys the trading agent using the Docker Hub image

set -e

echo "🚀 Starting Production Deployment..."

# Check if .env.prod exists
if [ ! -f ".env.prod" ]; then
    echo "❌ Error: .env.prod file not found!"
    echo "Please create .env.prod with your production environment variables."
    echo "You can use the following template:"
    echo ""
    echo "# Redis Configuration"
    echo "REDIS_PASSWORD=your_secure_redis_password_here"
    echo ""
    echo "# Security Configuration"
    echo "JWT_SECRET_KEY=your_jwt_secret_key_here_minimum_32_characters"
    echo "DEFAULT_ADMIN_PASSWORD=your_secure_admin_password_here"
    echo ""
    echo "# Trading API Keys"
    echo "CCXT_API_KEY=your_ccxt_api_key_here"
    echo "CCXT_API_SECRET=your_ccxt_api_secret_here"
    echo "ALPACA_KEY_ID=your_alpaca_key_id_here"
    echo "ALPACA_SECRET_KEY=your_alpaca_secret_key_here"
    echo ""
    echo "# Hugging Face Token"
    echo "HF_TOKEN=your_huggingface_token_here"
    echo ""
    echo "# Notification Services"
    echo "TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here"
    echo "TELEGRAM_CHAT_ID=your_telegram_chat_id_here"
    echo "SLACK_WEBHOOK_URL=your_slack_webhook_url_here"
    echo ""
    echo "# Logging Configuration"
    echo "AGENT_LOG_LEVEL=INFO"
    echo "AGENT_LOG_FORMAT=json"
    echo "ENABLE_MONITORING=1"
    exit 1
fi

# Pull the latest image from Docker Hub
echo "📥 Pulling latest image from Docker Hub..."
docker pull fadihussien/tradingtips:v1.0.0

# Stop existing containers if running
echo "🛑 Stopping existing containers..."
docker-compose -f docker-compose.prod.yml down || true

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p logs
mkdir -p config

# Start the production stack
echo "🚀 Starting production stack..."
docker-compose -f docker-compose.prod.yml --env-file .env.prod up -d

# Wait for services to be healthy
echo "⏳ Waiting for services to be healthy..."
sleep 30

# Check service status
echo "🔍 Checking service status..."
docker-compose -f docker-compose.prod.yml ps

# Show logs
echo "📋 Recent logs:"
docker-compose -f docker-compose.prod.yml logs --tail=20

echo ""
echo "✅ Production deployment completed!"
echo ""
echo "🌐 Web Interface: http://localhost:8000"
echo "📊 API Documentation: http://localhost:8000/docs"
echo "🔍 Health Check: http://localhost:8000/api/health"
echo ""
echo "📋 Useful commands:"
echo "  View logs: docker-compose -f docker-compose.prod.yml logs -f"
echo "  Stop services: docker-compose -f docker-compose.prod.yml down"
echo "  Restart services: docker-compose -f docker-compose.prod.yml restart"
echo "  Update image: docker pull fadihussien/tradingtips:v1.0.0 && docker-compose -f docker-compose.prod.yml up -d"
