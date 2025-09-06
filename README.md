# Trading Agent (notifications-only)

Compliance: Informational alerts only. Not financial advice. Paper trade first.

## 🚀 Quick Start

### Option 1: Using Makefile (Recommended)
```bash
# Clone the repository
git clone https://github.com/Fadih/tradeAi.git
cd tradeAi

# Setup development environment
make setup-dev
source .venv/bin/activate  # On Unix/macOS
# .venv\Scripts\activate   # On Windows

# Install dependencies
make install

# Quick start guide
make quick-start
```

### Option 2: Manual Setup
```bash
# From repo root
cd tradeAi

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Unix/macOS
# .venv\Scripts\activate   # On Windows

# Install dependencies
pip install -r requirements.txt
```

### Option 3: Docker (Production Ready)
```bash
# Build and run with Docker Compose
make docker-build
make docker-run

# Or use docker-compose directly
docker-compose up -d

# Stop the container
make docker-stop
```

### Option 4: Web Interface (Interactive Dashboard)
```bash
# Install web interface dependencies
make web-install

# Run web interface in development mode
make web-run

# Or use the startup script
python start_web.py

# Access dashboard at: http://localhost:8000
# API documentation at: http://localhost:8000/docs
```

## 🛠️ Build & Development

### Available Make Commands
```bash
make help          # Show all available commands
make install       # Install Python dependencies
make setup-dev     # Setup development environment
make test          # Run tests (if pytest is installed)
make lint          # Run linting checks
make format        # Format code with black
make clean         # Clean build artifacts
make check-deps    # Check for dependency conflicts
make build         # Build the trading agent
```

### Docker Commands
```bash
make docker-build  # Build Docker image
make docker-run    # Run Docker container
make docker-stop   # Stop Docker container
make docker-clean  # Clean Docker resources
```

### Development Workflow
```bash
# 1. Setup environment
make setup-dev
source .venv/bin/activate

# 2. Install dependencies
make install

# 3. Configure environment variables
cp .env.example .env
# Edit .env with your API keys

# 4. Test the setup
python -m agent.cli show-config

# 5. Get a trading tip
python -m agent.cli tip

# 6. Run backtest
python -m agent.cli backtest

# 7. Auto-tune parameters
python -m agent.cli tune
```

## 🌐 Web Interface

### Features
- **Real-time Dashboard**: Live monitoring of trading agent status
- **Signal Generation**: Interactive trading signal generation
- **Configuration Management**: View and update agent settings
- **Market Data Visualization**: Charts and technical indicators
- **API Documentation**: Interactive API docs with Swagger UI
- **Responsive Design**: Works on desktop and mobile devices

### Quick Start
```bash
# Install web interface dependencies
make web-install

# Start the web interface
make web-run

# Access the dashboard
open http://localhost:8000
```

### Web Interface Commands
```bash
make web-install   # Install web interface dependencies
make web-run       # Run web interface (development mode)
make web-build     # Build web interface for production
make web-test      # Test web interface endpoints
```

### Environment Variables for Web Interface
```bash
export WEB_HOST="0.0.0.0"        # Host to bind to (default: 0.0.0.0)
export WEB_PORT="8000"            # Port to run on (default: 8000)
export WEB_RELOAD="true"          # Enable auto-reload (default: false)
export WEB_LOG_LEVEL="info"       # Log level (default: info)
```

### Redis Configuration
```bash
export REDIS_HOST="localhost"     # Redis host (default: localhost)
export REDIS_PORT="6379"          # Redis port (default: 6379)
export REDIS_DB="0"               # Redis database (default: 0)
export REDIS_PASSWORD=""          # Redis password (optional)
```

### API Endpoints
- `GET /` - Main dashboard
- `GET /api/status` - Agent status
- `GET /api/signals` - Recent trading signals
- `POST /api/signals/generate` - Generate new signal
- `GET /api/config` - Current configuration
- `POST /api/config/update` - Update configuration
- `GET /api/market-data/{symbol}` - Market data for symbol
- `GET /api/health` - Health check
- `GET /api/docs` - Interactive API documentation

### Redis Endpoints
- `GET /api/redis/status` - Redis connection status and statistics
- `GET /api/redis/cache/clear` - Clear Redis cache by pattern
- `GET /api/redis/metrics/{metric_name}` - Get performance metrics from Redis

### Development Mode
```bash
# Start with auto-reload
export WEB_RELOAD=true
make web-run

# Or use the startup script
python start_web.py
```

### Production Deployment
```bash
# Build production image
make docker-build

# Run with Docker Compose
docker-compose up -d

# The web interface will be available at port 8000
```

## 📦 Docker Deployment

### Building the Image
```bash
# Build locally
make docker-build

# Or with docker-compose
docker-compose build
```

### Running with Docker Compose
```bash
# Start the service
docker-compose up -d

# View logs
docker-compose logs -f trading-agent

# Stop the service
docker-compose down

# Rebuild and restart
docker-compose up -d --build
```

### Docker Environment Variables
Create a `.env` file in the project root:
```bash
# Copy the example
cp .env.example .env

# Edit with your values
nano .env
```

### Docker Volumes
- `./config:/app/config:ro` - Configuration files (read-only)
- `./logs:/app/logs` - Log files
- `./.env:/app/.env:ro` - Environment variables (read-only)

### Health Checks
The Docker container includes health checks:
```bash
# Check container health
docker inspect trading-agent | grep Health -A 10

# View health check logs
docker logs trading-agent 2>&1 | grep "health check"
```

## 🔴 **Redis Integration**

### **Performance & Caching**
Redis provides **lightning-fast performance** and **intelligent caching** for your trading agent:

- **🚀 Market Data Caching** - Cache OHLCV data to reduce API calls
- **💾 Signal Storage** - Persistent storage of trading signals with TTL
- **⚡ Configuration Caching** - Fast access to frequently used settings
- **📊 Performance Metrics** - Store and analyze trading performance over time
- **🔔 Real-time Updates** - Pub/Sub for live market data and notifications

### **Quick Redis Setup**
```bash
# Install Redis
make redis-install

# Start Redis server
make redis-start

# Check Redis status
make redis-status

# Stop Redis server
make redis-stop
```

### **Redis with Docker**
```bash
# Start with Redis included
docker-compose up -d

# Redis will be available at localhost:6379
# Trading agent will automatically connect to Redis
```

### **Redis Benefits**
- **10x faster** data access compared to file-based storage
- **Automatic TTL** for data expiration and memory management
- **Pub/Sub** for real-time market updates across multiple instances
- **Persistence** with AOF (Append-Only File) for data durability
- **Horizontal scaling** support for high-traffic deployments

## 1) Setup

```bash
# From repo root
cd /Users/fhussein/Documents/repositories/huggingface/tradeAi

# Create venv
python3 -m venv .venv
source .venv/bin/activate

# Install core deps (no GPU/ML heavy packages yet)
pip install -r requirements.txt
```

Optional ML (for local FinBERT inference). Skip if you will use HF Inference API via `HF_TOKEN`.
```bash
# Optional
pip install transformers torch
```

## 2) Configure via environment

### Core Configuration
- `AGENT_TICKERS`: comma-separated universe; e.g. `BTC/USDT,ETH/USDT,SPY`
- `AGENT_TIMEFRAME`: e.g. `15m`, `1h`, `1d`
- `AGENT_NOTIFIER`: `console|telegram|slack`
- `AGENT_BUY_THRESHOLD`, `AGENT_SELL_THRESHOLD`: fused signal thresholds (default 0.5 / -0.5)

### Data Sources
- `CCXT_EXCHANGE`: crypto exchange (default: `binance`)
- `CCXT_API_KEY`, `CCXT_API_SECRET`: optional API keys for rate limits
- `ALPACA_KEY_ID`, `ALPACA_SECRET_KEY`: for equities/ETFs data
- `ALPACA_BASE_URL`: Alpaca API URL (default: paper trading)

### Sentiment & ML
- `HF_TOKEN`: Hugging Face API token (for Inference API)
- `HF_FIN_SENT_MODEL`: sentiment model id (default `ProsusAI/finbert`)

### Notifications
- `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`: for Telegram alerts
- `SLACK_WEBHOOK_URL`: Slack incoming webhook URL

### Logging Configuration
- `AGENT_LOG_LEVEL`: log level (`debug`, `info`, `warning`, `error`, `critical`)
- `AGENT_LOG_FORMAT`: log format style (`simple`, `verbose`)
- `AGENT_LOG_FILE`: optional file path for logging

**Log Levels:**
- `debug`: Detailed debugging information (most verbose)
- `info`: General information about program execution
- `warning`: Warning messages for potentially problematic situations
- `error`: Error messages for serious problems
- `critical`: Critical errors that may prevent the program from running

**Log Formats:**
- `simple`: `2024-01-15 10:30:45 | agent.cli | INFO | Trading Agent CLI starting`
- `verbose`: `2024-01-15 10:30:45 | agent.cli | INFO | main:45 | Trading Agent CLI starting`

Example (console mode):
```bash
export AGENT_TICKERS="BTC/USDT,ETH/USDT,SPY"
export AGENT_TIMEFRAME="1h"
export AGENT_NOTIFIER="console"
```

Example (Telegram):
```bash
export AGENT_NOTIFIER="telegram"
export TELEGRAM_BOT_TOKEN="<bot_token>"
export TELEGRAM_CHAT_ID="<chat_id>"
```

Example (Slack):
```bash
export AGENT_NOTIFIER="slack"
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/XXX/YYY/ZZZ"
```

Example (HF Inference API for sentiment):
```bash
export HF_TOKEN="hf_..."
# optional override
export HF_FIN_SENT_MODEL="ProsusAI/finbert"
```

Example (Logging configuration):
```bash
# Verbose debugging
export AGENT_LOG_LEVEL="debug"
export AGENT_LOG_FORMAT="verbose"

# File logging
export AGENT_LOG_FILE="/tmp/trading_agent.log"

# Production logging
export AGENT_LOG_LEVEL="warning"
export AGENT_LOG_FORMAT="simple"
```

## 3) Run the CLI

Show effective config:
```bash
python -m agent.cli show-config
```

RSI-only quick tips:
```bash
python -m agent.cli tip
```

Fused tips (technical + sentiment + RSS news):
```bash
python -m agent.cli run-once
```

Backtest current strategy on first ticker:
```bash
python -m agent.cli backtest --bars 500
```

Auto-tune parameters to maximize Sharpe:
```bash
python -m agent.cli tune --bars 500
```

Schedule fused tips (every 15 minutes by default):
```bash
python -m agent.cli schedule --cron "*/15 * * * *"
# Ctrl+C to stop
```

## 4) Strategy Components

### Technical Indicators
- **RSI(14)**: overbought/oversold detection
- **EMA(12,26)**: trend following with MACD
- **ATR(14)**: volatility-based stop/TP calculation
- **Regime Filter**: EMA(50) vs EMA(200) for bull/bear bias

### Signal Fusion
- Technical score: RSI + MACD histogram, normalized to [-1, 1]
- Sentiment score: FinBERT on RSS headlines, mapped to [-1, 1]
- Fused score: `w_tech * tech + w_sent * sentiment`
- Position: BUY if score ≥ threshold, SELL if ≤ -threshold

### Risk Management
- Stop loss: 2×ATR from entry
- Take profit: 3×ATR from entry
- Regime filter: only long in bull markets, short in bear markets

## 📊 **Signal Generation Process & Indicator Explanations**

### **How Trading Signals Are Generated**

The trading agent follows a sophisticated 14-step process to generate trading signals:

#### **Step 1: Data Collection**
- **OHLCV Data**: Fetches Open, High, Low, Close, Volume data
- **Crypto**: Uses CCXT (Binance) for symbols like `BTC/USDT`, `ETH/USDT`
- **Stocks/ETFs**: Uses Alpaca API for symbols like `SPY`, `AAPL`
- **Timeframes**: 1m, 5m, 15m, 30m, 1h, 4h, 1d

#### **Step 2: Technical Analysis**
The system calculates multiple technical indicators from OHLCV data:

##### **RSI (Relative Strength Index)**
```python
# Formula: RSI = 100 - (100 / (1 + RS))
# Where RS = Average Gain / Average Loss over 14 periods
```
- **Range**: 0-100
- **RSI > 70**: Overbought (price might fall) → Bearish signal
- **RSI < 30**: Oversold (price might rise) → Bullish signal
- **RSI = 50**: Neutral
- **Purpose**: Identifies momentum and potential reversal points

##### **MACD (Moving Average Convergence Divergence)**
```python
# MACD Line = EMA(12) - EMA(26)
# Signal Line = EMA(9) of MACD Line
# Histogram = MACD Line - Signal Line
```
- **MACD > Signal**: Bullish momentum
- **MACD < Signal**: Bearish momentum
- **Histogram**: Shows momentum strength and direction changes
- **Purpose**: Identifies trend changes and momentum shifts

##### **EMA (Exponential Moving Average)**
```python
# EMA = (Price × Multiplier) + (Previous EMA × (1 - Multiplier))
# Multiplier = 2 / (Period + 1)
```
- **EMA(12)**: Short-term trend
- **EMA(26)**: Long-term trend
- **EMA(50)**: Medium-term trend
- **EMA(200)**: Long-term trend
- **Purpose**: Smooths price data to identify trends

##### **ATR (Average True Range)**
```python
# True Range = max(High-Low, |High-PrevClose|, |Low-PrevClose|)
# ATR = EMA of True Range over 14 periods
```
- **High ATR**: High volatility (price moves a lot)
- **Low ATR**: Low volatility (price moves little)
- **Purpose**: Used for setting stop-loss and take-profit levels

#### **Step 3: Sentiment Analysis**
- **News Sources**: RSS feeds from Google News (crypto + stock keywords)
- **AI Model**: FinBERT (financial sentiment analysis)
- **Processing**: Local inference (faster) or HF Inference API (fallback)
- **Output**: Sentiment score from -1 (very negative) to +1 (very positive)

#### **Step 4: Signal Fusion**
```python
# Technical Score = (RSI_Score + MACD_Score) / 2
# Fused Score = (Technical_Weight × Technical_Score) + (Sentiment_Weight × Sentiment_Score)
# Default weights: Technical=60%, Sentiment=40%
```

#### **Step 5: Signal Decision**
```python
if Fused_Score >= Buy_Threshold:     # Default: 0.7
    Signal = "BUY"
elif Fused_Score <= Sell_Threshold:  # Default: -0.7
    Signal = "SELL"
else:
    Signal = "HOLD"
```

### **Real Example: ETH/USDT Signal Generation**

From your recent logs, here's how a signal was generated:

#### **OHLCV Data (Last 3 hours)**
```
Time: 2025-09-05 16:00:00 | Open: $4283.40 | High: $4303.75 | Low: $4274.56 | Close: $4297.81 | Volume: 17,892
Time: 2025-09-05 17:00:00 | Open: $4297.81 | High: $4299.21 | Low: $4272.35 | Close: $4289.99 | Volume: 13,261
Time: 2025-09-05 18:00:00 | Open: $4289.99 | High: $4301.87 | Low: $4284.38 | Close: $4294.90 | Volume: 4,435
```

#### **Technical Analysis Results**
- **RSI**: Calculated from 200 hours of close prices
- **MACD**: Based on EMA(12) and EMA(26) of close prices
- **Technical Score**: -0.09 (slightly bearish)

#### **Sentiment Analysis Results**
- **News Headlines**: 10 headlines from RSS feeds
- **AI Analysis**: FinBERT processed 3 sample headlines
- **Sentiment Score**: -1.0 (very negative sentiment)

#### **Final Signal**
- **Fused Score**: (0.6 × -0.09) + (0.4 × -1.0) = -0.45
- **Decision**: HOLD (below -0.7 sell threshold)
- **Confidence**: 45.2% (absolute value of fused score)

### **Indicator Interpretation Guide**

#### **RSI Interpretation**
- **0-30**: Oversold (potential buying opportunity)
- **30-50**: Bearish momentum
- **50-70**: Bullish momentum
- **70-100**: Overbought (potential selling opportunity)

#### **MACD Interpretation**
- **MACD > Signal**: Bullish momentum
- **MACD < Signal**: Bearish momentum
- **Histogram increasing**: Momentum strengthening
- **Histogram decreasing**: Momentum weakening

#### **Volume Analysis**
- **High Volume**: Strong conviction in price movement
- **Low Volume**: Weak conviction, potential reversal
- **Volume + Price**: Confirms trend strength

#### **Sentiment Analysis**
- **+1.0 to +0.5**: Very positive (bullish)
- **+0.5 to 0**: Slightly positive (neutral-bullish)
- **0**: Neutral
- **0 to -0.5**: Slightly negative (neutral-bearish)
- **-0.5 to -1.0**: Very negative (bearish)

### **Signal Quality Assessment**

#### **High Quality Signals**
- **Strong Technical + Strong Sentiment**: Both indicators agree
- **High Volume**: Confirms market conviction
- **Clear Trend**: Price moving in signal direction

#### **Medium Quality Signals**
- **Mixed Signals**: Technical and sentiment disagree
- **Medium Volume**: Moderate conviction
- **Sideways Movement**: Price consolidating

#### **Low Quality Signals**
- **Conflicting Indicators**: Technical and sentiment oppose
- **Low Volume**: Weak conviction
- **Choppy Price Action**: Unclear direction

### **Risk Management Integration**

#### **Stop Loss Calculation**
```python
Stop_Loss = Entry_Price ± (2 × ATR)
# For long positions: Entry - (2 × ATR)
# For short positions: Entry + (2 × ATR)
```

#### **Take Profit Calculation**
```python
Take_Profit = Entry_Price ± (3 × ATR)
# For long positions: Entry + (3 × ATR)
# For short positions: Entry - (3 × ATR)
```

#### **Position Sizing**
- **Risk per Trade**: 1-2% of account
- **ATR-based Sizing**: Adjust position size based on volatility
- **Correlation Limits**: Avoid overexposure to correlated assets

### **Signal Monitoring & Validation**

#### **Real-time Monitoring**
- **Web Dashboard**: Live signal updates
- **API Endpoints**: Programmatic access to signals
- **Notifications**: Telegram/Slack alerts for new signals

#### **Signal Validation**
- **Backtesting**: Historical performance validation
- **Paper Trading**: Live market testing without risk
- **Performance Metrics**: Win rate, Sharpe ratio, max drawdown

#### **Continuous Improvement**
- **Auto-tuning**: Automatic parameter optimization
- **Performance Tracking**: Monitor signal accuracy over time
- **Strategy Refinement**: Adjust based on market conditions

## 5) Auto-Tuning

The `tune` command automatically finds optimal parameters:
- **Thresholds**: buy/sell signal levels (0.3 to 0.6)
- **Weights**: technical vs sentiment importance (0.5 to 0.8)
- **Objective**: maximize Sharpe ratio
- **Constraints**: regime filter enabled, realistic trade frequency

Example output:
```json
{
  "symbol": "BTC/USDT",
  "best_params": {"buy_th": 0.3, "sell_th": -0.3, "w_tech": 0.5, "w_sent": 0.3},
  "sharpe": 6.695,
  "max_drawdown": -0.001,
  "win_rate": 0.667,
  "trades": 4
}
```

## 6) Data Sources

### Crypto (CCXT)
- Supports 100+ exchanges
- Automatic fallback to stubs if API unavailable
- Rate limiting and error handling

### Equities (Alpaca)
- Paper trading by default
- Real-time and historical data
- Fallback to stubs if keys not configured

### News Sentiment
- RSS feeds: Google News, crypto/stock keywords
- FinBERT sentiment analysis
- Local inference or HF Inference API

## 7) What's included now

### Core Features
- Real data: CCXT (crypto) + Alpaca (equities) with fallbacks
- Indicators: RSI, EMA, MACD, ATR with regime filtering
- Sentiment: FinBERT on RSS news + HF Inference API
- Fusion: configurable tech/sentiment weights
- Notifiers: console, Telegram, Slack
- Scheduler: minute-based cron via APScheduler

### Advanced Features
- Auto-tuning: grid search for optimal parameters
- Backtesting: Sharpe, drawdown, win-rate metrics
- Regime awareness: bull/bear market filtering
- Risk management: ATR-based stops and targets

### Logging & Monitoring
- Configurable log levels (DEBUG to CRITICAL)
- Structured logging with timestamps and module names
- File logging support
- Verbose debugging mode
- Performance metrics logging

## 8) Configuration Examples

### Basic Console Mode
```bash
export AGENT_TICKERS="BTC/USDT,ETH/USDT,SPY"
export AGENT_TIMEFRAME="1h"
export AGENT_NOTIFIER="console"
```

### Crypto Trading with Telegram
```bash
export AGENT_TICKERS="BTC/USDT,ETH/USDT,ADA/USDT"
export AGENT_TIMEFRAME="15m"
export AGENT_NOTIFIER="telegram"
export TELEGRAM_BOT_TOKEN="your_bot_token"
export TELEGRAM_CHAT_ID="your_chat_id"
export CCXT_EXCHANGE="binance"
```

### Equities with Alpaca
```bash
export AGENT_TICKERS="SPY,QQQ,AAPL"
export AGENT_TIMEFRAME="1h"
export AGENT_NOTIFIER="slack"
export SLACK_WEBHOOK_URL="your_webhook_url"
export ALPACA_KEY_ID="your_key"
export ALPACA_SECRET_KEY="your_secret"
```

### Sentiment Analysis
```bash
export HF_TOKEN="hf_your_token_here"
export HF_FIN_SENT_MODEL="ProsusAI/finbert"
```

### Custom Thresholds
```bash
export AGENT_BUY_THRESHOLD=0.3
export AGENT_SELL_THRESHOLD=-0.3
```

### Debug Logging
```bash
export AGENT_LOG_LEVEL="debug"
export AGENT_LOG_FORMAT="verbose"
export AGENT_LOG_FILE="/tmp/debug.log"
```

## 9) Workflow

1. **Setup**: Install deps, configure env vars
2. **Tune**: Run `python -m agent.cli tune` to find best params
3. **Test**: Use `python -m agent.cli backtest` to validate
4. **Monitor**: Run `python -m agent.cli run-once` for current signals
5. **Automate**: Schedule with `python -m agent.cli schedule`

## 🚀 Production Deployment

### Docker Production Setup
```bash
# Build production image
make docker-build

# Run with production environment
docker run -d --name trading-agent-prod \
  --restart unless-stopped \
  -e AGENT_LOG_LEVEL=warning \
  -e AGENT_LOG_FORMAT=simple \
  -v $(PWD)/config:/app/config:ro \
  -v $(PWD)/logs:/app/logs \
  -v $(PWD)/.env:/app/.env:ro \
  trading-agent:latest
```

### Docker Compose Production
```bash
# Start production stack
docker-compose -f docker-compose.yml up -d

# Monitor logs
docker-compose logs -f trading-agent

# Scale if needed
docker-compose up -d --scale trading-agent=2
```

### Systemd Service (Linux)
Create `/etc/systemd/system/trading-agent.service`:
```ini
[Unit]
Description=Trading Agent
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/path/to/tradeAi
ExecStart=/usr/bin/docker-compose up -d
ExecStop=/usr/bin/docker-compose down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable trading-agent
sudo systemctl start trading-agent
sudo systemctl status trading-agent
```

### Monitoring & Logging
```bash
# View real-time logs
docker logs -f trading-agent

# Check container health
docker inspect trading-agent | grep Health

# Monitor resource usage
docker stats trading-agent

# View logs from specific time
docker logs --since="2024-01-15T10:00:00" trading-agent
```

### Backup & Recovery
```bash
# Backup configuration and logs
tar -czf trading-agent-backup-$(date +%Y%m%d).tar.gz \
  config/ logs/ .env

# Restore from backup
tar -xzf trading-agent-backup-20240115.tar.gz

# Restart with restored config
docker-compose restart
```

## 10) Next Steps (Optional)

### Performance Improvements
- Add volatility filters (VIX, realized vol)
- Implement position sizing based on Kelly criterion
- Add correlation analysis for portfolio optimization

### Infrastructure
- ✅ Docker deployment with compose
- Centralized secrets management (Vault, AWS Secrets Manager)
- Logging and metrics dashboard (Grafana, Prometheus)
- Multi-timeframe analysis
- Kubernetes deployment manifests

### Advanced Features
- TimesFM forecasting integration
- Alternative sentiment sources (Twitter, Reddit)
- Machine learning signal enhancement
- Portfolio backtesting across multiple assets
- Web dashboard for real-time monitoring

## 🔧 Troubleshooting

### Common Issues

#### Docker Issues
```bash
# Container won't start
docker logs trading-agent

# Permission denied on volumes
sudo chown -R $USER:$USER config/ logs/

# Port already in use
docker-compose down
docker-compose up -d
```

#### Environment Variables
```bash
# Check if variables are loaded
python -m agent.cli show-config

# Test specific variable
echo $AGENT_LOG_LEVEL

# Reload environment
source .env
```

#### Dependencies
```bash
# Check for conflicts
make check-deps

# Reinstall dependencies
make clean
make install

# Update specific package
pip install --upgrade package-name
```

#### Logging Issues
```bash
# Check log file permissions
ls -la logs/

# Test logging
export AGENT_LOG_LEVEL=debug
python -m agent.cli tip

# View logs in real-time
tail -f logs/trading_agent.log
```

### Performance Tuning
```bash
# Run with debug logging to see bottlenecks
export AGENT_LOG_LEVEL=debug
python -m agent.cli run-once

# Profile memory usage
docker stats trading-agent

# Check system resources
htop
iostat
```

### Security Considerations
- Never commit `.env` files to version control
- Use Docker secrets for production deployments
- Regularly rotate API keys
- Monitor container logs for suspicious activity
- Use read-only volumes where possible

## 📚 Notes
- This agent is notifications-only. It does not place trades.
- Always validate signals with paper trading or backtesting before acting.
- Use the tune command to find optimal parameters for your market conditions.
- Monitor regime changes and adjust thresholds accordingly.
- Use debug logging to troubleshoot issues and understand signal generation.
- File logging is useful for production monitoring and debugging.
- Docker deployment provides isolation and easy scaling.
- Use the Makefile for consistent development workflows.
- Monitor container health checks in production environments.

---

## 🎉 **Congratulations!**

**You now have one of the most comprehensive trading agent systems available, combining the power of AI sentiment analysis, technical indicators, and a beautiful web interface. The agent is ready to help you make informed trading decisions!**

## 🚀 **What Makes This Trading Agent Special**

This is **not just another trading bot** - it's a **comprehensive AI-powered trading intelligence system** that combines:

- **🤖 Advanced AI Sentiment Analysis** - Real-time market sentiment using FinBERT
- **📊 Professional Technical Indicators** - RSI, MACD, EMA, ATR with regime filtering
- **🎯 Intelligent Signal Fusion** - Combines technical + sentiment for optimal decisions
- **🌐 Beautiful Web Dashboard** - Real-time monitoring and interactive controls
- **⚡ Auto-Tuning Engine** - Automatically optimizes parameters for maximum performance
- **📈 Comprehensive Backtesting** - Validate strategies before live trading
- **🔔 Multi-Platform Notifications** - Telegram, Slack, and console alerts
- **🏗️ Production-Ready Architecture** - Docker, logging, monitoring, and scaling

**This trading agent system is designed to give you the edge that professional traders have, with the simplicity that beginners need.**

---

**Ready to start trading smarter?** 🚀

```bash
# Quick start
make web-install
make web-run

# Open your browser to: http://localhost:8000
```

---

## 👨‍💻 **Developer**

**Developed by [Fadi Hussein](https://github.com/Fadih)**  
**Senior DevOps Engineer**

*This trading agent system represents the culmination of years of experience in DevOps, automation, and financial technology. Built with production-grade architecture, comprehensive monitoring, and enterprise-level scalability.*

**Connect with the developer:**
- 🐙 **GitHub**: [Fadih](https://github.com/Fadih)
- 🚀 **LinkedIn**: [Fadi Hussein](www.linkedin.com/in/fadi-hussein-8ab7403b)


---

**⭐ If this project helps you, please give it a star on GitHub!**
