# Trading Agent (notifications-only)

Compliance: Informational alerts only. Not financial advice. Paper trade first.

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

## 9) Workflow

1. **Setup**: Install deps, configure env vars
2. **Tune**: Run `python -m agent.cli tune` to find best params
3. **Test**: Use `python -m agent.cli backtest` to validate
4. **Monitor**: Run `python -m agent.cli run-once` for current signals
5. **Automate**: Schedule with `python -m agent.cli schedule`

## 10) Next Steps (Optional)

### Performance Improvements
- Add volatility filters (VIX, realized vol)
- Implement position sizing based on Kelly criterion
- Add correlation analysis for portfolio optimization

### Infrastructure
- Docker deployment with compose
- Centralized secrets management
- Logging and metrics dashboard
- Multi-timeframe analysis

### Advanced Features
- TimesFM forecasting integration
- Alternative sentiment sources (Twitter, Reddit)
- Machine learning signal enhancement
- Portfolio backtesting across multiple assets

## Notes
- This agent is notifications-only. It does not place trades.
- Always validate signals with paper trading or backtesting before acting.
- Use the tune command to find optimal parameters for your market conditions.
- Monitor regime changes and adjust thresholds accordingly.
