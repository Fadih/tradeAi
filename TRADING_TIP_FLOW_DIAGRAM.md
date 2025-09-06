# Trading AI Tips System - Complete Flow Diagram v2.0

## System Overview

The Trading AI Tips system is a comprehensive trading signal generation and monitoring platform that combines technical analysis, sentiment analysis, and automated monitoring to provide intelligent trading recommendations.

---

## 1. Signal Generation Flow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           USER INTERFACE (Frontend)                            │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 1. USER FILLS FORM                                                              │
│    ├── Custom Thresholds (Optional)                                             │
│    │   ├── Buy Threshold (default: 0.7)                                        │
│    │   ├── Sell Threshold (default: -0.7)                                      │
│    │   ├── Technical Weight (default: 0.6)                                     │
│    │   └── Sentiment Weight (default: 0.4)                                     │
│    ├── Market Selection (Required)                                              │
│    │   └── Symbol dropdown (BTC/USDT, ETH/USDT, SPY, etc.)                    │
│    └── Timeframe Selection (Required)                                           │
│        └── 15m, 1h, 4h, 1d                                                     │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 2. USER CLICKS "Generate Trading Tip"                                           │
│    ├── Frontend validation                                                      │
│    │   ├── Check if market is selected                                         │
│    │   └── Show loading state ("Generating...")                               │
│    ├── Prepare request payload                                                  │
│    │   ├── symbol, timeframe                                                   │
│    │   ├── buy_threshold (or null)                                             │
│    │   ├── sell_threshold (or null)                                            │
│    │   ├── technical_weight (or null)                                          │
│    │   └── sentiment_weight (or null)                                          │
│    └── Send POST request to /api/signals/generate                              │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           BACKEND API (FastAPI)                                │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 3. AUTHENTICATION & VALIDATION                                                  │
│    ├── Verify JWT token from Authorization header                               │
│    ├── Extract user information (username, role)                                │
│    ├── Validate request payload                                                 │
│    └── Log: "Starting signal generation for {symbol}"                           │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 4. MARKET DATA FETCHING                                                         │
│    ├── Determine data source based on symbol                                   │
│    │   ├── If symbol contains "/" → Crypto (CCXT)                             │
│    │   └── Else → Stock/ETF (Alpaca)                                          │
│    ├── Fetch OHLCV data                                                        │
│    │   ├── Crypto: fetch_ohlcv(symbol, timeframe)                             │
│    │   └── Stock: fetch_alpaca_ohlcv(symbol, timeframe)                       │
│    ├── Validate data (not None and not empty)                                 │
│    └── Extract: close, high, low, volume series                               │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 5. TECHNICAL INDICATORS CALCULATION                                             │
│    ├── RSI (Relative Strength Index)                                           │
│    │   └── compute_rsi(close) → RSI values                                     │
│    ├── EMA (Exponential Moving Averages)                                       │
│    │   ├── EMA 12: compute_ema(close, 12)                                      │
│    │   └── EMA 26: compute_ema(close, 26)                                      │
│    ├── MACD (Moving Average Convergence Divergence)                            │
│    │   └── compute_macd(close) → {macd, signal, hist}                         │
│    └── ATR (Average True Range)                                                │
│        └── compute_atr(high, low, close) → ATR values                         │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 6. TECHNICAL SCORE CALCULATION                                                  │
│    ├── Get latest values                                                       │
│    │   ├── current_close = close.iloc[-1]                                      │
│    │   ├── current_rsi = rsi.iloc[-1]                                          │
│    │   └── current_macd_hist = macd_hist.iloc[-1]                             │
│    ├── Normalize RSI to [-1, 1]                                                │
│    │   └── rsi_score = (current_rsi - 50) / 50                                │
│    ├── Normalize MACD to [-1, 1]                                               │
│    │   └── macd_score = max(min(current_macd_hist / 1000, 1), -1)             │
│    └── Calculate combined technical score                                      │
│        └── tech_score = (rsi_score + macd_score) / 2                          │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 7. ENHANCED SENTIMENT ANALYSIS (AI-Powered)                                    │
│    ├── Fetch news headlines                                                    │
│    │   ├── Stock market RSS: Google News (10 headlines)                       │
│    │   └── Crypto RSS: Google News (10 headlines)                             │
│    ├── Fetch Reddit posts (NEW)                                               │
│    │   ├── Crypto subreddits: r/Bitcoin, r/ethereum, r/cryptocurrency         │
│    │   └── Stock subreddits: r/stocks, r/investing, r/SecurityAnalysis        │
│    ├── Combine and sample data                                                 │
│    │   ├── Mix RSS headlines and Reddit posts                                 │
│    │   ├── Random shuffle for variety                                          │
│    │   └── Sample up to 20 texts for analysis                                 │
│    ├── AI Sentiment Analysis                                                   │
│    │   ├── Use FinBERT model (ProsusAI/finbert)                               │
│    │   ├── Analyze mixed sample of texts                                       │
│    │   └── Return sentiment score [-1, 1]                                     │
│    └── Handle errors (fallback to 0.0)                                        │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 8. SIGNAL FUSION & DECISION MAKING                                             │
│    ├── Apply custom weights (or defaults)                                      │
│    │   ├── tech_weight = custom_tech_weight || 0.6                            │
│    │   └── sentiment_weight = custom_sentiment_weight || 0.4                   │
│    ├── Calculate fused score                                                   │
│    │   └── fused_score = (tech_weight × tech_score) + (sentiment_weight × sentiment_score) │
│    ├── Apply custom thresholds (or defaults)                                   │
│    │   ├── buy_threshold = custom_buy_threshold || 0.7                        │
│    │   └── sell_threshold = custom_sell_threshold || -0.7                     │
│    └── Determine signal type                                                   │
│        ├── If fused_score >= buy_threshold → "BUY"                            │
│        ├── If fused_score <= sell_threshold → "SELL"                          │
│        └── Else → "HOLD"                                                      │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 9. RISK MANAGEMENT CALCULATION                                                  │
│    ├── Get current price and ATR                                               │
│    │   ├── current_close = close.iloc[-1]                                      │
│    │   └── current_atr = atr.iloc[-1]                                         │
│    ├── Calculate stop loss and take profit                                     │
│    │   ├── For BUY signals:                                                   │
│    │   │   ├── stop_loss = current_close - (2 × ATR)                         │
│    │   │   └── take_profit = current_close + (3 × ATR)                       │
│    │   └── For SELL signals:                                                  │
│    │       ├── stop_loss = current_close + (2 × ATR)                         │
│    │       └── take_profit = current_close - (3 × ATR)                       │
│    └── For HOLD signals: No risk management levels                             │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 10. SIGNAL OBJECT CREATION                                                      │
│     ├── Create TradingSignal object                                            │
│     │   ├── symbol, timeframe, timestamp, signal_type                         │
│     │   ├── confidence = abs(fused_score)                                      │
│     │   ├── technical_score, sentiment_score, fused_score                     │
│     │   ├── stop_loss, take_profit                                             │
│     │   ├── reasoning = "Technical: X, Sentiment: Y, Fused: Z"                │
│     │   └── applied thresholds and weights                                     │
│     └── Add user information                                                   │
│         └── username = current_user['username']                               │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 11. DATA PERSISTENCE (Redis)                                                    │
│     ├── Store signal in Redis                                                  │
│     │   ├── Key: signal:{timestamp}                                            │
│     │   ├── Store as JSON with timestamp                                       │
│     │   ├── Add to user signals list                                           │
│     │   ├── Add to symbol index                                                │
│     │   └── Add to global signals list                                         │
│     ├── Add signal history event                                               │
│     │   ├── Event type: "signal_created"                                       │
│     │   ├── Description: "Signal created: {signal_type} for {symbol}"         │
│     │   ├── Metadata: scores, confidence, thresholds, etc.                     │
│     │   └── Timestamp: signal creation time                                    │
│     └── Log activity                                                           │
│         ├── Activity type: "signal_generated"                                 │
│         ├── Description: "Generated {signal_type} for {symbol}"               │
│         ├── User: current_user['username']                                     │
│         └── Metadata: scores, confidence, etc.                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 12. RESPONSE TO FRONTEND                                                        │
│     ├── Return TradingSignal object as JSON                                    │
│     ├── Include all calculated values                                          │
│     └── HTTP 200 OK response                                                   │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           USER INTERFACE (Frontend)                            │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 13. FRONTEND PROCESSING                                                         │
│     ├── Receive signal response                                                 │
│     ├── Show signal modal with details                                          │
│     │   ├── Market, Recommendation, Confidence                                  │
│     │   ├── Technical, Sentiment, Fused scores                                 │
│     │   ├── Risk management levels                                              │
│     │   └── Applied thresholds and weights                                     │
│     ├── Refresh signals list                                                   │
│     ├── Update dashboard status                                                 │
│     ├── Show success notification                                               │
│     └── Reset button state                                                      │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 14. USER SEES RESULT                                                            │
│     ├── Signal modal displays                                                   │
│     ├── Recent Trading Tips list updates                                       │
│     ├── Dashboard counters update                                               │
│     └── Success notification shows                                             │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Automated Monitoring Flow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        MONITORING SYSTEM (Scheduler)                           │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 1. SCHEDULED MONITORING TRIGGER                                                 │
│    ├── Cron job runs every 2 minutes                                           │
│    ├── Get all active signals from Redis                                       │
│    ├── Group signals by symbol                                                 │
│    └── Skip admin signals (monitor user signals only)                          │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 2. FOR EACH SYMBOL GROUP                                                        │
│    ├── Fetch fresh market data                                                  │
│    │   ├── Crypto: fetch_ccxt_ohlcv(symbol, timeframe)                        │
│    │   └── Stock: fetch_alpaca_ohlcv(symbol, timeframe)                       │
│    ├── Get fresh sentiment data                                                 │
│    │   ├── Fetch RSS headlines (10 per feed)                                  │
│    │   ├── Fetch Reddit posts (3 per subreddit)                               │
│    │   ├── Mix and sample up to 20 texts                                      │
│    │   └── Analyze with FinBERT model                                          │
│    └── Calculate fresh technical indicators                                    │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 3. FOR EACH SIGNAL IN GROUP                                                     │
│    ├── Recalculate technical score                                             │
│    │   ├── Get fresh RSI and MACD values                                      │
│    │   ├── Normalize to [-1, 1] range                                         │
│    │   └── Calculate combined technical score                                  │
│    ├── Apply fresh sentiment score                                             │
│    ├── Recalculate fused score                                                 │
│    │   └── fused_score = (tech_weight × tech_score) + (sentiment_weight × sentiment_score) │
│    └── Determine new signal type                                               │
│        ├── If fused_score >= buy_threshold → "BUY"                            │
│        ├── If fused_score <= sell_threshold → "SELL"                          │
│        └── Else → "HOLD"                                                      │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 4. SIGNAL STATUS EVALUATION                                                     │
│    ├── Compare new signal type with original                                   │
│    ├── If status changed:                                                      │
│    │   ├── Update signal in Redis                                              │
│    │   ├── Add "status_changed" history event                                  │
│    │   ├── Log activity                                                        │
│    │   └── Mark as updated                                                     │
│    └── If status unchanged:                                                    │
│        ├── Add "monitoring_cycle" history event                                │
│        ├── Include fresh scores in metadata                                    │
│        └── Mark as monitored (not updated)                                     │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 5. MONITORING COMPLETION                                                        │
│    ├── Log monitoring results                                                  │
│    │   ├── Total signals monitored                                             │
│    │   ├── Number of signals updated                                           │
│    │   ├── Number of errors                                                    │
│    │   └── Timestamp of completion                                             │
│    └── Schedule next monitoring cycle                                          │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Position Management Flow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           USER INTERFACE (Frontend)                            │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 1. USER CREATES POSITION                                                        │
│    ├── Select signal from Recent Trading Tips                                  │
│    ├── Enter position details                                                  │
│    │   ├── Position size (USD amount)                                         │
│    │   ├── Entry price                                                        │
│    │   └── Notes (optional)                                                   │
│    └── Click "Create Position"                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 2. POSITION CREATION REQUEST                                                    │
│    ├── Send POST to /api/positions                                             │
│    ├── Include signal timestamp                                                │
│    ├── Include position details                                                │
│    └── Include user authentication                                             │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 3. BACKEND POSITION PROCESSING                                                  │
│    ├── Validate request and authentication                                     │
│    ├── Fetch signal details from Redis                                         │
│    ├── Create position object                                                  │
│    │   ├── Generate unique position ID                                         │
│    │   ├── Link to signal timestamp                                            │
│    │   ├── Set initial status as "open"                                        │
│    │   ├── Calculate initial PnL (0.0)                                         │
│    │   └── Set creation timestamp                                              │
│    └── Store position in Redis                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 4. POSITION MONITORING                                                          │
│    ├── Positions automatically update prices                                   │
│    ├── Calculate real-time PnL                                                 │
│    │   ├── For long positions: PnL = (current_price - entry_price) / entry_price │
│    │   └── For short positions: PnL = (entry_price - current_price) / entry_price │
│    ├── Update position status                                                  │
│    └── Store updated position in Redis                                         │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 5. POSITION MANAGEMENT                                                          │
│    ├── User can update position                                                │
│    │   ├── Close position (set exit price)                                     │
│    │   ├── Update notes                                                        │
│    │   └── Change status                                                       │
│    ├── Calculate final PnL                                                     │
│    ├── Update position in Redis                                                │
│    └── Generate performance analytics                                          │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. User Management Flow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           ADMIN INTERFACE                                      │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 1. USER LIFECYCLE MANAGEMENT                                                    │
│    ├── Create User                                                              │
│    │   ├── Set username, password, role                                        │
│    │   ├── Set personal information                                            │
│    │   ├── Set activation period (days)                                        │
│    │   └── Store in Redis with hashed password                                 │
│    ├── Edit User                                                                │
│    │   ├── Update personal information                                         │
│    │   ├── Extend activation period                                            │
│    │   └── Change role (user/admin)                                            │
│    ├── Deactivate User                                                          │
│    │   ├── Set activation_days to 0                                            │
│    │   ├── Set activation_expires_at to null                                   │
│    │   └── User cannot login                                                   │
│    └── Delete User                                                              │
│        ├── Remove user from Redis                                              │
│        ├── Optionally delete user's signals                                    │
│        └── Log deletion activity                                               │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 2. USER AUTHENTICATION                                                          │
│    ├── Login Request                                                            │
│    │   ├── Username and password                                               │
│    │   └── POST to /api/auth/login                                             │
│    ├── Authentication Process                                                   │
│    │   ├── Hash provided password                                              │
│    │   ├── Compare with stored hash                                            │
│    │   ├── Check activation status                                             │
│    │   └── Check activation expiry                                             │
│    ├── Token Generation                                                         │
│    │   ├── Generate JWT token                                                  │
│    │   ├── Include user role and permissions                                   │
│    │   └── Set expiration time                                                 │
│    └── Response                                                                 │
│        ├── Return access token                                                 │
│        ├── Update last_login timestamp                                         │
│        └── Log login activity                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           SYSTEM COMPONENTS                                    │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                    ┌───────────────────┼───────────────────┐
                    │                   │                   │
                    ▼                   ▼                   ▼
┌─────────────────────────┐ ┌─────────────────────────┐ ┌─────────────────────────┐
│    FRONTEND (Web UI)    │ │    BACKEND (FastAPI)    │ │    DATA LAYER          │
│                         │ │                         │ │                         │
│ ├── Login Page          │ │ ├── Authentication      │ │ ├── Redis Cache         │
│ ├── Dashboard           │ │ ├── Signal Generation   │ │ ├── User Data           │
│ ├── Admin Panel         │ │ ├── Monitoring System   │ │ ├── Signal Storage      │
│ ├── Signal Management   │ │ ├── Position Management │ │ ├── History Tracking    │
│ ├── Position Tracking   │ │ ├── User Management     │ │ └── Activity Logging    │
│ └── Settings            │ │ └── API Endpoints       │ │                         │
└─────────────────────────┘ └─────────────────────────┘ └─────────────────────────┘
                                        │
                    ┌───────────────────┼───────────────────┐
                    │                   │                   │
                    ▼                   ▼                   ▼
┌─────────────────────────┐ ┌─────────────────────────┐ ┌─────────────────────────┐
│   EXTERNAL APIS         │ │   AI/ML MODELS          │ │   MONITORING            │
│                         │ │                         │ │                         │
│ ├── CCXT (Crypto)       │ │ ├── FinBERT Sentiment   │ │ ├── Scheduler           │
│ ├── Alpaca (Stocks)     │ │ ├── Technical Indicators│ │ ├── Health Checks       │
│ ├── Google News RSS     │ │ ├── Signal Fusion       │ │ ├── Performance Metrics │
│ └── Reddit API          │ │ └── Risk Management     │ │ └── Error Tracking      │
└─────────────────────────┘ └─────────────────────────┘ └─────────────────────────┘
```

---

## Key Components & Technologies

### Data Sources
- **Crypto**: CCXT library → Binance API
- **Stocks/ETFs**: Alpaca API
- **News**: Google News RSS feeds
- **Social Sentiment**: Reddit API (r/Bitcoin, r/ethereum, r/stocks, etc.)

### AI/ML Components
- **Sentiment Analysis**: FinBERT (ProsusAI/finbert)
- **Technical Analysis**: RSI, EMA, MACD, ATR indicators
- **Signal Fusion**: Weighted combination of technical and sentiment scores

### Storage & Caching
- **Redis**: Signal persistence, user data, activity logging, position tracking
- **User Isolation**: Each user's signals and positions stored separately
- **History Tracking**: Complete audit trail of all signal events

### Security & Authentication
- **JWT Authentication**: Bearer token validation
- **User Context**: All signals and positions tagged with username
- **Role-based Access**: Admin and user permissions
- **Password Hashing**: SHA-256 password storage

### Monitoring & Automation
- **Scheduled Monitoring**: Every 2 minutes automatic signal evaluation
- **Real-time Updates**: Live position PnL calculation
- **Activity Logging**: Complete audit trail of all system activities
- **Health Monitoring**: System status and performance tracking

---

## Example Calculation (Enhanced)

```
Input: BTC/USDT, 1h timeframe
Custom Thresholds: Buy=0.5, Sell=-0.5, Tech=0.6, Sentiment=0.4

1. Technical Analysis:
   - RSI: 45 → RSI Score: (45-50)/50 = -0.1
   - MACD Hist: -200 → MACD Score: -200/1000 = -0.2
   - Technical Score: (-0.1 + -0.2) / 2 = -0.15

2. Enhanced Sentiment Analysis:
   - RSS Headlines: 10 crypto headlines
   - Reddit Posts: 6 posts from r/Bitcoin, r/ethereum
   - Mixed Sample: 16 texts randomly shuffled
   - FinBERT Analysis: -0.25
   - Sentiment Score: -0.25

3. Signal Fusion:
   - Fused Score: (0.6 × -0.15) + (0.4 × -0.25) = -0.09 + -0.10 = -0.19

4. Decision:
   - Fused Score (-0.19) > Sell Threshold (-0.5) → HOLD
   - Confidence: abs(-0.19) = 0.19 (19%)

5. Risk Management:
   - Current Price: $45,000
   - ATR: $1,500
   - No risk levels (HOLD signal)

6. Monitoring:
   - Signal stored with timestamp
   - History event created: "signal_created"
   - Monitoring scheduled for every 2 minutes
   - Future monitoring events: "monitoring_cycle" or "status_changed"
```

---

## Error Handling & Resilience

### Signal Generation Errors
- **Market Data Errors**: Fallback to error message with retry option
- **API Failures**: Network error notifications with graceful degradation
- **Sentiment Analysis Errors**: Fallback to neutral (0.0) with warning
- **Redis Errors**: Log warning, continue without persistence

### Monitoring Errors
- **Data Fetching Errors**: Skip problematic symbols, continue with others
- **Analysis Errors**: Log error, mark signal as error state
- **Redis Connection Errors**: Retry with exponential backoff

### Authentication & Authorization
- **Invalid Credentials**: Clear error message
- **Expired Tokens**: Automatic redirect to login
- **Insufficient Permissions**: 403 Forbidden with explanation
- **Account Deactivation**: Clear message with admin contact

### System Resilience
- **Redis Failures**: Graceful degradation, continue core functionality
- **External API Failures**: Fallback to cached data when available
- **High Load**: Rate limiting and queue management
- **Data Corruption**: Validation and recovery mechanisms

---

## Performance Optimizations

### Caching Strategy
- **Market Data**: Cache for 1 minute to reduce API calls
- **Sentiment Analysis**: Cache results for 5 minutes
- **User Sessions**: Redis-based session storage
- **Configuration**: In-memory config with periodic refresh

### Database Optimization
- **Signal Indexing**: Index by symbol, user, and timestamp
- **History Compression**: Archive old history events
- **Connection Pooling**: Redis connection pool management
- **Batch Operations**: Bulk updates for monitoring cycles

### API Performance
- **Response Compression**: Gzip compression for large responses
- **Pagination**: Limit large result sets
- **Async Processing**: Non-blocking I/O for external API calls
- **Rate Limiting**: Prevent API abuse

---

**Last Updated**: September 6, 2025
**Version**: 2.0
**System**: Trading AI Tips v2.0