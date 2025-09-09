# Trading System Flow Diagram - User Tip Generation

## Overview
This diagram shows the complete flow when a user generates trading tips, including all functions called at each step.

## User Tip Generation Flow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           USER GENERATES TIP                                    │
│  User clicks "Generate Tip" button in web interface                            │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 1. WEB INTERFACE (index.html)                                                  │
│    • generateTip() function called                                             │
│    • Collects user parameters (symbol, timeframe, weights, thresholds)         │
│    • Makes POST request to /api/signals/generate                               │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 2. FASTAPI ENDPOINT (web/main.py)                                              │
│    • @app.post("/api/signals/generate")                                        │
│    • verify_token() - Authentication check                                     │
│    • SignalRequest validation                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 3. CONFIGURATION LOADING                                                        │
│    • load_config_from_env() - Loads global config                              │
│    • get_config_value() - Retrieves config parameters                          │
│    • Sets default values from YAML if not provided by user                     │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 4. DATA FETCHING                                                                │
│    • fetch_ohlcv() - Gets historical price data                                │
│    • fetch_sentiment() - Gets sentiment analysis data                          │
│    • Data validation and error handling                                        │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 5. ENHANCED TIP GENERATION (agent/engine.py)                                   │
│    • make_fused_tip() - Main tip generation function                           │
│    • get_config_value() - Configuration access                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 6. ASSET TYPE DETECTION                                                         │
│    • Detects if symbol is crypto or stock                                       │
│    • Uses config.universe.crypto_symbols                                        │
│    • Uses config.universe.stock_symbols                                         │
│    • Sets is_crypto, is_stock flags                                            │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 7. SENTIMENT ENHANCEMENT                                                        │
│    • Applies asset-specific sentiment amplification                            │
│    • config.signals.sentiment_enhancement.crypto_amplification (1.2x)          │
│    • config.signals.sentiment_enhancement.stock_amplification (1.0x)           │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 8. TECHNICAL ANALYSIS (agent/indicators.py)                                    │
│    • score_technical() - Main technical analysis function                      │
│    • calculate_rsi() - RSI calculation                                         │
│    • calculate_ema() - EMA calculation                                         │
│    • calculate_macd() - MACD calculation                                       │
│    • calculate_atr() - ATR calculation                                         │
│    • score_multi_timeframe() - Multi-timeframe analysis                        │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 9. MULTI-TIMEFRAME ANALYSIS                                                    │
│    • Fetches data for multiple timeframes (15m, 1h, 4h)                       │
│    • config.signals.multi_timeframe.timeframes                                 │
│    • config.signals.multi_timeframe.weights                                    │
│    • Combines scores with weighted average                                     │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 10. MARKET REGIME DETECTION                                                     │
│     • Analyzes price trends to determine market regime                         │
│     • BULL, BEAR, or SIDEWAYS market detection                                 │
│     • Uses config.signals.regime_adjustments for regime-specific adjustments   │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 11. VOLATILITY ANALYSIS                                                         │
│     • calculate_volatility() - Calculates market volatility                    │
│     • config.signals.risk_management_by_asset.{asset_type}.volatility_threshold_high │
│     • config.signals.risk_management_by_asset.{asset_type}.volatility_threshold_low  │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 12. RISK MANAGEMENT CALCULATION                                                 │
│     • Base multipliers from config.signals.risk_management_by_asset.{asset_type} │
│     • Regime adjustments from config.signals.regime_adjustments                 │
│     • Volatility adjustments from config.signals.volatility_adjustments         │
│     • Final stop loss and take profit calculations                              │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 13. SIGNAL FUSION                                                               │
│     • Combines technical and sentiment scores                                   │
│     • config.signals.weights.technical_weight (0.5)                            │
│     • config.signals.weights.sentiment_weight (0.5)                            │
│     • config.signals.thresholds.buy_threshold (0.2)                            │
│     • config.signals.thresholds.sell_threshold (-0.2)                          │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 14. TIP OBJECT CREATION                                                         │
│     • Creates Tip object with suggestion (BUY/SELL/HOLD)                       │
│     • Populates meta dictionary with all analysis results                      │
│     • Includes asset_type, market_regime, volatility, risk metrics             │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 15. SIGNAL STORAGE (web/main.py)                                               │
│     • Creates TradingSignal object                                              │
│     • Stores in Redis cache                                                     │
│     • Adds to user's signal history                                             │
│     • Records signal generation event                                           │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 16. RESPONSE TO USER                                                            │
│     • Returns TradingSignal with all analysis results                          │
│     • Includes confidence, scores, stop loss, take profit                      │
│     • Enhanced reasoning with all analysis details                             │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 17. WEB INTERFACE UPDATE                                                        │
│     • Updates UI with new signal                                                │
│     • Shows signal type, confidence, reasoning                                  │
│     • Displays stop loss and take profit levels                                 │
│     • Updates signal history table                                              │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Key Functions Called at Each Step

### 1. Web Interface
- `generateTip()` - Main UI function
- `fetch()` - API call to backend

### 2. FastAPI Endpoint
- `verify_token()` - Authentication
- `SignalRequest` - Request validation

### 3. Configuration
- `load_config_from_env()` - Load global config
- `get_config_value()` - Access config parameters

### 4. Data Fetching
- `fetch_ohlcv()` - Price data
- `fetch_sentiment()` - Sentiment data

### 5. Tip Generation
- `make_fused_tip()` - Main generation function
- `get_config_value()` - Config access

### 6. Asset Detection
- Asset type detection logic
- Config-based asset classification

### 7. Sentiment Enhancement
- Sentiment amplification logic
- Asset-specific multipliers

### 8. Technical Analysis
- `score_technical()` - Main technical function
- `calculate_rsi()` - RSI indicator
- `calculate_ema()` - EMA indicator
- `calculate_macd()` - MACD indicator
- `calculate_atr()` - ATR indicator

### 9. Multi-timeframe
- `score_multi_timeframe()` - Multi-timeframe analysis
- `fetch_ohlcv()` - Data for multiple timeframes

### 10. Market Regime
- Market regime detection logic
- Regime-based adjustments

### 11. Volatility
- `calculate_volatility()` - Volatility calculation
- Volatility-based adjustments

### 12. Risk Management
- Risk calculation logic
- Stop loss and take profit calculation

### 13. Signal Fusion
- Score combination logic
- Threshold-based signal generation

### 14. Tip Creation
- `Tip` object creation
- Meta data population

### 15. Storage
- Redis storage
- Signal history update
- Event recording

### 16. Response
- `TradingSignal` object creation
- Response formatting

### 17. UI Update
- UI update functions
- Display formatting

## Configuration Parameters Used

### Asset Types
- `config.universe.crypto_symbols`
- `config.universe.stock_symbols`

### Sentiment Enhancement
- `config.signals.sentiment_enhancement.crypto_amplification`
- `config.signals.sentiment_enhancement.stock_amplification`

### Risk Management
- `config.signals.risk_management_by_asset.{asset_type}.base_stop_multiplier`
- `config.signals.risk_management_by_asset.{asset_type}.base_tp_multiplier`
- `config.signals.risk_management_by_asset.{asset_type}.volatility_threshold_high`
- `config.signals.risk_management_by_asset.{asset_type}.volatility_threshold_low`

### Regime Adjustments
- `config.signals.regime_adjustments.bull_market.buy_stop_adjustment`
- `config.signals.regime_adjustments.bull_market.sell_stop_adjustment`
- `config.signals.regime_adjustments.bear_market.buy_stop_adjustment`
- `config.signals.regime_adjustments.bear_market.sell_stop_adjustment`

### Volatility Adjustments
- `config.signals.volatility_adjustments.high_volatility_multiplier`
- `config.signals.volatility_adjustments.low_volatility_multiplier`
- `config.signals.volatility_adjustments.high_volatility_tp_adjustment`

### Multi-timeframe
- `config.signals.multi_timeframe.timeframes`
- `config.signals.multi_timeframe.weights`
- `config.signals.multi_timeframe.data_points`

### Signal Thresholds
- `config.signals.thresholds.buy_threshold`
- `config.signals.thresholds.sell_threshold`
- `config.signals.weights.technical_weight`
- `config.signals.weights.sentiment_weight`

## Summary

The trading system is now fully configuration-driven with no hardcoded values. Every parameter is loaded from the YAML configuration files, making the system:

1. **Universal** - Works for all asset types (crypto, forex, stocks)
2. **Configurable** - All parameters adjustable via YAML
3. **Enhanced** - Advanced analysis for all symbols
4. **Flexible** - Easy to modify without code changes
5. **Scalable** - Can handle any trading symbol

The system automatically detects asset types, applies appropriate analysis methods, and generates comprehensive trading signals with risk management parameters tailored to each asset class.
