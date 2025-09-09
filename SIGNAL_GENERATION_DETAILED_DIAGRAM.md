# Signal Generation - Detailed Method Calls & Data Flow (UPDATED)

## Overview
This diagram shows the complete signal generation process with all method calls, data calculations, configuration usage, and the new improvements including enhanced logging, user input priority system, and API integration.

## Complete Signal Generation Flow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           SIGNAL GENERATION ENTRY POINT                        │
│  User clicks "Generate Tip" → Web Interface → FastAPI → make_fused_tip()       │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 0. API ENDPOINT - /api/signals/generate (web/main.py:1805)                     │
│    • Authentication: verify_token()                                             │
│    • Request validation: SignalRequest model                                    │
│    • Data fetching: fetch_ohlcv() or fetch_alpaca_ohlcv()                      │
│    • Sentiment analysis: SentimentAnalyzer                                      │
│    • Enhanced logging: Configuration verification                               │
│    • Calls: make_fused_tip() with all parameters                                │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 0A. CONFIGURATION VERIFICATION & USER INPUT PRIORITY (web/main.py:1945-1976)   │
│    • DEBUG: config.signals.weights type and content                            │
│    • DEBUG: config.signals.thresholds type and content                         │
│    • User Input Parameters logging:                                            │
│      - technical_weight (user): request.technical_weight                       │
│      - sentiment_weight (user): request.sentiment_weight                       │
│      - buy_threshold (user): request.buy_threshold                             │
│      - sell_threshold (user): request.sell_threshold                           │
│    • Config Default Parameters logging:                                        │
│      - technical_weight (config): config.signals.weights["technical_weight"]   │
│      - sentiment_weight (config): config.signals.weights["sentiment_weight"]   │
│      - buy_threshold (config): config.thresholds.buy_threshold                 │
│      - sell_threshold (config): config.thresholds.sell_threshold               │
│    • Final Applied Parameters with source indication:                          │
│      - tech_weight: value (user/config)                                        │
│      - sentiment_weight: value (user/config)                                   │
│      - buy_threshold: value (user/config)                                      │
│      - sell_threshold: value (user/config)                                     │
│    • Asset Configuration logging:                                              │
│      - crypto_symbols: config.universe.crypto_symbols                          │
│      - stock_symbols: config.universe.stock_symbols                            │
│      - crypto_amplification: config.signals.sentiment_enhancement["crypto_amplification"] │
│      - stock_amplification: config.signals.sentiment_enhancement["stock_amplification"] │
│    • Multi-timeframe Configuration logging:                                    │
│      - enabled: config.signals.multi_timeframe["enabled"]                      │
│      - timeframes: config.signals.multi_timeframe["timeframes"]                │
│      - weights: config.signals.multi_timeframe["weights"]                      │
│    • Risk Management Configuration logging:                                    │
│      - crypto base_stop_multiplier: config.signals.risk_management_by_asset["crypto"]["base_stop_multiplier"] │
│      - crypto base_tp_multiplier: config.signals.risk_management_by_asset["crypto"]["base_tp_multiplier"] │
│      - stock base_stop_multiplier: config.signals.risk_management_by_asset["stock"]["base_stop_multiplier"] │
│      - stock base_tp_multiplier: config.signals.risk_management_by_asset["stock"]["base_tp_multiplier"] │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 1. make_fused_tip() - MAIN FUNCTION (agent/engine.py:82) ⭐ CORE OF IMPROVEMENTS ⭐ │
│    Parameters: symbol, timeframe, ohlcv, sentiment_score, w_tech, w_sent,      │
│                buy_th, sell_th, rsi_oversold, rsi_overbought, use_multi_timeframe, config │
│    • w_tech: user input OR config.signals.weights["technical_weight"]          │
│    • w_sent: user input OR config.signals.weights["sentiment_weight"]          │
│    • buy_th: user input OR config.thresholds.buy_threshold                     │
│    • sell_th: user input OR config.thresholds.sell_threshold                   │
│    ⚠️  THIS IS WHERE ALL THE NEW IMPROVED METHODS ARE CALLED! ⚠️                │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 2. ASSET TYPE DETECTION (agent/engine.py:83-93)                                │
│    • is_crypto = "/" in symbol                                                 │
│    • get_config_value(config, 'universe.crypto_symbols') → crypto_symbols      │
│    • get_config_value(config, 'universe.stock_symbols') → stock_symbols        │
│    • is_crypto_only = symbol.upper() in [s.upper() for s in crypto_symbols]   │
│    • is_stock = symbol.upper() in [s.upper() for s in stock_symbols]          │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 3. TECHNICAL ANALYSIS BRANCH (agent/engine.py:95-104)                          │
│    IF use_multi_timeframe:                                                     │
│      • from .indicators import score_multi_timeframe                           │
│      • tech = score_multi_timeframe(symbol, timeframe, config)                 │
│    ELSE:                                                                        │
│      • from .indicators import score_technical                                 │
│      • tech = score_technical(ohlcv, rsi_oversold, rsi_overbought)            │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 4A. score_multi_timeframe() - MULTI-TIMEFRAME ANALYSIS (agent/indicators.py:105) │
│    • get_config_value(config, 'signals.multi_timeframe.timeframes') → timeframes_list │
│    • get_config_value(config, 'signals.multi_timeframe.weights') → weights     │
│    • get_config_value(config, 'signals.multi_timeframe.data_points') → data_points │
│    • FOR each timeframe in timeframes_list:                                    │
│      - IF "/" in symbol: from ..data.ccxt_client import fetch_ohlcv            │
│      - ELSE: from ..data.alpaca_client import fetch_ohlcv                      │
│      - timeframes[tf] = fetch_func(symbol, tf, data_points)                    │
│    • FOR each tf, df in timeframes.items():                                    │
│      - scores[tf] = score_technical(df)                                        │
│    • multi_tf_score = sum(scores[tf] * weights[tf] for tf in timeframes)       │
│    • RETURN max(-1.0, min(1.0, multi_tf_score))                               │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 4B. score_technical() - SINGLE TIMEFRAME ANALYSIS (agent/indicators.py:66)     │
│    • compute_rsi(ohlcv["close"]) → rsi                                         │
│    • IF rsi <= rsi_oversold: rsi_score = 1.0                                   │
│    • ELIF rsi >= rsi_overbought: rsi_score = -1.0                              │
│    • ELSE: rsi_score = 1 - (rsi - rsi_oversold) * (2 / (rsi_overbought - rsi_oversold)) │
│    • compute_macd(ohlcv["close"]) → macd_df                                    │
│    • hist = macd_df["hist"].iloc[-50:]                                         │
│    • std = float(hist.std() or 1.0)                                            │
│    • macd_score = float((hist.iloc[-1] / (std * 2)))                           │
│    • macd_score = max(-1.0, min(1.0, macd_score))                              │
│    • volume = ohlcv["volume"].iloc[-20:]                                       │
│    • volume_ma = volume.rolling(10).mean().iloc[-1]                            │
│    • current_volume = volume.iloc[-1]                                          │
│    • volume_ratio = current_volume / volume_ma if volume_ma > 0 else 1.0       │
│    • volume_boost = min(0.3, max(-0.3, (volume_ratio - 1.0) * 0.5))           │
│    • base_score = (rsi_score + macd_score) / 2                                 │
│    • volume_confirmed_score = base_score + volume_boost                        │
│    • RETURN max(-1.0, min(1.0, volume_confirmed_score))                        │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 5. SENTIMENT ENHANCEMENT (agent/engine.py:106-118)                             │
│    • IF is_crypto_only:                                                        │
│      - amplification = get_config_value(config, 'signals.sentiment_enhancement.crypto_amplification', 1.2) │
│      - enhanced_sentiment = sentiment_score * amplification                    │
│    • ELIF is_stock:                                                            │
│      - amplification = get_config_value(config, 'signals.sentiment_enhancement.stock_amplification', 1.0) │
│      - enhanced_sentiment = sentiment_score * amplification                    │
│    • ELSE:                                                                      │
│      - amplification = get_config_value(config, 'signals.sentiment_enhancement.stock_amplification', 1.0) │
│      - enhanced_sentiment = sentiment_score * amplification                    │
│    • enhanced_sentiment = max(-1.0, min(1.0, enhanced_sentiment))             │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 6. SIGNAL FUSION (agent/engine.py:120-128)                                     │
│    • score = w_tech * tech + w_sent * enhanced_sentiment                       │
│    • IF score >= buy_th: suggestion = "BUY"                                    │
│    • ELIF score <= sell_th: suggestion = "SELL"                                │
│    • ELSE: suggestion = "NEUTRAL"                                              │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 7. MARKET REGIME DETECTION & FILTERING (agent/engine.py:130-139)               │
│    • market_regime = "N/A"                                                     │
│    • IF use_multi_timeframe:                                                   │
│      - from .indicators import detect_market_regime, apply_regime_filter       │
│      - market_regime = detect_market_regime(ohlcv)                             │
│      - suggestion = apply_regime_filter(tech, market_regime, suggestion)       │
│    • ELSE: market_regime = "UNKNOWN"                                           │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 7A. detect_market_regime() - REGIME DETECTION (agent/indicators.py:149)        │
│    • ema_50 = ohlcv["close"].ewm(span=50).mean().iloc[-1]                      │
│    • ema_200 = ohlcv["close"].ewm(span=200).mean().iloc[-1]                    │
│    • diff_pct = (ema_50 - ema_200) / ema_200 * 100                             │
│    • IF diff_pct > 2.0: RETURN "BULL"                                          │
│    • ELIF diff_pct < -2.0: RETURN "BEAR"                                       │
│    • ELSE: RETURN "SIDEWAYS"                                                   │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 7B. apply_regime_filter() - REGIME FILTERING (agent/indicators.py:177)         │
│    • IF market_regime == "BULL":                                               │
│      - IF suggestion == "SELL": RETURN "NEUTRAL"                               │
│      - ELIF suggestion == "BUY": RETURN "BUY"                                  │
│    • ELIF market_regime == "BEAR":                                             │
│      - IF suggestion == "BUY": RETURN "NEUTRAL"                                │
│      - ELIF suggestion == "SELL": RETURN "SELL"                                │
│    • ELIF market_regime == "SIDEWAYS":                                         │
│      - IF abs(tech_score) < 0.3: RETURN "NEUTRAL"                              │
│    • RETURN suggestion                                                          │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 8. RISK MANAGEMENT CALCULATION (agent/engine.py:140-189)                       │
│    • compute_atr(ohlcv["high"], ohlcv["low"], ohlcv["close"]) → atr            │
│    • close_val = ohlcv["close"].ffill().iloc[-1]                               │
│    • close = float(close_val) if math.isfinite(float(close_val)) else 1.0      │
│    • IF not math.isfinite(atr) or atr <= 0: atr = max(1e-6, 0.01 * close)     │
│    • volatility = ohlcv["close"].pct_change().std() * 100                      │
│    • IF is_crypto_only:                                                        │
│      - asset_type = 'crypto'                                                   │
│      - base_stop_multiplier = get_config_value(config, f'signals.risk_management_by_asset.{asset_type}.base_stop_multiplier', 2.0) │
│      - base_tp_multiplier = get_config_value(config, f'signals.risk_management_by_asset.{asset_type}.base_tp_multiplier', 2.5) │
│      - volatility_threshold_high = get_config_value(config, f'signals.risk_management_by_asset.{asset_type}.volatility_threshold_high', 3.0) │
│      - volatility_threshold_low = get_config_value(config, f'signals.risk_management_by_asset.{asset_type}.volatility_threshold_low', 1.0) │
│    • ELSE:                                                                      │
│      - asset_type = 'stock'                                                    │
│      - base_stop_multiplier = get_config_value(config, f'signals.risk_management_by_asset.{asset_type}.base_stop_multiplier', 1.5) │
│      - base_tp_multiplier = get_config_value(config, f'signals.risk_management_by_asset.{asset_type}.base_tp_multiplier', 2.0) │
│      - volatility_threshold_high = get_config_value(config, f'signals.risk_management_by_asset.{asset_type}.volatility_threshold_high', 2.0) │
│      - volatility_threshold_low = get_config_value(config, f'signals.risk_management_by_asset.{asset_type}.volatility_threshold_low', 0.5) │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 9. REGIME ADJUSTMENTS (agent/engine.py:165-178)                                │
│    • IF market_regime == "BULL":                                               │
│      - stop_adj = get_config_value(config, f'signals.regime_adjustments.bull_market.{"buy" if suggestion == "BUY" else "sell"}_stop_adjustment', 0.8 if suggestion == "BUY" else 1.3) │
│      - tp_adj = get_config_value(config, f'signals.regime_adjustments.bull_market.{"buy" if suggestion == "BUY" else "sell"}_tp_adjustment', 1.2 if suggestion == "BUY" else 0.8) │
│      - stop_multiplier = base_stop_multiplier * stop_adj                       │
│      - tp_multiplier = base_tp_multiplier * tp_adj                             │
│    • ELIF market_regime == "BEAR":                                             │
│      - stop_adj = get_config_value(config, f'signals.regime_adjustments.bear_market.{"buy" if suggestion == "BUY" else "sell"}_stop_adjustment', 1.3 if suggestion == "BUY" else 0.8) │
│      - tp_adj = get_config_value(config, f'signals.regime_adjustments.bear_market.{"buy" if suggestion == "BUY" else "sell"}_tp_adjustment', 0.8 if suggestion == "BUY" else 1.2) │
│      - stop_multiplier = base_stop_multiplier * stop_adj                       │
│      - tp_multiplier = base_tp_multiplier * tp_adj                             │
│    • ELSE:  # SIDEWAYS                                                          │
│      - stop_multiplier = base_stop_multiplier                                  │
│      - tp_multiplier = base_tp_multiplier                                      │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 10. VOLATILITY ADJUSTMENTS (agent/engine.py:180-189)                           │
│    • IF volatility > volatility_threshold_high:  # High volatility             │
│      - high_vol_mult = get_config_value(config, 'signals.volatility_adjustments.high_volatility_multiplier', 1.5) │
│      - high_vol_tp_adj = get_config_value(config, 'signals.volatility_adjustments.high_volatility_tp_adjustment', 1.2) │
│      - stop_multiplier *= high_vol_mult                                         │
│      - tp_multiplier *= high_vol_tp_adj                                         │
│    • ELIF volatility < volatility_threshold_low:  # Low volatility             │
│      - low_vol_mult = get_config_value(config, 'signals.volatility_adjustments.low_volatility_multiplier', 0.8) │
│      - stop_multiplier *= low_vol_mult                                          │
│      - tp_multiplier *= low_vol_mult                                            │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 11. STOP LOSS & TAKE PROFIT CALCULATION (agent/engine.py:191-200)              │
│    • IF suggestion == "BUY":                                                   │
│      - stop = close - (stop_multiplier * atr)                                  │
│      - tp = close + (tp_multiplier * atr)                                      │
│    • ELIF suggestion == "SELL":                                                │
│      - stop = close + (stop_multiplier * atr)                                  │
│      - tp = close - (tp_multiplier * atr)                                      │
│    • ELSE:  # NEUTRAL                                                           │
│      - stop = close                                                             │
│      - tp = close                                                               │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 12. TIP OBJECT CREATION (agent/engine.py:201-233)                              │
│    • RETURN Tip(                                                               │
│        symbol=symbol,                                                          │
│        timeframe=timeframe,                                                    │
│        indicator="FUSED(tech+sent)" + ("+MTF" if use_multi_timeframe else ""), │
│        value=float(score),                                                     │
│        suggestion=suggestion,                                                  │
│        meta={                                                                  │
│          "tech": float(tech),                                                  │
│          "sent": float(sentiment_score),                                       │
│          "enhanced_sentiment": float(enhanced_sentiment),                      │
│          "atr": float(atr),                                                    │
│          "close": close,                                                       │
│          "stop": stop,                                                         │
│          "tp": tp,                                                             │
│          "multi_timeframe": use_multi_timeframe,                               │
│          "volume_boost": "included" if use_multi_timeframe else "single_tf",   │
│          "market_regime": market_regime,                                       │
│          "regime_filtered": use_multi_timeframe,                               │
│          "volatility": float(volatility),                                      │
│          "stop_multiplier": float(stop_multiplier),                            │
│          "tp_multiplier": float(tp_multiplier),                                │
│          "risk_reward_ratio": float((tp - close) / (close - stop)) if suggestion == "BUY" and stop < close else float((close - tp) / (stop - close)) if suggestion == "SELL" and stop > close else 0.0, │
│          "position_size_recommendation": "conservative" if volatility > volatility_threshold_high else "moderate" if volatility > volatility_threshold_low else "aggressive", │
│          "sentiment_enhanced": is_crypto_only,                                 │
│          "tech_weight": float(w_tech),                                         │
│          "sentiment_weight": float(w_sent),                                    │
│          "asset_type": "crypto" if is_crypto_only else "stock",                │
│          "is_crypto": is_crypto_only,                                          │
│          "is_stock": is_stock,                                                 │
│          "base_stop_multiplier": float(base_stop_multiplier),                  │
│          "base_tp_multiplier": float(base_tp_multiplier),                      │
│        }                                                                       │
│      )                                                                         │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 13. API RESPONSE CREATION (web/main.py:2014-2040)                              │
│    • TradingSignal object creation:                                            │
│      - signal_type: tip.suggestion                                             │
│      - confidence: abs(tip.value)                                              │
│      - technical_score: tip.meta.get("tech", 0.0)                             │
│      - sentiment_score: tip.meta.get("sent", 0.0)                             │
│      - fused_score: tip.value                                                  │
│      - stop_loss: tip.meta.get("stop", 0.0)                                   │
│      - take_profit: tip.meta.get("tp", 0.0)                                   │
│      - applied_buy_threshold: buy_threshold                                    │
│      - applied_sell_threshold: sell_threshold                                  │
│      - applied_tech_weight: tech_weight                                        │
│      - applied_sentiment_weight: sentiment_weight                              │
│      - reasoning: Enhanced Analysis summary                                    │
│    • Redis storage: Store signal for user                                      │
│    • Activity logging: Log signal generation activity                          │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Key Technical Indicators Used

### 1. RSI (Relative Strength Index)
- **Function**: `compute_rsi(close, period=14)`
- **Calculation**: 
  - delta = close.diff()
  - gain = delta.clip(lower=0)
  - loss = -delta.clip(upper=0)
  - avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
  - avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
  - rs = avg_gain / avg_loss.replace(0, pd.NA)
  - rsi = 100 - (100 / (1 + rs))
- **Scoring**: Maps RSI to [-1, 1] using configurable thresholds

### 2. MACD (Moving Average Convergence Divergence)
- **Function**: `compute_macd(close, fast=12, slow=26, signal=9)`
- **Calculation**:
  - fast_ema = compute_ema(close, fast)
  - slow_ema = compute_ema(close, slow)
  - macd = fast_ema - slow_ema
  - signal_line = compute_ema(macd, signal)
  - hist = macd - signal_line
- **Scoring**: Normalizes histogram by rolling standard deviation

### 3. ATR (Average True Range)
- **Function**: `compute_atr(high, low, close, period=14)`
- **Calculation**:
  - prev_close = close.shift(1)
  - tr = max(high-low, |high-prev_close|, |low-prev_close|)
  - atr = tr.ewm(alpha=1/period, adjust=False).mean()
- **Usage**: Used for stop loss and take profit calculations

### 4. EMA (Exponential Moving Average)
- **Function**: `compute_ema(series, span)`
- **Calculation**: series.ewm(span=span, adjust=False).mean()
- **Usage**: Used in MACD and market regime detection

## Configuration Parameters Used

### Asset Type Detection
- `config.universe.crypto_symbols` - List of crypto symbols
- `config.universe.stock_symbols` - List of stock symbols

### Sentiment Enhancement
- `config.signals.sentiment_enhancement.crypto_amplification` - Crypto sentiment multiplier (1.2)
- `config.signals.sentiment_enhancement.stock_amplification` - Stock sentiment multiplier (1.0)

### Multi-timeframe Analysis
- `config.signals.multi_timeframe.timeframes` - List of timeframes ["15m", "1h", "4h"]
- `config.signals.multi_timeframe.weights` - Weights for each timeframe {"15m": 0.2, "1h": 0.5, "4h": 0.3}
- `config.signals.multi_timeframe.data_points` - Number of data points to fetch (50)

### Risk Management by Asset Type
- `config.signals.risk_management_by_asset.crypto.base_stop_multiplier` - Crypto stop multiplier (2.0)
- `config.signals.risk_management_by_asset.crypto.base_tp_multiplier` - Crypto TP multiplier (2.5)
- `config.signals.risk_management_by_asset.crypto.volatility_threshold_high` - Crypto high vol threshold (3.0)
- `config.signals.risk_management_by_asset.crypto.volatility_threshold_low` - Crypto low vol threshold (1.0)
- `config.signals.risk_management_by_asset.stock.base_stop_multiplier` - Stock stop multiplier (1.5)
- `config.signals.risk_management_by_asset.stock.base_tp_multiplier` - Stock TP multiplier (2.0)
- `config.signals.risk_management_by_asset.stock.volatility_threshold_high` - Stock high vol threshold (2.0)
- `config.signals.risk_management_by_asset.stock.volatility_threshold_low` - Stock low vol threshold (0.5)

### Market Regime Adjustments
- `config.signals.regime_adjustments.bull_market.buy_stop_adjustment` - Bull market buy stop adjustment (0.8)
- `config.signals.regime_adjustments.bull_market.sell_stop_adjustment` - Bull market sell stop adjustment (1.3)
- `config.signals.regime_adjustments.bull_market.buy_tp_adjustment` - Bull market buy TP adjustment (1.2)
- `config.signals.regime_adjustments.bull_market.sell_tp_adjustment` - Bull market sell TP adjustment (0.8)
- `config.signals.regime_adjustments.bear_market.buy_stop_adjustment` - Bear market buy stop adjustment (1.3)
- `config.signals.regime_adjustments.bear_market.sell_stop_adjustment` - Bear market sell stop adjustment (0.8)
- `config.signals.regime_adjustments.bear_market.buy_tp_adjustment` - Bear market buy TP adjustment (0.8)
- `config.signals.regime_adjustments.bear_market.sell_tp_adjustment` - Bear market sell TP adjustment (1.2)

### Volatility Adjustments
- `config.signals.volatility_adjustments.high_volatility_multiplier` - High volatility multiplier (1.5)
- `config.signals.volatility_adjustments.low_volatility_multiplier` - Low volatility multiplier (0.8)
- `config.signals.volatility_adjustments.high_volatility_tp_adjustment` - High volatility TP adjustment (1.2)

## Data Flow Summary

1. **Input**: Symbol, timeframe, OHLCV data, sentiment score, weights, thresholds
2. **Asset Detection**: Determines if symbol is crypto or stock
3. **Technical Analysis**: RSI + MACD + Volume analysis (single or multi-timeframe)
4. **Sentiment Enhancement**: Applies asset-specific amplification
5. **Signal Fusion**: Combines technical and sentiment scores
6. **Regime Detection**: EMA(50) vs EMA(200) to determine market regime
7. **Regime Filtering**: Adjusts signals based on market regime
8. **Risk Management**: Calculates ATR-based stop loss and take profit
9. **Regime Adjustments**: Modifies risk parameters based on market regime
10. **Volatility Adjustments**: Further modifies risk parameters based on volatility
11. **Output**: Tip object with signal, confidence, stop loss, take profit, and metadata

## Key Features

### Core Analysis Features
- **Multi-timeframe Analysis**: Analyzes 15m, 1h, and 4h timeframes with weighted scoring
- **Market Regime Detection**: Uses EMA(50) vs EMA(200) to detect bull/bear/sideways markets
- **Regime Filtering**: Prevents counter-trend signals in strong trending markets
- **Asset-Specific Parameters**: Different risk management for crypto vs stocks
- **Volatility Adjustment**: Adjusts risk parameters based on market volatility
- **Volume Confirmation**: Uses volume analysis to confirm technical signals

### Configuration & Customization
- **Configuration-Driven**: All parameters configurable via YAML files
- **User Input Priority**: User parameters override config defaults when provided
- **Asset Type Detection**: Automatic detection of crypto vs stock symbols
- **Sentiment Enhancement**: Asset-specific sentiment amplification (crypto: 1.2x, stock: 1.0x)

### API & Integration Features
- **RESTful API**: FastAPI-based signal generation endpoint
- **Authentication**: JWT token-based authentication
- **Request Validation**: Pydantic model validation for all inputs
- **Enhanced Logging**: Comprehensive configuration verification and parameter tracking
- **Response Transparency**: Shows which parameters were actually applied (user vs config)

### Monitoring & Debugging
- **Configuration Verification**: Detailed logging of all parameter sources
- **Parameter Tracking**: Complete visibility into user input vs config defaults
- **Activity Logging**: Redis-based activity tracking and signal storage
- **Error Handling**: Robust error handling with detailed traceback logging
- **Debug Information**: Type checking and content logging for configuration objects

### Production Features
- **Docker Integration**: Containerized deployment with health checks
- **Redis Integration**: Signal storage and user activity tracking
- **Multi-Asset Support**: Crypto and stock symbol support (forex removed)
- **Real-time Data**: Live market data integration via CCXT and Alpaca
- **Scalable Architecture**: Microservice-ready with proper separation of concerns

## Recent Improvements Made

### 1. Enhanced Configuration System
- **Fixed Configuration Access**: Resolved mixed dictionary/object access patterns
- **Standardized Access**: Proper dataclass attribute access for thresholds
- **Dictionary Access**: Proper dictionary access for weights and other config sections
- **Error Resolution**: Eliminated `'dict' object has no attribute 'technical_weight'` errors

### 2. User Input Priority System
- **Parameter Override**: User input parameters now properly override config defaults
- **Source Tracking**: Complete visibility into which parameters come from user vs config
- **Flexible Configuration**: System respects user preferences while maintaining sensible defaults

### 3. Enhanced Logging & Verification
- **Configuration Verification**: Detailed logging of all parameter sources
- **Debug Information**: Type checking and content logging for configuration objects
- **Parameter Tracking**: Shows exactly which values were applied and their source
- **Asset Configuration**: Logs crypto/stock symbol lists and amplification factors

### 4. API Integration Improvements
- **Response Enhancement**: API responses now include applied parameter tracking
- **Transparency**: Shows which parameters were actually used (user vs config)
- **Error Handling**: Robust error handling with detailed traceback logging
- **Activity Logging**: Redis-based signal storage and user activity tracking

### 5. Asset Type Specialization
- **Crypto Enhancement**: 1.2x sentiment amplification for crypto symbols
- **Stock Standardization**: 1.0x sentiment amplification for stock symbols
- **Asset-Specific Risk**: Different stop/take-profit multipliers for crypto vs stocks
- **Symbol Detection**: Automatic detection using universe configuration

### 6. Production Readiness
- **Docker Integration**: Proper containerized deployment with health checks
- **Configuration-Driven**: 100% configurable system without hardcoded values
- **Error Recovery**: Robust error handling and graceful degradation
- **Monitoring**: Comprehensive logging for production monitoring and debugging

## 🔄 **IMPORTANT: Flow Clarification**

**The correct flow is:**
```
User clicks "Generate Tip" → Web Interface → FastAPI → make_fused_tip()
```

**NOT:**
```
User clicks "Generate Tip" → Web Interface → FastAPI → /api/signals/generate
```

**Explanation:**
- `/api/signals/generate` is the **API endpoint URL**
- `make_fused_tip()` is the **actual function** that contains all the improvements
- The API endpoint calls `make_fused_tip()` internally
- All the new improved methods (multi-timeframe, regime detection, asset-specific logic) are inside `make_fused_tip()`

**So `make_fused_tip()` IS the core of all our improvements!** 🎯
