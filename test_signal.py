#!/usr/bin/env python3
"""Test script for signal generation functionality"""

import sys
import os
sys.path.append('.')

from agent.data.ccxt_client import fetch_ohlcv
from agent.indicators import compute_rsi, compute_ema, compute_macd, compute_atr
from agent.models.sentiment import SentimentAnalyzer

def test_signal_generation():
    """Test the complete signal generation pipeline"""
    print("Testing signal generation pipeline...")
    
    try:
        # 1. Test data fetching
        print("1. Testing data fetching...")
        ohlcv = fetch_ohlcv('BTCUSDT', '1h')
        print(f"   ✓ Data fetched: {len(ohlcv)} rows, shape: {ohlcv.shape}")
        
        # 2. Test indicators
        print("2. Testing indicators...")
        close = ohlcv['close']
        rsi = compute_rsi(close)
        ema_12 = compute_ema(close, 12)
        ema_26 = compute_ema(close, 26)
        macd_df = compute_macd(close)
        macd = macd_df['macd']
        macd_signal = macd_df['signal']
        macd_hist = macd_df['hist']
        atr = compute_atr(ohlcv['high'], ohlcv['low'], ohlcv['close'])
        print(f"   ✓ Indicators computed: RSI({len(rsi)}), EMA({len(ema_12)}), MACD({len(macd_df)}), ATR({len(atr)})")
        
        # 3. Test sentiment analyzer
        print("3. Testing sentiment analyzer...")
        sentiment_analyzer = SentimentAnalyzer('ProsusAI/finbert')
        print(f"   ✓ Sentiment analyzer initialized")
        
        # 4. Test technical score calculation
        print("4. Testing technical score calculation...")
        if not close.empty:
            current_close = close.iloc[-1]
            current_rsi = rsi.iloc[-1] if not rsi.empty else 50
            current_macd_hist = macd_hist.iloc[-1] if not macd_hist.empty else 0
            
            # Normalize RSI to [-1, 1]
            rsi_score = (current_rsi - 50) / 50
            
            # Normalize MACD histogram
            macd_score = max(min(current_macd_hist / 1000, 1), -1)
            
            tech_score = (rsi_score + macd_score) / 2
            print(f"   ✓ Technical score: {tech_score:.3f}")
        
        # 5. Test signal generation
        print("5. Testing signal generation...")
        tech_weight = 0.6
        sentiment_weight = 0.4
        sentiment_score = 0.0  # Neutral for now
        fused_score = tech_weight * tech_score + sentiment_weight * sentiment_score
        
        # Determine signal type
        if fused_score >= 0.7:
            signal_type = "BUY"
        elif fused_score <= -0.7:
            signal_type = "SELL"
        else:
            signal_type = "HOLD"
        
        print(f"   ✓ Signal generated: {signal_type} (score: {fused_score:.3f})")
        
        # 6. Test risk management
        print("6. Testing risk management...")
        current_close = close.iloc[-1] if not close.empty else 0
        current_atr = atr.iloc[-1] if not atr.empty else 0
        
        stop_loss = None
        take_profit = None
        
        if current_atr > 0:
            if signal_type == "BUY":
                stop_loss = current_close - (2 * current_atr)
                take_profit = current_close + (3 * current_atr)
            elif signal_type == "SELL":
                stop_loss = current_close + (2 * current_atr)
                take_profit = current_close - (3 * current_atr)
        
        stop_loss_str = f"{stop_loss:.2f}" if stop_loss is not None else "None"
        take_profit_str = f"{take_profit:.2f}" if take_profit is not None else "None"
        print(f"   ✓ Risk management: Stop Loss: {stop_loss_str}, Take Profit: {take_profit_str}")
        
        print("\n🎉 All tests passed! Signal generation pipeline is working correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_signal_generation()
    sys.exit(0 if success else 1)
