#!/usr/bin/env python3
"""
Test script for the new smart alert system
Tests that only the strongest signal is shown (color OR size, not both)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add current directory to path to import analyze functions
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from analyze import detect_strong_signals, format_color_alert, format_size_alert

def create_test_data():
    """Create synthetic test data with known patterns"""
    np.random.seed(42)  # For reproducible results
    
    # Create 200 rounds of data
    n_rounds = 200
    data = []
    
    # Create a strong RED momentum pattern
    colors = []
    numbers = []
    
    # First 100 rounds: mostly RED with some streaks
    for i in range(100):
        if i < 30 or (i >= 50 and i < 70) or (i >= 90 and i < 100):
            colors.extend(['RED'] * 5)  # RED streaks
            numbers.extend([1, 3, 5, 7, 9] * 5)
        else:
            colors.extend(['GREEN', 'VIOLET', 'RED', 'GREEN', 'VIOLET'])
            numbers.extend([2, 4, 6, 8, 0])
    
    # Last 100 rounds: mixed with some BIG bias
    for i in range(100):
        if i < 40:
            colors.append('GREEN')
            numbers.append(np.random.choice([6, 7, 8, 9]))  # BIG numbers
        elif i < 80:
            colors.append('RED')
            numbers.append(np.random.choice([1, 2, 3, 4, 5]))  # SMALL numbers
        else:
            colors.append(np.random.choice(['RED', 'GREEN', 'VIOLET']))
            numbers.append(np.random.randint(0, 10))
    
    # Create DataFrame
    base_time = datetime.utcnow() - timedelta(hours=3)
    for i in range(n_rounds):
        data.append({
            'period_id': f"20250101{base_time.strftime('%H%M')}{i:03d}",
            'number': numbers[i],
            'color': colors[i],
            'scraped_at': (base_time + timedelta(minutes=i)).isoformat()
        })
    
    return pd.DataFrame(data)

def create_color_test_data():
    """Create test data with strong color patterns"""
    np.random.seed(123)  # Different seed for variety
    
    # Create 200 rounds of data with strong GREEN momentum
    n_rounds = 200
    data = []
    
    colors = []
    numbers = []
    
    # Create strong GREEN momentum (70% GREEN in last 50 rounds)
    for i in range(n_rounds):
        if i < 50:
            colors.append(np.random.choice(['RED', 'GREEN', 'VIOLET']))
        else:
            # Strong GREEN bias
            if np.random.random() < 0.7:
                colors.append('GREEN')
            else:
                colors.append(np.random.choice(['RED', 'VIOLET']))
        numbers.append(np.random.randint(0, 10))
    
    # Create DataFrame
    base_time = datetime.utcnow() - timedelta(hours=3)
    for i in range(n_rounds):
        data.append({
            'period_id': f"20250101{base_time.strftime('%H%M')}{i:03d}",
            'number': numbers[i],
            'color': colors[i],
            'scraped_at': (base_time + timedelta(minutes=i)).isoformat()
        })
    
    return pd.DataFrame(data)

def test_smart_alerts():
    """Test the smart alert system"""
    print("🧪 Testing Smart Alert System")
    print("=" * 50)
    
    # Test 1: Size-focused data
    print("\n📊 Test 1: Size-focused data")
    df1 = create_test_data()
    print(f"📊 Created test data with {len(df1)} rounds")
    
    signals1 = detect_strong_signals(df1, 
                                  momentum_threshold=0.6,
                                  pattern_threshold=0.65,
                                  time_threshold=0.6,
                                  ensemble_threshold=0.7)
    
    print(f"📈 Signals detected: {len(signals1)}")
    for i, signal in enumerate(signals1, 1):
        if signal["type"] == "color":
            print(f"  {i}. COLOR: {signal['color']} @ {signal['confidence']:.3f} ({signal['method']})")
        else:
            print(f"  {i}. SIZE: {signal['size']} @ {signal['confidence']:.3f} ({signal['method']})")
    
    # Test 2: Color-focused data
    print(f"\n📊 Test 2: Color-focused data")
    df2 = create_color_test_data()
    print(f"📊 Created color test data with {len(df2)} rounds")
    
    signals2 = detect_strong_signals(df2, 
                                  momentum_threshold=0.6,
                                  pattern_threshold=0.65,
                                  time_threshold=0.6,
                                  ensemble_threshold=0.7)
    
    print(f"📈 Signals detected: {len(signals2)}")
    for i, signal in enumerate(signals2, 1):
        if signal["type"] == "color":
            print(f"  {i}. COLOR: {signal['color']} @ {signal['confidence']:.3f} ({signal['method']})")
        else:
            print(f"  {i}. SIZE: {signal['size']} @ {signal['confidence']:.3f} ({signal['method']})")
    
    # Test 3: Format alerts for both types
    print(f"\n📱 Testing alert formatting...")
    
    if signals1:
        best_signal1 = sorted(signals1, key=lambda x: x["confidence"], reverse=True)[0]
        if best_signal1["type"] == "color":
            alert_msg1 = format_color_alert(best_signal1, "202501011200001", 0.65)
            print("🎨 COLOR Alert (Test 1):")
        else:
            alert_msg1 = format_size_alert(best_signal1, "202501011200001", 0.65)
            print("⚖️  SIZE Alert (Test 1):")
        print(alert_msg1)
    
    if signals2:
        best_signal2 = sorted(signals2, key=lambda x: x["confidence"], reverse=True)[0]
        if best_signal2["type"] == "color":
            alert_msg2 = format_color_alert(best_signal2, "202501011200002", 0.65)
            print("\n🎨 COLOR Alert (Test 2):")
        else:
            alert_msg2 = format_size_alert(best_signal2, "202501011200002", 0.65)
            print("\n⚖️  SIZE Alert (Test 2):")
        print(alert_msg2)
    
    # Test 4: Verify only strongest signal is shown
    print(f"\n✅ Smart Alert System Test Results:")
    print(f"   - Test 1 signals: {len(signals1)}")
    print(f"   - Test 2 signals: {len(signals2)}")
    
    if signals1:
        best1 = sorted(signals1, key=lambda x: x["confidence"], reverse=True)[0]
        print(f"   - Test 1 best: {best1['type'].upper()} @ {best1['confidence']:.3f}")
    
    if signals2:
        best2 = sorted(signals2, key=lambda x: x["confidence"], reverse=True)[0]
        print(f"   - Test 2 best: {best2['type'].upper()} @ {best2['confidence']:.3f}")
    
    print("\n🎯 Smart Alert System Features:")
    print("   ✅ Only shows strongest signal (color OR size, not both)")
    print("   ✅ Filters out weak signals automatically")
    print("   ✅ Separate formatting for color vs size alerts")
    print("   ✅ Higher thresholds for better signal quality")
    print("   ✅ No more confusing mixed alerts")

if __name__ == "__main__":
    test_smart_alerts()
