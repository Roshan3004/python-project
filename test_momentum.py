#!/usr/bin/env python3
"""
Test script for WinGo Momentum Analysis System
This will help you see how many signals you can expect with different presets
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from momentum_config import get_preset_config, get_optimized_thresholds
from analyze import (
    analyze_color_momentum, 
    analyze_number_patterns, 
    analyze_time_based_patterns,
    detect_strong_signals,
    backtest_momentum_system
)

def generate_test_data(n_rounds=1000):
    """Generate realistic test data for WinGo"""
    np.random.seed(42)  # For reproducible results
    
    # Start from a recent time
    start_time = datetime.now() - timedelta(minutes=n_rounds)
    
    data = []
    for i in range(n_rounds):
        # Simulate some patterns (not completely random)
        if i % 50 == 0:  # Every 50 rounds, create a pattern
            # Create a streak
            streak_length = np.random.randint(3, 8)
            streak_color = np.random.choice(["RED", "GREEN", "VIOLET"])
            for j in range(streak_length):
                if i + j < n_rounds:
                    if streak_color == "RED":
                        number = np.random.choice([1, 3, 5, 7, 9])
                    elif streak_color == "GREEN":
                        number = np.random.choice([2, 4, 6, 8])
                    else:  # VIOLET
                        number = np.random.choice([0, 5])
                    
                    data.append({
                        "period_id": f"{(start_time + timedelta(minutes=i+j)).strftime('%Y%m%d%H%M')}",
                        "number": number,
                        "color": streak_color,
                        "scraped_at": (start_time + timedelta(minutes=i+j)).isoformat()
                    })
            i += streak_length - 1
        else:
            # Normal random round
            number = np.random.randint(0, 10)
            if number in [0, 5]:
                color = "VIOLET"
            elif number % 2 == 0:
                color = "GREEN"
            else:
                color = "RED"
            
            data.append({
                "period_id": f"{(start_time + timedelta(minutes=i)).strftime('%Y%m%d%H%M')}",
                "number": number,
                "color": color,
                "scraped_at": (start_time + timedelta(minutes=i)).isoformat()
            })
    
    return pd.DataFrame(data)

def test_presets():
    """Test different preset configurations"""
    print("🎯 Testing WinGo Momentum Analysis System")
    print("=" * 50)
    
    # Generate test data
    print("Generating test data...")
    df = generate_test_data(1000)
    print(f"Generated {len(df)} test rounds")
    print()
    
    # Test each preset
    presets = ["conservative", "balanced", "aggressive", "very_aggressive"]
    
    for preset in presets:
        print(f"📊 Testing {preset.upper()} preset:")
        preset_config = get_preset_config(preset)
        
        # Use momentum threshold for signal detection
        min_confidence = preset_config["momentum"]
        signals = detect_strong_signals(df, min_confidence=min_confidence)
        
        print(f"  Threshold: {min_confidence}")
        print(f"  Signals detected: {len(signals)}")
        
        if signals:
            # Show top 3 signals
            top_signals = sorted(signals, key=lambda x: x["confidence"], reverse=True)[:3]
            for i, signal in enumerate(top_signals, 1):
                print(f"    Signal {i}: {signal['color']} ({signal['method']}) - {signal['confidence']:.3f}")
        else:
            print("    No signals detected")
        
        print()
    
    # Test backtesting
    print("🔍 Testing backtesting system:")
    accuracy = backtest_momentum_system(df)
    print(f"  Estimated accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    print()
    
    # Show configuration options
    print("⚙️  Configuration options:")
    print("  --preset conservative: Fewer signals, higher accuracy")
    print("  --preset balanced: Moderate signals, balanced accuracy")
    print("  --preset aggressive: More signals, moderate accuracy")
    print("  --preset very_aggressive: Many signals, lower accuracy")
    print()
    print("  Example: python analyze.py --preset aggressive --max_signals 8")

def test_individual_methods():
    """Test individual analysis methods"""
    print("🔬 Testing Individual Analysis Methods")
    print("=" * 50)
    
    df = generate_test_data(500)
    
    # Test momentum analysis
    print("1. Color Momentum Analysis:")
    momentum_probs = analyze_color_momentum(df, lookback=25)
    for color, prob in momentum_probs.items():
        print(f"   {color}: {prob:.3f}")
    print()
    
    # Test number pattern analysis
    print("2. Number Pattern Analysis:")
    number_probs = analyze_number_patterns(df, lookback=35)
    for color, prob in number_probs.items():
        print(f"   {color}: {prob:.3f}")
    print()
    
    # Test time pattern analysis
    print("3. Time Pattern Analysis:")
    time_probs = analyze_time_based_patterns(df)
    for color, prob in time_probs.items():
        print(f"   {color}: {prob:.3f}")
    print()

if __name__ == "__main__":
    print("🚀 WinGo Momentum Analysis Test Suite")
    print()
    
    # Test individual methods
    test_individual_methods()
    print()
    
    # Test presets
    test_presets()
    
    print("✅ Testing complete!")
    print("\nTo run the actual analysis:")
    print("python analyze.py --preset aggressive --enable_alert --log_to_db")
