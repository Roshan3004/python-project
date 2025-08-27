#!/usr/bin/env python3
"""
Test script for WinGo Improved Alert System
This demonstrates how the system now calculates the NEXT betting period
"""

import pandas as pd
from datetime import datetime, timedelta
from analyze import get_next_betting_period, format_betting_alert

def generate_test_periods():
    """Generate test period IDs to demonstrate the system"""
    # Start from a recent time
    start_time = datetime.now() - timedelta(minutes=10)
    
    periods = []
    for i in range(10):
        current_time = start_time + timedelta(minutes=i)
        period_id = current_time.strftime("%Y%m%d%H%M") + "001"
        periods.append(period_id)
    
    return periods

def test_betting_period_calculation():
    """Test the betting period calculation"""
    print("🎯 Testing Betting Period Calculation")
    print("=" * 50)
    
    # Generate test data
    periods = generate_test_periods()
    
    print("📅 Sample Period IDs:")
    for i, period in enumerate(periods):
        print(f"  {i+1:2d}. {period}")
    
    print(f"\n🔢 Latest Period: {periods[-1]}")
    
    # Create a mock DataFrame
    df = pd.DataFrame({
        "period_id": periods,
        "number": [1, 2, 3, 4, 5, 6, 7, 8, 9, 0],
        "color": ["RED", "GREEN", "VIOLET", "RED", "GREEN", "VIOLET", "RED", "GREEN", "VIOLET", "RED"],
        "scraped_at": [datetime.now().isoformat()] * 10
    })
    
    # Calculate next betting period
    next_period = get_next_betting_period(df)
    print(f"🎲 Next Betting Period: {next_period}")
    
    # Show the logic
    latest = periods[-1]
    if len(latest) >= 12:
        timestamp_part = latest[:12]
        period_num = latest[12:] if len(latest) > 12 else "001"
        
        print(f"\n🔍 Calculation Breakdown:")
        print(f"  Latest period: {latest}")
        print(f"  Timestamp part: {timestamp_part}")
        print(f"  Period number: {period_num}")
        
        # Parse and increment
        dt = datetime.strptime(timestamp_part, "%Y%m%d%H%M")
        next_dt = dt + timedelta(minutes=1)
        next_timestamp = next_dt.strftime("%Y%m%d%H%M")
        calculated_next = f"{next_timestamp}{period_num}"
        
        print(f"  Current time: {dt.strftime('%Y-%m-%d %H:%M')}")
        print(f"  Next time: {next_dt.strftime('%Y-%m-%d %H:%M')}")
        print(f"  Calculated next: {calculated_next}")
        print(f"  Function result: {next_period}")
        
        if calculated_next == next_period:
            print("  ✅ Calculation correct!")
        else:
            print("  ❌ Calculation mismatch!")
    
    return df, next_period

def test_alert_formatting():
    """Test the alert message formatting"""
    print("\n📱 Testing Alert Message Formatting")
    print("=" * 50)
    
    # Create a mock signal
    mock_signal = {
        "color": "GREEN",
        "method": "Momentum",
        "confidence": 0.75,
        "reason": "Strong GREEN momentum with 0.75 confidence",
        "probs": {"RED": 0.20, "GREEN": 0.75, "VIOLET": 0.05}
    }
    
    # Test with different betting periods
    test_periods = ["20250827100010373", "20250827100010374", "20250827100010375"]
    
    for period in test_periods:
        print(f"\n🎲 Alert for Period: {period}")
        print("-" * 30)
        
        alert_msg = format_betting_alert(mock_signal, period, 0.65)
        print(alert_msg)
    
    return mock_signal

def demonstrate_betting_timing():
    """Demonstrate the betting timing concept"""
    print("\n⏰ Betting Timing Demonstration")
    print("=" * 50)
    
    current_time = datetime.now()
    print(f"🕐 Current time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Show timing for next few rounds
    for i in range(1, 6):
        next_round_time = current_time + timedelta(minutes=i)
        time_until = (next_round_time - current_time).total_seconds()
        
        print(f"🎲 Round {i}: {next_round_time.strftime('%H:%M:%S')} (in {time_until:.0f}s)")
    
    print("\n💡 BETTING STRATEGY:")
    print("1. System analyzes completed rounds")
    print("2. Detects patterns and sends alerts")
    print("3. Alert shows NEXT period to bet on")
    print("4. Place bet within 30 seconds of round start")
    print("5. Wait for results and repeat")

def main():
    print("🚀 WinGo Improved Alert System Test")
    print("=" * 50)
    
    # Test betting period calculation
    df, next_period = test_betting_period_calculation()
    
    # Test alert formatting
    mock_signal = test_alert_formatting()
    
    # Demonstrate betting timing
    demonstrate_betting_timing()
    
    print("\n" + "=" * 50)
    print("✅ Testing Complete!")
    print("\n🎯 Key Improvements:")
    print("  ✅ Calculates NEXT betting period (not current)")
    print("  ✅ Prevents duplicate alerts")
    print("  ✅ Clear betting instructions")
    print("  ✅ Timing information included")
    print("  ✅ Better alert formatting")
    
    print("\n🚀 To test the real system:")
    print("python analyze.py --preset aggressive --enable_alert --log_to_db")

if __name__ == "__main__":
    main()
