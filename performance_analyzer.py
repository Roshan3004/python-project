#!/usr/bin/env python3
"""
Performance analyzer based on your actual results
"""
from datetime import datetime, timedelta

def analyze_performance():
    """Analyze your actual performance data"""
    print("📊 Performance Analysis Based on Your Results")
    print("=" * 60)
    
    # Your actual data
    yesterday_signals = {
        "time_period": "11:30 PM - 1:00 AM",
        "signals": 4,
        "wins": 1,
        "losses": 3,
        "accuracy": 25.0
    }
    
    today_signals = {
        "time_period": "9:00 AM - Now",
        "signals": 4,
        "wins": 3,
        "losses": 1,
        "accuracy": 75.0
    }
    
    total_signals = yesterday_signals["signals"] + today_signals["signals"]
    total_wins = yesterday_signals["wins"] + today_signals["wins"]
    total_losses = yesterday_signals["losses"] + today_signals["losses"]
    overall_accuracy = (total_wins / total_signals) * 100
    
    print(f"📈 YESTERDAY ({yesterday_signals['time_period']}):")
    print(f"   Signals: {yesterday_signals['signals']}")
    print(f"   Wins: {yesterday_signals['wins']} | Losses: {yesterday_signals['losses']}")
    print(f"   Accuracy: {yesterday_signals['accuracy']:.1f}%")
    print()
    
    print(f"📈 TODAY ({today_signals['time_period']}):")
    print(f"   Signals: {today_signals['signals']}")
    print(f"   Wins: {today_signals['wins']} | Losses: {today_signals['losses']}")
    print(f"   Accuracy: {today_signals['accuracy']:.1f}%")
    print()
    
    print(f"📊 OVERALL PERFORMANCE:")
    print(f"   Total Signals: {total_signals}")
    print(f"   Total Wins: {total_wins}")
    print(f"   Total Losses: {total_losses}")
    print(f"   Overall Accuracy: {overall_accuracy:.1f}%")
    print()
    
    # Analysis and recommendations
    print("🔍 ANALYSIS:")
    print("=" * 40)
    
    if today_signals["accuracy"] > yesterday_signals["accuracy"]:
        print("✅ MORNING PERFORMANCE IS SUPERIOR")
        print("   - 75% vs 25% accuracy")
        print("   - 3x better performance in active hours")
        print("   - Market is more predictable during day")
    
    if overall_accuracy >= 50:
        print("✅ OVERALL PROFITABLE")
        print(f"   - {overall_accuracy:.1f}% accuracy is above break-even")
        print("   - System is working as designed")
    
    print()
    print("🎯 RECOMMENDATIONS:")
    print("=" * 40)
    
    print("1. ⏰ TIME-BASED OPTIMIZATION:")
    print("   - Focus on 9 AM - 11 PM (active hours)")
    print("   - Avoid late night signals (11 PM - 1 AM)")
    print("   - Use --disable_sleep_window for 24/7 operation")
    
    print("\n2. 🔧 PRESET OPTIMIZATION:")
    if today_signals["accuracy"] >= 70:
        print("   - Use --preset balanced (current is working well)")
        print("   - Consider --preset aggressive for more signals")
    else:
        print("   - Stick with --preset balanced")
        print("   - Avoid --preset very_aggressive")
    
    print("\n3. 📊 MONITORING STRATEGY:")
    print("   - Track hourly performance")
    print("   - Log which hours are most profitable")
    print("   - Adjust thresholds based on time of day")
    
    print("\n4. 🚀 IMMEDIATE ACTIONS:")
    print("   - Continue current settings")
    print("   - Monitor next 4-6 signals")
    print("   - If accuracy stays >60%, consider more aggressive settings")
    
    print("\n5. 📈 SCALING STRATEGY:")
    if overall_accuracy >= 60:
        print("   - INCREASE signal frequency")
        print("   - Use --preset aggressive")
        print("   - Consider --max_signals 6-8")
    elif overall_accuracy >= 50:
        print("   - MAINTAIN current settings")
        print("   - Focus on time-based optimization")
    else:
        print("   - REDUCE signal frequency")
        print("   - Use --preset conservative")
        print("   - Focus on quality over quantity")
    
    print("\n" + "=" * 60)
    print("💡 NEXT STEPS:")
    print("1. Run: python analyze.py --preset balanced --max_signals 6")
    print("2. Monitor performance for next 2-3 hours")
    print("3. If accuracy >60%, switch to --preset aggressive")
    print("4. Track which hours give best results")
    print("5. Consider time-based threshold adjustments")

if __name__ == "__main__":
    analyze_performance()
