#!/usr/bin/env python3
"""
Check Progressive Scaling Timeline
Shows the data scaling schedule from 23k to 80k over time
"""

from datetime import datetime, timedelta

def get_current_week_number() -> int:
    """Calculate current week number since start of progressive scaling"""
    start_date = datetime(2024, 12, 20)  # Adjust to when you want to start progressive scaling
    current_date = datetime.utcnow()
    weeks_elapsed = (current_date - start_date).days // 7
    return max(0, weeks_elapsed)

def show_progressive_timeline():
    """Show the progressive scaling timeline"""
    print("📈 Progressive Data Scaling Timeline")
    print("=" * 50)
    print("Week | Data Limit | Progress | Date (approx)")
    print("-" * 50)
    
    start_date = datetime(2024, 12, 20)  # Same as get_current_week_number
    current_week = get_current_week_number()
    
    for week in range(0, 15):  # Show first 15 weeks
        data_limit = min(23000 + (week * 10000), 80000)
        progress = min(100, (data_limit / 80000) * 100)
        approx_date = start_date + timedelta(weeks=week)
        
        status = "← CURRENT" if week == current_week else ""
        print(f"{week:4d} | {data_limit:9,} | {progress:6.1f}% | {approx_date.strftime('%Y-%m-%d')} {status}")
        
        if data_limit >= 80000:
            print(f"{week+1:4d} | {80000:9,} | 100.0% | REACHED TARGET")
            break
    
    print("\n📊 Current Status:")
    current_limit = min(23000 + (current_week * 10000), 80000)
    current_progress = min(100, (current_limit / 80000) * 100)
    print(f"   Week: {current_week}")
    print(f"   Data Limit: {current_limit:,} rows")
    print(f"   Progress: {current_progress:.1f}% to 80k target")
    
    if current_limit < 80000:
        weeks_to_target = ((80000 - current_limit) // 10000) + 1
        target_date = start_date + timedelta(weeks=current_week + weeks_to_target)
        print(f"   Target reached in: {weeks_to_target} weeks ({target_date.strftime('%Y-%m-%d')})")
    else:
        print("   ✅ Target reached!")

if __name__ == "__main__":
    show_progressive_timeline()
