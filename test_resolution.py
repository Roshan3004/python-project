#!/usr/bin/env python3
"""
Test script to verify that prediction resolution works correctly
"""
import os
import sys
from config import ScraperConfig
from analyze import resolve_unresolved_alerts

def test_resolution():
    """Test the resolve_unresolved_alerts function"""
    print("🧪 Testing prediction resolution...")
    
    # Load configuration
    cfg = ScraperConfig()
    
    if not cfg.neon_conn_str:
        print("❌ Error: Database connection string not found!")
        print("Please set NEON_CONN_STR environment variable")
        return False
    
    try:
        # Test the resolution function
        print("🔄 Running resolve_unresolved_alerts...")
        resolve_unresolved_alerts(cfg.neon_conn_str, batch_limit=10)
        print("✅ Resolution test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Resolution test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_resolution()
    sys.exit(0 if success else 1)
