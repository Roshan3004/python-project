#!/usr/bin/env python3
"""
Test script for hybrid ML workflow
Tests model training and loading functionality
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_test_data(n_rows: int = 1000) -> pd.DataFrame:
    """Create synthetic test data"""
    print(f"Creating {n_rows} rows of test data...")
    
    data = []
    base_time = datetime.utcnow() - timedelta(hours=24)
    
    for i in range(n_rows):
        # Simulate realistic patterns
        period_id = f"{(base_time + timedelta(minutes=i)).strftime('%Y%m%d%H%M')}001"
        number = np.random.choice([0,1,2,3,4,5,6,7,8,9], p=[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1])
        color = "VIOLET" if number in [0,5] else ("GREEN" if number % 2 == 0 else "RED")
        scraped_at = (base_time + timedelta(minutes=i)).isoformat()
        
        data.append({
            "period_id": period_id,
            "number": number,
            "color": color,
            "scraped_at": scraped_at
        })
    
    return pd.DataFrame(data)

def test_train_model():
    """Test model training"""
    print("🧪 Testing model training...")
    
    try:
        from train_model import train_full_model, save_model_with_versioning
        
        # Create test data
        df = create_test_data(2000)
        
        # Train model
        model, performance = train_full_model(df)
        
        # Save model
        model_file = save_model_with_versioning(model, performance, "test_models")
        
        print(f"✅ Training test passed: {model_file}")
        return True
        
    except Exception as e:
        print(f"❌ Training test failed: {e}")
        return False

def test_analyze_hybrid():
    """Test hybrid analysis"""
    print("🧪 Testing hybrid analysis...")
    
    try:
        from analyze import load_saved_model, save_model, build_ml_features, analyze_with_ml_model
        
        # Create test data
        df = create_test_data(1000)
        
        # Test feature building
        X, y = build_ml_features(df, is_training=True)
        print(f"✅ Features built: {X.shape}, targets: {y.shape if y is not None else 'None'}")
        
        # Test analysis
        result = analyze_with_ml_model(df, min_data_points=100)
        print(f"✅ Analysis result: {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ Analysis test failed: {e}")
        return False

def main():
    print("🚀 Testing Hybrid ML Workflow")
    print("=" * 50)
    
    # Test training
    train_ok = test_train_model()
    
    # Test analysis
    analyze_ok = test_analyze_hybrid()
    
    print("=" * 50)
    if train_ok and analyze_ok:
        print("✅ All tests passed! Hybrid ML workflow is ready.")
        return 0
    else:
        print("❌ Some tests failed. Check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
