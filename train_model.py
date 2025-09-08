#!/usr/bin/env python3
"""
Weekly Model Training Script
Trains a base LightGBM model on full historical data (25k+ rows)
Saves model with versioning and performance tracking
"""

import os
import sys
import pickle
import json
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from config import ScraperConfig
from analyze import analyze_big_small, load_neon, ensure_fresh_neon_data

def build_ml_features(df: pd.DataFrame, is_training: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Build ML features for training or prediction.
    Returns (features, targets) for training, (features, None) for prediction.
    """
    features_list = []
    targets = [] if is_training else None
    
    # Use more data for training (up to 25k rows)
    max_rows = 25000 if is_training else min(1000, len(df))
    train_data = df.tail(max_rows).copy()
    
    # For training, use more samples
    if is_training:
        max_samples = min(2000, len(train_data) - 100)  # More training samples
        start_idx = max(100, len(train_data) - max_samples - 100)
    else:
        max_samples = min(500, len(train_data) - 50)
        start_idx = max(50, len(train_data) - max_samples - 50)
    
    for i in range(start_idx, len(train_data)):
        features = []
        
        # Time features
        try:
            timestamp = pd.to_datetime(train_data.iloc[i]["scraped_at"])
            features.extend([timestamp.hour, timestamp.minute, timestamp.weekday()])
        except Exception:
            features.extend([12, 0, 0])
        
        # Historical color features (1, 2, 3 rounds ago)
        for lag in [1, 2, 3]:
            if i - lag >= 0:
                color = train_data.iloc[i - lag]["color"]
                features.append({"RED": 0, "GREEN": 1, "VIOLET": 2}.get(color, 0))
            else:
                features.append(0)
        
        # Frequency features (last 10, 30, 50)
        for window in [10, 30, 50]:
            if i >= window:
                window_data = train_data.iloc[i - window:i]
                red_freq = (window_data["color"] == "RED").sum() / window
                green_freq = (window_data["color"] == "GREEN").sum() / window
                violet_freq = (window_data["color"] == "VIOLET").sum() / window
                features.extend([red_freq, green_freq, violet_freq])
            else:
                features.extend([0.33, 0.33, 0.34])
        
        # Current streak based on previous color
        if i - 1 >= 0:
            current_color = train_data.iloc[i - 1]["color"]
            streak = 1
            for j in range(i - 2, max(0, i - 10), -1):
                if train_data.iloc[j]["color"] == current_color:
                    streak += 1
                else:
                    break
            features.append(min(streak, 10))
        else:
            features.append(1)
        
        # Number patterns (last 20)
        if i >= 20:
            recent_numbers = train_data.iloc[i - 20:i]["number"].astype(int).tolist()
            for num in range(10):
                features.append(recent_numbers.count(num) / 20)
            features.append(max(recent_numbers))
            features.append(min(recent_numbers))
            features.append(float(np.mean(recent_numbers)))
            features.append(float(np.std(recent_numbers)))
        else:
            features.extend([0.1] * 10)
            features.extend([5, 0, 5, 0])
        
        # Number volatility std over last 15 and 45
        for vol_window in [15, 45]:
            if i >= vol_window:
                nums_win = train_data.iloc[i - vol_window:i]["number"].astype(int).values
                features.append(float(np.std(nums_win)))
            else:
                features.append(0.0)
        
        # Time since last VIOLET
        lookback_slice = train_data.iloc[:i]
        last_violet_idx = None
        if len(lookback_slice) > 0:
            violet_positions = np.where(lookback_slice["color"].values == "VIOLET")[0]
            if violet_positions.size > 0:
                last_violet_idx = int(violet_positions[-1])
        if last_violet_idx is not None:
            features.append(float(i - last_violet_idx))
        else:
            features.append(float(min(i, 100)))
        
        # Markov BIG probability from analyze_big_small
        try:
            if i >= 20:
                temp_df = train_data.iloc[max(0, i - 60):i][["number", "color"]].copy()
                temp_df["number"] = temp_df["number"].astype(int)
                size_probs, _, _ = analyze_big_small(temp_df, lookback=min(60, len(temp_df)))
                features.append(float(size_probs.get("BIG", 0.5)))
            else:
                features.append(0.5)
        except Exception:
            features.append(0.5)
        
        # Lag numbers 1 and 2
        for lag in [1, 2]:
            if i - lag >= 0:
                try:
                    features.append(int(train_data.iloc[i - lag]["number"]))
                except Exception:
                    features.append(0)
            else:
                features.append(0)
        
        # Target for training
        if is_training:
            target_color = train_data.iloc[i]["color"]
            targets.append({"RED": 0, "GREEN": 1, "VIOLET": 2}.get(target_color, 0))
        
        features_list.append(features)
    
    X = np.array(features_list)
    y = np.array(targets) if is_training else None
    return X, y

def train_full_model(df: pd.DataFrame) -> Tuple[LGBMClassifier, Dict]:
    """Train full model on historical data with performance tracking"""
    print(f"🏋️ Training full model on {len(df)} rows...")
    
    X, y = build_ml_features(df, is_training=True)
    
    if len(X) < 1000:
        raise ValueError(f"Insufficient training data: {len(X)} samples (need 1000+)")
    
    # Split for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train with more conservative parameters for stability
    model = LGBMClassifier(
        n_estimators=300,  # More trees for better generalization
        max_depth=8,       # Deeper for complex patterns
        learning_rate=0.05,  # Lower for stability
        random_state=42,
        verbose=-1,
        class_weight='balanced',
        subsample=0.9,     # More data per tree
        colsample_bytree=0.9,  # More features per tree
        reg_alpha=0.1,     # L1 regularization
        reg_lambda=0.1     # L2 regularization
    )
    
    model.fit(X_train, y_train)
    
    # Validate model
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    
    # Performance metrics
    performance = {
        "accuracy": float(accuracy),
        "training_samples": len(X_train),
        "validation_samples": len(X_val),
        "feature_count": X.shape[1],
        "model_params": {
            "n_estimators": model.n_estimators,
            "max_depth": model.max_depth,
            "learning_rate": model.learning_rate
        },
        "trained_at": datetime.utcnow().isoformat(),
        "data_period": {
            "start": df["scraped_at"].min(),
            "end": df["scraped_at"].max(),
            "total_rows": len(df)
        },
        "progressive_scaling": {
            "enabled": True,
            "current_week": get_current_week_number(),
            "data_limit": len(df),
            "target_limit": 80000,
            "progress_percent": min(100, (len(df) / 80000) * 100)
        }
    }
    
    print(f"✅ Model trained: {accuracy:.3f} accuracy on {len(X_val)} validation samples")
    return model, performance

def save_model_with_versioning(model: LGBMClassifier, performance: Dict, model_dir: str = "models") -> str:
    """Save model with versioning and metadata"""
    os.makedirs(model_dir, exist_ok=True)
    
    # Create versioned filename
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    model_file = os.path.join(model_dir, f"lightgbm_model_{timestamp}.pkl")
    latest_file = os.path.join(model_dir, "lightgbm_model.pkl")
    metadata_file = os.path.join(model_dir, f"model_metadata_{timestamp}.json")
    
    # Save model
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    
    # Save latest version
    with open(latest_file, 'wb') as f:
        pickle.dump(model, f)
    
    # Save metadata
    with open(metadata_file, 'w') as f:
        json.dump(performance, f, indent=2)
    
    print(f"💾 Model saved: {model_file}")
    print(f"📊 Performance: {performance['accuracy']:.3f} accuracy")
    
    return model_file

def get_current_week_number() -> int:
    """Calculate current week number since start of progressive scaling"""
    # Start progressive scaling from today (Week 0 = 23k rows)
    # This means we start with current data and scale up over time
    start_date = datetime(2024, 12, 20)  # Adjust to when you want to start progressive scaling
    current_date = datetime.utcnow()
    weeks_elapsed = (current_date - start_date).days // 7
    return max(0, weeks_elapsed)

def validate_model_performance(model: LGBMClassifier, df: pd.DataFrame, recent_rows: int = 200) -> Dict:
    """Validate model on recent data"""
    print(f"🔍 Validating model on recent {recent_rows} rows...")
    
    recent_df = df.tail(recent_rows)
    X, y = build_ml_features(recent_df, is_training=True)
    
    if len(X) < 50:
        return {"validation_accuracy": 0.0, "validation_samples": 0}
    
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    
    return {
        "validation_accuracy": float(accuracy),
        "validation_samples": len(X),
        "validation_period": {
            "start": recent_df["scraped_at"].min(),
            "end": recent_df["scraped_at"].max()
        }
    }

def main():
    parser = argparse.ArgumentParser(description="Train LightGBM model on full historical data")
    parser.add_argument("--limit", type=int, default=80000, help="Maximum rows to load (progressive scaling)")
    parser.add_argument("--model_dir", default="models", help="Directory to save models")
    parser.add_argument("--validate", action="store_true", help="Validate model on recent data")
    parser.add_argument("--progressive", action="store_true", help="Use progressive data scaling (23k→80k)")
    args = parser.parse_args()
    
    print("🚀 WinGo Model Training")
    print("=" * 50)
    
    try:
        # Load data with progressive scaling
        cfg = ScraperConfig()
        
        if args.progressive:
            # Progressive scaling: start with 23k, add 10k each week until 80k
            current_week = get_current_week_number()
            progressive_limit = min(23000 + (current_week * 10000), 80000)
            actual_limit = min(progressive_limit, args.limit)
            print(f"📈 Progressive scaling: Week {current_week}, using {actual_limit:,} rows (target: 80k)")
        else:
            actual_limit = args.limit
            print(f"📊 Using fixed limit: {actual_limit:,} rows")
        
        df = ensure_fresh_neon_data(cfg, actual_limit, fresh_seconds=30, max_wait_seconds=20)
        
        if len(df) < 1000:
            print(f"❌ Insufficient data: {len(df)} rows (need 1000+)")
            return 1
        
        print(f"📊 Loaded {len(df):,} rows from database")
        
        # Train model
        model, performance = train_full_model(df)
        
        # Validate if requested
        if args.validate:
            validation = validate_model_performance(model, df)
            performance.update(validation)
            print(f"🔍 Recent validation: {validation['validation_accuracy']:.3f} accuracy")
        
        # Save model
        model_file = save_model_with_versioning(model, performance, args.model_dir)
        
        print("=" * 50)
        print("✅ Training completed successfully!")
        print(f"📁 Model saved: {model_file}")
        print(f"📊 Final accuracy: {performance['accuracy']:.3f}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
