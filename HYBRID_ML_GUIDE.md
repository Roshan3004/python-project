# Hybrid ML Workflow Guide

## Overview
This system uses a hybrid ML approach combining weekly full model training with daily fine-tuning for improved accuracy and efficiency.

## Architecture

### Weekly Training (`train_model.py`)
- **Schedule**: Every Sunday at 00:00 UTC (5:30 AM IST)
- **Data**: Progressive scaling (23k → 80k rows over 6 weeks)
- **Purpose**: Train robust base model with gradually increasing data
- **Output**: `models/lightgbm_model.pkl` + metadata

### Daily Analysis (`analyze.py`)
- **Schedule**: Every 10 minutes
- **Process**: 
  1. Load saved model
  2. Validate on recent data
  3. Fine-tune with last 500-1000 rows
  4. Make predictions
  5. Save updated model
- **Fallback**: Train from scratch if saved model fails

## Key Features

### Progressive Data Scaling
- **Week 0**: 23,000 rows (current data)
- **Week 1**: 33,000 rows (+10k)
- **Week 2**: 43,000 rows (+10k)
- **Week 3**: 53,000 rows (+10k)
- **Week 4**: 63,000 rows (+10k)
- **Week 5**: 73,000 rows (+10k)
- **Week 6**: 80,000 rows (target reached)
- **After Week 6**: Always use recent 80k rows

### Model Persistence
- **Loading**: `load_saved_model()` with error handling
- **Saving**: `save_model()` with directory creation
- **Validation**: `validate_model_on_recent_data()` for performance check

### Hybrid Training
- **Base Model**: Weekly full training with progressive data scaling
- **Fine-tuning**: Daily updates with recent 500-1000 rows
- **Fallback**: Train from scratch if model performance < 45%

### Advanced Features
- **Probability margin gate**: Requires top1-top2 ≥ 0.20
- **Entropy gate**: Filters low-certainty distributions
- **Recent performance penalty**: Adjusts thresholds based on recent accuracy
- **Sleep window**: Skips 1:00-9:00 IST (configurable)

## Setup

### 1. Environment Variables
```bash
export NEON_CONN_STR="postgresql://user:pass@host:port/db?sslmode=require"
export TELEGRAM_BOT_TOKEN="your_bot_token"
export TELEGRAM_CHAT_ID="your_chat_id"
```

### 2. Manual Testing
```bash
# Test training with progressive scaling
python train_model.py --progressive --validate

# Test training with fixed limit
python train_model.py --limit 5000 --validate

# Test analysis
python analyze.py --source db --fast_mode --preset balanced

# Check progressive scaling timeline
python check_progressive_scaling.py

# Test hybrid workflow
python test_hybrid_ml.py
```

### 3. GitHub Actions Setup
1. Add secrets in GitHub repository:
   - `NEON_CONN_STR`: Your Supabase connection string
   - `TELEGRAM_BOT_TOKEN`: Bot token for alerts
   - `TELEGRAM_CHAT_ID`: Chat ID for alerts

2. Workflows will run automatically:
   - **Weekly training**: Sunday 00:00 UTC
   - **Analysis**: Every 10 minutes

## Expected Performance

### Accuracy Improvements
- **Base model**: 3-8% improvement from 25k+ training data
- **Fine-tuning**: Adapts to recent patterns without losing base knowledge
- **Validation**: Prevents deployment of poor-performing models

### Signal Quality
- **Fewer false positives**: Margin and entropy gates filter fragile predictions
- **Better adaptation**: Recent performance penalty adjusts to market changes
- **Consistent performance**: Hybrid approach maintains stability

### Resource Usage
- **Training time**: 2-5 minutes weekly (acceptable)
- **Analysis time**: 10-30 seconds per run (efficient)
- **Storage**: Model files ~10-50MB (manageable)

## Monitoring

### Model Performance
- Check `models/model_metadata_*.json` for training metrics
- Monitor recent accuracy in analysis logs
- Watch for fallback to scratch training

### Alert Quality
- Track signal accuracy over time
- Monitor false positive rates
- Adjust thresholds based on performance

## Troubleshooting

### Common Issues
1. **Model loading fails**: Check file permissions and paths
2. **Poor recent accuracy**: Model may need retraining
3. **No signals generated**: Check thresholds and data quality
4. **GitHub Actions fails**: Verify secrets and dependencies

### Debug Commands
```bash
# Check model files
ls -la models/

# Test with verbose output
python analyze.py --source db --fast_mode --preset balanced --max_signals 1

# Validate model manually
python -c "from analyze import load_saved_model; print(load_saved_model())"
```

## Configuration

### Thresholds (CLI)
- `--min_prob_margin`: Probability margin gate (default: 0.20)
- `--max_entropy`: Entropy gate (default: 0.85)
- `--enable_recent_penalty`: Use recent performance penalty
- `--disable_sleep_window`: Override quiet hours

### Model Parameters
- **Training**: 300 trees, depth 8, learning rate 0.05
- **Fine-tuning**: Lower learning rate, early stopping
- **Validation**: 200 recent rows, 45% accuracy threshold

This hybrid approach provides the best of both worlds: rich historical patterns from weekly training and real-time adaptation from daily fine-tuning.
