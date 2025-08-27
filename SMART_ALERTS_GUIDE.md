# Smart Alert System Guide

## 🎯 Problem Solved

The previous system had these issues:
- **Mixed alerts**: Always showed both color AND size predictions in every alert
- **False signals**: Weak predictions were included, causing confusion
- **Low accuracy**: Too many signals with low confidence

## ✅ New Smart Alert System

### Key Improvements

1. **Single Strongest Signal Only**
   - Shows either color OR size, never both
   - Automatically selects the signal with highest confidence
   - No more confusing mixed alerts

2. **Stricter Quality Filters**
   - Higher confidence thresholds (0.65-0.70 instead of 0.58-0.62)
   - Only alerts when confidence is truly high
   - Filters out weak signals automatically

3. **Separate Alert Formats**
   - 🎨 **Color Alerts**: Clear color predictions with confidence
   - ⚖️ **Size Alerts**: Clear size predictions with confidence
   - No mixing of different prediction types

### Configuration Changes

#### GitHub Actions Workflow (`.github/workflows/analyze.yml`)
```yaml
# Stricter defaults for better quality
PRESET: balanced                    # Was: aggressive
MAX_SIGNALS: 3                      # Was: 5
COLOR_PROB: 0.68                    # Was: 0.64
```

#### Momentum Config (`momentum_config.py`)
```python
# Higher thresholds for better signal quality
MOMENTUM_CONFIDENCE_THRESHOLD = 0.65      # Was: 0.58
NUMBER_PATTERN_THRESHOLD = 0.68           # Was: 0.60
TIME_PATTERN_THRESHOLD = 0.65             # Was: 0.62
ENSEMBLE_THRESHOLD = 0.70                 # Was: 0.65

# Disabled aggressive mode for quality
ENABLE_AGGRESSIVE_SIGNALS = False         # Was: True
MAX_SIGNALS_PER_ANALYSIS = 3              # Was: 5
```

### Alert Examples

#### Color Alert
```
🎨 WinGo Color Signal: GREEN
🔢 Bet on Period: 202501011200002
📊 Method: NumberPattern
🎯 Confidence: 1.000
💡 Reason: Number pattern suggests GREEN correction with 1.000 confidence
📈 Probs: R=0.00 G=1.00 V=0.00
✅ System Accuracy: 65.0%
⏰ Alert Time (UTC): 2025-08-27 19:11:38
⏱️  Next Round ETA (UTC): 12:00:00 (0s)
🎲 Place bet on GREEN for the NEXT round!
💡 Tip: Bet within the next 30 seconds for best timing
```

#### Size Alert
```
⚖️  WinGo Size Signal: BIG
🔢 Bet on Period: 202501011200001
📊 Method: BigSmall
🎯 Confidence: 0.683
💡 Reason: Size analysis suggests BIG with 0.683 confidence
📈 Probs: BIG=0.68 SMALL=0.32
✅ System Accuracy: 65.0%
⏰ Alert Time (UTC): 2025-08-27 19:11:38
⏱️  Next Round ETA (UTC): 12:00:00 (0s)
🎲 Place bet on BIG for the NEXT round!
💡 Tip: Bet within the next 30 seconds for best timing
```

## 🔧 How It Works

### Signal Detection Process

1. **Multiple Analysis Methods**
   - Color Momentum Analysis
   - Number Pattern Analysis  
   - Time-based Pattern Analysis
   - Big/Small Analysis
   - Ensemble Analysis (when multiple methods agree)

2. **Signal Ranking**
   - All signals are ranked by confidence
   - Only the strongest signal is selected
   - Weak signals are automatically filtered out

3. **Alert Generation**
   - Color signals → Color alert format
   - Size signals → Size alert format
   - No mixing of different types

### Quality Control

- **Minimum Confidence**: 0.65 for color, 0.65 for size
- **Maximum Signals**: 3 per analysis run (was 5)
- **Time Between Alerts**: 15 minutes minimum (was 10)
- **Volatility Filter**: Reduced tolerance for unstable periods

## 📊 Expected Results

### Before (Old System)
- ❌ Mixed alerts (color + size in same message)
- ❌ Many weak signals (30-40% accuracy)
- ❌ Confusing predictions
- ❌ 5+ signals per run

### After (Smart System)
- ✅ Single strongest signal only
- ✅ Higher quality signals (65%+ accuracy target)
- ✅ Clear, focused predictions
- ✅ 1-3 signals per run (quality over quantity)

## 🚀 Usage

The system automatically runs every 10 minutes via GitHub Actions. No manual changes needed.

### Manual Testing
```bash
# Test the smart alert system
python test_smart_alerts.py

# Run analysis manually
python analyze.py --source db --preset balanced --max_signals 3 --enable_alert
```

## 🎯 Benefits

1. **Better Accuracy**: Higher confidence thresholds mean stronger signals
2. **Clearer Alerts**: Single prediction type per alert
3. **Less Confusion**: No mixed color + size predictions
4. **Focused Betting**: Clear, actionable signals
5. **Reduced Noise**: Fewer but higher quality alerts

## 📈 Monitoring

- Check Telegram for alert quality
- Monitor system accuracy in alert messages
- Use `test_smart_alerts.py` to verify system behavior
- Adjust thresholds in `momentum_config.py` if needed

The new smart alert system prioritizes **quality over quantity** and provides **clear, actionable signals** without confusion.
