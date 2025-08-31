"""
Configuration file for WinGo Momentum Analysis System
Tune these parameters to get your desired 7-15 signals per day
"""

# ====== SIGNAL GENERATION THRESHOLDS ======
# Manual thresholds (tuned for high accuracy - fewer but stronger signals)
MOMENTUM_CONFIDENCE_THRESHOLD = 0.72
NUMBER_PATTERN_THRESHOLD = 0.75
TIME_PATTERN_THRESHOLD = 0.78
ENSEMBLE_THRESHOLD = 0.80

# ====== LOOKBACK PERIODS ======
# Adjust these to change how much historical data is analyzed
MOMENTUM_LOOKBACK = 25                    # Default: 25 rounds (was 20)
NUMBER_PATTERN_LOOKBACK = 35              # Default: 35 rounds (was 30)
TIME_PATTERN_MIN_DATA = 50                # Default: 50 rounds minimum for time analysis

# ====== MOMENTUM SCORING WEIGHTS ======
# Adjust these to change how momentum is calculated
STREAK_BONUS_MULTIPLIER = 0.05            # Default: 0.05 (was 0.05)
RECENT_BIAS_MULTIPLIER = 0.10             # Default: 0.10 (was 0.10)
RECENT_ROUNDS_COUNT = 5                   # Default: 5 rounds for recent bias

# ====== NUMBER PATTERN ANALYSIS ======
# Adjust these for number pattern detection
UNDER_REPRESENTED_THRESHOLD = 0.7         # Default: 0.7 (30% below average)
MIN_UNDER_REPRESENTED_COUNT = 3           # Default: 3 numbers must be under-represented

# ====== TIME PATTERN ANALYSIS ======
# Adjust these for time-based patterns
TIME_SIMILARITY_HOURS = 2                 # Default: 2 hours (similar time periods)
MIN_HOURLY_DATA = 10                      # Default: 10 rounds minimum per hour

# ====== ENSEMBLE ANALYSIS ======
# Adjust these for combining multiple methods
MIN_SOURCES_FOR_ENSEMBLE = 2              # Ensure ensemble can trigger with 2 agreeing sources
ENSEMBLE_BONUS_THRESHOLD = 0.05           # Default: 0.05 higher than individual methods

# ====== AGGRESSIVE SIGNAL GENERATION ======
# Disabled for better signal quality (fewer but stronger signals)
ENABLE_AGGRESSIVE_SIGNALS = False         # Default: False (disabled for quality)
AGGRESSIVE_CONFIDENCE_THRESHOLD = 0.60    # Default: 0.60 (increased for quality)
SECONDARY_SIGNAL_BONUS = 0.03             # Default: 0.03 (reduced bonus)

# ====== BACKTESTING PARAMETERS ======
# Adjust these for accuracy estimation
BACKTEST_LOOKBACK = 300                   # Default: 300 rounds
BACKTEST_MIN_PREDICTIONS = 50             # Default: 50 minimum predictions for accuracy

# ====== ALERT FREQUENCY CONTROL ======
# Control how often alerts are sent (reduced for quality)
MAX_SIGNALS_PER_ANALYSIS = 3              # Default: 3 maximum signals per analysis run
MIN_TIME_BETWEEN_ALERTS = 15              # Default: 15 minutes between alerts for same color
ENABLE_DUPLICATE_ALERTS = False           # Default: False (avoid spam)

# ====== PERFORMANCE TUNING ======
# These affect system performance and signal quality
ENABLE_REAL_TIME_ANALYSIS = True          # Default: True (analyze every round)
BATCH_ANALYSIS_SIZE = 100                 # Default: 100 rounds per batch
ENABLE_CACHING = True                     # Default: True (cache analysis results)

# ====== SIGNAL QUALITY FILTERS ======
# Additional filters to ensure signal quality (stricter for better results)
MIN_SIGNAL_STRENGTH = 0.60                # Default: 0.60 (increased minimum confidence)
MAX_SIGNAL_STRENGTH = 0.95                # Default: 0.95 (maximum confidence cap)
ENABLE_VOLATILITY_FILTER = True           # Default: True (filter out volatile periods)
VOLATILITY_THRESHOLD = 0.12               # Default: 0.12 (reduced volatility tolerance)

# ====== TELEGRAM ALERT FORMATTING ======
# Customize alert messages
ALERT_EMOJI = "🚨"                        # Default: 🚨
INCLUDE_CONFIDENCE_IN_ALERT = True        # Default: True
INCLUDE_REASON_IN_ALERT = True            # Default: True
INCLUDE_PROBABILITIES_IN_ALERT = True     # Default: True
INCLUDE_ACCURACY_IN_ALERT = True          # Default: True

def get_optimized_thresholds():
    """Get thresholds optimized for 7-15 signals per day"""
    return {
        "momentum": MOMENTUM_CONFIDENCE_THRESHOLD,
        "number_pattern": NUMBER_PATTERN_THRESHOLD,
        "time_pattern": TIME_PATTERN_THRESHOLD,
        "ensemble": ENSEMBLE_THRESHOLD,
        "aggressive": AGGRESSIVE_CONFIDENCE_THRESHOLD
    }

def get_lookback_periods():
    """Get lookback periods for different analysis methods"""
    return {
        "momentum": MOMENTUM_LOOKBACK,
        "number_pattern": NUMBER_PATTERN_LOOKBACK,
        "time_pattern": TIME_PATTERN_MIN_DATA
    }

def is_aggressive_mode_enabled():
    """Check if aggressive signal generation is enabled"""
    return ENABLE_AGGRESSIVE_SIGNALS

def get_max_signals_per_run():
    """Get maximum signals per analysis run"""
    return MAX_SIGNALS_PER_ANALYSIS

# ====== QUICK TUNING PRESETS ======
def get_preset_config(preset_name: str):
    """Get predefined configurations for different signal frequencies"""
    presets = {
        "conservative": {
            "momentum": 0.62,
            "number_pattern": 0.65,
            "time_pattern": 0.67,
            "ensemble": 0.70,
            "aggressive": False
        },
        "balanced": {
            "momentum": 0.68,
            "number_pattern": 0.70,
            "time_pattern": 0.72,
            "ensemble": 0.74,
            "aggressive": True
        },
        "aggressive": {
            "momentum": 0.55,
            "number_pattern": 0.57,
            "time_pattern": 0.59,
            "ensemble": 0.61,
            "aggressive": True
        },
        "very_aggressive": {
            "momentum": 0.52,
            "number_pattern": 0.54,
            "time_pattern": 0.56,
            "ensemble": 0.58,
            "aggressive": True
        }
    }
    return presets.get(preset_name, presets["balanced"])

# ====== DYNAMIC THRESHOLD ADJUSTMENT ======
def adjust_thresholds_based_on_performance(accuracy: float, signals_per_day: int):
    """Dynamically adjust thresholds based on performance metrics"""
    if accuracy < 0.55 and signals_per_day < 5:
        # Too few signals, lower thresholds
        return {
            "momentum": max(0.52, MOMENTUM_CONFIDENCE_THRESHOLD - 0.03),
            "number_pattern": max(0.54, NUMBER_PATTERN_THRESHOLD - 0.03),
            "time_pattern": max(0.56, TIME_PATTERN_THRESHOLD - 0.03),
            "ensemble": max(0.58, ENSEMBLE_THRESHOLD - 0.03)
        }
    elif accuracy > 0.70 and signals_per_day > 20:
        # Too many signals, raise thresholds
        return {
            "momentum": min(0.65, MOMENTUM_CONFIDENCE_THRESHOLD + 0.02),
            "number_pattern": min(0.67, NUMBER_PATTERN_THRESHOLD + 0.02),
            "time_pattern": min(0.69, TIME_PATTERN_THRESHOLD + 0.02),
            "ensemble": min(0.71, ENSEMBLE_THRESHOLD + 0.02)
        }
    else:
        # Performance is good, keep current thresholds
        return get_optimized_thresholds()
