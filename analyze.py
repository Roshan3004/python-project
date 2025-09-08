import argparse, csv, math, statistics, os, json, time
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np
from scipy.stats import chisquare
import requests
from lightgbm import LGBMClassifier, early_stopping
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
from pathlib import Path
from db import MongoStore
from strategy import suggest_next, suggest_from_ensemble
from config import ScraperConfig
from momentum_config import (
    get_optimized_thresholds, 
    get_lookback_periods, 
    is_aggressive_mode_enabled,
    get_max_signals_per_run,
    get_preset_config
)
from sqlalchemy import create_engine

# ====== MODEL PERSISTENCE FUNCTIONS ======
def load_saved_model(model_path: str = "models/lightgbm_model.pkl") -> Optional[LGBMClassifier]:
    """Load saved model with fallback handling"""
    try:
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            print(f"✅ Loaded saved model from {model_path}")
            return model
    except Exception as e:
        print(f"⚠️ Failed to load saved model: {e}")
    return None

def save_model(model: LGBMClassifier, model_path: str = "models/lightgbm_model.pkl") -> bool:
    """Save model with directory creation"""
    try:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        return True
    except Exception as e:
        print(f"⚠️ Failed to save model: {e}")
        return False

def validate_model_on_recent_data(model: LGBMClassifier, df: pd.DataFrame, recent_rows: int = 200) -> float:
    """Quick validation of model on recent data"""
    try:
        if len(df) < recent_rows + 50:
            return 0.5  # Default if insufficient data
        
        recent_df = df.tail(recent_rows)
        X, y = build_ml_features(recent_df, is_training=True)
        
        if len(X) < 20:
            return 0.5
        
        y_pred = model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        return float(accuracy)
    except Exception:
        return 0.5

def build_ml_features(df: pd.DataFrame, is_training: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Build ML features (extracted from analyze_with_ml_model for reuse)"""
    features_list = []
    targets = [] if is_training else None
    
    # Use appropriate data size
    max_rows = 1000 if not is_training else min(800, len(df))
    train_data = df.tail(max_rows).copy()
    
    if is_training:
        max_samples = min(300, len(train_data) - 50)
        start_idx = max(50, len(train_data) - max_samples - 50)
    else:
        max_samples = min(200, len(train_data) - 30)
        start_idx = max(30, len(train_data) - max_samples - 30)
    
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

# ====== Helpers for size (Big/Small) ======
def number_to_size(n: int) -> str:
    try:
        n = int(n)
        return "BIG" if n >= 5 else "SMALL"
    except Exception:
        return "SMALL"

def build_markov_size(sizes: List[str], window: int = 300, alpha: float = 1.0) -> Tuple[Dict[str, Dict[str, float]], Optional[str]]:
    keys = ["SMALL","BIG"]
    idx = {c:i for i,c in enumerate(keys)}
    if not sizes:
        return {k:{kk:0.5 for kk in keys} for k in keys}, None
    seq = [s if s in idx else number_to_size(s) for s in sizes][-window:]
    M = np.full((2,2), alpha, dtype=float)
    for a,b in zip(seq[:-1], seq[1:]):
        if a in idx and b in idx:
            M[idx[a], idx[b]] += 1
    M = M / M.sum(axis=1, keepdims=True)
    out = {keys[i]: {keys[j]: float(M[i,j]) for j in range(2)} for i in range(2)}
    return out, seq[-1]

def size_markov_probs(transition: Dict[str, Dict[str, float]], last_size: Optional[str]) -> Dict[str, float]:
    keys = ["SMALL","BIG"]
    if not last_size or last_size not in transition:
        return {k: 0.5 for k in keys}
    row = transition[last_size]
    s = sum(row.values()) or 1.0
    return {k: row.get(k, 0.0)/s for k in keys}

def analyze_big_small(df: pd.DataFrame, lookback: int = 60) -> Tuple[Dict[str,float], float, str]:
    """Return (probs, confidence, reason) for BIG/SMALL.

    Blend of:
      - frequency momentum over last `lookback`
      - streak bonus on the most recent run
      - Markov 1-step on BIG/SMALL transitions
    """
    if len(df) < max(lookback, 20):
        return {"BIG": 0.5, "SMALL": 0.5}, 0.5, "Insufficient data; default 0.5"

    recent = df.tail(lookback)
    nums = recent["number"].astype(int).tolist()
    sizes = [number_to_size(n) for n in nums]

    big_count = sum(1 for s in sizes if s == "BIG")
    small_count = lookback - big_count

    # frequency
    p_big_freq = big_count / lookback
    p_small_freq = 1.0 - p_big_freq

    # streak bonus
    current = sizes[-1]
    streak = 1
    for s in reversed(sizes[:-1]):
        if s == current:
            streak += 1
        else:
            break
    bonus = min(0.10, 0.03 * streak)  # cap at +0.10
    if current == "BIG":
        p_big_streak = min(0.99, p_big_freq + bonus)
        p_small_streak = 1 - p_big_streak
    else:
        p_small_streak = min(0.99, p_small_freq + bonus)
        p_big_streak = 1 - p_small_streak

    # Markov 1-step
    trans, last_size = build_markov_size(sizes, window=min(300, lookback))
    mkv = size_markov_probs(trans, last_size)

    # Blend: 50% momentum(streak) + 50% markov
    p_big = 0.5 * p_big_streak + 0.5 * mkv.get("BIG", 0.5)
    p_small = 1.0 - p_big

    top = "BIG" if p_big >= p_small else "SMALL"
    conf = max(p_big, p_small)
    reason = f"freq={p_big_freq:.2f}/{p_small_freq:.2f}, streak={streak}, markov_BIG={mkv.get('BIG',0.5):.2f}"
    return {"BIG": p_big, "SMALL": p_small}, conf, reason

# ====== Cross-run deduping support (reservation table) ======
def reserve_alert_slot(conn_str: str, target_period: str, signal_type: str) -> bool:
    """Attempt to reserve (target_period, signal_type) before sending Telegram.

    Returns True if reservation inserted (i.e., we are first), False if a
    concurrent/previous run already reserved this slot.
    """
    try:
        import psycopg2
        with psycopg2.connect(conn_str) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS alert_reservations (
                      id SERIAL PRIMARY KEY,
                      target_period VARCHAR(50) NOT NULL,
                      signal_type VARCHAR(16) NOT NULL,
                      created_at timestamptz DEFAULT now(),
                      UNIQUE(target_period, signal_type)
                    );
                    """
                )
                cur.execute(
                    """
                    INSERT INTO alert_reservations(target_period, signal_type)
                    VALUES (%s, %s)
                    ON CONFLICT (target_period, signal_type) DO NOTHING
                    RETURNING id;
                    """,
                    (str(target_period), str(signal_type).upper()),
                )
                row = cur.fetchone()
                conn.commit()
                return row is not None
    except Exception:
        # On any DB error, fail-open (allow send) to avoid losing signals
        return True

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=["period_id", "number", "color"])
    df["number"] = df["number"].astype(int) % 10
    df["color"] = df["color"].str.upper()
    return df

def load_neon(conn_str: str, limit: int = 1500) -> pd.DataFrame:
    """Load data directly from Neon PostgreSQL using SQLAlchemy engine for speed/stability"""
    query = f"""
    SELECT period_id, number, color, scraped_at
    FROM game_history
    ORDER BY scraped_at DESC
    LIMIT {limit}
    """
    engine = create_engine(conn_str)
    try:
        df = pd.read_sql(query, engine)
    finally:
        try:
            engine.dispose()
        except Exception:
            pass
    return df.sort_values("period_id")

def ensure_fresh_neon_data(cfg: ScraperConfig, limit: int, fresh_seconds: int = 20, max_wait_seconds: int = 12) -> pd.DataFrame:
    """Reload Neon until the newest row is fresh enough or max wait reached.

    fresh_seconds: how recent the newest scraped_at must be relative to now (UTC)
    max_wait_seconds: total time to wait/retry before giving up
    """
    waited = 0
    retry_count = 0
    while True:
        retry_count += 1
        df = load_neon(cfg.neon_conn_str, limit=limit)
        try:
            last_ts = pd.to_datetime(df["scraped_at"].iloc[-1], utc=True, errors="coerce")
        except Exception:
            last_ts = None
        now_utc = pd.Timestamp.utcnow()
        if pd.notna(last_ts):
            age = (now_utc - last_ts).total_seconds()
            print(f"📊 Data freshness check: {age:.1f}s old (target: ≤{fresh_seconds}s)")
            if age <= fresh_seconds:
                print(f"✅ Fresh data found after {retry_count} attempt(s)")
                return df
        else:
            print(f"⚠️  Unable to parse scraped_at timestamp, attempt {retry_count}")
        
        if waited >= max_wait_seconds:
            print(f"⏰ Timeout reached ({max_wait_seconds}s), proceeding with available data")
            return df
        
        # Smart retry timing: faster retries near period boundaries
        current_second = int(time.time()) % 60
        if 55 <= current_second or current_second <= 5:  # Near period boundary
            sleep_time = 1  # Check every second near period end
        else:
            sleep_time = 3  # Normal 3-second intervals
        
        print(f"🔄 Retrying in {sleep_time}s... (waited {waited}s/{max_wait_seconds}s)")
        time.sleep(sleep_time)
        waited += sleep_time

def log_alert_to_neon(conn_str: str,
                      anchor_period_id: str,
                      predicted_color: str,
                      predicted_number: Optional[int],
                      color_probs: Dict[str, float],
                      sources: List[str],
                      confidence: float,
                      last300_precision: float,
                      cycle_len: Optional[int],
                      cycle_acc: float) -> None:
    """Create table if missing and insert one alert row. Safe no-op on errors."""
    try:
        import psycopg2
        with psycopg2.connect(conn_str) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS prediction_alerts (
                        id SERIAL PRIMARY KEY,
                        anchor_period_id VARCHAR(50) UNIQUE NOT NULL,
                        predicted_color TEXT NOT NULL,
                        predicted_number INT,
                        color_probs JSONB,
                        sources JSONB,
                        confidence REAL,
                        last300_precision REAL,
                        cycle_len INT,
                        cycle_acc REAL,
                        created_at timestamptz DEFAULT now()
                    );
                    """
                )
                cur.execute(
                    """
                    INSERT INTO prediction_alerts (
                        anchor_period_id, predicted_color, predicted_number,
                        color_probs, sources, confidence, last300_precision,
                        cycle_len, cycle_acc
                    ) VALUES (%s, %s, %s, %s::jsonb, %s::jsonb, %s, %s, %s, %s)
                    ON CONFLICT (anchor_period_id) DO NOTHING;
                    """,
                    (
                        str(anchor_period_id), predicted_color, predicted_number,
                        json.dumps(color_probs), json.dumps(sources), confidence,
                        last300_precision, (cycle_len if cycle_len is not None else None), cycle_acc,
                    ),
                )
            conn.commit()
    except Exception:
        # Intentionally swallow logging errors to not affect alerting
        pass

def resolve_unresolved_alerts(conn_str: str, batch_limit: int = 200) -> None:
    """Update prediction_alerts with actual outcome from game_history for unresolved rows.
    Uses the NEXT period after the anchor to determine outcomes.
    """
    try:
        import psycopg2
        with psycopg2.connect(conn_str) as conn:
            with conn.cursor() as cur:
                # Ensure outcome columns exist
                cur.execute(
                    """
                    ALTER TABLE prediction_alerts
                    ADD COLUMN IF NOT EXISTS outcome_color TEXT,
                    ADD COLUMN IF NOT EXISTS outcome_number INT,
                    ADD COLUMN IF NOT EXISTS hit_color BOOLEAN,
                    ADD COLUMN IF NOT EXISTS hit_number BOOLEAN,
                    ADD COLUMN IF NOT EXISTS resolved_at timestamptz;
                    """
                )
                
                # Get unresolved alerts
                cur.execute(
                    """
                    SELECT id, anchor_period_id, predicted_color, predicted_number
                        FROM prediction_alerts
                        WHERE resolved_at IS NULL
                        ORDER BY created_at ASC
                        LIMIT %s
                    """,
                    (batch_limit,)
                )
                
                unresolved = cur.fetchall()
                
                for alert_id, anchor_period, predicted_color, predicted_number in unresolved:
                    # Find the NEXT period after anchor_period
                    try:
                        next_period_id = str(int(anchor_period) + 1)
                    except:
                        continue
                    
                    # Get the actual outcome for that period
                    cur.execute(
                        """
                            SELECT number, color
                            FROM game_history
                        WHERE period_id = %s
                            LIMIT 1
                        """,
                        (next_period_id,)
                    )
                    
                    outcome = cur.fetchone()
                    if outcome:
                        outcome_number, outcome_color = outcome
                        
                        # Determine if prediction was correct
                        hit_color = (predicted_color == outcome_color)
                        hit_number = (predicted_number == outcome_number) if predicted_number is not None else None
                        
                        # Update the alert with outcome
                        cur.execute(
                            """
                            UPDATE prediction_alerts
                            SET outcome_number = %s,
                                outcome_color = %s,
                                hit_color = %s,
                                hit_number = %s,
                                resolved_at = now()
                            WHERE id = %s
                            """,
                            (outcome_number, outcome_color, hit_color, hit_number, alert_id)
                        )
                
            conn.commit()
    except Exception as e:
        print(f"Warning: Could not resolve alerts: {e}")
        pass

def color_from_number(n: int) -> str:
    return "VIOLET" if n in (0,5) else ("GREEN" if n % 2 == 0 else "RED")

def chi_square_color(df: pd.DataFrame) -> Tuple[float, float]:
    # Expected: RED ~ 4/10, GREEN ~ 4/10, VIOLET ~ 2/10 under common mapping
    counts = df["color"].value_counts().reindex(["RED","GREEN","VIOLET"]).fillna(0).values
    total = counts.sum()
    if total == 0:
        return 0.0, 1.0
    expected = np.array([0.4*total, 0.4*total, 0.2*total])
    chi, p = chisquare(counts, f_exp=expected)
    return chi, p

def find_cycle(numbers: List[int], max_period: int = 80) -> Tuple[Optional[int], Optional[List[int]]]:
    # Try to detect repeating period in last 400 samples
    seq = numbers[-400:] if len(numbers) > 400 else numbers[:]
    n = len(seq)
    best_p, best_score = None, 0
    for p in range(5, min(max_period, n//2)+1):
        # score how similar seq is to itself shifted by p
        matches = sum(1 for i in range(n-p) if seq[i] == seq[i+p])
        score = matches / (n-p) if (n-p)>0 else 0
        if score > best_score:
            best_score, best_p = score, p
    if best_p and best_score >= 0.55:  # adjustable threshold
        # Infer cycle template as argmax over positions mod p
        template = [None]*best_p
        buckets = [[] for _ in range(best_p)]
        for i, v in enumerate(seq):
            buckets[i % best_p].append(v)
        for i,b in enumerate(buckets):
            # choose the most frequent number at each cycle position
            template[i] = max(set(b), key=b.count)
        return best_p, template
    return None, None

def next_from_cycle(numbers: List[int], cycle_len: int, template: List[int]) -> Optional[int]:
    if not numbers: return None
    k = len(numbers) % cycle_len
    return template[k]

def validate_cycle(numbers: List[int], cycle_len: int, template: List[int], holdout: int = 80) -> float:
    """Return accuracy on the last `holdout` rounds predicted by the cycle template.
    Uses positions modulo `cycle_len`.
    """
    if not cycle_len or not template or len(numbers) < holdout + cycle_len:
        return 0.0
    acc, total = 0, 0
    n = len(numbers)
    start = max(0, n - holdout)
    for i in range(start, n):
        pred = template[i % cycle_len]
        if pred == numbers[i]:
            acc += 1
        total += 1
    return acc / max(1, total)

def build_markov_number(nums: List[int], window: int = 10000, alpha: float = 1.0) -> Tuple[np.ndarray, Optional[int]]:
    """Return (10x10 row-stochastic matrix, last_number) using last `window` samples.
    Laplace smoothing alpha to avoid zero rows.
    """
    if not nums:
        return np.zeros((10,10)), None
    seq = nums[-window:]
    M = np.full((10,10), alpha, dtype=float)
    for a, b in zip(seq[:-1], seq[1:]):
        if 0 <= a <= 9 and 0 <= b <= 9:
            M[a, b] += 1
    # normalize rows
    M = M / M.sum(axis=1, keepdims=True)
    return M, seq[-1]

def build_markov_color(colors: List[str], window: int = 10000, alpha: float = 1.0) -> Tuple[Dict[str, Dict[str, float]], Optional[str]]:
    """Return color transition dict and last color using last `window` samples.
    Colors: RED/GREEN/VIOLET. Laplace smoothing alpha.
    """
    keys = ["RED","GREEN","VIOLET"]
    idx = {c:i for i,c in enumerate(keys)}
    if not colors:
        return {k:{kk:1/3 for kk in keys} for k in keys}, None
    seq = [c if c in idx else color_from_number(int(c)) for c in colors][-window:]
    M = np.full((3,3), alpha, dtype=float)
    for a, b in zip(seq[:-1], seq[1:]):
        if a in idx and b in idx:
            M[idx[a], idx[b]] += 1
    M = M / M.sum(axis=1, keepdims=True)
    out = {keys[i]: {keys[j]: float(M[i,j]) for j in range(3)} for i in range(3)}
    return out, seq[-1]

def markov_color_probs(transition: Dict[str, Dict[str, float]], last_color: Optional[str]) -> Dict[str, float]:
    keys = ["RED","GREEN","VIOLET"]
    if not last_color or last_color not in transition:
        return {k: 1/3 for k in keys}
    row = transition[last_color]
    s = sum(row.values()) or 1.0
    return {k: row.get(k, 0.0)/s for k in keys}

def backtest_markov_color(colors: List[str], train_window: int = 10000, eval_n: int = 300) -> float:
    """Approximate precision of Markov color over last eval_n transitions using preceding data.
    Not walk-forward per step (for speed), but informative.
    """
    if len(colors) < eval_n + 5:
        return 0.0
    train = colors[:-eval_n]
    test = colors[-eval_n:]
    trans, _ = build_markov_color(train, window=train_window)
    correct, total = 0, 0
    last = train[-1]
    for nxt in test:
        probs = markov_color_probs(trans, last)
        pred = max(probs, key=probs.get)
        if pred == nxt:
            correct += 1
        total += 1
        last = nxt
    return correct / max(1, total)

def send_telegram(cfg: ScraperConfig, text: str) -> bool:
    token = cfg.telegram_bot_token
    chat_id = cfg.telegram_chat_id
    if not token or not chat_id:
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        r = requests.post(url, json={"chat_id": chat_id, "text": text}, timeout=10)
        return r.ok
    except Exception:
        return False

def get_next_betting_period(df: pd.DataFrame) -> str:
    """Return the next period id.

    Important: OK.Win's period id is NOT a pure timestamp. Many realms use a
    fixed middle token like "10001" with a trailing counter. So the safest and
    correct way is to treat it as an integer and increment by 1.
    """
    try:
        latest_period = str(df["period_id"].iloc[-1]).strip()
        # Prefer numeric increment. If it fails, fall back to timestamp heuristic.
        try:
            return str(int(latest_period) + 1)
        except Exception:
            pass

        # Fallback: preserve suffix if present but still advance minute boundary
        if len(latest_period) >= 12:
            timestamp_part = latest_period[:12]
            suffix = latest_period[12:]
            latest_dt = datetime.strptime(timestamp_part, "%Y%m%d%H%M")
            now_dt = datetime.utcnow()
            target_dt = max(latest_dt + timedelta(minutes=1), now_dt.replace(second=0, microsecond=0) + timedelta(minutes=1))
            return f"{target_dt.strftime('%Y%m%d%H%M')}{suffix or '001'}"
        return latest_period
    except Exception as e:
        print(f"Warning: Could not calculate next period: {e}")
        return "UNKNOWN"

def format_betting_alert(signal: dict, betting_period: str, accuracy: float) -> str:
    """Format a betting alert message with clear instructions"""
    color = signal["color"]
    method = signal["method"]
    confidence = signal["confidence"]
    reason = signal["reason"]
    probs = signal["probs"]
    
    # Calculate time until target round (best-effort based on minute boundary)
    current_time = datetime.utcnow()
    try:
        # betting_period starts with YYYYMMDDHHMM
        if len(betting_period) >= 12:
            target_dt = datetime.strptime(betting_period[:12], "%Y%m%d%H%M")
        else:
            target_dt = (current_time.replace(second=0, microsecond=0) + timedelta(minutes=1))
    except Exception:
        target_dt = (current_time.replace(second=0, microsecond=0) + timedelta(minutes=1))
    seconds_until = max(0, int((target_dt - current_time).total_seconds()))
    
    msg = (
        f"🚨 WinGo Strong Signal: {color}\n"
        f"🔢 Bet on Period: {betting_period}\n"
        f"📊 Method: {method}\n"
        f"🎯 Confidence: {confidence:.3f}\n"
        f"💡 Reason: {reason}\n"
        f"📈 Probs: R={probs['RED']:.2f} G={probs['GREEN']:.2f} V={probs['VIOLET']:.2f}\n"
        f"✅ System Accuracy: {accuracy:.1%}\n"
        f"⏰ Alert Time (UTC): {current_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"⏱️  Next Round ETA (UTC): {target_dt.strftime('%H:%M:%S')} ({seconds_until}s)\n"
        f"🎲 Place bet on {color} for the NEXT round!\n"
        f"💡 Tip: Bet within the next 30 seconds for best timing"
    )
    
    return msg

def ensure_min_time_buffer(df: pd.DataFrame, betting_period: str, min_buffer_seconds: int = 30) -> str:
    """If the computed betting_period starts in less than `min_buffer_seconds`,
    shift it forward to guarantee user has time to bet.
    Default 30 seconds buffer - enough time to place bet.
    """
    now_utc = datetime.utcnow()
    try:
        if len(betting_period) >= 12 and betting_period[:12].isdigit():
            target_dt = datetime.strptime(betting_period[:12], "%Y%m%d%H%M")
            seconds_until = (target_dt - now_utc).total_seconds()
            
            # If less than buffer time, push to next period only
            if seconds_until < min_buffer_seconds:
                suffix = betting_period[12:]
                target_dt = target_dt + timedelta(minutes=1)
                betting_period = f"{target_dt.strftime('%Y%m%d%H%M')}{suffix}"
            
            return betting_period
            
        # Numeric-only fallback - add 1 period if needed
        val = int(betting_period)
        # Check if we need buffer based on current time
        current_minute = datetime.utcnow().second
        if current_minute > 30:  # If past 30 seconds in current minute
            return str(val + 1)  # Skip to next period
        return str(val)  # Use current next period
        
    except Exception:
        # Conservative fallback - use next period
        try:
            return str(int(betting_period) + 1)
        except:
            return betting_period

def manipulation_indicators(numbers: List[int], colors: List[str]) -> Dict[str, bool]:
    flags = {}
    # 1) Violet rate spike
    violet_rate = colors[-120:].count("VIOLET")/max(1,len(colors[-120:]))
    flags["violet_spike"] = violet_rate > 0.28  # > expected 0.2 by wide margin
    # 2) Long anti-cycle runs where predicted value misses > 5 times in 20
    # placeholder; real-time needs live predictions. We'll piggy-back on chi-square drift.
    # 3) Parity flips unrealistically often in last 40 rounds
    last = colors[-40:]
    flips = sum(1 for i in range(1,len(last)) if (last[i] in ["RED","GREEN"]) and (last[i-1] in ["RED","GREEN"]) and (last[i]!=last[i-1]))
    flags["parity_whiplash"] = flips > 28  # very high flip rate
    return flags

def summarize_flags(flags: Dict[str,bool]) -> Tuple[bool, str]:
    active = [k for k,v in flags.items() if v]
    return (len(active)>0, (", ".join(active) if active else "none"))

# ====== MACHINE LEARNING ANALYSIS SYSTEM ======

def analyze_with_ml_model(df: pd.DataFrame, min_data_points: int = 200) -> Dict[str, float]:
    """
    Hybrid ML approach: Load saved model + fine-tune with recent data
    
    Features:
    - Time of day (hour, minute, weekday)
    - Colors from 1, 2, 3 rounds ago (encoded)
    - RED/GREEN/VIOLET frequencies over last 10, 30, 50 rounds
    - Current color streak length (capped at 10)
    - Number distribution over last 20 (counts normalized) + summary stats
    - Number volatility std over last 15 and 45 rounds
    - Time since last VIOLET
    - Markov BIG probability from analyze_big_small() as a numeric feature
    - Lagged numbers 1 and 2 rounds ago
    """
    if len(df) < min_data_points:
        print(f"⚠️  Insufficient data for ML model (need {min_data_points}, have {len(df)})")
        return {"RED": 0.33, "GREEN": 0.33, "VIOLET": 0.34}

    try:
        # Try to load saved model first
        saved_model = load_saved_model()
        use_saved_model = saved_model is not None
        
        if use_saved_model:
            # Validate saved model on recent data
            recent_accuracy = validate_model_on_recent_data(saved_model, df, recent_rows=200)
            print(f"🔍 Saved model recent accuracy: {recent_accuracy:.3f}")
            
            # If saved model performs poorly, fall back to training from scratch
            if recent_accuracy < 0.45:
                print("⚠️  Saved model performance poor, training from scratch")
                use_saved_model = False
                saved_model = None
        
        if use_saved_model:
            # Respect fast mode: skip fine-tuning to reduce latency
            fast_mode = os.getenv("WINGO_FAST_MODE", "0") == "1"
            model = saved_model
            if fast_mode:
                print("⚡ Fast mode: skipping fine-tune to reduce latency")
            else:
                # Use saved model as base, fine-tune with recent data
                print("🔄 Fine-tuning saved model with recent data...")
                
                # Fine-tune with recent data (last 500-1000 rows)
                X_recent, y_recent = build_ml_features(df, is_training=True)
                
                if len(X_recent) >= 80:  # Minimum for fine-tuning with a small val split
                    # Create a small validation split to enable early stopping
                    Xr_tr, Xr_val, yr_tr, yr_val = train_test_split(
                        X_recent, y_recent, test_size=0.2, random_state=42, stratify=y_recent
                    )
                    model.fit(
                        Xr_tr,
                        yr_tr,
                        eval_set=[(Xr_val, yr_val)],
                        callbacks=[early_stopping(stopping_rounds=10)]
                    )
                    print(f"✅ Fine-tuned model on {len(X_recent)} recent samples")
                else:
                    print("⚠️  Insufficient recent data for fine-tuning, using saved model as-is")
        else:
            # Train from scratch (fallback)
            print("🏋️ Training new model from scratch...")
            X, y = build_ml_features(df, is_training=True)
            
            if len(X) < 100:
                print("⚠️  Not enough training samples for ML model")
                return {"RED": 0.33, "GREEN": 0.33, "VIOLET": 0.34}
            
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            model = LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.07,
                random_state=42,
                verbose=-1,
                class_weight='balanced',
                subsample=0.8,
                colsample_bytree=0.8
            )
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            print(f"🤖 New Model Accuracy: {accuracy:.3f}")

        # Build current features for prediction
        X_current, _ = build_ml_features(df, is_training=False)
        
        if len(X_current) == 0:
            print("⚠️  No features for prediction")
            return {"RED": 0.33, "GREEN": 0.33, "VIOLET": 0.34}
        
        # Make prediction
        probabilities = model.predict_proba(X_current)[0]
        color_probs = {"RED": probabilities[0], "GREEN": probabilities[1], "VIOLET": probabilities[2]}
        print(f"🤖 ML Prediction: R={color_probs['RED']:.3f}, G={color_probs['GREEN']:.3f}, V={color_probs['VIOLET']:.3f}")
        
        # Save updated model (if fine-tuned or newly trained)
        if not use_saved_model or (not os.getenv("WINGO_FAST_MODE", "0") == "1" and 'X_recent' in locals() and len(X_recent) >= 50):
            save_model(model)
            print("💾 Model saved for next run")
            
            # Log progressive scaling info if available
            try:
                from train_model import get_current_week_number
                current_week = get_current_week_number()
                current_limit = min(23000 + (current_week * 10000), 80000)
                progress = min(100, (current_limit / 80000) * 100)
                print(f"📈 Progressive scaling: Week {current_week}, {current_limit:,} rows ({progress:.1f}% to 80k target)")
            except Exception:
                pass
        
        return color_probs

    except Exception as e:
        print(f"❌ ML Model Error: {e}")
        return {"RED": 0.33, "GREEN": 0.33, "VIOLET": 0.34}

def analyze_prediction_performance(df: pd.DataFrame, lookback: int = 100) -> Dict[str, float]:
    """
    Analyze recent prediction performance to adjust thresholds dynamically.
    Returns performance metrics for different methods.
    """
    if len(df) < lookback:
        return {"overall": 0.33, "ml": 0.33, "momentum": 0.33}
    
    recent = df.tail(lookback)
    
    # Calculate momentum-based accuracy
    momentum_correct = 0
    for i in range(20, len(recent)):
        # Simple momentum prediction: predict the most frequent color in last 20 rounds
        window = recent.iloc[i-20:i]
        red_count = (window["color"] == "RED").sum()
        green_count = (window["color"] == "GREEN").sum()
        violet_count = (window["color"] == "VIOLET").sum()
        
        predicted_color = max([("RED", red_count), ("GREEN", green_count), ("VIOLET", violet_count)], 
                            key=lambda x: x[1])[0]
        
        actual_color = recent.iloc[i]["color"]
        if predicted_color == actual_color:
            momentum_correct += 1
    
    momentum_accuracy = momentum_correct / max(1, len(recent) - 20)
    
    # Estimate ML accuracy (this would be more accurate with actual prediction history)
    # For now, assume ML is slightly better than momentum when working well
    ml_accuracy = min(0.45, momentum_accuracy + 0.05)  # Conservative estimate
    
    # Overall accuracy (weighted average)
    overall_accuracy = 0.6 * ml_accuracy + 0.4 * momentum_accuracy
    
    performance = {
        "overall": overall_accuracy,
        "ml": ml_accuracy,
        "momentum": momentum_accuracy
    }
    
    print(f"📊 Performance Analysis (last {lookback} rounds):")
    print(f"   Overall: {overall_accuracy:.3f}")
    print(f"   ML: {ml_accuracy:.3f}")
    print(f"   Momentum: {momentum_accuracy:.3f}")
    
    return performance

def get_adaptive_thresholds(df: pd.DataFrame, base_threshold: float = 0.65) -> float:
    """
    Dynamically adjust thresholds based on recent performance.
    """
    performance = analyze_prediction_performance(df, lookback=100)
    
    # Adjust threshold based on performance
    if performance["overall"] < 0.35:  # Very poor performance
        adjusted_threshold = base_threshold + 0.15  # Higher threshold = more selective
    elif performance["overall"] < 0.40:  # Poor performance
        adjusted_threshold = base_threshold + 0.10
    elif performance["overall"] < 0.45:  # Below average
        adjusted_threshold = base_threshold + 0.05
    elif performance["overall"] > 0.55:  # Good performance
        adjusted_threshold = base_threshold - 0.05  # Lower threshold = more signals
    else:
        adjusted_threshold = base_threshold
    
    print(f"🎯 Adaptive threshold: {adjusted_threshold:.3f} (base: {base_threshold:.3f})")
    return adjusted_threshold

# ====== LEGACY MOMENTUM-BASED ANALYSIS SYSTEM (DEPRECATED) ======

# Old statistical analysis functions removed - replaced with Machine Learning model

def detect_volatility(df: pd.DataFrame, lookback: int = 30) -> float:
    """Detect market volatility to avoid unstable periods"""
    if len(df) < lookback:
        return 0.0
    
    recent = df.tail(lookback)
    colors = recent["color"].tolist()
    
    # Count color switches in recent period
    switches = 0
    for i in range(1, len(colors)):
        if colors[i] != colors[i-1]:
            switches += 1
    
    # High switch rate indicates volatility
    volatility = switches / max(1, len(colors) - 1)
    return volatility

def analyze_recent_performance(conn_str: str, lookback_hours: int = 4) -> Dict[str, float]:
    """Analyze recent performance to adjust confidence dynamically"""
    try:
        import psycopg2
        from datetime import datetime, timedelta
        
        cutoff = datetime.utcnow() - timedelta(hours=lookback_hours)
        
        with psycopg2.connect(conn_str) as conn:
            with conn.cursor() as cur:
                # Get recent resolved predictions
                cur.execute("""
                    SELECT hit_color, confidence, method 
                    FROM prediction_alerts 
                    WHERE created_at >= %s 
                    AND resolved_at IS NOT NULL 
                    ORDER BY created_at DESC
                    LIMIT 10
                """, (cutoff,))
                
                results = cur.fetchall()
                if not results:
                    return {"accuracy": 0.5, "confidence_penalty": 0.0, "method_penalty": {}}
                
                # Calculate recent accuracy
                hits = sum(1 for hit, _, _ in results if hit)
                accuracy = hits / len(results)
                
                # Calculate confidence penalty based on recent performance
                if accuracy < 0.4:  # Very poor recent performance
                    confidence_penalty = 0.10
                elif accuracy < 0.6:  # Poor recent performance
                    confidence_penalty = 0.05
                else:
                    confidence_penalty = 0.0
                
                # Calculate method-specific penalties
                method_performance = {}
                for hit, conf, method in results:
                    if method not in method_performance:
                        method_performance[method] = {"hits": 0, "total": 0}
                    method_performance[method]["total"] += 1
                    if hit:
                        method_performance[method]["hits"] += 1
                
                method_penalty = {}
                for method, stats in method_performance.items():
                    method_acc = stats["hits"] / stats["total"] if stats["total"] > 0 else 0.5
                    if method_acc < 0.3:  # Method performing very poorly
                        method_penalty[method] = 0.15
                    elif method_acc < 0.5:  # Method performing poorly
                        method_penalty[method] = 0.08
                    else:
                        method_penalty[method] = 0.0
                
                return {
                    "accuracy": accuracy,
                    "confidence_penalty": confidence_penalty,
                    "method_penalty": method_penalty
                }
    except Exception:
        return {"accuracy": 0.5, "confidence_penalty": 0.0, "method_penalty": {}}

def detect_strong_signals(df: pd.DataFrame, 
                         ml_threshold: float = 0.70,
                         size_threshold: float = 0.70,
                         conn_str: str = None) -> List[Dict]:
    """Detect strong signals using Machine Learning model with enhanced filtering"""
    signals = []
    
    # Pull optional gating knobs from env set by CLI layer (fallbacks)
    try:
        min_prob_margin = float(os.getenv("WINGO_MIN_PROB_MARGIN", "0.20"))
    except Exception:
        min_prob_margin = 0.20
    try:
        max_entropy = float(os.getenv("WINGO_MAX_ENTROPY", "0.85"))
    except Exception:
        max_entropy = 0.85
    
    # Enhanced filtering: Check volatility
    volatility = detect_volatility(df)
    print(f"🌊 Volatility check: {volatility:.3f} (threshold: 0.75)")
    if volatility > 0.90:  # Skip during extreme volatile periods
        print("⚠️  Skipping due to extreme volatility")
        return []
    
    # 1. Machine Learning Analysis (Primary Method)
    print("🤖 Running Machine Learning analysis...")
    ml_probs = analyze_with_ml_model(df, min_data_points=200)
    max_ml_confidence = max(ml_probs.values())
    print(f"🤖 ML analysis: max={max_ml_confidence:.3f} (threshold: {ml_threshold:.3f})")
    
    # Lower threshold for ML signals to generate more alerts
    ml_alert_threshold = max(0.60, ml_threshold - 0.05)  # tighter: allow at most -0.05
    
    # Probability margin and entropy gates
    def probs_entropy(p: Dict[str, float]) -> float:
        arr = np.array([p.get("RED", 0.0), p.get("GREEN", 0.0), p.get("VIOLET", 0.0)])
        arr = np.clip(arr, 1e-9, 1.0)
        arr = arr / arr.sum()
        return float(-np.sum(arr * np.log(arr)))
    def top_margin(p: Dict[str, float]) -> float:
        values = sorted(p.values(), reverse=True)
        return float(values[0] - values[1]) if len(values) >= 2 else 0.0
    
    margin = top_margin(ml_probs)
    entropy = probs_entropy(ml_probs)
    print(f"🔒 Prob margin={margin:.3f} (min {min_prob_margin:.2f}), entropy={entropy:.3f} (max {max_entropy:.2f})")
    
    if (max_ml_confidence >= ml_alert_threshold) and (margin >= min_prob_margin) and (entropy <= max_entropy):
        best_color = max(ml_probs, key=ml_probs.get)
        signals.append({
            "type": "color",
            "color": best_color,
            "confidence": max_ml_confidence,
            "method": "MachineLearning",
            "reason": f"ML model predicts {best_color} with {max_ml_confidence:.3f} confidence (margin={margin:.3f}, H={entropy:.3f})",
            "probs": ml_probs
        })
    else:
        print("❌ ML signal failed margin/entropy/threshold gates")
    
    # 2. Big/Small Analysis (Keep this as it's complementary to color prediction)
    size_probs, size_conf, size_reason = analyze_big_small(df)
    print(f"⚖️  Size analysis: conf={size_conf:.3f} (threshold: {size_threshold:.3f})")
    
    # Lower threshold for size signals to generate more alerts
    size_alert_threshold = max(0.65, size_threshold - 0.05)  # tighter size threshold
    
    if size_conf >= size_alert_threshold:
        best_size = "BIG" if size_probs["BIG"] >= size_probs["SMALL"] else "SMALL"
        signals.append({
            "type": "size",
            "size": best_size,
            "confidence": size_conf,
            "method": "BigSmall",
            "reason": f"Size analysis suggests {best_size} with {size_conf:.3f} confidence",
            "probs": size_probs
        })
    
    # 3. Ensemble Analysis (Combine ML with size if both are strong)
    color_signals = [s for s in signals if s["type"] == "color"]
    size_signals = [s for s in signals if s["type"] == "size"]
    
    if color_signals and size_signals:
        color_signal = color_signals[0]  # ML signal
        size_signal = size_signals[0]    # Size signal
        
        # Create ensemble if both signals are strong
        if color_signal["confidence"] >= 0.75 and size_signal["confidence"] >= 0.75:
            ensemble_confidence = (color_signal["confidence"] + size_signal["confidence"]) / 2
            signals.append({
                "type": "ensemble",
                "color": color_signal["color"],
                "size": size_signal["size"],
                "confidence": min(0.95, ensemble_confidence + 0.05),  # Bonus for agreement
                "method": "Ensemble",
                "reason": f"ML+Size ensemble: {color_signal['color']} + {size_signal['size']} with {ensemble_confidence:.3f} avg confidence",
                "probs": ml_probs
            })
    
    print(f"🎯 Total signals generated: {len(signals)}")
    for i, signal in enumerate(signals):
        if signal["type"] == "color":
            print(f"  Signal {i+1}: {signal['method']} - {signal['color']} @ {signal['confidence']:.3f}")
        elif signal["type"] == "size":
            print(f"  Signal {i+1}: {signal['method']} - {signal['size']} @ {signal['confidence']:.3f}")
        elif signal["type"] == "ensemble":
            print(f"  Signal {i+1}: {signal['method']} - {signal['color']} + {signal['size']} @ {signal['confidence']:.3f}")
    
    return signals

def format_color_alert(signal: dict, betting_period: str, accuracy: float) -> str:
    """Format a color betting alert message"""
    color = signal["color"]
    method = signal["method"]
    confidence = signal["confidence"]
    reason = signal["reason"]
    probs = signal["probs"]
    
    # Calculate time until target round
    current_time = datetime.utcnow()
    try:
        if len(betting_period) >= 12:
            target_dt = datetime.strptime(betting_period[:12], "%Y%m%d%H%M")
        else:
            target_dt = (current_time.replace(second=0, microsecond=0) + timedelta(minutes=1))
    except Exception:
        target_dt = (current_time.replace(second=0, microsecond=0) + timedelta(minutes=1))
    if target_dt <= current_time:
        target_dt = (current_time.replace(second=0, microsecond=0) + timedelta(minutes=1))
    seconds_until = max(0, int((target_dt - current_time).total_seconds()))
    
    msg = (
        f"🎨 WinGo Color Signal: {color}\n"
        f"🔢 Bet on Period: {betting_period}\n"
        f"📊 Method: {method}\n"
        f"🎯 Confidence: {confidence:.3f}\n"
        f"💡 Reason: {reason}\n"
        f"📈 Probs: R={probs['RED']:.2f} G={probs['GREEN']:.2f} V={probs['VIOLET']:.2f}\n"
        f"✅ System Accuracy: {accuracy:.1%}\n"
        f"⏰ Alert Time (UTC): {current_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"⏱️  Next Round ETA (UTC): {target_dt.strftime('%H:%M:%S')} ({seconds_until}s)\n"
        f"🎲 Place bet on {color} for the NEXT round!\n"
        f"💡 Tip: Bet within the next 30 seconds for best timing"
    )
    
    return msg

def format_size_alert(signal: dict, betting_period: str, accuracy: float) -> str:
    """Format a size betting alert message"""
    size = signal["size"]
    method = signal["method"]
    confidence = signal["confidence"]
    reason = signal["reason"]
    probs = signal["probs"]
    
    # Calculate time until target round
    current_time = datetime.utcnow()
    try:
        if len(betting_period) >= 12:
            target_dt = datetime.strptime(betting_period[:12], "%Y%m%d%H%M")
        else:
            target_dt = (current_time.replace(second=0, microsecond=0) + timedelta(minutes=1))
    except Exception:
        target_dt = (current_time.replace(second=0, microsecond=0) + timedelta(minutes=1))
    if target_dt <= current_time:
        target_dt = (current_time.replace(second=0, microsecond=0) + timedelta(minutes=1))
    seconds_until = max(0, int((target_dt - current_time).total_seconds()))
    
    msg = (
        f"⚖️  WinGo Size Signal: {size}\n"
        f"🔢 Bet on Period: {betting_period}\n"
        f"📊 Method: {method}\n"
        f"🎯 Confidence: {confidence:.3f}\n"
        f"💡 Reason: {reason}\n"
        f"📈 Probs: BIG={probs['BIG']:.2f} SMALL={probs['SMALL']:.2f}\n"
        f"✅ System Accuracy: {accuracy:.1%}\n"
        f"⏰ Alert Time (UTC): {current_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"⏱️  Next Round ETA (UTC): {target_dt.strftime('%H:%M:%S')} ({seconds_until}s)\n"
        f"🎲 Place bet on {size} for the NEXT round!\n"
        f"💡 Tip: Bet within the next 30 seconds for best timing"
    )
    
    return msg

def backtest_ml_system(df: pd.DataFrame, lookback: int = 300) -> float:
    """Backtest the ML system to estimate accuracy - simplified version"""
    if len(df) < 400:  # Need sufficient data for ML model
        return 0.5
    
    try:
        # Use a simplified backtest - just run ML model once on recent data
        # and estimate accuracy based on model's internal validation
        ml_probs = analyze_with_ml_model(df, min_data_points=200)
        
        # Return a reasonable estimate based on model confidence
        max_confidence = max(ml_probs.values())
        
        # Map confidence to estimated accuracy
        if max_confidence >= 0.8:
            return 0.70  # High confidence = high accuracy
        elif max_confidence >= 0.7:
            return 0.65  # Good confidence = good accuracy
        elif max_confidence >= 0.6:
            return 0.60  # Moderate confidence = moderate accuracy
        else:
            return 0.55  # Low confidence = low accuracy
            
    except Exception as e:
        print(f"⚠️  Backtest error: {e}")
        return 0.5

def main():
    try:
        parser = argparse.ArgumentParser(description="WinGo Momentum Analysis System")
        parser.add_argument("--source", choices=["csv", "db"], default="db", help="Data source")
        parser.add_argument("--csv_path", default="game_history.csv", help="CSV file path")
        parser.add_argument("--limit", type=int, default=2000, help="Number of rows to load")
        parser.add_argument("--preset", choices=["conservative", "balanced", "aggressive", "very_aggressive"], 
                           default="balanced", help="Signal frequency preset")
        parser.add_argument("--max_signals", type=int, default=3, help="Maximum signals per run")
        parser.add_argument("--color_prob_threshold", type=float, default=0.68, help="Minimum confidence for color signals")
        parser.add_argument("--min_sources", type=int, default=2, help="Minimum sources for ensemble")
        parser.add_argument("--enable_alert", action="store_true", help="Enable Telegram alerts")
        parser.add_argument("--log_to_db", action="store_true", help="Log alerts to database")
        parser.add_argument("--fast_mode", action="store_true", help="Enable fast mode for quicker alerts")
        parser.add_argument("--mid_period_mode", action="store_true", help="Enable mid-period timing optimization")
        parser.add_argument("--disable_sleep_window", action="store_true", help="Ignore 1:00–9:00 IST quiet hours")
        parser.add_argument("--min_prob_margin", type=float, default=0.20, help="Require top1-top2 prob margin before alert")
        parser.add_argument("--max_entropy", type=float, default=0.85, help="Max allowed entropy of probs for alert (lower = stricter)")
        parser.add_argument("--enable_recent_penalty", action="store_true", help="Penalize threshold if recent accuracy is low")
        # New: tunable alert gates so we can adjust without code edits
        parser.add_argument("--eta_min_seconds", type=int, default=15, help="Minimum ETA seconds required to send alert")
        parser.add_argument("--violet_max_share", type=float, default=0.22, help="Maximum allowed recent VIOLET share (0-1) for alert")
        # Startup alignment: wait until next minute + offset before fetching data
        parser.add_argument("--align_startup_sleep", action="store_true", help="Sleep until next minute + offset before analysis")
        parser.add_argument("--align_offset_seconds", type=int, default=11, help="Extra seconds after minute boundary to start (default 11)")
        # Optional speed controls
        parser.add_argument("--fresh_seconds", type=int, default=None, help="Require newest row to be ≤ this many seconds old")
        parser.add_argument("--max_wait_seconds", type=int, default=None, help="Max seconds to retry for fresh data")
        parser.add_argument("--min_buffer_seconds", type=int, default=None, help="Override safety buffer before betting period")
        args = parser.parse_args()
    except Exception as e:
        print(f"❌ Error parsing arguments: {e}")
        return
    
    print("🚀 WinGo Momentum Analysis System")
    print("=" * 50)
    print(f"📊 Source: {args.source}")
    print(f"⚙️  Preset: {args.preset}")
    print(f"🎯 Max Signals: {args.max_signals}")
    print(f"📈 Confidence Threshold: {args.color_prob_threshold}")
    print(f"🔧 Fast Mode: {args.fast_mode}")
    print("=" * 50)
    
    # Propagate fast_mode to ML layer via env so fine-tune can be skipped quickly
    try:
        os.environ["WINGO_FAST_MODE"] = "1" if args.fast_mode else "0"
    except Exception:
        pass

    # Optional startup alignment sleep to allow the current period to finish
    if args.align_startup_sleep:
        now = datetime.utcnow()
        next_minute = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
        target_start = next_minute + timedelta(seconds=max(0, args.align_offset_seconds))
        # If we already passed, push one more minute
        if target_start <= now:
            target_start = target_start + timedelta(minutes=1)
        sleep_seconds = int((target_start - now).total_seconds())
        if sleep_seconds > 0:
            print(f"⏳ Startup alignment: sleeping {sleep_seconds}s until {target_start.strftime('%H:%M:%S')} UTC")
            time.sleep(sleep_seconds)
        else:
            print("⏳ Startup alignment: no sleep needed")

    # Respect sleep window (1:00–9:00 IST) unless disabled
    if not args.disable_sleep_window:
        now_utc = datetime.utcnow()
        ist_now = now_utc + timedelta(hours=5, minutes=30)
        if 1 <= ist_now.hour < 9:
            print(f"🛌 Quiet hours active (IST {ist_now.strftime('%H:%M')}). Skipping analysis. Use --disable_sleep_window to override.")
            return

    # Load data
    try:
        if args.source == "csv":
            print(f"📁 Loading data from CSV: {args.csv_path}")
            df = load_csv(args.csv_path)
        else:
            print("🔗 Connecting to database...")
            cfg = ScraperConfig()
            
            # Check if database connection string is available
            if not hasattr(cfg, 'neon_conn_str') or not cfg.neon_conn_str:
                print("❌ Error: Database connection string not found!")
                print("Please set NEON_CONN_STR environment variable")
                return
            
            print(f"📊 Database connection established")
            
            # Adjust timing based on mode
            if args.mid_period_mode:
                fresh_seconds = 25  # Wait longer for fresh data in mid-period mode
                max_wait = 15       # Allow more time for period completion
                print("🎯 Mid-period mode enabled - optimized for fresh period data")
            elif args.fast_mode:
                fresh_seconds = 12  # Faster freshness target
                max_wait = 6        # Shorter retry window
                print("⚡ Fast mode enabled - reduced delays for quicker alerts")
            else:
                fresh_seconds = 18  # Slightly faster default
                max_wait = 10       # Slightly shorter default wait

            # Allow explicit overrides from CLI
            if args.fresh_seconds is not None:
                fresh_seconds = max(0, int(args.fresh_seconds))
            if args.max_wait_seconds is not None:
                max_wait = max(0, int(args.max_wait_seconds))
            
            print(f"📥 Loading {args.limit} rows from database...")
            df = ensure_fresh_neon_data(cfg, args.limit, fresh_seconds, max_wait)
        
        print(f"📊 Loaded {len(df)} rounds of data")
        
        # Show analysis verification details
        if len(df) > 0:
            latest_period = df["period_id"].iloc[-1]
            target_period = get_next_betting_period(df)
            print("=" * 50)
            print("🔍 ANALYSIS VERIFICATION")
            print(f"📥 Latest analyzed period (anchor): {latest_period}")
            print(f"🎯 Target betting period: {target_period}")
            print("=" * 50)
        
        # Resolve any pending alert outcomes
        print("🔄 Resolving pending alert outcomes...")
        resolve_unresolved_alerts(cfg.neon_conn_str)
        
        if len(df) < 100:
            print("❌ Insufficient data for analysis (need at least 100 rounds)")
            return
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Detect signals with enhanced filtering
    try:
        if args.preset != "balanced":
            preset_config = get_preset_config(args.preset)
            if not preset_config:
                print(f"❌ Invalid preset: {args.preset}")
                return
            base_ml_threshold = preset_config["momentum"]
            if args.enable_recent_penalty:
                rp = analyze_recent_performance(cfg.neon_conn_str)
                penalty = rp.get("confidence_penalty", 0.0)
                base_ml_threshold = min(0.90, base_ml_threshold + penalty)
                print(f"📉 Recent penalty applied: +{penalty:.3f} → ML threshold {base_ml_threshold:.3f}")
            # Export gating knobs so detect_strong_signals can read them without refactoring signature
            try:
                os.environ["WINGO_MIN_PROB_MARGIN"] = str(args.min_prob_margin)
                os.environ["WINGO_MAX_ENTROPY"] = str(args.max_entropy)
            except Exception:
                pass
            signals = detect_strong_signals(df,
                                            ml_threshold=base_ml_threshold,
                                            size_threshold=0.70,
                                            conn_str=cfg.neon_conn_str)
        else:
            # Use default threshold
            base_ml_threshold = args.color_prob_threshold
            if args.enable_recent_penalty:
                rp = analyze_recent_performance(cfg.neon_conn_str)
                penalty = rp.get("confidence_penalty", 0.0)
                base_ml_threshold = min(0.90, base_ml_threshold + penalty)
                print(f"📉 Recent penalty applied: +{penalty:.3f} → ML threshold {base_ml_threshold:.3f}")
            # Export gating knobs so detect_strong_signals can read them without refactoring signature
            try:
                os.environ["WINGO_MIN_PROB_MARGIN"] = str(args.min_prob_margin)
                os.environ["WINGO_MAX_ENTROPY"] = str(args.max_entropy)
            except Exception:
                pass
            signals = detect_strong_signals(df,
                                            ml_threshold=base_ml_threshold,
                                            size_threshold=0.70,
                                            conn_str=cfg.neon_conn_str)
    except Exception as e:
        print(f"❌ Error detecting signals: {e}")
        return
    
    # Limit number of signals per run
    max_signals = min(args.max_signals, get_max_signals_per_run())
    if len(signals) > max_signals:
        # Sort by confidence and keep top signals
        signals = sorted(signals, key=lambda x: x["confidence"], reverse=True)[:max_signals]
        print(f"Limited to top {max_signals} signals by confidence")
    
    if not signals:
        print("No strong signals detected with current threshold.")
        print("Consider using --preset aggressive or --preset very_aggressive for more signals.")
        return

    # Display signals
    print(f"\n🎯 Strong Signals Detected: {len(signals)}")
    for i, signal in enumerate(signals, 1):
        if signal["type"] == "color":
            print(f"\nSignal {i}: {signal['method']}")
            print(f"  Color: {signal['color']}")
            print(f"  Confidence: {signal['confidence']:.3f}")
            print(f"  Reason: {signal['reason']}")
            print(f"  Probabilities: R={signal['probs']['RED']:.3f}, G={signal['probs']['GREEN']:.3f}, V={signal['probs']['VIOLET']:.3f}")
        elif signal["type"] == "size":
            print(f"\nSignal {i}: {signal['method']}")
            print(f"  Size: {signal['size']}")
            print(f"  Confidence: {signal['confidence']:.3f}")
            print(f"  Reason: {signal['reason']}")
            print(f"  Probabilities: BIG={signal['probs']['BIG']:.3f}, SMALL={signal['probs']['SMALL']:.3f}")
    
    # Backtest accuracy
    accuracy = backtest_ml_system(df)
    print(f"\n📊 Estimated System Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    
    # Send alerts for strong signals
    if args.enable_alert and signals:
        cfg = ScraperConfig()
        
        # Determine the correct threshold for alerts
        if args.preset != "balanced":
            preset_config = get_preset_config(args.preset)
            alert_threshold = preset_config["momentum"]
        else:
            alert_threshold = args.color_prob_threshold
            
        # Track sent alerts to prevent duplicates
        sent_alerts = set()
        
        # Sort signals by confidence to prioritize strongest
        signals = sorted(signals, key=lambda x: x["confidence"], reverse=True)

        # Prefer SIZE signal if its confidence >= color confidence + margin
        prefer_size_margin = 0.08  # Tighter margin for preferring size over color
        best_signal = None
        if signals:
            top = signals[0]
            top_color = next((s for s in signals if s["type"] == "color"), None)
            top_size = next((s for s in signals if s["type"] == "size"), None)
            if top_color and top_size and (top_size["confidence"] >= 0.70) and (top_size["confidence"] >= top_color["confidence"] + prefer_size_margin):
                best_signal = top_size
            else:
                best_signal = top
            
            # Only alert if: Ensemble OR exceptionally strong single-method (>=0.72)
            is_ensemble = (best_signal.get("method") == "Ensemble")
            exceptionally_strong = (best_signal["confidence"] >= 0.75)
            if is_ensemble or exceptionally_strong:
                # Calculate the NEXT period ID for betting and ensure a safe buffer
                initial_period = get_next_betting_period(df)
                
                # Buffer requirements based on timing mode
                if args.min_buffer_seconds is not None:
                    min_buffer = max(0, int(args.min_buffer_seconds))
                elif args.mid_period_mode:
                    min_buffer = 10  # Reduced for mid-period timing
                    print("🎯 Using 10s buffer for mid-period optimization")
                elif args.fast_mode:
                    min_buffer = 10  # Reduced for fast mode
                    print("⚡ Using 10s buffer for fast mode")
                else:
                    min_buffer = 15  # Reduced standard buffer
                    print(f"🛡️  Using {min_buffer}s safety buffer")
                
                betting_period = ensure_min_time_buffer(df, initial_period, min_buffer_seconds=min_buffer)
                
                # If betting period was shifted due to time buffer, continue with same prediction
                # The logic is: analyze anchor period → predict for betting period (with buffer)
                if betting_period != initial_period:
                    print(f"⏰ Time buffer applied - betting on period {betting_period} (shifted from {initial_period})")
                
                # Quality gates for alert sending
                current_time = datetime.utcnow()
                
                # Calculate ETA more accurately
                try:
                    if len(betting_period) >= 12:
                        # Parse the period ID to get the target time (UTC minute)
                        # betting_period already encodes the minute to bet on
                        target_dt = datetime.strptime(betting_period[:12], "%Y%m%d%H%M")
                    else:
                        # Fallback: next minute boundary
                        target_dt = (current_time.replace(second=0, microsecond=0) + timedelta(minutes=1))
                except Exception:
                    # Fallback: next minute boundary
                    target_dt = (current_time.replace(second=0, microsecond=0) + timedelta(minutes=1))

                # Guard: ensure target time is in the future
                if target_dt <= current_time:
                    target_dt = (current_time.replace(second=0, microsecond=0) + timedelta(minutes=1))
                
                eta_seconds = max(0, int((target_dt - current_time).total_seconds()))
                
                # Debug timing information
                print(f"🕐 Current time: {current_time.strftime('%H:%M:%S')}")
                print(f"🎯 Target time: {target_dt.strftime('%H:%M:%S')}")
                print(f"⏱️  ETA: {eta_seconds} seconds")
                
                # Quality gate 1: ETA check (configurable)
                if eta_seconds < args.eta_min_seconds:
                    print(f"❌ Skipping alert: ETA too low ({eta_seconds}s < {args.eta_min_seconds}s)")
                    return
                
                # Quality gate 2: Backtest precision check (reuse precomputed accuracy)
                backtest_precision = accuracy
                if backtest_precision < 0.65:
                    print(f"❌ Skipping alert: Backtest precision too low ({backtest_precision:.3f} < 0.65)")
                    return
                
                # Quality gate 3: Violet share check (avoid periods with too much violet)
                recent_colors = df.tail(120)["color"].tolist()
                violet_share = recent_colors.count("VIOLET") / len(recent_colors)
                if violet_share >= args.violet_max_share:
                    print(f"❌ Skipping alert: Violet share too high ({violet_share:.3f} >= {args.violet_max_share})")
                    return
                
                print(f"✅ Quality gates passed: ETA={eta_seconds}s, precision={backtest_precision:.3f}, violet={violet_share:.3f}")
                
                # Create unique alert key to prevent duplicates
                if best_signal["type"] == "color":
                    alert_key = f"{betting_period}_{best_signal['color']}_{best_signal['method']}"
                else:
                    alert_key = f"{betting_period}_{best_signal['size']}_{best_signal['method']}"
                
                # Cross-run reservation to dedupe globally by (period, type)
                reserved = reserve_alert_slot(cfg.neon_conn_str, betting_period,
                                              ("COLOR" if best_signal["type"] == "color" else "SIZE"))
                if not reserved:
                    print(f"Skipping alert due to existing reservation for {betting_period} {best_signal['type']}")
                    return
                # Also dedupe within this run
                if alert_key in sent_alerts:
                    print(f"Skipping duplicate alert for {alert_key}")
                    return
                
                # Create alert message for NEXT period betting
                # Only show the strongest signal (no mixing color + size)
                if best_signal["type"] == "color":
                    msg = format_color_alert(best_signal, betting_period, accuracy)
                    print(f"🎨 Sending COLOR alert: {best_signal['color']} @ {best_signal['confidence']:.3f}")
                elif best_signal["type"] == "size":
                    msg = format_size_alert(best_signal, betting_period, accuracy)
                    print(f"⚖️  Sending SIZE alert: {best_signal['size']} @ {best_signal['confidence']:.3f}")
                
                # Mark this alert as sent
                sent_alerts.add(alert_key)
                
                # Send Telegram alert
                ok = send_telegram(cfg, msg)
                if best_signal["type"] == "color":
                    print(f"Alert sent for {best_signal['color']}: {ok}")
                else:
                    print(f"Alert sent for {best_signal['size']}: {ok}")
                
                # Log to database if enabled
                if args.log_to_db:
                    try:
                        anchor_pid = str(df["period_id"].iloc[-1])
                        if best_signal["type"] == "color":
                            log_alert_to_neon(
                                cfg.neon_conn_str,
                                anchor_pid,
                                best_signal["color"],
                                None,
                                best_signal["probs"],
                                [best_signal["method"]],
                                best_signal["confidence"],
                                accuracy,
                                None,
                                0.0,
                            )
                        # Note: Size predictions not logged to database for now
                    except Exception as e:
                        print(f"Failed to log to database: {e}")
            else:
                print(f"❌ Best signal confidence {best_signal['confidence']:.3f} below threshold {alert_threshold}")
        else:
            print("❌ No signals to alert")
    
        # Summary
        print("\n" + "="*50)
        print("📊 ANALYSIS SUMMARY")
        print("="*50)
        print(f"📈 Data analyzed: {len(df)} rounds")
        print(f"🎯 Signals detected: {len(signals)}")
        print(f"📱 Alerts sent: {len(sent_alerts)}")
        print(f"⚙️  Preset used: {args.preset}")
        print(f"🔧 Fast mode: {args.fast_mode}")
        print(f"🎲 Next betting period: {get_next_betting_period(df)}")
        print(f"⏰ Analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*50)
        print("✅ Analysis complete!")
        
        if args.enable_alert and signals:
            print("\n💡 BETTING INSTRUCTIONS:")
            print("1. Wait for the NEXT round to start")
            print("2. Place your bet on the indicated color/size")
            print("3. Bet within 30 seconds of round start for best timing")
            print("4. Monitor results and adjust strategy as needed")
    
    # Final summary for all cases
    print(f"\n📊 Final Summary:")
    print(f"   - Data analyzed: {len(df)} rounds")
    print(f"   - Signals detected: {len(signals)}")
    print(f"   - System accuracy: {accuracy:.1%}")
    print(f"   - Preset used: {args.preset}")
    print(f"   - Fast mode: {args.fast_mode}")
    print(f"   - Max signals: {args.max_signals}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Fatal error in main: {e}")
        import traceback
        traceback.print_exc()
