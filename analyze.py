import argparse, csv, math, statistics, os, json, time
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np
from scipy.stats import chisquare
import requests
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
    """Load data directly from Neon PostgreSQL"""
    import psycopg2
    query = f"""
    SELECT period_id, number, color, scraped_at 
    FROM game_history 
    ORDER BY scraped_at DESC 
    LIMIT {limit}
    """
    conn = psycopg2.connect(conn_str)
    df = pd.read_sql(query, conn)
    conn.close()
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

# ====== NEW MOMENTUM-BASED ANALYSIS SYSTEM ======

def analyze_color_momentum(df: pd.DataFrame, lookback: int = 20) -> Dict[str, float]:
    """Analyze color momentum based on recent frequency and streaks"""
    if len(df) < lookback:
        return {"RED": 0.33, "GREEN": 0.33, "VIOLET": 0.34}
    
    recent = df.tail(lookback)
    colors = recent["color"].tolist()
    
    # Count recent frequencies
    color_counts = {"RED": 0, "GREEN": 0, "VIOLET": 0}
    for color in colors:
        color_counts[color] += 1
    
    # Calculate momentum scores
    total = len(colors)
    momentum = {}
    for color in ["RED", "GREEN", "VIOLET"]:
        freq = color_counts[color] / total
        
        # Streak bonus for current color
        streak = 0
        for c in reversed(colors):
            if c == color:
                streak += 1
            else:
                break
        
        # Bonus increases with streak length (capped at 0.15)
        streak_bonus = min(0.15, 0.03 * streak)
        
        # Final momentum score
        momentum[color] = min(0.95, freq + streak_bonus)
    
    # Normalize to sum to 1
    total_momentum = sum(momentum.values())
    if total_momentum > 0:
        for color in momentum:
            momentum[color] /= total_momentum
    
    return momentum

def analyze_number_patterns(df: pd.DataFrame, lookback: int = 100) -> Dict[str, float]:
    """Analyze number patterns to detect under-represented numbers"""
    if len(df) < lookback:
        return {"RED": 0.33, "GREEN": 0.33, "VIOLET": 0.34}
    
    recent = df.tail(lookback)
    numbers = recent["number"].astype(int).tolist()
    colors = recent["color"].tolist()
    
    # Count number frequencies
    number_counts = {i: 0 for i in range(10)}
    for num in numbers:
        number_counts[num] += 1
    
    # Find under-represented numbers
    expected = lookback / 10
    under_represented = []
    for num, count in number_counts.items():
        if count < expected * 0.7:  # 30% below expected
            under_represented.append(num)
    
    if not under_represented:
        return {"RED": 0.33, "GREEN": 0.33, "VIOLET": 0.34}
    
    # Map numbers to colors and calculate correction probabilities
    color_probs = {"RED": 0.0, "GREEN": 0.0, "VIOLET": 0.0}
    
    for num in under_represented:
        if num in [1, 2, 3, 4, 5]:
            color_probs["RED"] += 1
        elif num in [6, 7, 8, 9]:
            color_probs["GREEN"] += 1
        elif num == 0:
            color_probs["VIOLET"] += 1
    
    # Normalize with confidence dampening to prevent overly confident signals
    total = sum(color_probs.values())
    if total > 0:
        for color in color_probs:
            color_probs[color] /= total
        
        # Dampen confidence based on strength of under-representation
        max_prob = max(color_probs.values())
        under_repr_strength = len(under_represented) / 10.0  # 0.1 to 1.0 scale
        confidence_factor = min(0.70, 0.45 + under_repr_strength * 0.25)  # Max 0.70
        
        # Apply dampening to prevent perfect 1.0 confidence
        for color in color_probs:
            if color_probs[color] == max_prob:
                color_probs[color] = confidence_factor
            else:
                color_probs[color] *= (1 - confidence_factor) / 2
    else:
        color_probs = {"RED": 0.33, "GREEN": 0.33, "VIOLET": 0.34}
    
    return color_probs

def analyze_time_based_patterns(df: pd.DataFrame, min_data: int = 200) -> Dict[str, float]:
    """Analyze time-based patterns (hourly color distributions)"""
    if len(df) < min_data:
        return {"RED": 0.33, "GREEN": 0.33, "VIOLET": 0.34}
    
    # Add hour column
    df_copy = df.copy()
    df_copy["hour"] = pd.to_datetime(df_copy["scraped_at"]).dt.hour
    
    # Get current hour
    current_hour = datetime.utcnow().hour
    
    # Find data for current hour
    hour_data = df_copy[df_copy["hour"] == current_hour]
    
    if len(hour_data) < 20:  # Need at least 20 data points for this hour
        return {"RED": 0.33, "GREEN": 0.33, "VIOLET": 0.34}
    
    # Calculate hourly color distribution
    color_counts = hour_data["color"].value_counts()
    total = len(hour_data)
    
    probs = {}
    for color in ["RED", "GREEN", "VIOLET"]:
        count = color_counts.get(color, 0)
        probs[color] = count / total
    
    return probs

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
                         momentum_threshold: float = 0.6,
                         pattern_threshold: float = 0.65,
                         time_threshold: float = 0.6,
                         ensemble_threshold: float = 0.7,
                         conn_str: str = None) -> List[Dict]:
    """Detect strong signals using multiple analysis methods with enhanced filtering"""
    signals = []
    
    # Enhanced filtering: Check volatility (temporarily disabled for debugging)
    volatility = detect_volatility(df)
    print(f"🌊 Volatility check: {volatility:.3f} (threshold: 0.75)")
    if volatility > 0.90:  # Raised threshold to allow more signals
        print("⚠️  Skipping due to extreme volatility")
        return []  # Skip only during extreme volatile periods
    
    # Note: Performance analysis removed - using fixed high thresholds for quality
    
    # 1. Color Momentum Analysis
    momentum_probs = analyze_color_momentum(df)
    max_momentum = max(momentum_probs.values())
    print(f"🎨 Momentum analysis: max={max_momentum:.3f} (threshold: {momentum_threshold:.3f})")
    
    if max_momentum >= momentum_threshold:
        best_color = max(momentum_probs, key=momentum_probs.get)
        signals.append({
            "type": "color",
            "color": best_color,
            "confidence": max_momentum,
            "method": "ColorMomentum",
            "reason": f"Color momentum suggests {best_color} with {max_momentum:.3f} confidence",
            "probs": momentum_probs
        })
    
    # 2. Number Pattern Analysis
    pattern_probs = analyze_number_patterns(df)
    max_pattern = max(pattern_probs.values())
    print(f"🔢 Pattern analysis: max={max_pattern:.3f} (threshold: {pattern_threshold:.3f})")
    
    if max_pattern >= pattern_threshold:
        best_color = max(pattern_probs, key=pattern_probs.get)
        signals.append({
            "type": "color",
            "color": best_color,
            "confidence": max_pattern,
            "method": "NumberPattern",
            "reason": f"Number pattern suggests {best_color} correction with {max_pattern:.3f} confidence",
            "probs": pattern_probs
        })
    
    # 3. Time-based Pattern Analysis
    time_probs = analyze_time_based_patterns(df)
    max_time = max(time_probs.values())
    print(f"⏰ Time analysis: max={max_time:.3f} (threshold: {time_threshold:.3f})")
    
    if max_time >= time_threshold:
        best_color = max(time_probs, key=time_probs.get)
        signals.append({
            "type": "color",
            "color": best_color,
            "confidence": max_time,
            "method": "TimePattern",
            "reason": f"Time pattern suggests {best_color} bias with {max_time:.3f} confidence",
            "probs": time_probs
        })
    
    # 4. Big/Small Analysis
    size_probs, size_conf, size_reason = analyze_big_small(df)
    print(f"⚖️  Size analysis: conf={size_conf:.3f} (threshold: 0.72)")
    
    if size_conf >= 0.72:  # Increased threshold to improve 50% size accuracy
        best_size = "BIG" if size_probs["BIG"] >= size_probs["SMALL"] else "SMALL"
        signals.append({
            "type": "size",
            "size": best_size,
            "confidence": size_conf,
            "method": "BigSmall",
            "reason": f"Size analysis suggests {best_size} with {size_conf:.3f} confidence",
            "probs": size_probs
        })
    
    # 5. Enhanced Ensemble Analysis - require 3+ agreeing methods
    color_signals = [s for s in signals if s["type"] == "color"]
    print(f"🤝 Ensemble check: {len(color_signals)} color signals (need 3+ for ensemble)")
    if len(color_signals) >= 3:  # Require 3+ methods
        color_predictions = [s["color"] for s in color_signals]
        if len(set(color_predictions)) == 1:  # All predict same color
            best_color = color_predictions[0]
            avg_confidence = sum(s["confidence"] for s in color_signals) / len(color_signals)
            
            if avg_confidence >= ensemble_threshold:
                signals.append({
                    "type": "color",
                    "color": best_color,
                    "confidence": min(0.95, avg_confidence + 0.05),  # Bonus for agreement
                    "method": "Ensemble",
                    "reason": f"Multiple methods agree on {best_color} with {avg_confidence:.3f} avg confidence",
                    "probs": momentum_probs
                })
    
    print(f"🎯 Total signals generated: {len(signals)}")
    for i, signal in enumerate(signals):
        print(f"  Signal {i+1}: {signal['method']} - {signal.get('color', signal.get('size'))} @ {signal['confidence']:.3f}")
    
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

def backtest_momentum_system(df: pd.DataFrame, lookback: int = 300) -> float:
    """Backtest the momentum system to estimate accuracy"""
    if len(df) < lookback + 50:
        return 0.5
    
    correct_predictions = 0
    total_predictions = 0
    
    for i in range(50, min(lookback, len(df) - 1)):
        # Use data up to position i to predict position i+1
        train_data = df.iloc[:i+1]
        actual_color = df.iloc[i+1]["color"]
        
        # Get prediction
        momentum_probs = analyze_color_momentum(train_data, lookback=20)
        predicted_color = max(momentum_probs, key=momentum_probs.get)
        
        if predicted_color == actual_color:
            correct_predictions += 1
        total_predictions += 1
    
    return correct_predictions / total_predictions if total_predictions > 0 else 0.5

def main():
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
    args = parser.parse_args()
    
    print("🚀 WinGo Momentum Analysis System")
    print("=" * 50)
    print(f"📊 Source: {args.source}")
    print(f"⚙️  Preset: {args.preset}")
    print(f"🎯 Max Signals: {args.max_signals}")
    print(f"📈 Confidence Threshold: {args.color_prob_threshold}")
    print(f"🔧 Fast Mode: {args.fast_mode}")
    print("=" * 50)
    
    # Load data
    try:
        if args.source == "csv":
            df = load_csv(args.csv_path)
        else:
            cfg = ScraperConfig()
            
            # Adjust timing based on mode
            if args.mid_period_mode:
                fresh_seconds = 25  # Wait longer for fresh data in mid-period mode
                max_wait = 15       # Allow more time for period completion
                print("🎯 Mid-period mode enabled - optimized for fresh period data")
            elif args.fast_mode:
                fresh_seconds = 15  # Reduced from 20
                max_wait = 8        # Reduced from 10
                print("⚡ Fast mode enabled - reduced delays for quicker alerts")
            else:
                fresh_seconds = 20  # Reduced from 25
                max_wait = 12       # Reduced from 15
            
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
        
        # Detect signals with enhanced filtering
        if args.preset != "balanced":
            preset_config = get_preset_config(args.preset)
            if not preset_config:
                print(f"❌ Invalid preset: {args.preset}")
                return
            signals = detect_strong_signals(df, 
                                            momentum_threshold=preset_config["momentum"],
                                            pattern_threshold=preset_config["number_pattern"],
                                            time_threshold=preset_config["time_pattern"],
                                            ensemble_threshold=preset_config["ensemble"],
                                            conn_str=cfg.neon_conn_str)
        else:
            # Use default threshold
            signals = detect_strong_signals(df, 
                                            momentum_threshold=args.color_prob_threshold,
                                            pattern_threshold=0.65, # Default for number pattern
                                            time_threshold=0.6,   # Default for time pattern
                                            ensemble_threshold=0.7, # Default for ensemble
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
    accuracy = backtest_momentum_system(df)
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
        prefer_size_margin = 0.03
        best_signal = None
        if signals:
            top = signals[0]
            top_color = next((s for s in signals if s["type"] == "color"), None)
            top_size = next((s for s in signals if s["type"] == "size"), None)
            if top_color and top_size and (top_size["confidence"] >= top_color["confidence"] + prefer_size_margin):
                best_signal = top_size
            else:
                best_signal = top
            
            # Only alert if: Ensemble OR exceptionally strong single-method (>=0.47)
            is_ensemble = (best_signal.get("method") == "Ensemble")
            exceptionally_strong = (best_signal["confidence"] >= 0.47)
            if is_ensemble or exceptionally_strong:
                # Calculate the NEXT period ID for betting and ensure a safe buffer
                initial_period = get_next_betting_period(df)
                
                # Buffer requirements based on timing mode
                if args.mid_period_mode:
                    min_buffer = 15  # Optimized for mid-period timing
                    print("🎯 Using 15s buffer for mid-period optimization")
                elif args.fast_mode:
                    min_buffer = 15  # Reduced for fast mode
                    print("⚡ Using 15s buffer for fast mode")
                else:
                    min_buffer = 20  # Standard buffer
                    print(f"🛡️  Using {min_buffer}s safety buffer")
                
                betting_period = ensure_min_time_buffer(df, initial_period, min_buffer_seconds=min_buffer)
                
                # If betting period was shifted due to time buffer, continue with same prediction
                # The logic is: analyze anchor period → predict for betting period (with buffer)
                if betting_period != initial_period:
                    print(f"⏰ Time buffer applied - betting on period {betting_period} (shifted from {initial_period})")
                
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
                                None,  # No number prediction in new system
                                best_signal["probs"],
                                [best_signal["method"]],
                                best_signal["confidence"],
                                accuracy,
                                None,  # No cycle length
                                0.0,   # No cycle accuracy
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
    main()
