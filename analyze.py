import argparse, csv, math, statistics, os, json
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
    import numpy as np as _np  # local alias to protect global
    M = _np.full((2,2), alpha, dtype=float)
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
    Works in the same database; adds outcome columns if missing. Safe no-op on errors.
    """
    try:
        import psycopg2
        with psycopg2.connect(conn_str) as conn:
            with conn.cursor() as cur:
                # Ensure table exists and outcome columns are present
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
                    ALTER TABLE prediction_alerts
                    ADD COLUMN IF NOT EXISTS outcome_color TEXT,
                    ADD COLUMN IF NOT EXISTS outcome_number INT,
                    ADD COLUMN IF NOT EXISTS hit_color BOOLEAN,
                    ADD COLUMN IF NOT EXISTS hit_number BOOLEAN,
                    ADD COLUMN IF NOT EXISTS resolved_at timestamptz;
                    """
                )
                # Resolve a batch using the next period in game_history
                cur.execute(
                    """
                    WITH unresolved AS (
                        SELECT id, anchor_period_id
                        FROM prediction_alerts
                        WHERE resolved_at IS NULL
                        ORDER BY created_at ASC
                        LIMIT %s
                    ), next_round AS (
                        SELECT u.id,
                               g.number AS outcome_number,
                               g.color  AS outcome_color
                        FROM unresolved u
                        JOIN LATERAL (
                            SELECT number, color
                            FROM game_history
                            WHERE period_id > u.anchor_period_id
                            ORDER BY period_id
                            LIMIT 1
                        ) g ON TRUE
                    )
                    UPDATE prediction_alerts p
                    SET outcome_number = nr.outcome_number,
                        outcome_color  = nr.outcome_color,
                        hit_number     = CASE WHEN p.predicted_number IS NOT NULL THEN (p.predicted_number = nr.outcome_number) ELSE NULL END,
                        hit_color      = (p.predicted_color = nr.outcome_color),
                        resolved_at    = now()
                    FROM next_round nr
                    WHERE p.id = nr.id;
                    """,
                    (batch_limit,)
                )
            conn.commit()
    except Exception:
        # Avoid breaking analysis if resolution fails
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
            # Fallback: next minute
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

def ensure_min_time_buffer(df: pd.DataFrame, betting_period: str, min_buffer_seconds: int = 35) -> str:
    """If the computed betting_period starts in less than `min_buffer_seconds`,
    shift it forward by one more minute to guarantee user has time to bet.
    Works with both numeric-only ids and ids that start with YYYYMMDDHHMM.
    """
    now_utc = datetime.utcnow()
    try:
        if len(betting_period) >= 12 and betting_period[:12].isdigit():
            target_dt = datetime.strptime(betting_period[:12], "%Y%m%d%H%M")
            seconds_until = (target_dt - now_utc).total_seconds()
            if seconds_until < min_buffer_seconds:
                # push one minute ahead but keep suffix
                suffix = betting_period[12:]
                new_dt = target_dt + timedelta(minutes=1)
                return f"{new_dt.strftime('%Y%m%d%H%M')}{suffix}"
            return betting_period
        # Numeric-only fallback
        val = int(betting_period)
        # When we cannot parse time, still add +1 as conservative bump if we are too close
        # Heuristic: if the latest row is within 20s of the minute boundary, bump +1
        last_ts = pd.to_datetime(df["scraped_at"].iloc[-1], errors='coerce')
        if pd.notna(last_ts):
            sec = last_ts.second
            if sec >= 40:
                return str(val + 1)
        return betting_period
    except Exception:
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
    momentum_scores = {}
    
    for color in ["RED", "GREEN", "VIOLET"]:
        # Base frequency
        freq_score = color_counts[color] / total
        
        # Streak bonus (consecutive appearances)
        streak_bonus = 0
        current_streak = 0
        for c in reversed(colors):
            if c == color:
                current_streak += 1
                streak_bonus += 0.05 * current_streak
            else:
                break
        
        # Recent bias (last 5 rounds)
        recent_bias = 0
        last_5 = colors[-5:] if len(colors) >= 5 else colors
        recent_count = last_5.count(color)
        if recent_count > 0:
            recent_bias = 0.1 * recent_count
        
        # Combine scores
        momentum_scores[color] = freq_score + streak_bonus + recent_bias
    
    # Normalize to probabilities
    total_score = sum(momentum_scores.values())
    if total_score > 0:
        return {color: score / total_score for color, score in momentum_scores.items()}
    else:
        return {"RED": 0.33, "GREEN": 0.33, "VIOLET": 0.34}

def analyze_number_patterns(df: pd.DataFrame, lookback: int = 30) -> Dict[str, float]:
    """Analyze number patterns for better predictions"""
    if len(df) < lookback:
        return {"RED": 0.33, "GREEN": 0.33, "VIOLET": 0.34}
    
    recent = df.tail(lookback)
    numbers = recent["number"].tolist()
    
    # Analyze number distribution
    number_counts = [0] * 10
    for num in numbers:
        number_counts[num] += 1
    
    # Find under-represented numbers (potential for correction)
    avg_count = len(numbers) / 10
    under_rep = []
    for i, count in enumerate(number_counts):
        if count < avg_count * 0.7:  # 30% below average
            under_rep.append(i)
    
    # Calculate color probabilities based on under-represented numbers
    red_prob = sum(1 for i in under_rep if i % 2 == 1) / max(len(under_rep), 1)
    green_prob = sum(1 for i in under_rep if i % 2 == 0 and i not in [0, 5]) / max(len(under_rep), 1)
    violet_prob = sum(1 for i in under_rep if i in [0, 5]) / max(len(under_rep), 1)
    
    # Normalize
    total = red_prob + green_prob + violet_prob
    if total > 0:
        return {
            "RED": red_prob / total,
            "GREEN": green_prob / total,
            "VIOLET": violet_prob / total
        }
    else:
        return {"RED": 0.33, "GREEN": 0.33, "VIOLET": 0.34}

def analyze_time_based_patterns(df: pd.DataFrame) -> Dict[str, float]:
    """Analyze patterns based on time of day"""
    if len(df) < 50:
        return {"RED": 0.33, "GREEN": 0.33, "VIOLET": 0.34}
    
    # Add time information
    df_with_time = df.copy()
    df_with_time["hour"] = pd.to_datetime(df_with_time["scraped_at"]).dt.hour
    
    # Group by hour and analyze color distribution
    hourly_colors = {}
    for hour in range(24):
        hour_data = df_with_time[df_with_time["hour"] == hour]
        if len(hour_data) >= 10:  # Need sufficient data
            colors = hour_data["color"].tolist()
            hourly_colors[hour] = {
                "RED": colors.count("RED") / len(colors),
                "GREEN": colors.count("GREEN") / len(colors),
                "VIOLET": colors.count("VIOLET") / len(colors)
            }
    
    # Get current hour
    current_hour = datetime.now().hour
    
    # Find similar hours (within 2 hours)
    similar_hours = []
    for hour in hourly_colors:
        if abs(hour - current_hour) <= 2:
            similar_hours.append(hour)
    
    if similar_hours:
        # Average probabilities from similar hours
        avg_probs = {"RED": 0, "GREEN": 0, "VIOLET": 0}
        for hour in similar_hours:
            for color in ["RED", "GREEN", "VIOLET"]:
                avg_probs[color] += hourly_colors[hour][color]
        
        # Normalize
        total = sum(avg_probs.values())
        if total > 0:
            return {color: prob / total for color, prob in avg_probs.items()}
    
    return {"RED": 0.33, "GREEN": 0.33, "VIOLET": 0.34}

def detect_strong_signals(df: pd.DataFrame, min_confidence: float = 0.65) -> List[Dict]:
    """Detect strong signals using multiple analysis methods"""
    signals = []
    
    # Method 1: Color Momentum
    momentum_probs = analyze_color_momentum(df, lookback=25)
    momentum_confidence = max(momentum_probs.values())
    if momentum_confidence >= min_confidence:
        top_color = max(momentum_probs, key=momentum_probs.get)
        signals.append({
            "method": "Momentum",
            "color": top_color,
            "confidence": momentum_confidence,
            "probs": momentum_probs,
            "reason": f"Strong {top_color} momentum with {momentum_confidence:.2f} confidence"
        })
    
    # Method 2: Number Pattern Correction
    number_probs = analyze_number_patterns(df, lookback=35)
    number_confidence = max(number_probs.values())
    if number_confidence >= min_confidence:
        top_color = max(number_probs, key=number_probs.get)
        signals.append({
            "method": "NumberPattern",
            "color": top_color,
            "confidence": number_confidence,
            "probs": number_probs,
            "reason": f"Number pattern suggests {top_color} correction with {number_confidence:.2f} confidence"
        })
    
    # Method 3: Time-based Patterns
    time_probs = analyze_time_based_patterns(df)
    time_confidence = max(time_probs.values())
    if time_confidence >= min_confidence:
        top_color = max(time_probs, key=time_probs.get)
        signals.append({
            "method": "TimePattern",
            "color": top_color,
            "confidence": time_confidence,
            "probs": time_probs,
            "reason": f"Time-based pattern favors {top_color} with {time_confidence:.2f} confidence"
        })
    
    # Method 4: Combined Analysis (ensemble)
    if len(signals) >= 2:
        # Combine probabilities from multiple methods
        combined_probs = {"RED": 0, "GREEN": 0, "VIOLET": 0}
        total_weight = 0
        
        for signal in signals:
            weight = signal["confidence"]
            for color, prob in signal["probs"].items():
                combined_probs[color] += prob * weight
            total_weight += weight
        
        if total_weight > 0:
            combined_probs = {color: prob / total_weight for color, prob in combined_probs.items()}
            combined_confidence = max(combined_probs.values())
            
            if combined_confidence >= min_confidence + 0.05:  # Higher threshold for ensemble
                top_color = max(combined_probs, key=combined_probs.get)
                signals.append({
                    "method": "Ensemble",
                    "color": top_color,
                    "confidence": combined_confidence,
                    "probs": combined_probs,
                    "reason": f"Multiple methods agree on {top_color} with {combined_confidence:.2f} confidence"
                })
    
    return signals

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
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["csv", "db"], default="db")
    ap.add_argument("--path", default="data/history.csv")
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--enable_alert", action="store_true")
    ap.add_argument("--log_to_db", action="store_true")
    ap.add_argument("--color_prob_threshold", type=float, default=0.60)
    ap.add_argument("--min_sources", type=int, default=1)
    ap.add_argument("--preset", choices=["conservative", "balanced", "aggressive", "very_aggressive"], 
                   default="balanced", help="Use preset configuration for signal frequency")
    ap.add_argument("--max_signals", type=int, default=5, help="Maximum signals per analysis run")
    args = ap.parse_args()

    # Load preset configuration if specified
    if args.preset != "balanced":
        preset_config = get_preset_config(args.preset)
        print(f"Using {args.preset} preset configuration:")
        for method, threshold in preset_config.items():
            print(f"  {method}: {threshold}")
        print()

    # Load data
    try:
    if args.source == "csv":
        df = load_csv(args.path)
    else:
        cfg = ScraperConfig()
        df = load_neon(cfg.neon_conn_str, limit=args.limit)

        if df is None or df.empty:
            print("❌ No data loaded. Check database connection or CSV file.")
            return
            
        if len(df) < 100:
            print(f"⚠️  Insufficient data: {len(df)} rows. Need at least 100.")
            print("   The system will work better with more historical data.")
            if len(df) < 50:
                print("   Consider waiting for more data before running analysis.")
                return
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        print("   Check your database connection and credentials.")
        return

    print(f"=== WinGo Momentum Analysis ===")
    print(f"Data loaded: {len(df)} rows")
    print(f"Analysis preset: {args.preset}")
    print(f"Max signals per run: {args.max_signals}")
    print(f"Data source: {args.source}")
    print(f"Data limit: {args.limit}")
    print()
    
    # Detect strong signals with preset configuration
    try:
        if args.preset != "balanced":
            # Use preset thresholds
            preset_config = get_preset_config(args.preset)
            if not preset_config:
                print(f"❌ Invalid preset: {args.preset}")
                return
            signals = detect_strong_signals(df, min_confidence=preset_config["momentum"])
        else:
            # Use default threshold
            signals = detect_strong_signals(df, min_confidence=args.color_prob_threshold)
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
        print(f"\nSignal {i}: {signal['method']}")
        print(f"  Color: {signal['color']}")
        print(f"  Confidence: {signal['confidence']:.3f}")
        print(f"  Reason: {signal['reason']}")
        print(f"  Probabilities: RED={signal['probs']['RED']:.3f}, GREEN={signal['probs']['GREEN']:.3f}, VIOLET={signal['probs']['VIOLET']:.3f}")
    
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
        
        # Compute Big/Small once per run from the latest data
        size_probs, size_conf, size_reason = analyze_big_small(df, lookback=60)
        size_line = f"\n⚖️  Size: {'BIG' if size_probs['BIG']>=0.5 else 'SMALL'} @ {max(size_probs.values()):.2f} (reason: {size_reason})"

        for signal in signals:
            if signal["confidence"] >= alert_threshold:
                # Calculate the NEXT period ID for betting and ensure a safe buffer
                betting_period = get_next_betting_period(df)
                betting_period = ensure_min_time_buffer(df, betting_period, min_buffer_seconds=35)
                
                # Create unique alert key to prevent duplicates
                alert_key = f"{betting_period}_{signal['color']}_{signal['method']}"
                
                # Skip if we already sent this alert
                if alert_key in sent_alerts:
                    print(f"Skipping duplicate alert for {alert_key}")
                    continue
                
                # Create alert message for NEXT period betting
                # Build alert with Big/Small info appended
                msg = format_betting_alert(signal, betting_period, accuracy) + size_line
                
                # Mark this alert as sent
                sent_alerts.add(alert_key)
                
                # Send Telegram alert
                ok = send_telegram(cfg, msg)
                print(f"Alert sent for {signal['color']}: {ok}")
                
                # Log to database if enabled
                if args.log_to_db:
        try:
            anchor_pid = str(df["period_id"].iloc[-1])
            log_alert_to_neon(
                cfg.neon_conn_str,
                anchor_pid,
                            signal["color"],
                            None,  # No number prediction in new system
                            signal["probs"],
                            [signal["method"]],
                            signal["confidence"],
                            accuracy,
                            None,  # No cycle length
                            0.0,   # No cycle accuracy
                        )
                    except Exception as e:
                        print(f"Failed to log to database: {e}")
    
        # Summary
        print("\n" + "="*50)
        print("📊 ANALYSIS SUMMARY")
        print("="*50)
        print(f"📈 Data analyzed: {len(df)} rounds")
        print(f"🎯 Signals detected: {len(signals)}")
        print(f"📱 Alerts sent: {len(sent_alerts)}")
        print(f"⚙️  Preset used: {args.preset}")
        print(f"🎲 Next betting period: {get_next_betting_period(df)}")
        print(f"⏰ Analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*50)
        print("✅ Analysis complete!")
        
        if args.enable_alert and signals:
            print("\n💡 BETTING INSTRUCTIONS:")
            print("1. Wait for the NEXT round to start")
            print("2. Place your bet on the indicated color")
            print("3. Bet within 30 seconds of round start for best timing")
            print("4. Monitor results and adjust strategy as needed")
    
    # Final summary for all cases
    print(f"\n📊 Final Summary:")
    print(f"   - Data analyzed: {len(df)} rounds")
    print(f"   - Signals detected: {len(signals)}")
    print(f"   - System accuracy: {accuracy:.1%}")
    print(f"   - Preset used: {args.preset}")
    print(f"   - Max signals: {args.max_signals}")

if __name__ == "__main__":
    main()
