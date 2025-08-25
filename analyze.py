import argparse, csv, math, statistics, os, json
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timezone
import pandas as pd
import numpy as np
from scipy.stats import chisquare
import requests
from db import MongoStore
from strategy import suggest_next, suggest_from_ensemble
from config import ScraperConfig

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["csv","db"], default="csv")
    ap.add_argument("--path", default="data/history.csv", help="CSV path when --source=csv")
    ap.add_argument("--limit", type=int, default=20000, help="Rows to load from Neon when --source=db")
    ap.add_argument("--enable_alert", action="store_true", help="Send Telegram alert when high-confidence signal detected")
    ap.add_argument("--color_prob_threshold", type=float, default=0.62)
    ap.add_argument("--min_sources", type=int, default=2)
    ap.add_argument("--log_to_db", action="store_true", help="Persist BET alerts to Neon in prediction_alerts table")
    args = ap.parse_args()

    if args.source == "csv":
        df = load_csv(args.path)
    else:
        # Load from Neon directly
        cfg = ScraperConfig()
        df = load_neon(cfg.neon_conn_str, limit=args.limit)

    if df.empty:
        print("No data yet. Run simulator.py or scraper.py first.")
        return

    df = df.sort_values("period_id")
    nums = df["number"].astype(int).tolist()
    cols = [c.upper() for c in df["color"].tolist()]

    chi, p = chi_square_color(df.tail(400))
    cycle_len, template = find_cycle(nums)
    pred_next_cycle = next_from_cycle(nums, cycle_len, template) if cycle_len else None
    cycle_acc = validate_cycle(nums, cycle_len, template) if cycle_len else 0.0
    flags = manipulation_indicators(nums, cols)
    active, reason = summarize_flags(flags)

    # Build Markov models
    Mnum, last_num = build_markov_number(nums)
    num_vote = None
    num_prob = 0.0
    if last_num is not None:
        row = Mnum[last_num]
        num_vote = int(np.argmax(row))
        num_prob = float(row[num_vote])

    Mc, last_col = build_markov_color(cols)
    color_probs = markov_color_probs(Mc, last_col)

    # Simple ensemble decision
    sources = []
    if last_col is not None:
        sources.append("MarkovColor")
    if last_num is not None:
        sources.append("MarkovNumber")
    if cycle_len and cycle_acc >= 0.60:
        sources.append(f"Cycle{cycle_len}")
        # If cycle also gives a number, prefer it when it agrees with Markov
        if pred_next_cycle is not None:
            num_vote = pred_next_cycle if num_vote is None else num_vote

    sig = suggest_from_ensemble(color_probs, num_vote, sources, args.color_prob_threshold, args.min_sources)

    print("=== Analysis Report ===")
    print(f"Samples analyzed: {len(df)} (last 400 used for some tests)")
    print(f"Chi-square p-value (color balance): {p:.4f} -> {'OK (PRNG-likely)' if p>0.05 else 'Skewed'}")
    if cycle_len:
        print(f"Detected cycle length: {cycle_len}, val_acc={cycle_acc:.2f}, template (first 12): {template[:12]}")
        if pred_next_cycle is not None:
            print(f"Cycle predicted number: {pred_next_cycle}, color: {'VIOLET' if pred_next_cycle in (0,5) else ('GREEN' if pred_next_cycle%2==0 else 'RED')}")
    else:
        print("No reliable cycle detected.")
    print(f"Manipulation flags: {reason}")
    print(f"Markov number vote: {num_vote} @ {num_prob:.2f}")
    print(f"Color probs: {color_probs}")
    print(f"Signal: {sig.mode} -> {sig.suggestion} (confidence {sig.confidence:.2f}); sources={sig.sources}")

    # Backtest precision (approx.) and optional alert
    precision = backtest_markov_color(cols)
    print(f"Approx. Markov color precision over last 300: {precision:.2f}")

    if args.enable_alert and sig.suggestion.startswith("BET_") and sig.confidence >= args.color_prob_threshold and precision >= 0.60:
        cfg = ScraperConfig()
        top_color = sig.suggestion.replace("BET_", "")
        number_line = f", number={num_vote}" if num_vote is not None else ""
        # Determine anchor period (latest in DB) and a hint for the next target period
        try:
            anchor_pid = str(df["period_id"].iloc[-1])
        except Exception:
            anchor_pid = str(len(df))
        try:
            target_hint = f", next≈{int(anchor_pid)+1}"
        except Exception:
            target_hint = ", next round"
        msg = (
            f"WinGo signal: color={top_color}{number_line}\n"
            f"anchor={anchor_pid}{target_hint}\n"
            f"probs={{{'RED':round(color_probs.get('RED',0),2), 'GREEN':round(color_probs.get('GREEN',0),2), 'VIOLET':round(color_probs.get('VIOLET',0),2)}}}\n"
            f"confidence={sig.confidence:.2f} | last300_precision={precision:.2f}\n"
            f"cycle={'len='+str(cycle_len)+' acc='+str(round(cycle_acc,2)) if cycle_len else 'none'}\n"
            f"rows={len(df)} | at={datetime.now(timezone.utc).isoformat()}"
        )
        ok = send_telegram(cfg, msg)
        print(f"Alert sent: {ok}")
        # Optional: persist this alert for later evaluation
        if args.log_to_db:
            log_alert_to_neon(
                cfg.neon_conn_str,
                anchor_pid,
                top_color,
                int(num_vote) if num_vote is not None else None,
                {k: float(v) for k,v in color_probs.items()},
                sig.sources,
                float(sig.confidence),
                float(precision),
                int(cycle_len) if cycle_len else None,
                float(cycle_acc),
            )
            # Also resolve a batch of previous unresolved alerts
            resolve_unresolved_alerts(cfg.neon_conn_str, batch_limit=200)
    print("=======================")

if __name__ == "__main__":
    main()
