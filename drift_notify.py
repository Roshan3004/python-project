import psycopg2
import argparse
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List
from config import ScraperConfig
import requests


def send_telegram(token: str, chat_id: str, text: str) -> bool:
    if not token or not chat_id:
        return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text}, timeout=10
        )
        return r.ok
    except Exception:
        return False


def fetch_metrics(conn, days: int) -> Dict[str, Any]:
    with conn.cursor() as cur:
        cur.execute(
            """
            WITH base AS (
                SELECT created_at, hit_color, sources::text AS src
                FROM prediction_alerts
                WHERE resolved_at IS NOT NULL
                  AND created_at >= now() - interval '%s days'
            )
            SELECT
              COUNT(*) AS n,
              COALESCE(AVG(CASE WHEN hit_color THEN 1 ELSE 0 END)::float,0.0) AS p
            FROM base;
            """,
            (days,)
        )
        n, p = cur.fetchone()
        # by sources
        cur.execute(
            """
            WITH base AS (
                SELECT created_at, hit_color, COALESCE(sources::text,'[]') AS src
                FROM prediction_alerts
                WHERE resolved_at IS NOT NULL
                  AND created_at >= now() - interval '%s days'
            )
            SELECT src, COUNT(*) AS n, COALESCE(AVG(CASE WHEN hit_color THEN 1 ELSE 0 END)::float,0.0) AS p
            FROM base
            GROUP BY src
            HAVING COUNT(*) >= 30
            ORDER BY n DESC
            LIMIT 10;
            """,
            (days,)
        )
        rows = cur.fetchall() or []
    return {"n": int(n or 0), "p": float(p or 0.0), "by_src": rows}


def ensure_cooldown_table(conn):
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS drift_notifications (
              id SERIAL PRIMARY KEY,
              created_at timestamptz DEFAULT now(),
              message TEXT
            );
            """
        )
        conn.commit()


def cooldown_ok(conn, cooldown_days: int) -> bool:
    ensure_cooldown_table(conn)
    with conn.cursor() as cur:
        cur.execute("SELECT COALESCE(MAX(created_at), 'epoch') FROM drift_notifications;")
        (last_ts,) = cur.fetchone()
        if not last_ts:
            return True
        return (datetime.now(timezone.utc) - last_ts) >= timedelta(days=cooldown_days)


def record_notification(conn, message: str):
    ensure_cooldown_table(conn)
    with conn.cursor() as cur:
        cur.execute("INSERT INTO drift_notifications(message) VALUES (%s);", (message,))
        conn.commit()


def build_recommendation(m7: Dict[str, Any], m30: Dict[str, Any]) -> str:
    n7, p7 = m7["n"], m7["p"]
    n30, p30 = m30["n"], m30["p"]
    lines = [
        "Drift check:",
        f"7d: n={n7} color_precision={p7:.2f}",
        f"30d: n={n30} color_precision={p30:.2f}",
    ]
    recommendations: List[str] = []

    delta = p7 - p30
    # Core recs
    if n7 >= 50 and abs(delta) >= 0.05:
        if p7 < p30:
            # performance drop
            recommendations.append("Recommendation: raise color_prob_threshold (e.g., 0.62 → 0.65)")
            recommendations.append("Recommendation: consider min_sources=2 (or 3 if already 2)")
        else:
            recommendations.append("Recommendation: keep thresholds; recent performance is stronger than 30d baseline")

    # If 7d precision below target
    if n7 >= 50 and p7 < 0.60:
        if "Recommendation: raise color_prob_threshold (e.g., 0.62 → 0.65)" not in recommendations:
            recommendations.append("Recommendation: raise color_prob_threshold (e.g., 0.62 → 0.65)")
        if "Recommendation: consider min_sources=2 (or 3 if already 2)" not in recommendations:
            recommendations.append("Recommendation: consider min_sources=2 (or 3 if already 2)")

    # By source combos (prefer multi-source)
    best_combo = None
    for src, n, p in m7["by_src"]:
        if best_combo is None or p > best_combo[2]:
            best_combo = (src, int(n), float(p))
    if best_combo is not None and best_combo[1] >= 30 and best_combo[2] - p7 >= 0.05:
        recommendations.append(f"Source combo performing best: {best_combo[0]} (n={best_combo[1]}, color_hit={best_combo[2]:.2f}). Suggest min_sources=2 to favor agreement.")

    if not recommendations:
        recommendations.append("No significant drift detected. Keep current thresholds.")

    lines.append("\n".join(f"- {r}" for r in recommendations))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cooldown_days", type=int, default=3)
    ap.add_argument("--send", action="store_true", help="Send Telegram notification if drift detected or summary built")
    args = ap.parse_args()

    cfg = ScraperConfig()
    with psycopg2.connect(cfg.neon_conn_str) as conn:
        m7 = fetch_metrics(conn, 7)
        m30 = fetch_metrics(conn, 30)
        text = build_recommendation(m7, m30)
        print(text)

        if args.send and cooldown_ok(conn, args.cooldown_days):
            ok = send_telegram(cfg.telegram_bot_token, cfg.telegram_chat_id, text)
            print(f"Drift notification sent: {ok}")
            if ok:
                record_notification(conn, text)
        elif args.send:
            print("Cooldown active; no notification sent.")


if __name__ == "__main__":
    main()
