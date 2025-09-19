#!/usr/bin/env python3
import os, sys, time

from config import ScraperConfig
from analyze import resolve_unresolved_alerts
import psycopg2

def main():
    try:
        timeout_s = int(os.getenv("RESOLVE_TIMEOUT_S", "40"))
        interval_s = int(os.getenv("RESOLVE_INTERVAL_S", "5"))
        lookback_min = int(os.getenv("RESOLVE_LOOKBACK_MIN", "10"))
    except Exception:
        timeout_s, interval_s, lookback_min = 40, 5, 10

    cfg = ScraperConfig()
    if not cfg.neon_conn_str:
        print("ERROR: NEON_CONN_STR missing")
        sys.exit(1)

    # Fast exit: only resolve if there are unresolved alerts in recent window
    try:
        with psycopg2.connect(cfg.neon_conn_str) as conn:
            with conn.cursor() as cur:
                # Prefer recent unresolved count (last N minutes)
                cur.execute(
                    """
                    SELECT COUNT(*)
                    FROM prediction_alerts
                    WHERE resolved_at IS NULL
                      AND created_at >= NOW() - make_interval(mins => %s)
                    """,
                    (lookback_min,)
                )
                (pending_recent,) = cur.fetchone()
                if int(pending_recent or 0) == 0:
                    # Double-check: if absolutely no unresolved rows at all, exit too
                    cur.execute("SELECT COUNT(*) FROM prediction_alerts WHERE resolved_at IS NULL")
                    (pending_any,) = cur.fetchone()
                    if int(pending_any or 0) == 0:
                        print("resolve_wait: no unresolved alerts — exiting")
                        return
                    else:
                        print(f"resolve_wait: unresolved exist but none in last {lookback_min} min — exiting")
                        return
    except Exception as e:
        print(f"resolve_wait: recent check failed: {e}")

    deadline = time.time() + timeout_s
    attempt = 0
    while True:
        attempt += 1
        print(f"resolve_wait: attempt {attempt}")
        try:
            resolve_unresolved_alerts(cfg.neon_conn_str, batch_limit=50)
        except Exception as e:
            print(f"resolve_wait: resolve call failed: {e}")

        if time.time() >= deadline:
            break
        time.sleep(max(1, interval_s))

    print("resolve_wait: done")

if __name__ == "__main__":
    main()


