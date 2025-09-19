#!/usr/bin/env python3
import os, sys, time

from config import ScraperConfig
from analyze import resolve_unresolved_alerts

def main():
    try:
        timeout_s = int(os.getenv("RESOLVE_TIMEOUT_S", "40"))
        interval_s = int(os.getenv("RESOLVE_INTERVAL_S", "5"))
    except Exception:
        timeout_s, interval_s = 40, 5

    cfg = ScraperConfig()
    if not cfg.neon_conn_str:
        print("ERROR: NEON_CONN_STR missing")
        sys.exit(1)

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


