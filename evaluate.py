import argparse
from datetime import datetime, timedelta, timezone
import json
import psycopg2
from config import ScraperConfig


def fetch_summary(conn_str: str, days: int = 7) -> dict:
    """Return performance summary over the last N days from prediction_alerts resolved rows."""
    with psycopg2.connect(conn_str) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                WITH base AS (
                    SELECT
                        id,
                        created_at,
                        predicted_color,
                        predicted_number,
                        confidence,
                        last300_precision,
                        sources::text AS sources_text,
                        outcome_color,
                        outcome_number,
                        hit_color,
                        hit_number
                    FROM prediction_alerts
                    WHERE resolved_at IS NOT NULL
                      AND created_at >= now() - interval '%s days'
                )
                SELECT
                    COUNT(*) AS alerts,
                    COALESCE(AVG(CASE WHEN hit_color THEN 1 ELSE 0 END)::float, 0.0) AS color_hit,
                    COALESCE(AVG(CASE WHEN hit_number THEN 1 ELSE 0 END)::float, 0.0) AS number_hit
                FROM base;
                """,
                (days,)
            )
            overall = cur.fetchone()
            overall_summary = {
                "alerts": overall[0] or 0,
                "color_hit": float(overall[1] or 0.0),
                "number_hit": float(overall[2] or 0.0),
            }
            # By source combo (stringified list)
            cur.execute(
                """
                WITH base AS (
                    SELECT
                        CASE WHEN sources IS NULL THEN '[]' ELSE sources::text END AS src,
                        hit_color
                    FROM prediction_alerts
                    WHERE resolved_at IS NOT NULL
                      AND created_at >= now() - interval '%s days'
                )
                SELECT src, COUNT(*) AS n, COALESCE(AVG(CASE WHEN hit_color THEN 1 ELSE 0 END)::float,0.0) AS color_hit
                FROM base
                GROUP BY src
                ORDER BY n DESC
                LIMIT 10;
                """,
                (days,)
            )
            rows = cur.fetchall() or []
            by_sources = [{"sources": r[0], "n": r[1], "color_hit": float(r[2])} for r in rows]
    return {
        "window_days": days,
        "overall": overall_summary,
        "by_sources": by_sources,
    }


def format_summary(summary: dict, title: str) -> str:
    lines = [f"{title}", f"window={summary['window_days']}d"]
    o = summary["overall"]
    lines.append(f"overall: alerts={o['alerts']} color_hit={o['color_hit']:.2f} number_hit={o['number_hit']:.2f}")
    if summary["by_sources"]:
        lines.append("top source combos:")
        for r in summary["by_sources"][:5]:
            lines.append(f"- {r['sources']}: n={r['n']}, color_hit={r['color_hit']:.2f}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window_days", type=int, default=7)
    ap.add_argument("--title", default="WinGo performance summary")
    args = ap.parse_args()

    cfg = ScraperConfig()
    summary = fetch_summary(cfg.neon_conn_str, days=args.window_days)
    text = format_summary(summary, args.title)
    print(text)


if __name__ == "__main__":
    main()
