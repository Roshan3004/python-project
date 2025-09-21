import sqlite3
import logging
from datetime import datetime
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def save_to_sqlite(records: List[Dict[str, Any]], db_path: str = "wingo_data.db") -> int:
    """Save records to SQLite database"""
    if not records:
        return 0
        
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create table if not exists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS game_history (
                period_id TEXT PRIMARY KEY,
                number TEXT,
                color TEXT,
                scraped_at TIMESTAMP
            )
        """)
        
        # Insert records
        inserted = 0
        for record in records:
            cursor.execute("""
                INSERT OR IGNORE INTO game_history (period_id, number, color, scraped_at)
                VALUES (?, ?, ?, ?)
            """, (
                record['period_id'],
                record['number'],
                record['color'],
                record['scraped_at']
            ))
            if cursor.rowcount > 0:
                inserted += 1
        
        conn.commit()
        return inserted
    except Exception as e:
        logger.error(f"SQLite database error: {e}")
        return 0
    finally:
        if conn:
            conn.close()

def get_total_rows_sqlite(db_path: str = "wingo_data.db") -> int:
    """Return total rows in game_history."""
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS game_history (
                period_id TEXT PRIMARY KEY,
                number TEXT,
                color TEXT,
                scraped_at TIMESTAMP
            )
        """)
        cursor.execute("SELECT COUNT(*) FROM game_history")
        count = cursor.fetchone()[0]
        return int(count or 0)
    except Exception as e:
        logger.warning(f"Total rows check failed: {e}")
        return 0
    finally:
        if conn:
            conn.close()

def get_db_last_seen_sqlite(db_path: str = "wingo_data.db") -> str | None:
    """Return the max period_id from DB, or None if table empty/not exists."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS game_history (
                period_id TEXT PRIMARY KEY,
                number TEXT,
                color TEXT,
                scraped_at TIMESTAMP
            )
        """)
        conn.commit()
        cursor.execute("SELECT MAX(period_id) FROM game_history")
        row = cursor.fetchone()
        cursor.close()
        conn.close()
        return row[0] if row and row[0] else None
    except Exception as e:
        logger.warning(f"Could not read last_seen from SQLite DB: {e}")
        return None
