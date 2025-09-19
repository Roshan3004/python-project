#!/usr/bin/env python3
"""
Database diagnostic script to check prediction alerts and game history
"""
import os
import sys
from datetime import datetime
from config import ScraperConfig

def diagnose_database():
    """Diagnose database issues with prediction alerts"""
    print("🔍 Database Diagnostic Tool")
    print("=" * 50)
    
    # Load configuration
    cfg = ScraperConfig()
    
    if not cfg.neon_conn_str:
        print("❌ Error: Database connection string not found!")
        print("Please set NEON_CONN_STR environment variable")
        return False
    
    try:
        import psycopg2
        with psycopg2.connect(cfg.neon_conn_str) as conn:
            with conn.cursor() as cur:
                
                # Check if tables exist
                print("📋 Checking table existence...")
                cur.execute("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name IN ('prediction_alerts', 'game_history', 'alert_reservations')
                    ORDER BY table_name
                """)
                tables = [row[0] for row in cur.fetchall()]
                print(f"✅ Found tables: {', '.join(tables)}")
                
                # Check prediction_alerts structure
                print("\n📊 Checking prediction_alerts structure...")
                cur.execute("""
                    SELECT column_name, data_type, is_nullable
                    FROM information_schema.columns 
                    WHERE table_name = 'prediction_alerts' 
                    ORDER BY ordinal_position
                """)
                columns = cur.fetchall()
                print("Columns in prediction_alerts:")
                for col_name, data_type, nullable in columns:
                    print(f"  - {col_name}: {data_type} ({'NULL' if nullable == 'YES' else 'NOT NULL'})")
                
                # Check game_history structure
                print("\n🎮 Checking game_history structure...")
                cur.execute("""
                    SELECT column_name, data_type, is_nullable
                    FROM information_schema.columns 
                    WHERE table_name = 'game_history' 
                    ORDER BY ordinal_position
                """)
                columns = cur.fetchall()
                print("Columns in game_history:")
                for col_name, data_type, nullable in columns:
                    print(f"  - {col_name}: {data_type} ({'NULL' if nullable == 'YES' else 'NOT NULL'})")
                
                # Check prediction_alerts data
                print("\n📈 Checking prediction_alerts data...")
                cur.execute("SELECT COUNT(*) FROM prediction_alerts")
                total_alerts = cur.fetchone()[0]
                print(f"Total prediction alerts: {total_alerts}")
                
                cur.execute("SELECT COUNT(*) FROM prediction_alerts WHERE resolved_at IS NULL")
                unresolved_alerts = cur.fetchone()[0]
                print(f"Unresolved alerts: {unresolved_alerts}")
                
                cur.execute("SELECT COUNT(*) FROM prediction_alerts WHERE resolved_at IS NOT NULL")
                resolved_alerts = cur.fetchone()[0]
                print(f"Resolved alerts: {resolved_alerts}")
                
                # Show sample data
                print("\n📋 Sample prediction alerts (last 5):")
                cur.execute("""
                    SELECT id, anchor_period_id, predicted_color, predicted_number, 
                           outcome_color, outcome_number, hit_color, resolved_at, created_at
                    FROM prediction_alerts 
                    ORDER BY created_at DESC 
                    LIMIT 5
                """)
                alerts = cur.fetchall()
                for alert in alerts:
                    alert_id, anchor_period, pred_color, pred_number, outcome_color, outcome_number, hit_color, resolved_at, created_at = alert
                    print(f"  ID {alert_id}: {pred_color} for period {anchor_period} -> {outcome_color or 'NULL'} (resolved: {resolved_at or 'NO'})")
                
                # Check game_history data
                print("\n🎮 Checking game_history data...")
                cur.execute("SELECT COUNT(*) FROM game_history")
                total_games = cur.fetchone()[0]
                print(f"Total game history records: {total_games}")
                
                if total_games > 0:
                    cur.execute("""
                        SELECT period_id, number, color, scraped_at 
                        FROM game_history 
                        ORDER BY scraped_at DESC 
                        LIMIT 5
                    """)
                    games = cur.fetchall()
                    print("Sample game history (last 5):")
                    for game in games:
                        period_id, number, color, scraped_at = game
                        print(f"  Period {period_id}: {color} {number} at {scraped_at}")
                
                # Check for period ID format issues
                print("\n🔍 Checking period ID formats...")
                cur.execute("""
                    SELECT DISTINCT 
                        CASE 
                            WHEN period_id ~ '^[0-9]+$' THEN 'numeric'
                            ELSE 'non-numeric'
                        END as format_type,
                        COUNT(*) as count
                    FROM game_history 
                    GROUP BY format_type
                """)
                formats = cur.fetchall()
                for format_type, count in formats:
                    print(f"  {format_type} period IDs: {count}")
                
                # Check if there are any alerts that should be resolvable
                print("\n🔍 Checking for resolvable alerts...")
                cur.execute("""
                    SELECT pa.id, pa.anchor_period_id, pa.predicted_color, pa.created_at
                    FROM prediction_alerts pa
                    WHERE pa.resolved_at IS NULL
                    AND EXISTS (
                        SELECT 1 FROM game_history gh 
                        WHERE gh.period_id = pa.anchor_period_id
                    )
                    ORDER BY pa.created_at DESC
                    LIMIT 3
                """)
                resolvable = cur.fetchall()
                if resolvable:
                    print("Potentially resolvable alerts:")
                    for alert_id, anchor_period, pred_color, created_at in resolvable:
                        print(f"  Alert {alert_id}: {pred_color} for period {anchor_period} (created: {created_at})")
                else:
                    print("No obviously resolvable alerts found")
                
        print("\n✅ Database diagnostic completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Database diagnostic failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = diagnose_database()
    sys.exit(0 if success else 1)
