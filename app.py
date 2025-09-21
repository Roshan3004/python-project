from flask import Flask, request, jsonify
import subprocess
import threading
import time
import os

app = Flask(__name__)

# Global flag to control the scraper loop
scraper_running = False
scraper_thread = None

def run_scraper():
    """Run the scraper with timing validation"""
    try:
        from scraper import fetch_history_once, save_to_neon, get_db_last_seen
        from config import ScraperConfig
        import datetime
        
        # Check if we're in the right time window (5-10 seconds after minute)
        now = datetime.datetime.now()
        seconds_into_minute = now.second
        
        # Only scrape if we're between 5-10 seconds into the minute
        if seconds_into_minute < 5 or seconds_into_minute > 10:
            return {
                "status": "skipped", 
                "reason": f"Not in scrape window. Current second: {seconds_into_minute}, need 5-10 seconds",
                "current_time": now.strftime("%H:%M:%S")
            }
        
        cfg = ScraperConfig()
        last_seen = get_db_last_seen(cfg.neon_conn_str)
        recs = fetch_history_once(cfg)
        
        if recs:
            new_recs = [r for r in recs if (not last_seen) or r["period_id"] > last_seen]
            saved = save_to_neon(new_recs, cfg.neon_conn_str) if new_recs else 0
            return {
                "status": "success", 
                "fetched": len(recs), 
                "new": len(new_recs), 
                "saved": saved,
                "last_seen": new_recs[0]["period_id"] if new_recs else last_seen,
                "scraped_at": now.strftime("%H:%M:%S"),
                "seconds_into_minute": seconds_into_minute
            }
        else:
            return {
                "status": "success", 
                "fetched": 0, 
                "new": 0, 
                "saved": 0, 
                "last_seen": last_seen,
                "scraped_at": now.strftime("%H:%M:%S"),
                "seconds_into_minute": seconds_into_minute
            }
    except Exception as e:
        return {"status": "error", "message": str(e)[:100]}

@app.route('/')
def home():
    return jsonify({"status": "running", "message": "WinGo Scraper API"})

@app.route('/scrape', methods=['POST', 'GET'])
def scrape():
    """Endpoint to trigger scraping"""
    result = run_scraper()
    return jsonify(result)

@app.route('/trigger', methods=['POST', 'GET'])
def trigger():
    """Lightweight endpoint for cron job - minimal response"""
    try:
        # Run scraper in background
        threading.Thread(target=run_scraper, daemon=True).start()
        return jsonify({"status": "triggered", "timestamp": time.time()})
    except Exception as e:
        return jsonify({"status": "error", "message": "trigger failed"})

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "timestamp": time.time()})

def continuous_scraper_loop():
    """Run the scraper continuously like Railway"""
    global scraper_running
    
    from scraper import api_poll_loop
    from config import ScraperConfig
    
    cfg = ScraperConfig()
    scraper_running = True
    
    try:
        # Run the original continuous loop
        api_poll_loop(cfg)
    except Exception as e:
        print(f"Scraper loop error: {e}")
    finally:
        scraper_running = False

def start_continuous_scraper():
    """Start the continuous scraper in background"""
    global scraper_thread, scraper_running
    
    if not scraper_running:
        scraper_thread = threading.Thread(target=continuous_scraper_loop, daemon=True)
        scraper_thread.start()
        return True
    return False

if __name__ == '__main__':
    # Start continuous scraper when app starts
    start_continuous_scraper()
    app.run(host='0.0.0.0', port=5000)
