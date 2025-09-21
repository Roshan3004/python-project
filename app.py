from flask import Flask, request, jsonify
import subprocess
import threading
import time

app = Flask(__name__)

def run_scraper():
    """Run the scraper in a separate thread"""
    try:
        # Import and run your scraper logic
        from scraper import fetch_history_once, save_to_neon, get_db_last_seen
        from config import ScraperConfig
        
        cfg = ScraperConfig()
        last_seen = get_db_last_seen(cfg.neon_conn_str)
        recs = fetch_history_once(cfg)
        
        if recs:
            new_recs = [r for r in recs if (not last_seen) or r["period_id"] > last_seen]
            saved = save_to_neon(new_recs, cfg.neon_conn_str) if new_recs else 0
            return {"status": "success", "fetched": len(recs), "new": len(new_recs), "saved": saved}
        else:
            return {"status": "success", "fetched": 0, "new": 0, "saved": 0}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.route('/')
def home():
    return jsonify({"status": "running", "message": "WinGo Scraper API"})

@app.route('/scrape', methods=['POST', 'GET'])
def scrape():
    """Endpoint to trigger scraping"""
    result = run_scraper()
    return jsonify(result)

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "timestamp": time.time()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
