# WinGo 1-Min Pattern Lab (Ethical)
Tools to **collect**, **store**, and **analyze** WinGo 1-min results to detect PRNG cycles and possible manipulation.
> ⚠️ For educational use. No hacking, no bypassing security. We only analyze publicly visible history.

## Quick Start
```bash
# 1) Create & activate venv (optional)
python -m venv .venv && . .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) Install deps
pip install -r requirements.txt

# 3) Run simulator to see how detection works
python simulator.py

# 4) (Optional) Configure scraper in config.py and run:
python scraper.py

# 5) Analyze latest data
python analyze.py --source db         # if you've saved to Mongo
python analyze.py --source csv --path data/history.csv  # if using CSV
```

## Modules
- `config.py` – all settings (Mongo URL, selectors, base URL).
- `scraper.py` – pulls latest history into CSV and/or MongoDB.
- `db.py` – thin wrapper for Mongo operations.
- `analyze.py` – cycle detection, chi-square tests, manipulation flags, next-signal.
- `strategy.py` – translates analysis → "Bet/Skip + side" suggestions.
- `simulator.py` – generates synthetic rounds (90% PRNG, 10% overrides) so you can test logic today.

## Data Schema
```
period_id (str), number (int 0-9), color (str in {RED,GREEN,VIOLET}), ts (iso8601)
```

## Ethic Note
- We **do not** brute-force seeds or alter platform behavior.
- We only analyze outcomes to understand patterns and **when to avoid playing**.
