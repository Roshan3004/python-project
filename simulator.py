# Generates synthetic WinGo-like rounds so you can test analysis immediately.
import csv, os, random
from datetime import datetime, timedelta
from config import color_from_number, ScraperConfig

def prng_number(seed_state: int) -> int:
    # Simple LCG for demo (NOT secure): X_{n+1} = (aX + c) mod m
    a, c, m = 1103515245, 12345, 2**31
    x = (a*seed_state + c) % m
    # map to 0..9
    return (x % 10), x

def generate_rounds(n=800, override_rate=0.1, big_bet_bias_color="RED"):
    seed = 123456789
    out = []
    start = datetime.utcnow() - timedelta(minutes=n)
    for i in range(n):
        num, seed = prng_number(seed)
        # 10% of the time, override to hurt 'big crowd' (pretend big crowd on RED today)
        if random.random() < override_rate:
            # force to GREEN or VIOLET depending on parity needed
            if big_bet_bias_color == "RED":
                # choose even or 0/5
                num = random.choice([0,2,4,5,6,8])
        color = color_from_number(num)
        period_id = (start + timedelta(minutes=i)).strftime("%Y%m%d%H%M")
        out.append({
            "period_id": period_id,
            "number": num,
            "color": color,
            "ts": (start + timedelta(minutes=i)).isoformat()
        })
    return out

def save_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["period_id","number","color","ts"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

if __name__ == "__main__":
    cfg = ScraperConfig()
    rows = generate_rounds(n=1200, override_rate=0.12, big_bet_bias_color="RED")
    save_csv(cfg.csv_path, rows)
    print(f"Wrote {len(rows)} synthetic rows to {cfg.csv_path}. Now run:")
    print("python analyze.py --source csv --path data/history.csv")
