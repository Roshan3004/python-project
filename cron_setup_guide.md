# Cron Job Setup for Precise Timing

## Current Requirement:
- Scrape data 5-10 seconds into each new minute
- Example: Period 1445 at 12:44, scrape at 12:45:05-12:45:10

## Recommended Cron Job Setup:

### Option 1: Multiple Cron Jobs (Recommended)
Set up 6 cron jobs to hit your endpoint every 10 seconds:

1. **Cron Job 1**: Every minute at :05 seconds
   - URL: `https://python-project-wgn3.onrender.com/trigger`
   - Schedule: `5 * * * * *` (every minute at 5 seconds)

2. **Cron Job 2**: Every minute at :06 seconds
   - URL: `https://python-project-wgn3.onrender.com/trigger`
   - Schedule: `6 * * * * *` (every minute at 6 seconds)

3. **Cron Job 3**: Every minute at :07 seconds
   - URL: `https://python-project-wgn3.onrender.com/trigger`
   - Schedule: `7 * * * * *` (every minute at 7 seconds)

4. **Cron Job 4**: Every minute at :08 seconds
   - URL: `https://python-project-wgn3.onrender.com/trigger`
   - Schedule: `8 * * * * *` (every minute at 8 seconds)

5. **Cron Job 5**: Every minute at :09 seconds
   - URL: `https://python-project-wgn3.onrender.com/trigger`
   - Schedule: `9 * * * * *` (every minute at 9 seconds)

6. **Cron Job 6**: Every minute at :10 seconds
   - URL: `https://python-project-wgn3.onrender.com/trigger`
   - Schedule: `10 * * * * *` (every minute at 10 seconds)

### Option 2: Single Cron Job with 10-second intervals
- URL: `https://python-project-wgn3.onrender.com/trigger`
- Schedule: Every 10 seconds
- This will hit at :00, :10, :20, :30, :40, :50 seconds
- Only :10 seconds will be in your 5-10 window

## How It Works:
1. Cron jobs hit your endpoint every second from :05 to :10
2. Your app checks if current time is in 5-10 second window
3. If yes: scrapes data
4. If no: returns "skipped" status
5. Only one scrape per minute (the first one that hits the window)

## Benefits:
✅ Precise timing (5-10 seconds into minute)
✅ Multiple attempts to catch the window
✅ Only scrapes once per minute
✅ Perfect for prediction timing
