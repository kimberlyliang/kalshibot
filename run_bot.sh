#!/bin/bash
# Wrapper script for cron — ensures correct Python env and working directory.
# Cron runs with a bare environment, so we source conda and the .env explicitly.

BOT_DIR="/Users/kimberly/Documents/kalshi-btc-bot"
PYTHON="/Users/kimberly/miniconda3/bin/python3"
LOG="$BOT_DIR/logs/cron.log"

# Rotate log if > 5MB
if [ -f "$LOG" ] && [ $(stat -f%z "$LOG" 2>/dev/null || stat -c%s "$LOG") -gt 5242880 ]; then
    mv "$LOG" "$LOG.$(date +%Y%m%d_%H%M%S).bak"
fi

echo "======================================" >> "$LOG"
echo "Cron triggered: $(date -u '+%Y-%m-%d %H:%M:%S UTC')" >> "$LOG"

cd "$BOT_DIR" && "$PYTHON" src/bot.py >> "$LOG" 2>&1

echo "Exit code: $?" >> "$LOG"
