#!/bin/bash
# Wrapper script for cron — ensures correct Python env and working directory.
# Cron runs with a bare environment, so we source conda and the .env explicitly.


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/.env"

# Load BOT_DIR / PYTHON / LOG from .env if present.
if [ -f "$ENV_FILE" ]; then
    set -a
    . "$ENV_FILE"
    set +a
fi

# Safe fallbacks if .env is missing values.
BOT_DIR="${BOT_DIR:-$SCRIPT_DIR}"
PYTHON="${PYTHON:-python3}"
LOG="${LOG:-$BOT_DIR/logs/cron.log}"

# Rotate log if > 5MB
if [ -f "$LOG" ] && [ $(stat -f%z "$LOG" 2>/dev/null || stat -c%s "$LOG") -gt 5242880 ]; then
    mv "$LOG" "$LOG.$(date +%Y%m%d_%H%M%S).bak"
fi

echo "======================================" >> "$LOG"
echo "Cron triggered: $(date -u '+%Y-%m-%d %H:%M:%S UTC')" >> "$LOG"

cd "$BOT_DIR" && "$PYTHON" src/bot.py >> "$LOG" 2>&1

echo "Exit code: $?" >> "$LOG"
