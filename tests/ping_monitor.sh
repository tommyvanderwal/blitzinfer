#!/bin/bash
# Ping monitor with precise timestamps
# Usage: ./ping_monitor.sh <host> [log_file]

HOST="${1:-192.168.2.90}"
LOG="${2:-/tmp/ping_monitor.log}"

echo "=== PING MONITOR STARTED ===" | tee "$LOG"
echo "Target: $HOST" | tee -a "$LOG"
echo "Log: $LOG" | tee -a "$LOG"
echo "Started: $(date -Iseconds)" | tee -a "$LOG"
echo "" | tee -a "$LOG"

while true; do
    # Get precise timestamp
    TS=$(date +"%H:%M:%S.%3N")

    # Ping with 1 second timeout
    RESULT=$(ping -c 1 -W 1 "$HOST" 2>&1)

    if echo "$RESULT" | grep -q "time="; then
        # Extract ping time
        TIME=$(echo "$RESULT" | grep "time=" | sed 's/.*time=\([0-9.]*\).*/\1/')

        # Flag if ping is high (>10ms when it should be ~1ms)
        if (( $(echo "$TIME > 10" | bc -l) )); then
            echo "[$TS] ${TIME}ms  *** HIGH ***" | tee -a "$LOG"
        elif (( $(echo "$TIME > 5" | bc -l) )); then
            echo "[$TS] ${TIME}ms  * elevated *" | tee -a "$LOG"
        else
            echo "[$TS] ${TIME}ms" | tee -a "$LOG"
        fi
    else
        # Timeout or error
        echo "[$TS] TIMEOUT/ERROR" | tee -a "$LOG"
    fi

    # Ping every 500ms for better resolution
    sleep 0.5
done
