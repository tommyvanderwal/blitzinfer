#!/bin/bash
# CPU monitor - logs per-process CPU usage with timestamps
# Usage: ./cpu_monitor.sh [log_file]

LOG="${1:-/tmp/cpu_monitor.log}"

echo "=== CPU MONITOR STARTED ===" | tee "$LOG"
echo "Started: $(date -Iseconds)" | tee -a "$LOG"
echo "" | tee -a "$LOG"

while true; do
    TS=$(date +"%H:%M:%S.%3N")

    # Get top CPU consumers (top 5 processes)
    # Focus on processes using >5% CPU
    TOP_PROCS=$(ps -eo pid,comm,%cpu --sort=-%cpu | head -6 | tail -5 | awk '$3 > 5 {printf "%s(%s%%) ", $2, $3}')

    # Get overall CPU usage
    LOAD=$(cat /proc/loadavg | cut -d' ' -f1-3)

    # Check for any process using >80% CPU (warning sign)
    HIGH_CPU=$(ps -eo pid,comm,%cpu --sort=-%cpu | awk '$3 > 80 {printf "!!! %s (PID %s) at %s%% !!!", $2, $1, $3}')

    if [ -n "$HIGH_CPU" ]; then
        echo "[$TS] LOAD:$LOAD | $HIGH_CPU" | tee -a "$LOG"

        # Get thread info for high CPU process
        HIGH_PID=$(ps -eo pid,%cpu --sort=-%cpu | awk 'NR==2 {print $1}')
        if [ -n "$HIGH_PID" ]; then
            # Get top threads in that process
            THREADS=$(ps -L -p $HIGH_PID -o tid,%cpu,comm --sort=-%cpu 2>/dev/null | head -4 | tail -3 | awk '{printf "T%s(%s%%) ", $1, $2}')
            echo "[$TS]   Threads: $THREADS" | tee -a "$LOG"
        fi
    elif [ -n "$TOP_PROCS" ]; then
        echo "[$TS] LOAD:$LOAD | $TOP_PROCS" | tee -a "$LOG"
    else
        echo "[$TS] LOAD:$LOAD | idle" | tee -a "$LOG"
    fi

    sleep 0.5
done
