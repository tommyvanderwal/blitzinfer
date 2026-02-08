#!/bin/bash
# Power cycle the remote machine via Eve Energy Matter plug
# Usage: ./power_cycle.sh [delay_seconds]

DELAY=${1:-10}
HOST="192.168.2.90"

echo "=== Power Cycling Remote Machine ==="
echo "Target: $HOST"
echo "Off duration: ${DELAY}s"
echo ""

# Turn OFF
echo "[$(date +%H:%M:%S)] Turning power OFF..."
/snap/bin/chip-tool onoff off 1 1 2>&1 | grep -q "SUCCESS" && echo "  OFF successful" || echo "  OFF may have failed"

# Wait
echo "[$(date +%H:%M:%S)] Waiting ${DELAY} seconds..."
sleep $DELAY

# Turn ON
echo "[$(date +%H:%M:%S)] Turning power ON..."
/snap/bin/chip-tool onoff on 1 1 2>&1 | grep -q "SUCCESS" && echo "  ON successful" || echo "  ON may have failed"

echo ""
echo "[$(date +%H:%M:%S)] Power cycle complete. Waiting for system to boot..."

# Wait for system to come back
for i in {1..60}; do
    if ping -c 1 -W 2 $HOST &>/dev/null; then
        echo "[$(date +%H:%M:%S)] System is responding to ping!"

        # Wait a bit more for SSH
        sleep 10
        if ssh -o ConnectTimeout=5 -o BatchMode=yes tommy@$HOST 'echo ok' &>/dev/null; then
            echo "[$(date +%H:%M:%S)] SSH is available!"
            ssh tommy@$HOST 'uptime'
            exit 0
        fi
    fi
    echo -n "."
    sleep 5
done

echo ""
echo "[$(date +%H:%M:%S)] System did not come back after 5 minutes"
exit 1
