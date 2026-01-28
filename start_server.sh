#!/bin/bash
# BlitzInfer Server Startup Script
# Logs to /tmp/blitzinfer_server.log with timestamps

set -e

cd "$(dirname "$0")"

# Activate venv
source ./venv/bin/activate

# Set environment for vLLM single-process mode
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export CUDA_VISIBLE_DEVICES=0

# Log rotation - keep last 5 logs
for i in 4 3 2 1; do
    if [ -f "/tmp/blitzinfer_server.log.$i" ]; then
        mv "/tmp/blitzinfer_server.log.$i" "/tmp/blitzinfer_server.log.$((i+1))"
    fi
done
if [ -f "/tmp/blitzinfer_server.log" ]; then
    mv "/tmp/blitzinfer_server.log" "/tmp/blitzinfer_server.log.1"
fi

echo "=========================================="
echo "BlitzInfer Server Starting"
echo "Time: $(date)"
echo "Host: $(hostname)"
echo "Log: /tmp/blitzinfer_server.log"
echo "=========================================="

# Run server with unbuffered output
exec python -u -m blitzinfer.api.server 2>&1 | tee -a /tmp/blitzinfer_server.log
