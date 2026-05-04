#!/bin/bash
# One-shot helper: launch a retrain in the background as rfcai.
# Run from the box only. Logs to /tmp/rfcai_retrain.log.
set -e
cd /opt/rfcai/training
rm -f /tmp/rfcai_retrain.log
SEED=${RFCAI_SEED:-0}
sudo -u rfcai bash -c "cd /opt/rfcai/training && nohup .venv/bin/python -m scripts.auto_retrain --data-dir data/labeled/embedder --model-dir /home/rfcai/training/models/connector_classifier --force --balance-to-smallest --epochs 12 --seed $SEED > /tmp/rfcai_retrain.log 2>&1 < /dev/null & disown; echo PID=\$! seed=$SEED"
