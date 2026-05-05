#!/bin/bash
# One-shot helper: launch v19 retrain at 28 epochs as rfcai.
set -e
sudo -u rfcai bash -c "cd /opt/rfcai/training && nohup .venv/bin/python -m scripts.auto_retrain --data-dir data/labeled/embedder --model-dir /home/rfcai/training/models/connector_classifier --force --balance-to-smallest --epochs 28 --seed 0 > /tmp/rfcai_retrain.log 2>&1 < /dev/null & disown; echo PID=\$!"
