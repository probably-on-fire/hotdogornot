#!/bin/bash
# v20: same recipe as v18 but seed=1, for ensemble.
set -e
sudo -u rfcai bash -c "cd /opt/rfcai/training && nohup .venv/bin/python -m scripts.auto_retrain --data-dir data/labeled/embedder --model-dir /home/rfcai/training/models/connector_classifier --force --balance-to-smallest --epochs 20 --seed 1 > /tmp/rfcai_retrain.log 2>&1 < /dev/null & disown; echo PID=\$!"
