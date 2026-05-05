#!/bin/bash
# Kick the two-head training on the box.
set -e
sudo -u rfcai bash -c "cd /opt/rfcai/training && nohup .venv/bin/python scripts/exp_two_head_train.py --data-dir data/labeled/embedder --holdout-dir data/test_holdout --out-dir /home/rfcai/training/models/connector_classifier_2h --epochs 20 --balance-to-smallest --seed 0 > /tmp/two_head_train.log 2>&1 < /dev/null & disown; echo PID=\$!"
