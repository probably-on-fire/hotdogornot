#!/bin/bash
# Kick the prototypical networks experiment.
set -e
sudo -u rfcai cp /tmp/_proto.py /opt/rfcai/repo/training/scripts/exp_proto_train.py
sudo -u rfcai bash -c "cd /opt/rfcai/training && nohup .venv/bin/python -u scripts/exp_proto_train.py --data-dir data/labeled/embedder --holdout-dir data/test_holdout --out-dir /home/rfcai/training/models/connector_classifier_proto --epochs 30 --balance-to-smallest --seed 0 > /tmp/proto_train.log 2>&1 < /dev/null & disown; echo PID=\$!"
