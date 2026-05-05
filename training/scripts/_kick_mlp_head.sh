#!/bin/bash
# Kick the MLP-head experiment.
set -e
sudo -u rfcai cp /tmp/_arch.py /opt/rfcai/repo/training/scripts/exp_arch_variants.py
sudo -u rfcai bash -c "cd /opt/rfcai/training && nohup .venv/bin/python -u scripts/exp_arch_variants.py --variant mlp-head --data-dir data/labeled/embedder --holdout-dir data/test_holdout --out-dir /home/rfcai/training/models/connector_classifier_mlp --epochs 20 --balance-to-smallest --seed 0 > /tmp/mlp_head_train.log 2>&1 < /dev/null & disown; echo PID=\$!"
