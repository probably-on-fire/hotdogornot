#!/bin/bash
# Reconstruct labels.json for v8 (6 classes, SMA dropped). Run on box.
set -e
sudo -u rfcai bash -c 'cat > /home/rfcai/training/models/connector_classifier/labels.json' <<'JSON'
{
  "class_names": [
    "3.5mm-M",
    "3.5mm-F",
    "2.92mm-M",
    "2.92mm-F",
    "2.4mm-M",
    "2.4mm-F"
  ],
  "input_size": 224,
  "architecture": "resnet18",
  "n_train_samples": 1566,
  "n_val_samples": 388,
  "class_counts": {
    "3.5mm-M": 261,
    "3.5mm-F": 261,
    "2.92mm-M": 261,
    "2.92mm-F": 261,
    "2.4mm-M": 261,
    "2.4mm-F": 261
  }
}
JSON
systemctl restart rfcai-predict
echo "labels.json restored to v8 (6 classes); service restarted"
