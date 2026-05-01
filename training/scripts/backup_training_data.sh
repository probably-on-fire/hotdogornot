#!/usr/bin/env bash
# Snapshot the cleaned training data + test_holdout from the training box.
# Drops two timestamped tarballs into ./backups/ locally and leaves a
# matching pair on the box at /var/backups/rfcai/ for off-laptop recovery.
#
# Usage:
#   bash scripts/backup_training_data.sh
#
# Env overrides:
#   RFCAI_BOX       — ssh target (default: chris@192.168.20.235)
#   LOCAL_BACKUPS   — local destination dir (default: ./backups)
#
# What's in each tarball:
#   embedder_originals_<ts>.tar.gz
#       data/labeled/embedder/<CLASS>/*.jpg, EXCLUDING regenerable
#       synthetic variants (_z70, _z50, _mask). These are the user-
#       curated originals — the hours-of-cleanup output worth preserving.
#   test_holdout_<ts>.tar.gz
#       data/test_holdout/<CLASS>/*.jpg — the golden held-out test set.
#
# Restore:
#   ssh chris@192.168.20.235
#   sudo -u rfcai bash -c \
#     'cd /opt/rfcai/repo/training/data/labeled \
#      && tar xzf /var/backups/rfcai/embedder_originals_<ts>.tar.gz'

set -euo pipefail

BOX="${RFCAI_BOX:-chris@192.168.20.235}"
LOCAL_BACKUPS="${LOCAL_BACKUPS:-$(dirname "$0")/../backups}"
DATESTAMP=$(date -u +%Y%m%dT%H%M%S)

mkdir -p "$LOCAL_BACKUPS"

echo "[1/3] Building tarballs on $BOX (sudo password prompt expected)..."
ssh -t "$BOX" "sudo bash -c '
  mkdir -p /var/backups/rfcai
  cd /opt/rfcai/repo/training/data/labeled
  sudo -u rfcai tar \
    --exclude=\"*_z70.*\" --exclude=\"*_z50.*\" --exclude=\"*_mask.*\" \
    -czf /var/backups/rfcai/embedder_originals_$DATESTAMP.tar.gz embedder
  cd /opt/rfcai/repo/training/data
  sudo -u rfcai tar -czf /var/backups/rfcai/test_holdout_$DATESTAMP.tar.gz test_holdout
  chmod a+r /var/backups/rfcai/*.tar.gz
'"

echo "[2/3] Pulling tarballs to $LOCAL_BACKUPS..."
scp "$BOX:/var/backups/rfcai/embedder_originals_$DATESTAMP.tar.gz" "$LOCAL_BACKUPS/"
scp "$BOX:/var/backups/rfcai/test_holdout_$DATESTAMP.tar.gz" "$LOCAL_BACKUPS/"

echo "[3/3] Done. Snapshot:"
ls -lh "$LOCAL_BACKUPS"/*_${DATESTAMP}.tar.gz
