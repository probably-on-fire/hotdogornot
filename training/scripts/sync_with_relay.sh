#!/usr/bin/env bash
# sync_with_relay.sh — bidirectional sync between the training machine and
# the relay at aired.com. Runs from the training machine, called by a
# systemd timer (see deploy/systemd/rfcai-sync.timer).
#
# Pull direction: relay's incoming/ → local incoming/ (new uploads to process)
# Push direction: local models/connector_classifier/ → relay's models/
#                 (new model artifacts to serve)
#
# Required env (set in /etc/default/rfcai-sync, or systemd EnvironmentFile):
#   RFCAI_RELAY_HOST       hostname or tailscale name of the relay
#                          (e.g. aired.com or rfcai-relay)
#   RFCAI_RELAY_USER       SSH user on the relay (e.g. rfcai)
#   RFCAI_RELAY_INCOMING   path on the relay where uploads accumulate
#                          (e.g. /srv/rfcai/incoming)
#   RFCAI_RELAY_MODEL_DIR  path on the relay where the model lives
#                          (e.g. /srv/rfcai/models/connector_classifier)
#   RFCAI_LOCAL_INCOMING   local path the daemon watches
#                          (e.g. /home/rfcai/incoming)
#   RFCAI_LOCAL_MODEL_DIR  local path auto_retrain writes to
#                          (e.g. /home/rfcai/training/models/connector_classifier)
#
# Optional:
#   RFCAI_RELAY_PORT       SSH port (default 22)
#   RFCAI_SSH_KEY          path to SSH private key (default ~/.ssh/id_ed25519)
#
# Exits 0 on full success, 1 if either rsync direction failed.

set -euo pipefail

: "${RFCAI_RELAY_HOST:?missing RFCAI_RELAY_HOST}"
: "${RFCAI_RELAY_USER:?missing RFCAI_RELAY_USER}"
: "${RFCAI_RELAY_INCOMING:?missing RFCAI_RELAY_INCOMING}"
: "${RFCAI_RELAY_MODEL_DIR:?missing RFCAI_RELAY_MODEL_DIR}"
: "${RFCAI_LOCAL_INCOMING:?missing RFCAI_LOCAL_INCOMING}"
: "${RFCAI_LOCAL_MODEL_DIR:?missing RFCAI_LOCAL_MODEL_DIR}"

RFCAI_RELAY_PORT="${RFCAI_RELAY_PORT:-22}"
RFCAI_SSH_KEY="${RFCAI_SSH_KEY:-$HOME/.ssh/id_ed25519}"

SSH_OPTS="-p ${RFCAI_RELAY_PORT} -i ${RFCAI_SSH_KEY} -o StrictHostKeyChecking=accept-new -o BatchMode=yes"
RELAY="${RFCAI_RELAY_USER}@${RFCAI_RELAY_HOST}"

mkdir -p "${RFCAI_LOCAL_INCOMING}" "${RFCAI_LOCAL_MODEL_DIR}"

# --- Pull uploads ---------------------------------------------------------
# rsync only new files. Don't delete on either side — daemon writes
# _processed.json into each upload dir; cleanup is a separate concern
# (see scripts/cleanup_old_uploads.sh on the relay, run weekly).
echo "[sync] pulling uploads from ${RELAY}:${RFCAI_RELAY_INCOMING}/"
rsync -az --partial \
    -e "ssh ${SSH_OPTS}" \
    "${RELAY}:${RFCAI_RELAY_INCOMING}/" \
    "${RFCAI_LOCAL_INCOMING}/"
PULL_RC=$?

# --- Push model artifacts -------------------------------------------------
# Training run on the local box produces new weights + manifest; rsync
# pushes them to the relay's serving dir so /model/* picks up the new
# version on its next read. --delete removes stale versioned snapshots
# on the relay if we've cleaned them up locally.
echo "[sync] pushing model from ${RFCAI_LOCAL_MODEL_DIR} to ${RELAY}:${RFCAI_RELAY_MODEL_DIR}/"
rsync -az --partial --delete \
    -e "ssh ${SSH_OPTS}" \
    "${RFCAI_LOCAL_MODEL_DIR}/" \
    "${RELAY}:${RFCAI_RELAY_MODEL_DIR}/"
PUSH_RC=$?

if [ "$PULL_RC" -ne 0 ] || [ "$PUSH_RC" -ne 0 ]; then
    echo "[sync] FAILED pull_rc=$PULL_RC push_rc=$PUSH_RC" >&2
    exit 1
fi

echo "[sync] OK"
