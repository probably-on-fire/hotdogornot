#!/usr/bin/env bash
# setup_relay.sh — provisions an Ubuntu host (e.g. aired.com) as the relay.
#
# What it does:
#   - creates a system user `rfcai`
#   - clones this repo into /opt/rfcai/training
#   - creates a Python venv and installs only the relay deps (no torch/cv2)
#   - creates /srv/rfcai/{incoming,models/connector_classifier} owned by rfcai
#   - installs the rfcai-relay systemd unit
#   - prints next-step instructions for filling in /etc/default/rfcai-relay
#
# Run with sudo:
#   sudo ./deploy/setup_relay.sh https://github.com/<you>/<repo>.git
#
# Idempotent — safe to re-run.

set -euo pipefail

REPO_URL="${1:-}"
if [ -z "$REPO_URL" ]; then
    echo "usage: sudo $0 <git-repo-url>" >&2
    exit 1
fi

INSTALL_DIR=/opt/rfcai/training
RUN_USER=rfcai
RUN_HOME=/home/$RUN_USER
SRV_DIR=/srv/rfcai

if [ "$EUID" -ne 0 ]; then
    echo "must run as root (use sudo)" >&2
    exit 1
fi

echo "[setup] installing apt packages"
apt-get update -y
apt-get install -y --no-install-recommends \
    git python3 python3-venv python3-pip rsync openssh-client ca-certificates

echo "[setup] creating system user $RUN_USER"
if ! id "$RUN_USER" >/dev/null 2>&1; then
    useradd --system --create-home --shell /bin/bash "$RUN_USER"
fi

echo "[setup] preparing $SRV_DIR"
mkdir -p "$SRV_DIR/incoming" "$SRV_DIR/models/connector_classifier"
chown -R "$RUN_USER:$RUN_USER" "$SRV_DIR"

echo "[setup] cloning / updating repo at $INSTALL_DIR"
if [ -d "$INSTALL_DIR/.git" ]; then
    sudo -u "$RUN_USER" git -C "$INSTALL_DIR" pull --ff-only
else
    mkdir -p "$(dirname "$INSTALL_DIR")"
    chown -R "$RUN_USER:$RUN_USER" "$(dirname "$INSTALL_DIR")"
    # The repo's training/ dir IS what we want as INSTALL_DIR — clone the
    # parent and bind training/ in place. Simpler: clone to a parent path
    # and symlink training/ → INSTALL_DIR. We'll just clone the parent.
    PARENT_PATH=$(dirname "$INSTALL_DIR")
    sudo -u "$RUN_USER" git clone "$REPO_URL" "$PARENT_PATH/repo"
    if [ ! -e "$INSTALL_DIR" ]; then
        ln -s "$PARENT_PATH/repo/training" "$INSTALL_DIR"
    fi
fi

echo "[setup] creating venv at $INSTALL_DIR/.venv"
sudo -u "$RUN_USER" python3 -m venv "$INSTALL_DIR/.venv"
sudo -u "$RUN_USER" "$INSTALL_DIR/.venv/bin/pip" install --quiet --upgrade pip
# Relay-only deps. We don't install torch / opencv on the relay because it
# only forwards uploads + serves files — no inference, no training.
sudo -u "$RUN_USER" "$INSTALL_DIR/.venv/bin/pip" install --quiet \
    fastapi 'uvicorn[standard]' python-multipart

echo "[setup] installing systemd unit"
install -m 0644 "$INSTALL_DIR/deploy/systemd/rfcai-relay.service" /etc/systemd/system/rfcai-relay.service
systemctl daemon-reload

if [ ! -f /etc/default/rfcai-relay ]; then
    echo "[setup] writing /etc/default/rfcai-relay (with placeholder token — REPLACE IT)"
    install -m 0640 -o root -g root "$INSTALL_DIR/deploy/systemd/rfcai-relay.env.example" /etc/default/rfcai-relay
    sed -i "s|replace-me-with-openssl-rand-hex-32|$(openssl rand -hex 32)|" /etc/default/rfcai-relay
fi

cat <<EOF

[setup] DONE.

Next steps:
  1. Verify /etc/default/rfcai-relay (especially RFCAI_DEVICE_TOKEN —
     share this with the AR app team).
  2. Enable + start the service:
        sudo systemctl enable --now rfcai-relay
  3. Confirm:
        curl http://127.0.0.1:8000/healthz
  4. Put nginx (or caddy) in front for TLS on 443 — proxy to 127.0.0.1:8000.
  5. On the training machine, generate an SSH key for user '$RUN_USER@aired.com'
     and add its public key to /home/$RUN_USER/.ssh/authorized_keys.
        sudo -u $RUN_USER mkdir -p $RUN_HOME/.ssh
        sudo -u $RUN_USER chmod 700 $RUN_HOME/.ssh
        # paste training machine's public key into:
        $RUN_HOME/.ssh/authorized_keys
  6. Restrict that key to only sync paths in authorized_keys (recommended):
        command="rsync --server ...",no-port-forwarding,no-pty <key>
EOF
