#!/usr/bin/env bash
# setup_training.sh — provisions an Ubuntu host (e.g. ai.localradionetworks.com)
# as the training machine.
#
# What it does:
#   - creates a system user `rfcai`
#   - clones this repo into /opt/rfcai/training
#   - creates a Python venv and installs the full training deps
#     (torch + opencv + everything the ingestion daemon and trainer need)
#   - creates the local data directories owned by rfcai
#   - installs the ingestion-daemon, auto-retrain timer, and sync timer
#   - prints next-step instructions for the SSH key + env files
#
# Run with sudo:
#   sudo ./deploy/setup_training.sh https://github.com/<you>/<repo>.git
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

if [ "$EUID" -ne 0 ]; then
    echo "must run as root (use sudo)" >&2
    exit 1
fi

echo "[setup] installing apt packages"
apt-get update -y
apt-get install -y --no-install-recommends \
    git python3 python3-venv python3-pip rsync openssh-client \
    libgl1 libglib2.0-0 ca-certificates

echo "[setup] creating system user $RUN_USER"
if ! id "$RUN_USER" >/dev/null 2>&1; then
    useradd --system --create-home --shell /bin/bash "$RUN_USER"
fi

echo "[setup] preparing local data dirs under $RUN_HOME"
sudo -u "$RUN_USER" mkdir -p \
    "$RUN_HOME/incoming" \
    "$RUN_HOME/training/data/labeled/embedder" \
    "$RUN_HOME/training/data/quarantine" \
    "$RUN_HOME/training/models/connector_classifier"

echo "[setup] cloning / updating repo at $INSTALL_DIR"
if [ -d "$INSTALL_DIR/.git" ] || [ -L "$INSTALL_DIR" ]; then
    if [ -L "$INSTALL_DIR" ]; then
        TARGET=$(readlink -f "$INSTALL_DIR")
        sudo -u "$RUN_USER" git -C "$(dirname "$TARGET")" pull --ff-only
    else
        sudo -u "$RUN_USER" git -C "$INSTALL_DIR" pull --ff-only
    fi
else
    PARENT_PATH=$(dirname "$INSTALL_DIR")
    mkdir -p "$PARENT_PATH"
    chown -R "$RUN_USER:$RUN_USER" "$PARENT_PATH"
    sudo -u "$RUN_USER" git clone "$REPO_URL" "$PARENT_PATH/repo"
    if [ ! -e "$INSTALL_DIR" ]; then
        ln -s "$PARENT_PATH/repo/training" "$INSTALL_DIR"
    fi
fi

echo "[setup] creating venv + installing training deps"
sudo -u "$RUN_USER" python3 -m venv "$INSTALL_DIR/.venv"
sudo -u "$RUN_USER" "$INSTALL_DIR/.venv/bin/pip" install --quiet --upgrade pip
# Editable install pulls in pyproject deps (torch, opencv, etc.).
sudo -u "$RUN_USER" "$INSTALL_DIR/.venv/bin/pip" install --quiet -e "$INSTALL_DIR[dev]"

echo "[setup] installing systemd units"
install -m 0644 "$INSTALL_DIR/deploy/systemd/rfcai-ingestion-daemon.service" /etc/systemd/system/
install -m 0644 "$INSTALL_DIR/deploy/systemd/rfcai-auto-retrain.service"     /etc/systemd/system/
install -m 0644 "$INSTALL_DIR/deploy/systemd/rfcai-auto-retrain.timer"       /etc/systemd/system/
install -m 0644 "$INSTALL_DIR/deploy/systemd/rfcai-sync.service"             /etc/systemd/system/
install -m 0644 "$INSTALL_DIR/deploy/systemd/rfcai-sync.timer"               /etc/systemd/system/
systemctl daemon-reload

if [ ! -f /etc/default/rfcai-training ]; then
    install -m 0640 -o root -g root \
        "$INSTALL_DIR/deploy/systemd/rfcai-training.env.example" \
        /etc/default/rfcai-training
fi
if [ ! -f /etc/default/rfcai-sync ]; then
    install -m 0640 -o root -g root \
        "$INSTALL_DIR/deploy/systemd/rfcai-sync.env.example" \
        /etc/default/rfcai-sync
fi

# Ensure sync script is executable
chmod 0755 "$INSTALL_DIR/scripts/sync_with_relay.sh"

echo "[setup] generating SSH key for rfcai (if missing)"
if [ ! -f "$RUN_HOME/.ssh/id_ed25519" ]; then
    sudo -u "$RUN_USER" mkdir -p "$RUN_HOME/.ssh"
    sudo -u "$RUN_USER" chmod 700 "$RUN_HOME/.ssh"
    sudo -u "$RUN_USER" ssh-keygen -t ed25519 -N "" -f "$RUN_HOME/.ssh/id_ed25519" -C "rfcai-training@$(hostname)"
fi
PUB_KEY=$(cat "$RUN_HOME/.ssh/id_ed25519.pub")

cat <<EOF

[setup] DONE.

Next steps:
  1. Edit /etc/default/rfcai-sync if your relay host/user differs from
     the defaults (RFCAI_RELAY_HOST=aired.com, RFCAI_RELAY_USER=rfcai).
  2. Add this training machine's SSH public key to the relay host so
     rsync over SSH works without a password. Run on the relay:

        sudo -u rfcai mkdir -p /home/rfcai/.ssh
        sudo -u rfcai chmod 700 /home/rfcai/.ssh
        echo '$PUB_KEY' | sudo -u rfcai tee -a /home/rfcai/.ssh/authorized_keys
        sudo -u rfcai chmod 600 /home/rfcai/.ssh/authorized_keys

  3. Test the connection from this machine:

        sudo -u rfcai ssh rfcai@aired.com 'echo ok'

  4. Enable + start the timers / daemon:

        sudo systemctl enable --now rfcai-ingestion-daemon
        sudo systemctl enable --now rfcai-sync.timer
        sudo systemctl enable --now rfcai-auto-retrain.timer

  5. Watch logs:

        sudo journalctl -u rfcai-ingestion-daemon -f
        sudo journalctl -u rfcai-sync.service -f
        sudo journalctl -u rfcai-auto-retrain.service -f

EOF
