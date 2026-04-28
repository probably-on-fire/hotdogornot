# Deploy

Two Ubuntu machines:

- **Relay (`aired.com`)** — public-facing, accepts uploads, serves model files
- **Training (`ai.localradionetworks.com`)** — pulls uploads, retrains, pushes new model

## On the relay

```bash
git clone <REPO> /tmp/rfcai-bootstrap
sudo /tmp/rfcai-bootstrap/training/deploy/setup_relay.sh <REPO>
sudo systemctl enable --now rfcai-relay
```

Check `/etc/default/rfcai-relay` — the setup script generates a random
`RFCAI_DEVICE_TOKEN` for you. Share that token with the AR app team.

Put nginx in front for TLS:

```nginx
# /etc/nginx/sites-available/rfcai
server {
    listen 443 ssl http2;
    server_name aired.com;

    ssl_certificate     /etc/letsencrypt/live/aired.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/aired.com/privkey.pem;

    client_max_body_size 100M;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## On the training machine

```bash
git clone <REPO> /tmp/rfcai-bootstrap
sudo /tmp/rfcai-bootstrap/training/deploy/setup_training.sh <REPO>
```

The setup script generates an SSH key and prints the command to add it to
the relay's `authorized_keys`. Run that command on the relay.

Then:

```bash
sudo systemctl enable --now rfcai-ingestion-daemon
sudo systemctl enable --now rfcai-sync.timer
sudo systemctl enable --now rfcai-auto-retrain.timer
```

## Verify the loop

On the relay:

```bash
curl http://127.0.0.1:8000/healthz
# {"status":"ok","model_version":0}

curl -X POST http://127.0.0.1:8000/uploads \
  -H "X-Device-Token: $(grep RFCAI_DEVICE_TOKEN /etc/default/rfcai-relay | cut -d= -f2)" \
  -F "claimed_class=2.4mm-M" \
  -F "device_id=test" \
  -F "frames=@/path/to/test.jpg"
# {"upload_id":"...","n_frames_received":1,"claimed_class":"2.4mm-M"}
```

On the training machine, within ~2 minutes (next sync timer fire):

```bash
ls /home/rfcai/incoming/
# 20260428T120033_a8c7e1/

sudo journalctl -u rfcai-ingestion-daemon -n 20
# ...processed 20260428T... approve  (or quarantine)
```

After the first nightly retrain (or `sudo systemctl start rfcai-auto-retrain.service` for an immediate run):

```bash
ls /home/rfcai/training/models/connector_classifier/
# manifest.json  weights.0001.pt  weights.latest.pt  labels.json  version.json

# After the next sync.timer fires, those files appear on the relay too.
curl http://127.0.0.1:8000/model/version
# {"version":1}
```

## Files installed

| File                                       | Owner        | Purpose                                 |
|--------------------------------------------|--------------|-----------------------------------------|
| `/etc/systemd/system/rfcai-relay.service`  | root         | Relay (uvicorn)                         |
| `/etc/default/rfcai-relay`                 | root:root 0640 | Relay env (token, paths)             |
| `/etc/systemd/system/rfcai-ingestion-daemon.service` | root | Long-running upload processor       |
| `/etc/systemd/system/rfcai-auto-retrain.{service,timer}` | root | Nightly retrain                |
| `/etc/systemd/system/rfcai-sync.{service,timer}` | root | Bidirectional rsync, every 2 min       |
| `/etc/default/rfcai-training`              | root:root 0640 | Training env (paths)                  |
| `/etc/default/rfcai-sync`                  | root:root 0640 | Sync env (relay host, ssh key)        |
| `/opt/rfcai/training/`                     | rfcai        | Repo (training/ subtree)               |
| `/opt/rfcai/training/.venv/`               | rfcai        | Python venv                             |
| `/srv/rfcai/incoming/` (relay)             | rfcai        | Upload landing zone                     |
| `/srv/rfcai/models/connector_classifier/` (relay) | rfcai | Served model artifacts             |
| `/home/rfcai/incoming/` (training)         | rfcai        | Local upload mirror                    |
| `/home/rfcai/training/data/...`            | rfcai        | Labeled data + quarantine               |
| `/home/rfcai/training/models/...`          | rfcai        | Locally-trained model                  |

## Troubleshooting

**Relay 503 on /model/latest:** missing `RFCAI_DEVICE_TOKEN`. Set it in `/etc/default/rfcai-relay`, then `systemctl restart rfcai-relay`.

**Sync timer firing but nothing sync'ing:** check `sudo journalctl -u rfcai-sync.service -n 50`. Most likely SSH connectivity. Test manually as the rfcai user: `sudo -u rfcai ssh rfcai@aired.com 'echo ok'`.

**Auto-retrain skips every night:** dataset hasn't grown by `--min-new-samples` (default 20) since last train. Force a retrain with `sudo systemctl start rfcai-auto-retrain.service` after passing `--force` (you'll need to edit the unit's ExecStart for that one-time run).

**Daemon ignoring uploads:** they're missing the `.ready` sentinel. The relay writes it last. If you're testing manually, `touch /home/rfcai/incoming/<upload_id>/.ready` after dropping the frames.
