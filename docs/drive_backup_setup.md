# Drive backup for /labeler/upload-video — setup

Every video uploaded via the Flutter Contribute screen also gets pushed
to a shared Google Drive folder. Disabled by default; enable by setting
two env vars on the box.

## One-time setup (do this once on chris@aired.com's account)

1. **Google Cloud project.** Go to https://console.cloud.google.com.
   Create a project ("rfcai-drive-backup" is fine) or pick an existing one.

2. **Enable the Drive API.** APIs & Services → Library → search "Google
   Drive API" → Enable.

3. **Create a service account.**
   - IAM & Admin → Service Accounts → Create.
   - Name: `rfcai-drive-writer`. No roles needed at the project level.
   - Continue → Done.

4. **Generate a JSON key.**
   - Click the service account → Keys → Add Key → Create new key → JSON.
   - The JSON downloads. Save the **service-account email** somewhere
     — looks like `rfcai-drive-writer@<project>.iam.gserviceaccount.com`.

5. **Create the destination Drive folder.**
   - In Drive, create a folder (e.g., `aired-rfcai-uploads/`).
   - Right-click → Share → add **the service-account email** with
     **Editor** access.
   - Share the same folder with Jerry (and anyone else who should see
     the videos) as Viewer or Editor.
   - Grab the folder ID from the URL: `https://drive.google.com/drive/folders/<FOLDER_ID>`.

## Deploy the credentials onto the box

```bash
# On the LAN box (where the labeler service runs)
sudo install -m 0600 -o rfcai -g rfcai \
  ~/Downloads/rfcai-drive-writer-<key>.json \
  /etc/rfcai-drive-credentials.json

# Append to /etc/default/rfcai-relay (or the LAN-box equivalent)
sudo tee -a /etc/default/<unit-env-file> <<'EOF'
RFCAI_DRIVE_FOLDER_ID=<FOLDER_ID>
RFCAI_DRIVE_CREDENTIALS_PATH=/etc/rfcai-drive-credentials.json
EOF

# Install the client libs into the service's venv (one-time)
sudo -u rfcai /opt/rfcai/training/.venv/bin/pip install \
  google-api-python-client google-auth

# Restart the service
sudo systemctl restart <predict-service>
```

## Verify

Upload a video via Contribute (or curl). Response should include:

```json
{
  "saved_crops": ...,
  "predictions": [...],
  "drive_backup": {
    "id": "1abc...",
    "webViewLink": "https://drive.google.com/file/d/1abc.../view"
  }
}
```

If `drive_backup` is `null`, env vars aren't set or the credentials
file isn't readable. Check the service journal for
`Drive backup failed:` lines.

## Notes

- The Drive API scope is `drive.file` — the service account can only
  see/manage files it created itself; it cannot read the rest of the
  user's Drive.
- The folder must be shared with the service-account email for the
  files to land in the right place. Without the share, files land in
  the service account's own (invisible) Drive.
- The service account's Drive doesn't count against the user's quota,
  but files in a *shared* folder count against the folder owner's
  quota. That's chris@aired.com.
- Backup is best-effort: any failure is logged and ignored; the
  upload-video response is unaffected (still returns 200 + predictions).
