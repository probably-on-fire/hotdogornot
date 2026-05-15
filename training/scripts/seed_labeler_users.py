"""
Seed the labeler users table with the initial admin accounts.

Idempotent: re-running is safe; existing users are skipped (their
passwords are NOT rotated). Use --force to rotate passwords for
existing users.

Output: prints USERNAME=<u> PASSWORD=<p> for any user created or
rotated. The plaintext password appears only here — record it.
"""
from __future__ import annotations

import argparse
import secrets
import sys
from pathlib import Path

# Allow running directly: python -m scripts.seed_labeler_users
sys.path.insert(0, str(Path(__file__).parent.parent))

from rfconnectorai.server import auth


SEED_USERS = ["chris", "jdcrunchman", "zapperman"]


def _make_password() -> str:
    # 24 chars of url-safe base64 -> ~144 bits of entropy. Plenty for
    # a labeler login and short enough to communicate.
    return secrets.token_urlsafe(18)


def seed(db_path: Path, force: bool) -> None:
    auth.init_db(db_path)
    for username in SEED_USERS:
        existing = auth.get_user_by_username(db_path, username)
        if existing and not force:
            print(f"skip   username={username} (already exists; use --force to rotate)")
            continue
        password = _make_password()
        if existing:
            # Force-rotate: delete + recreate. Preserves the seed flow shape.
            auth.delete_user(db_path, existing.id)
        auth.create_user(db_path, username, password, role="admin")
        print(f"seeded username={username} password={password}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--db",
        type=Path,
        default=Path("/opt/rfcai/repo/training/data/labeler_users.db"),
        help="SQLite path. Default matches production layout.",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Rotate passwords for existing users (default: skip).",
    )
    args = ap.parse_args()
    seed(args.db, args.force)


if __name__ == "__main__":
    main()
