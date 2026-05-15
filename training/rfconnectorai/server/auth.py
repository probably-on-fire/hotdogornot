"""
Labeler auth module — user CRUD, scrypt password hashing.

Pure stdlib, no external dependencies. Designed to be imported by
labeler.py (Task 2) and the seed script.
"""
from __future__ import annotations

import hashlib
import secrets
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class UserExists(Exception):
    """Raised when create_user() finds an existing username."""


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class User:
    id: int
    username: str
    password_hash: str
    role: str
    created_at: str
    last_login_at: Optional[str]


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS users (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    username       TEXT    UNIQUE NOT NULL,
    password_hash  TEXT    NOT NULL,
    role           TEXT    NOT NULL DEFAULT 'admin',
    created_at     TEXT    NOT NULL DEFAULT (datetime('now')),
    last_login_at  TEXT
)
"""


def init_db(db_path: Path) -> None:
    """Create the users table if it does not already exist. Idempotent."""
    with sqlite3.connect(db_path) as conn:
        conn.execute(_CREATE_TABLE_SQL)
        conn.commit()


def _row_to_user(row: tuple) -> User:
    id_, username, password_hash, role, created_at, last_login_at = row
    return User(
        id=id_,
        username=username,
        password_hash=password_hash,
        role=role,
        created_at=created_at,
        last_login_at=last_login_at,
    )


# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------

_SCRYPT_N = 2**14  # n=16384 — fast on CPU, resistant to GPU
_SCRYPT_R = 8
_SCRYPT_P = 1
_SCRYPT_DKLEN = 32
_SALT_BYTES = 16


def hash_password(plaintext: str) -> str:
    """Return a scrypt hash string: ``scrypt$<salt_hex>$<hash_hex>``."""
    salt = secrets.token_bytes(_SALT_BYTES)
    dk = hashlib.scrypt(
        plaintext.encode(),
        salt=salt,
        n=_SCRYPT_N,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
        dklen=_SCRYPT_DKLEN,
    )
    return f"scrypt${salt.hex()}${dk.hex()}"


def verify_password(plaintext: str, stored_hash: str) -> bool:
    """
    Verify *plaintext* against *stored_hash*.

    Returns False on any parsing error or mismatch — never raises.
    Uses ``secrets.compare_digest`` for constant-time comparison.
    """
    try:
        parts = stored_hash.split("$")
        if len(parts) != 3 or parts[0] != "scrypt":
            return False
        algo, salt_hex, hash_hex = parts
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(hash_hex)
        actual = hashlib.scrypt(
            plaintext.encode(),
            salt=salt,
            n=_SCRYPT_N,
            r=_SCRYPT_R,
            p=_SCRYPT_P,
            dklen=_SCRYPT_DKLEN,
        )
        return secrets.compare_digest(actual, expected)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------


def create_user(
    db_path: Path,
    username: str,
    password: str,
    role: str = "admin",
) -> User:
    """
    Insert a new user and return the created User.

    Raises :class:`UserExists` if *username* already exists.
    """
    password_hash = hash_password(password)
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.execute(
                "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                (username, password_hash, role),
            )
            conn.commit()
            user_id = cursor.lastrowid
    except sqlite3.IntegrityError as exc:
        if "UNIQUE" in str(exc).upper():
            raise UserExists(f"User '{username}' already exists") from exc
        raise
    row = _fetch_row_by_id(db_path, user_id)
    if row is None:
        raise RuntimeError(f"Failed to fetch newly created user id={user_id}")
    return _row_to_user(row)


def _fetch_row_by_id(db_path: Path, user_id: int) -> Optional[tuple]:
    with sqlite3.connect(db_path) as conn:
        return conn.execute(
            "SELECT id, username, password_hash, role, created_at, last_login_at "
            "FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()


def get_user_by_username(db_path: Path, username: str) -> Optional[User]:
    """Return the User for *username*, or None if not found."""
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT id, username, password_hash, role, created_at, last_login_at "
            "FROM users WHERE username = ?",
            (username,),
        ).fetchone()
    if row is None:
        return None
    return _row_to_user(row)


def authenticate(db_path: Path, username: str, password: str) -> Optional[User]:
    """
    Verify credentials and return the User on success.

    Updates *last_login_at* on success. Returns None on any failure.
    """
    user = get_user_by_username(db_path, username)
    if user is None:
        return None
    if not verify_password(password, user.password_hash):
        return None
    # Update last_login_at
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE users SET last_login_at = datetime('now') WHERE id = ?",
            (user.id,),
        )
        conn.commit()
    # Re-fetch to return updated record
    row = _fetch_row_by_id(db_path, user.id)
    return _row_to_user(row) if row else user


def list_users(db_path: Path) -> list[User]:
    """Return all users ordered by id."""
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT id, username, password_hash, role, created_at, last_login_at "
            "FROM users ORDER BY id"
        ).fetchall()
    return [_row_to_user(r) for r in rows]


def delete_user(db_path: Path, user_id: int) -> None:
    """Delete the user with *user_id*. No-op if not found."""
    with sqlite3.connect(db_path) as conn:
        conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
        conn.commit()


def update_last_login(db_path: Path, user_id: int) -> None:
    """Set last_login_at to now for *user_id*. No-op if not found.

    Exposed as a standalone function for callers that already hold a
    User object and don't need to re-authenticate (e.g. session resume).
    ``authenticate()`` already calls this internally on success.
    """
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE users SET last_login_at = datetime('now') WHERE id = ?",
            (user_id,),
        )
        conn.commit()
