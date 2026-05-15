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
# DB connection helper (enables FK enforcement on every connection)
# ---------------------------------------------------------------------------


def _connect(db_path: Path) -> sqlite3.Connection:
    """Open a sqlite3 connection with PRAGMA foreign_keys = ON."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

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


@dataclass
class ApiToken:
    id: int
    user_id: int
    name: str
    created_at: str
    last_used_at: Optional[str]


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

_CREATE_USERS_SQL = """
CREATE TABLE IF NOT EXISTS users (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    username       TEXT    UNIQUE NOT NULL,
    password_hash  TEXT    NOT NULL,
    role           TEXT    NOT NULL DEFAULT 'admin',
    created_at     TEXT    NOT NULL DEFAULT (datetime('now')),
    last_login_at  TEXT
)
"""

_CREATE_API_TOKENS_SQL = """
CREATE TABLE IF NOT EXISTS api_tokens (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id      INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name         TEXT    NOT NULL,
    token_hash   TEXT    NOT NULL,
    created_at   TEXT    NOT NULL DEFAULT (datetime('now')),
    last_used_at TEXT
)
"""


def init_db(db_path: Path) -> None:
    """Create the users and api_tokens tables if they do not already exist. Idempotent."""
    with _connect(db_path) as conn:
        conn.execute(_CREATE_USERS_SQL)
        conn.execute(_CREATE_API_TOKENS_SQL)
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


def _row_to_token(row: tuple) -> ApiToken:
    return ApiToken(id=row[0], user_id=row[1], name=row[2],
                    created_at=row[3], last_used_at=row[4])


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
        with _connect(db_path) as conn:
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
    with _connect(db_path) as conn:
        return conn.execute(
            "SELECT id, username, password_hash, role, created_at, last_login_at "
            "FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()


def get_user_by_username(db_path: Path, username: str) -> Optional[User]:
    """Return the User for *username*, or None if not found."""
    with _connect(db_path) as conn:
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
    with _connect(db_path) as conn:
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
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT id, username, password_hash, role, created_at, last_login_at "
            "FROM users ORDER BY id"
        ).fetchall()
    return [_row_to_user(r) for r in rows]


def delete_user(db_path: Path, user_id: int) -> None:
    """Delete the user with *user_id*. No-op if not found."""
    with _connect(db_path) as conn:
        conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
        conn.commit()


def update_last_login(db_path: Path, user_id: int) -> None:
    """Set last_login_at to now for *user_id*. No-op if not found.

    Exposed as a standalone function for callers that already hold a
    User object and don't need to re-authenticate (e.g. session resume).
    ``authenticate()`` already calls this internally on success.
    """
    with _connect(db_path) as conn:
        conn.execute(
            "UPDATE users SET last_login_at = datetime('now') WHERE id = ?",
            (user_id,),
        )
        conn.commit()


# ---------------------------------------------------------------------------
# API token helpers
# ---------------------------------------------------------------------------


def create_token(db_path: Path, user_id: int, name: str) -> tuple[ApiToken, str]:
    """Generate a new API token for `user_id`.

    Returns (record, plaintext). The plaintext is shown to the caller
    exactly once — only the hash is stored. Subsequent verification
    uses verify_password() against the hash.
    """
    plaintext = secrets.token_urlsafe(32)
    token_hash = hash_password(plaintext)
    with _connect(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO api_tokens (user_id, name, token_hash) VALUES (?, ?, ?)",
            (user_id, name, token_hash),
        )
        token_id = cur.lastrowid
        row = conn.execute(
            "SELECT id, user_id, name, created_at, last_used_at FROM api_tokens WHERE id = ?",
            (token_id,),
        ).fetchone()
    return _row_to_token(row), plaintext


def authenticate_token(db_path: Path, plaintext_token: str) -> Optional[User]:
    """Look up a token across all rows; if any hash matches, update
    last_used_at and return the associated user. Returns None on
    mismatch, missing user, or malformed input."""
    if not plaintext_token:
        return None
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT id, user_id, token_hash FROM api_tokens"
        ).fetchall()
    for tok_id, user_id, token_hash in rows:
        if verify_password(plaintext_token, token_hash):
            # Update last_used_at + look up the user atomically.
            with _connect(db_path) as conn:
                conn.execute(
                    "UPDATE api_tokens SET last_used_at = datetime('now') WHERE id = ?",
                    (tok_id,),
                )
                urow = conn.execute(
                    "SELECT id, username, password_hash, role, created_at, last_login_at "
                    "FROM users WHERE id = ?",
                    (user_id,),
                ).fetchone()
            if urow is None:
                return None  # user was deleted; token is orphaned
            return User(*urow)
    return None


def list_tokens_for_user(db_path: Path, user_id: int) -> list[ApiToken]:
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT id, user_id, name, created_at, last_used_at "
            "FROM api_tokens WHERE user_id = ? ORDER BY id",
            (user_id,),
        ).fetchall()
    return [_row_to_token(r) for r in rows]


def delete_token(db_path: Path, token_id: int) -> None:
    with _connect(db_path) as conn:
        conn.execute("DELETE FROM api_tokens WHERE id = ?", (token_id,))
