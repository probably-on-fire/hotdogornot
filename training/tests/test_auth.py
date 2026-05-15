"""
Tests for the labeler auth module — user CRUD, password hashing,
seeding behavior. Pure data layer; no routes, no middleware.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from rfconnectorai.server import auth


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "test_users.db"


def test_init_db_creates_users_table(db_path):
    auth.init_db(db_path)
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='users'"
        ).fetchall()
    assert rows  # table exists


def test_hash_and_verify_password_roundtrip():
    h = auth.hash_password("hunter2")
    assert h.startswith("scrypt$")  # algo prefix
    assert auth.verify_password("hunter2", h) is True
    assert auth.verify_password("wrong", h) is False
    assert auth.verify_password("", h) is False


def test_verify_rejects_malformed_hash():
    assert auth.verify_password("anything", "not-a-valid-hash") is False
    assert auth.verify_password("anything", "") is False


def test_create_and_get_user(db_path):
    auth.init_db(db_path)
    user = auth.create_user(db_path, "alice", "secretpw", role="admin")
    assert user.username == "alice"
    assert user.role == "admin"
    assert user.id > 0

    fetched = auth.get_user_by_username(db_path, "alice")
    assert fetched is not None
    assert fetched.id == user.id
    assert fetched.username == "alice"
    assert fetched.role == "admin"


def test_create_user_rejects_duplicate(db_path):
    auth.init_db(db_path)
    auth.create_user(db_path, "alice", "pw1", role="admin")
    with pytest.raises(auth.UserExists):
        auth.create_user(db_path, "alice", "pw2", role="admin")


def test_get_user_by_username_returns_none_for_missing(db_path):
    auth.init_db(db_path)
    assert auth.get_user_by_username(db_path, "ghost") is None


def test_authenticate_returns_user_on_correct_password(db_path):
    auth.init_db(db_path)
    auth.create_user(db_path, "alice", "correct horse", role="admin")
    user = auth.authenticate(db_path, "alice", "correct horse")
    assert user is not None
    assert user.username == "alice"


def test_authenticate_returns_none_on_wrong_password(db_path):
    auth.init_db(db_path)
    auth.create_user(db_path, "alice", "correct horse", role="admin")
    assert auth.authenticate(db_path, "alice", "wrong") is None
    assert auth.authenticate(db_path, "alice", "") is None


def test_authenticate_returns_none_for_missing_user(db_path):
    auth.init_db(db_path)
    assert auth.authenticate(db_path, "ghost", "anything") is None


def test_list_users(db_path):
    auth.init_db(db_path)
    auth.create_user(db_path, "alice", "pw", role="admin")
    auth.create_user(db_path, "bob", "pw", role="admin")
    users = auth.list_users(db_path)
    names = [u.username for u in users]
    assert "alice" in names
    assert "bob" in names


def test_delete_user(db_path):
    auth.init_db(db_path)
    u = auth.create_user(db_path, "alice", "pw", role="admin")
    auth.delete_user(db_path, u.id)
    assert auth.get_user_by_username(db_path, "alice") is None


def test_init_db_is_idempotent(db_path):
    auth.init_db(db_path)
    auth.init_db(db_path)  # second call must not error
    auth.create_user(db_path, "alice", "pw", role="admin")
    auth.init_db(db_path)  # third call must not nuke data
    assert auth.get_user_by_username(db_path, "alice") is not None
