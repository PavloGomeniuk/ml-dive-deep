import pytest
import pytest_asyncio
import aiosqlite
import os
from datetime import datetime, timezone, timedelta

from app.database import DB_PATH
from app.sweep import sweep_expired


@pytest_asyncio.fixture
async def db_with_expired():
    async with aiosqlite.connect(DB_PATH, isolation_level=None) as db:
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute("BEGIN")
        # Insert an already-expired secret (SQLite-compatible UTC format)
        expired_at = (datetime.now(timezone.utc) - timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")
        await db.execute(
            """INSERT OR REPLACE INTO secrets
               (id, creator_token_hash, ciphertext, nonce, created_at, expires_at, max_views, view_count)
               VALUES ('expired-test-id', 'hash', 'ciphertext', 'nonce', datetime('now'), ?, 1, 0)""",
            (expired_at,),
        )
        # Insert a still-valid secret
        valid_at = (datetime.now(timezone.utc) + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")
        await db.execute(
            """INSERT OR REPLACE INTO secrets
               (id, creator_token_hash, ciphertext, nonce, created_at, expires_at, max_views, view_count)
               VALUES ('valid-test-id', 'hash', 'ciphertext', 'nonce', datetime('now'), ?, 1, 0)""",
            (valid_at,),
        )
        await db.execute("COMMIT")
    yield
    async with aiosqlite.connect(DB_PATH, isolation_level=None) as db:
        await db.execute("BEGIN")
        await db.execute("DELETE FROM secrets WHERE id IN ('expired-test-id', 'valid-test-id')")
        await db.execute("COMMIT")


@pytest.mark.asyncio
async def test_sweep_cleans_expired(db_with_expired):
    await sweep_expired()

    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute("SELECT id FROM secrets WHERE id = 'expired-test-id'")
        row = await cursor.fetchone()
        assert row is None, "Expired secret should be deleted"

        cursor2 = await db.execute("SELECT id FROM secrets WHERE id = 'valid-test-id'")
        row2 = await cursor2.fetchone()
        assert row2 is not None, "Valid secret should still exist"
