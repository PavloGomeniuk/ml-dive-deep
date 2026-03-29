import os
import logging
from contextlib import asynccontextmanager

import aiosqlite

logger = logging.getLogger(__name__)

DB_PATH = os.environ.get("DB_PATH", "/app/data/secrets.db")

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS secrets (
    id TEXT PRIMARY KEY,
    creator_token_hash TEXT NOT NULL,
    ciphertext TEXT NOT NULL,
    nonce TEXT NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    max_views INTEGER,
    view_count INTEGER NOT NULL DEFAULT 0,
    first_viewed_at TEXT
)
"""


async def get_db() -> aiosqlite.Connection:
    # isolation_level=None disables Python sqlite3's implicit transaction management
    # so we control BEGIN/COMMIT/ROLLBACK explicitly.
    db = await aiosqlite.connect(DB_PATH, timeout=10, isolation_level=None)
    db.row_factory = aiosqlite.Row
    await db.execute("PRAGMA journal_mode=WAL")
    await db.execute("PRAGMA busy_timeout=5000")
    await db.execute("PRAGMA foreign_keys=ON")
    return db


async def init_db():
    async with aiosqlite.connect(DB_PATH, isolation_level=None) as db:
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute(CREATE_TABLE_SQL)
    logger.info({"event": "db_initialized", "db_path": DB_PATH})


@asynccontextmanager
async def db_context():
    db = await get_db()
    try:
        yield db
    finally:
        await db.close()
