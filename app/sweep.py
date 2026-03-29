import logging

import aiosqlite

from app.database import DB_PATH

logger = logging.getLogger(__name__)


async def sweep_expired():
    """Delete expired secrets. Runs on startup and every 15 minutes."""
    async with aiosqlite.connect(DB_PATH, isolation_level=None) as db:
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute("PRAGMA busy_timeout=5000")
        await db.execute("BEGIN")
        cursor = await db.execute(
            "DELETE FROM secrets WHERE expires_at < datetime('now') RETURNING id"
        )
        deleted_rows = await cursor.fetchall()
        deleted = len(deleted_rows)
        cursor2 = await db.execute("SELECT COUNT(*) FROM secrets")
        row = await cursor2.fetchone()
        remaining = row[0] if row else 0
        await db.execute("COMMIT")

    logger.info({"event": "sweep", "deleted": deleted, "remaining": remaining})
