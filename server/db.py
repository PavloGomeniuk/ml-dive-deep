import time
from pathlib import Path

import aiosqlite

DB_PATH = str(Path(__file__).parent / "data" / "family.db")


async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id       INTEGER PRIMARY KEY AUTOINCREMENT,
                user     TEXT    NOT NULL,
                content  TEXT    NOT NULL,
                type     TEXT    NOT NULL DEFAULT 'text',
                filename TEXT,
                ts       REAL    NOT NULL
            )
        """)
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_ts ON messages(ts DESC)"
        )
        await db.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id         TEXT PRIMARY KEY,
                name       TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        """)
        await db.commit()


async def get_messages(limit: int = 100) -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT id, user, content, type, filename, ts "
            "FROM messages ORDER BY ts DESC LIMIT ?",
            (limit,),
        ) as cursor:
            rows = await cursor.fetchall()
    return [dict(r) for r in reversed(rows)]


async def add_message(
    user: str,
    content: str,
    type: str = "text",
    filename: str | None = None,
) -> int:
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            "INSERT INTO messages (user, content, type, filename, ts) VALUES (?, ?, ?, ?, ?)",
            (user, content, type, filename, time.time()),
        )
        await db.commit()
        return cursor.lastrowid


async def create_session(session_id: str, name: str) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT OR REPLACE INTO sessions (id, name, created_at) VALUES (?, ?, ?)",
            (session_id, name, time.time()),
        )
        await db.commit()


async def get_session(session_id: str) -> dict | None:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT id, name FROM sessions WHERE id = ?", (session_id,)
        ) as cursor:
            row = await cursor.fetchone()
    return dict(row) if row else None


async def load_all_sessions() -> dict[str, str]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT id, name FROM sessions") as cursor:
            rows = await cursor.fetchall()
    return {r["id"]: r["name"] for r in rows}
