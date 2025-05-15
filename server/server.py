import asyncio
import json
import re
import time
import uuid
from pathlib import Path

from aiohttp import web

import config
import db

UPLOAD_DIR = Path(__file__).parent / "uploads"
STATIC_DIR = Path(__file__).parent / "static"

# session_id -> name (all valid cookie sessions)
sessions_cache: dict[str, str] = {}

# session_id -> {"name": str, "ws": WebSocketResponse}
online: dict[str, dict] = {}

upload_lock = asyncio.Lock()

MAGIC: dict[bytes, str] = {
    b"\x89PNG": "image/png",
    b"\xff\xd8": "image/jpeg",
    b"GIF8": "image/gif",
    b"RIFF": "image/webp",
    b"%PDF": "application/pdf",
    b"PK\x03\x04": "application/zip",
}
BLOCKED_EXTS = {".exe", ".sh", ".py", ".bat", ".cmd", ".ps1", ".php", ".rb"}
IMAGE_MIMES = {"image/png", "image/jpeg", "image/gif", "image/webp"}


# ── helpers ───────────────────────────────────────────────────────────────────

async def _broadcast(msg: dict, exclude: str | None = None) -> None:
    data = json.dumps(msg)
    dead: list[str] = []
    for sid, info in list(online.items()):
        if sid == exclude:
            continue
        try:
            await info["ws"].send_str(data)
        except Exception:
            dead.append(sid)
    for sid in dead:
        await _remove_peer(sid)


async def _broadcast_presence() -> None:
    peers = [{"id": sid, "name": info["name"]} for sid, info in online.items()]
    await _broadcast({"type": "presence", "peers": peers})


async def _remove_peer(session_id: str) -> None:
    info = online.pop(session_id, None)
    if info:
        await _broadcast(
            {"type": "peer_left", "id": session_id, "name": info["name"]}
        )
        await _broadcast_presence()


def _require_session(request: web.Request) -> str | None:
    sid = request.cookies.get("session_id")
    if sid and sid in sessions_cache:
        return sid
    return None


# ── routes ────────────────────────────────────────────────────────────────────

async def index(request: web.Request) -> web.FileResponse:
    return web.FileResponse(STATIC_DIR / "index.html")


async def api_join(request: web.Request) -> web.Response:
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    name = (data.get("name") or "").strip()[:50]
    code = (data.get("invite_code") or "").strip()

    if not name:
        return web.json_response({"error": "Name required"}, status=400)
    if code != config.INVITE_CODE:
        return web.json_response({"error": "Wrong invite code"}, status=401)

    active_names = {info["name"].lower() for info in online.values()}
    if name.lower() in active_names:
        return web.json_response({"error": "Name already in use"}, status=409)

    session_id = str(uuid.uuid4())
    sessions_cache[session_id] = name
    await db.create_session(session_id, name)

    resp = web.json_response({"ok": True, "name": name})
    resp.set_cookie(
        "session_id",
        session_id,
        httponly=True,
        samesite="Lax",
        max_age=30 * 24 * 3600,
    )
    return resp


async def api_messages(request: web.Request) -> web.Response:
    if not _require_session(request):
        return web.json_response({"error": "Unauthorized"}, status=401)
    msgs = await db.get_messages(100)
    return web.json_response(msgs)


async def upload(request: web.Request) -> web.Response:
    sid = _require_session(request)
    if not sid:
        return web.json_response({"error": "Unauthorized"}, status=401)

    name = sessions_cache[sid]

    cl = request.content_length
    if cl and cl > config.UPLOAD_MAX_BYTES:
        return web.json_response({"error": "File too large (max 50MB)"}, status=413)

    reader = await request.multipart()
    field = await reader.next()
    if field is None or field.name != "file":
        return web.json_response({"error": "Expected field named 'file'"}, status=400)

    original = field.filename or "upload"
    # Strip path components before sanitizing characters
    basename = Path(original).name or original
    safe = re.sub(r"[^a-zA-Z0-9._-]", "_", basename).lstrip(".")
    # Also strip any remaining .. sequences
    safe = safe.replace("..", "_")
    if not safe:
        safe = "upload"

    ext = Path(safe).suffix.lower()
    if ext in BLOCKED_EXTS:
        return web.json_response({"error": f"File type not allowed: {ext}"}, status=400)

    data = await field.read()
    head = data[:8]

    if len(data) > config.UPLOAD_MAX_BYTES:
        return web.json_response({"error": "File too large (max 50MB)"}, status=413)

    mime = next((v for k, v in MAGIC.items() if head.startswith(k)), "application/octet-stream")

    async with upload_lock:
        total = sum(f.stat().st_size for f in UPLOAD_DIR.iterdir() if f.is_file())
        if total + len(data) > config.UPLOAD_QUOTA_BYTES:
            return web.json_response({"error": "Server storage full"}, status=507)
        file_id = f"{uuid.uuid4()}_{safe}"
        dest = UPLOAD_DIR / file_id
        try:
            dest.write_bytes(data)
        except OSError:
            return web.json_response({"error": "Server storage full"}, status=507)

    is_image = mime in IMAGE_MIMES
    msg_type = "image" if is_image else "file"
    url = f"/uploads/{file_id}"
    ts = time.time()
    await db.add_message(name, url, type=msg_type, filename=safe)

    msg = {
        "type": "chat",
        "user": name,
        "content": url,
        "msg_type": msg_type,
        "filename": safe,
        "ts": ts,
    }
    await _broadcast(msg)

    return web.json_response({"ok": True, "url": url, "filename": safe, "type": msg_type, "ts": ts})


async def ws_handler(request: web.Request) -> web.WebSocketResponse:
    sid = _require_session(request)
    if not sid:
        return web.Response(status=401)

    name = sessions_cache[sid]
    ws = web.WebSocketResponse(heartbeat=15)
    await ws.prepare(request)

    online[sid] = {"name": name, "ws": ws}

    peers = [
        {"id": s, "name": info["name"]}
        for s, info in online.items()
        if s != sid
    ]
    await ws.send_str(json.dumps({
        "type": "welcome",
        "id": sid,
        "name": name,
        "peers": peers,
        "stun": config.STUN_SERVERS,
    }))

    await _broadcast({"type": "peer_joined", "id": sid, "name": name}, exclude=sid)
    await _broadcast_presence()

    try:
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                try:
                    payload = json.loads(msg.data)
                except json.JSONDecodeError:
                    continue

                mtype = payload.get("type")

                if mtype == "chat":
                    content = (payload.get("content") or "").strip()
                    if content:
                        ts = time.time()
                        await db.add_message(name, content, type="text")
                        await _broadcast({
                            "type": "chat",
                            "user": name,
                            "content": content,
                            "msg_type": "text",
                            "ts": ts,
                        })

                elif mtype in ("offer", "answer", "ice-candidate"):
                    target = payload.get("target")
                    if target and target in online:
                        payload["from"] = sid
                        try:
                            await online[target]["ws"].send_str(json.dumps(payload))
                        except Exception:
                            pass

            elif msg.type in (web.WSMsgType.ERROR, web.WSMsgType.CLOSE):
                break
    except Exception:
        pass
    finally:
        await _remove_peer(sid)

    return ws


# ── app startup ───────────────────────────────────────────────────────────────

async def on_startup(app: web.Application) -> None:
    UPLOAD_DIR.mkdir(exist_ok=True)
    Path(db.DB_PATH).parent.mkdir(exist_ok=True)
    await db.init_db()
    cached = await db.load_all_sessions()
    sessions_cache.update(cached)


def create_app() -> web.Application:
    app = web.Application()
    app.on_startup.append(on_startup)
    app.router.add_get("/", index)
    app.router.add_post("/api/join", api_join)
    app.router.add_get("/api/messages", api_messages)
    app.router.add_post("/upload", upload)
    app.router.add_get("/ws", ws_handler)
    app.router.add_static("/uploads", UPLOAD_DIR, show_index=False)
    app.router.add_static("/static", STATIC_DIR, show_index=False)
    return app


if __name__ == "__main__":
    web.run_app(create_app(), port=config.PORT)
