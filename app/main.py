import asyncio
import hashlib
import hmac
import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta
from typing import Optional
from urllib.parse import urlparse

import aiosqlite
import httpx
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from cryptography.exceptions import InvalidTag
from fastapi import FastAPI, Form, HTTPException, Request, UploadFile, File
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from ulid import ULID

from app.crypto import derive_key, encrypt, decrypt
from app.database import init_db, db_context, DB_PATH
from app.models import (
    SecretCreate, SecretCreateResponse, SecretViewResponse,
    RevokeRequest, HealthResponse, EXPIRES_IN_MAP
)
from app.sweep import sweep_expired

logging.basicConfig(
    level=logging.INFO,
    format='{"ts": "%(asctime)s", "level": "%(levelname)s", "msg": %(message)s}',
)
logger = logging.getLogger(__name__)

SECRET_KEY = os.environ.get("SECRET_KEY", "")
NOTIFICATION_WEBHOOK_URL = os.environ.get("NOTIFICATION_WEBHOOK_URL", "")
MAX_SECRET_SIZE_KB = int(os.environ.get("MAX_SECRET_SIZE_KB", "1024"))
RATE_LIMIT_PER_IP = int(os.environ.get("RATE_LIMIT_PER_IP", "10"))
PORT = int(os.environ.get("PORT", "8000"))
BASE_URL = os.environ.get("BASE_URL", f"http://localhost:{PORT}")

# In-memory rate limiter: ip -> list of timestamps
_rate_limit_store: dict[str, list[float]] = {}

scheduler = AsyncIOScheduler()


def _check_startup():
    if not SECRET_KEY:
        raise RuntimeError("SECRET_KEY environment variable is required")
    if len(SECRET_KEY) < 32:
        logger.warning(
            '"SECRET_KEY is under 32 characters — encryption strength depends on key entropy. '
            'Use: openssl rand -hex 32"'
        )
    if NOTIFICATION_WEBHOOK_URL:
        parsed = urlparse(NOTIFICATION_WEBHOOK_URL)
        if parsed.scheme != "https":
            raise RuntimeError(
                f"NOTIFICATION_WEBHOOK_URL must use https:// scheme, got: {parsed.scheme}://"
            )


@asynccontextmanager
async def lifespan(app: FastAPI):
    _check_startup()
    await init_db()
    await sweep_expired()
    scheduler.add_job(sweep_expired, "interval", minutes=15, id="sweep")
    scheduler.start()
    logger.info(f'{{"event": "started", "port": {PORT}, "db_path": "{DB_PATH}"}}')
    yield
    scheduler.shutdown(wait=False)


app = FastAPI(title="SecretDrop", lifespan=lifespan)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
static_dir = os.path.join(BASE_DIR, "static")
templates_dir = os.path.join(BASE_DIR, "templates")

app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=templates_dir)

AES_KEY = None  # initialized after SECRET_KEY validated


def get_aes_key() -> bytes:
    return derive_key(SECRET_KEY)


# ---------- Exception handlers ----------

_ERROR_MESSAGES = {
    410: ("Secret Gone", "This secret has expired or has already been viewed."),
    429: ("Too Many Requests", "Too many requests. Please wait a moment and try again."),
    403: ("Forbidden", "Invalid creator token."),
    413: ("Payload Too Large", f"File too large. Maximum size is {MAX_SECRET_SIZE_KB}KB."),
    400: ("Bad Request", "File contains non-text content. Base64-encode it before sharing."),
}


def _wants_html(request: Request) -> bool:
    return "text/html" in request.headers.get("accept", "")


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    if _wants_html(request):
        title, message = _ERROR_MESSAGES.get(exc.status_code, ("Error", "Something went wrong."))
        return templates.TemplateResponse(
            "error.html",
            {"request": request, "title": title, "message": message},
            status_code=exc.status_code,
        )
    return JSONResponse(status_code=exc.status_code, content=exc.detail)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f'"event": "unhandled_exception", "error": "{type(exc).__name__}"')
    if _wants_html(request):
        return templates.TemplateResponse(
            "error.html",
            {"request": request, "title": "Error", "message": "Something went wrong. Please try again."},
            status_code=500,
        )
    return JSONResponse(status_code=500, content={"error": "internal_error"})


# ---------- Rate limiting ----------

def _is_rate_limited(ip: str) -> bool:
    import time
    now = time.time()
    window = 60.0
    timestamps = _rate_limit_store.get(ip, [])
    timestamps = [t for t in timestamps if now - t < window]
    if len(timestamps) >= RATE_LIMIT_PER_IP:
        _rate_limit_store[ip] = timestamps
        return True
    timestamps.append(now)
    _rate_limit_store[ip] = timestamps
    return False


# ---------- Webhook ----------

async def _fire_webhook(secret_id: str, viewed_at: str):
    if not NOTIFICATION_WEBHOOK_URL:
        return
    payload = {
        "event": "first_view",
        "secret_id": secret_id,
        "viewed_at": viewed_at,
    }
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.post(NOTIFICATION_WEBHOOK_URL, json=payload)
    except Exception as e:
        webhook_hash = hashlib.sha256(NOTIFICATION_WEBHOOK_URL.encode()).hexdigest()[:8]
        logger.warning(
            f'{{"event": "webhook_failed", "id": "{secret_id}", "error": "{e}", "url_hash": "{webhook_hash}"}}'
        )


# ---------- Routes ----------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "max_size_kb": MAX_SECRET_SIZE_KB},
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    try:
        async with db_context() as db:
            await db.execute("SELECT 1")
        return {"status": "ok", "db": "ok"}
    except Exception:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "db": "unavailable"},
        )


@app.post("/secrets", response_model=SecretCreateResponse)
async def create_secret(request: Request):
    content_type = request.headers.get("content-type", "")
    if "multipart/form-data" in content_type or "application/x-www-form-urlencoded" in content_type:
        form = await request.form()
        payload_field = form.get("payload")
        if hasattr(payload_field, "read"):
            raw = await payload_field.read()
            try:
                payload_text = raw.decode("utf-8")
            except UnicodeDecodeError:
                raise HTTPException(
                    status_code=400,
                    detail={"error": "payload_not_utf8", "hint": "base64-encode binary content before sharing"},
                )
        else:
            payload_text = str(payload_field) if payload_field else ""

        expires_in = str(form.get("expires_in", "24h"))
        max_views_raw = form.get("max_views", "1")
        try:
            sc = SecretCreate(payload=payload_text, expires_in=expires_in, max_views=max_views_raw)
        except Exception as e:
            raise HTTPException(status_code=422, detail=str(e))
    elif "application/json" in content_type:
        try:
            data = await request.json()
            sc = SecretCreate(**data)
        except Exception as e:
            raise HTTPException(status_code=422, detail=str(e))
    else:
        raise HTTPException(status_code=422, detail="Content-Type must be application/json or multipart/form-data")

    payload = sc.payload
    max_views = sc.max_views  # int or None

    # Validate UTF-8 (already a str here, but check for JSON path)
    try:
        payload.encode("utf-8")
    except UnicodeEncodeError:
        raise HTTPException(
            status_code=400,
            detail={"error": "payload_not_utf8", "hint": "base64-encode binary content before sharing"},
        )

    # Size check
    size_kb = len(payload.encode("utf-8")) / 1024
    if size_kb > MAX_SECRET_SIZE_KB:
        raise HTTPException(
            status_code=413,
            detail={"error": "payload_too_large", "max_kb": MAX_SECRET_SIZE_KB},
        )

    secret_id = str(ULID())
    creator_token = str(uuid.uuid4())
    creator_token_hash = hashlib.sha256(creator_token.encode()).hexdigest()

    now = datetime.now(timezone.utc)
    seconds = EXPIRES_IN_MAP[sc.expires_in]
    expires_at = now + timedelta(seconds=seconds)

    # Store as SQLite-compatible UTC string (no timezone suffix)
    now_str = now.strftime("%Y-%m-%d %H:%M:%S")
    expires_at_str = expires_at.strftime("%Y-%m-%d %H:%M:%S")

    ciphertext, nonce = encrypt(payload, get_aes_key())

    async with db_context() as db:
        await db.execute("BEGIN")
        await db.execute(
            """INSERT INTO secrets
               (id, creator_token_hash, ciphertext, nonce, created_at, expires_at, max_views, view_count)
               VALUES (?, ?, ?, ?, ?, ?, ?, 0)""",
            (
                secret_id,
                creator_token_hash,
                ciphertext,
                nonce,
                now_str,
                expires_at_str,
                max_views,
            ),
        )
        await db.execute("COMMIT")

    logger.info(
        f'{{"event": "created", "id": "{secret_id}", "expires_at": "{expires_at_str}", "max_views": {max_views}}}'
    )

    link = f"{BASE_URL}/secrets/{secret_id}"
    return SecretCreateResponse(
        link=link,
        creator_token=creator_token,
        expires_at=expires_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
    )


@app.get("/secrets/{secret_id}")
async def view_secret(secret_id: str, request: Request):
    client_ip = request.client.host if request.client else "unknown"

    if _is_rate_limited(client_ip):
        raise HTTPException(
            status_code=429,
            detail={"error": "rate_limited", "retry_after": 60},
        )

    fire_webhook = False
    webhook_viewed_at = None
    payload_text = None

    async with db_context() as db:
        try:
            await db.execute("BEGIN IMMEDIATE")

            # Step 1: SELECT
            cursor = await db.execute(
                "SELECT id, ciphertext, nonce, expires_at, max_views, view_count, first_viewed_at "
                "FROM secrets WHERE id = ? AND expires_at > datetime('now')",
                (secret_id,),
            )
            row = await cursor.fetchone()
            if not row:
                await db.execute("ROLLBACK")
                raise HTTPException(status_code=410, detail={"error": "gone"})

            # Step 2: Atomic UPDATE
            cursor2 = await db.execute(
                """UPDATE secrets
                   SET view_count = view_count + 1,
                       first_viewed_at = COALESCE(first_viewed_at, datetime('now'))
                   WHERE id = ? AND (max_views IS NULL OR view_count < max_views)
                   RETURNING view_count, first_viewed_at, max_views""",
                (secret_id,),
            )
            updated = await cursor2.fetchone()
            if not updated:
                await db.execute("ROLLBACK")
                raise HTTPException(status_code=410, detail={"error": "gone"})

            new_view_count = updated[0]
            first_viewed_at = updated[1]
            max_views = updated[2]

            # Step 3: Delete if view limit hit
            if max_views is not None and new_view_count >= max_views:
                await db.execute("DELETE FROM secrets WHERE id = ?", (secret_id,))

            # Step 4: Decrypt
            try:
                payload_text = decrypt(row["ciphertext"], row["nonce"], get_aes_key())
            except InvalidTag:
                await db.execute("ROLLBACK")
                raise HTTPException(status_code=410, detail={"error": "gone"})

            await db.execute("COMMIT")

            # Determine if webhook should fire (first_viewed_at was just set)
            if new_view_count == 1:
                fire_webhook = True
                webhook_viewed_at = first_viewed_at

        except HTTPException:
            raise
        except Exception as e:
            try:
                await db.execute("ROLLBACK")
            except Exception:
                pass
            raise e

    logger.info(
        f'{{"event": "viewed", "id": "{secret_id}", "view_count": {new_view_count}, "ip": "{client_ip}", "first_view": {fire_webhook}}}'
    )

    # Step 5: Fire webhook async after commit
    if fire_webhook and webhook_viewed_at:
        asyncio.create_task(_fire_webhook(secret_id, webhook_viewed_at))

    # Check if this is a browser request (Accept: text/html)
    accept = request.headers.get("accept", "")
    if "text/html" in accept:
        return templates.TemplateResponse(
            "view.html",
            {"request": request, "payload": payload_text},
        )

    return SecretViewResponse(payload=payload_text, content_type="text/plain")


@app.post("/secrets/{secret_id}/revoke")
async def revoke_secret(secret_id: str, body: RevokeRequest):
    submitted_hash = hashlib.sha256(body.creator_token.encode()).hexdigest()

    async with db_context() as db:
        await db.execute("BEGIN")
        cursor = await db.execute(
            "SELECT creator_token_hash FROM secrets WHERE id = ?",
            (secret_id,),
        )
        row = await cursor.fetchone()
        if not row:
            await db.execute("ROLLBACK")
            raise HTTPException(status_code=410, detail={"error": "gone"})

        stored_hash = row["creator_token_hash"]
        if not hmac.compare_digest(submitted_hash, stored_hash):
            await db.execute("ROLLBACK")
            raise HTTPException(status_code=403, detail={"error": "forbidden"})

        await db.execute("DELETE FROM secrets WHERE id = ?", (secret_id,))
        await db.execute("COMMIT")

    logger.info(f'{{"event": "revoked", "id": "{secret_id}"}}')
    return {"status": "revoked"}
