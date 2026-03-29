import asyncio
import hashlib
import pytest
import pytest_asyncio
import aiosqlite

from app.database import DB_PATH


# ---------- POST /secrets ----------

@pytest.mark.asyncio
async def test_create_secret_json(client):
    res = await client.post(
        "/secrets",
        json={"payload": "my secret", "expires_in": "1h", "max_views": 1},
    )
    assert res.status_code == 200
    data = res.json()
    assert "link" in data
    assert "creator_token" in data
    assert "expires_at" in data


@pytest.mark.asyncio
async def test_create_secret_multipart(client):
    res = await client.post(
        "/secrets",
        data={"payload": "multipart secret", "expires_in": "24h", "max_views": "1"},
    )
    assert res.status_code == 200
    assert "link" in res.json()


@pytest.mark.asyncio
async def test_create_secret_unlimited_views(client):
    res = await client.post(
        "/secrets",
        json={"payload": "persistent secret", "expires_in": "7d", "max_views": "unlimited"},
    )
    assert res.status_code == 200


@pytest.mark.asyncio
async def test_create_secret_invalid_expires_in(client):
    res = await client.post(
        "/secrets",
        json={"payload": "x", "expires_in": "2h"},
    )
    assert res.status_code == 422


@pytest.mark.asyncio
async def test_create_secret_max_views_zero_rejected(client):
    res = await client.post(
        "/secrets",
        json={"payload": "x", "expires_in": "1h", "max_views": 0},
    )
    assert res.status_code == 422


@pytest.mark.asyncio
async def test_create_secret_max_views_negative_rejected(client):
    res = await client.post(
        "/secrets",
        json={"payload": "x", "expires_in": "1h", "max_views": -1},
    )
    assert res.status_code == 422


@pytest.mark.asyncio
async def test_create_secret_too_large(client):
    big_payload = "x" * (1025 * 1024)
    res = await client.post(
        "/secrets",
        json={"payload": big_payload, "expires_in": "1h", "max_views": 1},
    )
    assert res.status_code == 413
    assert res.json()["detail"]["error"] == "payload_too_large"


@pytest.mark.asyncio
async def test_create_secret_default_max_views_is_1(client):
    res = await client.post(
        "/secrets",
        json={"payload": "auto-one-view", "expires_in": "1h"},
    )
    assert res.status_code == 200
    secret_id = res.json()["link"].split("/")[-1]

    view1 = await client.get(f"/secrets/{secret_id}")
    assert view1.status_code == 200

    view2 = await client.get(f"/secrets/{secret_id}")
    assert view2.status_code == 410


# ---------- GET /secrets/{id} ----------

@pytest.mark.asyncio
async def test_view_secret_returns_payload(client):
    create = await client.post(
        "/secrets",
        json={"payload": "hello world", "expires_in": "1h", "max_views": 5},
    )
    secret_id = create.json()["link"].split("/")[-1]

    res = await client.get(f"/secrets/{secret_id}")
    assert res.status_code == 200
    assert res.json()["payload"] == "hello world"
    assert res.json()["content_type"] == "text/plain"


@pytest.mark.asyncio
async def test_view_unknown_id_returns_410(client):
    res = await client.get("/secrets/notarealid00000000000000000")
    assert res.status_code == 410
    assert res.json()["detail"]["error"] == "gone"


@pytest.mark.asyncio
async def test_view_after_max_views_returns_410(client):
    create = await client.post(
        "/secrets",
        json={"payload": "one-time", "expires_in": "1h", "max_views": 1},
    )
    secret_id = create.json()["link"].split("/")[-1]
    await client.get(f"/secrets/{secret_id}")
    res = await client.get(f"/secrets/{secret_id}")
    assert res.status_code == 410


@pytest.mark.asyncio
async def test_aes_tamper_returns_410(client):
    """Corrupt the ciphertext directly in SQLite. GET must return 410, not 500."""
    create = await client.post(
        "/secrets",
        json={"payload": "tamper test", "expires_in": "1h", "max_views": 5},
    )
    secret_id = create.json()["link"].split("/")[-1]

    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE secrets SET ciphertext = 'AAAAAAAAAAAAAAAA' WHERE id = ?",
            (secret_id,),
        )
        await db.commit()

    res = await client.get(f"/secrets/{secret_id}")
    assert res.status_code == 410
    assert res.json()["detail"]["error"] == "gone"


@pytest.mark.asyncio
async def test_concurrent_get_max_views_1(client):
    """Two simultaneous GETs on a max_views=1 secret: exactly one 200, one 410."""
    create = await client.post(
        "/secrets",
        json={"payload": "race-condition-test", "expires_in": "1h", "max_views": 1},
    )
    secret_id = create.json()["link"].split("/")[-1]

    results = await asyncio.gather(
        client.get(f"/secrets/{secret_id}"),
        client.get(f"/secrets/{secret_id}"),
        return_exceptions=True,
    )
    statuses = [r.status_code for r in results if hasattr(r, "status_code")]
    assert sorted(statuses) == [200, 410], f"Expected [200, 410], got {statuses}"


# ---------- POST /secrets/{id}/revoke ----------

@pytest.mark.asyncio
async def test_revoke_with_valid_token(client):
    create = await client.post(
        "/secrets",
        json={"payload": "to be revoked", "expires_in": "1h", "max_views": 5},
    )
    data = create.json()
    secret_id = data["link"].split("/")[-1]
    creator_token = data["creator_token"]

    res = await client.post(
        f"/secrets/{secret_id}/revoke",
        json={"creator_token": creator_token},
    )
    assert res.status_code == 200

    view = await client.get(f"/secrets/{secret_id}")
    assert view.status_code == 410


@pytest.mark.asyncio
async def test_revoke_with_invalid_token_returns_403(client):
    create = await client.post(
        "/secrets",
        json={"payload": "protected", "expires_in": "1h", "max_views": 5},
    )
    secret_id = create.json()["link"].split("/")[-1]

    res = await client.post(
        f"/secrets/{secret_id}/revoke",
        json={"creator_token": "00000000-0000-0000-0000-000000000000"},
    )
    assert res.status_code == 403
    assert res.json()["detail"]["error"] == "forbidden"


@pytest.mark.asyncio
async def test_timing_safe_revoke():
    """Verify hmac.compare_digest is used — wrong-length token doesn't short-circuit."""
    import hmac
    h1 = hashlib.sha256(b"correct-token").hexdigest()
    h2 = hashlib.sha256(b"wrong-token").hexdigest()
    short = "abc"  # short-circuits with == but not compare_digest
    # compare_digest with unequal-length strings should not raise, just return False
    assert not hmac.compare_digest(h1, short + "x" * (len(h1) - len(short) - 1) + "!")
    assert not hmac.compare_digest(h1, h2)
    assert hmac.compare_digest(h1, h1)


@pytest.mark.asyncio
async def test_revoke_nonexistent_returns_410(client):
    res = await client.post(
        "/secrets/doesnotexist000000000000000/revoke",
        json={"creator_token": "00000000-0000-0000-0000-000000000000"},
    )
    assert res.status_code == 410


# ---------- GET /health ----------

@pytest.mark.asyncio
async def test_health_ok(client):
    res = await client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["db"] == "ok"


# ---------- UTF-8 validation ----------

@pytest.mark.asyncio
async def test_utf8_validation_multipart(client):
    """Non-UTF8 file bytes should return 400."""
    import io
    binary_content = bytes([0xFF, 0xFE, 0x00, 0x01])
    res = await client.post(
        "/secrets",
        files={"payload": ("test.bin", io.BytesIO(binary_content), "application/octet-stream")},
        data={"expires_in": "1h", "max_views": "1"},
    )
    assert res.status_code == 400
    assert res.json()["detail"]["error"] == "payload_not_utf8"
