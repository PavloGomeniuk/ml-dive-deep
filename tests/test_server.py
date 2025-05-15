"""
Backend tests for FamilyRoom server.
Run: pytest tests/ -v
"""
import asyncio
import json
import sys
import os

import pytest
import pytest_asyncio
from aiohttp.test_utils import TestClient, TestServer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'server'))

import config
import db
from server import create_app, sessions_cache, online

# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, 'DB_PATH', str(tmp_path / 'test.db'))
    sessions_cache.clear()
    online.clear()
    return tmp_path


@pytest_asyncio.fixture
async def client(tmp_db, aiohttp_client):
    app = create_app()
    return await aiohttp_client(app)


# ── /api/join ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_join_success(client):
    resp = await client.post('/api/join', json={'name': 'Mum', 'invite_code': config.INVITE_CODE})
    assert resp.status == 200
    body = await resp.json()
    assert body['ok'] is True
    assert body['name'] == 'Mum'
    cookies = client.session.cookie_jar.filter_cookies(resp.url)
    assert 'session_id' in cookies


@pytest.mark.asyncio
async def test_join_wrong_code(client):
    resp = await client.post('/api/join', json={'name': 'Mum', 'invite_code': 'wrong'})
    assert resp.status == 401
    body = await resp.json()
    assert 'error' in body


@pytest.mark.asyncio
async def test_join_empty_name(client):
    resp = await client.post('/api/join', json={'name': '', 'invite_code': config.INVITE_CODE})
    assert resp.status == 400


@pytest.mark.asyncio
async def test_join_duplicate_name(client):
    # First join succeeds
    r1 = await client.post('/api/join', json={'name': 'Mum', 'invite_code': config.INVITE_CODE})
    assert r1.status == 200
    cookies = client.session.cookie_jar.filter_cookies(r1.url)
    assert 'session_id' in cookies
    sid = cookies['session_id'].value

    # Simulate active WS connection in `online`
    online[sid] = {'name': 'Mum', 'ws': None}

    # Second join with same name rejected
    r2 = await client.post('/api/join', json={'name': 'Mum', 'invite_code': config.INVITE_CODE})
    assert r2.status == 409
    body = await r2.json()
    assert 'already in use' in body['error'].lower()


@pytest.mark.asyncio
async def test_join_case_insensitive_duplicate(client):
    r1 = await client.post('/api/join', json={'name': 'Dad', 'invite_code': config.INVITE_CODE})
    assert r1.status == 200
    cookies = client.session.cookie_jar.filter_cookies(r1.url)
    if 'session_id' in cookies:
        online[cookies['session_id'].value] = {'name': 'Dad', 'ws': None}

    r2 = await client.post('/api/join', json={'name': 'dad', 'invite_code': config.INVITE_CODE})
    assert r2.status == 409


# ── /api/messages ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_messages_unauthorized(client):
    resp = await client.get('/api/messages')
    assert resp.status == 401


@pytest.mark.asyncio
async def test_messages_empty_after_join(client):
    await client.post('/api/join', json={'name': 'Ana', 'invite_code': config.INVITE_CODE})
    resp = await client.get('/api/messages')
    assert resp.status == 200
    body = await resp.json()
    assert isinstance(body, list)
    assert len(body) == 0


@pytest.mark.asyncio
async def test_messages_after_insert(client, tmp_db):
    await client.post('/api/join', json={'name': 'Ana', 'invite_code': config.INVITE_CODE})
    await db.add_message('Ana', 'hello family')
    resp = await client.get('/api/messages')
    assert resp.status == 200
    body = await resp.json()
    assert len(body) == 1
    assert body[0]['content'] == 'hello family'
    assert body[0]['user'] == 'Ana'


@pytest.mark.asyncio
async def test_messages_limited_to_100(client, tmp_db):
    await client.post('/api/join', json={'name': 'Ana', 'invite_code': config.INVITE_CODE})
    for i in range(105):
        await db.add_message('Ana', f'msg {i}')
    resp = await client.get('/api/messages')
    body = await resp.json()
    assert len(body) == 100


# ── /upload ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_upload_unauthorized(client):
    data = {'file': (b'fake', 'test.txt', 'text/plain')}
    resp = await client.post('/upload', data=data)
    assert resp.status == 401


@pytest.mark.asyncio
async def test_upload_png_image(client, tmp_path, monkeypatch):
    from server import UPLOAD_DIR
    import server
    monkeypatch.setattr(server, 'UPLOAD_DIR', tmp_path / 'uploads')
    (tmp_path / 'uploads').mkdir()

    await client.post('/api/join', json={'name': 'Ana', 'invite_code': config.INVITE_CODE})

    png_bytes = (
        b'\x89PNG\r\n\x1a\n'
        b'\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x00'
        b'\x00\x00\x00:~\x9bU\x00\x00\x00\nIDATx\x9cc`\x00\x00\x00\x02'
        b'\x00\x01\xe2!\xbc3\x00\x00\x00\x00IEND\xaeB`\x82'
    )
    data = {'file': png_bytes}
    from aiohttp import FormData
    form = FormData()
    form.add_field('file', png_bytes, filename='photo.png', content_type='image/png')
    resp = await client.post('/upload', data=form)
    assert resp.status == 200
    body = await resp.json()
    assert body['type'] == 'image'
    assert 'photo.png' in body['filename']


@pytest.mark.asyncio
async def test_upload_blocked_extension(client, tmp_path, monkeypatch):
    from server import UPLOAD_DIR
    import server
    monkeypatch.setattr(server, 'UPLOAD_DIR', tmp_path / 'uploads')
    (tmp_path / 'uploads').mkdir()

    await client.post('/api/join', json={'name': 'Ana', 'invite_code': config.INVITE_CODE})
    from aiohttp import FormData
    form = FormData()
    form.add_field('file', b'#!/bin/sh\nrm -rf /', filename='evil.sh', content_type='text/plain')
    resp = await client.post('/upload', data=form)
    assert resp.status == 400
    body = await resp.json()
    assert 'not allowed' in body['error'].lower()


@pytest.mark.asyncio
async def test_upload_filename_sanitization(client, tmp_path, monkeypatch):
    import server
    monkeypatch.setattr(server, 'UPLOAD_DIR', tmp_path / 'uploads')
    (tmp_path / 'uploads').mkdir()

    await client.post('/api/join', json={'name': 'Ana', 'invite_code': config.INVITE_CODE})
    png_bytes = (
        b'\x89PNG\r\n\x1a\n'
        b'\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x00'
        b'\x00\x00\x00:~\x9bU\x00\x00\x00\nIDATx\x9cc`\x00\x00\x00\x02'
        b'\x00\x01\xe2!\xbc3\x00\x00\x00\x00IEND\xaeB`\x82'
    )
    from aiohttp import FormData
    form = FormData()
    # Path traversal attempt
    form.add_field('file', png_bytes, filename='../../etc/passwd.png', content_type='image/png')
    resp = await client.post('/upload', data=form)
    assert resp.status == 200
    body = await resp.json()
    # Filename must not contain ..
    assert '..' not in body['filename']
    assert '/' not in body['filename']


# ── db.py ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_db_wal_mode(tmp_db):
    await db.init_db()
    import aiosqlite
    async with aiosqlite.connect(db.DB_PATH) as conn:
        async with conn.execute("PRAGMA journal_mode") as cur:
            row = await cur.fetchone()
    assert row[0] == 'wal'


@pytest.mark.asyncio
async def test_db_add_and_get_message(tmp_db):
    await db.init_db()
    await db.add_message('Mum', 'hello', type='text')
    msgs = await db.get_messages()
    assert len(msgs) == 1
    assert msgs[0]['content'] == 'hello'
    assert msgs[0]['user'] == 'Mum'
    assert msgs[0]['type'] == 'text'


@pytest.mark.asyncio
async def test_db_session_create_and_load(tmp_db):
    await db.init_db()
    await db.create_session('sess-1', 'Dad')
    loaded = await db.load_all_sessions()
    assert loaded.get('sess-1') == 'Dad'


@pytest.mark.asyncio
async def test_db_message_order(tmp_db):
    await db.init_db()
    import asyncio
    for i in range(5):
        await db.add_message('Ana', f'msg {i}')
        await asyncio.sleep(0.01)
    msgs = await db.get_messages()
    contents = [m['content'] for m in msgs]
    assert contents == [f'msg {i}' for i in range(5)]
