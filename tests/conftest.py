import os
import tempfile
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

os.environ.setdefault("SECRET_KEY", "test-secret-key-for-testing-only-32chars!!")
os.environ.setdefault("BASE_URL", "http://testserver")

# Use a temp DB for each test session
_tmp = tempfile.mktemp(suffix=".db")
os.environ["DB_PATH"] = _tmp


@pytest_asyncio.fixture(scope="session")
async def app():
    from app.main import app as _app
    from app.database import init_db
    await init_db()
    return _app


@pytest_asyncio.fixture
async def client(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as ac:
        yield ac
