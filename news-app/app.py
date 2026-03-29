from pathlib import Path
import json
import os

from flask import Flask, jsonify, send_from_directory
from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv

from fetcher import fetch_and_cache, load_fallback, CACHE_PATH, BASE_DIR

load_dotenv(BASE_DIR / ".env")

# Startup check — warn early, not silently
if not os.environ.get("NEWSAPI_KEY"):
    print(
        "\n  WARNING: NEWSAPI_KEY not set in .env\n"
        "  Security news will use BBC Tech RSS only.\n"
        "  Get a free key at https://newsapi.org and add it to news-app/.env\n"
    )

STATIC_DIR = BASE_DIR / "static"
_debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() in ("1", "true")

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")
app.debug = _debug_mode


@app.route("/")
def index():
    return send_from_directory(str(STATIC_DIR), "index.html")


@app.route("/api/news")
def get_news():
    try:
        if CACHE_PATH.exists():
            with open(CACHE_PATH) as f:
                data = json.load(f)
            return jsonify(data)
        # First run — fetch immediately and cache
        data = fetch_and_cache()
        return jsonify(data)
    except Exception as e:
        print(f"Error serving /api/news: {e}")
        return jsonify(load_fallback())


# APScheduler guard: in Flask debug mode the reloader forks a child process.
# WERKZEUG_RUN_MAIN is set only in the child, so we start the scheduler once
# (in the child when debug=True, or in the main process when debug=False).
if os.environ.get("WERKZEUG_RUN_MAIN") or not _debug_mode:
    _scheduler = BackgroundScheduler()
    _scheduler.add_job(fetch_and_cache, "interval", hours=24, id="daily_fetch")
    _scheduler.start()


if __name__ == "__main__":
    app.run(debug=_debug_mode, host="0.0.0.0", port=5000)
