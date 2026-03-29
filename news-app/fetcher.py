from pathlib import Path
from datetime import datetime, timezone
import json
import os

import feedparser
import requests
from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent
CACHE_PATH = BASE_DIR / "cache.json"
FALLBACK_PATH = BASE_DIR / "fallback_news.json"

load_dotenv(BASE_DIR / ".env")

BBC_WORLD_RSS = "https://feeds.bbci.co.uk/news/world/rss.xml"
BBC_TECH_RSS = "https://feeds.bbci.co.uk/news/technology/rss.xml"

SECURITY_KEYWORDS = {
    "security", "hack", "hacker", "hacking", "vulnerability", "breach",
    "cyber", "malware", "ransomware", "phishing", "exploit", "zero-day",
    "zero day", "cve", "attack", "intrusion", "spyware", "trojan",
}


def _parse_feed(url):
    try:
        feed = feedparser.parse(url)
        articles = []
        for entry in feed.entries[:15]:
            title = entry.get("title", "").strip()
            if not title:
                continue
            articles.append({
                "title": title,
                "summary": entry.get("summary", "").strip(),
                "link": entry.get("link", ""),
                "published": entry.get("published", ""),
                "source": "BBC",
            })
        return articles
    except Exception as e:
        print(f"RSS fetch error ({url}): {e}")
        return []


def _fetch_newsapi_security():
    api_key = os.environ.get("NEWSAPI_KEY")
    if not api_key:
        return []
    try:
        resp = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                "q": "cybersecurity OR hack OR vulnerability OR malware OR ransomware",
                "language": "en",
                "pageSize": 10,
                "sortBy": "publishedAt",
                "apiKey": api_key,
            },
            timeout=10,
        )
        resp.raise_for_status()
        articles = []
        for item in resp.json().get("articles", []):
            title = (item.get("title") or "").strip()
            if not title or title == "[Removed]":
                continue
            articles.append({
                "title": title,
                "summary": (item.get("description") or "").strip(),
                "link": item.get("url", ""),
                "published": item.get("publishedAt", ""),
                "source": (item.get("source") or {}).get("name", "NewsAPI"),
            })
        return articles
    except Exception as e:
        print(f"NewsAPI fetch error: {e}")
        return []


def _is_security(article):
    text = (article["title"] + " " + article["summary"]).lower()
    return any(kw in text for kw in SECURITY_KEYWORDS)


def _dedup(articles):
    seen, out = set(), []
    for a in articles:
        key = a["title"].lower()[:60]
        if key not in seen:
            seen.add(key)
            out.append(a)
    return out


def load_fallback():
    try:
        with open(FALLBACK_PATH) as f:
            return json.load(f)
    except Exception:
        return {"world_news": [], "web_security": []}


def fetch_and_cache():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Fetching news...")

    world_raw = _parse_feed(BBC_WORLD_RSS)
    tech_raw = _parse_feed(BBC_TECH_RSS)
    security_api = _fetch_newsapi_security()

    world_news = _dedup(world_raw)[:10]
    security_candidates = tech_raw + security_api
    web_security = _dedup([a for a in security_candidates if _is_security(a)])[:10]

    fallback = load_fallback()

    if not world_news:
        print("World RSS empty — using fallback world news")
        world_news = fallback.get("world_news", [])

    if not web_security:
        print("Security feeds empty — using fallback security news")
        web_security = fallback.get("web_security", [])

    result = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "world_news": world_news,
        "web_security": web_security,
    }

    with open(CACHE_PATH, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Cached {len(world_news)} world + {len(web_security)} security articles")
    return result


if __name__ == "__main__":
    fetch_and_cache()
