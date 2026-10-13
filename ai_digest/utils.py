from __future__ import annotations

import hashlib
import html
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import feedparser
import httpx

from . import config

log = logging.getLogger("ai_digest")
_TRACKING = re.compile(r"^(utm_|ref$|ref_src$|fbclid$|gclid$|mc_)")


def normalize_url(url: str) -> str:
    parts = urlsplit(url.strip())
    query = urlencode([(k, v) for k, v in parse_qsl(parts.query) if not _TRACKING.match(k)])
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, query, ""))


def clean_text(raw: str, limit: int = 500) -> str:
    text = html.unescape(re.sub(r"<[^>]+>", " ", raw or ""))
    return re.sub(r"\s+", " ", text).strip()[:limit]


def make_item(title, url, source, published, snippet, origin) -> dict:
    norm = normalize_url(url)
    return {
        "id": hashlib.sha1(norm.encode()).hexdigest()[:12],
        "title": clean_text(title, 300),
        "url": norm,
        "source": source,
        "published": published.isoformat() if published else None,
        "snippet": clean_text(snippet),
        "origin": origin,
    }


def parse_json_array(text: str) -> list[dict]:
    start, end = (text or "").find("["), (text or "").rfind("]")
    if start == -1 or end <= start:
        return []
    try:
        return [d for d in json.loads(text[start : end + 1]) if isinstance(d, dict)]
    except json.JSONDecodeError:
        return []


# RSS
def _entry_datetime(entry) -> datetime | None:
    for key in ("published_parsed", "updated_parsed"):
        if entry.get(key):
            return datetime(*entry[key][:6], tzinfo=timezone.utc)
    return None


def fetch_feed(name: str, url: str, limit: int, cutoff: datetime) -> list[dict]:
    try:
        resp = httpx.get(url, headers={"User-Agent": config.USER_AGENT},
                         timeout=20, follow_redirects=True)
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        log.warning("Feed failed: %s (%s)", name, exc)
        return []
    items = []
    for entry in feedparser.parse(resp.content).entries:
        link, title = entry.get("link"), entry.get("title", "")
        if not link or not title:
            continue
        published = _entry_datetime(entry)
        if published and published < cutoff:
            continue
        items.append(make_item(title, link, name, published, entry.get("summary", ""), "rss"))
        if len(items) >= limit:
            break
    log.info("  %-26s %d items", name, len(items))
    return items


def collect_rss(cutoff: datetime) -> list[dict]:
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(fetch_feed, n, u, lim, cutoff) for n, u, lim in config.FEEDS]
        for fut in as_completed(futures):
            results.extend(fut.result())
    return results


# Web pages
def probe_url(url: str) -> tuple[str, str] | None:
    """Follow redirects. Return (final_url, page_title) if the page exists, else None.
    Also turns Gemini's grounding redirect links into real article URLs."""
    try:
        with httpx.stream("GET", url, headers={"User-Agent": config.USER_AGENT},
                          timeout=15, follow_redirects=True) as r:
            if r.status_code >= 400 and r.status_code not in (403, 429):  # 403/429 = real but blocks bots
                return None
            head = b""
            for chunk in r.iter_bytes():
                head += chunk
                if len(head) > 65536:
                    break
            m = re.search(rb"<title[^>]*>(.*?)</title>", head, re.I | re.S)
            return str(r.url), clean_text(m.group(1).decode("utf-8", "ignore"), 300) if m else ""
    except httpx.HTTPError:
        return None


def fetch_article(url: str) -> dict:
    """Download a web page and return its main text (max ~12,000 characters).

    Args:
        url: The full article URL to read.

    Returns:
        A dict with 'url', 'status' ('ok' or 'error'), and 'text'.
    """
    import trafilatura

    try:
        resp = httpx.get(url, headers={"User-Agent": config.USER_AGENT},
                         timeout=20, follow_redirects=True)
        resp.raise_for_status()
        text = trafilatura.extract(resp.text) or ""
        return {"url": url, "status": "ok", "text": text[:12000]}
    except Exception as exc:  # noqa: BLE001 - report any failure back to the agent
        return {"url": url, "status": "error", "text": f"Could not fetch: {exc}"}


# History
def load_history() -> list[dict]:
    try:
        return json.loads(config.HISTORY_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return []
