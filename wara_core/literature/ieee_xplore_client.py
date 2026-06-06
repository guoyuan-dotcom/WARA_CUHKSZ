from __future__ import annotations

import json
import logging
import os
import random
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from .models import Author, Paper

logger = logging.getLogger(__name__)

_BASE_URL = "https://ieeexploreapi.ieee.org/api/v1/search/articles"
_MAX_PER_REQUEST = 200
_MAX_RETRIES = int(os.environ.get("WCL_IEEE_XPLORE_MAX_RETRIES", "2"))
_MAX_WAIT_SEC = int(os.environ.get("WCL_IEEE_XPLORE_MAX_WAIT_SEC", "8"))
_RATE_LIMIT_SEC = float(os.environ.get("WCL_IEEE_XPLORE_RATE_LIMIT_SEC", "1.0"))

_last_request_time: float = 0.0
_rate_lock = threading.Lock()


def search_ieee_xplore(
    query: str,
    *,
    limit: int = 20,
    year_min: int = 0,
    api_key: str = "",
) -> list[Paper]:
    """Search IEEE Xplore metadata and return normalized Paper objects."""

    api_key = api_key or os.environ.get("IEEE_XPLORE_API_KEY", "") or os.environ.get("IEEE_API_KEY", "")
    setattr(search_ieee_xplore, "last_error", "")
    if not api_key:
        logger.info("IEEE Xplore search skipped: IEEE_XPLORE_API_KEY is not set")
        setattr(search_ieee_xplore, "last_error", "missing_api_key")
        return []

    query = str(query or "").strip()
    if not query:
        return []

    global _last_request_time  # noqa: PLW0603
    with _rate_lock:
        now = time.monotonic()
        elapsed = now - _last_request_time
        if elapsed < _RATE_LIMIT_SEC:
            time.sleep(_RATE_LIMIT_SEC - elapsed)
        _last_request_time = time.monotonic()

    limit = max(1, min(int(limit), _MAX_PER_REQUEST))
    params: dict[str, str] = {
        "apikey": api_key,
        "format": "json",
        "querytext": query,
        "max_records": str(limit),
        "sort_field": "article_title",
        "sort_order": "asc",
    }
    if year_min > 0:
        params["start_year"] = str(int(year_min))

    url = f"{_BASE_URL}?{urllib.parse.urlencode(params)}"
    data = _request_with_retry(url)
    if data is None:
        return []

    articles = data.get("articles") or data.get("article") or []
    if isinstance(articles, dict):
        articles = [articles]
    if not isinstance(articles, list):
        return []

    papers: list[Paper] = []
    for item in articles:
        if not isinstance(item, dict):
            continue
        try:
            papers.append(_parse_ieee_article(item))
        except Exception:  # noqa: BLE001
            logger.debug("Failed to parse IEEE Xplore article: %s", item.get("article_number", "?"))
    return papers


def _request_with_retry(url: str) -> dict[str, Any] | None:
    for attempt in range(_MAX_RETRIES):
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "Accept": "application/json",
                    "User-Agent": "WARA/1.0 IEEE-Xplore-Metadata",
                },
            )
            with urllib.request.urlopen(req, timeout=None) as resp:
                body = resp.read().decode("utf-8")
                setattr(search_ieee_xplore, "last_error", "")
                return json.loads(body)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            detail = exc.headers.get("X-Error-Detail-Header", "") if exc.headers else ""
            mashery_code = exc.headers.get("X-Mashery-Error-Code", "") if exc.headers else ""
            error_summary = " ".join(part for part in (str(exc.code), mashery_code, detail, _clean_text(body)) if part).strip()
            setattr(search_ieee_xplore, "last_error", error_summary)
            if exc.code in (429, 500, 502, 503, 504):
                wait = min(2 ** (attempt + 1), _MAX_WAIT_SEC)
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                if retry_after:
                    try:
                        wait = min(float(retry_after), _MAX_WAIT_SEC)
                    except (TypeError, ValueError):
                        pass
                jitter = random.uniform(0, wait * 0.2)
                logger.warning("IEEE Xplore HTTP %d. Retry %d/%d in %.1fs", exc.code, attempt + 1, _MAX_RETRIES, wait + jitter)
                time.sleep(wait + jitter)
                continue
            logger.warning("IEEE Xplore HTTP %d for %s", exc.code, _redact_api_key(url))
            return None
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
            wait = min(2 ** attempt, _MAX_WAIT_SEC)
            jitter = random.uniform(0, wait * 0.2)
            logger.warning("IEEE Xplore request failed (%s). Retry %d/%d in %.1fs", exc, attempt + 1, _MAX_RETRIES, wait + jitter)
            time.sleep(wait + jitter)

    logger.error("IEEE Xplore request exhausted retries for: %s", _redact_api_key(url))
    return None


def _parse_ieee_article(item: dict[str, Any]) -> Paper:
    article_number = str(item.get("article_number") or item.get("arnumber") or item.get("document_id") or "")
    title = _clean_text(item.get("title") or item.get("article_title") or "")
    venue = _clean_text(item.get("publication_title") or item.get("conference_title") or item.get("journal_title") or "")
    doi = _clean_doi(item.get("doi") or "")
    url = str(item.get("html_url") or item.get("abstract_url") or item.get("pdf_url") or "")
    if not url and article_number:
        url = f"https://ieeexplore.ieee.org/document/{article_number}"
    return Paper(
        paper_id=f"ieee:{article_number or doi or _slug(title)}",
        title=title,
        authors=_parse_authors(item.get("authors")),
        year=_parse_year(item.get("publication_year") or item.get("publication_date")),
        abstract=_clean_text(item.get("abstract") or ""),
        venue=venue,
        citation_count=_parse_int(item.get("citing_paper_count")),
        doi=doi,
        url=url,
        source="ieee_xplore",
    )


def _parse_authors(raw: Any) -> tuple[Author, ...]:
    if isinstance(raw, dict):
        raw = raw.get("authors") or raw.get("author") or []
    if isinstance(raw, list):
        authors: list[Author] = []
        for entry in raw:
            if isinstance(entry, dict):
                name = _clean_text(entry.get("full_name") or entry.get("name") or entry.get("author") or "")
                affiliation = _clean_text(entry.get("affiliation") or "")
            else:
                name = _clean_text(entry)
                affiliation = ""
            if name:
                authors.append(Author(name=name, affiliation=affiliation))
        return tuple(authors)
    if isinstance(raw, str):
        names = [part.strip() for part in re.split(r";|\band\b|,", raw) if part.strip()]
        return tuple(Author(name=name) for name in names)
    return ()


def _parse_year(value: Any) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    match = re.search(r"(19|20)\d{2}", str(value or ""))
    return int(match.group(0)) if match else 0


def _parse_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _clean_text(value: Any) -> str:
    text = str(value or "")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _clean_doi(value: Any) -> str:
    return re.sub(r"^https?://(?:dx\.)?doi\.org/", "", _clean_text(value), flags=re.IGNORECASE).strip()


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:80] or "unknown"


def _redact_api_key(url: str) -> str:
    return re.sub(r"(apikey=)[^&]+", r"\1***", url)


__all__ = ["search_ieee_xplore", "_parse_ieee_article"]
