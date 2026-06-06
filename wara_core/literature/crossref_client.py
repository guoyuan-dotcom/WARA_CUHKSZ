from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from wara_core.literature.models import Author, Paper


logger = logging.getLogger(__name__)

_BASE_URL = "https://api.crossref.org/works"
_MAX_RETRIES = int(os.environ.get("WARA_CROSSREF_MAX_RETRIES", "2") or 2)
_MAILTO = os.environ.get("WARA_CROSSREF_MAILTO", "").strip()


def search_crossref(query: str, *, limit: int = 20, year_min: int = 0) -> list[Paper]:
    """Search Crossref metadata.

    Crossref is used as a stable metadata verifier. It is not a
    browser-scraping source and usually returns DOI-backed records, which makes
    it safer than letting an LLM invent bibliography entries.
    """

    params = {
        "query.bibliographic": query,
        "rows": str(max(1, min(limit, 50))),
        "select": "DOI,title,author,published-print,published-online,container-title,is-referenced-by-count,URL,type",
    }
    filters = []
    if year_min:
        filters.append(f"from-pub-date:{int(year_min)}")
    if filters:
        params["filter"] = ",".join(filters)
    if _MAILTO:
        params["mailto"] = _MAILTO
    url = _BASE_URL + "?" + urllib.parse.urlencode(params)
    headers = {
        "User-Agent": "WARA/1.0 Crossref metadata verifier",
        "Accept": "application/json",
    }
    last_error = ""
    for attempt in range(_MAX_RETRIES + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=None) as resp:
                payload = json.loads(resp.read().decode("utf-8", errors="ignore"))
            items = payload.get("message", {}).get("items", [])
            papers = [_parse_crossref_item(item) for item in items if isinstance(item, dict)]
            result = [paper for paper in papers if paper is not None]
            setattr(search_crossref, "last_error", "")
            return result
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt >= _MAX_RETRIES:
                break
            time.sleep(min(2.0 * (attempt + 1), 5.0))
    logger.warning("Crossref search failed for %r: %s", query, last_error)
    setattr(search_crossref, "last_error", last_error)
    return []


def _parse_crossref_item(item: dict[str, Any]) -> Paper | None:
    title = _first_text(item.get("title"))
    if not title:
        return None
    doi = str(item.get("DOI") or "").strip()
    year = _published_year(item)
    authors = tuple(
        Author(name=_author_name(author))
        for author in item.get("author", [])
        if isinstance(author, dict) and _author_name(author)
    )
    venue = _first_text(item.get("container-title"))
    citation_count = _safe_int(item.get("is-referenced-by-count"))
    url = str(item.get("URL") or "").strip()
    paper_id = f"crossref:{doi.lower()}" if doi else f"crossref:{abs(hash(title))}"
    return Paper(
        paper_id=paper_id,
        title=title,
        authors=authors,
        year=year,
        venue=venue,
        citation_count=citation_count,
        doi=doi,
        url=url,
        source="crossref",
    )


def _first_text(value: Any) -> str:
    if isinstance(value, list):
        return str(value[0] or "").strip() if value else ""
    return str(value or "").strip()


def _author_name(author: dict[str, Any]) -> str:
    given = str(author.get("given") or "").strip()
    family = str(author.get("family") or "").strip()
    name = " ".join(part for part in (given, family) if part)
    return name or str(author.get("name") or "").strip()


def _published_year(item: dict[str, Any]) -> int:
    for key in ("published-print", "published-online", "published"):
        parts = item.get(key, {}).get("date-parts") if isinstance(item.get(key), dict) else None
        if isinstance(parts, list) and parts and isinstance(parts[0], list) and parts[0]:
            return _safe_int(parts[0][0])
    return 0


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
