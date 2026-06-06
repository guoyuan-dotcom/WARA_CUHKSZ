"""Unified literature search with deduplication.

Combines results from OpenAlex, Semantic Scholar, and arXiv,
deduplicates by DOI → arXiv ID → fuzzy title match, and returns
a merged list sorted by citation count (descending).

Source priority: OpenAlex (most generous limits) → Semantic Scholar → arXiv.
If any source hits rate limits, remaining sources compensate automatically.

Public API
----------
- ``search_papers(query, limit, sources, year_min, deduplicate)``
  → ``list[Paper]``
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict
import importlib
import logging
import os
import re
import time
import urllib.error
from typing import Optional, cast

from wara_core.literature.arxiv_client import search_arxiv
from wara_core.literature.crossref_client import search_crossref
from wara_core.literature.ieee_xplore_client import search_ieee_xplore
from wara_core.literature.models import Author, Paper
from wara_core.literature.openalex_client import search_openalex
from wara_core.literature.semantic_scholar import search_semantic_scholar

logger = logging.getLogger(__name__)

# Wireless communications papers are heavily concentrated in IEEE venues, so
# IEEE Xplore should be first when an API key is available. Crossref is the
# stable no-key metadata fallback; Semantic Scholar adds citation counts.
# OpenAlex/arXiv remain available but are not the Phase 1 stability baseline
# because they can hit timeout/rate-limit bursts. Google Scholar is supported as
# an explicit supplement only because it is prone to blocking/CAPTCHA.
_DEFAULT_SOURCES = ("ieee_xplore", "crossref", "semantic_scholar")


CacheGet = Callable[[str, str, int], Optional[list[dict[str, object]]]]
CachePut = Callable[[str, str, int, list[dict[str, object]]], None]


def _resolve_sources(sources: Sequence[str]) -> tuple[str, ...]:
    """Apply optional runtime source filtering before any external request."""
    requested = [_normalize_source(src) for src in sources]
    override = os.environ.get("WCL_LITERATURE_SOURCES", "").strip()
    if override:
        requested = [_normalize_source(item) for item in override.split(",") if item.strip()]
    if os.environ.get("WCL_DISABLE_IEEE_XPLORE", "").strip().lower() in {"1", "true", "yes", "on"}:
        requested = [src for src in requested if src != "ieee_xplore"]
    if os.environ.get("WCL_DISABLE_SEMANTIC_SCHOLAR", "").strip().lower() in {"1", "true", "yes", "on"}:
        requested = [src for src in requested if src not in {"semantic_scholar", "s2"}]
    if os.environ.get("WCL_DISABLE_ARXIV", "").strip().lower() in {"1", "true", "yes", "on"}:
        requested = [src for src in requested if src != "arxiv"]
    return tuple(requested or ("openalex",))


def _normalize_source(source: object) -> str:
    value = str(source).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "ieee": "ieee_xplore",
        "xplore": "ieee_xplore",
        "ieeexplore": "ieee_xplore",
        "ieee_xplore_api": "ieee_xplore",
        "cross_ref": "crossref",
        "cross-ref": "crossref",
        "s2": "semantic_scholar",
        "scholar": "google_scholar",
        "google": "google_scholar",
        "googlescholar": "google_scholar",
    }
    return aliases.get(value, value)


def _cache_api() -> tuple[CacheGet, CachePut]:
    cache_mod = importlib.import_module("wara_core.literature.cache")
    return cast(CacheGet, cache_mod.get_cached), cast(CachePut, cache_mod.put_cache)


def _papers_to_dicts(papers: list[Paper]) -> list[dict[str, object]]:
    """Convert papers to serializable dicts for caching."""
    return [asdict(p) for p in papers]


def _as_int(value: object, default: int = 0) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def _dicts_to_papers(dicts: list[dict[str, object]]) -> list[Paper]:
    """Reconstruct Paper objects from cached dicts."""
    papers: list[Paper] = []
    for d in dicts:
        try:
            authors_raw = d.get("authors", ())
            if not isinstance(authors_raw, list):
                authors_raw = []
            authors = tuple(
                Author(
                    name=str(cast(dict[str, object], a).get("name", "")),
                    affiliation=str(cast(dict[str, object], a).get("affiliation", "")),
                )
                for a in authors_raw
                if isinstance(a, dict)
            )
            paper_id = cast(str, d["paper_id"])
            title = cast(str, d["title"])
            papers.append(
                Paper(
                    paper_id=paper_id,
                    title=title,
                    authors=authors,
                    year=_as_int(d.get("year", 0), 0),
                    abstract=str(d.get("abstract", "")),
                    venue=str(d.get("venue", "")),
                    citation_count=_as_int(d.get("citation_count", 0), 0),
                    doi=str(d.get("doi", "")),
                    arxiv_id=str(d.get("arxiv_id", "")),
                    url=str(d.get("url", "")),
                    source=str(d.get("source", "")),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return papers


def search_papers(
    query: str,
    *,
    limit: int = 20,
    sources: Sequence[str] = _DEFAULT_SOURCES,
    year_min: int = 0,
    deduplicate: bool = True,
    s2_api_key: str = "",
) -> list[Paper]:
    """Search multiple academic sources and return deduplicated results.

    Parameters
    ----------
    query:
        Free-text search query.
    limit:
        Maximum results *per source*.
    sources:
        Which backends to query.  Default: both S2 and arXiv.
    year_min:
        If >0, pass to backends that support year filtering.
    deduplicate:
        Whether to remove duplicates across sources.
    s2_api_key:
        Optional Semantic Scholar API key.

    Returns
    -------
    list[Paper]
        Merged results, sorted by citation_count descending.
    """
    all_papers: list[Paper] = []
    cache_get: CacheGet
    cache_put: CachePut
    cache_get, cache_put = _cache_api()
    sources = _resolve_sources(sources)

    source_stats: dict[str, int] = {}  # track per-source counts
    source_errors: dict[str, str] = {}
    cache_hits = 0

    for src in sources:
        src_lower = src.lower().replace("-", "_").replace(" ", "_")
        cache_source = (
            "semantic_scholar" if src_lower in ("semantic_scholar", "s2") else src_lower
        )
        try:
            if src_lower == "openalex":
                papers = search_openalex(
                    query,
                    limit=limit,
                    year_min=year_min,
                )
                all_papers.extend(papers)
                cache_put(query, "openalex", limit, _papers_to_dicts(papers))
                source_stats["openalex"] = len(papers)
                logger.info(
                    "OpenAlex returned %d papers for %r", len(papers), query
                )
                time.sleep(0.5)

            elif src_lower == "crossref":
                papers = search_crossref(
                    query,
                    limit=limit,
                    year_min=year_min,
                )
                all_papers.extend(papers)
                cache_put(query, "crossref", limit, _papers_to_dicts(papers))
                source_stats["crossref"] = len(papers)
                crossref_error = str(getattr(search_crossref, "last_error", "") or "")
                if crossref_error:
                    source_errors["crossref"] = crossref_error
                logger.info("Crossref returned %d papers for %r", len(papers), query)
                time.sleep(0.25)

            elif src_lower == "ieee_xplore":
                api_key = os.environ.get("IEEE_XPLORE_API_KEY", "") or os.environ.get("IEEE_API_KEY", "")
                if not api_key:
                    logger.info("IEEE Xplore skipped for %r: IEEE_XPLORE_API_KEY is not set", query)
                    source_stats["ieee_xplore"] = 0
                    continue
                papers = search_ieee_xplore(
                    query,
                    limit=limit,
                    year_min=year_min,
                    api_key=api_key,
                )
                all_papers.extend(papers)
                cache_put(query, "ieee_xplore", limit, _papers_to_dicts(papers))
                source_stats["ieee_xplore"] = len(papers)
                ieee_error = str(getattr(search_ieee_xplore, "last_error", "") or "")
                if ieee_error:
                    source_errors["ieee_xplore"] = ieee_error
                logger.info(
                    "IEEE Xplore returned %d papers for %r", len(papers), query
                )

            elif src_lower in ("semantic_scholar", "s2"):
                papers = search_semantic_scholar(
                    query,
                    limit=limit,
                    year_min=year_min,
                    api_key=s2_api_key,
                )
                all_papers.extend(papers)
                cache_put(query, "semantic_scholar", limit, _papers_to_dicts(papers))
                source_stats["semantic_scholar"] = len(papers)
                logger.info(
                    "Semantic Scholar returned %d papers for %r", len(papers), query
                )
                # Rate-limit gap before next source
                time.sleep(1.0)

            elif src_lower == "google_scholar":
                from wara_core.literature.google_scholar import GoogleScholarClient

                scholar_delay = float(os.environ.get("WCL_GOOGLE_SCHOLAR_DELAY", "2.0"))
                scholar_limit = min(limit, int(os.environ.get("WCL_GOOGLE_SCHOLAR_LIMIT", str(limit))))
                client = GoogleScholarClient(
                    inter_request_delay=scholar_delay,
                    use_proxy=os.environ.get("WCL_GOOGLE_SCHOLAR_USE_PROXY", "").strip().lower()
                    in {"1", "true", "yes", "on"},
                )
                scholar_papers = client.search(query, limit=scholar_limit)
                papers = [paper.to_literature_paper() for paper in scholar_papers]
                all_papers.extend(papers)
                cache_put(query, "google_scholar", limit, _papers_to_dicts(papers))
                source_stats["google_scholar"] = len(papers)
                logger.info(
                    "Google Scholar returned %d papers for %r", len(papers), query
                )

            elif src_lower == "arxiv":
                papers = search_arxiv(query, limit=limit, year_min=year_min)
                all_papers.extend(papers)
                cache_put(query, "arxiv", limit, _papers_to_dicts(papers))
                source_stats["arxiv"] = len(papers)
                logger.info("arXiv returned %d papers for %r", len(papers), query)

            else:
                logger.warning("Unknown literature source: %s (skipped)", src)
        except (
            ImportError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
            urllib.error.HTTPError,
            urllib.error.URLError,
        ):
            logger.warning(
                "[rate-limit] Source %s failed for %r — trying cache", src, query
            )
            cached = cache_get(query, cache_source, limit)
            if cached:
                papers = _dicts_to_papers(cached)
                all_papers.extend(papers)
                cache_hits += len(papers)
                logger.info(
                    "[cache] HIT: %d papers for %s/%r", len(papers), src, query
                )
            else:
                logger.warning(
                    "No cache available for %s/%r — skipping", src, query
                )
                source_stats.setdefault(src_lower, 0)
                source_errors[src_lower] = "source_failed_no_cache"

    # Summary log
    total = len(all_papers)
    parts = [f"{src}: {n}" for src, n in source_stats.items()]
    if cache_hits:
        parts.append(f"cache: {cache_hits}")
    setattr(search_papers, "last_source_stats", dict(source_stats))
    setattr(search_papers, "last_source_errors", dict(source_errors))
    setattr(search_papers, "last_cache_hits", cache_hits)
    logger.info(
        "[literature] Found %d papers (%s) for %r",
        total,
        ", ".join(parts) if parts else "none",
        query,
    )

    if deduplicate:
        all_papers = _deduplicate(all_papers)

    # Sort by citation count descending, then year descending
    all_papers.sort(key=lambda p: (p.citation_count, p.year), reverse=True)

    return all_papers


def search_papers_multi_query(
    queries: list[str],
    *,
    limit_per_query: int = 20,
    sources: Sequence[str] = _DEFAULT_SOURCES,
    year_min: int = 0,
    s2_api_key: str = "",
    inter_query_delay: float = 1.5,
) -> list[Paper]:
    """Run multiple queries and return deduplicated union.

    Adds a delay between queries to respect rate limits.
    """
    all_papers: list[Paper] = []
    aggregate_stats: dict[str, int] = {}
    aggregate_errors: dict[str, str] = {}
    total_cache_hits = 0

    for i, q in enumerate(queries):
        if i > 0:
            time.sleep(inter_query_delay)
        results = search_papers(
            q,
            limit=limit_per_query,
            sources=sources,
            year_min=year_min,
            s2_api_key=s2_api_key,
            deduplicate=False,  # we dedup globally below
        )
        all_papers.extend(results)
        last_stats = getattr(search_papers, "last_source_stats", {})
        if isinstance(last_stats, dict):
            for source, count in last_stats.items():
                aggregate_stats[source] = aggregate_stats.get(source, 0) + int(count or 0)
        last_errors = getattr(search_papers, "last_source_errors", {})
        if isinstance(last_errors, dict):
            aggregate_errors.update({str(source): str(error) for source, error in last_errors.items()})
        total_cache_hits += int(getattr(search_papers, "last_cache_hits", 0) or 0)
        logger.info("Query %d/%d %r → %d papers", i + 1, len(queries), q, len(results))

    deduped = _deduplicate(all_papers)
    deduped.sort(key=lambda p: (p.citation_count, p.year), reverse=True)
    setattr(search_papers_multi_query, "last_source_stats", aggregate_stats)
    setattr(search_papers_multi_query, "last_source_errors", aggregate_errors)
    setattr(search_papers_multi_query, "last_cache_hits", total_cache_hits)
    return deduped


# ------------------------------------------------------------------
# Deduplication
# ------------------------------------------------------------------


def _normalise_title(title: str) -> str:
    """Lower-case, strip punctuation, collapse whitespace."""
    t = title.lower()
    t = re.sub(r"[^a-z0-9\s]", "", t)
    return re.sub(r"\s+", " ", t).strip()


def _deduplicate(papers: list[Paper]) -> list[Paper]:
    """Remove duplicates.  Priority: DOI > arXiv ID > fuzzy title.

    When a duplicate is found, the entry with higher citation_count wins
    (i.e. Semantic Scholar data is preferred over arXiv-only data).
    """
    seen_doi: dict[str, int] = {}
    seen_arxiv: dict[str, int] = {}
    seen_title: dict[str, int] = {}
    result: list[Paper] = []

    def _update_indices(p: Paper, idx: int) -> None:
        """Register all identifiers of *p* in the lookup dicts at *idx*."""
        if p.doi:
            seen_doi[p.doi.lower().strip()] = idx
        if p.arxiv_id:
            seen_arxiv[p.arxiv_id.strip()] = idx
        norm = _normalise_title(p.title)
        if norm:
            seen_title[norm] = idx

    def _replace_at(old: Paper, new: Paper, idx: int) -> None:
        """Replace paper at *idx* and clean up stale index entries."""
        # Remove old identifiers that the new paper does NOT share
        if old.doi:
            old_doi = old.doi.lower().strip()
            new_doi = new.doi.lower().strip() if new.doi else ""
            if old_doi != new_doi and seen_doi.get(old_doi) == idx:
                del seen_doi[old_doi]
        if old.arxiv_id:
            old_ax = old.arxiv_id.strip()
            new_ax = new.arxiv_id.strip() if new.arxiv_id else ""
            if old_ax != new_ax and seen_arxiv.get(old_ax) == idx:
                del seen_arxiv[old_ax]
        old_norm = _normalise_title(old.title)
        new_norm = _normalise_title(new.title)
        if old_norm and old_norm != new_norm and seen_title.get(old_norm) == idx:
            del seen_title[old_norm]
        result[idx] = new
        _update_indices(new, idx)

    for paper in papers:
        is_dup = False

        # Check DOI
        if paper.doi:
            doi_key = paper.doi.lower().strip()
            if doi_key in seen_doi:
                idx = seen_doi[doi_key]
                if paper.citation_count > result[idx].citation_count:
                    _replace_at(result[idx], paper, idx)
                is_dup = True

        # Check arXiv ID
        if not is_dup and paper.arxiv_id:
            ax_key = paper.arxiv_id.strip()
            if ax_key in seen_arxiv:
                idx = seen_arxiv[ax_key]
                if paper.citation_count > result[idx].citation_count:
                    _replace_at(result[idx], paper, idx)
                is_dup = True

        # Check fuzzy title
        if not is_dup:
            norm = _normalise_title(paper.title)
            if norm and norm in seen_title:
                idx = seen_title[norm]
                if paper.citation_count > result[idx].citation_count:
                    _replace_at(result[idx], paper, idx)
                is_dup = True

        if is_dup:
            continue

        # Not a duplicate — store indices and append
        new_idx = len(result)
        _update_indices(paper, new_idx)
        result.append(paper)

    return result


def papers_to_bibtex(papers: Sequence[Paper]) -> str:
    """Generate a combined BibTeX file from a list of papers."""
    entries = [p.to_bibtex() for p in papers]
    return "\n\n".join(entries) + "\n"
