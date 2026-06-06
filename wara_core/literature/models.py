from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field


_STOPWORDS = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "from",
        "that",
        "this",
        "into",
        "over",
        "under",
        "using",
        "based",
        "toward",
        "towards",
        "between",
        "wireless",
        "communication",
        "communications",
    }
)


@dataclass(frozen=True)
class Author:
    name: str
    affiliation: str = ""

    def last_name(self) -> str:
        raw = (self.name.strip().split() or ["unknown"])[-1]
        folded = unicodedata.normalize("NFKD", raw).encode("ascii", "ignore").decode("ascii")
        return re.sub(r"[^a-zA-Z]", "", folded).lower() or "unknown"


@dataclass(frozen=True)
class Paper:
    paper_id: str
    title: str
    authors: tuple[Author, ...] = ()
    year: int = 0
    abstract: str = ""
    venue: str = ""
    citation_count: int = 0
    doi: str = ""
    arxiv_id: str = ""
    url: str = ""
    source: str = ""
    _bibtex_override: str = field(default="", repr=False)

    @property
    def cite_key(self) -> str:
        last = self.authors[0].last_name() if self.authors else "anon"
        year = str(self.year) if self.year else "0000"
        keyword = ""
        for word in self.title.split():
            cleaned = re.sub(r"[^a-zA-Z]", "", word).lower()
            if len(cleaned) > 3 and cleaned not in _STOPWORDS:
                keyword = cleaned
                break
        return f"{last}{year}{keyword}"

    def to_bibtex(self) -> str:
        if self._bibtex_override:
            return self._bibtex_override.strip()
        authors = " and ".join(author.name for author in self.authors) or "Unknown"
        venue = self.venue or "Unknown"
        entry_type = "inproceedings" if re.search(r"\b(proc|conference|workshop|symposium)\b", venue, re.IGNORECASE) else "article"
        venue_field = "booktitle" if entry_type == "inproceedings" else "journal"
        lines = [
            f"@{entry_type}{{{self.cite_key},",
            f"  title = {{{self.title}}},",
            f"  author = {{{authors}}},",
            f"  year = {{{self.year or 'Unknown'}}},",
            f"  {venue_field} = {{{venue}}},",
        ]
        if self.doi:
            lines.append(f"  doi = {{{self.doi}}},")
        if self.arxiv_id:
            lines.append(f"  eprint = {{{self.arxiv_id}}},")
            lines.append("  archiveprefix = {arXiv},")
        if self.url:
            lines.append(f"  url = {{{self.url}}},")
        lines.append("}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "authors": [{"name": author.name, "affiliation": author.affiliation} for author in self.authors],
            "year": self.year,
            "abstract": self.abstract,
            "venue": self.venue,
            "citation_count": self.citation_count,
            "doi": self.doi,
            "arxiv_id": self.arxiv_id,
            "url": self.url,
            "source": self.source,
            "cite_key": self.cite_key,
        }


__all__ = ["Author", "Paper"]
