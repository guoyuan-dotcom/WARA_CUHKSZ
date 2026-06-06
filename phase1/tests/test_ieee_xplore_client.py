from __future__ import annotations

import sys
import unittest
from pathlib import Path


PHASE1_ROOT = Path(__file__).resolve().parents[1]
ENGINE = PHASE1_ROOT / "engine"
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from wara_core.literature.ieee_xplore_client import _parse_ieee_article  # noqa: E402


class IeeeXploreClientTest(unittest.TestCase):
    def test_parse_ieee_article_into_paper_model(self) -> None:
        paper = _parse_ieee_article(
            {
                "article_number": "1234567",
                "title": "Integrated Sensing Communication and Powering Beamforming",
                "publication_title": "IEEE Wireless Communications Letters",
                "publication_year": "2026",
                "abstract": "<p>Wireless beamforming abstract.</p>",
                "doi": "https://doi.org/10.1109/LWC.2026.1234567",
                "html_url": "https://ieeexplore.ieee.org/document/1234567",
                "citing_paper_count": "5",
                "authors": {
                    "authors": [
                        {"full_name": "Jane Doe", "affiliation": "Example University"},
                        {"full_name": "John Smith"},
                    ]
                },
            }
        )

        self.assertEqual(paper.source, "ieee_xplore")
        self.assertEqual(paper.paper_id, "ieee:1234567")
        self.assertEqual(paper.year, 2026)
        self.assertEqual(paper.venue, "IEEE Wireless Communications Letters")
        self.assertEqual(paper.citation_count, 5)
        self.assertEqual(paper.doi, "10.1109/LWC.2026.1234567")
        self.assertEqual([author.name for author in paper.authors], ["Jane Doe", "John Smith"])
        self.assertIn("Wireless beamforming abstract.", paper.abstract)


if __name__ == "__main__":
    unittest.main()
