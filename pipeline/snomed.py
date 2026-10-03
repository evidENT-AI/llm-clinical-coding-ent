#!/usr/bin/env python3
"""Search SNOMED CT concepts on a Snowstorm terminology server."""

import json
import logging
from pathlib import Path

import requests

from pipeline.config import SNOMED_SERVER_URL

logger = logging.getLogger(__name__)

SEMANTIC_TAGS = {"diagnos": "disorder", "procedure": "procedure"}


def semantic_tag(qualifier):
    """Return the SNOMED CT semantic tag for a label category.

    Args:
        qualifier: Label category, e.g. "diagnoses" or "procedures".

    Returns:
        The semantic tag, or None for other categories.
    """
    qualifier = str(qualifier).lower()
    for key, tag in SEMANTIC_TAGS.items():
        if key in qualifier:
            return tag
    return None


class SnomedSearcher:
    """Thin client for the Snowstorm description search endpoint.

    Attributes:
        server_url: Base URL of the Snowstorm server.
        timeout: Request timeout in seconds.
    """

    def __init__(self, server_url=SNOMED_SERVER_URL, timeout=30):
        self.server_url = server_url
        self.timeout = timeout

    def search(self, term, tag=None):
        """Search active descriptions for a term.

        Args:
            term: Search text.
            tag: Optional semantic tag filter, e.g. "disorder".

        Returns:
            The raw Snowstorm response, or `{"items": []}` on a request error.
        """
        params = {
            "term": term,
            "active": "true",
            "conceptActive": "true",
            "searchMode": "STANDARD",
        }
        if tag:
            params["semanticTag"] = tag
        try:
            response = requests.get(f"{self.server_url}/browser/MAIN/descriptions",
                                    params=params, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as error:
            logger.error("SNOMED search failed for %r: %s", term, error)
            return {"items": []}

    def candidates(self, term, tag=None):
        """Search and return candidate concepts as `{"items": [{id, term}]}`.

        Args:
            term: Search text.
            tag: Optional semantic tag filter.

        Returns:
            Simplified results.
        """
        return simplify(self.search(term, tag))

    def search_labels(self, labels, output_dir):
        """Search every label and save full and simplified results.

        Results for row `i` are written to `<i>_<text>_full.json` and
        `<i>_<text>.json`.

        Args:
            labels: DataFrame with `text` and `qualifier` columns.
            output_dir: Directory for the result files.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for idx, row in labels.iterrows():
            results = self.search(row["text"], semantic_tag(row["qualifier"]))
            stem = f"{idx}_{safe_name(row['text'])}"
            _write(output_dir / f"{stem}_full.json", results)
            _write(output_dir / f"{stem}.json", simplify(results))


def simplify(results):
    """Reduce a Snowstorm response to concept ids and display terms.

    The fully specified name is preferred, then the preferred term, then the
    matched description.

    Args:
        results: Raw Snowstorm response.

    Returns:
        `{"items": [{"term": ..., "id": ...}]}`.
    """
    items = []
    for item in results.get("items", []):
        concept = item.get("concept", {})
        term = (concept.get("fsn") or {}).get("term")
        if term is None:
            term = (concept.get("pt") or {}).get("term")
        if term is None:
            term = item.get("term")
        if term and "conceptId" in concept:
            items.append({"term": term, "id": concept["conceptId"]})
    return {"items": items}


def safe_name(text):
    """Return text safe to use in a file name."""
    return str(text).replace("/", "_").replace("\\", "_")


def _write(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
