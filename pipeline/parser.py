#!/usr/bin/env python3
"""Parse and standardise the JSON labels returned by the model.

Repairs truncated JSON, maps free-text field values onto the allowed values
(by lookup, then spaCy and string similarity) and converts labels to a table.
"""

import json
import logging
import re

import pandas as pd
import spacy
from thefuzz import fuzz

logger = logging.getLogger(__name__)

FIELDS = (
    "context", "qualifier", "laterality", "presence", "primary_secondary",
    "experiencer", "treatment_stage", "snomed_ct",
)

ALLOWED = {
    "qualifier": {"Diagnoses", "Procedures"},
    "laterality": {"left", "right", "bilateral", "NA"},
    "presence": {"confirmed", "suspected", "resolved", "negated", "NA"},
    "primary_secondary": {"primary", "secondary", "NA"},
    "experiencer": {"Patient", "Family", "NA"},
    "treatment_stage": {"pre-treatment", "current", "post-treatment", "NA"},
}

SYNONYMS = {
    "qualifier": {
        "diagnosis": "Diagnoses",
        "diagnoses": "Diagnoses",
        "condition": "Diagnoses",
        "medical condition": "Diagnoses",
        "disease": "Diagnoses",
        "procedure": "Procedures",
        "procedures": "Procedures",
        "surgery": "Procedures",
        "operation": "Procedures",
        "surgical procedure": "Procedures",
    },
    "presence": {
        "present": "confirmed",
        "positive": "confirmed",
        "found": "confirmed",
        "observed": "confirmed",
        "suspect": "suspected",
        "possible": "suspected",
        "probable": "suspected",
        "past": "resolved",
        "historical": "resolved",
        "previous": "resolved",
        "absent": "negated",
        "negative": "negated",
        "none": "negated",
        "denied": "negated",
    },
    "treatment_stage": {
        "planned": "pre-treatment",
        "recommended": "pre-treatment",
        "scheduled": "pre-treatment",
        "pending": "pre-treatment",
        "proposed": "pre-treatment",
        "ongoing": "current",
        "active": "current",
        "in progress": "current",
        "current": "current",
        "undergoing": "current",
        "completed": "post-treatment",
        "done": "post-treatment",
        "performed": "post-treatment",
        "past": "post-treatment",
        "historical": "post-treatment",
    },
}


class LabelParser:
    """Extract, repair and standardise model labels.

    Attributes:
        threshold: Minimum combined similarity for a fuzzy match.
        nlp: Loaded spaCy pipeline used for semantic similarity.
    """

    def __init__(self, threshold=0.9, spacy_model="en_core_web_lg"):
        """Load the spaCy model.

        Args:
            threshold: Minimum combined similarity for a fuzzy match.
            spacy_model: Name of an installed spaCy model with word vectors.
        """
        self.threshold = threshold
        self.nlp = spacy.load(spacy_model)
        self._docs = {}

    def _doc(self, text):
        if text not in self._docs:
            self._docs[text] = self.nlp(text.lower())
        return self._docs[text]

    def closest_match(self, value, options, field):
        """Return the allowed value most similar to `value`.

        Similarity combines spaCy vectors (0.4), Levenshtein ratio (0.3) and
        token-sort ratio (0.3). A token that exactly equals an option wins.

        Args:
            value: Raw value from the model.
            options: Allowed values for the field.
            field: Field name.

        Returns:
            Tuple of (best option, similarity).
        """
        if not value or value.lower() == "na":
            return "NA", 0.0
        tokens = {token.text.lower() for token in self._doc(value)}
        for option in options:
            if option.lower() in tokens:
                return option, 1.0

        best, best_score = "NA", 0.0
        for option in options:
            score = (
                0.4 * self._doc(value).similarity(self._doc(option))
                + 0.3 * fuzz.ratio(value.lower(), option.lower()) / 100
                + 0.3 * fuzz.token_sort_ratio(value.lower(), option.lower()) / 100
            )
            if field == "qualifier" and any(
                    t in value.lower()
                    for t in ("diagnosis", "diagnoses", "procedure", "procedures")):
                score *= 1.2
            if score > best_score:
                best, best_score = option, score
        return best, best_score

    def standardise(self, field, value):
        """Map a raw value onto the allowed values for `field`.

        Args:
            field: Field name.
            value: Raw value from the model.

        Returns:
            The allowed value, or "NA" if nothing matches.
        """
        if value is None or value.lower() == "na":
            return "NA"
        value = value.lower()
        if value.title() in ALLOWED.get(field, {}):
            return value.title()
        mapped = SYNONYMS.get(field, {}).get(value)
        if mapped:
            return mapped
        if field in ALLOWED:
            match, score = self.closest_match(value, ALLOWED[field], field)
            if score >= self.threshold:
                return match
            logger.warning("No match for %r in %s (best %r, %.2f)",
                           value, field, match, score)
        return _keyword_match(field, value)

    def clean_label(self, entry):
        """Return a label with every field present and standardised.

        Args:
            entry: Field values for one label.

        Returns:
            Dict of standardised field values.
        """
        cleaned = {}
        for field in FIELDS:
            value = entry.get(field)
            if value is None:
                cleaned[field] = "NA"
            elif field in ALLOWED or field in SYNONYMS:
                cleaned[field] = self.standardise(field, str(value))
            else:
                cleaned[field] = str(value)
        return cleaned

    def parse(self, text, allow_partial=False):
        """Parse a model response into `{"labels": {text: fields}}`.

        Args:
            text: Raw model response.
            allow_partial: If the JSON cannot be repaired, return the complete
                labels found so far with `_partial` and `_last_key` set.

        Returns:
            Parsed labels.

        Raises:
            ValueError: If no JSON or no labels are found.
            json.JSONDecodeError: If the JSON is invalid and no partial result
                is allowed or found.
        """
        try:
            json_str = extract_json(text)
            if not json_str:
                raise ValueError("No JSON found in output")
            labels = json.loads(json_str).get("labels", {})
            if not labels:
                raise ValueError("No labels found in JSON")
            if isinstance(labels, list):
                labels = {item.pop("text"): item for item in labels
                          if isinstance(item, dict) and "text" in item}
            elif not isinstance(labels, dict):
                raise ValueError(f"Labels must be a dict or list, got {type(labels)}")
            cleaned = {}
            for label, entry in labels.items():
                try:
                    cleaned[label] = self.clean_label(entry)
                except Exception as error:
                    logger.error("Skipping label %r: %s", label, error)
            return {"labels": cleaned}
        except json.JSONDecodeError:
            if allow_partial:
                last_key, partial = complete_entries(text)
                if partial:
                    return {"labels": partial, "_partial": True, "_last_key": last_key}
            raise


def _keyword_match(field, value):
    """Fallback keyword rules for fields the similarity match missed."""
    if field == "experiencer":
        if any(t in value for t in ("patient", "self", "subject", "individual")):
            return "Patient"
        if any(t in value for t in ("family", "relative", "parent", "sibling",
                                    "mother", "father")):
            return "Family"
    elif field == "primary_secondary":
        if any(t in value for t in ("primary", "main", "principal", "chief")):
            return "primary"
        if any(t in value for t in ("secondary", "additional", "associated", "other")):
            return "secondary"
    elif field == "laterality":
        if any(t in value for t in ("left", "l", "left-sided", "left side")):
            return "left"
        if any(t in value for t in ("right", "r", "right-sided", "right side")):
            return "right"
        if any(t in value for t in ("bilateral", "both", "two-sided", "both sides")):
            return "bilateral"
    return "NA"


def extract_json(text):
    """Return the JSON object in a response, closing it if truncated.

    Args:
        text: Raw model response, possibly in a markdown code block.

    Returns:
        A JSON string, or None if none could be recovered.
    """
    fenced = re.search(r"```(?:json)?\s*(\{.*)\s*```", text, re.DOTALL)
    if fenced:
        json_str = fenced.group(1).rstrip()
        if not json_str.endswith("}"):
            missing = json_str.count("{") - json_str.count("}")
            if missing > 0:
                json_str += "}" * missing
        return json_str

    found = re.search(r"\{.*", text, re.DOTALL)
    if not found:
        return None
    json_str = found.group()
    try:
        json.loads(json_str)
        return json_str
    except json.JSONDecodeError as error:
        pos = getattr(error, "pos", len(json_str))
        braces = json_str[:pos].count("{") - json_str[:pos].count("}")
        brackets = json_str[:pos].count("[") - json_str[:pos].count("]")
        fixed = json_str[:pos].rstrip()
        if fixed.endswith(":"):
            fixed += ' "NA"'
        elif fixed.endswith(","):
            fixed = fixed[:-1]
        elif not fixed.endswith(('"', "}", "]")) and fixed.count('"') % 2 == 1:
            fixed += '"'
        fixed += "}" * braces + "]" * brackets
        try:
            json.loads(fixed)
            return fixed
        except json.JSONDecodeError:
            logger.error("Could not repair truncated JSON")
            return None


def complete_entries(text):
    """Return the labels that were fully written before a truncation.

    Args:
        text: Truncated model response.

    Returns:
        Tuple of (last complete label, dict of complete labels).
    """
    clean = text.strip()
    if clean.startswith("```json"):
        clean = clean[7:].strip()
    if clean.endswith("```"):
        clean = clean[:-3].strip()
    start = re.search(r'"labels"\s*:\s*\{', clean)
    if not start:
        return None, {}

    content = clean[start.end():]
    entries, last_key = {}, None
    depth, in_string, escaped = 1, False, False
    for i, char in enumerate(content):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 1:
                chunk = content[:i + 1]
                if not chunk.strip().endswith(","):
                    chunk = chunk.rstrip() + ","
                try:
                    parsed = json.loads(('{"labels": {' + chunk + "}}").replace(",}", "}"))
                except json.JSONDecodeError:
                    continue
                if parsed.get("labels"):
                    entries = parsed["labels"]
                    last_key = list(entries)[-1]
    return last_key, entries


def merge_labels(responses):
    """Merge several parsed responses into one, later labels winning.

    Args:
        responses: Parsed responses, each with a `labels` dict.

    Returns:
        A single `{"labels": ...}` dict.
    """
    merged = {}
    for response in responses:
        if isinstance(response, dict) and "labels" in response:
            merged.update(response["labels"])
    return {"labels": merged}


def to_dataframe(parsed):
    """Convert parsed labels to a table with one row per label.

    Args:
        parsed: Output of `LabelParser.parse`.

    Returns:
        DataFrame with lower-cased text and missing values as None.
    """
    labels = parsed.get("labels", parsed)
    df = pd.DataFrame([{"text": text, **fields} for text, fields in labels.items()])
    df = df.apply(lambda col: col.str.lower() if col.dtype == "object" else col)
    return df.replace(["na", "NA", "None", ""], None)
