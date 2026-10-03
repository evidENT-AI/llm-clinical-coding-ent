#!/usr/bin/env python3
"""Code one inpatient episode with an LLM.

Steps: pre-prompt, worked example, in-context learning on an example episode,
extraction of diagnoses and procedures, SNOMED CT grounding, mapping to
ICD-10 and OPCS-4 (Stage 1), then scoping to the admission's codes (Stage 2).
"""

import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path

import openai
import pandas as pd

from pipeline import prompts
from pipeline.classification import ClassificationMapper
from pipeline.config import (CLASSIFICATION_LOOKUP, EXAMPLE_ANNOTATION_PATH,
                             EXAMPLE_NOTE_PATH, LOCAL_API_KEY, LOCAL_BASE_URL,
                             MODELS, OPENAI_API_KEY, SEED, TEMPERATURE)
from pipeline.parser import merge_labels, to_dataframe
from pipeline.snomed import SnomedSearcher, safe_name

logger = logging.getLogger(__name__)

NO_CODE = {"nan", "na", "none", ""}


def make_client(settings):
    """Create an OpenAI-compatible client for a model's provider.

    Args:
        settings: Entry from `config.MODELS`.

    Returns:
        An `openai.OpenAI` or `openai.AzureOpenAI` client.
    """
    provider = settings["provider"]
    if provider == "local":
        return openai.OpenAI(api_key=LOCAL_API_KEY,
                             base_url=settings.get("base_url", LOCAL_BASE_URL))
    if provider == "azure":
        return openai.AzureOpenAI(api_version=settings["api_version"],
                                  azure_endpoint=settings["azure_endpoint"],
                                  api_key=settings["azure_api_key"],
                                  max_retries=8)
    return openai.OpenAI(api_key=OPENAI_API_KEY)


def format_episode(bundle):
    """Join an episode's notes in date order, each headed by date and class.

    The anchor note is not marked, so the model must infer the admission's
    focus from the notes themselves.

    Args:
        bundle: Episode bundle with a `notes` list.

    Returns:
        The text given to the model.
    """
    sections = ["Clinical notes for this admission:"]
    for i, note in enumerate(bundle["notes"], 1):
        sections.append(f"--- Note {i} | {note.get('note_date', '')} | "
                        f"{note.get('note_class', '')} ---\n{note.get('text', '') or ''}")
    return "\n\n".join(sections)


def load_mapper(path=CLASSIFICATION_LOOKUP):
    """Load the SNOMED CT to ICD-10/OPCS-4 lookup, or None if not built."""
    try:
        return ClassificationMapper(path)
    except FileNotFoundError:
        logger.warning("No classification lookup at %s; codes will be empty", path)
        return None


class EpisodeLabeller:
    """Run the coding pipeline for one model over individual episodes.

    Attributes:
        key: Model key in `config.MODELS`.
        settings: Model settings.
        model: Model name sent to the API.
    """

    def __init__(self, key, parser, searcher=None, mapper=None,
                 example_note=EXAMPLE_NOTE_PATH, example_annotation=EXAMPLE_ANNOTATION_PATH):
        """Set up the client and shared resources.

        Args:
            key: Model key in `config.MODELS`.
            parser: A `LabelParser`.
            searcher: A `SnomedSearcher`; one is created if omitted.
            mapper: A `ClassificationMapper`, or None to skip mapping.
            example_note: Episode bundle used for in-context learning.
            example_annotation: Gold annotation for the example episode.
        """
        self.key = key
        self.settings = MODELS[key]
        self.model = self.settings["model"]
        self.client = make_client(self.settings)
        self.parser = parser
        self.searcher = searcher or SnomedSearcher()
        self.mapper = mapper
        self.example_note = Path(example_note)
        self.example_annotation = Path(example_annotation)

    def annotate(self, episode_path, output_dir):
        """Code one episode and write all outputs to `output_dir/<episode>`.

        Args:
            episode_path: Episode bundle JSON.
            output_dir: Directory for this model's run.

        Returns:
            Run metadata, including token use, time and estimated cost.
        """
        episode_path = Path(episode_path)
        self.episode_id = episode_path.stem
        self.out = Path(output_dir) / self.episode_id
        self.out.mkdir(parents=True, exist_ok=True)
        self.messages = []
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "n_llm_calls": 0}
        self._n_responses = 0

        handler = logging.FileHandler(self.out / "run.log")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logging.getLogger("pipeline").addHandler(handler)
        start = time.monotonic()
        try:
            bundle = json.loads(episode_path.read_text(encoding="utf-8"))
            text = format_episode(bundle)
            _write_json(self.out / "episode.json", bundle)
            (self.out / "episode.txt").write_text(text, encoding="utf-8")

            self.messages = [{"role": "system", "content": prompts.PREPROMPT}]
            self.prompt(prompts.EXAMPLE_PROMPT)
            self.learn_from_example()
            labels = self.extract(text)
            labels = self.ground_snomed(labels)
            candidates = self.add_classification(labels)
            self.scope(candidates, text)
            _write_json(self.out / "conversation_log.json", self.messages)
            return self.write_metadata(time.monotonic() - start, bundle, text)
        finally:
            logging.getLogger("pipeline").removeHandler(handler)
            handler.close()

    def prompt(self, message, parse=False):
        """Send a user message in the current conversation.

        Args:
            message: Prompt text.
            parse: Parse the reply as labels, recovering truncated JSON.

        Returns:
            The reply text, or parsed labels if `parse` is set.
        """
        self.messages.append({"role": "user", "content": message})
        settings = self.settings
        request = {
            "model": self.model,
            "messages": self.messages,
            settings.get("token_param", "max_tokens"): settings["max_tokens"],
        }
        if settings["provider"] == "azure":
            request["model"] = settings["azure_deployment"] or self.model
        if settings.get("supports_temperature", True):
            request["temperature"] = TEMPERATURE
        if settings.get("supports_seed", True):
            request["seed"] = SEED
        if settings.get("reasoning_effort"):
            request["reasoning_effort"] = settings["reasoning_effort"]

        response = self.client.chat.completions.create(**request)
        reply = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": reply})
        usage = getattr(response, "usage", None)
        if usage is not None:
            self.usage["prompt_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
            self.usage["completion_tokens"] += getattr(usage, "completion_tokens", 0) or 0
        self.usage["n_llm_calls"] += 1
        self._save_response(reply)

        if not parse:
            return reply
        try:
            return self.parser.parse(reply)
        except Exception as error:
            logger.warning("Invalid JSON (%s); attempting recovery", error)
            return self._recover_json(reply)

    def _save_response(self, reply):
        self._n_responses += 1
        folder = self.out / "raw_responses"
        folder.mkdir(exist_ok=True)
        (folder / f"{self._n_responses:03d}.txt").write_text(reply or "", encoding="utf-8")

    def _recover_json(self, reply, attempts=3):
        """Ask the model to continue a truncated reply, else to reformat it."""
        try:
            partial = self.parser.parse(reply, allow_partial=True)
            if partial.get("_partial"):
                responses = [{"labels": partial["labels"]}]
                last_key = partial["_last_key"]
                for attempt in range(attempts):
                    try:
                        request = prompts.continuation_prompt(len(partial["labels"]), last_key)
                        more = self.parser.parse(self.prompt(request), allow_partial=True)
                    except Exception as error:
                        logger.error("Continuation %d failed: %s", attempt + 1, error)
                        if attempt == attempts - 1:
                            break
                        continue
                    if not more.get("labels"):
                        break
                    responses.append(more)
                    if not more.get("_partial"):
                        break
                    partial, last_key = more, more["_last_key"]
                return merge_labels(responses)
        except Exception as error:
            logger.error("Partial parse failed: %s", error)
        try:
            return self.parser.parse(self.prompt(prompts.json_retry_prompt(reply)))
        except Exception as error:
            raise ValueError("Failed to obtain valid JSON") from error

    def _fresh_prompt(self, message):
        """Prompt in an empty context, leaving the main conversation intact."""
        saved, self.messages = self.messages, []
        try:
            return self.prompt(message)
        finally:
            self.messages = saved

    def learn_from_example(self):
        """Annotate the example episode, then show the model the gold annotation."""
        example = json.loads(self.example_note.read_text(encoding="utf-8"))
        attempt = self.prompt(prompts.extraction_prompt(format_episode(example)), parse=True)
        _write_json(self.out / "zero_shot_extraction.json", attempt or {"labels": {}})
        self.prompt(prompts.feedback_prompt(self.example_annotation))

    def extract(self, text):
        """Extract diagnoses and procedures from the episode.

        Args:
            text: Formatted episode notes.

        Returns:
            DataFrame with one row per label.
        """
        extraction = self.prompt(prompts.extraction_prompt(text), parse=True)
        _write_json(self.out / "initial_extraction.json", extraction or {"labels": {}})
        labels = to_dataframe(extraction)
        labels.to_csv(self.out / "initial_output.csv", index=False)
        return labels

    def ground_snomed(self, labels, max_rounds=3):
        """Assign a SNOMED CT concept to every label.

        Args:
            labels: Extracted labels.
            max_rounds: Maximum search rounds per label.

        Returns:
            Labels with the `snomed_ct` column replaced.
        """
        folder = self.out / "snowstorm_results"
        self.searcher.search_labels(labels, folder)
        codes = [self._resolve_snomed(labels.iloc[i], i, folder, max_rounds)
                 for i in range(len(labels))]
        labels = labels.copy()
        labels.loc[:, "snomed_ct"] = pd.Series(codes, index=labels.index)
        labels.to_csv(self.out / "final_output.csv", index=False)
        return labels

    def _resolve_snomed(self, row, idx, folder, max_rounds):
        """Choose a concept for one label with a bounded search loop.

        With no candidates the model proposes a simpler term; with unsuitable
        candidates it may reply `SEARCH: <term>`. Terms are never repeated.

        Returns:
            Concept id, or None if unresolved.
        """
        term = str(row["text"])
        context = str(row.get("context", "") or "")
        qualifier = str(row.get("qualifier", "") or "")
        tried = {term.strip().lower()}
        path = folder / f"{idx}_{safe_name(term)}.json"
        try:
            results = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        except (json.JSONDecodeError, OSError):
            results = {}

        for round_ in range(max_rounds):
            items = (results or {}).get("items", [])
            if items:
                options = "\n".join(f"- ID: {it['id']}, Term: {it['term']}" for it in items)
                reply = (self._fresh_prompt(prompts.SNOMED_SELECTION_PROMPT.format(
                    text=term, context=context, qualifier=qualifier,
                    snomed_options=options)) or "").strip()
                if reply.isdigit():
                    return reply
                search = re.match(r"^\s*SEARCH:\s*(.+)$", reply, re.IGNORECASE)
                if search and round_ < max_rounds - 1:
                    new_term = search.group(1).strip().strip('"').strip()
                    if new_term and new_term.lower() not in tried:
                        tried.add(new_term.lower())
                        term, results = new_term, self.searcher.candidates(new_term)
                        continue
                return None
            if round_ >= max_rounds - 1:
                return None
            new_term = (self._fresh_prompt(prompts.SNOMED_REFINEMENT_PROMPT.format(
                text=term, context=context)) or "").strip().strip('"').strip()
            if not new_term or new_term.lower() in tried:
                return None
            tried.add(new_term.lower())
            term, results = new_term, self.searcher.candidates(new_term)
        return None

    def add_classification(self, labels):
        """Map each label's concept to ICD-10 or OPCS-4 and save Stage 1.

        Args:
            labels: Labels with SNOMED CT concepts.

        Returns:
            Stage 1 table with `icd10`, `opcs4` and candidate columns.
        """
        icd10, opcs4, alternatives = [], [], []
        for _, row in labels.iterrows():
            qualifier = str(row.get("qualifier", "")).strip().lower()
            kind = {"diagnoses": "diagnosis", "procedures": "procedure"}.get(qualifier)
            sctid = row.get("snomed_ct")
            result = None
            if self.mapper and kind and sctid and str(sctid).lower() not in NO_CODE:
                try:
                    result = self.mapper.map(str(sctid), kind)
                except Exception as error:
                    logger.warning("Mapping failed for %s: %s", sctid, error)
            icd10.append(result["code"] if result and kind == "diagnosis" else None)
            opcs4.append(result["code"] if result and kind == "procedure" else None)
            alternatives.append(";".join(result["all_group1_targets"]) if result else None)

        stage1 = labels.copy()
        stage1["icd10"] = icd10
        stage1["opcs4"] = opcs4
        stage1["classification_candidates"] = alternatives
        self._save_stage(stage1, "stage1_output", "wide_extraction")
        return stage1

    def scope(self, stage1, text):
        """Select the admission's principal and secondary codes (Stage 2).

        The model picks by id from the Stage 1 candidates, so selected rows keep
        their codes: one principal and up to two secondary diagnoses, and the
        same for procedures.

        Args:
            stage1: Stage 1 table.
            text: Formatted episode notes.
        """
        if stage1 is None or len(stage1) == 0:
            _write_json(self.out / "stage2_output.json",
                        {"episode_id": self.episode_id, "labels": []})
            return
        stage1 = stage1.reset_index(drop=True)

        lines = []
        for i, row in stage1.iterrows():
            code = _code_for(row)
            code = "" if code is None or str(code).lower() == "nan" else f" | {code}"
            lines.append(f"  [{i}] {row.get('qualifier')} | {str(row.get('text'))[:80]}{code}")
        reply = self._fresh_prompt(prompts.scoping_prompt("\n".join(lines), text))

        try:
            found = re.search(r"\{.*\}", reply, re.DOTALL)
            selection = json.loads(found.group(0)) if found else {}
        except (json.JSONDecodeError, AttributeError, TypeError) as error:
            logger.error("Could not parse scoping selection: %s", error)
            selection = {}

        chosen = (_pick(stage1, selection, "diagnoses")
                  + _pick(stage1, selection, "procedures"))
        stage2 = pd.DataFrame(chosen) if chosen else stage1.iloc[0:0].copy()
        self._save_stage(stage2, "stage2_output", "scoped")

    def _save_stage(self, table, name, stage):
        table.to_csv(self.out / f"{name}.csv", index=False)
        labels = table.where(pd.notna(table), None).to_dict(orient="records") if len(table) else []
        _write_json(self.out / f"{name}.json",
                    {"episode_id": self.episode_id, "stage": stage, "labels": labels})

    def write_metadata(self, elapsed, bundle, text):
        """Write run settings, token use, time and estimated cost."""
        settings = self.settings
        in_price, out_price = settings["price_per_mtok"]
        tokens_in, tokens_out = self.usage["prompt_tokens"], self.usage["completion_tokens"]
        metadata = {
            "episode_id": self.episode_id,
            "model": self.model,
            "model_key": self.key,
            "provider": settings["provider"],
            "temperature": TEMPERATURE if settings.get("supports_temperature", True) else None,
            "seed": SEED if settings.get("supports_seed", True) else None,
            "reasoning_effort": settings.get("reasoning_effort"),
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": round(elapsed, 2),
            **self.usage,
            "total_tokens": tokens_in + tokens_out,
            "estimated_cost_usd": round(tokens_in / 1e6 * in_price
                                        + tokens_out / 1e6 * out_price, 6),
            "n_notes": len(bundle.get("notes", [])),
            "n_input_chars": len(text),
        }
        _write_json(self.out / "run_metadata.json", metadata)
        return metadata


def _code_for(row):
    qualifier = str(row.get("qualifier", "")).strip().lower()
    if qualifier == "diagnoses":
        return row.get("icd10")
    if qualifier == "procedures":
        return row.get("opcs4")
    return None


def _normalise(value):
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


def _resolve(stage1, value, category):
    """Match a selection to a Stage 1 row by id, then code, then text."""
    if value is None:
        return None
    text = str(value).strip()
    rows = [(i, row) for i, row in stage1.iterrows()
            if str(row.get("qualifier", "")).strip().lower() == category]
    try:
        i = int(float(text))
        if 0 <= i < len(stage1) and any(j == i for j, _ in rows):
            return i
    except (ValueError, TypeError):
        pass
    code_col = "icd10" if category == "diagnoses" else "opcs4"
    if _normalise(text):
        for i, row in rows:
            if _normalise(row.get(code_col)) == _normalise(text):
                return i
    if text.lower():
        for i, row in rows:
            if text.lower() in str(row.get("text", "")).lower():
                return i
    return None


def _pick(stage1, selection, category):
    """Return the principal and up to two secondary rows for a category."""
    block = selection.get(category, {}) if isinstance(selection, dict) else {}
    refs = [(block.get("principal"), "primary")]
    refs += [(ref, "secondary") for ref in (block.get("secondary") or [])[:2]]
    chosen, seen = [], set()
    for ref, position in refs:
        idx = _resolve(stage1, ref, category)
        if idx is not None and idx not in seen:
            seen.add(idx)
            row = stage1.iloc[idx].to_dict()
            row["primary_secondary"] = position
            chosen.append(row)
    return chosen


def _write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
