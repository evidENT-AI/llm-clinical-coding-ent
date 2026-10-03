#!/usr/bin/env python3
"""Build one note bundle per annotated episode from an OMOP snapshot.

Each reference-clinician episode is anchored to the source note that contains
most of the clinician's verbatim annotation snippets (ties broken by the
nearest coder admission date). All of the patient's notes from `pre_days`
before to `post_days` after the anchor are bundled in date order. The output
contains de-identified patient text and must stay in the secure environment.

    python extraction/build_episode_notes.py --omop <snapshot>/public/omop \
        --linkage linkage.csv --annotations reference.xlsx \
        --coder-admissions coder_admissions.csv --out llm_notes
"""

import argparse
import datetime as dt
import json
import re
from pathlib import Path

import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds

MIN_SNIPPET = 12
PROBE_LENGTH = 60


def norm_id(value):
    """Return a hospital number as digits only, without zero padding."""
    return re.sub(r"[^0-9]", "", re.sub(r"\.0$", "", str(value).strip())).lstrip("0")


def canon(text):
    """Return lower-case text with punctuation removed, for snippet matching."""
    return re.sub(r"[^a-z0-9 ]", " ", re.sub(r"\s+", " ", str(text).lower())).strip()


def parse_date(value):
    """Parse a datetime, an Excel serial number or a day-first date string."""
    if pd.isna(value):
        return None
    if isinstance(value, (pd.Timestamp, dt.datetime, dt.date)):
        return pd.Timestamp(value).date()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return (pd.Timestamp("1899-12-30") + pd.to_timedelta(int(value), unit="D")).date()
    parsed = pd.to_datetime(str(value).split(" ")[0], dayfirst=True, errors="coerce")
    return None if pd.isna(parsed) else parsed.date()


def swapped(day):
    """Return the day/month swap of an ambiguous date, else None."""
    if not isinstance(day, dt.date) or day.day > 12 or day.day == day.month:
        return None
    return dt.date(day.year, day.day, day.month)


def load_cohort(annotations, sheet, linkage):
    """Return one row per annotated episode with person id and snippets.

    Args:
        annotations: Reference clinician workbook.
        sheet: Worksheet name.
        linkage: CSV mapping LocalPatientIdentifier to OMOP person_id.

    Returns:
        DataFrame with raw_id, person_id and snippets.
    """
    link = pd.read_csv(linkage)
    person = dict(zip(link.LocalPatientIdentifier.map(norm_id), link.person_id))
    ref = pd.read_excel(annotations, sheet_name=sheet)
    ref = ref[ref.LocalPatientIdentifier.notna()].copy()
    ref["raw_id"] = ref.LocalPatientIdentifier.map(lambda x: re.sub(r"\.0$", "", str(x).strip()))
    ref["nid"] = ref.LocalPatientIdentifier.map(norm_id)

    episodes = []
    for (raw, nid), group in ref.groupby(["raw_id", "nid"]):
        snippets = [canon(row[col]) for _, row in group.iterrows()
                    for col in ("context", "text") if len(canon(row[col])) >= MIN_SNIPPET]
        episodes.append({"raw_id": raw, "person_id": person.get(nid), "snippets": snippets})
    return pd.DataFrame(episodes)


def load_notes(omop, persons):
    """Load the notes of the given people from the OMOP note table."""
    notes = ds.dataset(Path(omop) / "note", format="parquet").to_table(
        columns=["note_id", "person_id", "note_date", "note_class_concept_id", "note_text"],
        filter=pc.field("person_id").isin(persons)).to_pandas()
    notes["note_date"] = pd.to_datetime(notes.note_date).dt.date
    notes["canon"] = notes.note_text.fillna("").map(canon)
    classes = ds.dataset(Path(omop) / "concept", format="parquet").to_table(
        columns=["concept_id", "concept_name"],
        filter=pc.field("concept_id").isin(notes.note_class_concept_id.unique().tolist())
    ).to_pandas()
    notes["note_class"] = notes.note_class_concept_id.map(
        dict(zip(classes.concept_id, classes.concept_name)))
    return notes


def load_coder_admissions(path):
    """Return person_id -> sorted coder admission dates (Excel serials)."""
    coder = pd.read_csv(path, low_memory=False, usecols=["person_id", "AdmissionDate"])
    coder = coder.dropna(subset=["person_id"])
    coder["admission"] = (pd.Timestamp("1899-12-30") + pd.to_timedelta(
        pd.to_numeric(coder.AdmissionDate, errors="coerce"), unit="D")).dt.date
    coder = coder.dropna(subset=["admission"])
    return {int(pid): sorted(set(g.admission)) for pid, g in coder.groupby("person_id")}


def find_anchor(snippets, notes, coder_dates):
    """Return the note matching most snippets, nearest a coder admission.

    Returns:
        Tuple of (note row, snippets matched, snippets tried, days to the
        nearest coder admission or None), or None if nothing matches.
    """
    probes = [s[:PROBE_LENGTH] for s in snippets if len(s) >= MIN_SNIPPET]
    if notes.empty or not probes:
        return None
    scores = notes.canon.map(lambda text: sum(p in text for p in probes))
    if scores.max() == 0:
        return None
    top = notes[scores == scores.max()].copy()
    gap = None
    if coder_dates:
        top["gap"] = top.note_date.map(lambda d: min(abs((d - c).days) for c in coder_dates))
        top = top.sort_values(["gap", "note_date"])
        gap = int(top.iloc[0].gap)
    else:
        top = top.sort_values("note_date")
    return top.iloc[0], int(scores.max()), len(probes), gap


def build_bundle(episode, notes, coder_dates, args):
    """Return the note bundle for one episode, or None if no anchor is found."""
    hit = find_anchor(episode.snippets, notes, coder_dates)
    if hit is None:
        return None
    anchor, score, n_probes, gap = hit
    low = anchor.note_date - dt.timedelta(days=args.pre_days)
    high = anchor.note_date + dt.timedelta(days=args.post_days)
    window = notes[((notes.note_date >= low) & (notes.note_date <= high))
                   | (notes.note_id == anchor.note_id)]
    window = window.drop_duplicates("note_id").sort_values("note_date")
    episode_id = f"{re.sub(r'[^A-Za-z0-9]', '', episode.raw_id)}_{anchor.note_date:%Y%m%d}"
    return {
        "episode_id": episode_id,
        "anchor_note_id": int(anchor.note_id),
        "anchor_date": str(anchor.note_date),
        "anchor_snippets_matched": [score, n_probes],
        "coder_gap_days": gap,
        "anchor_suspect": gap is None or gap > args.suspect_days,
        "window": [str(low), str(high)],
        "n_notes": len(window),
        "notes": [{"note_id": int(n.note_id), "note_date": str(n.note_date),
                   "note_class": n.note_class, "is_anchor": bool(n.note_id == anchor.note_id),
                   "text": n.note_text if isinstance(n.note_text, str) else ""}
                  for n in window.itertuples()],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--omop", type=Path, required=True, help="OMOP parquet directory")
    parser.add_argument("--linkage", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--sheet", default="Data collection")
    parser.add_argument("--coder-admissions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pre-days", type=int, default=90)
    parser.add_argument("--post-days", type=int, default=30)
    parser.add_argument("--suspect-days", type=int, default=45,
                        help="flag anchors further than this from a coder admission")
    args = parser.parse_args()

    cohort = load_cohort(args.annotations, args.sheet, args.linkage)
    linked = cohort[cohort.person_id.notna()].copy()
    linked["person_id"] = linked.person_id.astype(int)
    notes = load_notes(args.omop, linked.person_id.unique().tolist())
    coder = load_coder_admissions(args.coder_admissions)

    folder = args.out / "episodes"
    folder.mkdir(parents=True, exist_ok=True)
    written = 0
    for episode in linked.itertuples():
        bundle = build_bundle(episode, notes[notes.person_id == episode.person_id],
                              coder.get(episode.person_id, []), args)
        if bundle is not None:
            (folder / f"{bundle['episode_id']}.json").write_text(
                json.dumps(bundle, indent=2), encoding="utf-8")
            written += 1
    print(f"{written} of {len(cohort)} episodes bundled "
          f"({len(cohort) - len(linked)} not linked to the OMOP snapshot)")


if __name__ == "__main__":
    main()
