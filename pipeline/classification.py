#!/usr/bin/env python3
"""Map SNOMED CT concepts to ICD-10 (diagnoses) and OPCS-4 (procedures).

Uses the complex map reference sets in the NHS SNOMED CT UK Monolith release.
The release is licensed through NHS TRUD, so the lookup is built locally:

    python -m pipeline.classification --build --release uk_sct2mo_42.4.0_*.zip
    python -m pipeline.classification --map 195677001 --kind diagnosis
"""

import argparse
import csv
import io
import json
import re
import zipfile
from pathlib import Path

from pipeline.config import CLASSIFICATION_LOOKUP

ICD10_REFSET = "999002271000000101"   # ICD-10 5th edition UK complex map
OPCS4_REFSET = "1891651000000103"     # OPCS-4.11 complex map
REFSETS = {"diagnosis": ICD10_REFSET, "procedure": OPCS4_REFSET}
UNCONDITIONAL = {"", "OTHERWISE TRUE", "TRUE", "ALWAYS"}


def read_map_rows(release):
    """Yield active ICD-10 and OPCS-4 map rows from an RF2 release.

    Args:
        release: Monolith release zip, or an extracted RF2 directory.

    Yields:
        Dicts with refset, sctid, group, priority, rule, advice and target.
    """
    def rows(handle):
        reader = csv.reader(io.TextIOWrapper(handle, encoding="utf-8"), delimiter="\t")
        next(reader, None)
        for r in reader:
            if len(r) >= 11 and r[2] == "1" and r[4] in (ICD10_REFSET, OPCS4_REFSET):
                yield {"refset": r[4], "sctid": r[5], "group": int(r[6]),
                       "priority": int(r[7]), "rule": r[8].strip(),
                       "advice": r[9], "target": r[10]}

    release = Path(release)
    if release.suffix == ".zip":
        with zipfile.ZipFile(release) as z:
            for name in z.namelist():
                if "der2_iisssciRefset_ExtendedMap" in name and name.endswith(".txt"):
                    with z.open(name) as handle:
                        yield from rows(handle)
    else:
        for path in release.rglob("der2_iisssciRefset_ExtendedMap*.txt"):
            with open(path, "rb") as handle:
                yield from rows(handle)


def build_lookup(release, out):
    """Write a compact `{refset: {sctid: [candidates]}}` lookup as JSON.

    Args:
        release: Monolith release zip or RF2 directory.
        out: Output JSON path.
    """
    lookup = {ICD10_REFSET: {}, OPCS4_REFSET: {}}
    for row in read_map_rows(release):
        lookup[row["refset"]].setdefault(row["sctid"], []).append(
            {k: row[k] for k in ("group", "priority", "rule", "advice", "target")})
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(lookup), encoding="utf-8")
    print(f"ICD-10 concepts {len(lookup[ICD10_REFSET])}, "
          f"OPCS-4 concepts {len(lookup[OPCS4_REFSET])} -> {out}")


def dotted(code):
    """Return a code in dotted form, e.g. J039 -> J03.9."""
    code = re.sub(r"[^A-Z0-9]", "", str(code).upper())
    return f"{code[:3]}.{code[3:]}" if len(code) > 3 else code


def select_target(candidates):
    """Choose the primary map target for a concept.

    Takes map group 1, prefers unconditional rules (patient attributes such as
    age and sex are unknown), excludes codes flagged as needing an additional
    code (ICD-10 asterisk codes) unless nothing else remains, then takes the
    lowest priority.

    Args:
        candidates: Map rows for one concept.

    Returns:
        The chosen candidate, or None if the concept has no group 1 target.
    """
    group1 = [c for c in candidates if c["group"] == 1 and c["target"]]
    if not group1:
        return None
    unconditional = [c for c in group1
                     if c["rule"].upper() in UNCONDITIONAL
                     or c["rule"].upper().startswith("OTHERWISE")] or group1
    standalone = [c for c in unconditional
                  if "ADDITIONAL CODE MANDATORY" not in (c["advice"] or "").upper()]
    return min(standalone or unconditional, key=lambda c: c["priority"])


class ClassificationMapper:
    """Look up ICD-10 or OPCS-4 codes for SNOMED CT concepts."""

    def __init__(self, lookup_path=CLASSIFICATION_LOOKUP):
        """Load a lookup built by `build_lookup`.

        Args:
            lookup_path: Path to the lookup JSON.
        """
        data = json.loads(Path(lookup_path).read_text(encoding="utf-8"))
        self.lookup = {refset: data.get(refset, {}) for refset in REFSETS.values()}

    def map(self, sctid, kind):
        """Map one concept to its classification code.

        Args:
            sctid: SNOMED CT concept id.
            kind: "diagnosis" (ICD-10) or "procedure" (OPCS-4).

        Returns:
            Dict with `code` (None if unmapped) and all group 1 targets.
        """
        candidates = self.lookup[REFSETS[kind]].get(str(sctid), [])
        best = select_target(candidates)
        return {
            "sctid": str(sctid),
            "kind": kind,
            "code": dotted(best["target"]) if best else None,
            "advice": best["advice"] if best else None,
            "all_group1_targets": sorted({dotted(c["target"]) for c in candidates
                                          if c["group"] == 1 and c["target"]}),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--build", action="store_true", help="build the lookup")
    parser.add_argument("--release", type=Path, help="monolith zip or RF2 directory")
    parser.add_argument("--lookup", type=Path, default=CLASSIFICATION_LOOKUP)
    parser.add_argument("--map", dest="sctid", help="SNOMED CT concept id to map")
    parser.add_argument("--kind", choices=REFSETS, default="diagnosis")
    args = parser.parse_args()

    if args.build:
        if not args.release:
            parser.error("--build requires --release")
        build_lookup(args.release, args.lookup)
    if args.sctid:
        print(json.dumps(ClassificationMapper(args.lookup).map(args.sctid, args.kind), indent=2))


if __name__ == "__main__":
    main()
