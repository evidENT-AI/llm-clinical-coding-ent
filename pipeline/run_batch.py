#!/usr/bin/env python3
"""Code every episode in a directory with one model.

Outputs go to `<results>/<run id>/<model>/<episode>/`, with a run manifest.
Example:

    python -m pipeline.run_batch --model GPT4O --notes-dir data/synthetic/episodes
"""

import argparse
import json
import logging
import subprocess
import time
from datetime import datetime
from pathlib import Path

from pipeline.config import MODELS, NOTES_DIR, REPO_ROOT, RESULTS_DIR
from pipeline.labeller import EpisodeLabeller, load_mapper
from pipeline.parser import LabelParser

logger = logging.getLogger("pipeline")


def git_commit():
    """Return the short commit hash of this repository, or "unknown"."""
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--notes-dir", type=Path, default=NOTES_DIR)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--exclude", nargs="*", default=[],
                        help="episode ids to skip, e.g. the in-context example")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logger.setLevel(logging.INFO)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    model = MODELS[args.model]["model"]
    out = args.results_dir / run_id / model
    out.mkdir(parents=True, exist_ok=True)
    episodes = [p for p in sorted(args.notes_dir.glob("*.json")) if p.stem not in args.exclude]

    manifest = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "model_key": args.model,
        "git_commit": git_commit(),
        "notes_dir": str(args.notes_dir),
        "n_episodes": len(episodes),
        "excluded": args.exclude,
    }
    (out.parent / "run_manifest.json").write_text(json.dumps(manifest, indent=2))

    labeller = EpisodeLabeller(args.model, LabelParser(), mapper=load_mapper())
    ok = failed = 0
    start = time.time()
    for i, path in enumerate(episodes, 1):
        tick = time.time()
        try:
            labeller.annotate(path, out)
            ok += 1
            status = "ok"
        except Exception as error:
            failed += 1
            status = f"error: {error}"
            logger.exception("Episode %s failed", path.stem)
        logger.info("[%d/%d] %s %s (%.0fs)", i, len(episodes), path.stem, status,
                    time.time() - tick)
    logger.info("Done: %d ok, %d failed in %.1f min -> %s", ok, failed,
                (time.time() - start) / 60, out)


if __name__ == "__main__":
    main()
