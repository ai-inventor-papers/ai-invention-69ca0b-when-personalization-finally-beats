#!/usr/bin/env python3
"""Assemble out/data_out.json in the exp_sel_data_out shape.

Because NO additional real catalog passed the acceptance bar, the aggregate
carries the machine-readable REJECTION LOG as its dataset rows (each rejected
candidate = one example: input = candidate probe, output = "REJECTED", plus
metadata_source/criterion_failed/counts), and top-level metadata records
n_accepted=0 with the explicit honest-outcome finding. This keeps the artifact
schema-valid and lets any downstream confirm-round discovery see the truthful
empty result rather than a padded catalog set.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
from loguru import logger

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"

@logger.catch(reraise=True)
def main() -> None:
    OUT.mkdir(exist_ok=True)
    rej = json.loads((HERE / "rejections.json").read_text())
    examples = []
    for r in rej["rejections"]:
        examples.append({
            "input": json.dumps({"source": r["source"], "attempt": r["attempt"]}, separators=(",", ":")),
            "output": "REJECTED",
            "metadata_source": r["source"],
            "metadata_attempt": r["attempt"],
            "metadata_criterion_failed": r["criterion_failed"],
            "metadata_verdict": r["verdict"],
            "metadata_counts": json.dumps(r["counts"], separators=(",", ":")),
        })
    data_out = {
        "metadata": {
            "corpus": "small_ecommerce_catalog_screen_iter2",
            "schema": "catalog_schema.json (iter-1 common schema)",
            "iteration": 2,
            "artifact": "gen_art_dataset_1 (real-catalog confirm-evidence widening)",
            "outcome": "no_additional_real_catalog_accepted",
            "n_accepted": 0,
            "n_rejection_records": len(examples),
            "k_grid": [0, 1, 2, 3, 5, 8],
            "finding": ("The public freely-downloadable landscape of genuinely small "
                        "(<=1000-item) per-user purchase logs with item category+price is "
                        "effectively empty in this environment: Retailrocket's most complete "
                        "free mirror yields at most 96 users with >=4 purchases (best "
                        "configuration) and has no price; Dunnhumby is gated behind login/form; "
                        "the boutique/marketplace sweep produced no qualifying candidate. "
                        "The acceptance bar was not lowered. This empty result is itself the "
                        "corpus-contribution finding; see search_log.md and rejections.json."),
        },
        "datasets": [
            {"dataset": "acquisition_rejection_log", "examples": examples},
        ],
    }
    (OUT / "data_out.json").write_text(json.dumps(data_out, indent=2))
    logger.info(f"wrote out/data_out.json ({len(examples)} rejection rows, n_accepted=0)")

if __name__ == "__main__":
    main()