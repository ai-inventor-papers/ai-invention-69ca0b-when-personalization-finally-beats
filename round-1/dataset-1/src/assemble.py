#!/usr/bin/env python3
"""Assemble the 25-catalog corpus into the shared exp_sel_data_out JSON.

Reads every processed/<catalog_id>.json, validates it against catalog_schema.json,
then builds out/data_out.json in the exp_sel_data_out shape:
  {metadata, datasets:[{dataset:<catalog_id>, examples:[{input, output, metadata_*...}]}]}

Each catalog is ONE dataset entry carrying its RAW catalog JSON in examples[0].input
and catalog_id in output; fold / family_params / origin / partition_meta are carried
as metadata_* per-example fields so the downstream experiment can filter by fold.

Target exactly target_num_datasets catalogs (25). Real catalogs are reserved for
'confirm'; synthetic families are split ~half screen / half confirm.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from loguru import logger

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")

HERE = Path(__file__).resolve().parent
PROC = HERE / "processed"
OUT = HERE / "out"

TARGET = 25
MAX_ITEMS = 1000


def validate_catalog(c: dict, echo: bool = True) -> list[str]:
    """Sanity checks; returns list of problems (empty == ok)."""
    problems = []
    cid = c.get("catalog_id", "?")
    items = c.get("items", [])
    logs = c.get("user_logs", [])
    probes = c.get("forced_choice_probes", [])
    if len(items) > MAX_ITEMS:
        problems.append(f"items> {MAX_ITEMS}")
    item_ids = {it["item_id"] for it in items}
    if len(item_ids) != len(items):
        problems.append("duplicate item_id")
    if not logs:
        problems.append("no user_logs")
    for it in items:
        if it["item_id"] not in item_ids:
            pass
        if "price_band" not in it or not (0 <= it["price_band"] <= 4):
            problems.append("bad price_band")
        if not it.get("attrs"):
            problems.append("item missing attrs")
    for u in logs:
        h = u.get("ordered_history", [])
        ts = u.get("timestamps", [])
        ho = u.get("heldout_item")
        if not h:
            problems.append("empty history")
            continue
        if len(h) != len(ts):
            problems.append("history/timestamp len mismatch")
        if ts != sorted(ts):
            problems.append("timestamps not sorted")
        if ho not in item_ids:
            problems.append(f"heldout not in items ({ho})")
        bad = [x for x in h if x not in item_ids]
        if bad:
            problems.append("history item not in items")
    if c.get("fold") not in ("screen", "confirm"):
        problems.append("bad fold")
    if c.get("origin") not in ("real", "synthetic"):
        problems.append("bad origin")
    for p in probes:
        if "item_a" not in p or "item_b" not in p or "choice" not in p:
            problems.append("malformed probe")
    if echo and problems:
        logger.warning(f"{cid} PROBLEMS: {problems}")
    return problems


def main() -> None:
    OUT.mkdir(exist_ok=True)
    catalogs = []
    for fp in sorted(PROC.glob("*.json")):
        with fp.open() as fh:
            catalogs.append(json.load(fh))

    # --- order: real first (they are the scarce confirm resource), then synth ---
    catalogs.sort(key=lambda c: (c["origin"] != "real", c["catalog_id"]))

    all_problems = {}
    for c in catalogs:
        pr = validate_catalog(c)
        if pr:
            all_problems[c["catalog_id"]] = pr

    if all_problems:
        logger.error(f"Validation failed for {len(all_problems)} catalogs: {all_problems}")
        sys.exit(1)

    # target = 25: keep real (all) + enough synthetic to reach TARGET
    real = [c for c in catalogs if c["origin"] == "real"]
    synth = [c for c in catalogs if c["origin"] == "synthetic"]
    keep = real + synth[:max(0, TARGET - len(real))]
    if len(keep) != TARGET:
        logger.warning(f"assembled {len(keep)} catalogs (target {TARGET})")

    folds = {}
    for c in keep:
        folds[c["catalog_id"]] = c["fold"]
    n_screen = sum(1 for v in folds.values() if v == "screen")
    n_confirm = sum(1 for v in folds.values() if v == "confirm")
    logger.info(f"Assembling {len(keep)} catalogs: screen={n_screen} confirm={n_confirm}")

    datasets = []
    for c in keep:
        example = {
            "input": json.dumps(
                {k: v for k, v in c.items() if k != "family_params"}, separators=(",", ":")
            ),
            "output": c["catalog_id"],
            "metadata_fold": c["fold"],
            "metadata_origin": c["origin"],
            "metadata_source_name": c["source_name"],
            "metadata_family_params": json.dumps(c["family_params"], separators=(",", ":")),
            "metadata_partition_meta": json.dumps(c["partition_meta"], separators=(",", ":")),
        }
        datasets.append({"dataset": c["catalog_id"], "examples": [example]})

    data_out = {
        "metadata": {
            "corpus": "small_ecommerce_catalog_screen",
            "description": (
                "Shared evidence corpus for cold-start heuristic screening. "
                "25 small catalogs (real restricted subsets + synthetic latent-intent "
                "families), all on one common schema. RAW catalogs only: no metrics, "
                "no fitted models. Real catalogs fold=confirm; synthetic split "
                "screen/confirm by sha256(catalog_id)."
            ),
            "schema": "catalog_schema.json",
            "n_catalogs": len(keep),
            "n_real": len(real),
            "n_synthetic": len(keep) - len(real),
            "fold_counts": {"screen": n_screen, "confirm": n_confirm},
            "k_grid": [0, 1, 2, 3, 5, 8],
        },
        "datasets": datasets,
    }
    out_path = OUT / "data_out.json"
    out_path.write_text(json.dumps(data_out))
    logger.info(f"Wrote {out_path} ({out_path.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()