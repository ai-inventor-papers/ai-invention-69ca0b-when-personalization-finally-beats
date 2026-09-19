#!/usr/bin/env python3
"""Canonical entry point: assemble the 25-catalog cold-start screen corpus.

Reads the standardized catalogs in processed/ (written by generator.py for the
19 synthetic latent-intent families and process_real.py for the 6 real
restricted subsets), validates each one, and emits out/full_data_out.json (the
full corpus), out/mini_data_out.json (3 catalogs) and out/preview_data_out.json
(10 catalogs, strings truncated) in the exp_sel_data_out shape:

    {metadata, datasets:[{dataset:<catalog_id>, examples:[{input, output,
      metadata_fold, metadata_family_params, metadata_origin,
      metadata_source_name, metadata_partition_meta}]}]}

Each catalog is ONE raw unit (items + user_logs + heldout + forced-choice
probes) carried as examples[0].input, group by group. RAW data only: no
metrics, no fitted models, no derived statistics.

Usage:
    uv run data.py          # rebuild out/{full,mini,preview}_data_out.json
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
MINI_N = 3
PREVIEW_N = 10
TRUNC = 200


def validate_catalog(c: dict) -> list[str]:
    problems = []
    items = c.get("items", [])
    logs = c.get("user_logs", [])
    probes = c.get("forced_choice_probes", [])
    item_ids = {it["item_id"] for it in items}
    if len(items) > MAX_ITEMS:
        problems.append("items>MAX")
    if len(item_ids) != len(items):
        problems.append("duplicate item_id")
    if not logs:
        problems.append("no user_logs")
    if c.get("fold") not in ("screen", "confirm"):
        problems.append("bad fold")
    if c.get("origin") not in ("real", "synthetic"):
        problems.append("bad origin")
    for it in items:
        if not (0 <= it.get("price_band", -1) <= 4):
            problems.append("bad price_band")
    for u in logs:
        h, ts, ho = u.get("ordered_history", []), u.get("timestamps", []), u.get("heldout_item")
        if not h or len(h) != len(ts) or ts != sorted(ts):
            problems.append("bad history/timestamps")
        if ho not in item_ids:
            problems.append("heldout not in items")
        if any(x not in item_ids for x in h):
            problems.append("history item not in items")
    for p in probes:
        if not all(k in p for k in ("user_id", "item_a", "item_b", "choice")):
            problems.append("malformed probe")
    return problems


def truncate(o, n: int = TRUNC):
    if isinstance(o, dict):
        return {k: truncate(v, n) for k, v in o.items()}
    if isinstance(o, list):
        return [truncate(x, n) for x in o]
    if isinstance(o, str):
        return o if len(o) <= n else o[:n] + "..."
    return o


def example_for(c: dict) -> dict:
    return {
        "input": json.dumps({k: v for k, v in c.items() if k != "family_params"},
                            separators=(",", ":")),
        "output": c["catalog_id"],
        "metadata_fold": c["fold"],
        "metadata_origin": c["origin"],
        "metadata_source_name": c["source_name"],
        "metadata_family_params": json.dumps(c["family_params"], separators=(",", ":")),
        "metadata_partition_meta": json.dumps(c["partition_meta"], separators=(",", ":")),
    }


def build() -> None:
    OUT.mkdir(exist_ok=True)
    catalogs = [json.loads(f.read_text()) for f in sorted(PROC.glob("*.json"))]
    catalogs.sort(key=lambda c: (c["origin"] != "real", c["catalog_id"]))

    bad = {c["catalog_id"]: p for c in catalogs if (p := validate_catalog(c))}
    if bad:
        logger.error(f"Validation failed: {bad}")
        sys.exit(1)

    real = [c for c in catalogs if c["origin"] == "real"]
    synth = [c for c in catalogs if c["origin"] == "synthetic"]
    keep = real + synth[: max(0, TARGET - len(real))]
    if len(keep) != TARGET:
        logger.warning(f"assembled {len(keep)} catalogs (target {TARGET})")

    folds = {c["catalog_id"]: c["fold"] for c in keep}
    n_screen = sum(1 for v in folds.values() if v == "screen")
    meta = {
        "corpus": "small_ecommerce_catalog_screen",
        "description": "25 small catalogs for cold-start heuristic screening "
                       "(shared evidence, one common schema). RAW only.",
        "schema": "catalog_schema.json",
        "n_catalogs": len(keep),
        "n_real": len(real),
        "n_synthetic": len(keep) - len(real),
        "fold_counts": {"screen": n_screen, "confirm": len(keep) - n_screen},
        "k_grid": [0, 1, 2, 3, 5, 8],
    }

    datasets_full = [{"dataset": c["catalog_id"], "examples": [example_for(c)]} for c in keep]

    def emit(name, ds):
        (OUT / name).write_text(json.dumps({"metadata": meta, "datasets": ds}))

    emit("full_data_out.json", datasets_full)
    emit("mini_data_out.json", datasets_full[:MINI_N])
    preview_ds = [
        {"dataset": d["dataset"], "examples": [truncate(e) for e in d["examples"]]}
        for d in datasets_full[:PREVIEW_N]
    ]
    emit("preview_data_out.json", preview_ds)
    logger.info(f"wrote full({len(datasets_full)}) mini({MINI_N}) preview({PREVIEW_N}) "
                f"| screen={n_screen} confirm={len(keep)-n_screen}")


if __name__ == "__main__":
    build()