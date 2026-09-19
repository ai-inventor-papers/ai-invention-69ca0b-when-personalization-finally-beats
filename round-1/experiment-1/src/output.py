#!/usr/bin/env python3
"""Serialization: method_out.json parts + aggregate CSVs + provenance.

The full output is streamed into parts of <= PART_SIZE_MB each (schema
exp_gen_sol_out; every part is a self-contained {"metadata":..., "datasets":
[...]} and validates on its own).  Per-example payload (additionalProperties
== false in the schema, so ONLY these keys appear):
  input  : str JSON {"catalog","user","k","history"}
  output : str next purchase item id
  predict_{h}     : str JSON array of top-50 ranked item ids (omitted when the
                    cell is not applicable, e.g. active_elic2 at k>0)
  metadata_catalog, metadata_user, metadata_k, metadata_fold (user's
  ground-truth half: 'screen'|'confirm'), metadata_catalog_fold (corpus
  phase), metadata_true, metadata_rank_{h} (int rank r, or '' for NA),
  metadata_headroom, metadata_sales_hhi, metadata_sales_norm_entropy,
  metadata_attr_entropy, metadata_mean_hist, metadata_mi_ceiling (per-k MC
  ceiling, synthetic only), metadata_supported (n_valid >= MIN_EXAMPLES),
  metadata_n_valid.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
from loguru import logger

from synthetic import K_VALUES, MIN_EXAMPLES
from heuristics import HEURISTIC_NAMES

PART_SIZE_MB = 85


def hkey(h: str) -> str:
    """Sanitize a heuristic name for use inside metadata_rank_*/predict_* keys:
    the exp_gen_sol_out schema restricts keys to [a-zA-Z0-9_] (no dots), so
    'recency_pop_0.1' -> 'recency_pop_0_1', 'lambda_hybrid_0.25' ->
    'lambda_hybrid_0_25'.  The unsanitized name is retained in
    metadata['heuristic_inventory'] and in the aggregates CSV."""
    return h.replace(".", "_")


def _ex_float(v) -> float:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    return float(v)


def build_examples(payload: dict, meta: dict) -> list[dict]:
    """Reconstruct schema-conformant example dicts from a per-catalog payload.

    Item ids may be re-encoded to short aliases when the original ids are
    long (pool catalogs); the alias -> original mapping is recorded in
    meta['id_mappings'][cat_id].
    """
    cat_id: str = payload["catalog"]
    arrs = payload["arrays"]
    diag = payload.get("diagnostics") or {}
    mi = diag.get("mi_ceiling") or {}
    headroom = _ex_float(diag.get("headroom"))
    hhi = _ex_float(diag.get("sales_hhi"))
    nent = _ex_float(diag.get("sales_norm_entropy"))
    aent = _ex_float(diag.get("attr_entropy"))
    mhist = _ex_float(diag.get("mean_history_length"))
    n_items = int(payload["n_items"])
    item_ids = meta["item_ids"][cat_id]
    user_ids = meta["user_ids"][cat_id]
    catalog_fold = payload.get("fold", "screen")
    screen_pos = meta["screen_pos"][cat_id]  # dict user_slot -> screen position
    id_map = (meta.get("id_mappings") or {}).get(cat_id) or {}

    def code(x: str) -> str:
        return id_map.get(x, x)

    examples: list[dict] = []
    na_rank = {h: "" for h in HEURISTIC_NAMES}

    for k in K_VALUES:
        sk = str(k)
        if f"{sk}/n" not in arrs:
            continue
        nv = int(arrs[f"{sk}/n"])
        if nv == 0:
            continue
        users = arrs[f"{sk}/user"]
        gt = arrs[f"{sk}/gt"]
        ctx = arrs[f"{sk}/ctx"] if k >= 1 else np.empty((nv, 0), dtype=np.int16)
        confirm_frac = arrs.get(f"{sk}/confirm_frac")
        supported = bool(nv >= MIN_EXAMPLES)
        # gather heuristic rank arrays once
        hr: dict[str, np.ndarray] = {}
        top: dict[str, np.ndarray] = {}
        for h in HEURISTIC_NAMES:
            rk = arrs.get(f"{sk}/{h}/r")
            if rk is not None:
                hr[h] = rk
                top[h] = arrs.get(f"{sk}/{h}/top")
        for j in range(nv):
            u_slot = int(users[j])
            true_i = int(gt[j])
            hist = [code(str(item_ids[int(c)])) for c in ctx[j] if int(c) >= 0]
            example: dict = {
                "input": json.dumps(
                    {"catalog": cat_id, "user": str(user_ids[u_slot]),
                     "k": k, "history": hist}, separators=(",", ":")),
                "output": code(str(item_ids[true_i])),
                "metadata_catalog": cat_id,
                "metadata_user": str(user_ids[u_slot]),
                "metadata_k": k,
                "metadata_fold": "confirm" if (k + 1) > screen_pos[u_slot] else "screen",
                "metadata_catalog_fold": catalog_fold,
                "metadata_true": code(str(item_ids[true_i])),
                "metadata_headroom": headroom,
                "metadata_sales_hhi": hhi,
                "metadata_sales_norm_entropy": nent,
                "metadata_attr_entropy": aent,
                "metadata_mean_hist": mhist,
                "metadata_supported": supported,
                "metadata_n_valid": nv,
            }
            if mi and "k%d" % k in mi:
                example["metadata_mi_ceiling"] = _ex_float(mi.get(f"k{k}"))
            for h in HEURISTIC_NAMES:
                if h in hr:
                    r = int(hr[h][j])
                    example[f"metadata_rank_{hkey(h)}"] = r
                    tl = top[h][j]
                    valid = [int(x) for x in tl if int(x) >= 0]
                    example[f"predict_{hkey(h)}"] = json.dumps(
                        [code(str(item_ids[x])) for x in valid[: min(50, n_items)]],
                        separators=(",", ":"))
                else:
                    example[f"metadata_rank_{hkey(h)}"] = na_rank[h]
            examples.append(example)
    return examples


def write_parts(catalogs_in_order: list[str], payloads_by_id: dict[str, dict],
                meta: dict, ser: dict, out_dir: Path,
                wall_started_at: float | None = None,
                part_size_mb: int = PART_SIZE_MB) -> list[Path]:
    """Stream examples for the given catalogs (sorted order) into part files
    under out_dir/method_out/method_out_N.json.  Returns part file paths.

    meta: the experiment metadata dict (heuristic inventory, split protocol,
    tuning disclosure, provenance, ...) placed verbatim in every part's
    top-level "metadata".  ser: serialization support dict with per-catalog
    item_ids/user_ids/screen_pos/id_mappings (used by build_examples; NOT
    written into the parts' metadata).  If wall_started_at is given, each
    part's metadata.total_wall_seconds is stamped with the cumulative wall
    time at flush (compute + serialization so far; the last part's stamp is
    the full run wall)."""
    out_dir = Path(out_dir)
    part_dir = out_dir / "method_out"
    part_dir.mkdir(parents=True, exist_ok=True)
    parts: list[Path] = []
    current: list[dict] = []
    current_size = 0
    limit = part_size_mb * 1024 * 1024
    idx = 0

    def flush(force: bool = False) -> None:
        nonlocal current, current_size, idx
        if current and (force or current_size >= limit):
            idx += 1
            path = part_dir / f"method_out_{idx:04d}.json"
            doc_meta = dict(meta)
            if wall_started_at is not None:
                doc_meta["total_wall_seconds"] = round(time.time() - wall_started_at, 1)
            doc = {"metadata": doc_meta, "datasets": current}
            path.write_text(json.dumps(doc))
            parts.append(path)
            logger.info(f"wrote part {path.name}: {len(current)} datasets, "
                        f"{current_size / 1e6:.1f} MB")
            current = []
            current_size = 0

    for cid in catalogs_in_order:
        payload = payloads_by_id[cid]
        examples = build_examples(payload, ser)
        ds = {"dataset": cid, "examples": examples}
        blob = json.dumps(ds)
        if blob_size(blob) > limit and not current:
            # single oversized catalog: write it as its own part (rare)
            current = [ds]
            flush(force=True)
            continue
        current.append(ds)
        current_size += blob_size(blob)
        flush()
    flush(force=True)
    if not parts:
        raise RuntimeError("no output parts written")
    parts.sort(key=lambda p: int(p.stem.rsplit("_", 1)[1]))
    manifest = out_dir / "method_out_manifest.json"
    manifest.write_text(json.dumps({
        "parts": [p.name for p in parts],
        "catalogs": catalogs_in_order,
        "count": len(catalogs_in_order),
    }, indent=1))
    return parts


def blob_size(s: str) -> int:
    return len(s.encode("utf-8"))


def write_aggregates(payloads_by_id: dict[str, dict], out_dir: Path) -> Path:
    """Per (catalog, heuristic, k): n, mean rank, NDCG@5/10, Recall@5/10."""
    out_dir = Path(out_dir)
    rows: list[list] = []
    for cid in sorted(payloads_by_id):
        p = payloads_by_id[cid]
        arrs = p["arrays"]
        for k in K_VALUES:
            sk = str(k)
            if f"{sk}/n" not in arrs:
                continue
            nv = int(arrs[f"{sk}/n"])
            if nv == 0:
                continue
            for h in HEURISTIC_NAMES:
                r = arrs.get(f"{sk}/{h}/r")
                if r is None:
                    rows.append([cid, h, k, nv, "", "", "", "", ""])
                    continue
                r = r.astype(np.float64)
                mean_r = float(r.mean()) if nv else ""
                ndcg5 = float((np.where(r <= 5, 1 / np.log2(r + 1), 0)).mean())
                ndcg10 = float((np.where(r <= 10, 1 / np.log2(r + 1), 0)).mean())
                rec5 = float((r <= 5).mean())
                rec10 = float((r <= 10).mean())
                rows.append([cid, h, k, nv, f"{mean_r:.4f}", f"{ndcg5:.5f}",
                             f"{ndcg10:.5f}", f"{rec5:.5f}", f"{rec10:.5f}"])
    path = out_dir / "aggregates_by_catalog_heuristic_k.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["catalog_id", "heuristic", "k", "n_users", "mean_rank",
                    "ndcg@5", "ndcg@10", "recall@5", "recall@10"])
        w.writerows(rows)
    return path


def write_catalog_diagnostics(payloads_by_id: dict[str, dict], out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    rows: list[list] = []
    for cid in sorted(payloads_by_id):
        p = payloads_by_id[cid]
        d = p.get("diagnostics") or {}
        rows.append([
            cid, p.get("origin"), p.get("source_name"), p.get("fold"),
            d.get("n_items"), d.get("n_users"), f"{d.get('sales_hhi', '')}",
            f"{d.get('sales_norm_entropy', '')}", f"{d.get('attr_entropy', '')}",
            f"{d.get('mean_history_length', '')}", f"{d.get('headroom', '')}",
        ])
    path = out_dir / "catalog_diagnostics.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["catalog_id", "origin", "source", "fold", "n_items",
                    "n_users", "sales_hhi", "sales_norm_entropy", "attr_entropy",
                    "mean_history_length", "headroom"])
        w.writerows(rows)
    return path


