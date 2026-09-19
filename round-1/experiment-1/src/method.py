#!/usr/bin/env python3
"""Cold-start heuristic sweep across small e-commerce catalogs.

CPU-only.  Runs ~16 lightweight heuristics (popularity family, content-kNN
family, co-purchase, lambda hybrids, Bayesian active elicitation) on a
factorial grid of synthetic small-catalog families (plus any real catalogs
found in the shared pool) and emits, per (catalog, user, k, heuristic), the
rank r of the user's true next purchase and the top-50 ranking -- the
complete signal for recomputing NDCG@k/Recall@k at any cut and for
bootstrap CIs downstream.

Usage:
  uv run method.py --mode smoke            # 1 tiny catalog end-to-end
  uv run method.py --mode corners          # 10 grid-corner catalogs
  uv run method.py --mode full             # full 192-catalog grid + pool
  uv run method.py --mode full --max-catalogs 20
  uv run method.py --checks                # unit/logic checks C1-C8
"""

from __future__ import annotations

import json
import multiprocessing as mp
import resource
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from loguru import logger

from catalog_pool import load_pool_catalogs
from synthetic import Catalog, N_USERS_DEFAULT, grid_specs, smoke_specs, corner_specs, generate_catalog
from heuristics import HEURISTIC_SPECS
import output as output_mod

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(str(HERE / "logs" / "run.log"), rotation="30 MB", level="DEBUG")


def _detect_cpus() -> int:
    try:
        parts = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if parts and parts[0] != "max":
            return max(1, int(parts[0]) // int(parts[1]))
    except (FileNotFoundError, ValueError):
        pass
    try:
        import os
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        pass
    return 4


def _container_ram_gb() -> float | None:
    for p in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            v = Path(p).read_text().strip()
            if v != "max" and int(v) < 1_000_000_000_000:
                return int(v) / 1e9
        except (FileNotFoundError, ValueError):
            pass
    return None


def set_limits() -> None:
    import os
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VEC_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[k] = "1"
    mem_gb = _container_ram_gb() or 16.0
    budget = int(min(mem_gb * 0.62, 18.0) * 1024 ** 3)
    try:
        resource.setrlimit(resource.RLIMIT_AS, (budget, budget))
    except (ValueError, OSError) as e:
        logger.warning(f"could not set RLIMIT_AS: {e}")
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (3 * 3600 + 1800, 3 * 3600 + 1800))
    except (ValueError, OSError) as e:
        logger.warning(f"could not set RLIMIT_CPU: {e}")


# ---------------------------------------------------------------------------
# Worker entry (module-level for spawn pickling)
# ---------------------------------------------------------------------------


def _worker_run(task: dict) -> dict:
    """Runs in a spawn worker.  task: {'kind': 'synthetic'|'pool', ...}.
    Returns a small summary + the full compact payload dict (arrays)."""
    t0 = time.time()
    if task["kind"] == "synthetic":
        cat = generate_catalog(task["cfg"])
    else:
        cat = task["catalog"]
    from eval_metrics import evaluate_catalog
    payload = evaluate_catalog(cat)
    payload["catalog"] = cat.catalog_id
    summary = {
        "catalog_id": cat.catalog_id,
        "origin": cat.origin,
        "source_name": cat.source_name,
        "source_path": cat.source_path,
        "fold": cat.fold,
        "n_items": cat.n_items,
        "n_users": cat.n_users,
        "wall_s": round(time.time() - t0, 3),
        "cells": {k: int(payload["arrays"][f"{k}/n"]) for k in
                  [str(x) for x in (0, 1, 2, 3, 5, 8)]
                  if f"{k}/n" in payload["arrays"]},
        "item_ids": [str(x) for x in cat.item_ids],
        "user_ids": [u["user_id"] for u in cat.users],
        "screen_pos": [max(0, int(len(u["item_seq"]) * 0.8)) for u in cat.users],
    }
    return {"summary": summary, "payload": payload}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_meta(workers: int, tasks: list[dict], pool_skipped: list[dict],
               wall_s: float, repro_command: str,
               id_mappings: dict[str, dict[str, str]] | None = None) -> dict:
    meta = {
        "experiment": "Cold-start heuristic sweep across small e-commerce catalogs",
        "description": (
            "Per (catalog, user, k, heuristic): rank r of the user's true next "
            "purchase under temporal leave-one-out-at-k with screen-half-fitted, "
            "leave-one-own-history-out (L1O) parameters.  The integer r is the "
            "complete signal for NDCG@k/Recall@k at any cut <= 50 and for "
            "bootstrap CIs over users."),
        "heuristic_inventory": HEURISTIC_SPECS,
        "k_values": [0, 1, 2, 3, 5, 8],
        "metrics": ["NDCG@5", "NDCG@10", "Recall@5", "Recall@10",
                    "(recomputable from metadata_rank_{h} via "
                    "NDCG_cut = 1/log2(r+1) if r<=cut else 0; Recall_cut = 1 if r<=cut else 0)"],
        "split_protocol": {
            "rule": "temporal 80/20 at user level: screen = positions 1..floor(0.8*L), "
                    "confirm = rest; ground truth for cell (u,k) = purchase at position "
                    "k+1 (valid iff k+1 <= L); context = positions 1..k",
            "fit": "ALL fitted parameters (popularity counts, category/price shares, "
                   "recency decays, TF-IDF, co-occurrence) use ONLY screen-half events; "
                   "per cell the user's own events at positions 1..k+1 (context + ground "
                   "truth) are excluded (L1O-pop): no leakage, confirm events never "
                   "enter fitted parameters",
            "fold_rule": "catalog-level fold: screen (this iteration's corpus) vs "
                         "confirm (reserved for iteration-2); per-example metadata_fold "
                         "= user's ground-truth half ('screen' if k+1 <= floor(0.8L) else 'confirm')",
            "rng": "np.random.default_rng(seed) per catalog; RandomState(0) for "
                   "active-elicitation pair sampling",
        },
        "tuning_disclosure": (
            "Every heuristic parameter is FIXED & DISCLOSED a priori (lambda in "
            "{0.25,0.5,0.75}, recency windows in {0.1,0.5,1.0} quantiles of the screen "
            "span, content-kNN neighborhoods in {1,3,5}, popularity exponent beta=0.5, "
            "band weight gamma=1.0) or chosen ONLY on screen-half users via the "
            "first-j-predict-j+1 inner split (best popularity family for the headroom "
            "diagnostic).  The confirm half is NEVER used to select anything."),
        "filtering": "for k>=1 the user's own context items are excluded from every "
                     "ranking (uniform across heuristics); k=0 has nothing to filter",
        "not_applicable_cells": (
            "content_knn_{1,3,5}, last_item_nbhd, pop_scaled_content, co_purchase are "
            "undefined at k=0 (no history); active_elic2 is defined only at k=0.  In "
            "those cells metadata_rank_{h}='' and predict_{h} is omitted."),
        "degenerate_cells": (
            "at k=0 the lambda_hybrid_* scores reduce to (1-lam)*normalized_popularity "
            "(no content signal) -- reported, marked degenerate in rank distributions."),
        "grid_spec": {
            "n_items": [20, 100, 500, 1000],
            "entropy": ["LOW (2 cats, 2 price bands, 8 tags)", "HIGH (20 cats, 4 price bands, 64 tags)"],
            "zipf_alpha": [0.5, 1.5],
            "epsilon_stickiness": [0.2, 0.6],
            "mean_history": [4, 10],
            "seeds": [0, 1, 2],
            "n_catalogs": 192,
            "n_users_per_catalog": N_USERS_DEFAULT,
        },
        "active_elicitation_disclosure": {
            "questions": 2,
            "selection": "greedy max expected Shannon-entropy reduction of the category "
                         "posterior; all pairs enumerated for |V|<=250 else a fixed "
                         "20k-pair random subset per question (RandomState(catalog seed))",
            "answer_noise": 0.2,
            "answers_simulated_from": "synthetic: beta_u argmax-category with noise; "
                                      "real/pool: option whose category matches the "
                                      "held-out next purchase",
            "final_ranking": "L1O popularity prior re-weighted by the category posterior",
        },
        "prior_art": {
            "pop_global/recency": "Ji et al., A Re-visit of the Popularity Baseline, SIGIR 2020",
            "lambda_hybrid": "B2P P3 = lambda*pop + (1-lambda)*content, Chaimalas et al., RecSys 2023",
            "content_knn/pop_scaled": "content-based cold-start textbook remedy (Basu et al. 1998; Lops et al. 2011)",
            "co_purchase": "association rules: Agrawal & Srikant, VLDB 1994",
            "active_elic2": "Bayesian active/adaptive elicitation; adaptive submodularity, "
                            "Golovin & Krause, ICML 2010 -- no canonical cold-start paper "
                            "(the gap this sweep characterizes)",
        },
        "cpu_only": True,
        "api_spend_usd": 0.0,
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "workers": workers,
        "n_catalogs": len(tasks),
        "pool_files_skipped": pool_skipped,
        "total_wall_seconds": round(wall_s, 1),
        "reproduce": repro_command,
        "output_schema": "exp_gen_sol_out (aii-json), split across method_out/method_out_XXXX.json parts",
        "id_encoding": (
            "Example item ids are the catalogs' own ids, except for pool catalogs whose "
            "item ids exceed 12 chars: those are re-encoded to short aliases p0, p1, ... "
            "inside input/output/predict_/metadata_true; the alias->original mapping is "
            "recorded in metadata_id_mappings per catalog (lossless re-encoding)."),
        "key_sanitization": (
            "exp_gen_sol_out keys are restricted to [a-zA-Z0-9_], so '.' in heuristic "
            "names is replaced by '_' in metadata_rank_*/predict_* keys: "
            "'recency_pop_0.1' -> 'recency_pop_0_1', 'lambda_hybrid_0.25' -> "
            "'lambda_hybrid_0_25'.  The original names are in heuristic_inventory "
            "and in aggregates_by_catalog_heuristic_k.csv."),
        "min_examples_per_cell": 50,
        "metadata_id_mappings": id_mappings or {},
    }
    return meta


def main() -> None:
    import argparse
    import os
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["smoke", "corners", "full"], default="full")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--n-users", type=int, default=N_USERS_DEFAULT)
    parser.add_argument("--max-catalogs", type=int, default=0)
    parser.add_argument("--outdir", type=str, default=str(HERE / "out"))
    parser.add_argument("--checks", action="store_true")
    parser.add_argument("--check", type=str, default="", help="comma-separated C1..C8")
    args = parser.parse_args()

    set_limits()
    logger.info(f"mode={args.mode} workers={args.workers} n_users={args.n_users}")

    if args.checks or args.check:
        import checks
        names = [c.strip() for c in args.check.split(",") if c.strip()] if args.check else []
        ok = checks.run_checks(names if names else None)
        sys.exit(0 if ok else 1)

    t_start = time.time()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "payloads").mkdir(parents=True, exist_ok=True)

    # corpus acquisition: pool first, inline fallback covers the grid
    pool_cats, pool_skipped = load_pool_catalogs()
    if pool_cats:
        logger.info(f"consuming {len(pool_cats)} pooled catalogs")
    else:
        logger.info("no usable pool catalogs -> full inline synthetic fallback")

    if args.mode == "smoke":
        specs = smoke_specs(args.n_users)
    elif args.mode == "corners":
        specs = corner_specs(args.n_users)
    else:
        specs = grid_specs(args.n_users)

    tasks: list[dict] = []
    for c in pool_cats:
        tasks.append({"kind": "pool", "catalog": c})
    for cfg in specs:
        tasks.append({"kind": "synthetic", "cfg": cfg})
    if args.max_catalogs and args.max_catalogs > 0:
        tasks = tasks[: args.max_catalogs]
    n_pool_tasks = sum(1 for t in tasks if t["kind"] == "pool")
    logger.info(f"tasks: {len(tasks)} catalogs "
                f"({n_pool_tasks} pool + {len(tasks) - n_pool_tasks} synthetic)")

    from eval_metrics import evaluate_catalog  # noqa: F401  (warm import)

    cpus = _detect_cpus()
    workers = args.workers or min(cpus, 4)
    payloads_by_id: dict[str, dict] = {}
    summaries_by_id: dict[str, dict] = {}
    provenance: list[dict] = []

    if workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=workers,
                                 mp_context=mp.get_context("spawn")) as pool:
            futs = {pool.submit(_worker_run, t): t for t in tasks}
            for fut in as_completed(futs):
                t = futs[fut]
                cid = (t.get("cfg", {}).get("catalog_id")
                       if t["kind"] == "synthetic" else t["catalog"].catalog_id)
                try:
                    res = fut.result()
                except Exception as e:
                    logger.error(f"catalog {cid} FAILED: {type(e).__name__}: {e}")
                    provenance.append({"catalog_id": cid, "status": "failed",
                                       "error": f"{type(e).__name__}: {e}"})
                    continue
                payloads_by_id[res["summary"]["catalog_id"]] = res["payload"]
                summaries_by_id[res["summary"]["catalog_id"]] = res["summary"]
                s = res["summary"]
                provenance.append({
                    "catalog_id": s["catalog_id"], "status": "ok", "origin": s["origin"],
                    "source_name": s["source_name"], "source_path": s["source_path"],
                    "fold": s["fold"], "n_items": s["n_items"], "n_users": s["n_users"],
                    "wall_s": s["wall_s"], "cells": s["cells"],
                })
                logger.info(f"done {s['catalog_id']} ({s['n_items']} items, "
                            f"{s['n_users']} users, {s['wall_s']}s)")
    else:
        for t in tasks:
            cid = (t.get("cfg", {}).get("catalog_id")
                   if t["kind"] == "synthetic" else t["catalog"].catalog_id)
            try:
                res = _worker_run(t)
            except Exception as e:
                logger.error(f"catalog {cid} FAILED: {type(e).__name__}: {e}")
                provenance.append({"catalog_id": cid, "status": "failed",
                                   "error": f"{type(e).__name__}: {e}"})
                continue
            payloads_by_id[res["summary"]["catalog_id"]] = res["payload"]
            summaries_by_id[res["summary"]["catalog_id"]] = res["summary"]
            s = res["summary"]
            provenance.append({
                "catalog_id": s["catalog_id"], "status": "ok", "origin": s["origin"],
                "source_name": s["source_name"], "source_path": s["source_path"],
                "fold": s["fold"], "n_items": s["n_items"], "n_users": s["n_users"],
                "wall_s": s["wall_s"], "cells": s["cells"],
            })
            logger.info(f"done {s['catalog_id']} ({s['wall_s']}s)")

    if not payloads_by_id:
        logger.error("no catalogs produced any output")
        sys.exit(1)

    # runtime metadata needed by output.build_examples; long item ids of pool
    # catalogs are re-encoded to short aliases (size; mapping recorded below)
    id_mappings: dict[str, dict[str, str]] = {}
    for cid in payloads_by_id:
        ids = summaries_by_id[cid]["item_ids"]
        if not ids:
            continue
        if max(len(x) for x in ids) > 12:
            id_mappings[cid] = {x: f"p{j}" for j, x in enumerate(ids)}
    run_meta = {
        "item_ids": {cid: summaries_by_id[cid]["item_ids"] for cid in payloads_by_id},
        "user_ids": {cid: summaries_by_id[cid]["user_ids"] for cid in payloads_by_id},
        "screen_pos": {cid: summaries_by_id[cid]["screen_pos"] for cid in payloads_by_id},
        "id_mappings": id_mappings,
    }

    repro = "uv run method.py --mode full"
    if args.mode != "full":
        repro = f"uv run method.py --mode {args.mode}"

    order = sorted(payloads_by_id)
    # write_parts stamps each part's metadata.total_wall_seconds with the
    # cumulative wall at flush (compute + serialization); in-memory meta gets
    # the final wall for the provenance/manifest readers.
    meta = build_meta(workers, tasks, pool_skipped, 0.0, repro, id_mappings)
    parts = output_mod.write_parts(order, payloads_by_id, meta, run_meta, outdir,
                                   wall_started_at=t_start)
    agg = output_mod.write_aggregates(payloads_by_id, outdir)
    diag_path = output_mod.write_catalog_diagnostics(payloads_by_id, outdir)
    meta["total_wall_seconds"] = round(time.time() - t_start, 1)

    prov_path = outdir / "provenance.jsonl"
    with open(prov_path, "w") as f:
        for row in provenance:
            f.write(json.dumps(row) + "\n")

    wall = time.time() - t_start
    logger.info(f"TOTAL wall: {wall:.1f}s; catalogs OK: {len(payloads_by_id)}")
    logger.info(f"parts: {[p.name for p in parts]}")
    logger.info(f"aggregates: {agg.name}; diagnostics: {diag_path.name}")
    logger.info(f"provenance: {prov_path.name}")
    logger.info(f"reproduce: {repro}")


if __name__ == "__main__":
    main()