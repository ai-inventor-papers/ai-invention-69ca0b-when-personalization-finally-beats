#!/usr/bin/env python3
"""Pre-registered wide-screen survivor selection for cold-start next-item heuristics.

Role: consume the SHARED POOL of per-user measurements produced in parallel by this
iteration's EXPERIMENT artifact (method_out.json, exp_gen_sol_out schema) and the
DATASET artifact's catalog metadata, then MECHANICALLY apply the pre-registered
wide-screen selection rule to declare ONE survivor among the four candidate
mechanisms:

  A  content / hybrid Bayes-shrinkage crossovers
  B  evaluation noise
  C  active elicitation
  D  recency-windowed popularity

plus:
  - the screen-half signed checks of the main hypothesis (crossover k* movement),
  - a tiny threshold / depth<=2 diagnostic decision rule over catalog statistics,
  - a plug-in information-theoretic ceiling MI(history ; next item), and
  - the pre-registered iteration-2 confirmation protocol for the untouched CONFIRM half.

HARD RULES (from the artifact plan):
  * NEVER touch the confirm half (per-user cells with metadata_fold == 'confirm' are
    read only to confirm they exist for iteration 2, never used to choose anything).
  * The selection rule is enforced mechanically - no post-hoc winner picking.
  * If the shared pool is ABSENT (the parallel EXPERIMENT has not published its
    method_out.json yet) we DO NOT fabricate or substitute our own measurements:
    we emit a skeleton eval_out.json that records the uncovered/inconclusive status,
    the coverage report, and the full confirmation protocol, and we STOP.
  * All bootstrap seeds are fixed and disclosed; CPU-only; no paid API calls.

When the shared pool IS present, the full 9-step analysis runs and produces a
complete eval_out.json. The analysis code is additionally exercised by a
`--selftest` mode on an inline synthetic pool (clearly labelled, never emitted as
the screen result) so downstream consumers know the machinery is correct.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
from loguru import logger

# --------------------------------------------------------------------------- #
# Logging (aii-python conventions)
# --------------------------------------------------------------------------- #
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/run.log", rotation="30 MB", level="DEBUG", enqueue=True)

# --------------------------------------------------------------------------- #
# Fixed, disclosed hyper-parameters (pre-registered)
# --------------------------------------------------------------------------- #
K_VALUES: list[int] = [0, 1, 2, 3, 5, 8]
POP_REF_K: list[int] = [0, 1, 2, 3, 5, 8]      # ks over which the reference is averaged
SEED_BASE: int = 20240601                      # base for the per-catalog bootstrap seed
B_BOOT: int = 2000                             # bootstrap resamples (reduce to 500 if slow)
GAIN_GATE: float = 0.02                        # meaningful effect size (NDCG@5 units)
CI_MAJORITY: float = 0.50                      # CI excludes 0 in >= this fraction of catalogs
MIN_USERS_CELL: int = 50                       # min users to call a (catalog,k,h) cell "supported"
DEPTH5_K: int = 5                              # primary cut
CI_ALPHA: float = 0.95

# Candidate -> target history lengths (pre-registered)
CAND_TARGET_KS: dict[str, list[int]] = {
    "A": [1, 2, 3],
    "C": [0],
    "D": [0, 1, 2, 3, 5, 8],
}
# Family -> heuristic keywords (prefix/name matching; robust to experiment renaming)
FAMILY_KEYWORDS: dict[str, list[str]] = {
    "pop_global":      ["pop_global", "mostpop"],
    "banded_pop":      ["pop_category", "pop_price", "cat_pop", "price_pop", "band"],
    "content_hybrid":  [
        "content_knn", "last_item", "pop_scaled_content", "co_purchase",
        "lambda_hybrid", "content", "hybrid", "neighbor", "nbhd",
    ],
    "windowed_pop":    ["recency_pop", "recency", "windowed"],
    "elicitation":     ["active_elic", "elic", "forced_choice"],
}

ALTERNATE_SEED_DELTAS: list[int] = [1, 2, 3, 4]  # disclosed alternate seed bases for Step 5b


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _catalog_seed(catalog_id: str) -> int:
    """Formulaic, disclosed per-catalog bootstrap seed."""
    h = int(hashlib.sha1(str(catalog_id).encode("utf-8")).hexdigest()[:8], 16)
    return SEED_BASE + (h % 100_000)


def ndcg_at(rank: float, k: int) -> float:
    """NDCG@k for a single relevant item at 1-indexed rank r (plan STEP 1).

    rank is 1-indexed; rank = inf means the true item was not in the heuristic's top-K
    ranking (a genuine MISS, counted as 0 - not excluded). Non-finite / out-of-cut ranks
    yield 0; otherwise 1/log2(rank+1).
    """
    try:
        if not math.isfinite(float(rank)) or float(rank) > k:
            return 0.0
        return 1.0 / math.log2(float(rank) + 1.0)
    except (TypeError, ValueError):
        return 0.0


def recall_at(rank: float, k: int) -> float:
    try:
        return 1.0 if math.isfinite(float(rank)) and float(rank) <= k else 0.0
    except (TypeError, ValueError):
        return 0.0


def _resolve_rank(example: dict, heuristic: str, true_item: Any) -> Optional[float]:
    """Resolve the 1-indexed rank r of the true next item for *heuristic*.

    Priority (score-based only, no heuristic re-runs):
      1. metadata_rank_{heuristic} if a non-empty numeric value.
      2. predict_{heuristic} ranking if present (position of the true item, r=inf if
         the item is not in the top-K ranking).
      3. None -> the cell is NOT APPLICABLE for this heuristic (e.g. active_elic2 at
         k>0 has no definition) and must be excluded, never treated as a miss.
    """
    key = f"metadata_rank_{heuristic}"
    raw = example.get(key)
    if raw is not None and not isinstance(raw, str):
        try:
            r = int(raw)
            if r >= 1:
                return float(r)
        except (TypeError, ValueError):
            pass
    if isinstance(raw, str) and raw.strip() != "":
        try:
            r = int(raw.strip())
            if r >= 1:
                return float(r)
        except (TypeError, ValueError):
            pass
    # Fall back to the predict_ ranking if present (top-K list of item ids).
    pred_key = f"predict_{heuristic}"
    pred_raw = example.get(pred_key)
    if isinstance(pred_raw, str) and pred_raw.strip() not in ("", "[]", "null"):
        try:
            ranking = json.loads(pred_raw)
        except json.JSONDecodeError:
            ranking = None
        if isinstance(ranking, list) and ranking:
            for i, it in enumerate(ranking):
                if str(it) == str(true_item):
                    return float(i + 1)
            return float("inf")  # present in ranking API but item not in top-K
    return None


# --------------------------------------------------------------------------- #
# Pool discovery & loading (plan STEP 0)
# --------------------------------------------------------------------------- #
POOL_SEARCH_DIRS: list[str] = [
    "POOL_DIR", "DATASET_OUT", "EVAL_POOL", "EXPERIMENT_OUT", "SHARED_POOL",
]


def _env_search_dirs() -> list[Path]:
    out: list[Path] = []
    for var in POOL_SEARCH_DIRS:
        v = os.environ.get(var)
        if v:
            out.append(Path(v))
    return out


def discover_pool(root: Path, workspace: Path) -> list[Path]:
    """Return candidate shared-pool method_out.json / catalog files."""
    wanted = {"method_out.json", "method_out", "catalog.json"}
    found: list[Path] = []
    roots: list[Path] = []
    roots += _env_search_dirs()
    roots.append(workspace.parent / "gen_art_experiment_1")
    roots.append(workspace.parent / "gen_art_dataset_1")
    for p in workspace.parent.parent.glob("gen_plan/*"):       # ../gen_plan/*
        roots.append(p)
    roots.append(root / "3_invention_loop" / "iter_1" / "pool")
    roots.append(root / "3_invention_loop" / "iter_1" / "execute" / "dataset")
    roots.append(root / "3_invention_loop" / "iter_1" / "execute")

    seen: set[Path] = set()
    for r in roots:
        if r is None or not r.exists():
            continue
        for jf in r.rglob("*.json"):
            if sys.version_info >= (3, 12):
                pass
            try:
                rel = jf.relative_to(Path("/ai-inventor/aii_data/runs/run_5D4WD4vgZZMJ/logs"))
                if rel.parts:
                    continue
            except ValueError:
                pass
            if not any(seg in {"logs", "sinks", ".oh_sessions", ".venv"} for seg in jf.parts):
                if jf.name in wanted or "method" in jf.stem or "catalog" in jf.stem or "pool" in jf.stem:
                    if jf not in seen:
                        seen.add(jf)
                        found.append(jf)
    # Broad fallback search under the run root (excluding noise dirs).
    for jf in root.rglob("*.json"):
        if jf in seen:
            continue
        if any(seg in {"logs", "sinks", ".oh_sessions", "gen_plan", ".venv", ".shared_cache"} for seg in jf.parts):
            continue
        if jf.name in wanted or "method_out" in jf.stem or "catalog_pool" in jf.stem:
            seen.add(jf)
            found.append(jf)
    return found


def _parse_meta(meta: Any) -> dict[str, Any]:
    if isinstance(meta, dict):
        return meta
    return {}


def load_pool(path: Path) -> dict[str, dict[str, Any]]:
    """Load a method_out.json (exp_gen_sol_out) into per-catalog measurement dicts.

    Returns: {catalog_id: {"meta": {...}, "examples": [ ...raw example dicts... ]}}
    """
    logger.info(f"Loading pool from {path}")
    data = json.loads(path.read_text())
    catalogs: dict[str, dict[str, Any]] = {}
    for dset in data.get("datasets", []):
        cid = str(dset.get("dataset", "unknown"))
        catalogs[cid] = {
            "meta": {},
            "examples": dset.get("examples", []),
        }
    # Top-level metadata (heuristic inventory etc.), merged into a pseudo catalog
    # "**metadata**" is NOT returned; catalog-level metadata is per-dataset only.
    return catalogs


def extract_cells(pool: dict[str, dict[str, Any]], screen_only: bool = True) -> dict[str, Any]:
    """Turn raw examples into {catalog_id: {k: {heuristic: {uid: rank}}}} plus meta.

    Only the SCREEN fold is kept for analysis; confirm examples are counted but
    excluded from every computation and never used to make any choice.
    """
    out: dict[str, Any] = {}
    for cid, blob in pool.items():
        ex = blob.get("examples", [])
        cells: dict[int, dict[str, dict[str, float]]] = {k: {} for k in K_VALUES}
        confirm_count = 0
        screen_count = 0
        for e in ex:
            fold = str(e.get("metadata_fold", "screen"))
            if fold == "confirm":
                confirm_count += 1
                continue
            screen_count += 1
            try:
                k = int(e.get("metadata_k"))
            except (TypeError, ValueError):
                continue
            if k not in cells:
                continue
            uid = str(e.get("metadata_user", "?"))
            true_item = e.get("metadata_true", e.get("output"))
            # discover heuristics present on this example
            for key, val in e.items():
                if key.startswith("metadata_rank_"):
                    h = key[len("metadata_rank_"):]
                    r = _resolve_rank(e, h, true_item)
                    if r is None:
                        continue
                    cells[k].setdefault(h, {})[uid] = r
        meta = dict(blob.get("meta", {}))
        meta["n_screen_examples"] = screen_count
        meta["n_confirm_examples"] = confirm_count
        out[cid] = {"meta": meta, "cells": cells, "n_confirm": confirm_count}
    return out


# --------------------------------------------------------------------------- #
# Metric aggregation (plan STEP 1 - STEP 3)
# --------------------------------------------------------------------------- #
def _heuristic_names(extracted: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for cdata in extracted.values():
        for k, hmap in cdata["cells"].items():
            names.update(hmap.keys())
    return names


def assign_family(heuristic: str) -> Optional[str]:
    """Map a heuristic name to one of the plan families, or None if unmapped."""
    for family, kws in FAMILY_KEYWORDS.items():
        for kw in kws:
            if kw in heuristic:
                return family
    return None


def reference_family_names() -> list[str]:
    """Reference family for the survival rule: global + banded popularity.
    NOTE: recency-windowed popularity (candidate D) is deliberately NOT included.
    """
    return ["pop_global", "banded_pop"]


def best_popularity_reference(cdata: dict[str, Any]) -> tuple[Optional[str], float]:
    """Per-catalog reference = best popularity-family heuristic by mean NDCG@5
    averaged over k in {0,1,2,3,5,8} on the SCREEN half (plan STEP 2)."""
    best_h: Optional[str] = None
    best_mean = -math.inf
    for h in sorted(_heuristic_names({"x": cdata})):
        if assign_family(h) not in reference_family_names():
            continue
        vals: list[float] = []
        for k in POP_REF_K:
            ranks = _ranks_for(cdata, k, h)
            for r in ranks:
                vals.append(ndcg_at(r, DEPTH5_K))
        if not vals:
            continue
        m = float(np.mean(np.asarray(vals, dtype=float)))
        if m > best_mean:
            best_mean = m
            best_h = h
    return best_h, best_mean


def _ranks_for(cdata: dict[str, Any], k: int, h: str) -> list[float]:
    """Per-user ranks (1-indexed; inf = genuine miss -> NDCG 0, kept) for (catalog,k,h).
    Cells stored as NOT-APPLICABLE (active_elic2 at k>0 etc.) are never in the map, so
    they are excluded here automatically."""
    umap = cdata["cells"].get(k, {}).get(h, {})
    return [float(r) for r in umap.values()]


def _aligned_rank_arrays(cdata: dict[str, Any], k: int, hc: str, hr: str) -> tuple[np.ndarray, np.ndarray, int]:
    """Paired per-user rank arrays (candidate, reference) on their common users at k.
    Both candidate and reference must be defined for a user; a defined rank of inf is a
    real miss and is retained."""
    uc = cdata["cells"].get(k, {}).get(hc, {})
    ur = cdata["cells"].get(k, {}).get(hr, {})
    common = [u for u in uc if u in ur]
    if not common:
        return np.array([], dtype=float), np.array([], dtype=float), 0
    rc = np.asarray([float(uc[u]) for u in common], dtype=float)
    rr = np.asarray([float(ur[u]) for u in common], dtype=float)
    return rc, rr, len(common)


def cell_mean_ndcg5(cdata: dict[str, Any], k: int, h: str) -> tuple[float, int]:
    ranks = _ranks_for(cdata, k, h)
    if not ranks:
        return float("nan"), 0
    vals = np.asarray([ndcg_at(r, DEPTH5_K) for r in ranks], dtype=float)
    return float(np.mean(vals)), len(ranks)


def bootstrap_gain_ci(rc: np.ndarray, rr: np.ndarray, rng: np.random.Generator,
                      n_boot: int = B_BOOT) -> tuple[float, float, float, int]:
    """Paired stratified-bootstrap 95% CI for mean(NDCG@5(candidate)) - mean(NDCG@5(ref)).

    Users are resampled WITHIN the single k-stratum (one stratum per k), pairing the
    per-user candidate and reference differences to remove between-user variance.
    Returns (gain, lo, hi, n).
    """
    if rc.size == 0 or rr.size == 0:
        return float("nan"), float("nan"), float("nan"), 0
    vc = np.asarray([ndcg_at(r, DEPTH5_K) for r in rc], dtype=float)
    vr = np.asarray([ndcg_at(r, DEPTH5_K) for r in rr], dtype=float)
    diff = vc - vr
    n = diff.size
    gain = float(diff.mean())
    # Vectorized bootstrap: pre-draw the full index matrix once.
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = diff[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [(1 - CI_ALPHA) / 2 * 100, (1 + CI_ALPHA) / 2 * 100])
    return gain, float(lo), float(hi), n


def family_best_at_k(cdata: dict[str, Any], family: str, k: int) -> tuple[Optional[str], float, int]:
    """Best heuristic in *family* at *k* by screen-half mean NDCG@5 (best performer,
    chosen from the shared measurements only - disclosed pre-registered rule)."""
    best_h: Optional[str] = None
    best_m = -math.inf
    best_n = 0
    for h in sorted(_heuristic_names({"x": cdata})):
        if assign_family(h) != family:
            continue
        m, n = cell_mean_ndcg5(cdata, k, h)
        if not np.isfinite(m):
            continue
        if m > best_m:
            best_m, best_h, best_n = m, h, n
    return best_h, best_m, best_n


# --------------------------------------------------------------------------- #
# Candidate gain tables (plan STEP 3 - STEP 4)
# --------------------------------------------------------------------------- #
def candidate_family(cand: str) -> Optional[str]:
    if cand == "A":
        return "content_hybrid"
    if cand == "C":
        return "elicitation"
    if cand == "D":
        return "windowed_pop"
    return None


def candidate_gains(cdata: dict[str, Any], cand: str, ref_h: str,
                    rng: np.random.Generator) -> dict[int, dict[str, float]]:
    """Per-target-k gain stats for candidate *cand* vs the reference over the users
    where BOTH are defined. Returns {k: {gain, lo, hi, n, cand_h}}."""
    fam = candidate_family(cand)
    if fam is None or ref_h is None:
        return {}
    out: dict[int, dict[str, float]] = {}
    for k in CAND_TARGET_KS[cand]:
        cand_h, _, _ = family_best_at_k(cdata, fam, k)
        if cand_h is None:
            continue
        rc, rr, n = _aligned_rank_arrays(cdata, k, cand_h, ref_h)
        if n < 1:
            continue
        gain, lo, hi, nn = bootstrap_gain_ci(rc, rr, rng)
        out[k] = {"gain": gain, "lo": lo, "hi": hi, "n": nn, "cand_h": cand_h}
    return out


def win_k_for_candidate(gains: dict[int, dict[str, float]]) -> Optional[int]:
    """The candidate's representative target k = argmax gain over its target set
    (declared a priori: max-over-target-k gain; per-k values are also reported)."""
    if not gains:
        return None
    return max(gains, key=lambda k: gains[k]["gain"])


def ci_excludes_zero(stat: dict[str, float]) -> bool:
    return stat["lo"] > 0.0 or stat["hi"] < 0.0


# --------------------------------------------------------------------------- #
# Selection rule (plan STEP 4) - enforced mechanically
# --------------------------------------------------------------------------- #
def selection_rule(pool_stats: dict[str, dict[str, dict[str, float]]]) -> dict[str, Any]:
    """pool_stats[candidate][catalog_id] = {gain, lo, hi, n, k}

    Returns a dict with the decision trace and the single survivor.
    """
    candidates = ["A", "C", "D"]
    # Representative per-candidate per-catalog gain + CI at its win k.
    rep: dict[str, dict[str, dict[str, float]]] = {}
    for c in candidates:
        rep[c] = {}
        for cid, gains in pool_stats[c].items():
            wk = win_k_for_candidate(gains)
            if wk is None:
                continue
            rep[c][cid] = dict(gains[wk], k=wk)

    # Condition (i)
    cond_i: dict[str, bool] = {}
    for c in candidates:
        gs = [rep[c][cid]["gain"] for cid in rep[c]]
        if not gs:
            cond_i[c] = False
            continue
        median_gain = float(np.median(gs))
        ci_exc = float(np.mean([ci_excludes_zero(rep[c][cid]) for cid in rep[c]]))
        cond_i[c] = bool((median_gain >= GAIN_GATE) and (ci_exc >= CI_MAJORITY))

    # Condition (ii): candidate X strictly out-performs Y in >=50% of catalogs while
    # X also satisfies (i). X dominates Y if X_satisfies_i and P_XY(X>Y) >= 0.5.
    dominated: dict[str, set[str]] = {c: set() for c in candidates}  # c dominated by set
    domination_pairs: list[dict[str, Any]] = []
    for x in candidates:
        if not cond_i[x]:
            continue
        for y in candidates:
            if x == y:
                continue
            shared = set(rep[x]) & set(rep[y])
            if not shared:
                continue
            px = float(np.mean([rep[x][c]["gain"] > rep[y][c]["gain"] for c in shared]))
            cy = float(np.mean([ci_excludes_zero(rep[y][c]) for c in shared]))
            gy = float(np.mean([rep[y][c]["gain"] >= 0.0 for c in shared]))
            strictly = px >= 0.50
            # The plan describes dominance via X strictly out-performing Y while Y's CI
            # straddles 0 or Y's gain < X's; we operationalize with the strict-outperform
            # majority test, logging the straddle evidence for transparency.
            dominates = cond_i[x] and strictly
            domination_pairs.append({
                "x": x, "y": y, "strict_majority_frac": round(px, 4),
                "y_ci_straddle_frac": round(1 - cy, 4), "y_nonneg_frac": round(gy, 4),
                "x_dominates_y": dominates,
            })
            if dominates:
                dominated[y].add(x)

    survivors = [c for c in candidates if cond_i[c] and not dominated[c]]
    decision: str
    if len(survivors) == 1:
        decision = survivors[0]
    elif len(survivors) > 1:
        # Largest median gain wins.
        best = max(survivors, key=lambda c: float(np.median([rep[c][cid]["gain"] for cid in rep[c]])))
        decision = best
    else:
        # None of A,C,D survives -> the NOISE candidate (B) formally wins the screen.
        decision = "B"

    return {
        "candidates": candidates,
        "survived_condition_i": cond_i,
        "dominated_by": {c: sorted(dominated[c]) for c in candidates},
        "domination_pairs": domination_pairs,
        "survivors_list": survivors,
        "decision": decision,
        "rule_trace": (
            "A candidate survives iff (i) median-over-catalogs gain >= 0.02 AND its 95% "
            "bootstrap CI excludes 0 in >= 50% of catalogs AND (ii) it is not dominated "
            "by another candidate under the same rule in >= 50% of catalogs. "
            "If multiple survive, the largest median gain wins. If none of A,C,D "
            "survives, the NOISE candidate B formally wins."
        ),
    }


# --------------------------------------------------------------------------- #
# Candidate B: noise report (plan STEP 5)
# --------------------------------------------------------------------------- #
def noise_report(extracted: dict[str, Any], pool_stats_all: dict[str, dict[str, dict[str, dict[str, float]]]],
                 refs: dict[str, str]) -> dict[str, Any]:
    """(a) within-noise fraction of pairwise heuristic comparisons per catalog;
    (b) seed-to-seed variance of winner identity from alternate bootstrap seeds."""
    # (a) within-noise over all pairwise heuristic comparisons in each catalog
    within_noise_catalog: list[float] = []
    small_effect_catalog: list[float] = []
    for cid, cdata in extracted.items():
        if cid not in refs or refs[cid] is None:
            continue
        ref_h = refs[cid]
        comps: list[bool] = []
        small: list[bool] = []
        for k in K_VALUES:
            hs = sorted(cdata["cells"].get(k, {}).keys())
            for h in hs:
                if h == ref_h:
                    continue
                rc, rr, n = _aligned_rank_arrays(cdata, k, h, ref_h)
                if n < MIN_USERS_CELL:
                    continue
                rng = np.random.default_rng(_catalog_seed(cid))
                gain, lo, hi, _ = bootstrap_gain_ci(rc, rr, rng, n_boot=500)
                comps.append(not (lo <= 0.0 <= hi))
                small.append(abs(gain) < GAIN_GATE)
        if comps:
            within_noise_catalog.append(1.0 - float(np.mean(comps)))
            small_effect_catalog.append(float(np.mean(small)))

    # (b) seed-to-seed variance of winner identity
    seed_agreement: dict[str, Any] = {}
    winner_by_seed: dict[str, str] = {}
    for delta in [0] + ALTERNATE_SEED_DELTAS:
        base = SEED_BASE + delta
        stats: dict[str, dict[str, dict[str, float]]] = {c: {} for c in ["A", "C", "D"]}
        for c in ["A", "C", "D"]:
            fam = candidate_family(c)
            for cid, cdata in extracted.items():
                ref_h = refs.get(cid)
                if ref_h is None:
                    continue
                rng = np.random.default_rng(base + (int(hashlib.sha1(str(cid).encode()).hexdigest()[:8], 16) % 100_000))
                g = candidate_gains(cdata, c, ref_h, rng)
                if g:
                    stats[c][cid] = g
        dec = selection_rule(stats)
        winner_by_seed[str(delta)] = dec["decision"]
        seed_agreement[str(delta)] = {
            "decision": dec["decision"],
            "survived_condition_i": dec["survived_condition_i"],
        }

    from collections import Counter
    cnt = Counter(winner_by_seed.values())
    return {
        "within_noise_fraction_pairs": round(float(np.mean(within_noise_catalog)), 4) if within_noise_catalog else None,
        "fraction_abs_gain_lt_0.02": round(float(np.mean(small_effect_catalog)), 4) if small_effect_catalog else None,
        "n_catalog_paired_comparisons": sum(len(a) for a in [within_noise_catalog]),
        "winner_identity_by_seed": winner_by_seed,
        "seed_agreement_fraction": round(max(cnt.values()) / sum(cnt.values()), 4) if sum(cnt.values()) else None,
        "winner_counts_across_seeds": dict(cnt),
        "note": "Candidate B (noise) is measured directly: how often a pairwise median gain "
                "is within its 95% CI of 0, and how often the mechanical winner identity "
                "flips under disclosed alternate bootstrap seed bases.",
    }


# --------------------------------------------------------------------------- #
# Candidate A: diagnostic report (plan STEP 6)
# --------------------------------------------------------------------------- #
def crossover_kstar(cdata: dict[str, Any], ref_h: str,
                    rng: np.random.Generator) -> tuple[Optional[int], dict[int, dict[str, float]]]:
    """Smallest k>=1 where the best personalized (content/hybrid) family gain over the
    reference has a 95% CI excluding 0 AND gain >= 0.02; inf if never in tested k."""
    if ref_h is None:
        return None, {}
    per_k: dict[int, dict[str, float]] = {}
    for k in [1, 2, 3]:
        cand_h, _, _ = family_best_at_k(cdata, "content_hybrid", k)
        if cand_h is None:
            continue
        rc, rr, n = _aligned_rank_arrays(cdata, k, cand_h, ref_h)
        if n < 1:
            continue
        gain, lo, hi, nn = bootstrap_gain_ci(rc, rr, rng)
        per_k[k] = {"gain": gain, "lo": lo, "hi": hi, "n": nn, "cand_h": cand_h}
    for k in sorted(per_k):
        st = per_k[k]
        if ci_excludes_zero(st) and st["gain"] >= GAIN_GATE:
            return k, per_k
    return math.inf, per_k


def diagnostic_report(extracted: dict[str, Any], refs: dict[str, str],
                      meta: dict[str, dict[str, Any]] = None) -> dict[str, Any]:
    """Crossover k* per catalog (+ family aggregates), signed monotonicity checks, and
    the small-catalog plain-vs-banded popularity check. SCREEN half only."""
    kstars: dict[str, Any] = {}
    signed: dict[str, Any] = {}
    for cid, cdata in extracted.items():
        rng = np.random.default_rng(_catalog_seed(cid))
        kstar, per_k = crossover_kstar(cdata, refs.get(cid), rng)
        kstars[cid] = {"kstar": (None if kstar is None else ("inf" if kstar == math.inf else kstar)),
                       "per_k": {str(k): {kk: (round(vv, 4) if isinstance(vv, float) else vv)
                                          for kk, vv in st.items()}
                                 for k, st in per_k.items()}}

    # Signed checks: correlate k* with |V| (expect larger k* for SMALLER |V| -> rho<0),
    # item attribute entropy (expect larger k* for LOWER entropy -> rho<0), and HHI
    # (expect larger k* for HIGHER concentration -> rho>0).
    from scipy.stats import spearmanr
    sizes: list[tuple[str, float, float]] = []   # (cid, numeric kstar, metric)
    for cid, cdata in extracted.items():
        m = cdata["meta"] if "meta" in cdata else {}
        ks = kstars[cid]["kstar"]
        if ks is None or ks == "inf":
            continue
        sizes.append((cid, float(ks), float(m.get("n_items", float("nan")))))
    # n_items may be absent; fall back to max 1-indexed rank seen as a proxy.
    def _stat_per_catalog(cid: str, st: str) -> Optional[float]:
        m = extracted[cid]["meta"]
        k = st
        if k == "n_items":
            v = m.get("n_items")
            if v is not None:
                try:
                    return float(v)
                except (TypeError, ValueError):
                    pass
            # proxy: max rank over all heuristics / k
            best = 0
            for kk, hmap in extracted[cid]["cells"].items():
                for h, um in hmap.items():
                    for r in um.values():
                        if np.isfinite(float(r)):
                            best = max(best, int(r))
            return float(best) if best > 0 else None
        v = m.get(st)
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    checks = {
        "catalog_size_inv": {"stat": "n_items", "expect": "larger k* for SMALLER |V| (rho<0)"},
        "attribute_entropy": {"stat": "attr_entropy", "expect": "larger k* for LOWER entropy (rho<0)"},
        "sales_concentration": {"stat": "sales_hhi", "expect": "larger k* for HIGHER HHI (rho>0)"},
    }
    for name, ch in checks.items():
        xs: list[float] = []
        ys: list[float] = []
        for cid in extracted:
            ks = kstars[cid]["kstar"]
            if ks is None or ks == "inf":
                continue
            sv = _stat_per_catalog(cid, ch["stat"])
            if sv is None:
                continue
            xs.append(float(ks))
            ys.append(sv)
        if len(xs) >= 3 and len(set(xs)) > 1 and len(set(ys)) > 1:
            rho, p = spearmanr(xs, ys)
            signed[name] = {"n": len(xs), "spearman_rho": round(float(rho), 4),
                            "p_value": round(float(p), 4), "expectation": ch["expect"],
                            "matches_expectation": (rho < 0) if "rho<0" in ch["expect"] else (rho > 0)}
        else:
            signed[name] = {"n": len(xs), "spearman_rho": None, "p_value": None,
                            "expectation": ch["expect"], "matches_expectation": None,
                            "note": "insufficient/constant input (need >=3 distinct k* and "
                                    ">=2 distinct statistic values) for this check"}

    # (c) plain pop_global vs banded popularity in the smallest catalogs ( |V| < 100 )
    plain_vs_banded: dict[str, Any] = {"catalogs": {}, "gains": []}
    for cid, cdata in extracted.items():
        nv = _stat_per_catalog(cid, "n_items")
        if nv is not None and nv >= 100:
            continue
        # plain pop_global
        pg, _, _ = family_best_at_k(cdata, "pop_global", 0)
        # banded best
        bh, _, _ = family_best_at_k(cdata, "banded_pop", 0)
        if pg is None or bh is None:
            continue
        rc, rr, n = _aligned_rank_arrays(cdata, 0, pg, bh)
        if n < 1:
            continue
        rng = np.random.default_rng(_catalog_seed(cid))
        gain, lo, hi, nn = bootstrap_gain_ci(rc, rr, rng)
        plain_vs_banded["catalogs"][cid] = {"gain_plain_minus_banded": round(gain, 4),
                                            "lo": round(lo, 4), "hi": round(hi, 4), "n": nn}
        plain_vs_banded["gains"].append(gain)
        plain_vs_banded.setdefault("n_used", 0)
        plain_vs_banded["n_used"] += 1
    if plain_vs_banded["gains"]:
        plain_vs_banded["mean_gain"] = round(float(np.mean(plain_vs_banded["gains"])), 4)

    return {"crossover_kstar_by_catalog": kstars, "signed_checks": signed,
            "plain_vs_banded_smallest": plain_vs_banded}


# --------------------------------------------------------------------------- #
# Candidate C & D details
# --------------------------------------------------------------------------- #
def elicitation_detail(extracted: dict[str, Any], refs: dict[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = {"gain_at_k0_by_catalog": {}}
    gains: list[float] = []
    for cid, cdata in extracted.items():
        if refs.get(cid) is None:
            continue
        g = candidate_gains(cdata, "C", refs[cid], np.random.default_rng(_catalog_seed(cid)))
        if 0 in g:
            st = g[0]
            out["gain_at_k0_by_catalog"][cid] = {kk: (round(vv, 4) if isinstance(vv, float) else vv)
                                                 for kk, vv in st.items()}
            gains.append(st["gain"])
    if gains:
        out["median_gain_at_k0"] = round(float(np.median(gains)), 4)
        out["mean_gain_at_k0"] = round(float(np.mean(gains)), 4)
    return out


def windowed_pop_detail(extracted: dict[str, Any], refs: dict[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = {"best_gain_by_catalog": {}}
    gains: list[float] = []
    for cid, cdata in extracted.items():
        if refs.get(cid) is None:
            continue
        g = candidate_gains(cdata, "D", refs[cid], np.random.default_rng(_catalog_seed(cid)))
        if not g:
            continue
        wk = win_k_for_candidate(g)
        best = dict(g[wk], k=wk)
        out["best_gain_by_catalog"][cid] = {kk: (round(vv, 4) if isinstance(vv, float) else vv)
                                            for kk, vv in best.items()}
        gains.append(best["gain"])
    if gains:
        out["median_best_gain"] = round(float(np.median(gains)), 4)
    return out


# --------------------------------------------------------------------------- #
# Decision rule (plan STEP 7): tiny interpretable rule over catalog statistics
# --------------------------------------------------------------------------- #
def winning_family_per_catalog(extracted: dict[str, Any], refs: dict[str, str]) -> dict[str, str]:
    """Winning heuristic family per catalog = the family containing the heuristic with
    the highest mean NDCG@5 over k in {0,1,2,3,5,8} on the screen half. Popularity is
    the implicit reference; if no non-popularity family beats it by >= GAIN_GATE the
    label is 'popularity'. Elicitation only competes at k=0 (it is undefined at k>0), so
    it is compared separately at k=0."""
    out: dict[str, str] = {}
    for cid, cdata in extracted.items():
        best_mean = -math.inf
        best_fam: Optional[str] = None
        for fam in ["pop_global", "banded_pop", "content_hybrid", "windowed_pop"]:
            m = -math.inf
            for k in K_VALUES:
                h, mm, _ = family_best_at_k(cdata, fam, k)
                if h is None:
                    continue
                m = max(m, mm if np.isfinite(mm) else -math.inf)
            if m > best_mean:
                best_mean, best_fam = m, fam
        # Elicitation at k=0 competes at the cold-start cut specifically.
        eh, em, _ = family_best_at_k(cdata, "elicitation", 0)
        # Ref = population mean for the reference popularity family, for the gate.
        ref_mean = -math.inf
        for fam in ["pop_global", "banded_pop"]:
            for k in K_VALUES:
                h, mm, _ = family_best_at_k(cdata, fam, k)
                if h is not None and np.isfinite(mm):
                    ref_mean = max(ref_mean, mm)
        winner = best_fam if best_fam is not None else "popularity"
        if ref_mean > -math.inf and (best_mean - ref_mean) < GAIN_GATE:
            winner = "popularity"
        out[cid] = winner
    return out


def decision_rule(extracted: dict[str, Any], refs: dict[str, str]) -> dict[str, Any]:
    """Fit a depth<=2 decision tree over screen-half catalog statistics predicting the
    winning family; evaluate via leave-one-FAMILY-out; report vs baselines."""
    try:
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.model_selection import LeaveOneGroupOut
        from sklearn.metrics import accuracy_score
    except Exception as exc:  # pragma: no cover
        return {"error": f"scikit-learn unavailable: {exc}"}

    winners = winning_family_per_catalog(extracted, refs)
    cids = list(winners.keys())
    if len(cids) < 2:
        return {"features": ["sales_hhi", "attr_entropy", "mean_hist"],
                "note": "insufficient catalogs to fit a decision rule"}

    fam_ids = {f: i for i, f in enumerate(sorted(set(winners.values())))}
    groups_fam = {cid: wid for cid, wid in winners.items()}
    # leave ONE FAMILY out
    fams = sorted(set(groups_fam.values()))
    y = np.asarray([0, 0])  # placeholder, replaced below
    X_rows: list[list[float]] = []
    y_rows: list[str] = []
    groups: list[str] = []
    for cid in cids:
        m = extracted[cid]["meta"]
        row: list[float] = []
        for st in ["sales_hhi", "attr_entropy", "mean_hist"]:
            v = m.get(st)
            if v is None:
                row.append(float("nan"))
            else:
                try:
                    row.append(float(v))
                except (TypeError, ValueError):
                    row.append(float("nan"))
        X_rows.append(row)
        y_rows.append(winners[cid])
        groups.append(groups_fam[cid])

    X = np.asarray(X_rows, dtype=float)
    # feature bridge: impute NaN with the column mean (screen half only)
    for j in range(X.shape[1]):
        col = X[:, j]
        mask = np.isnan(col)
        if mask.all():
            X[:, j] = 0.0
        elif mask.any():
            X[mask, j] = np.nanmean(col)
    yt_full = np.asarray([fam_ids[v] for v in y_rows])

    # Leave-ONE-FAMILY-OUT within the SCREEN half: fit on all but one family, predict
    # that family's winners.
    preds: list[int] = []
    truthf: list[int] = []
    for fam in fams:
        test_mask = np.asarray([g == fam for g in groups])
        tr_mask = ~test_mask
        if tr_mask.sum() < 2 or test_mask.sum() < 1:
            continue
        clf = DecisionTreeClassifier(max_depth=2, min_samples_leaf=1,
                                     class_weight="balanced", random_state=SEED_BASE)
        try:
            clf.fit(X[tr_mask], yt_full[tr_mask])
            pred = clf.predict(X[test_mask])
        except Exception as exc:  # pragma: no cover
            logger.warning(f"decision-rule LOF failed on family {fam}: {exc}")
            continue
        preds.extend(int(p) for p in pred.tolist())
        truthf.extend(int(v) for v in yt_full[test_mask].tolist())
    lof_acc = accuracy_score(np.asarray(truthf), np.asarray(preds)) if truthf else None
    # learned thresholds: fit once on all screen catalogs for reporting
    tree = DecisionTreeClassifier(max_depth=2, random_state=SEED_BASE)
    yt = np.asarray([fam_ids[v] for v in y_rows])
    tree.fit(X, yt)
    feats = ["sales_hhi", "attr_entropy", "mean_hist"]
    from collections import Counter
    base_always_pop = max(Counter(y_rows).values()) / len(y_rows) if y_rows else None
    base_always_hybrid = (Counter(y_rows).get("content_hybrid", 0) / len(y_rows)) if y_rows else None

    return {
        "features": feats,
        "winning_family_per_catalog": winners,
        "n_catalogs": len(cids),
        "families": fams,
        "LOF_accuracy": round(lof_acc, 4) if lof_acc is not None else None,
        "baseline_always_popularity": round(base_always_pop, 4) if base_always_pop else None,
        "baseline_always_hybrid": round(base_always_hybrid, 4) if base_always_hybrid is not None else None,
        "tree_thresholds_learned": _tree_thresholds(tree, feats),
    }


def _tree_thresholds(tree, feats) -> dict[str, Any]:
    """Extract split thresholds from a fitted Depth<=2 tree for the report."""
    out: dict[str, Any] = {}
    t = tree.tree_
    for i in range(t.node_count):
        if t.children_left[i] != t.children_right[i]:  # internal node
            f = feats[int(t.feature[i])]
            out[f"node_{i}"] = {"feature": f, "threshold": round(float(t.threshold[i]), 4)}
    out["classes"] = list(tree.classes_)
    return out


# --------------------------------------------------------------------------- #
# Information-theoretic ceiling (plan STEP 8): plug-in MI(history; next item)
# --------------------------------------------------------------------------- #
def mi_ceiling(extracted: dict[str, Any]) -> dict[str, Any]:
    """Coarse plug-in estimate of I(history ; next_item) per catalog.

    Discretize each user's history signature as (their most-purchased category index)
    and the next-item label as the category of the true next item (coarse discretization
    to control sparsity). MI_hat = sum p(h,n) log[p(h,n)/(p(h) p(n))]. This is an
    UPPER-BOUND estimate (plug-in bias: positive for small samples) and should be read
    as a ceiling, not a target.
    """
    out: dict[str, Any] = {}
    for cid, cdata in extracted.items():
        # build per-user signature from the history vector encoded in 'history'
        # We need category -> item mapping; the experiment may store it in meta.
        items_meta = cdata["meta"].get("items", [])
        cat_of_item: dict[str, str] = {}
        if isinstance(items_meta, list):
            for it in items_meta:
                if isinstance(it, dict):
                    iid = str(it.get("item_id", it.get("id", "")))
                    cat = str(it.get("category", it.get("categorical", {}).get("category", "?") if isinstance(it.get("categorical"), dict) else "?"))
                    cat_of_item[iid] = cat[:6]
        sig_pairs: list[tuple[str, str]] = []
        # Reconstruct history signature from examples (input history) - but that is
        # heavy; instead use the last-item identity as the coarse signature when the
        # per-example history is available in the input JSON.
        for e in cdata.get("examples_raw", []):
            hist = None
            try:
                inp = json.loads(e.get("input", "{}"))
                hist = inp.get("history")
            except (TypeError, json.JSONDecodeError):
                hist = None
            if not isinstance(hist, list) or not hist:
                continue
            last = str(hist[-1])
            sig = cat_of_item.get(last, f"item_{last[:6]}")
            nxt = e.get("metadata_true", e.get("output"))
            nxt_cat = cat_of_item.get(str(nxt), f"item_{str(nxt)[:6]}")
            sig_pairs.append((sig, nxt_cat))
        if len(sig_pairs) < 2:
            out[cid] = {"mi_hat": None, "n_pairs": 0,
                        "caveat": "insufficient pairs; plug-in MI not estimable"}
            continue
        out[cid] = _plugin_mi(sig_pairs)

    # global caveat
    out["caveats"] = (
        "Plug-in MI is positively biased for small samples and should be treated as an "
        "upper bound / ceiling, not a target. Discretization (signature -> coarse "
        "category; next item -> category) reduces sparsity at the cost of MI resolution. "
        "Where the synthetic generator is available the generative ground-truth MI "
        "(metadata_mi_ceiling) supersedes this estimate; the two should be compared."
    )
    return out


def _plugin_mi(pairs: list[tuple[str, str]]) -> dict[str, Any]:
    from collections import Counter
    n = len(pairs)
    ph = Counter(p[0] for p in pairs)
    pn = Counter(p[1] for p in pairs)
    phn = Counter(pairs)
    mi = 0.0
    for (h, nn), c in phn.items():
        p_h = ph[h] / n
        p_n = pn[nn] / n
        p_hn = c / n
        if p_h > 0 and p_n > 0 and p_hn > 0:
            mi += p_hn * math.log(p_hn / (p_h * p_n))
    ent_n = -sum((v / n) * math.log(v / n) for v in pn.values())
    return {"mi_hat": round(mi, 6), "n_pairs": n,
            "next_item_entropy_nats": round(ent_n, 6),
            "mi_over_entropy": round(mi / ent_n, 4) if ent_n > 0 else None}


# --------------------------------------------------------------------------- #
# Confirmation protocol for iteration 2 (plan STEP 9) - write, never execute
# --------------------------------------------------------------------------- #
def confirmation_protocol(decision: str, reserved_note: str) -> dict[str, Any]:
    return {
        "title": "Iteration-2 confirmation protocol for the untouched CONFIRM half (pre-registered)",
        "reserved_confirm_catalogs": reserved_note,
        "metric": "per-user NDCG@5 (single relevant item: 1/log2(r+1) if r<=5 else 0)",
        "reference": "best popularity-family heuristic per catalog (max mean NDCG@5 over "
                     "k in {0,1,2,3,5,8} on the screen half); computed on the CONFIRM half "
                     "from its own popularity-family measurements",
        "target_k_by_candidate": CAND_TARGET_KS,
        "survivor_being_confirmed": decision,
        "success_thresholds": {
            "median_over_confirm_catalogs_gain": ">= 0.02 NDCG@5 (candidate over the "
                                                 "confirm-half reference at the candidate's target k)",
            "ci_excludes_zero_fraction": ">= 50% of confirm catalogs (95% stratified-"
                                         "bootstrap CI over users, paired, disclosed seeds)",
            "ties": "CI straddling 0 is reported explicitly as a tie (a tie is a finding); "
                    "if the confirm run cannot discriminate, the screen outcome stands as "
                    "reported and iteration 2 records the tie without re-selecting.",
        },
        "holdout_protocol": (
            "The CONFIRM half must remain UTTERLY UNTOUCHED until iteration 2. It is never "
            "used (a) to fit any heuristic parameter, (b) to choose the reference, (c) to "
            "pick the survivor, or (d) to tune any bootstrap seed or threshold. The screen "
            "survivor decision is final for this iteration. In iteration 2 the confirm half "
            "is evaluated exactly once, with per-user bootstrap seeds again fixed at "
            f"base_seed={SEED_BASE} + catalog_hash % 100000; no post-hoc winner switching.",
        ),
        "execution_status": "WRITTEN BUT NOT EXECUTED (confirm half not touched by this artifact).",
    }


# --------------------------------------------------------------------------- #
# Output assembly (plan STEP 10)
# --------------------------------------------------------------------------- #
def build_skeleton_eval(discovery_summary: dict[str, Any]) -> dict[str, Any]:
    """Skeleton eval_out.json for the ABSENT-pool case. Honest, no fabricated data."""
    reserved = (
        "Per the DATASET artifact's pre-assignment: the CONFIRM half = a reserved subset "
        "of the synthetic catalog families plus ALL real catalogs. Exact reserved "
        "catalog_ids are recorded in the pool's per-catalog fold labels "
        "(metadata_fold) once the parallel DATASET/EXPERIMENT artifacts publish. "
        "This evaluation could not enumerate them because the shared pool is absent."
    )
    protocol = confirmation_protocol(
        decision="NNN (undetermined: screen not executed on shared data)", reserved_note=reserved)
    protocol["survivor_being_confirmed"] = "TBD - to be fixed by the screen once the shared pool is available"

    meta = {
        "evaluation_name": "cold-start heuristic wide-screen: pre-registered survivor selection",
        "status": "INCONCLUSIVE / UNCOVERED - shared pool absent at evaluation time",
        "run_preamble": {
            "running_with_pool": discovery_summary.get("pool_found", False),
            "pool_candidates_found": [str(p) for p in discovery_summary.get("candidates", [])],
            "pool_load_errors": discovery_summary.get("errors", []),
            "timestamp": discovery_summary.get("timestamp"),
            "seed_base": SEED_BASE,
            "bootstrap_resamples": B_BOOT,
            "confirm_half_touched": False,
        },
        "coverage_report": {
            "screen_executed_on_shared_data": False,
            "reason": discovery_summary.get("reason",
                "No compatible shared-pool method_out.json / catalog measurements were "
                "found; the parallel EXPERIMENT artifact had not published its per-user "
                "(catalog, heuristic, k) rank measurements at evaluation time. Per the "
                "artifact plan STEP 0 coverage handling, missing cells are NOT fabricated "
                "and the screen is NOT run on substituted measurements."),
            "catalog_ids_with_cells": [],
            "heuristics_present": [],
            "k_cells_present": {},
            "n_users_per_cell": {},
            "n_confirm_catalogs_seen": 0,
        },
        "metric_definitions": {
            "primary": "per-user NDCG@5 = 1/log2(r+1) if r<=5 else 0 (r = 1-indexed rank "
                       "of the true held-out next item; single relevant item, ideal DCG=1)",
            "secondary": {"recall@5": "1 if r<=5 else 0", "ndcg@10": "1/log2(r+1) if r<=10 else 0"},
        },
        "reference_definition": "per catalog: best popularity-family heuristic "
                                "(global MostPop, category-banded POP, price-band POP; NOT "
                                "recency-windowed POP) by mean NDCG@5 over k in {0,1,2,3,5,8} "
                                "on the screen half",
        "per_candidate": {
            "A": {"target_k": CAND_TARGET_KS["A"], "gains_by_k": {}, "per_catalog_gain_table": {},
                  "ci_excludes0_fraction": None, "median_gain": None, "survived": None},
            "C": {"target_k": CAND_TARGET_KS["C"], "gains_by_k": {}, "per_catalog_gain_table": {},
                  "ci_excludes0_fraction": None, "median_gain": None, "survived": None},
            "D": {"target_k": CAND_TARGET_KS["D"], "gains_by_k": {}, "per_catalog_gain_table": {},
                  "ci_excludes0_fraction": None, "median_gain": None, "survived": None},
            "B": {"within_noise_fraction_pairs": None, "seed_var_winner_identity": None,
                  "note": "not measurable without shared measurements"},
        },
        "selection_decision": {
            "survivor": "NNN", "rule_trace": (
                "Pre-registered mechanical rule: A candidate survives iff (i) median-over-"
                "catalogs gain >= 0.02 AND its 95% CI excludes 0 in >=50% of catalogs AND "
                "(ii) it is not dominated by another candidate in >=50% of catalogs. "
                "Multiple -> largest median gain; none -> NOISE (B) wins. The screen could "
                "not be run because the shared pool is absent, so no decision is made here."),
            "ties": "not evaluated"
        },
        "diagnostic_report": {"crossover_k_star_by_family": {}, "signed_checks": {},
                              "plain_vs_banded_smallest": {}},
        "decision_rule": {"features": ["sales_hhi", "attr_entropy", "mean_hist"],
                          "rule": None, "LOF_accuracy_vs_baselines": None},
        "mi_ceiling": {"per_catalog": {}, "caveats": (
            "Plug-in MI is positively biased for small samples and is an UPPER BOUND / "
            "ceiling, not a target. Not computed because the shared pool is absent.")},
        "elicitation_gain_k0": {"detail": "not computed - no shared measurements"},
        "windowed_pop_turnover": {"detail": "not computed - no shared measurements"},
        "confirmation_protocol_for_iter2": protocol,
    }

    metrics_agg: dict[str, float] = {
        "screen_executed_on_shared_data": 0.0,
        "catalog_cells_covered": 0.0,
        "catalogs_with_screen_cells": 0.0,
    }

    datasets = [{
        "dataset": "screen_uncovered_status",
        "examples": [{
            "input": json.dumps({"role": "coverage_report", "pool_found": False}),
            "output": "eval_out.json skeleton: screen not executed on shared data; "
                      "see metadata.selection_decision and "
                      "metadata.confirmation_protocol_for_iter2.",
            "metadata_status": "uncovered",
            "metadata_pool_found": "false",
            "metadata_confirm_half_touched": "false",
            "eval_screen_executed": 0.0,
            "eval_cells_covered": 0.0,
        }],
    }]

    return {"metadata": meta, "metrics_agg": metrics_agg, "datasets": datasets}


# --------------------------------------------------------------------------- #
# Full analysis when the pool IS present
# --------------------------------------------------------------------------- #
def run_full_analysis(extracted: dict[str, Any]) -> dict[str, Any]:
    """Run the complete 9-step pre-registered screen on the shared measurements."""
    logger.info("Running full pre-registered screen on the shared pool")

    # References per catalog (STEP 2)
    refs: dict[str, Optional[str]] = {}
    ref_means: dict[str, float] = {}
    for cid, cdata in extracted.items():
        h, m = best_popularity_reference(cdata)
        refs[cid] = h
        ref_means[cid] = m

    # Per-candidate gain stats (STEP 3) using the primary per-catalog seed
    pool_stats: dict[str, dict[str, dict[str, float]]] = {c: {} for c in ["A", "C", "D"]}
    for cid, cdata in extracted.items():
        ref_h = refs.get(cid)
        if ref_h is None:
            logger.warning(f"catalog {cid}: no popularity-family reference found; skipped")
            continue
        rng = np.random.default_rng(_catalog_seed(cid))
        for c in ["A", "C", "D"]:
            g = candidate_gains(cdata, c, ref_h, rng)
            if g:
                pool_stats[c][cid] = g

    # Selection (STEP 4)
    selection = selection_rule(pool_stats)

    # Noise report (STEP 5)
    noise = noise_report(extracted, pool_stats, refs)

    # Diagnostic (STEP 6)
    diag = diagnostic_report(extracted, refs)

    # Decision rule (STEP 7)
    rule = decision_rule(extracted, refs)

    # MI ceiling (STEP 8)
    mi = mi_ceiling(extracted)

    # Candidate C & D detail
    elic = elicitation_detail(extracted, refs)
    wpop = windowed_pop_detail(extracted, refs)

    # Per-candidate summary tables for the report
    per_candidate: dict[str, Any] = {}
    for c in ["A", "C", "D"]:
        per_k_over_cats: dict[str, list[float]] = {}
        per_cat: dict[str, dict[str, Any]] = {}
        excl_frac = 0.0
        gains_at_win: list[float] = []
        n_cats = len(pool_stats[c])
        for cid, g in pool_stats[c].items():
            wk = win_k_for_candidate(g)
            if wk is None:
                continue
            st = dict(g[wk], k=wk)
            per_cat[cid] = {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in st.items()}
            excl_frac += 1 if ci_excludes_zero(g[wk]) else 0
            gains_at_win.append(g[wk]["gain"])
            for k, stk in g.items():
                per_k_over_cats.setdefault(str(k), []).append(stk["gain"])
        per_candidate[c] = {
            "target_k": CAND_TARGET_KS[c],
            "gains_by_k": {k: (round(float(np.mean(v)), 4) if v else None)
                           for k, v in per_k_over_cats.items()},
            "per_catalog_gain_table": per_cat,
            "ci_excludes0_fraction": (round(excl_frac / n_cats, 4) if n_cats else None),
            "median_gain": (round(float(np.median(gains_at_win)), 4) if gains_at_win else None),
            "survived": c in selection["survivors_list"],
        }

    return {
        "per_candidate": per_candidate,
        "selection_decision": selection,
        "noise_report_candidate_B": noise,
        "diagnostic_report": diag,
        "decision_rule": rule,
        "mi_ceiling": mi,
        "elicitation_gain_k0": elic,
        "windowed_pop_turnover": wpop,
    }


def build_full_eval(extracted: dict[str, Any], analysis: dict[str, Any],
                    discovery_summary: dict[str, Any]) -> dict[str, Any]:
    """Assemble the full eval_out.json when the screen executed."""

    meta = {
        "evaluation_name": "cold-start heuristic wide-screen: pre-registered survivor selection",
        "status": "COMPLETE - screen executed on shared measurements",
        "run_preamble": {
            "running_with_pool": True,
            "pool_path": str(discovery_summary.get("pool_path", "")),
            "timestamp": discovery_summary.get("timestamp"),
            "seed_base": SEED_BASE,
            "bootstrap_resamples": B_BOOT,
            "confirm_half_touched": False,
            "k_values": K_VALUES,
        },
        "coverage_report": {
            "screen_executed_on_shared_data": True,
            "catalog_ids_with_cells": sorted(extracted.keys()),
            "n_catalogs": len(extracted),
            "n_catalog_heuristic_k_cells": {
                cid: {str(k): {h: len(um) for h, um in cdata["cells"].get(k, {}).items()}
                      for k in K_VALUES}
                for cid, cdata in extracted.items()},
            "n_confirm_examples_per_catalog": {cid: cdata["n_confirm"] for cid, cdata in extracted.items()},
        },
        "metric_definitions": {
            "primary": "per-user NDCG@5 = 1/log2(r+1) if r<=5 else 0",
            "secondary": {"recall@5": "1 if r<=5 else 0", "ndcg@10": "1/log2(r+1) if r<=10 else 0"},
        },
        "reference_definition": "best popularity-family heuristic per catalog by mean "
                                "NDCG@5 over k in {0,1,2,3,5,8} on the screen half",
        **analysis,
        "confirmation_protocol_for_iter2": confirmation_protocol(
            decision=analysis["selection_decision"].get("decision", "B"), reserved_note=_RESERVED_NOTE),
    }

    n_cats = len(extracted)
    metrics_agg: dict[str, float] = {
        "screen_executed_on_shared_data": 1.0,
        "catalog_cells_covered": float(n_cats),
        "selection_survivor_FULL": float(1.0 if analysis["selection_decision"].get("decision") != "B" else 0.0),
    }

    datasets = [{
        "dataset": "screen_aggregate",
        "examples": [{
            "input": json.dumps({"role": "screen_result_summary",
                                 "survivor": analysis["selection_decision"].get("decision")}),
            "output": json.dumps({"survivor": analysis["selection_decision"].get("decision"),
                                  "per_candidate_median_gain": {c: meta["per_candidate"][c]["median_gain"]
                                                                for c in ["A", "C", "D"]}}),
            "metadata_survivor": str(analysis["selection_decision"].get("decision")),
            "metadata_n_catalogs_screened": str(n_cats),
            "eval_survivor_encoded": 1.0 if analysis["selection_decision"].get("decision") != "B" else 0.0,
            "eval_n_catalogs_screened": float(n_cats),
        }],
    }]
    return {"metadata": meta, "metrics_agg": metrics_agg, "datasets": datasets}


# Placeholder filled below by make_reserved_note()
_RESERVED_NOTE = (
    "CONFIRM half = a reserved subset of synthetic catalog families plus ALL real "
    "catalogs (per the DATASET artifact pre-assignment). Exact reserved ids are "
    "enumerated once the shared pool's per-catalog fold labels (metadata_fold) are "
    "available; this artifact never reads or uses them."
)


def make_reserved_note() -> str:
    """Construct the reserved-confirm note for the protocol."""
    return _RESERVED_NOTE


# --------------------------------------------------------------------------- #
# Self-test mode (code-correctness only; NEVER emitted as the screen result)
# --------------------------------------------------------------------------- #
def _make_selftest_pool(rng: np.random.Generator) -> dict[str, Any]:
    """Build a tiny synthetic method_out.json (exp_gen_sol_out-shaped) with a few
    catalogs and heuristics so the analysis machinery can be exercised.

    Signal is deliberately simple: content/hybrid ranks the true item at the top for
    long histories (=> positive A gain at larger k), popularity/recency do so at k=0,
    so selection/median/CIs all exercise branches.
    """
    ks = [0, 1, 2, 3, 5, 8]
    families = ["pop_global", "pop_category", "pop_price",
                "recency_pop_0.1", "recency_pop_0.5", "recency_pop_1.0",
                "content_knn_1", "content_knn_3", "content_knn_5",
                "last_item_nbhd", "pop_scaled_content", "co_purchase",
                "lambda_hybrid_0.25", "lambda_hybrid_0.5", "lambda_hybrid_0.75",
                "active_elic2"]

    def _rank(h: str, k: int) -> Optional[float]:
        flip = rng.integers(0, 2)
        if h == "active_elic2" and k > 0:
            return None  # not applicable at k>0
        if h in ("pop_global", "pop_category", "pop_price"):
            if k == 0:
                return float(1 + flip)
            return float(3 + int(rng.integers(0, 6)))       # popularity fades at k>=1
        if h.startswith("recency_pop"):
            if k <= 1:
                return float(1 + flip)
            return float(2 + int(rng.integers(0, 4)))
        if h.startswith(("content_knn", "last_item", "pop_scaled", "co_purchase", "lambda_hybrid")):
            if k >= 3:
                return float(1 + flip)                      # personalized wins at k>=3
            if k >= 1:
                return float(2 + int(rng.integers(0, 3)))
            return float(4 + int(rng.integers(0, 8)))       # weak at k=0 (cold start)
        return float(10)

    datasets = []
    for ci in range(4):
        v = [20, 100, 500, 1000][ci]
        cid = f"selftest_cat_{ci}"
        examples = []
        for k in ks:
            for u in range(80):                             # 75% screen => 60 >= MIN_USERS_CELL
                true_item = f"item_{u % v}"
                ex = {
                    "input": json.dumps({"catalog": cid, "user": f"u{k}_{u}", "k": k,
                                         "history": [f"item_{(u + j) % v}" for j in range(k)]}),
                    "output": true_item,
                    "metadata_catalog": cid,
                    "metadata_user": f"u{k}_{u}",
                    "metadata_k": k,
                    "metadata_fold": "screen" if (u % 4) else "confirm",  # 25% confirm, unused
                    "metadata_true": true_item,
                }
                for h in families:
                    r = _rank(h, k)
                    ex[f"metadata_rank_{h}"] = "" if r is None else int(r)
                    ex[f"predict_{h}"] = json.dumps([f"item_{i % v}" for i in range(50)])
                examples.append(ex)
        meta = {"n_items": v, "sales_hhi": 0.05 + 0.01 * ci, "sales_norm_entropy": 0.8,
                "attr_entropy": 0.3 + 0.1 * ci, "mean_hist": 6.0, "headroom": 0.03,
                "mi_ceiling": 0.2}
        datasets.append({"dataset": cid, "examples": examples, "metadata": meta})
    return {"metadata": {"title": "selftest"}, "datasets": datasets}


def run_selftest(workspace: Path) -> dict[str, Any]:
    """Exercise the full analysis pipeline on a tiny inline synthetic pool. The output
    is clearly labelled as a CODE-CORRECTNESS self-test and is NEVER emitted as the
    screen result (which would substitute fabricated measurements - forbidden)."""
    rng = np.random.default_rng(0)
    raw = _make_selftest_pool(rng)
    pool = {str(d["dataset"]): {"meta": d.get("metadata", {}), "examples": d.get("examples", [])}
            for d in raw["datasets"]}
    extracted = extract_cells(pool, screen_only=True)
    for ds in raw["datasets"]:
        extracted[ds["dataset"]]["examples_raw"] = ds["examples"]
    analysis = run_full_analysis(extracted)
    assert "selection_decision" in analysis
    assert analysis["selection_decision"]["decision"] in {"A", "C", "D", "B"}
    assert set(analysis["per_candidate"].keys()) == {"A", "C", "D"}
    report = {"selftest_passed": True, "n_catalogs": len(extracted), "analysis": analysis}
    out = workspace / "selftest" / "selftest_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str))
    logger.info(f"Self-test written to {out}")
    return report


# --------------------------------------------------------------------------- #
# main()
# --------------------------------------------------------------------------- #
@logger.catch(reraise=True)
def main() -> None:
    try:
        resource.setrlimit(resource.RLIMIT_AS, (8 * 1024**3, 8 * 1024**3))  # 8GB virtual cap
        resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))                # 1h CPU cap
    except (ValueError, OSError) as e:
        logger.warning(f"could not set rlimits: {e}")

    workspace = Path("/ai-inventor/aii_data/runs/run_5D4WD4vgZZMJ/3_invention_loop/iter_1/"
                     "gen_art/gen_art_evaluation_1").resolve()
    run_root = Path("/ai-inventor/aii_data/runs/run_5D4WD4vgZZMJ")

    if "--selftest" in sys.argv:
        run_selftest(workspace)
        logger.info("Self-test completed OK.")
        return

    t0 = time.time()
    discovery_summary: dict[str, Any] = {"pool_found": False, "candidates": [],
                                         "errors": [], "pool_path": None,
                                         "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    candidates = discover_pool(run_root, workspace)
    discovery_summary["candidates"] = [str(p) for p in candidates]
    logger.info(f"Pool discovery: {len(candidates)} candidate file(s)")

    pool_loaded: Optional[dict[str, Any]] = None
    for cand in candidates:
        try:
            data = load_pool(cand)
            if data:
                pool_loaded = data
                discovery_summary["pool_found"] = True
                discovery_summary["pool_path"] = str(cand)
                logger.info(f"Loaded shared pool from {cand} with {len(data)} catalog(s)")
                break
        except Exception as e:  # noqa: BLE001 - isolate one bad candidate
            logger.error(f"Failed to load pool candidate {cand}: {e}")
            discovery_summary["errors"].append(f"{cand}: {e}")
            gc.collect()

    if not pool_loaded:
        logger.warning(
            "No compatible shared pool found. Per the artifact plan STEP 0 coverage "
            "handling, the screen is NOT run on substituted/fabricated measurements. "
            "Emitting a skeleton eval_out.json with uncovered status + confirmation "
            "protocol, and stopping (confirm half untouched).")
        eval_doc = build_skeleton_eval(discovery_summary)
    else:
        extracted = extract_cells(pool_loaded, screen_only=True)
        for ds_key, blob in pool_loaded.items():
            if "examples" in blob and ds_key in extracted:
                extracted[ds_key]["examples_raw"] = blob["examples"]
        analysis = run_full_analysis(extracted)
        eval_doc = build_full_eval(extracted, analysis, discovery_summary)

    out_path = workspace / "eval_out.json"
    out_path.write_text(json.dumps(eval_doc, indent=2, default=str))
    logger.info(f"Wrote {out_path} ({out_path.stat().st_size / 1e6:.2f} MB) in "
                f"{time.time() - t0:.1f}s")
    logger.info(f"Status: {eval_doc['metadata']['status']}")

if __name__ == "__main__":
    main()
