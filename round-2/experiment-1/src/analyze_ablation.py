#!/usr/bin/env python3
"""Iteration-2 post-processing: headroom per phi, bootstrap CIs,
null-vs-extended table, null-replication comparison, diagnostic robustness.

Reads the evaluation payload cache (out/payloads/<catalog_id>.pkl: rank
arrays + diagnostics + per-user split positions) and the generation gate CSV
(out/marginals_match_gate.csv), plus the PUBLISHED iteration-1 aggregates
(gen_art_experiment_1/out/aggregates_by_catalog_heuristic_k.csv and
catalog_diagnostics.csv) for the null-replication gate.

Headline metric (mirrors iteration-1):
    headroom(cell)  = mean over k in {1,2,3,5,8} of
        [ NDCG@5(content_knn_3) - NDCG@5(best popularity-family heuristic) ]
    with best-pop chosen ONLY on screen-inner cells (first-j-predict-j+1
    discipline; confirm half never selects anything).
  * headroom     : evaluated on the screen-inner cells (iteration-1 exact);
  * headroom_full: same best-pop, evaluated on ALL valid cells (more power;
                   primary ablation metric).
  NDCG@5 = 1/log2(r+1) if r<=5 else 0; user-level stratified bootstrap
  (n_boot=1000, rng = default_rng(7 + seed*104729 + k)) within (catalog, k).

Outputs (into out/):
  ablation_headroom_by_phi.csv, bootstrap_headroom_CIs.csv,
  null_replication_comparison.csv, null_vs_extended.csv,
  diagnostic_robustness.csv, headline_summary.json
"""

from __future__ import annotations

import csv
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from loguru import logger

from heuristics import HEURISTIC_NAMES, POPULARITY_FAMILY_NAMES
from ablation_grid import parse_cell_id
from synthetic import K_VALUES

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "out"
PAYLOAD_DIR = OUT_DIR / "payloads"
GATE_CSV = OUT_DIR / "marginals_match_gate.csv"

IT1_EXPERIMENT = Path(
    "/ai-inventor/aii_data/runs/run_5D4WD4vgZZMJ/3_invention_loop/iter_1/"
    "gen_art/gen_art_experiment_1"
)
IT1_AGG = IT1_EXPERIMENT / "out" / "aggregates_by_catalog_heuristic_k.csv"
IT1_DIAG = IT1_EXPERIMENT / "out" / "catalog_diagnostics.csv"

INNER_KS = (1, 2, 3, 5, 8)
N_BOOT = 1000
BUILD_TOL = 0.02          # paper's "+0.02 don't-build" boundary
OVERTAKE_HEURISTICS = ("content_knn_3", "lambda_hybrid_0.25",
                       "lambda_hybrid_0.5", "lambda_hybrid_0.75")


def _ndcg5(r: np.ndarray) -> np.ndarray:
    return np.where(r <= 5, 1.0 / np.log2(r + 1.0), 0.0)


def load_payloads(pldir: Path = PAYLOAD_DIR) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for p in sorted(pldir.glob("*.pkl")):
        with open(p, "rb") as f:
            out[p.stem] = pickle.load(f)
    return out


def load_gate_rows() -> dict[str, dict]:
    rows: dict[str, dict] = {}
    if not GATE_CSV.exists():
        return rows
    with open(GATE_CSV, newline="") as f:
        for r in csv.DictReader(f):
            rows[r["catalog_id"]] = r
    return rows


# ---------------------------------------------------------------------------
# Per-catalog cell metrics
# ---------------------------------------------------------------------------

def _cell_ndcg5_by_heur(payload: dict, k: int) -> dict[str, np.ndarray]:
    """Per-user NDCG@5 (full cell) for every heuristic defined at k."""
    arrs = payload["arrays"]
    out: dict[str, np.ndarray] = {}
    for h in HEURISTIC_NAMES:
        key = f"{k}/{h}/r"
        if key in arrs:
            out[h] = _ndcg5(arrs[key].astype(np.float64))
    return out


def catalog_metrics(payload: dict) -> dict:
    """All per-(catalog, k) metrics + headroom variants for one payload."""
    arrs = payload["arrays"]
    extra = payload.get("_extra") or {}
    split_pos = extra.get("user_split_pos")
    res: dict = {"cells": {}}
    for k in K_VALUES:
        sk = str(k)
        if f"{sk}/n" not in arrs:
            continue
        users = arrs[f"{sk}/user"].astype(np.int64)
        nd5 = _cell_ndcg5_by_heur(payload, k)
        inner = None
        if split_pos is not None and k >= 1:
            inner = (k + 1) <= split_pos[users]
        mean5 = {h: float(v.mean()) for h, v in nd5.items()}
        res["cells"][k] = {
            "users": users,
            "nd5": nd5,
            "inner": inner,
            "mean5": mean5,
            "n": int(len(users)),
        }
    # ---- headroom (iteration-1 exact) ----
    diag = payload.get("diagnostics") or {}
    res["headroom_payload"] = diag.get("headroom")
    # recompute from per-user ranks (screen-inner cells; best pop on inner)
    d_inner: list[float] = []
    d_full: list[float] = []
    for k in INNER_KS:
        c = res["cells"].get(k)
        if c is None or c["inner"] is None:
            continue
        inner_idx = np.where(c["inner"])[0]
        if len(inner_idx) == 0:
            continue
        pop_names = [h for h in POPULARITY_FAMILY_NAMES if h in c["nd5"]]
        if not pop_names or "content_knn_3" not in c["nd5"]:
            continue
        best_pop = max(pop_names, key=lambda h: float(c["nd5"][h][inner_idx].mean()))
        d_inner.append(float(
            (c["nd5"]["content_knn_3"][inner_idx]
             - c["nd5"][best_pop][inner_idx]).mean()))
        d_full.append(float(
            (c["nd5"]["content_knn_3"] - c["nd5"][best_pop]).mean()))
    res["headroom"] = float(np.mean(d_inner)) if d_inner else float("nan")
    res["headroom_full"] = float(np.mean(d_full)) if d_full else float("nan")
    res["inner_cells_used"] = len(d_inner)
    return res


# ---------------------------------------------------------------------------
# Bootstrap (user-level stratified within (catalog, k))
# ---------------------------------------------------------------------------

def _boot_rng(seed: int, k: int) -> np.random.Generator:
    return np.random.default_rng(7 + seed * 104729 + k)


def bootstrap_metrics(metrics: dict, seed_param: int) -> dict:
    """Bootstrap CIs on headroom / headroom_full / per-(k,h) NDCG@5 means.
    Returns {headroom: (m, lo, hi), headroom_full: (m, lo, hi),
             per_k: {k: {h: (m, lo, hi)}}, bootstrap_n: B}."""
    B = N_BOOT
    res: dict = {"headroom": (float("nan"), float("nan"), float("nan")),
                 "headroom_full": (float("nan"), float("nan"), float("nan"))}
    Bb = 100
    per_k: dict[int, dict[str, tuple[float, float, float]]] = {}
    for k in K_VALUES:
        c = metrics["cells"].get(k)
        if c is None:
            continue
        rng = _boot_rng(seed_param, k)
        nv = c["n"]
        # per-heuristic ndcg5 bootstrap (vectorized in blocks of Bb); one
        # rng stream per (seed, k) -> deterministic across re-runs
        h_stats: dict[str, tuple[float, float, float]] = {}
        for h, v in c["nd5"].items():
            means = np.empty(B, dtype=np.float64)
            for b0 in range(0, B, Bb):
                b1 = min(b0 + Bb, B)
                idx = rng.integers(0, nv, size=(b1 - b0, nv))
                means[b0:b1] = v[idx].mean(axis=1)
            hi = float(np.percentile(means, 97.5))
            lo = float(np.percentile(means, 2.5))
            h_stats[h] = (float(v.mean()), lo, hi)
        per_k[k] = h_stats
        if k not in INNER_KS or "content_knn_3" not in c["nd5"]:
            continue
        pop_names = [h for h in POPULARITY_FAMILY_NAMES if h in c["nd5"]]
        if not pop_names:
            continue
        # headroom_full bootstrap: mean over k of mean(cknn3 - best_pop)
        best_pop_name = max(
            pop_names,
            key=lambda h: _inner_mean(c["nd5"][h], c))
        d_full = c["nd5"]["content_knn_3"] - c["nd5"][best_pop_name]
        means = np.empty(B, dtype=np.float64)
        for b0 in range(0, B, Bb):
            b1 = min(b0 + Bb, B)
            idx = rng.integers(0, nv, size=(b1 - b0, nv))
            means[b0:b1] = d_full[idx].mean(axis=1)
        d_final = means  # per-b mean at this k; combined below
        res.setdefault("_dists", {})[k] = d_final
        res.setdefault("_best_pop", {})[k] = best_pop_name
        # headroom (inner) bootstrap on the inner subset
        if c["inner"] is not None and c["inner"].any():
            inner_idx = np.where(c["inner"])[0]
            n_in = len(inner_idx)
            d_in = c["nd5"]["content_knn_3"][inner_idx] \
                - c["nd5"][best_pop_name][inner_idx]
            means_i = np.empty(B, dtype=np.float64)
            for b0 in range(0, B, Bb):
                b1 = min(b0 + Bb, B)
                idx = rng.integers(0, n_in, size=(b1 - b0, n_in))
                means_i[b0:b1] = d_in[idx].mean(axis=1)
            res.setdefault("_inner_dists", {})[k] = means_i
    # aggregate headroom/headroom_full across k (mean of per-k boot means)
    dists = res.get("_dists") or {}
    if dists:
        joined = np.mean(np.stack([dists[k] for k in dists]), axis=0)
        res["headroom_full"] = (float(np.mean(joined)),
                                float(np.percentile(joined, 2.5)),
                                float(np.percentile(joined, 97.5)))
    inner_dists = res.get("_inner_dists") or {}
    if inner_dists:
        joined = np.mean(np.stack(list(inner_dists.values())), axis=0)
        res["headroom"] = (float(np.mean(joined)),
                           float(np.percentile(joined, 2.5)),
                           float(np.percentile(joined, 97.5)))
    res["per_k"] = per_k
    res["bootstrap_n"] = B
    return res


def _inner_mean(v: np.ndarray, c: dict) -> float:
    if c["inner"] is not None and c["inner"].any():
        return float(v[c["inner"]].mean())
    return float(v.mean())


# ---------------------------------------------------------------------------
# Analyze
# ---------------------------------------------------------------------------

def analyze(payloads: dict, workers: int = 1) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gate_rows = load_gate_rows()
    cell_ids = [cid for cid in payloads if parse_cell_id(cid) is not None]

    # ---------- per-catalog metrics + bootstrap ----------
    logger.info("computing per-catalog metrics + bootstrap ...")
    rows_ab: list[dict] = []
    rows_boot: list[dict] = []
    meta_by_id: dict[str, dict] = {}
    for cid in sorted(payloads):
        if parse_cell_id(cid) is None:
            continue  # main-grid cells only
        cfg = parse_cell_id(cid)
        p = payloads[cid]
        extra = p.get("_extra") or {}
        fp = json.loads(extra.get("family_params_json", "{}"))
        phi_eff = float(fp.get("phi_effective", cfg["phi"]))
        phi_capped = bool(fp.get("phi_capped", False))
        m = catalog_metrics(p)
        seed_param = cfg["seed"]
        b = bootstrap_metrics(m, seed_param)
        g = gate_rows.get(cid, {})
        # winner per k (full cell, all heuristics defined at k)
        winners: list[str] = []
        for k in K_VALUES:
            ck = m["cells"].get(k)
            if ck is None or not ck["mean5"]:
                winners.append(f"{k}:na")
            else:
                w = max(ck["mean5"], key=ck["mean5"].get)
                winners.append(f"{k}:{w}")
        # overtake (CI-based) per k among OVERTAKE_HEURISTICS vs pop_global
        overtake_any = False
        overtake_detail: list[str] = []
        for k in INNER_KS:
            pk = b.get("per_k", {}).get(k)
            if not pk or "pop_global" not in pk:
                continue
            pop = pk["pop_global"]
            for h in OVERTAKE_HEURISTICS:
                if h not in pk:
                    continue
                lo = pk[h][1] - pop[2]
                if lo > 0:
                    overtake_any = True
                    overtake_detail.append(f"k{k}:{h}")
        hf = b["headroom_full"]
        hd = b["headroom"]
        rows_ab.append({
            "catalog_id": cid,
            "n_items": cfg["n_items"], "entropy": cfg["entropy"],
            "zipf_s": cfg["zipf_s"], "seed": cfg["seed"],
            "phi_nominal": cfg["phi"], "phi_effective": round(phi_eff, 4),
            "phi_capped": phi_capped,
            "headroom": _f(hd[0]), "headroom_lo": _f(hd[1]),
            "headroom_hi": _f(hd[2]),
            "headroom_full": _f(hf[0]), "headroom_full_lo": _f(hf[1]),
            "headroom_full_hi": _f(hf[2]),
            "delta_vs_phi0": "", "delta_lo": "", "delta_hi": "",
            "delta_dominant": "",
            "winner_per_k": ";".join(winners),
            "overtake_any": overtake_any,
            "overtake_detail": "+".join(overtake_detail),
            "gate_KS": _f(g.get("gate_KS", "")), "gate_max_abs_dev": _f(g.get("gate_max_abs_dev", "")),
            "dHHI": _f(g.get("dHHI", "")), "dNormEntropy": _f(g.get("dNormEntropy", "")),
            "inner_cells_used": m["inner_cells_used"],
        })
        meta_by_id[cid] = {"cfg": cfg, "phi_eff": phi_eff, "b": b, "m": m,
                           "diag": p.get("diagnostics") or {}}
        for k in K_VALUES:
            ck = m["cells"].get(k)
            pk = b.get("per_k", {}).get(k, {})
            if not ck:
                continue
            for h in HEURISTIC_NAMES:
                if h not in ck["nd5"]:
                    continue
                v = pk.get(h)
                rows_boot.append({
                    "catalog_id": cid, "k": k, "heuristic": h,
                    "ndcg5": _f(float(ck["nd5"][h].mean())),
                    "ci_lo": _f(v[1]) if v else "", "ci_hi": _f(v[2]) if v else "",
                })
    _write_ablation_headroom(rows_ab)
    _write_bootstrap(rows_boot)

    # ---------- delta vs phi0 (paired by (config, seed)) ----------
    phi0_idx = {}
    for cid, meta in meta_by_id.items():
        cfg = meta["cfg"]
        if cfg["phi"] == 0.0:
            phi0_idx[(cfg["n_items"], cfg["entropy"], cfg["zipf_s"], cfg["seed"])] = cid
    for row in rows_ab:
        key = (row["n_items"], row["entropy"], row["zipf_s"], row["seed"])
        b0 = phi0_idx.get(key)
        if b0 is None or row["phi_nominal"] == 0.0:
            continue
        hf0 = meta_by_id[b0]["b"]["headroom_full"]
        hf1 = meta_by_id[row["catalog_id"]]["b"]["headroom_full"]
        lo = hf1[1] - hf0[2]
        hi = hf1[2] - hf0[1]
        row["delta_vs_phi0"] = _f(hf1[0] - hf0[0])
        row["delta_lo"] = _f(lo)
        row["delta_hi"] = _f(hi)
        row["delta_dominant"] = str(lo > 0).lower()
    _write_ablation_headroom(rows_ab)

    # ---------- null-vs-extended table ----------
    _write_null_vs_extended(rows_ab)

    # ---------- null replication comparison ----------
    _write_null_replication_comparison(payloads)

    # ---------- diagnostic robustness ----------
    _write_diagnostic_robustness(rows_ab, meta_by_id)

    # ---------- headline summary ----------
    _write_headline_summary(rows_ab, meta_by_id)

    logger.info("analyze_ablation finished")


def _f(v) -> str:
    if v is None or v == "" or (isinstance(v, float) and np.isnan(v)):
        return ""
    return f"{float(v):.6f}"


def _write_ablation_headroom(rows: list[dict]) -> None:
    path = OUT_DIR / "ablation_headroom_by_phi.csv"
    fields = ["catalog_id", "n_items", "entropy", "zipf_s", "seed",
              "phi_nominal", "phi_effective", "phi_capped",
              "headroom", "headroom_lo", "headroom_hi",
              "headroom_full", "headroom_full_lo", "headroom_full_hi",
              "delta_vs_phi0", "delta_lo", "delta_hi", "delta_dominant",
              "winner_per_k", "overtake_any", "overtake_detail",
              "gate_KS", "gate_max_abs_dev", "dHHI", "dNormEntropy",
              "inner_cells_used"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    logger.info(f"wrote {path.name}: {len(rows)} rows")


def _write_bootstrap(rows: list[dict]) -> None:
    path = OUT_DIR / "bootstrap_headroom_CIs.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["catalog_id", "k", "heuristic",
                                          "ndcg5", "ci_lo", "ci_hi"])
        w.writeheader()
        w.writerows(rows)
    logger.info(f"wrote {path.name}: {len(rows)} rows")


def _region(grid_cfg: dict) -> tuple:
    return (grid_cfg["n_items"], grid_cfg["entropy"], grid_cfg["zipf_s"])


def _write_null_vs_extended(rows_ab: list[dict]) -> None:
    from collections import defaultdict
    regions: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows_ab:
        regions[(_region(r))].append(r)
    out_rows: list[dict] = []
    for reg, rows in sorted(regions.items()):
        size, entropy, zipf = reg
        row: dict = {"n_items": size, "entropy": entropy, "zipf_s": zipf}
        for phi in (0.0, 0.25, 0.5, 1.0):
            sel = [r for r in rows if r["phi_nominal"] == phi]
            if not sel:
                continue
            hf_all = np.array([float(r["headroom_full"]) for r in sel
                               if r["headroom_full"] != ""])
            hd_all = np.array([float(r["headroom"]) for r in sel
                               if r["headroom"] != ""])
            row[f"n_phi{phi}"] = len(sel)
            if len(hf_all):
                row[f"headroom_full_phi{phi}"] = _f(hf_all.mean())
                row[f"headroom_phi{phi}"] = _f(hd_all.mean()) if len(hd_all) else ""
            row[f"phi{phi}_mean_phi_eff"] = _f(np.mean([
                float(r["phi_effective"]) for r in sel if r["phi_effective"] != ""]))
        # winners at phi=0 and phi=1 (k=5, full cell)
        def winner_at_phi(phi: float) -> str:
            sel = [r for r in rows if r["phi_nominal"] == phi]
            w5 = [r["winner_per_k"] for r in sel if r["winner_per_k"]]
            if not w5:
                return ""
            names = []
            for wk in w5:
                parts = dict(seg.split(":", 1) for seg in wk.split(";"))
                names.append(parts.get("5", "na"))
            from collections import Counter
            return Counter(names).most_common(1)[0][0]
        row["winner_k5_phi0"] = winner_at_phi(0.0)
        row["winner_k5_phi1"] = winner_at_phi(1.0)
        # null 'never overtakes' holds at phi>0?
        any_overtake = any(r["overtake_any"] for r in rows
                           if float(r["phi_nominal"]) > 0.0)
        row["content_or_hybrid_overtakes_at_phi_gt0"] = any_overtake
        # min effective phi at which overtake first happens in this region
        cands = sorted((float(r["phi_effective"]), float(r["phi_nominal"]))
                       for r in rows if r["overtake_any"])
        row["min_phi_eff_overtake"] = _f(cands[0][0]) if cands else "none"
        row["min_phi_nom_overtake"] = _f(cands[0][1]) if cands else "none"
        # delta headroom phi=1 vs phi=0 (mean over seeds + dominant count)
        sel1 = [r for r in rows if r["phi_nominal"] == 1.0
                and r["delta_vs_phi0"] != ""]
        deltas = np.array([float(r["delta_vs_phi0"]) for r in sel1])
        doms = [r for r in sel1 if r["delta_dominant"] == "true"]
        row["delta_hf_phi1_minus_phi0"] = _f(deltas.mean()) if len(deltas) else ""
        row["n_cells_delta_dominant"] = len(doms) if sel1 else ""
        out_rows.append(row)
    # Not every region row populates every (conditional) phi column, so the
    # fieldnames are the UNION of all keys across rows, in first-seen order.
    fieldnames: list[str] = []
    for r in out_rows:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)
    path = OUT_DIR / "null_vs_extended.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)
    logger.info(f"wrote {path.name}: {len(out_rows)} region rows")


def _write_null_replication_comparison(payloads: dict) -> None:
    import pandas as pd
    if not IT1_AGG.exists():
        logger.warning("published iteration-1 aggregates NOT FOUND; "
                       "null-replication comparison skipped")
        return
    pub = pd.read_csv(IT1_AGG)
    pub_diag = pd.read_csv(IT1_DIAG)
    pub_diag = pub_diag.set_index("catalog_id")
    # The published iteration-1 aggregates print NDCG@5 with 5 decimal
    # places (e.g. 0.09050).  A 5-dp print of a value in [0,1] carries a
    # quantization error of up to 0.5e-5 = 5e-6, so the STRICT 1e-6 gate is
    # unattainable against the published file and the effective gate is
    # 5e-6 + epsilon.  Both are reported; the summary gate uses pass_print5
    # AND the 1e-3 headroom tolerance (plan: "NDCG@5 within 1e-6, headroom
    # within 1e-3" <= both interpreted against the published representation;
    # measured: 100% of cells satisfy published == round(mine, 5), i.e. the
    # recomputation is bit-consistent with iteration-1's up to print format).
    PRINT5_TOL = 5e-6 + 1e-9
    out_rows: list[dict] = []
    n_cell_checked = 0
    n_cell_pass_1e6 = 0
    n_cell_pass_print5 = 0
    n_head_fail_1e3 = 0
    n_head_checked = 0
    for cid, p in sorted(payloads.items()):
        if parse_cell_id(cid) is not None:
            continue  # null-replication catalogs only
        arrs = p["arrays"]
        for k in K_VALUES:
            sk = str(k)
            if f"{sk}/n" not in arrs:
                continue
            for h in HEURISTIC_NAMES:
                rk = arrs.get(f"{sk}/{h}/r")
                if rk is None:
                    continue
                mine = float(_ndcg5(rk.astype(np.float64)).mean())
                row_pub = pub[(pub.catalog_id == cid) & (pub.heuristic == h)
                              & (pub.k == k)]
                if len(row_pub) == 0:
                    continue
                pubv = float(row_pub["ndcg@5"].iloc[0])
                delta = mine - pubv
                pass_1e6 = abs(delta) <= 1e-6
                pass_print5 = abs(delta) <= PRINT5_TOL
                n_cell_checked += 1
                n_cell_pass_1e6 += int(pass_1e6)
                n_cell_pass_print5 += int(pass_print5)
                out_rows.append({
                    "catalog_id": cid, "heuristic": h, "k": k,
                    "published_ndcg5": f"{pubv:.7f}", "mine_ndcg5": f"{mine:.7f}",
                    "delta": f"{delta:.3e}", "pass_1e6": pass_1e6,
                    "pass_print5": pass_print5,
                })
        # headroom comparison
        if cid in pub_diag.index:
            pub_h = float(pub_diag.loc[cid, "headroom"])
            mine_h = float((p.get("diagnostics") or {}).get("headroom") or float("nan"))
            delta_h = mine_h - pub_h if np.isfinite(mine_h) else float("nan")
            passed_h = bool(np.isfinite(delta_h) and abs(delta_h) <= 1e-3)
            n_head_checked += 1
            if not passed_h:
                n_head_fail_1e3 += 1
            out_rows.append({
                "catalog_id": cid, "heuristic": "HEADROOM", "k": "",
                "published_ndcg5": f"{pub_h:.7f}",
                "mine_ndcg5": f"{mine_h:.7f}" if np.isfinite(mine_h) else "",
                "delta": f"{delta_h:.3e}" if np.isfinite(delta_h) else "",
                "pass_1e6": passed_h, "pass_print5": passed_h,
            })
    path = OUT_DIR / "null_replication_comparison.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["catalog_id", "heuristic", "k",
                                          "published_ndcg5", "mine_ndcg5",
                                          "delta", "pass_1e6", "pass_print5"])
        w.writeheader()
        w.writerows(out_rows)
    summary = {
        "cells_checked": n_cell_checked,
        "cells_pass_1e6": n_cell_pass_1e6,
        "cells_pass_print5": n_cell_pass_print5,
        "headroom_checked": n_head_checked,
        "headroom_failed_1e3": n_head_fail_1e3,
        "basis": (
            "published iteration-1 aggregates print NDCG@5 to 5dp "
            "(max quantization error 5e-6); pass_1e6 is the strict "
            "tolerance and is expected to reflect print quantization, "
            "pass_print5 (<=5e-6+1e-9) is the effective gate; headroom "
            "gate is |delta| <= 1e-3"),
        "gate": bool(n_cell_checked
                     and n_cell_pass_print5 == n_cell_checked
                     and n_head_fail_1e3 == 0),
    }
    (OUT_DIR / "null_replication_gate.json").write_text(
        json.dumps(summary, indent=1))
    logger.info(f"null-replication comparison: {summary}")


def _write_diagnostic_robustness(rows_ab: list[dict],
                                 meta_by_id: dict[str, dict]) -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    import pandas as pd

    feat_rows: list[dict] = []
    for r in rows_ab:
        cid = r["catalog_id"]
        p = meta_by_id[cid]
        diag = p.get("diag")
        if not diag:
            continue
        feat_rows.append({
            "catalog_id": cid,
            "n_items": r["n_items"], "entropy": r["entropy"],
            "zipf_s": r["zipf_s"], "seed": r["seed"],
            "phi_effective": float(r["phi_effective"]) if r["phi_effective"] != "" else float(r["phi_nominal"]),
            "sales_hhi": float(diag.get("sales_hhi", float("nan"))),
            "sales_norm_entropy": float(diag.get("sales_norm_entropy", float("nan"))),
            "attr_entropy": float(diag.get("attr_entropy", float("nan"))),
            "headroom_full": float(r["headroom_full"]) if r["headroom_full"] != "" else float("nan"),
            "label_build": float(r["headroom_full"] != "" and float(r["headroom_full"]) > BUILD_TOL),
        })
    df = pd.DataFrame(feat_rows)
    if df.empty:
        logger.warning("no main-grid cells with diagnostics; robustness skipped")
        return
    feat_cols = ["sales_hhi", "sales_norm_entropy", "attr_entropy"]
    X0 = df.loc[df.phi_effective <= 1e-9, feat_cols]
    y0 = df.loc[df.phi_effective <= 1e-9, "label_build"]
    scaler = StandardScaler().fit(X0)
    clf = None
    if len(np.unique(y0)) >= 2:
        clf = LogisticRegression(C=1.0, max_iter=5000, random_state=0)
        clf.fit(scaler.transform(X0), y0)
    elif len(y0) > 0:
        logger.warning("phi=0 subset has a single build label; LR fit skipped "
                       "(acc_lr reported empty; rule-based columns still valid)")
    # single-feature HHI threshold rule fitted on phi0 (max accuracy)
    tau = _best_hhi_threshold(df, X0, y0, feat_cols)

    # baselines
    out_rows: list[dict] = []
    phi_levels = sorted(df.phi_effective.unique())
    hhi_terc = df.loc[df.phi_effective <= 1e-9, "sales_hhi"].quantile(2 / 3)
    first_fail_acc: float | None = None
    acc0 = None
    for phi in phi_levels:
        sub = df[np.isclose(df.phi_effective, phi, atol=1e-9)]
        if len(sub) == 0:
            continue
        y = sub.label_build.values
        if clf is not None:
            pred_lr = clf.predict(scaler.transform(sub[feat_cols]))
        else:
            pred_lr = np.zeros(len(sub))
        pred_pop = np.zeros(len(sub))
        pred_build = np.ones(len(sub))
        pred_size = (sub.n_items.values >= 100).astype(float)
        pred_hhi = (sub.sales_hhi.values >= tau).astype(float)
        acc = {
            "phi_effective": round(float(phi), 4),
            "n_cells": len(sub),
            "n_build_label": int(y.sum()),
            "acc_lr": (_f(np.mean(pred_lr == y)) if clf is not None else ""),
            "acc_always_pop": _f(np.mean(pred_pop == y)),
            "acc_always_build": _f(np.mean(pred_build == y)),
            "acc_size_rule": _f(np.mean(pred_size == y)),
            "acc_hhi_rule": _f(np.mean(pred_hhi == y)),
        }
        # concentrated (top-third HHI) false-negative rate
        conc = sub[sub.sales_hhi >= hhi_terc]
        if len(conc) and clf is not None:
            fn = int(((pred_lr[sub.sales_hhi.values >= hhi_terc] == 0)
                      & (y[sub.sales_hhi.values >= hhi_terc] == 1)).sum())
            acc["fn_lr_concentrated"] = _f(fn / len(conc))
        else:
            acc["fn_lr_concentrated"] = ""
        out_rows.append(acc)
        if acc0 is None:
            acc0 = float(acc["acc_lr"]) if acc["acc_lr"] != "" else float("nan")
        if first_fail_acc is None and acc["acc_lr"] != "" \
                and abs(float(acc["acc_lr"]) - acc0) > 0.05:
            first_fail_acc = float(phi)
    if first_fail_acc:
        for r in out_rows:
            if abs(r["phi_effective"] - round(first_fail_acc, 4)) < 1e-9:
                r["first_failure_phi"] = "THIS_ROW"
            else:
                r.setdefault("first_failure_phi", "")
    path = OUT_DIR / "diagnostic_robustness.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "phi_effective", "n_cells", "n_build_label",
            "acc_lr", "acc_always_pop", "acc_always_build",
            "acc_size_rule", "acc_hhi_rule", "fn_lr_concentrated",
            "first_failure_phi"])
        w.writeheader()
        w.writerows(out_rows)
    logger.info(f"wrote {path.name}: {len(out_rows)} phi levels "
                f"(first failure at phi_eff={first_fail_acc})")


def _best_hhi_threshold(df, X0, y0, feat_cols) -> float:
    """Threshold on sales_hhi maximizing phi0 accuracy (disclosed rule)."""
    vals = np.sort(df.loc[df.phi_effective <= 1e-9, "sales_hhi"].unique())
    if len(vals) == 0 or len(y0) == 0:
        return 0.5  # degenerate phi0 subset: default threshold, documented
    best_t, best_a = vals[0], -1.0
    for t in vals:
        pred = (df.loc[df.phi_effective <= 1e-9, "sales_hhi"].values >= t)
        a = np.mean(pred == y0.values)
        if a > best_a:
            best_a, best_t = a, t
    return float(best_t)


def _write_headline_summary(rows_ab: list[dict],
                            meta_by_id: dict[str, dict]) -> None:
    import numpy as _np
    phi_levels = sorted({round(float(r["phi_nominal"]), 2) for r in rows_ab})
    summary: dict = {
        "n_main_grid_cells": len(rows_ab),
        "headroom_full_mean_by_nominal_phi": {},
        "headroom_inner_mean_by_nominal_phi": {},
        "n_capped_cells": sum(1 for r in rows_ab if r["phi_capped"]),
        "capped_cells": [r["catalog_id"] for r in rows_ab if r["phi_capped"]][:40],
        "n_overtake_cells": sum(1 for r in rows_ab if r["overtake_any"]),
        "overtake_cells": [r["catalog_id"] for r in rows_ab
                           if r["overtake_any"]][:40],
        "mean_phi_effective_by_nominal": {},
        "delta_dominant_cells_phi1": sum(
            1 for r in rows_ab
            if r["phi_nominal"] == 1.0 and r["delta_dominant"] == "true"),
        "headroom_definition": (
            "headroom = mean over k in {1,2,3,5,8} of [NDCG@5(content_knn_3) "
            "- NDCG@5(best popularity-family)]; best pop chosen on screen-inner "
            "cells (iteration-1 discipline); headroom_full evaluated on ALL "
            "valid users, headroom on screen-inner users only"),
    }
    for phi in phi_levels:
        sel = [r for r in rows_ab if r["phi_nominal"] == phi]
        hf = [float(r["headroom_full"]) for r in sel if r["headroom_full"] != ""]
        hd = [float(r["headroom"]) for r in sel if r["headroom"] != ""]
        pe = [float(r["phi_effective"]) for r in sel if r["phi_effective"] != ""]
        summary["headroom_full_mean_by_nominal_phi"][str(phi)] = (
            round(float(_np.mean(hf)), 5)) if hf else None
        summary["headroom_inner_mean_by_nominal_phi"][str(phi)] = (
            round(float(_np.mean(hd)), 5)) if hd else None
        summary["mean_phi_effective_by_nominal"][str(phi)] = (
            round(float(_np.mean(pe)), 4)) if pe else None
    (OUT_DIR / "headline_summary.json").write_text(
        json.dumps(summary, indent=1))
    logger.info("wrote headline_summary.json")


@logger.catch(reraise=True)
def main() -> None:
    import argparse
    import os
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload-dir", type=str, default=str(PAYLOAD_DIR))
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = "1"
    payloads = load_payloads(Path(args.payload_dir))
    logger.info(f"loaded {len(payloads)} payloads")
    analyze(payloads, workers=args.workers)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stdout, level="INFO",
               format="{time:HH:mm:ss}|{level:<7}|{message}")
    main()