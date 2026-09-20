#!/usr/bin/env python3
"""Iteration-2 checks C9-C12 (in addition to the ported C1-C8).

C9  marginals-match gate: for a (config, seed) cell the phi>0 catalog's
    screen-half marginal stays within the GATE_TOL of its phi=0 catalog;
C10 tag-encoding self-check: latent tag d present on item v iff z_v[d] > 0
    (column-wise agreement == 1.0 at design) while distractor tags are
    uncorrelated with z_v;
C11 phi-monotonicity sanity (NON-GATING, reported): mean headroom(phi=1) >=
    mean headroom(phi=0) across the smoke cells;
C12 null-replication byte-identity: build_family_legacy output reproduces
    the iteration-1 pool families byte-for-byte, and the phi-aware phi=0
    user_logs equal the legacy user_logs element-for-element.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from loguru import logger

import generator_phi as gphi
from ablation_grid import make_params, smoke_specs

_HERE = Path(__file__).resolve().parent
_IT1_PROCESSED = (
    "/ai-inventor/aii_data/runs/run_5D4WD4vgZZMJ/3_invention_loop/iter_1/"
    "gen_art/gen_art_dataset_1/processed"
)


# ---------------------------------------------------------------------------
def c9_marginals_gate() -> tuple[bool, str]:
    """C9: the PRODUCTION gate path (generate_cell_with_gate) must converge
    to a catalog whose screen-half marginal matches phi=0 within the
    disclosed tolerances (KS<=0.05, max_abs_dev<=0.02, |dHHI|<=0.02,
    |dNormEntropy|<=0.03), either at the nominal phi or at a LOGGED
    effective strength (phi_effective <= phi).  Uses the harshest tiny cell
    (|V|=20, LOW, zipf=0.5, phi=1.0) where the raw kernel is known to
    flatten the marginal."""
    p0 = make_params(20, "LOW", 0.5, 0.0, 0)
    p1 = make_params(20, "LOW", 0.5, 1.0, 0)
    cat0 = gphi.build_family(p0)
    cat1, rep = gphi.generate_cell_with_gate(p1, cat0)
    passed = bool(rep["pass"])
    phi_eff = rep.get("phi_effective", 1.0)
    detail = (f"KS={rep['gate_KS']:.4f} maxabs={rep['gate_max_abs_dev']:.4f} "
              f"dHHI={rep['dHHI']:.5f} dNE={rep['dNormEntropy']:.5f} "
              f"pass={passed} phi_eff={phi_eff:.3f} capped={rep.get('capped', False)}")
    return passed, detail


def c10_tag_encoding() -> tuple[bool, str]:
    """C10: latent tags are exactly sign(z_v); distractor tags are not."""
    rng = gphi.latent_rng_for(0)
    Z, tags_per_item, tag_names = gphi.make_latent_structure(
        rng, 200, gphi.LATENT_D_C, gphi.N_DISTRACTOR_TAGS
    )
    n = Z.shape[0]
    d = gphi.LATENT_D_C
    agreed = 0
    total = 0
    for v in range(n):
        for t in range(d):
            has = f"lz_{t}" in tags_per_item[v]
            want = bool(Z[v, t] > 0.0)
            agreed += int(has == want)
            total += 1
    frac = agreed / total
    if frac < 0.9999:
        return False, f"latent tag agreement = {frac:.6f} (needs ~1.0)"
    # distractor correlation check
    dist_presence = np.zeros((n, gphi.N_DISTRACTOR_TAGS), dtype=float)
    for v in range(n):
        for j in range(gphi.N_DISTRACTOR_TAGS):
            dist_presence[v, j] = float(f"lx_{j}" in tags_per_item[v])
    max_corr = 0.0
    for j in range(gphi.N_DISTRACTOR_TAGS):
        for t in range(d):
            corr = abs(float(np.corrcoef(dist_presence[:, j], np.sign(Z[:, t]))[0, 1]))
            max_corr = max(max_corr, corr)
    if max_corr > 0.35:
        return False, f"distractor tag max |corr| with z_v = {max_corr:.3f} (>0.35)"
    return True, (f"latent agreement={frac:.4f}; distractor max|c|={max_corr:.3f} "
                  f"(design holds)")


def c11_phi_monotonicity() -> tuple[bool, str]:
    """C11 (NON-GATING): mean headroom(phi=1) >= mean headroom(phi=0) over
    the smoke cells.  Reported loudly, does not gate."""
    from eval_metrics import evaluate_catalog, compute_headroom  # noqa: F401
    from catalog_pool import build_catalog, _adapt_direct
    from pathlib import Path

    headrooms: dict[float, list[float]] = {0.0: [], 1.0: []}
    cells = [make_params(20, "LOW", 1.5, 0.0, 0), make_params(20, "LOW", 1.5, 1.0, 0),
             make_params(100, "HIGH", 1.5, 0.0, 1), make_params(100, "HIGH", 1.5, 1.0, 1)]
    for p in cells:
        cat = gphi.build_family(p)
        intern, err = build_catalog(_adapt_direct(cat, Path("x.json"))[0], Path("x.json"))
        if intern is None:
            return False, f"catalog {p['catalog_id']} failed adaptation: {err}"
        payload = evaluate_catalog(intern)
        headrooms[float(p["phi"])].append(float(payload["diagnostics"]["headroom"]))
    m0 = float(np.mean(headrooms[0.0])) if headrooms[0.0] else float("nan")
    m1 = float(np.mean(headrooms[1.0])) if headrooms[1.0] else float("nan")
    detail = (f"smoke cells: mean headroom(phi=0)={m0:.4f}, mean headroom(phi=1)="
              f"{m1:.4f} (holding: {m1 >= m0})")
    return bool(m1 >= m0), detail


def c12_null_byte_identity() -> tuple[bool, str]:
    """C12: (a) build_family_legacy reproduces the iteration-1 pool
    families byte-for-byte (sha256); (b) the phi-aware phi=0 user_logs equal
    the legacy user_logs element-for-element for a sample cell."""
    proc = Path(_IT1_PROCESSED)
    ids = ["synth_v20_s0", "synth_v200_s0", "synth_v1000_s1",
           "synth_tstat_s0", "synth_attr_many_s0"]
    ok = True
    checked = 0
    for fam in gphi.family_grid():
        if fam["catalog_id"] not in ids:
            continue
        cat = gphi.build_family_legacy(fam)
        mine = json.dumps(cat)
        orig = (proc / f"{fam['catalog_id']}.json").read_text()
        if mine != orig:
            return False, f"byte mismatch for {fam['catalog_id']}"
        checked += 1
    if checked < 3:
        return False, f"only {checked} families checked"
    # (b) phi-aware phi=0 user_logs == legacy user_logs on a tiny cell
    p = make_params(20, "LOW", 1.5, 0.0, 0)
    cat_phi = gphi.build_family(p)
    p_legacy = {k: v for k, v in p.items() if k not in ("phi", "emit_tags", "tighten")}
    cat_legacy = gphi.build_family_legacy(p_legacy)
    logs_a = cat_phi["user_logs"]
    logs_b = cat_legacy["user_logs"]
    if len(logs_a) != len(logs_b) or any(
        u["ordered_history"] != v["ordered_history"]
        or u["heldout_item"] != v["heldout_item"]
        or u["timestamps"] != v["timestamps"]
        for u, v in zip(logs_a, logs_b)
    ):
        return False, "phi-aware phi=0 user_logs differ from legacy logs"
    # and items' RNG-driven fields (category/price_band/base_weight) equal
    for a, b in zip(cat_phi["items"], cat_legacy["items"]):
        for key in ("item_id", "category", "price_band", "attribute_diversity"):
            if a[key] != b[key]:
                return False, f"item field {key} differs at phi=0"
    return True, (f"{checked} pool families byte-identical (sha256) + phi-aware "
                  f"phi=0 logs/items == legacy")


def _check_functions_phi() -> dict:
    return {
        "C9": c9_marginals_gate,
        "C10": c10_tag_encoding,
        "C11": c11_phi_monotonicity,
        "C12": c12_null_byte_identity,
    }


def run_checks_phi(names: list[str] | None = None, gate_c11: bool = False) -> bool:
    """Returns True if all (non-C11) checks pass.  C11 is reported and gates
    only when gate_c11 is True."""
    fns = _check_functions_phi()
    sel = names or list(fns)
    ok_all = True
    for name in sel:
        if name not in fns:
            logger.error(f"unknown check {name}")
            ok_all = False
            continue
        try:
            passed, detail = fns[name]()
        except Exception as e:  # pragma: no cover (defensive)
            passed, detail = False, f"EXCEPTION {type(e).__name__}: {e}"
        if name == "C11" and not gate_c11:
            logger.warning(f"{name} CHECKPOINT (non-gating): "
                           f"{'held' if passed else 'did NOT hold'} | {detail}")
            continue
        logger.info(f"{name}: {'PASS' if passed else 'FAIL'} | {detail}")
        ok_all = ok_all and passed
    return ok_all


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
    ok = run_checks_phi()
    sys.exit(0 if ok else 1)