#!/usr/bin/env python3
"""Measurement: marginal-gate pass rates for factor-normalization designs.

Compares two latent-factor constructions on a sample of (config, seed)
cells at phi=1:
  raw  : z_v ~ N(0, I) (current design)
  norm : z_v <- z_v / ||z_v||  (unit-normed factors -> equal per-item
         preference variance, so the within-category softmax kernel has the
         same marginal distribution for every item -> marginal preserved in
         expectation to second order)
Each cell is measured at gate tolerances with no tightening.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import generator_phi as gphi
from ablation_grid import make_params

_CELLS = [
    (20, "LOW", 1.5), (20, "LOW", 0.5), (20, "HIGH", 1.5),
    (100, "LOW", 1.5), (100, "HIGH", 0.5), (100, "HIGH", 1.5),
    (500, "LOW", 0.5), (500, "HIGH", 1.5), (500, "LOW", 1.5),
]


def build_cell(p: dict, norm: bool) -> dict:
    rng = gphi._seeded(p["seed"])
    items, _ = gphi.make_items(rng, p["n_items"], p["n_attrs"], p["zipf_s"])
    turn = gphi.make_turnover(rng, p["n_items"], p["dynamic"], 365)
    lr = gphi.latent_rng_for(p["seed"])
    Z = lr.standard_normal(size=(p["n_items"], gphi.LATENT_D_C))
    if norm:
        Z = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)
    sim = gphi.sample_histories_phi(
        items, turn, p["n_users"], p["mean_hist"], p["alpha"], p["beta"],
        seed=p["seed"] + 1000, phi=float(p["phi"]), Z=Z, latent_rng=lr,
    )
    cat = {
        "catalog_id": p["catalog_id"], "items": items,
        "user_logs": sim["user_logs"], "forced_choice_probes": sim["probes"],
    }
    return cat


def main() -> None:
    print(f"{'cell':34s} {'design':5s} {'KS':>8s} {'maxabs':>8s} {'dHHI':>9s} {'dNE':>8s} PASS")
    for (ni, ent, zf) in _CELLS:
        p0 = make_params(ni, ent, zf, 0.0, 0)
        p1 = make_params(ni, ent, zf, 1.0, 0)
        cat0 = build_cell(p0, norm=False)
        for norm in (False, True):
            cat1 = build_cell(p1, norm=norm)
            rep = gphi.marginals_match(cat1, cat0)
            print(f"{p1['catalog_id']:34s} {'norm' if norm else 'raw ':5s} "
                  f"{rep['gate_KS']:8.4f} {rep['gate_max_abs_dev']:8.4f} "
                  f"{rep['dHHI']:9.4f} {rep['dNormEntropy']:8.4f} "
                  f"{'yes' if rep['pass'] else 'no '}")


if __name__ == "__main__":
    main()