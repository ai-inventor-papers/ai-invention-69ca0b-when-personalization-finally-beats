#!/usr/bin/env python3
"""Iteration-2 ablation grid.

Main factorial: |V| in {20, 100, 500} x entropy {LOW, HIGH} x zipf_s
{0.5, 1.5} x phi {0, 0.25, 0.5, 1.0} x seed {0, 1} = 96 cells, plus seed 2
on a stratified subset (one phi per size/entropy/zipf corner, phi=0.5) to
reach 108 main-grid catalogs (~120-150 with the 22 null-replication
catalogs).  Cross-phi controls are HELD: mean_hist=10, alpha=0.3 (sticky),
beta=2.0, dynamic turnover, horizon 365d.  n_users scaled by |V|
(600 / 1000 / 1500) so every (catalog, k) cell stays above MIN_EXAMPLES=50
supported users.  entropy LOW -> n_attrs=2, HIGH -> n_attrs=30.

Cell ids: phi_V{n_items}_E{entropy}_z{zipf_s}_p{phi}_s{seed}.
"""

from __future__ import annotations

import re

PHI_LEVELS: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0)
SIZES: tuple[int, ...] = (20, 100, 500)
ENTROPIES: tuple[str, ...] = ("LOW", "HIGH")
ZIPFS: tuple[float, ...] = (0.5, 1.5)
ENTROPY_N_ATTRS = {"LOW": 2, "HIGH": 30}
SIZE_N_USERS = {20: 600, 100: 1000, 500: 1500}
# Holds across phi (controls, not axes)
MEAN_HIST = 10
ALPHA = 0.3
BETA = 2.0
DYNAMIC = True
HORIZON_DAYS = 365
N_PROBES = 3
PROBE_NOISE = 0.0

_SEED2_PHI = 0.5


def _fmt_phi(phi: float) -> str:
    if phi == int(phi):
        return f"{int(phi):d}"
    return f"{phi:.2f}".rstrip("0")


def catalog_id_for(n_items: int, entropy: str, zipf_s: float, phi: float,
                   seed: int) -> str:
    return (f"phi_V{n_items}_E{entropy}_z{zipf_s}"
            f"_p{_fmt_phi(phi)}_s{seed}")


def make_params(n_items: int, entropy: str, zipf_s: float, phi: float,
                seed: int) -> dict:
    """One main-grid generation-parameter dict."""
    return {
        "catalog_id": catalog_id_for(n_items, entropy, zipf_s, phi, seed),
        "n_items": n_items,
        "n_attrs": ENTROPY_N_ATTRS[entropy],
        "entropy_level": entropy,
        "zipf_s": zipf_s,
        "alpha": ALPHA,
        "beta": BETA,
        "mean_hist": MEAN_HIST,
        "n_users": SIZE_N_USERS[n_items],
        "dynamic": DYNAMIC,
        "horizon_days": HORIZON_DAYS,
        "n_probes": N_PROBES,
        "probe_noise": PROBE_NOISE,
        "phi": phi,
        "seed": seed,
        "emit_tags": True,
        "tighten": 0,
    }


def main_grid_specs() -> list[dict]:
    """96 main-factorial cells (seeds 0,1 at all phi)."""
    cells: list[dict] = []
    for n_items in SIZES:
        for entropy in ENTROPIES:
            for zipf_s in ZIPFS:
                for phi in PHI_LEVELS:
                    for seed in (0, 1):
                        cells.append(make_params(n_items, entropy, zipf_s,
                                                 phi, seed))
    return cells


def seed2_specs() -> list[dict]:
    """12 stratified seed-2 cells: one phi (=0.5) per size/entropy/zipf
    corner (extends the 96-cell factorial, documented in the README)."""
    cells: list[dict] = []
    for n_items in SIZES:
        for entropy in ENTROPIES:
            for zipf_s in ZIPFS:
                cells.append(make_params(n_items, entropy, zipf_s,
                                         _SEED2_PHI, 2))
    return cells


def full_main_grid_specs() -> list[dict]:
    return main_grid_specs() + seed2_specs()


def smoke_specs() -> list[dict]:
    """Smoke: one tiny corner at phi=0 and phi=1 (entity of T2)."""
    return [
        make_params(20, "LOW", 1.5, 0.0, 0),
        make_params(20, "LOW", 1.5, 1.0, 0),
    ]


_CELL_RE = re.compile(
    r"^phi_V(\d+)_E(LOW|HIGH)_z([0-9.]+)_p([0-9.]+)_s(\d+)$"
)


def parse_cell_id(catalog_id: str) -> dict | None:
    """Parse a phi_V... main-grid cell id back into config fields."""
    m = _CELL_RE.match(catalog_id)
    if not m:
        return None
    return {
        "n_items": int(m.group(1)),
        "entropy": m.group(2),
        "zipf_s": float(m.group(3)),
        "phi": float(m.group(4)),
        "seed": int(m.group(5)),
    }