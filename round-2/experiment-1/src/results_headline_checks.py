#!/usr/bin/env python3
"""Re-compute every headline count of README Section 6 from the named CSVs.

Every number quoted in README Section 6.x is produced by the print statements
below (plain pandas filters).  Running this script on the committed outputs
must reproduce the README table.  Exit code 0 = all asserted anchors match.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"

ab = pd.read_csv(OUT / "ablation_headroom_by_phi.csv")
boot = pd.read_csv(OUT / "bootstrap_headroom_CIs.csv")
gate = pd.read_csv(OUT / "marginals_match_gate.csv")
bytecheck = pd.read_csv(OUT / "null_replication_bytecheck.csv")
nrc = pd.read_csv(OUT / "null_replication_comparison.csv")
nvs = pd.read_csv(OUT / "null_vs_extended.csv")
diagr = pd.read_csv(OUT / "diagnostic_robustness.csv")
headline = json.loads((OUT / "headline_summary.json").read_text())
nrg = json.loads((OUT / "null_replication_gate.json").read_text())

fails: list[str] = []


def chk(name: str, got, want) -> None:
    ok = bool(got == want)
    print(f"{'OK ' if ok else 'FAIL'} {name}: got={got} want={want}")
    if not ok:
        fails.append(f"{name}: got={got} want={want}")


print("== 6.1 null-replication gate ==")
chk("pool families byte-identical", int((bytecheck.byte_identical == True).sum()), 19)
print(f"  all 19 byte-identical: {(bytecheck.byte_identical == True).all()}")
chk("NDCG@5 cells pass_print5", int((nrc.pass_print5 == True).sum()), int(nrc.pass_print5.notna().sum()))
hr = nrc[nrc.heuristic == "HEADROOM"]
chk("headroom rows within 1e-3", int((hr.pass_1e6 == True).sum()), len(hr))
chk("null-replication overall gate", bool(nrg["gate"]), True)

print("\n== 6.2 marginals-match gate ==")
chk("cells passing gate", int((gate.gate_pass == True).sum()), len(gate))
chk("capped cells", int((gate.phi_capped == True).sum()), 21)
mef = headline["mean_phi_effective_by_nominal"]
print(f"  mean phi_eff by nominal: {mef}")
chk("mean phi_eff at nominal 1.0 ~ 0.749", round(float(mef["1.0"]), 3), 0.749)

print("\n== 6.3 headroom per phi ==")
for phi in ("0.0", "0.25", "0.5", "1.0"):
    sel = ab[ab.phi_nominal == float(phi)]
    hf = sel.headroom_full.mean()
    hm = sel.headroom.mean()
    print(f"  phi={phi}: n={len(sel)} mean headroom_full={hf:.5f} mean headroom(inner)={hm:.5f}")
chk("mean headroom_full phi0", round(float(headline["headroom_full_mean_by_nominal_phi"]["0.0"]), 5), -0.16733)
chk("mean headroom_full phi1", round(float(headline["headroom_full_mean_by_nominal_phi"]["1.0"]), 5), -0.15192)
best = ab.loc[ab.phi_nominal == 1.0, "headroom_full"].max()
print(f"  max headroom_full at phi=1: {best:.4f} (cell {ab.loc[ab.phi_nominal==1.0,'headroom_full'].idxmax()})")
chk("max headroom_full phi1 == -0.0743", round(float(best), 4), -0.0743)

print("\n== 6.4 overtake / crossover ==")
chk("cells overtake_any", int((ab.overtake_any == True).sum()), 0)
chk("regions with phi>0 overtake", int((nvs.content_or_hybrid_overtakes_at_phi_gt0 == True).sum()), 0)
chk("min_phi_eff_overtake all none", bool((nvs.min_phi_eff_overtake == "none").all()), True)
w5 = {}
for phi in (0.0, 1.0):
    c = pd.Series([dict(x.split(":", 1) for x in w.split(";")).get("5", "na")
                   for w in ab[ab.phi_nominal == phi].winner_per_k])
    w5[phi] = c.value_counts().to_dict()
    print(f"  k=5 winner counts phi={phi}: {w5[phi]}")
chk("k=5 winner phi0 pop_global 17", w5[0.0].get("pop_global", 0), 17)
chk("k=5 winner phi1 pop_global 14", w5[1.0].get("pop_global", 0), 14)
chk("k=5 winner phi1 lambda_hybrid_0.25 5", w5[1.0].get("lambda_hybrid_0.25", 0), 5)

print("\n== 6.5 bootstrap CIs / decomposition ==")
dom = ab[(ab.phi_nominal == 1.0) & (ab.delta_dominant == True)]
chk("phi1 CI-dominant cells", len(dom), 7)
print(f"  dominant regions: {sorted(set(zip(dom.n_items, dom.entropy, dom.zipf_s)))}")
nd = ab[(ab.phi_nominal > 0) & (ab.delta_dominant.notna()) & (ab.delta_vs_phi0.notna())]
chk("paired delta rows at phi=1", int((nd.phi_nominal == 1.0).sum()), 24)
# decomposition over twin-paired cells (phi1 vs phi0 by n_items/entropy/zipf_s/seed)
pop_fam = ["pop_global", "pop_category", "pop_price", "recency_pop_0.1",
           "recency_pop_0.5", "recency_pop_1.0", "pop_scaled_content"]
agg = {}
for _, r in ab.iterrows():
    cid = r.catalog_id
    key = (r.n_items, r.entropy, r.zipf_s, r.seed, r.phi_nominal)
    vals = []
    for k in (1, 2, 3, 5, 8):
        ck = boot[(boot.catalog_id == cid) & (boot.heuristic == "content_knn_3")
                  & (boot.k == k)]
        if not len(ck):
            continue
        ckn = float(ck.ndcg5.iloc[0])
        bpv = [float(boot[(boot.catalog_id == cid) & (boot.heuristic == h)
                          & (boot.k == k)].ndcg5.iloc[0]) for h in pop_fam
               if len(boot[(boot.catalog_id == cid) & (boot.heuristic == h)
                           & (boot.k == k)])]
        if not bpv:
            continue
        vals.append((ckn, max(bpv), ckn - max(bpv)))
    agg[key] = np.mean(vals, axis=0)
pairs = [(k0, k1) for k1 in agg if k1[4] == 1.0 and (k0 := k1[:4] + (0.0,)) in agg]
dck = np.array([agg[k1][0] - agg[k0][0] for k0, k1 in pairs])
dbp = np.array([agg[k1][1] - agg[k0][1] for k0, k1 in pairs])
dh = np.array([agg[k1][2] - agg[k0][2] for k0, k1 in pairs])
print(f"  paired cells: {len(pairs)} | d_content_knn_3 mean={dck.mean():+.5f} "
      f"frac>0={(dck > 0).mean():.2f}")
print(f"  d_best_pop mean={dbp.mean():+.5f} frac<0={(dbp < 0).mean():.2f} | "
      f"d_headroom mean={dh.mean():+.5f} (check dck-dbp: {dck.mean() - dbp.mean():+.5f})")
chk("d_content_knn_3 mean ~ +0.0017", round(float(dck.mean()), 4), 0.0017)
chk("d_best_pop mean ~ -0.0136", round(float(dbp.mean()), 4), -0.0136)
chk("d_headroom mean ~ +0.0154", round(float(dh.mean()), 4), 0.0154)
for hname in ("content_knn_3", "pop_global"):
    sub = boot[(boot.heuristic == hname) & (boot.k == 8)].copy()
    sub["phi"] = sub.catalog_id.map(dict(zip(ab.catalog_id, ab.phi_nominal)))
    g = sub.groupby("phi").ndcg5.mean()
    print(f"  k=8 {hname}: phi0={g.get(0.0, float('nan')):.4f} phi1={g.get(1.0, float('nan')):.4f}")

print("\n== 6.6 null-vs-extended ==")
chk("region rows", len(nvs), 12)

print("\n== 6.7 diagnostic robustness ==")
chk("build labels at any phi_eff", int(diagr.n_build_label.sum()), 0)
print(f"  first_failure_phi values: {diagr.first_failure_phi.unique()}")
print(f"  HHI-rule accuracy per phi_eff: {diagr.acc_hhi_rule.iloc[0]} (phi0) .. "
      f"{diagr.acc_hhi_rule.iloc[-1]} (phi=1)")

print()
if fails:
    print(f"FAILED ANCHORS: {len(fails)}")
    for f in fails:
        print(" -", f)
    raise SystemExit(1)
print("ALL HEADLINE ANCHORS MATCH")