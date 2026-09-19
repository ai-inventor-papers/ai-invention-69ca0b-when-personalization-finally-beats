#!/usr/bin/env python3
"""Shared-pool catalog discovery + normalization adapter.

Searches the iteration's shared pool for catalog JSON files produced by the
parallel DATASET artifact.  Any candidate JSON is parsed and normalized to a
Catalog (synthetic.Catalog); files that cannot be parsed into >=1 usable
catalog (real histories + item attributes) are skipped with a logged reason.
If nothing usable is found the experiment falls back to the inline synthetic
generator (source='inline_fallback').
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from loguru import logger

from synthetic import Catalog

# Paths searched for pool catalogs (in order); env vars may add more.
_POOL_ROOTS: list[str] = [
    "/ai-inventor/aii_data/runs/run_5D4WD4vgZZMJ/3_invention_loop/iter_1/pool",
    "/ai-inventor/aii_data/runs/run_5D4WD4vgZZMJ/3_invention_loop/iter_1/execute/dataset",
    "/ai-inventor/aii_data/runs/run_5D4WD4vgZZMJ/3_invention_loop/iter_1/execute",
    "/ai-inventor/aii_data/runs/run_5D4WD4vgZZMJ/3_invention_loop/iter_1/gen_art/gen_art_dataset_1",
    "/ai-inventor/aii_data/runs/run_5D4WD4vgZZMJ/3_invention_loop/iter_1/gen_plan/gen_plan_dataset_1",
]


def _candidate_roots() -> list[Path]:
    roots: list[Path] = []
    for env in ("POOL_DIR", "DATASET_OUT", "POOL_PATHS"):
        v = os.environ.get(env)
        if v:
            roots.append(Path(v))
    for r in _POOL_ROOTS:
        roots.append(Path(r))
    seen: set[Path] = set()
    out: list[Path] = []
    for r in roots:
        r = r.resolve()
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def _skip_path(p: Path) -> bool:
    parts = p.parts
    for ex in ("logs", "sinks"):
        if ex in parts:
            return True
    # aggregate/preview outputs under any "out" directory are not catalogs
    if "out" in parts:
        return True
    # this workspace and the gen_plan workspaces are not pool sources
    if "gen_art_experiment_1" in parts or "gen_plan_experiment_1" in parts:
        return True
    if p.name.startswith("."):
        return True
    return False


def discover_candidate_files() -> list[Path]:
    """All candidate JSON files under the pool roots (dedup, sorted)."""
    files: dict[str, Path] = {}
    for root in _candidate_roots():
        if not root.exists():
            continue
        for p in sorted(root.rglob("*.json")):
            if _skip_path(p):
                continue
            files[str(p)] = p
    return [files[k] for k in sorted(files)]


def _as_str(v) -> str:
    return str(v)


def _adapt_direct(d: dict, filepath: Path) -> list[dict]:
    """Normalize one catalog dict (any of several producer schemes) into the
    canonical internal form.  Returns a list of catalog dicts (usually 1)."""
    items_raw = d.get("items") or d.get("catalog_items")
    logs = d.get("user_logs") or d.get("users") or d.get("purchases")
    if not items_raw or not logs:
        return []
    items: list[dict] = []
    id2idx: dict[str, int] = {}
    for i, it in enumerate(items_raw):
        if isinstance(it, str):
            items.append({"item_id": it, "category": None, "price_band": None, "tags": []})
        elif isinstance(it, dict):
            attrs = it.get("attrs") or it.get("attributes") or {}
            if not isinstance(attrs, dict):
                attrs = {}
            cat = attrs.get("category") or it.get("category")
            pb = attrs.get("price_band") or it.get("price_band") or attrs.get("price")
            tags = it.get("tags") or attrs.get("tags") or []
            if not isinstance(tags, list):
                tags = [tags] if tags else []
            items.append({
                "item_id": _as_str(it.get("item_id") or it.get("id") or f"i{i}"),
                "category": _as_str(cat) if cat is not None else None,
                "price_band": _as_str(pb) if pb is not None else None,
                "tags": [_as_str(t) for t in tags],
                "popularity_weight": it.get("popularity_weight"),
            })
        else:
            return []
        id2idx[items[-1]["item_id"]] = i

    users: list[dict] = []
    unmapped_users = 0
    for u in logs:
        if not isinstance(u, dict):
            continue
        uid = _as_str(u.get("user_id") or u.get("userId") or u.get("id") or f"u{len(users)}")
        hist = u.get("ordered_history") or u.get("history") or u.get("purchases") or u.get("items")
        t = u.get("timestamps") or u.get("times")
        if not hist:
            continue
        seq: list[int] = []
        ts: list[float] = []
        ok = True
        for j, h in enumerate(hist):
            iid, tt = None, float(j)
            if isinstance(h, (list, tuple)):
                iid, tt = h[0], float(h[1]) if len(h) > 1 else float(j)
            else:
                iid = h
                if t is not None and j < len(t):
                    tt = float(t[j])
            if iid not in id2idx:
                ok = False
                break
            seq.append(id2idx[iid])
            ts.append(tt)
        if not ok or not seq:
            unmapped_users += 1
            continue
        order = np.argsort(np.asarray(ts, dtype=np.float64), kind="stable")
        users.append({
            "user_id": uid,
            "item_seq": np.asarray(seq, dtype=np.int32)[order],
            "t_seq": np.asarray(ts, dtype=np.float64)[order],
            "beta": None,
            "epsilon": None,
            "argmax_cat": None,
        })
    if not users:
        return []
    catalog_id = _as_str(d.get("catalog_id") or d.get("dataset") or d.get("id")
                         or filepath.stem)
    notes = list(d.get("notes") or [])
    if unmapped_users:
        notes.append(
            f"{unmapped_users} user histories contain items outside the catalog's "
            f"item universe (unmapped); only fully-mapped users are kept"
        )
    return [{
        "catalog_id": catalog_id,
        "items": items,
        "users": users,
        "fold": _as_str(d.get("fold") or "confirm"),
        "family_params": d.get("family_params") or d.get("partition_meta") or {},
        "origin": _as_str(d.get("origin") or "real"),
        "source_name": _as_str(d.get("source_name") or "pool"),
        "notes": notes,
    }]


def adapt_file(filepath: Path) -> tuple[list[dict], str | None]:
    """Parse one JSON file into a list of internal catalog dicts.
    Returns (catalogs, None) or ([], reason)."""
    try:
        data = json.loads(filepath.read_text())
    except (json.JSONDecodeError, OSError) as e:
        return [], f"unreadable JSON: {e}"
    cat_dicts: list[dict] = []
    if isinstance(data, dict):
        if "datasets" in data and isinstance(data["datasets"], list):
            for entry in data["datasets"]:
                if not isinstance(entry, dict):
                    continue
                inner: list[dict] = []
                if "items" in entry or "user_logs" in entry or "users" in entry:
                    inner = _adapt_direct(entry, filepath)
                elif "examples" in entry:
                    # aggregate-example scheme (per-user rows with metadata)
                    inner = _adapt_examples(entry, filepath)
                cat_dicts.extend(inner)
        else:
            cat_dicts = _adapt_direct(data, filepath)
            if not cat_dicts and isinstance(data.get("examples"), list):
                cat_dicts = _adapt_examples(data, filepath)
    elif isinstance(data, list):
        for entry in data:
            if isinstance(entry, dict):
                cat_dicts.extend(_adapt_direct(entry, filepath))
    return cat_dicts, None


def _adapt_examples(d: dict, filepath: Path) -> list[dict]:
    """Produce scheme: examples[] rows with item/user/attr/history metadata."""
    examples = d.get("examples")
    if not isinstance(examples, list) or not examples:
        return []
    items: dict[str, dict] = {}
    history_by_user: dict[str, list[tuple[str, float]]] = {}
    user_ids_in_order: list[str] = []
    for ex in examples:
        if not isinstance(ex, dict):
            continue
        meta = ex.get("metadata") if isinstance(ex.get("metadata"), dict) else ex
        uid = _as_str(meta.get("user_id") or ex.get("user"))
        iid = _as_str(meta.get("item_id") or meta.get("item") or ex.get("item"))
        if not uid or not iid:
            continue
        t = float(meta.get("timestamp", meta.get("t", len(history_by_user.get(uid, [])))))
        cat = meta.get("category")
        pb = meta.get("price_band") or meta.get("price")
        items.setdefault(iid, {"item_id": iid, "category": cat, "price_band": pb, "tags": []})
        if uid not in history_by_user:
            history_by_user[uid] = []
            user_ids_in_order.append(uid)
        history_by_user[uid].append((iid, t))
    if not user_ids_in_order:
        return []
    items_list = [items[i] for i in sorted(items)]
    id2idx = {it["item_id"]: j for j, it in enumerate(items_list)}
    users: list[dict] = []
    for uid in user_ids_in_order:
        hist = sorted(history_by_user[uid], key=lambda x: x[1])
        users.append({
            "user_id": uid,
            "item_seq": np.asarray([id2idx[h[0]] for h in hist], dtype=np.int32),
            "t_seq": np.asarray([h[1] for h in hist], dtype=np.float64),
            "beta": None, "epsilon": None, "argmax_cat": None,
        })
    return [{
        "catalog_id": _as_str(d.get("catalog_id") or d.get("dataset") or filepath.stem),
        "items": items_list, "users": users,
        "fold": _as_str(d.get("fold") or "confirm"),
        "family_params": d.get("family_params") or {},
        "origin": "real", "source_name": "pool", "notes": [],
    }]


def build_catalog(cd: dict, filepath: Path) -> tuple[Catalog | None, str | None]:
    """Internal catalog dict -> Catalog.  Returns (cat, None) or (None, reason)."""
    items = cd.get("items") or []
    if not items:
        return None, "no items"
    # map item_id -> internal index & build attribute vocabularies
    cats: list[str] = []
    bands: list[str] = []
    tags: set[str] = set()
    for it in items:
        c = it.get("category")
        b = it.get("price_band")
        if c is not None:
            cats.append(_as_str(c))
        else:
            cats.append("__none__")
        bands.append(_as_str(b) if b is not None else "__none__")
        tags.update(_as_str(t) for t in it.get("tags") or [])
    cat_vocab = sorted(set(cats))
    band_vocab = sorted(set(bands))
    tag_vocab = sorted(tags)
    cat_idx = {c: i for i, c in enumerate(cat_vocab)}
    band_idx = {b: i for i, b in enumerate(band_vocab)}
    tag_idx = {t: i for i, t in enumerate(tag_vocab)}
    n = len(items)
    categories = np.array([cat_idx[c] for c in cats], dtype=np.int32)
    price_bands = np.array([band_idx[b] for b in bands], dtype=np.int32)
    tag_matrix = np.zeros((n, len(tag_vocab)), dtype=bool)
    for i, it in enumerate(items):
        for t in it.get("tags") or []:
            tag_matrix[i, tag_idx[_as_str(t)]] = True
    item_ids = np.array([it["item_id"] for it in items], dtype=object)
    if len(set(item_ids.tolist())) != n:
        return None, "duplicate item ids"
    pw = np.array([it.get("popularity_weight") for it in items], dtype=np.float64)
    pw = pw if np.isfinite(pw).all() and pw.size == n else None
    users = cd.get("users") or []
    n_users = len(users)
    note_unmapped = next((n for n in (cd.get("notes") or []) if "unmapped" in n), "")
    if n_users < 10:
        reason = f"too few users ({n_users})"
        if note_unmapped:
            reason += f"; {note_unmapped}"
        return None, reason
    total_events = sum(len(u["item_seq"]) for u in users)
    if total_events < 50:
        return None, f"too few events ({total_events})"
    if n > 2000 or total_events > 2_000_000:
        return None, f"oversized pool catalog (n={n}, events={total_events}); skipped (fallback plan 6)"
    notes = list(cd.get("notes") or [])
    user_subsample = None
    if n_users > 1000:
        # fallback plan 6: bounded stratified subsample (fixed seed, recorded)
        rng = np.random.RandomState(0)
        sel = rng.choice(n_users, size=1000, replace=False)
        sel.sort()
        user_subsample = {"n_original": n_users, "n_sampled": 1000, "seed": 0,
                          "rule": "fixed RandomState(0) selection, recorded"}
        users = [users[j] for j in sel]
        n_users = len(users)
        notes.append(f"user_subsample={user_subsample}")
    intern_cat = Catalog(
        catalog_id=cd.get("catalog_id") or filepath.stem,
        origin=cd.get("origin") or "real",
        source_name=cd.get("source_name") or "pool",
        fold=cd.get("fold") or "confirm",
        family_params=cd.get("family_params") or {},
        rng_seed=None,
        item_ids=item_ids,
        categories=categories,
        price_bands=price_bands,
        tag_matrix=tag_matrix,
        popularity_weight=pw,
        users=users,
        source_path=str(filepath),
        notes=notes,
    )
    if user_subsample:
        intern_cat.family_params["user_subsample"] = user_subsample
    return intern_cat, None


def load_pool_catalogs() -> tuple[list[Catalog], list[dict]]:
    """Discover + adapt pool catalogs.  Returns (catalogs, skipped_log) where
    skipped_log entries are {file, reason}."""
    skipped: list[dict] = []
    catalogs: list[Catalog] = []
    for fp in discover_candidate_files():
        cat_dicts, err = adapt_file(fp)
        if err:
            skipped.append({"file": str(fp), "reason": err})
            continue
        if not cat_dicts:
            skipped.append({"file": str(fp), "reason": "no parseable catalog"})
            continue
        for cd in cat_dicts:
            try:
                intern_cat, reason = build_catalog(cd, fp)
            except Exception as e:  # isolate per-catalog failures
                intern_cat, reason = None, f"adaptation error: {type(e).__name__}: {e}"
            if intern_cat is None:
                skipped.append({"file": str(fp), "reason": reason})
            else:
                logger.info(f"pool catalog {intern_cat.catalog_id} from {fp} "
                            f"({intern_cat.n_items} items, {intern_cat.n_users} users)")
                catalogs.append(intern_cat)
    return catalogs, skipped