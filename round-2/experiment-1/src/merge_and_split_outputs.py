#!/usr/bin/env python3
"""Merge out/method_out/*.json parts into one method_out.json, then apply
the aii-file-size-limit procedure (100 MB limit): split the oversized
method_out.json / full_method_out.json into part directories, and generate
mini/preview variants with the aii-json format script's exact semantics
(datasets-grouped branch of aii_json_format_mini_preview.py):

  full    = identical to input
  mini    = ALL datasets, examples sliced to the first 3 per dataset
  preview = datasets sliced to the first 3, examples to the first 3,
            strings truncated to 200 chars + '...'

The merged full output (~5 GB) cannot be loaded by the stock format script
(json.load of 5 GB would exceed the 32 GB container), so the mini/preview
branch is implemented here with IDENTICAL semantics (verified against the
script source and against iteration-1's committed mini/preview shapes:
mini = n datasets x 3 examples, preview = 3 datasets x 3 examples with
200-char string truncation).

Split layout (file-size-limit skill, 100 MB limit):
  method_out/        <= 100 MB parts named method_out_NNNN.json,
                        hard links to out/method_out/*.json (same inode,
                        zero extra disk; parts already <= 61 MB each)
  full_method_out/   <= 100 MB parts named full_method_out_NNNN.json,
                        hard links to the same canonical parts
The oversized single files are deleted once their part directories exist,
exactly as the skill prescribes.
"""

from __future__ import annotations

import gc
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARTS_DIR = HERE / "out" / "method_out"

MAX_STRING_LENGTH = 200
TRUNCATE_MARKER = "..."
MAX_ARRAY_ITEMS = 3


def truncate_value(value):
    """Exact replica of aii_json_format_mini_preview.truncate_value."""
    if isinstance(value, list):
        return [truncate_value(item) for item in value[:MAX_ARRAY_ITEMS]]
    if isinstance(value, str):
        if len(value) > MAX_STRING_LENGTH:
            return value[:MAX_STRING_LENGTH] + TRUNCATE_MARKER
        return value
    if isinstance(value, dict):
        return {key: truncate_value(val) for key, val in value.items()}
    return value


def merge_parts(out_path: Path) -> None:
    """Streaming merge of part files into one datasets-grouped dict."""
    parts = sorted(PARTS_DIR.glob("method_out_*.json"))
    if not parts:
        raise SystemExit(f"no parts found under {PARTS_DIR}")
    total = 0
    with open(out_path, "w", encoding="utf-8") as out:
        out.write('{"metadata": {"method": "method.py", '
                  '"pipeline": "cold-start 16-heuristic sweep", '
                  '"parts_merged": %d, '
                  '"merged_by": "merge_and_split_outputs.py"}, "datasets": ['
                  % len(parts))
        first = True
        for i, p in enumerate(parts):
            with open(p, encoding="utf-8") as f:
                part = json.load(f)
            for ds in part["datasets"]:
                payload = json.dumps(ds, ensure_ascii=False)
                if not first:
                    out.write(",")
                out.write(payload)
                first = False
                total += 1
                if total % 10 == 0:
                    sys.stdout.write(f"\r  merged datasets: {total}")
                    sys.stdout.flush()
            del part
            gc.collect()
        out.write("]}\n")
    print(f"\nmerged {total} datasets from {len(parts)} parts -> {out_path.name} "
          f"({out_path.stat().st_size / 1e9:.2f} GB)")


def slice_examples_to(ds: dict, n: int) -> dict:
    return {**ds, "examples": ds.get("examples", [])[:n]}


def make_mini_preview(parts: list[Path]) -> None:
    """mini = all datasets x 3 examples; preview = first-3 datasets, 3
    examples, strings truncated (format-script semantics)."""
    all_ds: list[dict] = []
    for p in parts:
        with open(p, encoding="utf-8") as f:
            part = json.load(f)
        all_ds.extend(part["datasets"])
        del part
        gc.collect()
    mini = {"metadata": {"source": "merge_and_split_outputs.py",
                         "full_parts": len(parts)},
            "datasets": [slice_examples_to(d, MAX_ARRAY_ITEMS) for d in all_ds]}
    (HERE / "mini_method_out.json").write_text(
        json.dumps(mini, ensure_ascii=False))
    preview = truncate_value(
        {"metadata": mini["metadata"],
         "datasets": [slice_examples_to(d, MAX_ARRAY_ITEMS)
                      for d in all_ds[:MAX_ARRAY_ITEMS]]})
    (HERE / "preview_method_out.json").write_text(
        json.dumps(preview, ensure_ascii=False))
    print(f"mini: {len(mini['datasets'])} datasets "
          f"({sum(len(d['examples']) for d in mini['datasets'])} examples); "
          f"preview: {len(preview['datasets'])} datasets "
          f"({sum(len(d['examples']) for d in preview['datasets'])} examples)")


def link_parts(dest_dir: Path, prefix: str) -> None:
    """Hard-link canonical parts into dest_dir as {prefix}_NNNN.json."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    for p in sorted(PARTS_DIR.glob("method_out_*.json")):
        number = p.name.split("_", 2)[-1]          # "method_out_0001.json" -> "0001.json"
        target = dest_dir / f"{prefix}_{number}"
        if target.exists():
            target.unlink()
        os.link(p, target)
    print(f"linked {len(list(dest_dir.glob('*.json')))} parts into {dest_dir.name}/")


def main() -> None:
    # The merged single-file form is TRANSIENT: it exists only while the
    # part views are built, then it is deleted.  The committed layout is the
    # split-parts form (100 MB-limit procedure); no file in this workspace
    # may exceed 100 MB.  A leftover temp from an interrupted run is removed.
    tmp = HERE / "method_out.merged.tmp"
    if tmp.exists():
        tmp.unlink()
    # 1) merged full output (temp; never committed)
    merge_parts(tmp)
    # 2) mini + preview (format-script semantics)
    parts = sorted(PARTS_DIR.glob("method_out_*.json"))
    make_mini_preview(parts)
    # 3) split the oversized merged content into <= 100 MB part directories
    link_parts(HERE / "method_out", "method_out")
    link_parts(HERE / "full_method_out", "full_method_out")
    # 4) delete the transient oversized file (file-size-limit procedure)
    tmp.unlink()
    # 5) manifest
    part_bytes = sum(p.stat().st_size for p in parts)
    manif = {"canonical_parts": "out/method_out/method_out_NNNN.json",
             "split_views": ["method_out/method_out_NNNN.json",
                             "full_method_out/full_method_out_NNNN.json"],
             "method_out_parts": len(parts),
             "merged_size_gb": round(part_bytes / 1e9, 2),
             "file_size_limit": "100 MB per part (max observed 61 MB)",
             "reconstruct": ("concatenate the parts in sorted NNNN order: "
                             "{\"metadata\": ..., \"datasets\": [d1, d2, ...]} "
                             "or stream-merge with merge_parts()"),
             "note": ("hard links share inodes with out/method_out/*.json; "
                      "duplicate views are excluded from repo upload via "
                      "upload_ignore_regexes")}
    (HERE / "full_method_out" / "full_method_out_manifest.json").write_text(
        json.dumps(manif, indent=1))
    print(f"split done: {len(parts)} parts, "
          f"{part_bytes / 1e9:.2f} GB total, transient merged file deleted")


if __name__ == "__main__":
    main()