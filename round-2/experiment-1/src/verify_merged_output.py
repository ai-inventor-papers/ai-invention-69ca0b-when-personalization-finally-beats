#!/usr/bin/env python3
"""Structural verification of the full method output.

The full 5.35 GB output is stored as SPLIT PARTS (100 MB-limit procedure;
the single merged file exists only transiently while building the parts).
This script verifies the split form, which is what downstream code reads:

  * the split views method_out/ and full_method_out/ (and the canonical
    out/method_out/) hold 103 parts each, all < 100 MB;
  * every part parses as exp_gen_sol_out-shaped JSON;
  * the parts concatenate to exactly 131 datasets, first dataset olist_all;
  * the three part views share inodes (same content, zero duplicate disk).

If a transient single file (method_out.json) exists, the byte-level
header/trailer + boundary-count checks from the merge are additionally run
on it (the same checks that ran when the merged file was built)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MERGED = HERE / "method_out.json"          # transient single file (optional)
PARTS_DIR = HERE / "out" / "method_out"    # canonical parts
SPLIT_DIR = HERE / "method_out"            # 100 MB-limit split view
FULL_SPLIT_DIR = HERE / "full_method_out"  # 100 MB-limit split view
LIMIT = 100 * 1024 * 1024

ok = True


def _check(name: str, cond: bool, detail: str = "") -> None:
    global ok
    ok = ok and bool(cond)
    print(f"{'OK ' if cond else 'FAIL'} {name} {detail}")


# --- 1) part views exist, same count, all < 100 MB, same inodes -----------
# part-only globs: numbered parts end in _NNNN.json (the *_manifest.json
# in full_method_out/ is metadata, not a part)
canon = sorted(PARTS_DIR.glob("method_out_*.json"))
split = sorted(p for p in SPLIT_DIR.glob("method_out_*.json") if "_manifest" not in p.name)
full = sorted(p for p in FULL_SPLIT_DIR.glob("full_method_out_*.json") if "_manifest" not in p.name)
_check("canonical parts count", len(canon) == 103, f"({len(canon)})")
_check("split view parts count", len(split) == 103, f"({len(split)})")
_check("full split view parts count", len(full) == 103, f"({len(full)})")
max_part = max(p.stat().st_size for p in canon)
_check("all parts <= 100 MB", max_part <= LIMIT, f"(max {max_part/1e6:.1f} MB)")
inode_same = ({p.stat().st_ino for p in canon}
              == {p.stat().st_ino for p in split}
              == {p.stat().st_ino for p in full})
_check("split views share canonical inodes", inode_same)

# --- 2) every part parses; datasets concatenate to 131 --------------------
total_datasets = 0
first_dataset: str | None = None
examples_total = 0
for p in canon:
    with open(p, encoding="utf-8") as f:
        part = json.load(f)
    ds = part["datasets"]
    total_datasets += len(ds)
    examples_total += sum(len(d["examples"]) for d in ds)
    if first_dataset is None and ds:
        first_dataset = ds[0]["dataset"]
_check("all parts parse; 131 datasets total", total_datasets == 131,
      f"(got {total_datasets})")
_check("total examples", examples_total >= 50, f"({examples_total})")
_check("first dataset is olist_all", first_dataset == "olist_all",
       f"({first_dataset})")

# --- 3) if a transient merged single file exists, byte-verify it ----------
if MERGED.exists():
    with open(MERGED, "rb") as f:
        head = f.read(128)
        f.seek(max(0, MERGED.stat().st_size - 128))
        tail = f.read(128)
    _check("single-file header", head.startswith(b'{"metadata": '))
    _check("single-file trailer", tail.rstrip().endswith(b"]}"))
    count = 0
    needle = b'{"dataset"'
    prev = b""
    with open(MERGED, "rb") as f:
        while True:
            chunk = f.read(8 * 1024 * 1024)
            if not chunk:
                break
            data = prev + chunk
            pos = data.find(needle)
            while pos >= 0:
                count += 1
                pos = data.find(needle, pos + len(needle))
            prev = data[-len(needle) + 1:]
    _check("single-file dataset count", count == 131, f"({count})")
    print("transient single-file byte checks done")

if not ok:
    print("VERIFY MERGED: FAILED")
    sys.exit(1)
print("VERIFY MERGED: OK")