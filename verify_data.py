#!/usr/bin/env python3
"""Check every released data file against MANIFEST.sha256.

  python3 verify_data.py        # exits non-zero if anything is missing or altered
"""
import csv, hashlib, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
missing, altered, ok = [], [], 0
with (ROOT / "MANIFEST.sha256").open() as fh:
    for row in csv.DictReader(fh):
        p = ROOT / row["path"]
        if not p.is_file():
            missing.append(row["path"])
        elif hashlib.sha256(p.read_bytes()).hexdigest() != row["sha256"]:
            altered.append(row["path"])
        else:
            ok += 1

print(f"{ok} files verified")
for label, items in (("missing", missing), ("altered", altered)):
    if items:
        print(f"\n{len(items)} {label}:")
        for i in items[:20]:
            print("  ", i)
        if len(items) > 20:
            print(f"   ... and {len(items) - 20} more")
sys.exit(1 if (missing or altered) else 0)
