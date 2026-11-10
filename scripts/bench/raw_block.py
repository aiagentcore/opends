#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Run the raw block mode bench on the target and print its tables.

Runs tasks/bench_raw_block.yaml (scratch file, HOMI stack, the sweep of
bench_raw_block_aisio, teardown) into cijoe-output-bench-raw-block/, then
collects the RESULT lines and prints one markdown table per operation
(read, write): rows are (mode, chunk, qd), columns the file handle and the raw
handle in MiB/s. --report-only skips the run and formats the last output.
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _helpers import ok, run_cijoe

OUT = "cijoe-output-bench-raw-block"
RESULT = re.compile(r"^RESULT (.*)$", re.M)


def collect(out_dir):
    rows = {}
    for p in sorted(Path(out_dir).rglob("cmd_*.output")):
        for m in RESULT.finditer(p.read_text(errors="replace")):
            kv = dict(f.split("=", 1) for f in m.group(1).split())
            key = (kv["op"], kv["mode"], int(kv["chunk"]), int(kv["qd"]))
            rows.setdefault(key, {})[kv["handle"]] = (
                None if "error" in kv else float(kv["MiBs"]))
    return rows


def human(n):
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024 or unit == "GiB":
            return f"{n:g} {unit}"
        n //= 1024
    return str(n)


def table(rows, op):
    lines = [f"\n### {op}\n",
             "| mode   | chunk   | qd | file (MiB/s) | raw (MiB/s) | raw/file |",
             "|--------|---------|----|--------------|-------------|----------|"]
    order = {"sync": 0, "async": 1, "stream": 2, "batch": 3}
    for key in sorted((k for k in rows if k[0] == op),
                      key=lambda k: (order.get(k[1], 9), k[2], k[3])):
        _, mode, chunk, qd = key
        f = rows[key].get("file")
        r = rows[key].get("raw")
        fs = f"{f:.0f}" if f else "error"
        rs = f"{r:.0f}" if r else "error"
        ratio = f"{r / f:.2f}" if f and r else "-"
        lines.append(f"| {mode:<6} | {human(chunk):>7} | {qd:>2} | {fs:>12} | "
                     f"{rs:>11} | {ratio:>8} |")
    return "\n".join(lines)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--report-only", action="store_true",
                    help="Format the last run's output without running.")
args = parser.parse_args()

if not args.report_only:
    run_cijoe("tasks/bench_raw_block.yaml", out=OUT)
    ok("bench_raw_block")

rows = collect(OUT)
if not rows:
    sys.exit(f"no RESULT lines under {OUT}")
for op in ("read", "write"):
    print(table(rows, op))
