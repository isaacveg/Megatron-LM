#!/usr/bin/env python3

import argparse
import csv
import json
import os
import re
from typing import Dict, Iterable, List, Optional


TRAIN_RE = re.compile(
    r"iteration\s+(\d+)/\s*(\d+)\s+\|\s+consumed samples:\s+(\d+)\s+\|"
    r"\s+elapsed time per iteration \(ms\):\s+([0-9.]+)\s+\|.*?"
    r"lm loss:\s+([0-9.E+-]+)\s+\|"
)
VAL_RE = re.compile(
    r"validation loss at (.+?) \| .*? value:\s+([0-9.E+-]+)\s+\| .*? PPL:\s+([0-9.E+-]+)\s+\|"
)
VAL_ITER_RE = re.compile(r"iteration\s+(\d+)")

FIELDNAMES = [
    "series_rank",
    "series_id",
    "label",
    "variant",
    "source_log",
    "segment_index",
    "phase",
    "split",
    "eval_scope",
    "metric",
    "iteration",
    "train_iters",
    "consumed_samples",
    "iter_ms",
    "value",
]


def _parse_eval_scope(prefix: str) -> Dict[str, str]:
    if "on test set" in prefix:
        return {"split": "test", "eval_scope": "final_test_set"}
    if "on validation set" in prefix:
        return {"split": "validation", "eval_scope": "final_validation_set"}
    return {"split": "validation", "eval_scope": "periodic_validation"}


def _in_range(iteration: Optional[int], min_iteration: Optional[int], max_iteration: Optional[int]) -> bool:
    if iteration is None:
        return False
    if min_iteration is not None and iteration < min_iteration:
        return False
    if max_iteration is not None and iteration > max_iteration:
        return False
    return True


def _parse_segment_rows(
    log_path: str,
    series_rank: int,
    series_id: str,
    label: str,
    variant: str,
    segment_index: int,
    min_iteration: Optional[int],
    max_iteration: Optional[int],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []

    with open(log_path, "r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()

            train_match = TRAIN_RE.search(line)
            if train_match:
                iteration = int(train_match.group(1))
                if not _in_range(iteration, min_iteration, max_iteration):
                    continue
                rows.append(
                    {
                        "series_rank": series_rank,
                        "series_id": series_id,
                        "label": label,
                        "variant": variant,
                        "source_log": log_path,
                        "segment_index": segment_index,
                        "phase": "train",
                        "split": "train",
                        "eval_scope": "",
                        "metric": "loss",
                        "iteration": iteration,
                        "train_iters": int(train_match.group(2)),
                        "consumed_samples": int(train_match.group(3)),
                        "iter_ms": float(train_match.group(4)),
                        "value": float(train_match.group(5)),
                    }
                )
                continue

            val_match = VAL_RE.search(line)
            if not val_match:
                continue

            prefix = val_match.group(1)
            iter_match = VAL_ITER_RE.search(prefix)
            iteration = int(iter_match.group(1)) if iter_match else None
            if not _in_range(iteration, min_iteration, max_iteration):
                continue

            eval_meta = _parse_eval_scope(prefix)
            common = {
                "series_rank": series_rank,
                "series_id": series_id,
                "label": label,
                "variant": variant,
                "source_log": log_path,
                "segment_index": segment_index,
                "phase": "eval",
                "split": eval_meta["split"],
                "eval_scope": eval_meta["eval_scope"],
                "iteration": iteration,
                "train_iters": "",
                "consumed_samples": "",
                "iter_ms": "",
            }
            rows.append({**common, "metric": "loss", "value": float(val_match.group(2))})
            rows.append({**common, "metric": "ppl", "value": float(val_match.group(3))})

    return rows


def build_rows(manifest_path: str) -> List[Dict[str, object]]:
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    rows: List[Dict[str, object]] = []
    for series_rank, run in enumerate(manifest["runs"]):
        for segment_index, segment in enumerate(run["segments"]):
            rows.extend(
                _parse_segment_rows(
                    log_path=segment["log_path"],
                    series_rank=series_rank,
                    series_id=run["series_id"],
                    label=run["label"],
                    variant=run["variant"],
                    segment_index=segment_index,
                    min_iteration=segment.get("min_iteration"),
                    max_iteration=segment.get("max_iteration"),
                )
            )

    rows.sort(
        key=lambda row: (
            int(row["series_rank"]),
            row["phase"],
            row["split"],
            row["metric"],
            int(row["iteration"]),
            row["eval_scope"],
        )
    )
    return rows


def write_csv(rows: Iterable[Dict[str, object]], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a unified curve table from experiment logs.")
    parser.add_argument("manifest", help="Path to the run manifest JSON.")
    parser.add_argument("output_csv", help="Where to write the combined curve CSV.")
    args = parser.parse_args()

    rows = build_rows(args.manifest)
    write_csv(rows, args.output_csv)
    print(f"Wrote {len(rows)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
