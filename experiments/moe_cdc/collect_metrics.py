#!/usr/bin/env python3

import argparse
import csv
import os
import re
from statistics import mean


TRAIN_RE = re.compile(
    r"iteration\s+(\d+)/\s*(\d+)\s+\|\s+consumed samples:\s+(\d+)\s+\|"
    r"\s+elapsed time per iteration \(ms\):\s+([0-9.]+)\s+\|.*?"
    r"lm loss:\s+([0-9.E+-]+)\s+\|"
)
VAL_RE = re.compile(
    r"validation loss at (.+?) \| .*? value:\s+([0-9.E+-]+)\s+\| .*? PPL:\s+([0-9.E+-]+)\s+\|"
)
VAL_ITER_RE = re.compile(r"iteration\s+(\d+)")
CDC_RE = re.compile(r"\[CDC\] Communication:\s+([0-9.]+)\s+MB in\s+([0-9.]+)s")


def _safe_mean(values):
    return mean(values) if values else None


def _fmt(value):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def parse_log(path):
    train_rows = []
    val_rows = []
    cdc_rows = []

    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()

            match = TRAIN_RE.search(line)
            if match:
                train_rows.append(
                    {
                        "iteration": int(match.group(1)),
                        "train_iters": int(match.group(2)),
                        "consumed_samples": int(match.group(3)),
                        "iter_ms": float(match.group(4)),
                        "train_loss": float(match.group(5)),
                    }
                )
                continue

            match = VAL_RE.search(line)
            if match:
                prefix = match.group(1)
                iter_match = VAL_ITER_RE.search(prefix)
                val_rows.append(
                    {
                        "prefix": prefix,
                        "iteration": int(iter_match.group(1)) if iter_match else None,
                        "val_loss": float(match.group(2)),
                        "val_ppl": float(match.group(3)),
                    }
                )
                continue

            match = CDC_RE.search(line)
            if match:
                cdc_rows.append(
                    {
                        "comm_mb": float(match.group(1)),
                        "comm_s": float(match.group(2)),
                    }
                )

    latest_train = train_rows[-1] if train_rows else {}
    latest_val = val_rows[-1] if val_rows else {}
    metrics = {
        "final_train_iter": latest_train.get("iteration"),
        "train_iters": latest_train.get("train_iters"),
        "final_train_loss": latest_train.get("train_loss"),
        "last_iter_ms": latest_train.get("iter_ms"),
        "avg_iter_ms": _safe_mean([row["iter_ms"] for row in train_rows]),
        "last_eval_iter": latest_val.get("iteration"),
        "last_eval_loss": latest_val.get("val_loss"),
        "last_eval_ppl": latest_val.get("val_ppl"),
        "avg_cdc_comm_mb": _safe_mean([row["comm_mb"] for row in cdc_rows]),
        "avg_cdc_comm_s": _safe_mean([row["comm_s"] for row in cdc_rows]),
        "max_cdc_comm_mb": max((row["comm_mb"] for row in cdc_rows), default=None),
    }
    return metrics


def append_csv(path, row):
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not file_exists or os.path.getsize(path) == 0:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Summarize MoE CDC training logs.")
    parser.add_argument("log_path", help="Path to a Megatron training log.")
    parser.add_argument("--run-name", default=None, help="Override run name.")
    parser.add_argument("--variant", default=None, help="Experiment variant label.")
    parser.add_argument("--seed", default=None, help="Seed label for the result row.")
    parser.add_argument("--cdc-algorithm", default=None, help="Optional CDC algorithm label.")
    parser.add_argument("--cdc-param-mode", default=None, help="Optional CDC param mode label.")
    parser.add_argument("--cdc-sync-interval", default=None, help="Optional CDC sync interval label.")
    parser.add_argument("--cdc-num-shards", default=None, help="Optional CDC num shards label.")
    parser.add_argument("--cdc-alpha", default=None, help="Optional CDC alpha label.")
    parser.add_argument("--notes", default="", help="Optional notes for CSV output.")
    parser.add_argument("--append-csv", default=None, help="Append a result row to this CSV.")
    args = parser.parse_args()

    metrics = parse_log(args.log_path)
    run_name = args.run_name or os.path.splitext(os.path.basename(args.log_path))[0]
    variant = args.variant or run_name

    print(f"run_name: {_fmt(run_name)}")
    print(f"variant: {_fmt(variant)}")
    print(f"final_train_iter: {_fmt(metrics['final_train_iter'])}")
    print(f"final_train_loss: {_fmt(metrics['final_train_loss'])}")
    print(f"last_eval_iter: {_fmt(metrics['last_eval_iter'])}")
    print(f"last_eval_loss: {_fmt(metrics['last_eval_loss'])}")
    print(f"last_eval_ppl: {_fmt(metrics['last_eval_ppl'])}")
    print(f"avg_iter_ms: {_fmt(metrics['avg_iter_ms'])}")
    print(f"last_iter_ms: {_fmt(metrics['last_iter_ms'])}")
    print(f"avg_cdc_comm_mb: {_fmt(metrics['avg_cdc_comm_mb'])}")
    print(f"avg_cdc_comm_s: {_fmt(metrics['avg_cdc_comm_s'])}")
    print(f"max_cdc_comm_mb: {_fmt(metrics['max_cdc_comm_mb'])}")

    if args.append_csv:
        row = {
            "run_name": run_name,
            "variant": variant,
            "seed": args.seed or "",
            "train_iters": metrics["train_iters"] or "",
            "cdc_algorithm": args.cdc_algorithm or "",
            "cdc_param_mode": args.cdc_param_mode or "",
            "cdc_sync_interval": args.cdc_sync_interval or "",
            "cdc_num_shards": args.cdc_num_shards or "",
            "cdc_alpha": args.cdc_alpha or "",
            "final_train_iter": metrics["final_train_iter"] or "",
            "final_train_loss": metrics["final_train_loss"] or "",
            "last_eval_iter": metrics["last_eval_iter"] or "",
            "last_eval_loss": metrics["last_eval_loss"] or "",
            "last_eval_ppl": metrics["last_eval_ppl"] or "",
            "avg_iter_ms": metrics["avg_iter_ms"] or "",
            "last_iter_ms": metrics["last_iter_ms"] or "",
            "avg_cdc_comm_mb": metrics["avg_cdc_comm_mb"] or "",
            "avg_cdc_comm_s": metrics["avg_cdc_comm_s"] or "",
            "max_cdc_comm_mb": metrics["max_cdc_comm_mb"] or "",
            "max_memory_gb": "",
            "notes": args.notes,
        }
        append_csv(args.append_csv, row)


if __name__ == "__main__":
    main()
