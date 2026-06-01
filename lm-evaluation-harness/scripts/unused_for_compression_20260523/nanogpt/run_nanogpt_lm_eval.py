#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path

from lm_eval import evaluator, utils


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run lm-eval on local nanoGPT checkpoint.")
    p.add_argument("--ckpt_path", type=str, required=True, help="Path to ckpt file or run dir.")
    p.add_argument("--nanogpt_repo_root", type=str, default="", help="nanoGPT repo root.")
    p.add_argument("--tasks", type=str, default="hellaswag", help="Comma-separated task names.")
    p.add_argument("--output_path", type=str, required=True, help="JSON output path.")
    p.add_argument("--batch_size", type=str, default="1")
    p.add_argument("--max_batch_size", type=int, default=64)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="bf16")
    p.add_argument("--num_fewshot", type=int, default=0)
    p.add_argument("--limit", type=float, default=None)
    p.add_argument("--max_length", type=int, default=None)
    p.add_argument("--xsa_forward_only", type=str, default="", help="Optional override: true/false.")
    p.add_argument("--xsa_forward_target", type=str, default="", help="Optional override.")
    p.add_argument("--xsa_forward_ref", type=str, default="", help="Optional override.")
    p.add_argument("--xsa_forward_space", type=str, default="", help="Optional override.")
    p.add_argument("--xsa_forward_op", type=str, default="", help="Optional override.")
    p.add_argument("--xsa_forward_alpha", type=float, default=None, help="Optional override.")
    return p.parse_args()


def emit_stage(message: str) -> None:
    print(message, flush=True)


class _Heartbeat:
    def __init__(self, label: str, interval_sec: float = 60.0) -> None:
        self.label = label
        self.interval_sec = interval_sec
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._start_time = time.time()

    def _run(self) -> None:
        while not self._stop.wait(self.interval_sec):
            elapsed = time.time() - self._start_time
            emit_stage(
                f"[HEARTBEAT] {self.label} alive elapsed_sec={elapsed:.1f}"
            )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)


def main() -> None:
    args = parse_args()
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    model_kwargs = dict(
        ckpt_path=args.ckpt_path,
        nanogpt_repo_root=(args.nanogpt_repo_root or None),
        dtype=args.dtype,
        max_length=args.max_length,
    )
    if args.xsa_forward_only != "":
        model_kwargs["xsa_forward_only"] = args.xsa_forward_only
    if args.xsa_forward_target:
        model_kwargs["xsa_forward_target"] = args.xsa_forward_target
    if args.xsa_forward_ref:
        model_kwargs["xsa_forward_ref"] = args.xsa_forward_ref
    if args.xsa_forward_space:
        model_kwargs["xsa_forward_space"] = args.xsa_forward_space
    if args.xsa_forward_op:
        model_kwargs["xsa_forward_op"] = args.xsa_forward_op
    if args.xsa_forward_alpha is not None:
        model_kwargs["xsa_forward_alpha"] = args.xsa_forward_alpha

    emit_stage(
        f"[STAGE] simple_evaluate_start model=nanogpt ckpt={args.ckpt_path} device={args.device} dtype={args.dtype} batch_size={args.batch_size} tasks={','.join(tasks)} num_tasks={len(tasks)}"
    )
    st = time.time()
    heartbeat = _Heartbeat("simple_evaluate", interval_sec=60.0)
    heartbeat.start()
    try:
        results = evaluator.simple_evaluate(
            model="nanogpt",
            model_args=model_kwargs,
            tasks=tasks,
            num_fewshot=args.num_fewshot,
            limit=args.limit,
            batch_size=args.batch_size,
            max_batch_size=args.max_batch_size,
            device=args.device,
            log_samples=False,
        )
    finally:
        heartbeat.stop()
    elapsed = time.time() - st
    emit_stage(f"[STAGE] simple_evaluate_done elapsed_sec={elapsed:.1f}")

    if isinstance(results.get("config"), dict):
        results["config"]["tasks"] = tasks
        results["config"]["ckpt_path"] = str(Path(args.ckpt_path).expanduser().resolve())
        results["config"]["dtype"] = args.dtype
        results["config"]["batch_size"] = args.batch_size
        results["config"]["max_batch_size"] = args.max_batch_size
    out_path = Path(args.output_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    emit_stage(f"[STAGE] result_write_start path={out_path}")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=2,
            default=utils.handle_non_serializable,
        )
    emit_stage(f"[STAGE] result_write_done path={out_path}")

    result_block = results.get("results", {}) if isinstance(results, dict) else {}
    if isinstance(result_block, dict):
        emit_stage("[INFO] Metric summary:")
        for task_name, task_metrics in result_block.items():
            if not isinstance(task_metrics, dict):
                continue
            preferred = [
                "acc_norm,none",
                "acc,none",
                "exact_match,strict-match",
                "exact_match,none",
                "f1,none",
            ]
            shown = False
            for key in preferred:
                val = task_metrics.get(key)
                if isinstance(val, (int, float)):
                    emit_stage(f"[RESULT] task={task_name} metric={key} value={val:.6f}")
                    shown = True
                    break
            if shown:
                continue
            for key, val in task_metrics.items():
                if isinstance(val, (int, float)) and "_stderr" not in key:
                    emit_stage(f"[RESULT] task={task_name} metric={key} value={val:.6f}")
                    break

    try:
        print(utils.make_table(results))
    except ModuleNotFoundError as exc:
        if "pytablewriter" in str(exc):
            print("[WARN] pytablewriter is not installed; skip pretty table output.")
        else:
            raise
    emit_stage(f"[INFO] Result JSON: {out_path}")


if __name__ == "__main__":
    main()
