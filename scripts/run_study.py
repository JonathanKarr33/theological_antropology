#!/usr/bin/env python3
"""
Run the theological-anthropology LLM study.

Design (see README.md + questions.json):
  10 items × 2 framings (neutral / catholic) × 5 runs  = 100 cells per model

Backends (local MLX):
  --models qwen7b llama phi4mini

Examples (from repo root):
  .venv/bin/python scripts/run_study.py --models qwen7b --items Q01 --framings neutral --runs 1
  .venv/bin/python scripts/run_study.py --models qwen7b llama phi4mini
  .venv/bin/python scripts/run_study.py --models phi4mini
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

from backends import build_backend  # noqa: E402


FIELDNAMES = [
    "model_key",
    "model_id",
    "backend",
    "item_id",
    "theme",
    "framing",
    "run",
    "prompt",
    "response",
    "latency_s",
    "error",
    "timestamp_utc",
    "temp",
    "max_tokens",
]


def load_questions(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def cells(
    items: List[Dict[str, Any]],
    framings: List[str],
    runs: int,
) -> Iterable[Tuple[Dict[str, Any], str, int, str]]:
    for item in items:
        for framing in framings:
            prompt = item.get(framing)
            if not prompt:
                raise KeyError(f"Item {item['id']} missing framing '{framing}'")
            for run in range(1, runs + 1):
                yield item, framing, run, prompt


def load_done(csv_path: Path) -> Set[Tuple[str, str, str, int]]:
    done: Set[Tuple[str, str, str, int]] = set()
    if not csv_path.exists():
        return done
    with csv_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                # Preserve failed/empty attempts in the output, but retry them on resume.
                if row.get("error", "").strip() or not row.get("response", "").strip():
                    continue
                done.add(
                    (
                        row["model_key"],
                        row["item_id"],
                        row["framing"],
                        int(row["run"]),
                    )
                )
            except (KeyError, ValueError):
                continue
    return done


def append_row(csv_path: Path, row: Dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def also_jsonl(jsonl_path: Path, row: Dict[str, Any]) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Theological anthropology LLM study runner")
    p.add_argument(
        "--questions",
        type=Path,
        default=SCRIPTS / "questions.json",
        help="Path to questions.json",
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=["qwen7b"],
        help="Model keys: qwen7b llama phi4mini",
    )
    p.add_argument(
        "--items",
        nargs="+",
        default=None,
        help="Optional subset of item ids, e.g. Q01 Q02",
    )
    p.add_argument(
        "--framings",
        nargs="+",
        default=["neutral", "catholic"],
        choices=["neutral", "catholic"],
    )
    p.add_argument("--runs", type=int, default=5, help="Runs per (item × framing)")
    p.add_argument("--temp", type=float, default=1.0)
    p.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum response tokens; 256 is substantially faster on a Mac",
    )
    p.add_argument(
        "--small",
        action="store_true",
        help="Use smaller MLX builds (Llama 3.2 3B, Gemma 2 2B) if RAM is tight",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "outputs",
        help="Directory for CSV/JSONL outputs",
    )
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing results and re-run (appends duplicates unless you delete first)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned cells only; do not call models",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    bank = load_questions(args.questions)
    items = bank["items"]
    if args.items:
        wanted = set(args.items)
        items = [it for it in items if it["id"] in wanted]
        missing = wanted - {it["id"] for it in items}
        if missing:
            raise SystemExit(f"Unknown item ids: {sorted(missing)}")

    runs = args.runs if args.runs > 0 else int(bank.get("runs_per_cell", 5))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    out_csv = args.out_dir / f"responses_{stamp}.csv"
    out_jsonl = args.out_dir / f"responses_{stamp}.jsonl"

    planned = list(cells(items, args.framings, runs))
    total_cells = len(planned) * len(args.models)
    print(
        f"Study: {len(items)} items × {len(args.framings)} framings × {runs} runs "
        f"= {len(planned)} cells/model; models={args.models}; total={total_cells}",
        flush=True,
    )
    print(f"Output: {out_csv}", flush=True)

    if args.dry_run:
        for model_key in args.models:
            for item, framing, run, prompt in planned:
                print(
                    f"  [{model_key}] {item['id']} {framing} run={run}: {prompt[:70]}..."
                )
        return

    done = set() if args.no_resume else load_done(out_csv)
    if done:
        print(f"Resume: {len(done)} cells already in {out_csv.name}", flush=True)

    work: List[Tuple[str, Dict[str, Any], str, int, str]] = []
    for model_key in args.models:
        for item, framing, run, prompt in planned:
            key = (model_key, item["id"], framing, run)
            if key not in done:
                work.append((model_key, item, framing, run, prompt))

    if not work:
        print("Nothing to do — all cells already complete.", flush=True)
        return

    completed_selected = total_cells - len(work)
    print(
        f"Selected cells: {completed_selected}/{total_cells} complete; "
        f"{len(work)} remaining. Each successful response is saved immediately.",
        flush=True,
    )

    backends: Dict[str, Any] = {}
    active_key: str | None = None
    n_new = 0
    t0 = time.time()

    with tqdm(
        total=total_cells,
        initial=completed_selected,
        desc="Study",
        unit="cell",
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
    ) as pbar:
        for model_key, item, framing, run, prompt in work:
            if model_key not in backends:
                # Keep only one local MLX model resident to avoid Metal OOM.
                if active_key and active_key in backends:
                    prev = backends[active_key]
                    if hasattr(prev, "unload"):
                        pbar.set_postfix_str(f"unloading {active_key}")
                        prev.unload()
                    del backends[active_key]
                pbar.set_postfix_str(f"loading {model_key}")
                backend = build_backend(
                    model_key,
                    max_tokens=args.max_tokens,
                    temp=args.temp,
                    small=args.small,
                )
                if hasattr(backend, "load"):
                    backend.load()
                backends[model_key] = backend
                active_key = model_key

            backend = backends[model_key]
            pbar.set_postfix_str(f"{model_key} {item['id']} {framing} r{run}")

            result = backend.generate(prompt)
            model_id = getattr(backend, "model_id", result.model_id)
            row = {
                "model_key": model_key,
                "model_id": model_id,
                "backend": result.backend,
                "item_id": item["id"],
                "theme": item.get("theme", ""),
                "framing": framing,
                "run": run,
                "prompt": prompt,
                "response": result.text,
                "latency_s": f"{result.latency_s:.3f}",
                "error": result.error or "",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "temp": args.temp,
                "max_tokens": args.max_tokens,
            }
            append_row(out_csv, row)
            also_jsonl(out_jsonl, row)
            done.add((model_key, item["id"], framing, run))
            n_new += 1

            status = "ERR" if result.error else "ok"
            pbar.set_postfix_str(
                f"{model_key} {item['id']} {framing} r{run} {status} {result.latency_s:.1f}s"
            )
            pbar.update(1)

    print(
        f"Done. Wrote {n_new} new cells in {time.time() - t0:.1f}s → {out_csv}",
        flush=True,
    )


if __name__ == "__main__":
    main()
