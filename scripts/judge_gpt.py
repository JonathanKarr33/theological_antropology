#!/usr/bin/env python3
"""
LLM-as-judge via OpenAI (cheap GPT-5 mini) — parallel rise-style loop.

  for each pending response (up to --workers in flight, default 10):
    1. call OpenAI chat completions (default: gpt-5-mini)
    2. parse JSON judgment
    3. append immediately to judgments_*.csv / .jsonl
    4. advance tqdm progress bar
    5. STOP the whole run on rate limit / quota / billing errors

Rubric / system prompt: README.md section between
  <!-- JUDGE_PROMPT_START --> … <!-- JUDGE_PROMPT_END -->

Examples:
  # smoke
  .venv/bin/python scripts/judge_gpt.py --limit 10

  # all pending (10 parallel API calls)
  .venv/bin/python scripts/judge_gpt.py

  # tune parallelism
  .venv/bin/python scripts/judge_gpt.py --workers 10
"""

from __future__ import annotations

import argparse
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

from judge_lib import (  # noqa: E402
    build_judge_prompt,
    extract_json,
    filter_rows,
    is_judgable,
    judgment_id,
    load_done,
    load_questions,
    load_responses,
    parse_judgment_payload,
    response_key,
    upsert_judgment,
)
from openai_api import (  # noqa: E402
    DEFAULT_API_MODEL,
    ApiResult,
    is_stop_error,
    resolve_api_key,
    run_openai_chat,
)

DEFAULT_PROMPT_MD = REPO_ROOT / "README.md"
DEFAULT_WORKERS = 10
_JUDGE_BLOCK = re.compile(
    r"<!--\s*JUDGE_PROMPT_START\s*-->(.*?)<!--\s*JUDGE_PROMPT_END\s*-->",
    re.DOTALL | re.IGNORECASE,
)
_write_lock = threading.Lock()


def load_system_prompt(path: Path) -> str:
    if not path.is_file():
        return (
            "You are scoring theological anthropology for an empirical study. "
            "Output only one JSON object matching the requested schema."
        )
    text = path.read_text(encoding="utf-8")
    m = _JUDGE_BLOCK.search(text)
    if m:
        return m.group(1).strip()
    return text.strip()


def newest_responses(out_dir: Path) -> Path:
    cands = sorted(out_dir.glob("responses_*.csv"), key=lambda p: p.stat().st_mtime)
    if not cands:
        raise SystemExit(f"No responses_*.csv under {out_dir}")
    return cands[-1]


def score_one(
    row: Dict[str, str],
    *,
    items: Dict[str, Dict[str, Any]],
    system: str,
    model: str,
    timeout_s: int,
    attempts: int = 3,
) -> Dict[str, Any]:
    item = items.get(row["item_id"])
    jid = judgment_id(row)
    if item is None:
        judgment = parse_judgment_payload(
            None, "", 0.0, f"unknown item_id {row['item_id']}"
        )
        return {
            "judgment_id": jid,
            "model_key": row["model_key"],
            "item_id": row["item_id"],
            "framing": row["framing"],
            "run": row["run"],
            "theme": row.get("theme", ""),
            "prompt": row["prompt"],
            "response": row["response"],
            "judge_model": model,
            "judge_backend": "openai",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "stop": False,
            **judgment,
        }

    prompt = build_judge_prompt(
        item=item,
        framing=row["framing"],
        prompt=row["prompt"],
        response=row["response"],
        jid=jid,
    )

    last_result: Optional[ApiResult] = None
    last_judgment: Dict[str, Any] = {}
    total_latency = 0.0
    for attempt in range(1, max(1, attempts) + 1):
        t_call = time.time()
        result = run_openai_chat(
            prompt,
            system=system,
            model=model,
            timeout_s=timeout_s,
        )
        total_latency += time.time() - t_call
        last_result = result
        if result.stop_run:
            break
        judgment = parse_judgment_payload(
            extract_json(result.text) if result.text else None,
            result.text,
            total_latency,
            result.error or "",
        )
        last_judgment = judgment
        # Retry empty / unparseable replies (common with gpt-5-mini).
        retryable = (not result.text) or bool(judgment.get("error"))
        if not retryable:
            break
        if attempt < attempts and not is_stop_error(
            result.error or "", http_status=result.http_status
        ):
            time.sleep(0.4 * attempt)
            continue
        break

    assert last_result is not None
    if not last_judgment:
        last_judgment = parse_judgment_payload(
            extract_json(last_result.text) if last_result.text else None,
            last_result.text,
            total_latency,
            last_result.error or "",
        )
    stop = bool(last_result.stop_run) or is_stop_error(
        last_result.error or "", http_status=last_result.http_status
    )
    return {
        "judgment_id": jid,
        "model_key": row["model_key"],
        "item_id": row["item_id"],
        "framing": row["framing"],
        "run": row["run"],
        "theme": row.get("theme", item.get("theme", "")),
        "prompt": row["prompt"],
        "response": row["response"],
        "judge_model": last_result.model or model,
        "judge_backend": "openai",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "stop": stop,
        **last_judgment,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--responses", type=Path, default=None)
    p.add_argument("--questions", type=Path, default=SCRIPTS / "questions.json")
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs")
    p.add_argument("--prompt-md", type=Path, default=DEFAULT_PROMPT_MD)
    p.add_argument("--models", nargs="+", default=None, help="Filter study model_key(s)")
    p.add_argument("--items", nargs="+", default=None)
    p.add_argument(
        "--framings",
        nargs="+",
        choices=["neutral", "catholic"],
        default=None,
    )
    p.add_argument("--limit", type=int, default=0, help="Max judgments this run (0=all)")
    p.add_argument("--timeout", type=int, default=180)
    p.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Parallel OpenAI requests (default: {DEFAULT_WORKERS})",
    )
    p.add_argument(
        "--model",
        default=DEFAULT_API_MODEL,
        help=f"OpenAI judge model (default: {DEFAULT_API_MODEL})",
    )
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    responses_path = args.responses or newest_responses(args.out_dir)
    items = load_questions(args.questions)
    system = load_system_prompt(args.prompt_md)
    workers = max(1, int(args.workers))

    class NS:
        pass

    filt = NS()
    filt.models = args.models
    filt.items = args.items
    filt.framings = args.framings

    rows = filter_rows([r for r in load_responses(responses_path) if is_judgable(r)], filt)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    out_csv = args.out_dir / f"judgments_{stamp}.csv"
    out_jsonl = args.out_dir / f"judgments_{stamp}.jsonl"

    already: Set[Tuple[str, str, str, str]] = set() if args.no_resume else load_done(out_csv)
    work: List[Dict[str, str]] = [r for r in rows if response_key(r) not in already]
    if args.limit and args.limit > 0:
        work = work[: args.limit]

    key = resolve_api_key()
    print(
        f"Judge GPT: {len(rows)} eligible; {len(already)} done; "
        f"{len(work)} pending → {out_csv}",
        flush=True,
    )
    print(
        f"Model: {args.model} | workers={workers} | OPENAI_API_KEY: "
        f"{'found' if key else 'MISSING'} | rubric: {args.prompt_md}",
        flush=True,
    )
    if already:
        print(
            f"Resume: skipping {len(already)} already-scored; "
            "each new judgment is saved immediately.",
            flush=True,
        )

    if args.dry_run:
        for r in work[:20]:
            print(f"  would judge {r['model_key']} {r['item_id']} {r['framing']} r{r['run']}")
        if len(work) > 20:
            print(f"  ... and {len(work) - 20} more")
        return

    if not key:
        raise SystemExit(
            "Set OPENAI_API_KEY (env or repo .env), then re-run:\n"
            "  echo 'OPENAI_API_KEY=sk-...' > .env\n"
            "  .venv/bin/python scripts/judge_gpt.py --limit 10"
        )

    if not work:
        print("Nothing to judge.", flush=True)
        return

    n_ok = n_err = 0
    stopped_early = False
    t0 = time.time()

    with tqdm(
        total=len(work),
        desc=f"Judge ({args.model}×{workers})",
        unit="resp",
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
    ) as pbar:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures: Dict[Future, str] = {}
            for row in work:
                fut = pool.submit(
                    score_one,
                    row,
                    items=items,
                    system=system,
                    model=args.model,
                    timeout_s=args.timeout,
                )
                futures[fut] = judgment_id(row)

            for fut in as_completed(futures):
                try:
                    out = fut.result()
                except Exception as e:
                    jid = futures[fut]
                    out = {
                        "judgment_id": jid,
                        "model_key": "",
                        "item_id": "",
                        "framing": "",
                        "run": "",
                        "theme": "",
                        "prompt": "",
                        "response": "",
                        "judge_model": args.model,
                        "judge_backend": "openai",
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "alignment": "",
                        "rationale": "",
                        "judge_raw": "",
                        "error": str(e),
                        "latency_s": "0.000",
                        "stop": is_stop_error(str(e)),
                        "invokes_religion": "",
                        "capacity_language": "",
                        "relational_language": "",
                        "ranks_persons": "",
                        "personhood_disclaimer": "",
                    }

                stop = bool(out.pop("stop", False))
                with _write_lock:
                    # Upsert: drop any prior failure/success for this cell, then write.
                    upsert_judgment(out_csv, out_jsonl, out)

                if out.get("error"):
                    n_err += 1
                    pbar.set_postfix_str(f"ERR {str(out['error'])[:50]}")
                else:
                    n_ok += 1
                    pbar.set_postfix_str(
                        f"saved {out.get('judgment_id')} → {out.get('alignment')}"
                    )
                pbar.update(1)

                if stop:
                    stopped_early = True
                    print(
                        "\nSTOPPED: rate limit / quota / billing error "
                        f"(cancelling remaining workers).\n  {out.get('error', '')[:300]}",
                        flush=True,
                    )
                    for other in futures:
                        other.cancel()
                    break

    print(
        f"Done. ok={n_ok} err={n_err}"
        f"{' (stopped early)' if stopped_early else ''} "
        f"in {time.time() - t0:.1f}s → {out_csv}",
        flush=True,
    )
    if stopped_early:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
