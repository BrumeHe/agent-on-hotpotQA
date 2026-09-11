"""Run one agent over a sampled subset of HotpotQA dev (fullwiki protocol).

Sample protocol matches the ReAct paper: idxs = range(7405) shuffled with
random.Random(seed); take idxs[offset : offset+n]. Paper: seed 233, n 500.
Tuning set: offset 0. Held-out set: offset 500 (never tune on it).

Usage (from project root):
  .venv/Scripts/python src/run_eval.py --method react --n 20 --workers 4
  .venv/Scripts/python src/run_eval.py --method react --n 500 --workers 16 --env rag --tag rag
  .venv/Scripts/python src/run_eval.py --method react --n 500 --workers 16 --env fallback --tag fallback
  .venv/Scripts/python src/run_eval.py --method react --n 500 --workers 16 --prompt concise --tag concise
  .venv/Scripts/python src/run_eval.py --method react --n 500 --workers 16 --selfcheck --tag selfcheck
  .venv/Scripts/python src/run_eval.py --method react --n 500 --workers 16 --memory case --tag case
  # resume a killed run:
  .venv/Scripts/python src/run_eval.py --method react --n 500 --workers 16 --resume results/react_n500_..._.jsonl
"""

import argparse
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

from agents import (ActOnlyAgent, ConciseReActAgent, ContentFallbackReActAgent,
                    CoTAgent, CotScContentReActAgent, CotScReActAgent,
                    DirectAgent, FallbackReActAgent, FallbackV2ReActAgent,
                    RAGReActAgent, ReActAgent, ReActSCAgent,
                    SelfCheckReActAgent, SelfCheckStrictReActAgent)
from evaluate import aggregate, score_answer
from llm import LLMClient
from memory import CaseMemory, CompositeMemory, NullMemory, ReflectionMemory

DATA_PATH = "data/hotpotqa_fullwiki_dev.parquet"


def load_data(path):
    df = pd.read_parquet(path)
    records = []
    for idx, row in enumerate(df.itertuples(index=False)):
        records.append({"idx": idx, "id": row.id, "question": row.question,
                        "answer": row.answer, "type": row.type,
                        "level": row.level})
    return records


def sample_indices(n_total, seed, n, offset):
    idxs = list(range(n_total))
    random.Random(seed).shuffle(idxs)
    return idxs[offset:offset + n]


def load_memory(spec, memory_dir):
    if spec == "none":
        return NullMemory()
    banks = {"case": (CaseMemory, "bank_cases.jsonl"),
             "refl": (ReflectionMemory, "bank_lessons.jsonl")}
    parts = []
    for token in spec.split("+"):
        if token not in banks:
            raise SystemExit(f"unknown --memory component {token!r}")
        cls, fname = banks[token]
        path = os.path.join(memory_dir, fname)
        if not os.path.exists(path):
            raise SystemExit(
                f"missing {path}; run scripts/build_memory.py first")
        parts.append(cls(path))
    return parts[0] if len(parts) == 1 else CompositeMemory(parts)


def make_agent(args, llm):
    if args.method == "direct":
        return DirectAgent(llm)
    if args.method == "cot":
        return CoTAgent(llm)
    if args.method == "actonly":
        return ActOnlyAgent(llm, args.wiki_db, max_steps=args.max_steps)
    memory = load_memory(args.memory, args.memory_dir)
    if args.selfcheck_strict:
        cls = SelfCheckStrictReActAgent
    elif args.cotsc and args.env == "content":
        cls = CotScContentReActAgent  # combo arm
    elif args.cotsc:
        cls = CotScReActAgent
    elif args.sc:
        cls = ReActSCAgent
    elif args.selfcheck:
        cls = SelfCheckReActAgent
    elif args.prompt == "concise":
        cls = ConciseReActAgent
    elif args.env == "rag":
        cls = RAGReActAgent
    elif args.env == "fallback":
        cls = FallbackReActAgent
    elif args.env == "fallback2":
        cls = FallbackV2ReActAgent
    elif args.env == "content":
        cls = ContentFallbackReActAgent
    else:
        cls = ReActAgent
    return cls(llm, args.wiki_db, memory=memory, max_steps=args.max_steps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True,
                    choices=["direct", "cot", "react", "actonly"])
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=233)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=7)
    ap.add_argument("--model", default="deepseek-v4-flash")
    ap.add_argument("--wiki-db", default="data/wiki_full.db")
    ap.add_argument("--data", default=DATA_PATH)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--tag", default="")
    ap.add_argument("--resume", default="",
                    help="append to this partial jsonl, skipping completed ids")
    ap.add_argument("--memory", default="none",
                    help="none | case | refl | case+refl (react only)")
    ap.add_argument("--env", default="base",
                    choices=["base", "rag", "fallback", "fallback2", "content"],
                    help="rag = question-aware obs; fallback = auto-simplify "
                         "retry; fallback2 = follow FTS similar-title hits; "
                         "content = BM25 content search on title miss")
    ap.add_argument("--prompt", default="base", choices=["base", "concise"],
                    help="concise = answer-granularity instruction (react only)")
    ap.add_argument("--selfcheck", action="store_true",
                    help="verify evidence before accepting finish[] (react only)")
    ap.add_argument("--selfcheck-strict", action="store_true",
                    help="cite-evidence verification gate (react only)")
    ap.add_argument("--cotsc", action="store_true",
                    help="CoT self-consistency vote on forced finish "
                         "(react only, paper's ReAct->CoT-SC combo)")
    ap.add_argument("--sc", action="store_true",
                    help="ReAct-SC: 5 trajectories at temp 0.7, majority vote "
                         "(react only)")
    ap.add_argument("--memory-dir", default="memory")
    args = ap.parse_args()

    records = load_data(args.data)
    sel = sample_indices(len(records), args.seed, args.n, args.offset)
    llm = LLMClient(model=args.model)
    agent = make_agent(args, llm)  # stateless per episode -> shared across threads
    os.makedirs(args.out_dir, exist_ok=True)

    if args.resume:
        jsonl_path = args.resume
        base = os.path.splitext(os.path.basename(jsonl_path))[0]
        if not base.startswith(args.method + "_"):
            raise SystemExit(
                f"--resume file {base} does not match --method {args.method}")
        with open(jsonl_path, "rb") as f:
            data = f.read()
        cut = data.rfind(b"\n")
        if cut != len(data) - 1:  # drop a torn final line left by a kill
            with open(jsonl_path, "r+b") as f:
                f.truncate(cut + 1)
            data = data[:cut + 1]
        done_ids = set()
        for line in data.decode("utf-8").splitlines():
            try:
                done_ids.add(json.loads(line)["id"])
            except json.JSONDecodeError:
                continue
        n_before = len(sel)
        sel = [i for i in sel if records[i]["id"] not in done_ids]
        print(f"resume: {n_before - len(sel)} done, {len(sel)} remaining "
              f"-> {jsonl_path}")
    else:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        tag = f"_{args.tag}" if args.tag else ""
        base = (f"{args.method}_n{args.n}_seed{args.seed}_off{args.offset}"
                f"{tag}_{stamp}")
        jsonl_path = os.path.join(args.out_dir, base + ".jsonl")
    write_lock = threading.Lock()

    def work(rec):
        t0 = time.time()
        out = agent.run(rec["question"])
        scores = score_answer(out["answer"], rec["answer"])
        row = {**rec, "pred": out["answer"], **scores,
               "steps": out["steps"], "n_calls": out["n_calls"],
               "n_badcalls": out["n_badcalls"], "searches": out.get("searches", 0),
               "prompt_tokens": out["prompt_tokens"],
               "completion_tokens": out["completion_tokens"],
               "latency_s": round(time.time() - t0, 2),
               "forced_finish": out.get("forced_finish", False),
               "steps_detail": out.get("steps_detail"),
               "trace": out["trace"]}
        with write_lock, open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row

    rows = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(work, records[i]) for i in sel]
        with tqdm(total=len(futs), desc=args.method) as bar:
            for fut in as_completed(futs):
                rows.append(fut.result())
                bar.update(1)
                em = sum(r["em"] for r in rows) / len(rows)
                bar.set_postfix(em=f"{em:.3f}")

    if args.resume:  # aggregate over the whole file, not just this session
        with open(jsonl_path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    react_cfg = ({"memory": args.memory, "env": args.env,
                  "prompt": args.prompt, "selfcheck": args.selfcheck,
                  "selfcheck_strict": args.selfcheck_strict, "cotsc": args.cotsc,
                  "sc": args.sc}
                 if args.method == "react" else {})
    summary = {"method": args.method, "model": args.model, "seed": args.seed,
               "offset": args.offset, "max_steps": args.max_steps,
               **react_cfg,
               **aggregate(rows), "elapsed_s": round(time.time() - t0, 1),
               "llm": llm.summary()}
    sum_path = os.path.join(args.out_dir, base + "_summary.json")
    with open(sum_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"traces -> {jsonl_path}")


if __name__ == "__main__":
    main()
