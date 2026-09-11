"""Mine memory banks from tuning-slice traces (NEVER the held-out slice).

Cases:   a failed search[X] followed later by a successful search[Y] in the
         same episode -> {"miss": X, "fix": Y}  (rule-based, no LLM)
Lessons: one LLM call per failed episode -> a one-line transferable rule
         (Reflexion-style, --lessons flag)

Usage (from project root):
  .venv/Scripts/python scripts/build_memory.py results/react_n500_seed233_off0_baseline_*.jsonl
  .venv/Scripts/python scripts/build_memory.py results/react_n500_..._baseline.jsonl --lessons
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from memory import mine_cases  # noqa: E402
from llm import LLMClient  # noqa: E402
from replay_trace import load, steps_of  # noqa: E402  (sibling script)

LESSON_PROMPT = """A HotpotQA agent searched Wikipedia and answered wrong.

Question: {question}
Gold answer: {gold}
Agent answer: {pred}
Trace (action -> observation):
{trace}

Write ONE general rule (max 25 words) that would have avoided this mistake \
and that transfers to other questions. Output only the rule."""


def compress_trace(row, obs_chars=120, total_chars=2500):
    lines = []
    for s in steps_of(row):
        obs = " ".join(str(s.get("obs", "")).split())[:obs_chars]
        lines.append(f"{s.get('action', '')} -> {obs}")
    return "\n".join(lines)[:total_chars]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", help="tuning-slice results file (off0 only!)")
    ap.add_argument("--out-dir", default="memory")
    ap.add_argument("--lessons", action="store_true",
                    help="also mine LLM lessons from failed episodes")
    ap.add_argument("--model", default="deepseek-v4-flash")
    args = ap.parse_args()

    if "_off0_" not in os.path.basename(args.jsonl):
        print("[warn] input does not look like the tuning slice (off0); "
              "never mine memory from held-out runs", file=sys.stderr)

    rows = load(args.jsonl)
    os.makedirs(args.out_dir, exist_ok=True)

    # --- case bank ---
    counts = {}
    for row in rows:
        row = dict(row, steps_detail=steps_of(row))  # legacy fallback parse
        for c in mine_cases(row):
            key = (c["miss"].strip().lower(), c["fix"].strip().lower())
            entry = counts.setdefault(
                key, {"miss": c["miss"], "fix": c["fix"], "n": 0})
            entry["n"] += 1
    cases = sorted(counts.values(), key=lambda e: -e["n"])
    cases_path = os.path.join(args.out_dir, "bank_cases.jsonl")
    with open(cases_path, "w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"{len(cases)} unique cases -> {cases_path}")

    # --- lesson bank ---
    if not args.lessons:
        return
    fails = [r for r in rows if not r.get("em")]
    llm = LLMClient(model=args.model)
    lessons_path = os.path.join(args.out_dir, "bank_lessons.jsonl")
    with open(lessons_path, "w", encoding="utf-8") as f:
        for i, row in enumerate(fails, 1):
            prompt = LESSON_PROMPT.format(
                question=row["question"], gold=row["answer"],
                pred=row.get("pred", ""), trace=compress_trace(row))
            res = llm.generate(prompt, stop=None, max_tokens=60)
            lesson = res.text.strip().split("\n")[0].strip()
            if lesson:
                f.write(json.dumps({"qid": row["id"], "question": row["question"],
                                    "lesson": lesson}, ensure_ascii=False) + "\n")
            print(f"\rlessons {i}/{len(fails)}", end="", flush=True)
    print(f"\n{len(fails)} failed episodes mined -> {lessons_path}")


if __name__ == "__main__":
    main()
