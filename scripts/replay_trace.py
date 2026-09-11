"""Inspect and replay agent traces from a results jsonl.

Usage (from project root):
  .venv/Scripts/python scripts/replay_trace.py results/react_xxx.jsonl --list
  .venv/Scripts/python scripts/replay_trace.py results/react_xxx.jsonl --fail
  .venv/Scripts/python scripts/replay_trace.py results/react_xxx.jsonl --signals
  .venv/Scripts/python scripts/replay_trace.py results/react_xxx.jsonl --id 5a758...
  .venv/Scripts/python scripts/replay_trace.py a.jsonl --diff b.jsonl --id 5a758...
"""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from hotpot_evaluate_v1 import normalize_answer  # noqa: E402

# Fallback parser for old runs without steps_detail: cut the few-shot prefix
# (everything up to the last "Question: ") and split Thought/Action/Observation.
STEP_RE = re.compile(
    r"(?:Thought (\d+): (.*?)\n)?Action (\d+): (.*?)\nObservation \3: (.*?)"
    r"(?=(?:\nThought \d+: )|(?:\nAction \d+: )|\Z)", re.S)

WRAPPER_RE = re.compile(r"^(the\s+)?answer\s*(is|:)?\s*", re.I)


def load(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def steps_of(row):
    if row.get("steps_detail"):
        return row["steps_detail"]
    tail = row.get("trace", "").rsplit("Question: ", 1)[-1]
    tail = tail.split("\n", 1)[1] if "\n" in tail else ""
    steps = []
    for m in STEP_RE.finditer(tail):
        steps.append({"i": int(m.group(3)), "thought": m.group(2) or "",
                      "action": m.group(4).strip(), "obs": m.group(5),
                      "badcall": False})
    return steps


def trunc(s, n):
    s = " ".join(str(s).split())
    return s if n <= 0 or len(s) <= n else s[:n] + "..."


def forced_finish(row):
    ff = row.get("forced_finish")
    if ff is None:  # legacy rows: forced finish => env.steps exceeds max_steps
        ff = (row.get("steps") or 0) >= 8
    return bool(ff)


def find(rows, qid=None, idx=None):
    for r in rows:
        if qid is not None and r["id"] == qid:
            return r
        if idx is not None and r["idx"] == idx:
            return r
    return None


def show(row, obs_chars=400):
    print(f"# idx {row['idx']} | {row['id']} | {row['type']}/{row['level']}")
    print(f"Q: {row['question']}")
    print(f"gold: {row['answer']}")
    print(f"pred: {row.get('pred')}  | EM {row.get('em')} "
          f"F1 {(row.get('f1') or 0):.3f}")
    print(f"steps={row.get('steps')} calls={row.get('n_calls')} "
          f"badcalls={row.get('n_badcalls')} searches={row.get('searches')} "
          f"forced_finish={forced_finish(row)} "
          f"tokens={row.get('prompt_tokens')}+{row.get('completion_tokens')} "
          f"latency={row.get('latency_s')}s")
    for s in steps_of(row):
        print(f"\n[{s['i']}]{' (badcall)' if s.get('badcall') else ''}")
        if s.get("thought"):
            print(f"  Thought: {s['thought']}")
        print(f"  Action:  {s['action']}")
        print(f"  Obs:     {trunc(s.get('obs', ''), obs_chars)}")
        if s.get("raw"):
            print(f"  Raw:     {trunc(s['raw'], 200)}")
    print()


def list_rows(rows, fail_only=False):
    for r in sorted(rows, key=lambda r: r["idx"]):
        if fail_only and r.get("em"):
            continue
        print(f"{r['idx']:>5}  em={r.get('em')} f1={(r.get('f1') or 0):.2f} "
              f"st={r.get('steps')} sch={r.get('searches')} "
              f"bad={r.get('n_badcalls')} ff={int(forced_finish(r))} "
              f"pred={trunc(r.get('pred'), 40)!r} "
              f"gold={trunc(r['answer'], 40)!r}")


def signals(rows):
    """Failure triage counters over em==0 rows (heuristic, not official EM)."""
    fail = [r for r in rows if not r.get("em")]
    c = {"failures": len(fail), "forced_finish": 0, "any_badcall": 0,
         "zero_searches": 0, "search_miss": 0, "format_flippable": 0}
    for r in fail:
        if forced_finish(r):
            c["forced_finish"] += 1
        if r.get("n_badcalls"):
            c["any_badcall"] += 1
        if not r.get("searches"):
            c["zero_searches"] += 1
        steps = steps_of(r)
        if any(str(s.get("obs", "")).startswith("Could not find")
               for s in steps):
            c["search_miss"] += 1
        stripped = WRAPPER_RE.sub("", str(r.get("pred", "")).strip())
        if (stripped != str(r.get("pred", "")).strip()
                and normalize_answer(stripped) == normalize_answer(r["answer"])):
            c["format_flippable"] += 1
    print(f"failures: {c['failures']} / {len(rows)}")
    for k in ("forced_finish", "any_badcall", "zero_searches",
              "search_miss", "format_flippable"):
        pct = c[k] / len(fail) * 100 if fail else 0
        print(f"  {k:<18} {c[k]:>4}  ({pct:.0f}% of failures)")


def diff(path_a, path_b, qid=None, idx=None):
    ra = find(load(path_a), qid, idx)
    rb = find(load(path_b), qid, idx)
    for name, r in (("A " + os.path.basename(path_a), ra),
                    ("B " + os.path.basename(path_b), rb)):
        if r is None:
            print(f"{name}: question not found")
            continue
        print(f"--- {name}: EM {r.get('em')} F1 {(r.get('f1') or 0):.3f} "
              f"steps={r.get('steps')} searches={r.get('searches')}")
        for s in steps_of(r):
            print(f"  [{s['i']}] {trunc(s['action'], 60):<60} "
                  f"-> {trunc(s.get('obs', ''), 90)}")
        print(f"  pred={r.get('pred')!r}  gold={r['answer']!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--fail", action="store_true", help="list em==0 rows only")
    ap.add_argument("--signals", action="store_true",
                    help="failure triage counters")
    ap.add_argument("--id")
    ap.add_argument("--idx", type=int)
    ap.add_argument("--diff", metavar="OTHER_JSONL")
    ap.add_argument("--obs-chars", type=int, default=400,
                    help="observation truncation per step, 0 = full")
    args = ap.parse_args()

    if args.diff:
        diff(args.jsonl, args.diff, args.id, args.idx)
        return
    rows = load(args.jsonl)
    if args.id is not None or args.idx is not None:
        row = find(rows, args.id, args.idx)
        if row is None:
            sys.exit("question not found")
        show(row, args.obs_chars)
    elif args.signals:
        signals(rows)
    else:
        list_rows(rows, fail_only=args.fail)


if __name__ == "__main__":
    main()
