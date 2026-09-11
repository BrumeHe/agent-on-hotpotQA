"""Cross-episode memory: a pluggable scaffold component with three hooks.

  inject(question)              -> str, inserted into the prompt before step 1
  on_search_miss(entity)        -> str | None, appended to a "Could not find"
                                   observation
  on_episode_end(row)           -> list of learned cases; the same code path
                                   is used for offline bank mining

Discipline: banks are mined ONLY from the tuning slice (seed 233, offset 0)
and frozen before any held-out run. NullMemory makes the agent loop
byte-identical to the no-memory baseline.
"""

import json
import math
import re
from collections import Counter

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(s):
    return TOKEN_RE.findall(str(s).lower())


class BM25:
    """Tiny stdlib BM25 over a small in-memory corpus (banks, one page)."""

    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs = [tokenize(d) for d in docs]
        self.dl = [len(d) for d in self.docs]
        self.avgdl = sum(self.dl) / max(1, len(self.dl))
        self.n = len(self.docs)
        df = Counter()
        for d in self.docs:
            for t in set(d):
                df[t] += 1
        self.idf = {t: math.log(1 + (self.n - f + 0.5) / (f + 0.5))
                    for t, f in df.items()}
        self.tf = [Counter(d) for d in self.docs]

    def top(self, query, k=1, min_score=0.0):
        """Return [(score, doc_idx)] with score > min_score, best first."""
        q = tokenize(query)
        scored = []
        for i in range(self.n):
            s = 0.0
            for t in q:
                if t not in self.idf:
                    continue
                f = self.tf[i][t]
                s += (self.idf[t] * f * (self.k1 + 1)
                      / (f + self.k1 * (1 - self.b + self.b * self.dl[i]
                                        / self.avgdl)))
            if s > min_score:
                scored.append((s, i))
        scored.sort(reverse=True)
        return scored[:k]


class NullMemory:
    name = "none"

    def inject(self, question):
        return ""

    def on_search_miss(self, entity):
        return None

    def on_episode_end(self, row):
        return []


def mine_cases(row):
    """Extract (failed search entity -> next successful search entity) pairs
    from one episode's steps_detail. Shared by online learning and by
    scripts/build_memory.py for offline bank mining."""
    cases = []
    pending = []
    for s in row.get("steps_detail") or []:
        action = str(s.get("action", ""))
        obs = str(s.get("obs", ""))
        if not action.lower().startswith("search["):  # raw output may be "Search["
            continue
        entity = action[action.find("[") + 1:-1]
        if obs.startswith("Could not find"):
            pending.append(entity)
        elif pending:
            for miss in pending:
                if miss.strip().lower() != entity.strip().lower():
                    cases.append({"miss": miss, "fix": entity})
            pending = []
    return cases


class _BankMemory:
    def _load(self, bank_path):
        self.entries = []
        with open(bank_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.entries.append(json.loads(line))


class CaseMemory(_BankMemory):
    """Remembers which query reformulation historically fixed a search miss,
    and suggests it when a similar miss occurs. mode="hint" appends the
    suggestion to the observation; mode="rewrite" returns the replacement
    query for the caller to re-issue."""

    name = "case"

    def __init__(self, bank_path, min_score=0.0, mode="hint"):
        self._load(bank_path)
        self.bm25 = BM25([e["miss"] for e in self.entries]) if self.entries else None
        self.min_score = min_score
        self.mode = mode
        self.stats = Counter()

    def inject(self, question):
        return ""

    def on_search_miss(self, entity):
        if not self.bm25:
            return None
        top = self.bm25.top(entity, k=1, min_score=self.min_score)
        if not top:
            self.stats["no_match"] += 1
            return None
        fix = self.entries[top[0][1]]["fix"]
        self.stats["hint"] += 1
        if self.mode == "rewrite":
            return fix
        return f'A past similar miss was fixed by searching "{fix}".'

    def on_episode_end(self, row):
        return mine_cases(row)


class CompositeMemory:
    """Fan hooks out to several memories: injects are concatenated, the
    first non-None search-miss hint wins."""

    name = "composite"

    def __init__(self, parts):
        self.parts = parts

    def inject(self, question):
        return "".join(p.inject(question) for p in self.parts)

    def on_search_miss(self, entity):
        for p in self.parts:
            hint = p.on_search_miss(entity)
            if hint:
                return hint
        return None

    def on_episode_end(self, row):
        cases = []
        for p in self.parts:
            cases += p.on_episode_end(row)
        return cases


class ReflectionMemory(_BankMemory):
    """Reflexion-style bank of one-line lessons mined from failed tuning
    episodes; injects the lessons of the most similar past questions."""

    name = "refl"

    def __init__(self, bank_path, k=2, min_score=0.0):
        self._load(bank_path)
        self.bm25 = (BM25([e["question"] for e in self.entries])
                     if self.entries else None)
        self.k = k
        self.min_score = min_score
        self.stats = Counter()

    def inject(self, question):
        if not self.bm25:
            return ""
        top = self.bm25.top(question, k=self.k, min_score=self.min_score)
        self.stats["injected"] += bool(top)
        if not top:
            return ""
        lines = [f"- {self.entries[i]['lesson']}" for _, i in top]
        return "Lessons from similar past questions:\n" + "\n".join(lines) + "\n"

    def on_search_miss(self, entity):
        return None

    def on_episode_end(self, row):
        return []
