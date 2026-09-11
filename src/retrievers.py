"""Retrieval-side upgrades for the wiki env (the R in RAG).

Recall axis:  baseline WikiEnv already does title exact match + FTS5 similar
              titles; that stays the entry protocol.
Read axis:    baseline search[] returns the page's first 5 sentences.
              QuestionAwareEnv instead returns the 5 sentences most relevant
              to the question (BM25 within the page, kept in document order),
              falling back to first-5 when nothing overlaps.
Query axis:   reformulation middleware lives in memory.py (CaseMemory), not
              here.

Dense title embeddings are a documented stretch goal (needs an embedding
model via hf-mirror + onnxruntime); deliberately not implemented yet.
"""

from memory import BM25
from wiki_env import WikiEnv, normalize_title

import sqlite3


def page_sentences(page):
    # same sentence splitting as WikiEnv.construct_lookup_list
    paragraphs = [p.strip() for p in page.split("\n") if p.strip()]
    sentences = []
    for p in paragraphs:
        sentences += p.split(". ")
    return [s.strip() + "." for s in sentences if s.strip()]


class FallbackWikiEnv(WikiEnv):
    """Recall-axis upgrade: on a search miss, deterministically retry with
    progressively shorter queries (drop trailing tokens) inside the same
    step, annotating the observation when a retry hits. Auto-retries do not
    count as agent searches."""

    MAX_RETRIES = 3

    def search_step(self, entity):
        super().search_step(entity)
        if not self.obs.startswith("Could not find"):
            return
        tokens = entity.split()
        for k in range(1, min(self.MAX_RETRIES + 1, len(tokens))):
            shorter = " ".join(tokens[:-k])
            if not self._fetch_page(shorter):
                continue
            super().search_step(shorter)
            self.num_searches -= 1  # auto-retry is not an agent action
            self.obs = f"[auto-retry: {shorter}] " + self.obs
            return


class FallbackV2WikiEnv(WikiEnv):
    """Recall-axis upgrade v2: on a search miss, follow the env's own FTS
    'Similar' suggestions instead of blindly shortening the query — v1's
    tail-chop degraded entities to bare first names ("Danny Green" -> "Danny")
    and hit name-index pages. Candidates must contain ALL query tokens
    (the OR-fallback half of _similar_titles is unranked garbage: it mapped
    "South Dade Senior High School" -> "Austrian School"), and
    disambiguation-looking pages are skipped.
    Auto-retries do not count as agent searches."""

    MAX_CANDIDATES = 8

    @staticmethod
    def _tokens(text):
        return {"".join(c for c in t if c.isalnum())
                for t in normalize_title(text).split()} - {""}

    def search_step(self, entity):
        super().search_step(entity)
        if not self.obs.startswith("Could not find"):
            return
        q_tokens = self._tokens(entity)
        for cand in self._similar_titles(entity, limit=self.MAX_CANDIDATES):
            row = self._fetch_page(cand)
            if row is None:
                continue
            title, text = row
            if not q_tokens <= self._tokens(title):
                continue
            if "may refer to:" in text[:300].lower() \
                    or "disambiguation" in title.lower():
                continue
            super().search_step(cand)
            self.num_searches -= 1  # auto-retry is not an agent action
            self.obs = f"[auto-retry: {cand}] " + self.obs
            return


class ContentFallbackWikiEnv(WikiEnv):
    """Recall-axis upgrade v3: on a title miss, fall back to CONTENT search —
    FTS5 + bm25() ranking over the intro paragraph of every page
    (data/wiki_content.db, built by scripts/build_content_index.py).

    The query is AND-matched on all alnum tokens; if nothing matches, drop
    trailing qualifier tokens (models append qualifiers: "Michael Braz
    composer Georgia") and retry, up to MAX_DROPS times. This is DrQA-style
    passage retrieval grafted onto the ReAct action space: it does not depend
    on the model guessing the exact page title. The hit page is fetched from
    the main DB and returned as a normal first-5 observation with a
    [content-match: Title] marker. Auto-retries do not count as searches."""

    MAX_DROPS = 2

    def __init__(self, db_path, content_db="data/wiki_content.db"):
        super().__init__(db_path)
        self.content = sqlite3.connect(content_db, check_same_thread=False)

    @staticmethod
    def _qtokens(text):
        return [t for t in ("".join(c for c in tok if c.isalnum())
                            for tok in normalize_title(text).split()) if t]

    def _content_search(self, entity):
        tokens = self._qtokens(entity)
        for drop in range(0, min(self.MAX_DROPS + 1, len(tokens))):
            keep = tokens[:len(tokens) - drop]
            if not keep:
                break
            query = " AND ".join(f'"{t}"' for t in keep)
            try:
                rows = self.content.execute(
                    "SELECT title FROM content_fts WHERE intro MATCH ? "
                    "ORDER BY bm25(content_fts) LIMIT 3", (query,)).fetchall()
            except sqlite3.Error:
                continue
            for (title,) in rows:
                row = self._fetch_page(title)
                if row is None:
                    continue
                if "may refer to:" in row[1][:300].lower() \
                        or "disambiguation" in row[0].lower():
                    continue
                return row[0]
        return None

    def search_step(self, entity):
        super().search_step(entity)
        if not self.obs.startswith("Could not find"):
            return
        hit = self._content_search(entity)
        if hit is None:
            return
        super().search_step(hit)
        self.num_searches -= 1  # auto-retry is not an agent action
        self.obs = f"[content-match: {hit}] " + self.obs


class QuestionAwareEnv(WikiEnv):
    """search[] observation = question-relevant sentences instead of first-5.
    Everything else (action space, lookup, miss format) is unchanged."""

    def __init__(self, db_path, question, k=5):
        super().__init__(db_path)
        self.question = question
        self.k = k

    def search_step(self, entity):
        super().search_step(entity)
        if self.obs.startswith("Could not find"):
            return
        sents = page_sentences(self.page)
        if len(sents) <= self.k:
            return
        ranked = BM25(sents).top(self.question, k=self.k)
        if ranked:
            chosen = sorted(i for _, i in ranked)
            self.obs = " ".join(sents[i] for i in chosen)
