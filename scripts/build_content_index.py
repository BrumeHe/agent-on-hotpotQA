"""Build a content-level FTS5 index over every page's intro paragraph.

Source: data/wiki_full.db pages(title, text). Target: data/wiki_content.db
with content_fts(title UNINDEXED, intro). The intro is the first paragraph
after the title line that passes WikiEnv's >2-word filter — i.e. the same
lead the baseline search[] observation is drawn from.

This powers ContentFallbackWikiEnv: when title-exact search misses, the agent
falls back to BM25-ranked content retrieval (DrQA-style) instead of guessing
more titles.

Builds into a .tmp file and atomically renames on completion.

Usage: .venv/Scripts/python scripts/build_content_index.py
"""

import os
import sqlite3
import sys
import time

from tqdm import tqdm

SRC = "data/wiki_full.db"
OUT = "data/wiki_content.db"
TMP = OUT + ".tmp"
CHUNK = 20000


def intro_of(text):
    parts = text.split("\n")
    for p in parts[1:]:  # skip the title line
        p = p.strip()
        if len(p.split(" ")) > 2:
            return p
    return ""


def main():
    t0 = time.time()
    if os.path.exists(TMP):
        os.remove(TMP)
    src = sqlite3.connect(SRC)
    n_total = src.execute("SELECT COUNT(*) FROM pages").fetchone()[0]
    dst = sqlite3.connect(TMP)
    dst.execute("PRAGMA journal_mode=OFF")
    dst.execute("PRAGMA synchronous=OFF")
    dst.execute("CREATE VIRTUAL TABLE content_fts USING fts5(title UNINDEXED, intro)")

    cur = src.execute("SELECT title, text FROM pages ORDER BY rowid")
    buf = []
    n_kept = 0
    with tqdm(total=n_total, desc="content-index") as bar:
        while True:
            rows = cur.fetchmany(CHUNK)
            if not rows:
                break
            for title, text in rows:
                intro = intro_of(text)
                if intro:
                    buf.append((title, intro))
            if buf:
                dst.executemany("INSERT INTO content_fts(title, intro) VALUES (?, ?)", buf)
                dst.commit()
                n_kept += len(buf)
                buf.clear()
            bar.update(len(rows))
    dst.execute("INSERT INTO content_fts(content_fts) VALUES ('optimize')")
    dst.commit()
    dst.close()
    src.close()
    os.replace(TMP, OUT)
    print(f"done: {n_kept}/{n_total} intros indexed -> {OUT} "
          f"({time.time() - t0:.0f}s, {os.path.getsize(OUT) / 2**30:.2f} GiB)")


if __name__ == "__main__":
    sys.exit(main())
