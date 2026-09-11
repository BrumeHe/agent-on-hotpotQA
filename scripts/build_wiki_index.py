"""Build the sqlite wiki index from an extracted HotpotQA Wikipedia dump.

The dump extracts to a directory tree of .bz2 files; each line is one JSON
page: {"id", "url", "title", "text": [[para1_sent1, ...], ...], ...}.
The abstracts dump additionally carries text_with_links / charoffset_with_links
which we ignore. Hyperlink markup in `text` is stripped (anchor text kept).

Output schema:
  pages(title TEXT, title_norm TEXT UNIQUE, text TEXT)   -- paragraphs joined by \n
  titles_fts: FTS5 over title_norm (external content -> pages.rowid)

Crash safety: the db is built at OUT.tmp with journaling enabled and each
completed .bz2 file recorded in done_files; a killed build resumes with
--resume. On success the tmp file is atomically renamed to OUT, so --out
never holds a partial database.

Usage:
  .venv/Scripts/python scripts/build_wiki_index.py --src data/wiki_abstracts --out data/wiki_abstracts.db
  .venv/Scripts/python scripts/build_wiki_index.py --src data/wiki_full --out data/wiki_full.db --resume
"""

import argparse
import bz2
import json
import os
import sqlite3
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from wiki_env import normalize_title, strip_links  # noqa: E402


def list_files(src):
    files = []
    for root, _, names in os.walk(src):
        for name in sorted(names):
            if name.endswith(".bz2"):
                files.append(os.path.join(root, name))
    files.sort()
    return files


def iter_file_pages(path):
    try:
        with bz2.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
    except (OSError, EOFError) as e:
        print(f"\n[warn] skip corrupt file {path}: {e}", file=sys.stderr)


def page_to_text(page):
    paras = []
    for para in page.get("text") or []:
        s = strip_links("".join(para)).strip()
        if s:
            paras.append(s)
    return "\n".join(paras)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="debug: stop after N pages")
    ap.add_argument("--resume", action="store_true",
                    help="resume a killed build from OUT.tmp via done_files")
    args = ap.parse_args()

    out_tmp = args.out + ".tmp"
    resume = args.resume and os.path.exists(out_tmp)
    if not resume:
        if os.path.exists(out_tmp):
            os.remove(out_tmp)
        if os.path.exists(args.out):
            os.remove(args.out)
    db = sqlite3.connect(out_tmp)
    # journal survives process kills; synchronous=OFF only risks OS crashes
    db.execute("PRAGMA journal_mode = DELETE")
    db.execute("PRAGMA synchronous = OFF")
    db.execute("CREATE TABLE IF NOT EXISTS pages ("
               "title TEXT, title_norm TEXT UNIQUE, text TEXT)")
    db.execute("CREATE TABLE IF NOT EXISTS done_files (path TEXT PRIMARY KEY)")
    db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS titles_fts USING fts5("
               "title_norm, content='pages', content_rowid='rowid')")

    files = list_files(args.src)
    done = {r[0] for r in db.execute("SELECT path FROM done_files")}
    n = db.execute("SELECT COUNT(*) FROM pages").fetchone()[0]
    print(f"{len(files)} bz2 files under {args.src}; "
          f"{len(done)} already done, {n} pages indexed", flush=True)

    t0 = time.time()
    n0 = n
    batch = []
    stopped = False
    for fi, path in enumerate(files):
        if path in done:
            continue
        for page in iter_file_pages(path):
            title = page.get("title") or ""
            text = page_to_text(page)
            if not title or not text:
                continue
            batch.append((title, normalize_title(title), text))
            if len(batch) >= 5000:
                db.executemany(
                    "INSERT OR IGNORE INTO pages VALUES (?, ?, ?)", batch)
                n += len(batch)
                batch = []
        if batch:
            db.executemany(
                "INSERT OR IGNORE INTO pages VALUES (?, ?, ?)", batch)
            n += len(batch)
            batch = []
        db.execute("INSERT OR REPLACE INTO done_files VALUES (?)", (path,))
        db.commit()
        elapsed = max(time.time() - t0, 1e-9)
        print(f"\r{n} pages, {(n - n0) / elapsed:.0f} pages/s   ",
              end="", flush=True)
        if args.limit and n >= args.limit:
            stopped = True
            break

    if stopped:
        db.close()
        print(f"\ndebug stop at {n} pages; tmp left at {out_tmp}")
        return
    print(f"\ninserted {n} pages in {time.time() - t0:.0f}s, rebuilding FTS...")
    t1 = time.time()
    db.execute("INSERT INTO titles_fts(titles_fts) VALUES('rebuild')")
    db.commit()
    db.execute("PRAGMA optimize")
    total = db.execute("SELECT COUNT(*) FROM pages").fetchone()[0]
    db.close()
    os.replace(out_tmp, args.out)
    print(f"FTS rebuilt in {time.time() - t1:.0f}s; total rows {total}; -> {args.out}")


if __name__ == "__main__":
    main()
