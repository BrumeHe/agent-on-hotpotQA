"""Offline Wikipedia environment, porting ReAct's official wikienv.py
(https://github.com/ysymyth/ReAct) onto the local HotpotQA 2017-10-01 dump.

Action space and observation formats are identical to the original:
  search[entity] -> first 5 sentences of the page, or
                    "Could not find {entity}. Similar: [...]."
  lookup[keyword] -> next sentence containing keyword in the current page
  finish[answer]  -> end episode
  think[...]      -> "Nice thought."

Known deviation: the original scraped en.wikipedia.org live and re-searched
"[entity]" upon hitting a disambiguation page. Here a disambiguation page is
returned like a normal page; its first sentences enumerate the options
("... may refer to: ..."), so the agent can pick one explicitly.
"""

import re
import sqlite3

LINK_RE = re.compile(r'<a href="[^"]*">(.*?)</a>', re.S)
TAG_RE = re.compile(r"<[^>]+>")


def strip_links(text):
    """Remove wiki markup, keeping the visible anchor text."""
    text = LINK_RE.sub(lambda m: m.group(1), text)
    return TAG_RE.sub("", text)


def normalize_title(title):
    return " ".join(title.lower().split())


class WikiEnv:
    def __init__(self, db_path):
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.page = None
        self.obs = None
        self.lookup_keyword = None
        self.lookup_list = None
        self.lookup_cnt = None
        self.steps = 0
        self.answer = None
        self.num_searches = 0
        self.current_title = None

    def reset(self):
        self.obs = ("Interact with Wikipedia using search[], lookup[], and "
                    "finish[].\n")
        self.page = None
        self.lookup_keyword = self.lookup_list = self.lookup_cnt = None
        self.steps = 0
        self.answer = None
        self.current_title = None
        return self.obs

    @staticmethod
    def get_page_obs(page):
        # identical to official wikienv.get_page_obs
        paragraphs = page.split("\n")
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        sentences = []
        for p in paragraphs:
            sentences += p.split('. ')
        sentences = [s.strip() + '.' for s in sentences if s.strip()]
        return ' '.join(sentences[:5])

    def construct_lookup_list(self, keyword):
        # identical to official wikienv.construct_lookup_list
        if self.page is None:
            return []
        paragraphs = self.page.split("\n")
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        sentences = []
        for p in paragraphs:
            sentences += p.split('. ')
        sentences = [s.strip() + '.' for s in sentences if s.strip()]
        parts = [p for p in sentences if keyword.lower() in p.lower()]
        return parts

    def _fetch_page(self, entity):
        cur = self.db.execute(
            "SELECT title, text FROM pages WHERE title_norm = ?",
            (normalize_title(entity),))
        return cur.fetchone()

    def _similar_titles(self, entity, limit=5):
        norm = normalize_title(entity)
        tokens = [t.replace('"', '') for t in norm.split() if t.replace('"', '')]
        seen, out = set(), []

        def add(rows):
            for (title,) in rows:
                if title not in seen:
                    seen.add(title)
                    out.append(title)

        if tokens:
            for query in (" AND ".join(f'"{t}"' for t in tokens),
                          " OR ".join(f'"{t}"' for t in tokens)):
                try:
                    add(self.db.execute(
                        "SELECT p.title FROM titles_fts JOIN pages p "
                        "ON p.rowid = titles_fts.rowid "
                        "WHERE titles_fts MATCH ? LIMIT ?",
                        (query, limit)).fetchall())
                except sqlite3.Error:
                    pass
                if len(out) >= limit:
                    break
        if len(out) < limit:
            add(self.db.execute(
                "SELECT title FROM pages WHERE title_norm LIKE ? LIMIT ?",
                (f"%{norm}%", limit)).fetchall())
        return out[:limit]

    def search_step(self, entity):
        self.num_searches += 1
        row = self._fetch_page(entity)
        if row is None:
            self.obs = f"Could not find {entity}. Similar: {self._similar_titles(entity)}."
            return
        title, text = row
        page = ""
        for p in text.split("\n"):
            p = p.strip()
            if len(p.split(" ")) > 2:
                page += p
                if not p.endswith("\n"):
                    page += "\n"
        self.page = page
        self.current_title = title
        self.obs = self.get_page_obs(self.page)
        self.lookup_keyword = self.lookup_list = self.lookup_cnt = None

    def step(self, action):
        done = False
        action = action.strip()
        if self.answer is not None:  # already finished
            return self.obs, True

        if action.startswith("search[") and action.endswith("]"):
            self.search_step(action[len("search["):-1])
        elif action.startswith("lookup[") and action.endswith("]"):
            keyword = action[len("lookup["):-1]
            if self.lookup_keyword != keyword:  # reset lookup
                self.lookup_keyword = keyword
                self.lookup_list = self.construct_lookup_list(keyword)
                self.lookup_cnt = 0
            if self.lookup_cnt >= len(self.lookup_list):
                self.obs = "No more results.\n"
            else:
                self.obs = (f"(Result {self.lookup_cnt + 1} / {len(self.lookup_list)}) "
                            + self.lookup_list[self.lookup_cnt])
                self.lookup_cnt += 1
        elif action.startswith("finish[") and action.endswith("]"):
            self.answer = action[len("finish["):-1]
            done = True
            self.obs = "Episode finished.\n"
        elif action.startswith("think[") and action.endswith("]"):
            self.obs = "Nice thought."
        else:
            self.obs = f"Invalid action: {action}"

        self.steps += 1
        return self.obs, done
