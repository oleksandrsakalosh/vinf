#!/usr/bin/env python
import os
import csv
import re
from collections import defaultdict

import lucene # type: ignore
from java.nio.file import Paths # type: ignore

from org.apache.lucene.analysis.standard import StandardAnalyzer # type: ignore
from org.apache.lucene.store import NIOFSDirectory # type: ignore
from org.apache.lucene.index import IndexWriter, IndexWriterConfig # type: ignore
from org.apache.lucene.document import ( # type: ignore
    Document,
    Field,
    StringField,
    TextField,
    StoredField,
    IntPoint,
)

DATA_DIR = os.environ.get("DATA_DIR", "/data")
INDEX_DIR = os.environ.get("INDEX_DIR", "/index")

SHOWS_TSV = os.path.join(DATA_DIR, "extracted_shows.tsv")
EPISODES_TSV = os.path.join(DATA_DIR, "extracted_episodes.tsv")
CREDITS_TSV = os.path.join(DATA_DIR, "extracted_credits.tsv")
WIKI_TSV = os.path.join(DATA_DIR, "wiki_shows_test.tsv")


def norm_title(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


def load_episodes_by_show(path: str):
    episodes_by_show = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            episodes_by_show[row["show_url"]].append(row)
    return episodes_by_show


def load_credits_by_show(path: str):
    credits_by_show = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            credits_by_show[row["show_url"]].append(row)
    return credits_by_show


def load_wiki_shows(path: str):
    wiki_by_key = {}
    with open(path, encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            nt = row.get("norm_title") or norm_title(row.get("title", ""))
            sy = row.get("start_year")
            key = (nt, int(sy)) if sy and sy.isdigit() else (nt, None)
            # keep first for now
            wiki_by_key.setdefault(key, row)
    return wiki_by_key


def find_wiki_for_show(show_row, wiki_by_key):
    nt = norm_title(show_row.get("title", ""))
    year = None
    iso = show_row.get("date_start_iso")
    if iso and len(iso) >= 4 and iso[:4].isdigit():
        year = int(iso[:4])

    if year is not None:
        w = wiki_by_key.get((nt, year))
        if w:
            return w
    return wiki_by_key.get((nt, None))


def build_index():
    print("Loading TSVs...")
    episodes_by_show = load_episodes_by_show(EPISODES_TSV)
    credits_by_show = load_credits_by_show(CREDITS_TSV)
    wiki_by_key = load_wiki_shows(WIKI_TSV)

    print("Initializing Lucene VM...")
    lucene.initVM(vmargs=["-Djava.awt.headless=true"])
    print("Lucene version:", lucene.VERSION)

    os.makedirs(INDEX_DIR, exist_ok=True)

    store = NIOFSDirectory(Paths.get(INDEX_DIR))
    analyzer = StandardAnalyzer()
    config = IndexWriterConfig(analyzer)
    config.setOpenMode(IndexWriterConfig.OpenMode.CREATE)

    writer = IndexWriter(store, config)

    print("Indexing shows from", SHOWS_TSV)
    with open(SHOWS_TSV, encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            show_url = row["url"]
            wiki = find_wiki_for_show(row, wiki_by_key)

            doc = Document()

            # -------- IDs --------
            doc.add(StringField("show_id", show_url, Field.Store.YES))
            if wiki:
                doc.add(StringField("page_id", wiki.get("page_id", ""), Field.Store.YES))

            # -------- Titles --------
            title = row.get("title") or ""
            doc.add(TextField("title", title, Field.Store.YES))
            doc.add(StringField("title_exact", title.lower(), Field.Store.YES))

            if wiki:
                alt_title = wiki.get("alt_title") or ""
                native_title = wiki.get("native_title") or ""
                if alt_title:
                    doc.add(TextField("alt_title", alt_title, Field.Store.YES))
                if native_title:
                    doc.add(TextField("native_title", native_title, Field.Store.YES))

            # -------- Description (prefer extracted, fallback to wiki) --------
            desc = row.get("description") or (wiki.get("description") if wiki else "") or ""
            doc.add(TextField("description", desc, Field.Store.YES))

            # -------- Helper for numeric fields --------
            def add_int_field(name: str, val: str | None):
                if val and str(val).isdigit():
                    v = int(val)
                    doc.add(IntPoint(name, v))
                    doc.add(StoredField(name, v))

            add_int_field("runtime_minutes", row.get("runtime_minutes"))
            add_int_field("season_count", row.get("season_count"))
            add_int_field("episode_count", row.get("episode_count"))

            # year_start from date_start_iso or wiki.page_start_year
            year_start = None
            iso = row.get("date_start_iso")
            if iso and len(iso) >= 4 and iso[:4].isdigit():
                year_start = int(iso[:4])
            elif wiki and wiki.get("page_start_year") and wiki["page_start_year"].isdigit():
                year_start = int(wiki["page_start_year"])
            if year_start is not None:
                doc.add(IntPoint("year_start", year_start))
                doc.add(StoredField("year_start", year_start))

            # -------- Categorical fields --------
            genres = row.get("genres") or ""
            if genres:
                doc.add(TextField("genres", genres, Field.Store.YES))
                for g in re.split(r"[;,]", genres):
                    g = g.strip()
                    if g:
                        doc.add(StringField("genres_exact", g.lower(), Field.Store.YES))

            if row.get("country"):
                doc.add(StringField("country_exact", row["country"], Field.Store.YES))
            if row.get("network"):
                doc.add(StringField("network_exact", row["network"], Field.Store.YES))

            if wiki and wiki.get("language"):
                doc.add(StringField("language_exact", wiki["language"], Field.Store.YES))

            # -------- People from wiki --------
            if wiki:
                for fname in [
                    "creator",
                    "developer",
                    "showrunner",
                    "writer",
                    "director",
                    "starring",
                    "composer",
                ]:
                    val = wiki.get(fname) or ""
                    if val:
                        doc.add(TextField(fname, val, Field.Store.YES))

            # -------- Episodes aggregation --------
            ep_rows = episodes_by_show.get(show_url, [])
            ep_titles = [e.get("episode_title", "") for e in ep_rows if e.get("episode_title")]
            if ep_titles:
                doc.add(TextField("episode_titles", " \n".join(ep_titles), Field.Store.YES))

            # -------- Credits aggregation --------
            cr_rows = credits_by_show.get(show_url, [])
            cast_chunks = []
            for c in cr_rows:
                name = (c.get("actor_name") or "").strip()
                role = (c.get("role") or "").strip()
                if not name:
                    continue
                cast_chunks.append(f"{name} {role}" if role else name)

            if cast_chunks:
                doc.add(TextField("cast_all", " ".join(cast_chunks), Field.Store.YES))

            # -------- Keywords --------
            if row.get("keywords"):
                doc.add(TextField("keywords", row["keywords"], Field.Store.YES))

            # -------- Catch-all content field --------
            full_content_parts = [
                title,
                desc,
                genres,
                row.get("keywords", ""),
                " ".join(ep_titles),
                " ".join(cast_chunks),
            ]
            if wiki:
                for fname in [
                    "creator",
                    "developer",
                    "showrunner",
                    "writer",
                    "director",
                    "starring",
                    "categories",
                ]:
                    full_content_parts.append(wiki.get(fname, ""))

            full_content = " ".join(p for p in full_content_parts if p)
            doc.add(TextField("content_all", full_content, Field.Store.NO))

            writer.addDocument(doc)

    print("Committing index…")
    writer.commit()
    writer.close()
    print("Index built in", INDEX_DIR)


if __name__ == "__main__":
    print("DATA_DIR =", DATA_DIR)
    print("INDEX_DIR =", INDEX_DIR)
    build_index()
