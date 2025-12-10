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

DATA_DIR = os.environ.get("DATA_DIR", "/extracted")
INDEX_DIR = os.environ.get("INDEX_DIR", "/index")

EPISODES_TSV = os.path.join(DATA_DIR, "extracted_episodes.tsv")
CREDITS_TSV = os.path.join(DATA_DIR, "extracted_credits.tsv")
JOINED_TSV = os.path.join(DATA_DIR, "joined_shows_wiki.tsv")


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


def build_index():
    print("Loading TSVs...")
    episodes_by_show = load_episodes_by_show(EPISODES_TSV)
    credits_by_show = load_credits_by_show(CREDITS_TSV)

    print("Initializing Lucene VM...")
    lucene.initVM(vmargs=["-Djava.awt.headless=true"])
    print("Lucene version:", lucene.VERSION)

    os.makedirs(INDEX_DIR, exist_ok=True)

    store = NIOFSDirectory(Paths.get(INDEX_DIR))
    analyzer = StandardAnalyzer()
    config = IndexWriterConfig(analyzer)
    config.setOpenMode(IndexWriterConfig.OpenMode.CREATE)

    writer = IndexWriter(store, config)

    print("Indexing shows from", JOINED_TSV)
    with open(JOINED_TSV, encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            show_name = row.get("title") or ""
            if not show_name:
                show_name = row.get("wiki_title") or ""

            doc = Document()

            # -------- IDs --------
            doc.add(StringField("show_name", show_name, Field.Store.YES))
            doc.add(StringField("url", row.get("url") or "", Field.Store.YES))

            # -------- Titles --------
            alt_title = row.get("wiki_alt_title") or ""
            native_title = row.get("wiki_native_title") or ""
            norm_title_val = row.get("wiki_norm_title")
            
            if alt_title:
                doc.add(TextField("alt_title", alt_title, Field.Store.YES))
            if native_title:
                doc.add(TextField("native_title", native_title, Field.Store.YES))
            doc.add(StringField("norm_title", norm_title_val, Field.Store.YES))

            # -------- Description --------
            desc = row.get("description") or ""

            if desc:
                doc.add(TextField("description", desc, Field.Store.YES))

            wiki_desc = row.get("wiki_description") or ""
            if wiki_desc and wiki_desc != desc:
                doc.add(TextField("wiki_description", wiki_desc, Field.Store.YES))

            # -------- Helper for numeric fields --------
            def add_int_field(name: str, val: str | None):
                if val and str(val).isdigit():
                    v = int(val)
                    doc.add(IntPoint(name, v))
                    doc.add(StoredField(name, v))

            add_int_field("runtime_minutes", row.get("runtime_minutes"))
            add_int_field("season_count", row.get("season_count"))
            add_int_field("episode_count", row.get("episode_count"))

            # -------- Start year --------
            year_start = row.get("wiki_page_start_year") or row.get("start_year")
            
            add_int_field("start_year", year_start)

            # -------- Categorical fields --------
            genres = row.get("genres") or ""
            if genres:
                doc.add(TextField("genres", genres, Field.Store.YES))
                for g in re.split(r"[;,]", genres):
                    g = g.strip()
                    if g:
                        doc.add(StringField("genres_exact", g.lower(), Field.Store.YES))

            wiki_cats = row.get("wiki_categories") or ""
            if wiki_cats:
                doc.add(TextField("wiki_categories", wiki_cats, Field.Store.YES))
                for c in re.split(r"[;,]", wiki_cats):
                    c = c.strip()
                    if c:
                        doc.add(StringField("category_exact", c.lower(), Field.Store.YES))

            if row.get("country"):
                doc.add(StringField("country_exact", row["country"], Field.Store.YES))
            if row.get("network"):
                doc.add(StringField("network_exact", row["network"], Field.Store.YES))

            if row.get("wiki_language"):
                doc.add(StringField("language_exact", row["wiki_language"], Field.Store.YES))

            # -------- People from wiki --------
            for fname in [
                "wiki_creator",
                "wiki_developer",
                "wiki_showrunner",
                "wiki_writer",
                "wiki_director",
                "wiki_composer",
            ]:
                val = row.get(fname) or ""
                if val:
                    doc.add(TextField(fname, val, Field.Store.YES))

            # -------- Episodes aggregation --------
            show_url = row.get("url") or ""

            ep_rows = episodes_by_show.get(show_url, [])
            ep_titles = [e.get("episode_title", "") for e in ep_rows if e.get("episode_title")]
            if ep_titles:
                doc.add(TextField("episode_titles", " \n".join(ep_titles), Field.Store.YES))

            # -------- Credits aggregation --------
            if not row.get("wiki_starring"):
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
                    cast_all_text = " ".join(cast_chunks) if cast_chunks else ""
            else:
                wiki_starring = row.get("wiki_starring") or ""
                doc.add(TextField("cast_all", wiki_starring, Field.Store.YES))
                cast_all_text = wiki_starring

            # -------- Keywords --------
            if row.get("keywords"):
                doc.add(TextField("keywords", row["keywords"], Field.Store.YES))

            # -------- Catch-all content field --------
            full_content_parts = [
                show_name,
                alt_title,
                native_title,
                desc,
                wiki_desc,
                genres,
                " ".join(ep_titles),
            ]

            if cast_all_text:
                full_content_parts.append(cast_all_text)

            for fname in [
                "keywords",
                "wiki_creator",
                "wiki_developer",
                "wiki_showrunner",
                "wiki_writer",
                "wiki_director",
                "wiki_categories",
            ]:
                val = row.get(fname, "")
                if val:
                    full_content_parts.append(val)

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
