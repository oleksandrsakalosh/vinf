#!/usr/bin/env python
import os
import sys
import re
import lucene # type: ignore

from java.nio.file import Paths # type: ignore
from java.util import HashMap # type: ignore
from java.util import HashMap  # type: ignore

from org.apache.lucene.search import MatchAllDocsQuery  # type: ignore
from org.apache.lucene.analysis.standard import StandardAnalyzer # type: ignore
from org.apache.lucene.index import DirectoryReader # type: ignore
from org.apache.lucene.queryparser.classic import MultiFieldQueryParser # pyright: ignore
from org.apache.lucene.store import NIOFSDirectory # type: ignore
from org.apache.lucene.search import IndexSearcher # type: ignore
from org.apache.lucene.search import BooleanQuery, BooleanClause # type: ignore
from org.apache.lucene.search import TermQuery # type: ignore
from org.apache.lucene.index import Term # type: ignore
from org.apache.lucene.document import IntPoint # type: ignore

class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    GREY = "\033[90m"


INDEX_DIR = os.environ.get("INDEX_DIR", "/index")

MEDIUM_DISAMBIG_RE = re.compile(
    r"\s*\((american|british|canadian|australian|indian|south korean|japanese)\s+"
    r"(tv|television)\s+series\)\s*$",
    re.IGNORECASE,
)

YEAR_PARENS_RE = re.compile(r"\s*\(\d{4}.*?\)\s*$")
PUNCT_SEP_RE = re.compile(r"[-–—_:;/\.,!?]+")
MULTISPACE_RE = re.compile(r"\s+")

def norm_query(text: str) -> str:
    if not text:
        return ""

    s = text.strip().lower()
    s = MEDIUM_DISAMBIG_RE.sub("", s)
    s = YEAR_PARENS_RE.sub("", s)
    s = PUNCT_SEP_RE.sub(" ", s)
    s = MULTISPACE_RE.sub(" ", s)

    return s.strip()


FIELDS = [
    "show_name",
    "alt_title",
    "native_title",
    "description",
    "wiki_description",
    "cast_all",
    "wiki_categories",
    "keywords",
    "content_all", 
]

BOOSTS = {
    "show_name": 5.0,
    "alt_title": 4.0,
    "native_title": 4.0,
    "description": 3.0,
    "wiki_description": 2.0,
    "cast_all": 1.0,
    "wiki_categories": 0.5,
    "keywords": 2.0,
    "content_all": 0.5,
}

def build_query(command: str, analyzer: StandardAnalyzer):
    tokens = command.split()
    filters = []
    free_terms = []

    for t in tokens:
        if t.startswith("country:"):
            val = t.split(":", 1)[1].lower()
            filters.append(TermQuery(Term("country_exact", val)))

        elif t.startswith("genre:"):
            val = t.split(":", 1)[1].lower()
            filters.append(TermQuery(Term("genres_exact", val)))

        elif t.startswith("year>="):
            try:
                val = int(t.split(">=", 1)[1])
                filters.append(IntPoint.newRangeQuery("start_year", val, 9999))
            except ValueError:
                pass 

        elif t.startswith("year<="):
            try:
                val = int(t.split("<=", 1)[1])
                filters.append(IntPoint.newRangeQuery("start_year", 0, val))
            except ValueError:
                pass 

        else:
            free_terms.append(t)

    if free_terms:
        free_text_raw = " ".join(free_terms)
        free_text_norm = norm_query(free_text_raw)

        if free_text_norm:
            flags = [BooleanClause.Occur.SHOULD] * len(FIELDS)
            main_query = MultiFieldQueryParser.parse(
                free_text_norm,
                FIELDS,
                flags,
                analyzer,
            )
        else:
            main_query = MatchAllDocsQuery()
    else:
        main_query = MatchAllDocsQuery()

    if not filters:
        return main_query

    bq = BooleanQuery.Builder()
    bq.add(main_query, BooleanClause.Occur.MUST)
    for f in filters:
        bq.add(f, BooleanClause.Occur.FILTER)
    return bq.build()


def run(searcher: IndexSearcher, analyzer: StandardAnalyzer):
    stored_fields = searcher.storedFields()

    while True:
        print()
        print(f"{C.GREY}Hit Enter on an empty line to quit.{C.RESET}")
        try:
            command = input(f"{C.CYAN}Query{C.RESET}{C.GREY}:{C.RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if not command:
            return

        print(f"\n{C.GREY}Searching for:{C.RESET} {C.YELLOW!s}{command!r}{C.RESET}\n")

        query = build_query(command, analyzer)

        top_n = 5
        hits = searcher.search(query, top_n)
        score_docs = hits.scoreDocs

        print(
            f"{C.GREEN}{len(score_docs)}{C.RESET} total matching documents "
            f"(showing up to {C.GREEN}{top_n}{C.RESET})\n"
        )

        if not score_docs:
            print(f"{C.RED}No results.{C.RESET}")
            continue

        for i, score_doc in enumerate(score_docs, start=1):
            doc = stored_fields.document(score_doc.doc)

            title = doc.get("show_name") or "<no title>"
            start_year = doc.get("start_year") or "n/a"
            country = doc.get("country_exact") or "n/a"
            genres = doc.get("genres") or ""
            network = doc.get("network_exact") or "?"
            runtime = doc.get("runtime_minutes") or "?"
            episodes = doc.get("episode_count") or "?"
            url = doc.get("url") or ""
            description = doc.get("description") or ""
            wiki_description = doc.get("wiki_description") or ""

            # Title line
            print(
                f"{C.MAGENTA}{i:2d}.{C.RESET} "
                f"{C.BOLD}{title}{C.RESET}  "
                f"{C.GREY}[score={score_doc.score:.3f}]{C.RESET}"
            )

            # Metadata line
            meta_line = (
                f"    {C.GREY}Year:{C.RESET} {C.YELLOW}{start_year}{C.RESET}  "
                f"{C.GREY}Country:{C.RESET} {country}  "
                f"{C.GREY}Network:{C.RESET} {network}  "
                f"{C.GREY}Runtime:{C.RESET} {runtime} min  "
                f"{C.GREY}Episodes:{C.RESET} {episodes}"
            )
            print(meta_line)

            if genres:
                print(f"    {C.GREY}Genres:{C.RESET} {C.CYAN}{genres}{C.RESET}")

            if url:
                print(f"    {C.GREY}URL:{C.RESET} {C.BLUE}{url}{C.RESET}")

            if description:
                short = description[:200]
                if len(description) > 200:
                    short += "..."
                print(
                    f"    {C.GREY}Description:{C.RESET} "
                    f"{C.DIM}{short}{C.RESET}"
                )

            if wiki_description:
                short = wiki_description[:200]
                if len(wiki_description) > 200:
                    short += "..."
                print(
                    f"    {C.GREY}Wiki:{C.RESET} "
                    f"{C.DIM}{short}{C.RESET}"
                )

            print(f"{C.GREY}" + "-" * 60 + f"{C.RESET}")



def main():
    lucene.initVM(vmargs=['-Djava.awt.headless=true'])
    print("Lucene version:", lucene.VERSION)

    index_dir = INDEX_DIR
    if len(sys.argv) >= 2:
        index_dir = sys.argv[1]

    if not os.path.exists(index_dir):
        print(f"Index directory does not exist: {index_dir}")
        sys.exit(1)

    index_path = Paths.get(index_dir)

    directory = NIOFSDirectory(index_path)
    reader = DirectoryReader.open(directory)
    searcher = IndexSearcher(reader)
    analyzer = StandardAnalyzer()

    try:
        run(searcher, analyzer)
    finally:
        reader.close()
        directory.close()


if __name__ == "__main__":
    main()
