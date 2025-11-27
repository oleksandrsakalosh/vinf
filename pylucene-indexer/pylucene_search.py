#!/usr/bin/env python

import os
import sys
import lucene # type: ignore

from java.nio.file import Paths # type: ignore

from org.apache.lucene.analysis.standard import StandardAnalyzer # type: ignore
from org.apache.lucene.index import DirectoryReader # type: ignore
from org.apache.lucene.queryparser.classic import QueryParser # pyright: ignore
from org.apache.lucene.store import NIOFSDirectory # type: ignore
from org.apache.lucene.search import IndexSearcher # type: ignore


INDEX_DIR = os.environ.get("INDEX_DIR", "/index")
SEARCH_FIELD = "content_all"


def run(searcher: IndexSearcher, analyzer: StandardAnalyzer):
    """
    Interactive search loop: prompts for a query, searches, prints top hits.
    """
    stored_fields = searcher.storedFields()

    while True:
        print()
        print("Hit Enter on an empty line to quit.")
        command = input("Query: ").strip()
        if not command:
            return

        print(f"\nSearching for: {command!r}")

        parser = QueryParser(SEARCH_FIELD, analyzer)
        query = parser.parse(command)

        top_n = 5
        hits = searcher.search(query, top_n)
        score_docs = hits.scoreDocs

        print(f"{len(score_docs)} total matching documents (showing up to {top_n})\n")

        for i, score_doc in enumerate(score_docs, start=1):
            doc = stored_fields.document(score_doc.doc)

            title = doc.get("title") or "<no title>"
            start_year = doc.get("start_year") or "n/a"
            country = doc.get("country") or "n/a"
            genres = doc.get("genres") or ""
            network = doc.get("network") or ""
            url = doc.get("url") or ""

            print(f"{i:2d}. {title} ({start_year})  [score={score_doc.score:.3f}]")
            if country or network:
                print(f"    Country: {country}   Network: {network}")
            if genres:
                print(f"    Genres: {genres}")
            if url:
                print(f"    URL: {url}")
            print("-" * 60)


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
        # clean up
        reader.close()
        directory.close()


if __name__ == "__main__":
    main()
