from pathlib import Path
from indexer import Indexer

EP = Path("logs/index_episodes.json")
SH = Path("logs/index_shows.json")
IDX = Path("logs/index_unified.json")

if __name__ == "__main__":
    idx = Indexer.load(IDX)
    print(f"Loaded {idx.N} docs from {IDX}")
    print("Tip: filter by type, e.g.,  type:show fantasy ;  type:actor cranston ;  type:episode fly")
    while True:
        q = input("query> ").strip()
        if not q: break
        hits = idx.search(q, top_k=12)
        groups = {"show": [], "episode": [], "actor": []}
        for s, d in hits:
            groups[d.doc_type].append((s, d))
        for t in ("show", "episode", "actor"):
            if not groups[t]: 
                continue
            print(f"\n=== {t.upper()}S ===")
            for score, doc in groups[t][:6]:  # show top 6 per type
                url = doc.url or "(no url)"
                print(f"{score:0.3f}  {doc.title}  → {url}")
        print()
