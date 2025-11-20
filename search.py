from __future__ import annotations
from pathlib import Path
import sys, re, json, gzip, pickle, math, csv
from collections import Counter

SPARK_DIR = "spark"

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
FG_GREEN = "\033[92m"
FG_CYAN = "\033[96m"
FG_YELLOW = "\033[93m"
FG_MAGENTA = "\033[95m"
FG_RED = "\033[91m"
FG_WHITE = "\033[97m"

def color(text, c): return f"{c}{text}{RESET}"

SPLIT_RE = re.compile(r"[^a-zA-Z0-9]+")
STOPWORDS = {
    "a","an","and","are","as","at","be","by","for","from","in","into","is","it","no","not",
    "of","on","or","s","such","t","that","the","their","then","there","these","they","this",
    "to","was","will","with","you","your","yours","our","ours","he","she","we","i","me","my",
    "mine","his","her","its","them","those","these","what","which","who","whom","why","how",
}

def tokenize(text: str) -> list[str]:
    if not text: return []
    toks = [t.lower() for t in SPLIT_RE.split(text) if len(t) >= 2]
    return [t for t in toks if t not in STOPWORDS]

def load_vocab(vocab_path: Path) -> list[str]:
    return vocab_path.read_text(encoding="utf-8").splitlines()

def load_idf(idf_path: Path) -> list[float]:
    return json.loads(idf_path.read_text(encoding="utf-8"))["idf"]

def load_docs_table(docs_tsv: Path) -> dict[int, tuple[str,str]]:
    """Returns: doc_id -> (url, title)"""
    out = {}
    with docs_tsv.open("r", encoding="utf-8", newline="") as f:
        r = csv.reader(f, delimiter="\t")
        header = next(r, None)
        for row in r:
            if not row: continue
            # expected columns: doc_id, url, title
            try:
                did = int(row[0])
            except:
                continue
            url = row[1] if len(row) > 1 else ""
            title = row[2] if len(row) > 2 else ""
            out[did] = (url, title)
    return out

def load_tfidf_rows_with_meta(rows_path: Path):
    """
    Returns two dicts:
      vecs: doc_id -> (indices, values)
      meta: doc_id -> (url, title)
    """
    vecs, meta = {}, {}
    with gzip.open(rows_path, "rb") as f:
        while True:
            try:
                did, url, title, idx, val = pickle.load(f)
                did = int(did)
                vecs[did] = (idx, val)
                meta[did] = (url, title)
            except EOFError:
                break
    return vecs, meta

def build_index(artifacts_dir: Path):
    vocab = load_vocab(artifacts_dir / "vocab.txt")
    idf = load_idf(artifacts_dir / "idf.json")

    # Load vectors + embedded metadata
    vecs, meta = load_tfidf_rows_with_meta(artifacts_dir / "tfidf_rows.pkl.gz")

    # If docs.tsv exists, it can augment/override metadata (optional)
    docs = {}
    docs_tsv = artifacts_dir / "docs.tsv"
    if docs_tsv.exists():
        docs = load_docs_table(docs_tsv)

    # Merge: prefer embedded meta, fallback to docs.tsv
    merged = {}
    for did in vecs.keys():
        if did in docs:
            merged[did] = docs[did]
        elif did in meta:
            merged[did] = meta[did]
        else:
            merged[did] = ("", "")

    term2idx = {t:i for i,t in enumerate(vocab)}
    doc_norm = {
        did: math.sqrt(sum(v*v for v in vecs[did][1])) or 1e-12
        for did in vecs
    }
    return term2idx, idf, merged, vecs, doc_norm

def make_query_vector(q: str, term2idx: dict[str,int], idf: list[float]) -> dict[int, float]:
    toks = tokenize(q)
    if not toks: return {}
    tf = Counter(toks)   # raw term freq (same as CountVectorizer default)
    qvec = {}
    for term, cnt in tf.items():
        j = term2idx.get(term)
        if j is None:
            continue
        # tf * idf
        qvec[j] = float(cnt) * float(idf[j])
    return qvec

def qnorm(qvec: dict[int,float]) -> float:
    s = sum(v*v for v in qvec.values())
    return math.sqrt(s) if s>0 else 1e-12

def search(q: str, top_k: int,
           term2idx, idf, docs, vecs, doc_norm):
    qvec = make_query_vector(q, term2idx, idf)
    if not qvec:
        return []

    qn = qnorm(qvec)
    scores = []

    # Iterate all docs (13k scale is fine); compute dot only on shared indices
    # Dot(q,d) = sum_{j in intersection} q_j * d_j
    q_idxs = set(qvec.keys())
    for did, (idx_list, val_list) in vecs.items():
        # build dot by scanning doc indices and checking if in q
        dot = 0.0
        for j, v in zip(idx_list, val_list):
            if j in q_idxs:
                dot += qvec[j] * v
        denom = qn * doc_norm[did]
        if denom > 0:
            scores.append((dot / denom, did))

    # sort by score desc
    scores.sort(key=lambda x: x[0], reverse=True)
    return scores[:top_k]

def print_hits(hits, docs):
    if not hits:
        print(color("No results.", FG_RED))
        return
    print()
    for score, did in hits:
        url, title = docs.get(did, ("", ""))
        print(
            f"{color(f'{score:0.3f}', FG_GREEN)}  "
            f"{color('SHOW:', FG_CYAN)} "
            f"{color(BOLD + (title or '(no title)') + RESET, FG_CYAN)}  "
            f"{color('→', FG_WHITE)} {color(url or '(no url)', FG_WHITE)}"
        )

def main():
    sprark_dir = Path(SPARK_DIR)
    needed = ["vocab.txt", "idf.json", "docs.tsv", "tfidf_rows.pkl.gz"]
    missing = [p for p in needed if not (sprark_dir / p).exists()]
    if missing:
        print(color(f"Missing artifacts in {sprark_dir}: {', '.join(missing)}", FG_RED))
        sys.exit(1)

    print(color(f"Loading index from {sprark_dir} ...", FG_GREEN))
    term2idx, idf, docs, vecs, doc_norm = build_index(sprark_dir)
    print(color(f"Ready. Documents: {len(docs)} | Vocab: {len(term2idx)}", FG_GREEN))

    while True:
        try:
            q = input(color("query> ", FG_GREEN)).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            break

        hits = search(q, top_k=10, term2idx=term2idx, idf=idf,
                      docs=docs, vecs=vecs, doc_norm=doc_norm)
        print_hits(hits, docs)
        print()

if __name__ == "__main__":
    main()