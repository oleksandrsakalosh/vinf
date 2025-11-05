from __future__ import annotations
import csv, json, math, re
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Tuple, Iterable

EXTRACTED_DIR = Path("./extracted")

SHOWS = EXTRACTED_DIR/"extracted_shows.tsv"
EPISODES = EXTRACTED_DIR/"extracted_episodes.tsv"
CREDITS = EXTRACTED_DIR/"extracted_credits.tsv"
ACTORS = EXTRACTED_DIR/"extracted_actors.tsv"

INDEXER_DIR = Path("./indexer")


TOKEN_RE = re.compile(r"[a-z0-9]+")

def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())

@dataclass
class Doc:
    doc_id: int
    url: str
    title: str
    keywords: Optional[str] = None
    date_start: Optional[str] = None
    date_start_iso: Optional[str] = None
    date_end: Optional[str] = None
    date_end_iso: Optional[str] = None
    status: Optional[str] = None
    network: Optional[str] = None
    country: Optional[str] = None
    runtime_minutes: Optional[int] = None
    episode_count: Optional[int] = None
    genres: Optional[str] = None
    cast: Optional[str] = None
    characters: Optional[str] = None
    description: Optional[str] = None
    season_count: Optional[int] = None
    episodes: Optional[str] = None


class Indexer:
    def __init__(self, 
                 tf_sublinear=True, 
                 idf_mode: str = "smooth",   # "classic" | "smooth" | "prob" | "invdf"
                 ):
        self.tf_sublinear = tf_sublinear
        self.idf_mode = idf_mode
        self.normalize = True

        self.docs: list[Doc] = []
        self.N = 0
        
        self.postings: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        self.df: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.doc_norm: Dict[int, float] = {}

    def _tfw(self, tf: int) -> float:
        return (1.0 + math.log(tf)) if (self.tf_sublinear and tf > 0) else float(tf)

    def build(self, iterable: Iterable[Tuple[str, str, str, Dict[str, str]]], field_weights: Dict[str, Dict[str, float]]):
        per_doc_counts: list[Counter] = []

        for doc_type, title, url, fields in iterable:
            counts = Counter()
            weights = field_weights.get(doc_type, {})
            for fname, text in (fields or {}).items():
                w = float(weights.get(fname, 1.0))
                if w <= 0 or not text:
                    continue
                toks = tokenize(text)
                if not toks:
                    continue
                c = Counter(toks)
                for t, tf in c.items():
                    counts[t] += max(1, int(round(tf * w)))
            if not counts:
                continue
            doc_id = len(self.docs)
            self.docs.append(Doc(
                    doc_id=doc_id, 
                    url=url, 
                    title=title, 
                    keywords=fields.get("keywords"), 
                    status=fields.get("status"), 
                    network=fields.get("network"), 
                    country=fields.get("country"), 
                    genres=fields.get("genres"), 
                    cast=fields.get("cast"), 
                    characters=fields.get("characters"), 
                    description=fields.get("description"), 
                    episodes=fields.get("episodes")
                )
            )
            per_doc_counts.append(counts)

        self.N = len(self.docs)
        if self.N == 0:
            return

        for doc in self.docs:
            counts = per_doc_counts[doc.doc_id]
            for term, tf in counts.items():
                self.postings[term].append((doc.doc_id, tf))
            for term in counts.keys():
                self.df[term] = self.df.get(term, 0) + 1

        # idf
        if self.idf_mode == "classic":
            # idf = ln(N / df)
            for t, df in self.df.items():
                self.idf[t] = math.log(self.N / df) if df > 0 else 0.0

        elif self.idf_mode == "prob":
            # idf = ln( (N - df + 0.5) / (df + 0.5) )
            for t, df in self.df.items():
                num = (self.N - df + 0.5)
                den = (df + 0.5)
                if den <= 0:
                    self.idf[t] = 0.0
                else:
                    self.idf[t] = math.log(num / den)

        elif self.idf_mode == "invdf":
            # idf_inv(t) = 1 / (1 + df(t))
            for t, df in self.df.items():
                self.idf[t] = 1.0 / (1.0 + df)

        else:  # "smooth" default
            # smoothed idf = ln((1 + N)/(1 + df)) + 1
            for t, df in self.df.items():
                self.idf[t] = math.log((1 + self.N) / (1 + df)) + 1.0

        # norms
        for doc in self.docs:
            counts = per_doc_counts[doc.doc_id]
            s = 0.0
            for t, tf in counts.items():
                w_td = self._tfw(tf) * self.idf.get(t, 0.0)
                s += w_td * w_td
            self.doc_norm[doc.doc_id] = math.sqrt(max(s, 1e-12))

    def build_docs(self, shows, episodes, credits):
        episodes_by_show: Dict[str, List[str]] = defaultdict(list)

        for ep in episodes:
            show_url = ep.get("show_url", "") or ""
            title = ep.get("episode_title", "") or ""
            airdate_iso = ep.get("airdate_iso", "") or ""

            episodes_by_show[show_url].append(f"{title} ({airdate_iso})")

        actors_by_show: Dict[str, List[str]] = defaultdict(list)
        characters_by_show: Dict[str, List[str]] = defaultdict(list)

        for cr in credits:
            show_url = cr.get("show_url", "") or ""
            actor_name = cr.get("actor_name", "") or ""
            character_name = cr.get("role", "") or ""

            if actor_name:
                if actor_name not in actors_by_show[show_url]:
                    actors_by_show[show_url].append(actor_name)
            if character_name:
                if character_name not in characters_by_show[show_url]:
                    characters_by_show[show_url].append(character_name)

        # SHOW docs
        for s in shows:
            show_url = s.get("url", "") or ""
            title = s.get("title", "") or show_url
            genres = (s.get("genres") or "").replace("|", " ")

            episode_list = episodes_by_show.get(show_url, [])
            episode_str = "\n".join(episode_list)

            cast_list = sorted(actors_by_show.get(show_url, []))
            cast_str = ", ".join(cast_list)

            character_list = sorted(characters_by_show.get(show_url, []))
            character_str = ", ".join(character_list)

            fields = {
                "show_title": s.get("title", "") or "",
                "genres": genres,
                "description": s.get("description", "") or "",
                "cast": cast_str,
                "status": s.get("status", "") or "",
                "network": s.get("network", "") or "",
                "country": s.get("country", "") or "",
                "episodes": episode_str,
                "characters": character_str,
                "keywords": s.get("keywords", "") or "",
            }
            yield ("show", title, show_url, fields)

    def search(self, query: str, top_k: int = 20) -> list[tuple[float, Doc]]:
        if not query or self.N == 0:
            return []
        q_tokens = tokenize(query.strip())
        if not q_tokens:
            return []

        q_counts = Counter(q_tokens)
        q_vec: dict[str, float] = {}
        for t, tf in q_counts.items():
            q_vec[t] = self._tfw(tf) * self.idf.get(t, 0.0)

        q_norm = math.sqrt(sum(w*w for w in q_vec.values())) or 1.0 if self.normalize else 1.0

        scores: Dict[int, float] = defaultdict(float)
        for t, wq in q_vec.items():
            posts = self.postings.get(t)
            if not posts or wq == 0.0:
                continue
            idfw = self.idf.get(t, 0.0)
            for doc_id, tf in posts:
                w_td = self._tfw(tf) * idfw
                scores[doc_id] += w_td * wq

        if self.normalize:
            for d in list(scores.keys()):
                scores[d] /= (self.doc_norm.get(d, 1.0) * q_norm)

        for doc_id in list(scores.keys()):
            if " ".join(q_tokens) in self.docs[doc_id].title.lower():
                scores[doc_id] *= 1.3

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [(s, self.docs[d]) for d, s in ranked]
    
    def load_tsv(self, path: Path):
        with path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f, delimiter="\t"))

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "params": {"tf_sublinear": self.tf_sublinear, "idf_mode": self.idf_mode},
            "docs": [doc.__dict__ for doc in self.docs],
            "N": self.N,
            "postings": self.postings,
            "df": self.df,
            "idf": self.idf,
            "doc_norm": self.doc_norm,
        }
        path.write_text(json.dumps(data))

    @staticmethod
    def load(path: Path) -> "Indexer":
        obj = json.loads(path.read_text())
        idx = Indexer(**obj["params"])
        idx.docs = [Doc(**d) for d in obj["docs"]]
        idx.N = obj["N"]
        idx.postings = {t: [tuple(x) for x in v] for t, v in obj["postings"].items()}
        idx.df = obj["df"]
        idx.idf = {k: float(v) for k, v in obj["idf"].items()}
        idx.doc_norm = {int(k): float(v) for k, v in obj["doc_norm"].items()}
        return idx

    def run(self, weights: dict):
        shows = self.load_tsv(SHOWS)
        episodes = self.load_tsv(EPISODES)
        credits = self.load_tsv(CREDITS)

        all_docs = self.build_docs(shows, episodes, credits)
        self.build(all_docs, weights)
        self.save(INDEXER_DIR/f"index_{self.idf_mode}.json")


if __name__ == "__main__":
    weights = {
        "show": {
            "show_title": 2.1,
            "genres": 1.4,
            "description": 1.0,
            "cast": 0.9,
            "status": 0.5,
            "network": 0.5,
            "country": 0.5,
            "episodes": 1.0,
            "characters": 1.0,
            "keywords": 1.0,
        },
    }

    print("Building classic index...")
    idx = Indexer(tf_sublinear=True, idf_mode="classic")
    idx.run(weights)
    print(f"Indexed {idx.N} documents with classic indexing, saved to {INDEXER_DIR/f"index_{idx.idf_mode}.json"}")

    print("Building smooth index...")
    idx2 = Indexer(tf_sublinear=True, idf_mode="smooth")
    idx2.run(weights)
    print(f"Indexed {idx2.N} documents with smooth indexing, saved to {INDEXER_DIR/f"index_{idx2.idf_mode}.json"}")

    print("Building probabilistic index...")
    idx3 = Indexer(tf_sublinear=True, idf_mode="prob")
    idx3.run(weights)
    print(f"Indexed {idx3.N} documents with probabilistic indexing, saved to {INDEXER_DIR/f"index_{idx3.idf_mode}.json"}")

    print("Building inverse document frequency index...")
    idx4 = Indexer(tf_sublinear=True, idf_mode="invdf")
    idx4.run(weights)
    print(f"Indexed {idx4.N} documents with inverse document frequency indexing, saved to {INDEXER_DIR/f"index_{idx4.idf_mode}.json"}")
