from __future__ import annotations
import csv, json, math, re
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Iterable, Optional

LOGS = Path("./logs")

SHOWS = LOGS/"extracted_shows.tsv"
EPISODES = LOGS/"extracted_episodes.tsv"
CREDITS = LOGS/"extracted_credits.tsv"

OUT = LOGS/"index_unified.json"


TOKEN_RE = re.compile(r"[a-z0-9]+")

def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())

@dataclass
class Doc:
    doc_id: int
    doc_type: str      # "show" | "episode" | "actor"
    url: str
    title: str
    extra: dict

class Indexer:
    def __init__(self, tf_sublinear=True, idf_smoothing=True, normalize=True):
        self.tf_sublinear = tf_sublinear
        self.idf_smoothing = idf_smoothing
        self.normalize = normalize

        self.docs: list[Doc] = []
        self.N = 0
        # term -> list of (doc_id, tf)
        self.postings: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        self.df: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.doc_norm: Dict[int, float] = {}

    def _tfw(self, tf: int) -> float:
        return (1.0 + math.log(tf)) if (self.tf_sublinear and tf > 0) else float(tf)

    def build(self, iterable: Iterable[Tuple[str, str, str, Dict[str, str]]], field_weights: Dict[str, Dict[str, float]]):
        """
        iterable yields: (doc_type, title, url, fields_dict)
          - doc_type: "show" | "episode" | "actor"
          - fields_dict: name -> text
        field_weights: {"show": {...}, "episode": {...}, "actor": {...}}
        """
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
            self.docs.append(Doc(doc_id=doc_id, doc_type=doc_type, url=url, title=title, extra=fields))
            per_doc_counts.append(counts)

        self.N = len(self.docs)
        if self.N == 0:
            return

        # postings + df
        for doc in self.docs:
            counts = per_doc_counts[doc.doc_id]
            for term, tf in counts.items():
                self.postings[term].append((doc.doc_id, tf))
            for term in counts.keys():
                self.df[term] = self.df.get(term, 0) + 1

        # idf
        if self.idf_smoothing:
            self.idf = {t: math.log((1 + self.N) / (1 + df)) + 1.0 for t, df in self.df.items()}
        else:
            self.idf = {t: math.log(self.N / df) for t, df in self.df.items()}

        # norms
        for doc in self.docs:
            counts = per_doc_counts[doc.doc_id]
            s = 0.0
            for t, tf in counts.items():
                w_td = self._tfw(tf) * self.idf.get(t, 0.0)
                s += w_td * w_td
            self.doc_norm[doc.doc_id] = math.sqrt(max(s, 1e-12))

    def _parse_query(self, query: str):
        # simple filter: type:show|episode|actor
        q = query.strip()
        doc_type = None
        m = re.search(r"\btype:(show|episode|actor)\b", q, re.I)
        if m:
            doc_type = m.group(1).lower()
            q = (q[:m.start()] + q[m.end():]).strip()
        return doc_type, tokenize(q)

    def search(self, query: str, top_k: int = 20) -> list[tuple[float, Doc]]:
        if not query or self.N == 0:
            return []
        type_filter, q_tokens = self._parse_query(query)
        if not q_tokens:
            # return top docs by length? keep simple: empty = no results
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
                if type_filter and self.docs[doc_id].doc_type != type_filter:
                    continue
                w_td = self._tfw(tf) * idfw
                scores[doc_id] += w_td * wq

        if self.normalize:
            for d in list(scores.keys()):
                scores[d] /= (self.doc_norm.get(d, 1.0) * q_norm)

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [(s, self.docs[d]) for d, s in ranked]

    # persistence
    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "params": {"tf_sublinear": self.tf_sublinear, "idf_smoothing": self.idf_smoothing, "normalize": self.normalize},
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
    
    def load_tsv(self, path: Path):
        with path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f, delimiter="\t"))

    def build_docs(self, shows, episodes, credits):
        # helpers
        shows_by_url = {s["url"]: s for s in shows}

        # show_url -> [actor_name] and actor_name -> [(show_title, role)]
        show_cast = defaultdict(list)
        actor_map = defaultdict(list)
        for c in credits:
            url = c.get("show_url","")
            name = c.get("actor_name","")
            role = c.get("role","")
            if not name:
                continue
            show_cast[url].append(name)
            show_title = shows_by_url.get(url, {}).get("title","")
            actor_map[name].append((show_title, role))

        # --- SHOW docs ---
        for s in shows:
            url = s.get("url","")
            keywords = s.get("keywords","").replace("|"," ")
            title = s.get("title","") or url
            genres = (s.get("genres") or "").replace("|", " ")
            cast = " ".join(show_cast.get(url, []))
            fields = {
                "show_title": s.get("title",""),
                "genres": genres,
                "description": s.get("description",""),
                "keywords": keywords,
                "cast": cast,
                "status": s.get("status",""),
                "network": s.get("network",""),
                "country": s.get("country",""),
            }
            yield ("show", title, url, fields)

        # --- EPISODE docs ---
        for e in episodes:
            url = e.get("show_url","")
            s = shows_by_url.get(url, {})
            show_title = s.get("title","")
            ep_title = e.get("episode_title","")
            season = e.get("season_number","")
            epno = e.get("episode_number_in_season","")
            title = f"{show_title} — S{season}E{epno}: {ep_title}".strip(" -—:")
            genres = (s.get("genres") or "").replace("|"," ")
            cast = " ".join(show_cast.get(url, []))
            fields = {
                "episode_title": ep_title,
                "show_title": show_title,
                "genres": genres,
                "cast": cast,
                "description": s.get("description",""),
                "status": s.get("status",""),
                "network": s.get("network",""),
                "country": s.get("country",""),
            }
            # keep some metadata for display
            fields["meta"] = f"S{season}E{epno} {e.get('airdate_iso','')}".strip()
            yield ("episode", title, url, fields)

        # --- ACTOR docs ---
        for actor, items in actor_map.items():
            # aggregate all shows/roles into a single context text
            shows_str = " ".join(t for (t, _) in items if t)
            roles_str = " ".join(r for (_, r) in items if r)
            # for URL we can keep empty or use a pseudo-url
            title = actor
            fields = {
                "actor_name": actor,
                "shows": shows_str,
                "roles": roles_str,
            }
            yield ("actor", title, "", fields)


    def run(self, weights: dict):
        shows = self.load_tsv(SHOWS)
        episodes = self.load_tsv(EPISODES)
        credits = self.load_tsv(CREDITS)

        all_docs = self.build_docs(shows, episodes, credits)
        self.build(all_docs, weights)
        self.save(OUT)


if __name__ == "__main__":
    weights = {
        "show": {
            "keywords": 1.0,
            "show_title": 2.0,
            "genres": 1.3,
            "description": 1.2,
            "cast": 1.0,
            "status": 0.5,
            "network": 0.5,
            "country": 0.5,
        },
        "episode": {
            "episode_title": 2.2,
            "show_title": 1.6,
            "genres": 1.2,
            "cast": 1.0,
            "description": 0.9,
            "status": 0.4,
            "network": 0.4,
            "country": 0.4,
            "meta": 0.6,
        },
        "actor": {
            "actor_name": 2.2,
            "shows": 1.4,
            "roles": 1.0,
        }
    }

    idx = Indexer(tf_sublinear=True, idf_smoothing=True, normalize=True)
    idx.run(weights)
    print(f"Indexed {idx.N} documents, saved to {OUT}")

