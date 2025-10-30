from __future__ import annotations
import csv, json, math, re
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Iterable

EXTRACTED_DIR = Path("./extracted")

SHOWS = EXTRACTED_DIR/"extracted_shows.tsv"
EPISODES = EXTRACTED_DIR/"extracted_episodes.tsv"
CREDITS = EXTRACTED_DIR/"extracted_credits.tsv"
ACTORS = EXTRACTED_DIR/"extracted_actors.tsv"

INDEXER_DIR = Path("./indexer")
OUT = INDEXER_DIR/"index.json"


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
    def __init__(self, 
                 tf_sublinear=True, 
                 idf_mode: str = "smooth",   # "classic" | "smooth" | "prob"
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
            self.docs.append(Doc(doc_id=doc_id, doc_type=doc_type, url=url, title=title, extra=fields))
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

    def build_docs(self, shows, episodes, credits, actors):
        shows_by_url = {s["url"]: s for s in shows}

        actor_urls = {}
        for a in actors:
            name = (a.get("actor_name") or "").strip()
            url  = (a.get("actor_url") or "").strip()
            if name:
                actor_urls[name] = url

        show_cast: dict[str, set[str]] = defaultdict(set)

        actor_map: dict[str, dict] = defaultdict(lambda: {"url": "", "shows": defaultdict(lambda: {"title": "", "roles": set()})})

        for c in credits:
            show_url = (c.get("show_url") or "").strip()
            actor_name = (c.get("actor_name") or "").strip()
            role = (c.get("role") or "").strip()
            if not show_url or not actor_name:
                continue

            show_cast[show_url].add(actor_name)

            show_title = shows_by_url.get(show_url, {}).get("title", "").strip()

            actor_entry = actor_map[actor_name]
            
            if actor_name in actor_urls:
                actor_entry["url"] = actor_urls[actor_name]

            show_entry = actor_entry["shows"][show_url]
            if show_title:
                show_entry["title"] = show_title
            if role:
                show_entry["roles"].add(role)

        # SHOW docs
        for s in shows:
            show_url = s.get("url", "") or ""
            title = s.get("title", "") or show_url
            genres = (s.get("genres") or "").replace("|", " ")
            
            cast_list = sorted(show_cast.get(show_url, []))
            cast_str = ", ".join(cast_list)

            fields = {
                "show_title": s.get("title", "") or "",
                "genres": genres,
                "description": s.get("description", "") or "",
                "cast": cast_str,
                "status": s.get("status", "") or "",
                "network": s.get("network", "") or "",
                "country": s.get("country", "") or "",
            }
            yield ("show", title, show_url, fields)

        # EPISODE docs
        for e in episodes:
            show_url = e.get("show_url", "") or ""
            s = shows_by_url.get(show_url, {})
            show_title = s.get("title", "") or ""
            ep_title = e.get("episode_title", "") or ""
            season = e.get("season_number", "") or ""
            epno = e.get("episode_number_in_season", "") or ""
            title = f"{show_title} — S{season}E{epno}: {ep_title}".strip(" -—:")
            ep_url = e.get("episode_url", "") or ""

            genres = (s.get("genres") or "").replace("|", " ")
            cast_list = sorted(show_cast.get(show_url, []))
            cast_str = ", ".join(cast_list)

            fields = {
                "episode_title": ep_title,
                "show_title": show_title,
                "genres": genres,
                "cast": cast_str,
                "description": s.get("description", "") or "",
                "status": s.get("status", "") or "",
                "network": s.get("network", "") or "",
                "country": s.get("country", "") or "",
                "meta": f"S{season}E{epno} {e.get('airdate_iso','')}".strip(),
            }
            yield ("episode", title, ep_url, fields)

        # ACTOR docs
        for actor_name, ainfo in actor_map.items():
            actor_url = ainfo.get("url", "") or ""
            shows_blob = []
            roles_blob = []

            for s_url, sinfo in ainfo["shows"].items():
                stitle = sinfo.get("title", "") or s_url
                shows_blob.append(stitle)

                for role in sorted(sinfo["roles"]):
                    roles_blob.append(f"{stitle}: {role}")

            fields = {
                "actor_name": actor_name,
                "shows": ", ".join(shows_blob),
                "roles": ", ".join(roles_blob),
            }
            yield ("actor", actor_name, actor_url, fields)


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
        actors = self.load_tsv(ACTORS)

        all_docs = self.build_docs(shows, episodes, credits, actors)
        self.build(all_docs, weights)
        self.save(OUT)


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
        },
        "episode": {
            "episode_title": 2.0,
            "show_title": 1.6,
            "genres": 1.2,
            "cast": 1.0,
            "description": 0.8,
            "status": 0.4,
            "network": 0.4,
            "country": 0.4,
            "meta": 0.6,
        },
        "actor": {
            "actor_name": 2.3,
            "shows": 1.5,
            "roles": 1.1,
        },
    }

    idx = Indexer(tf_sublinear=True, idf_mode="prob")
    idx.run(weights)
    print(f"Indexed {idx.N} documents, saved to {OUT}")

