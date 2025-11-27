from __future__ import annotations
from pathlib import Path
import os, sys, csv, re
from datetime import datetime

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.sql import SparkSession, DataFrame, functions as F, types as T

DATA_DIR      = "enwiki-dumps/enwiki-latest"
EXTRACTED_DIR = "extracted"

SHOWS_FILE    = f"{EXTRACTED_DIR}/extracted_shows.tsv"
EPISODES_FILE = f"{EXTRACTED_DIR}/extracted_episodes.tsv"
ACTORS_FILE   = f"{EXTRACTED_DIR}/extracted_actors.tsv"

WIKI_SHOWS_TSV    = f"{EXTRACTED_DIR}/wiki_shows.tsv"
WIKI_EPISODES_TSV = f"{EXTRACTED_DIR}/wiki_episodes.tsv"
WIKI_ACTORS_TSV   = f"{EXTRACTED_DIR}/wiki_actors.tsv"

WIKI_CATEGORIES_FILE   = f"{EXTRACTED_DIR}/television_categories.txt"
WIKI_CHUNK_MAP_TSV     = f"{EXTRACTED_DIR}/wiki_chunk_progress.tsv"

from logger import WikiLogger
logger = WikiLogger()

SHOW_COLS = [
    "page_id",
    "title",
    "norm_title",
    "entity_type",
    "start_year",
    "page_start_year",
    "infobox_type",
    "alt_title",
    "native_title",
    "based_on",
    "inspired_by",
    "creator",
    "developer",
    "showrunner",
    "writer",
    "screenplay",
    "teleplay",
    "story",
    "director",
    "creative_director",
    "presenter",
    "starring",
    "judges",
    "voices",
    "narrator",
    "theme_music_composer",
    "open_theme",
    "end_theme",
    "composer",
    "language",
    "num_specials",
    "executive_producer",
    "producer",
    "news_editor",
    "location",
    "cinematography",
    "animator",
    "editor",
    "camera",
    "production_company",
    "budget",
    "list_episodes",
    "related",
    "description",
    "categories",
]


def create_spark(app_name: str = "ShowWikiExtractor") -> SparkSession:
    return (
        SparkSession.builder
        .appName(app_name)
        .master("local[12]")
        .config("spark.driver.memory", "4g")
        .config("spark.executor.memory", "4g")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.shuffle.partitions", "96")
        .config("spark.sql.files.maxPartitionBytes", "134217728")
        .config("spark.local.dir", "D:/spark_tmp")
        .config("spark.sql.execution.pyspark.udf.faulthandler.enabled", "true")
        .config("spark.python.worker.faulthandler.enabled", "true")
        .getOrCreate()
    )


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def norm_title(col):
    """
    Normalize titles for matching shows / episodes / actors with Wikipedia pages.
    """
    c = F.lower(F.trim(col))
    c = F.regexp_replace(
        c,
        r"\s*\((american|british|canadian|australian|indian|south korean|japanese)\s+(tv|television)\s+series\)\s*$",
        "",
    )
    c = F.regexp_replace(c, r"\s*\(\d{4}\s+tv\s+series\)\s*$", "")
    c = F.regexp_replace(c, r"\s*\(\d{4}.*?\)\s*$", "")
    c = F.regexp_replace(c, r"[-–—_:;/\.,!?]+", " ")
    c = F.regexp_replace(c, r"\s+", " ")
    return F.trim(c)


def _clean_wiki_value(value: str) -> str:
    """
    Clean a single infobox value
    """
    if not value:
        return ""

    v = value

    # --- 0) Special-case 'Collapsible list' and similar list templates ---
    # Keep only bullet lines; ignore title/etc.
    if "{{Collapsible list" in v:
        bullet_lines = [
            ln for ln in v.splitlines()
            if ln.lstrip().startswith("*")
        ]
        if bullet_lines:
            v = "\n".join(bullet_lines)

    # --- 1) Kill <ref>...</ref> and <ref .../> ---
    v = re.sub(r"<ref[^>]*>.*?</ref>", " ", v, flags=re.IGNORECASE | re.DOTALL)
    v = re.sub(r"<ref[^>]*/>", " ", v, flags=re.IGNORECASE)

    # --- 2) Unwrap Plainlist-like wrappers but keep inner text ---
    v = re.sub(
        r"\{\{\s*(plainlist|unbulleted list|ubl|hlist|idp|Efn)\s*\|",
        "",
        v,
        flags=re.IGNORECASE,
    )

    # --- 3) Drop leftover '{{' and '}}' (after unwrapping list templates) ---
    v = v.replace("{{", " ").replace("}}", " ")

    # --- 4) Wiki links: [[Foo|Bar]] -> Bar, [[Foo]] -> Foo ---
    v = re.sub(r"\[\[(?:[^\|\]]*\|)?([^\]]+)\]\]", r"\1", v)

    # --- 5) Drop HTML-like tags ---
    v = re.sub(r"<[^>]+>", " ", v)

    # --- 6) Remove bullets/numbering at line starts ---
    v = re.sub(r"^[\*\#:\;\u2022]+\s*", "", v, flags=re.MULTILINE)

    # --- 7) Newlines -> commas for list-ish fields ---
    v = re.sub(r"[\r\n]+", ", ", v)

    # --- 8) Collapse whitespace & trim separators ---
    v = re.sub(r"\s+", " ", v)
    v = v.strip(" ,;")

    return v


def extract_infobox_py(raw_text: str) -> dict:
    """
    Extract ONLY the 'Infobox television' template into a dict.
    Handles multiline values and nested templates.
    """
    if not raw_text:
        return {}

    # 1) find start of {{Infobox television ...}}
    m = re.search(r"\{\{\s*infobox\s+television\b", raw_text, flags=re.IGNORECASE)
    if not m:
        return {}

    start = m.start()
    text = raw_text

    # 2) walk with brace depth to find the matching closing }}
    depth = 0
    pos = start
    end = None
    n = len(text)

    while pos < n - 1:
        if text.startswith("{{", pos):
            depth += 1
            pos += 2
            continue
        if text.startswith("}}", pos):
            depth -= 1
            pos += 2
            if depth == 0:
                end = pos
                break
            continue
        pos += 1

    if end is None:
        end = n

    template_block = text[start:end]
    lines = template_block.splitlines()
    if not lines:
        return {}

    info: dict[str, str] = {"_infobox_type": "television"}

    # 3) parameter lines (skip header {{Infobox television, drop final }} line)
    body_lines = lines[1:]
    if body_lines and body_lines[-1].strip().startswith("}}"):
        body_lines = body_lines[:-1]

    param_re = re.compile(r"^\|\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$")

    current_key = None
    current_val_lines: list[str] = []

    for line in body_lines:
        m2 = param_re.match(line.rstrip("\n"))
        if m2:
            if current_key is not None:
                raw_val = "\n".join(current_val_lines).strip()
                if raw_val:
                    info[current_key] = _clean_wiki_value(raw_val)

            current_key = m2.group(1).lower()
            first_val = m2.group(2)
            current_val_lines = [first_val]
        else:
            if current_key is not None:
                current_val_lines.append(line)

    if current_key is not None:
        raw_val = "\n".join(current_val_lines).strip()
        if raw_val:
            info[current_key] = _clean_wiki_value(raw_val)

    return info


def remove_first_infobox_block(raw_text: str) -> str:
    """
    Remove the first Infobox block from wikitext.
    Uses brace depth so it works with nested templates and multiline values.
    """
    if not raw_text:
        return ""

    # Find '{{Infobox ...' (any type)
    m = re.search(r"\{\{\s*infobox\b", raw_text, flags=re.IGNORECASE)
    if not m:
        return raw_text  # no infobox found

    start = m.start()
    text = raw_text
    n = len(text)

    depth = 0
    pos = start
    end = None

    while pos < n - 1:
        if text.startswith("{{", pos):
            depth += 1
            pos += 2
            continue
        if text.startswith("}}", pos):
            depth -= 1
            pos += 2
            if depth == 0:
                end = pos
                break
            continue
        pos += 1

    if end is None:
        end = n

    while end < n and text[end] in " \t\r\n":
        end += 1

    return text[:start] + text[end:]



def extract_categories_py(raw_text: str) -> str | None:
    if not raw_text:
        return None

    cats: list[str] = []
    for line in raw_text.splitlines():
        if "[[Category:" in line:
            for m in re.findall(r"\[\[Category:([^|\]]+)", line):
                c = m.strip()
                if c and c not in cats:
                    cats.append(c)

    return ", ".join(cats) if cats else None

def remove_all_templates(text: str) -> str:
    """
    Remove all {{...}} templates from wikitext, including nested templates,
    using brace depth. This is intentionally lossy but fine for summaries.
    """
    if not text:
        return ""

    result_chars = []
    depth = 0
    i = 0
    n = len(text)

    while i < n:
        if text.startswith("{{", i):
            depth += 1
            i += 2
            continue
        if text.startswith("}}", i) and depth > 0:
            depth -= 1
            i += 2
            continue
        if depth > 0:
            i += 1
            continue

        result_chars.append(text[i])
        i += 1

    return "".join(result_chars)

def clean_text_py(raw_text: str) -> str:
    if not raw_text:
        return ""

    # 1) remove the whole infobox
    text = remove_first_infobox_block(raw_text)

    # 2) remove HTML comments
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)

    # 3) remove all <ref>...</ref> and <ref .../>
    text = re.sub(r"<ref[^>]*>.*?</ref>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<ref[^>]*/>", " ", text, flags=re.IGNORECASE)

    # 4) remove ALL templates {{...}}
    text = remove_all_templates(text)

    # 5) unwrap wiki italics/bold markup: '''''Title''''' -> Title, ''Title'' -> Title
    text = re.sub(r"''+([^'].*?)''+", r"\1", text)

    # 6) remove File/Image links
    text = re.sub(r"\[\[(File|Image):[^\]]+\]\]", " ", text)

    # 7) wiki links [[Foo|Bar]] -> Bar, [[Foo]] -> Foo
    text = re.sub(r"\[\[(?:[^\|\]]*\|)?([^\]]+)\]\]", r"\1", text)

    # 8) URLs
    text = re.sub(r"https?://\S+", " ", text)

    # 9) strip section headings == Heading ==
    text = re.sub(r"={2,}\s*(.*?)\s*={2,}", r" \1 ", text)

    # 10) drop HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # 11) compress spaces/tabs only, keep newlines
    text = re.sub(r"[ \t]{2,}", " ", text)

    # 12) normalize excessive blank lines (3+ -> 2)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()

def extract_first_paragraph_py(clean_text: str) -> str | None:
    if not clean_text:
        return None

    text = clean_text.strip()

    # 1) Try paragraph split on blank lines
    parts = re.split(r"\n\s*\n", text)
    first_block = ""
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if len(p) < 40:
            # skip very short fragments (e.g. artifacts)
            continue
        first_block = p
        break

    if not first_block:
        first_block = text

    # 2) Fallback / refinement: use first 2–3 sentences from that block
    sentences = re.split(r'(?<=[.!?])\s+', first_block)
    if sentences:
        candidate = " ".join(sentences[:3]).strip()
    else:
        candidate = first_block

    if not candidate:
        candidate = first_block

    # 3) FINAL: normalize ALL whitespace (including newlines) to single spaces
    candidate = re.sub(r"\s+", " ", candidate).strip()

    return candidate

def extract_start_year_from_infobox_py(info: dict) -> int | None:
    if not info:
        return None

    candidate_keys = [
        "original_release",
        "first_aired",
        "released",
        "broadcast",
        "air_date",
        "release_date",
    ]

    for key in candidate_keys:
        v = info.get(key)
        if not v:
            continue
        m = re.search(r"\b(19|20)\d{2}\b", v)
        if m:
            try:
                return int(m.group(0))
            except ValueError:
                pass
    return None


def make_category_match_fn(keywords: list[str]):
    kw_lower = [k.lower() for k in keywords]

    def _matches(categories: str | None) -> bool:
        if not categories or not kw_lower:
            return False

        cats = [
            c.strip().lower()
            for c in categories.split("|")
            if c.strip()
        ]

        for c in cats:
            for kw in kw_lower:
                if kw in c:
                    return True
        return False

    return _matches


@F.udf(returnType=T.MapType(T.StringType(), T.StringType()))
def extract_infobox(raw_text: str):
    """
    Extract first Infobox as a map of key -> value (strings).
    Stores the infobox template type under key '_infobox_type'.
    """
    if not raw_text:
        return {}

    lines = raw_text.splitlines()
    start_idx = None
    for i, line in enumerate(lines):
        if line.lstrip().lower().startswith("{{infobox"):
            start_idx = i
            break

    if start_idx is None:
        return {}

    info = {}

    header_line = lines[start_idx].strip()
    m = re.match(r"\{\{Infobox\s+([^\|\n]+)", header_line, flags=re.IGNORECASE)
    if m:
        info["_infobox_type"] = m.group(1).strip()

    for line in lines[start_idx + 1 :]:
        stripped = line.strip()
        if stripped.startswith("}}"):
            break
        if not stripped.startswith("|"):
            continue

        stripped = stripped[1:] 
        if "=" not in stripped:
            continue

        key, value = stripped.split("=", 1)
        key = key.strip()
        value = _clean_wiki_value(value)
        if key and value:
            info[key] = value

    return info


@F.udf(returnType=T.StringType())
def extract_first_paragraph(clean_text: str):
    """
    Take cleaned article text and return ~first paragraph (first non-trivial block).
    """
    if not clean_text:
        return None

    text = clean_text.strip()
    parts = re.split(r"\n\s*\n", text)
    for p in parts:
        p = p.strip()
        if len(p) >= 40:  # skip tiny fragments / headers
            return p
    return parts[0].strip() if parts else None


@F.udf(returnType=T.StringType())
def extract_categories(raw_text: str):
    """
    Extract [[Category:...]] entries and join them with '|'.
    """
    if not raw_text:
        return None

    cats = []
    for line in raw_text.splitlines():
        if "[[Category:" in line:
            for m in re.findall(r"\[\[Category:([^|\]]+)", line):
                c = m.strip()
                if c and c not in cats:
                    cats.append(c)

    if not cats:
        return None
    return "|".join(cats)


@F.udf(returnType=T.IntegerType())
def extract_start_year_from_infobox(info: dict | None):
    """
    Try to extract a start year (e.g. 1961) from the infobox.
    Useful for disambiguating shows with the same title by year.
    """
    if not info:
        return None

    candidate_keys = [
        "original_release",
        "first_aired",
        "released",
        "broadcast",
        "air_date",
        "release_date",
    ]

    for key in candidate_keys:
        v = info.get(key)
        if not v:
            continue
        m = re.search(r"\b(19|20)\d{2}\b", v)
        if m:
            try:
                return int(m.group(0))
            except ValueError:
                pass
    return None

def make_category_match_udf(keywords: list[str]):
    """
    Build a UDF that returns True if at least one keyword appears
    in at least one of the page's categories (substring, case-insensitive).
    """
    kw_lower = [k.lower() for k in keywords]

    @F.udf(returnType=T.BooleanType())
    def _matches(categories: str | None):
        if not categories or not kw_lower:
            return False

        cats = [
            c.strip().lower()
            for c in categories.split("|")
            if c.strip()
        ]

        for c in cats:
            for kw in kw_lower:
                if kw in c:
                    return True
        return False

    return _matches


class SparkWikiExtractor:
    def __init__(self):
        self.spark: SparkSession = create_spark()
        self.allowed_categories = self._load_allowed_categories()
        self.category_match_fn = (
            make_category_match_fn(self.allowed_categories)
            if self.allowed_categories
            else None
        )
        self.processed_chunks = self._load_processed_chunks()
        self.load_extracted_data()

    def _load_allowed_categories(self) -> list[str]:
        """
        Load list of topic-related *keywords* from extracted/wiki_media_categories.txt.
        Each line = one keyword (lowercased match against category names).
        """
        path = Path(WIKI_CATEGORIES_FILE)
        if not path.exists():
            logger.log(
                f"Category keyword file not found: {WIKI_CATEGORIES_FILE}. "
                f"No category filtering will be applied."
            )
            return []

        keywords: list[str] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            c = line.strip()
            if not c or c.startswith("#"):
                continue
            keywords.append(c)

        logger.log(
            f"Loaded {len(keywords)} category keywords "
            f"from {WIKI_CATEGORIES_FILE}"
        )
        return keywords

    def _load_processed_chunks(self) -> set[str]:
        """
        Load mapping of already-processed chunk files from WIKI_CHUNK_MAP_TSV.
        Format: chunk_file \t rows \t processed_at
        """
        path = Path(WIKI_CHUNK_MAP_TSV)
        if not path.exists():
            return set()

        processed = set()
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader, None)
            for row in reader:
                if not row:
                    continue
                chunk_file = row[0]
                processed.add(chunk_file)
        logger.log(f"Loaded {len(processed)} previously processed chunk(s).")
        return processed

    def _append_chunk_mapping(self, chunk_path: str, rows: int):
        """
        Append a record about a processed chunk to the mapping TSV.
        """
        path = Path(WIKI_CHUNK_MAP_TSV)
        _ensure_dir(path.parent)

        file_exists = path.exists()
        with path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            if not file_exists:
                writer.writerow(["chunk_file", "rows", "processed_at"])
            writer.writerow([
                chunk_path,
                rows,
                datetime.utcnow().isoformat(timespec="seconds")
            ])

        self.processed_chunks.add(chunk_path)

    def load_extracted_data(self):
        logger.log("Loading extracted TSV files (titles / names only)...")

        # Shows: title + date_start_iso -> start_year
        shows_raw = (
            self.spark.read
            .option("header", True)
            .option("sep", "\t")
            .csv(SHOWS_FILE)
        )

        self.shows = (
            shows_raw
            .select(
                F.col("title"),
                F.col("date_start_iso"),
            )
            .where(F.col("title").isNotNull())
            .withColumn(
                "start_year",
                F.substring("date_start_iso", 1, 4).cast("int")
            )
            .distinct()
        )

        # # Episodes: only episode titles
        # self.episodes = (
        #     self.spark.read
        #     .option("header", True)
        #     .option("sep", "\t")
        #     .csv(EPISODES_FILE)
        #     .select("episode_title")
        #     .where(F.col("episode_title").isNotNull())
        #     .distinct()
        # )

        # # Actors: try actor_name, fallback to name/title
        # actors_raw = (
        #     self.spark.read
        #     .option("header", True)
        #     .option("sep", "\t")
        #     .csv(ACTORS_FILE)
        # )

        # self.actors = (
        #     actors_raw
        #     .select(F.col("actor_name"))
        #     .where(F.col("actor_name").isNotNull())
        #     .distinct()
        # )

        logger.log(
            f"Loaded TSVs: shows={self.shows.count()}, "
            # f"episodes={self.episodes.count()}, actors={self.actors.count()}"
        )

    def build_title_whitelist(self) -> DataFrame:
        """
        Build a whitelist of normalized titles from shows, episodes, and actors.
        For shows we also keep start_year derived from date_start_iso.
        Also tag each row with entity_type ('show', 'episode', 'actor').
        """
        logger.log("Building whitelist of titles from shows, episodes, and actors...")

        show_titles = (
            self.shows
            .select(
                F.col("title").alias("page_title"),
                F.col("start_year").alias("start_year"),
                F.lit("show").alias("entity_type"),
            )
        )

        # episode_titles = (
        #     self.episodes
        #     .select(
        #         F.col("episode_title").alias("page_title"),
        #         F.lit(None).cast("int").alias("start_year"),
        #         F.lit("episode").alias("entity_type"),
        #     )
        # )

        # actor_titles = (
        #     self.actors
        #     .select(
        #         F.col("actor_name").alias("page_title"),
        #         F.lit(None).cast("int").alias("start_year"),
        #         F.lit("actor").alias("entity_type"),
        #     )
        # )

        # all_titles = (
        #     show_titles
        #     .unionByName(episode_titles)
        #     .unionByName(actor_titles)
        #     .where(F.col("page_title").isNotNull())
        # )

        whitelist = (
            # all_titles
            show_titles
            .withColumn("norm_title", norm_title(F.col("page_title")))
            .select("norm_title", "start_year", "entity_type")
            .dropDuplicates(["norm_title", "entity_type", "start_year"])
        )

        logger.log(
            f"Title whitelist built. Unique normalized titles (by entity_type+year): "
            f"{whitelist.count()}"
        )
        return whitelist

    def parse_wiki_dump_chunk(self, dump_path: str, title_whitelist: DataFrame) -> DataFrame:
        logger.log(f"Parsing Wikipedia dump chunk: {dump_path}")

        wl = F.broadcast(title_whitelist)

        wiki = (
            self.spark.read.format("xml")
            .option("rowTag", "page")
            .load(dump_path)
            .select(
                F.col("id").cast("long").alias("page_id"),
                F.col("title").cast("string").alias("title"),
                F.col("revision.text._VALUE").cast("string").alias("raw_text"),
            )
        )

        wiki = wiki.withColumn("norm_title", norm_title(F.col("title")))

        # Keep pages whose normalized title appears in whitelist, skip redirects
        candidates = (
            wiki
            .join(wl, on="norm_title", how="inner")
            .filter(~F.lower(F.col("raw_text")).startswith("#redirect"))
            .select(
                "page_id",
                "title",
                "norm_title",
                "raw_text",
                "entity_type",
                "start_year",
            )
        )

        logger.log(
            f"Candidate pages in chunk after title whitelist & redirect filter: "
            f"{candidates.count()}"
        )

        # clean = F.col("raw_text")
        # clean = F.regexp_replace(clean, r"\{\{[^}{]*\}\}", " ")
        # clean = F.regexp_replace(clean, r"\[\[(File|Image):[^\]]+\]\]", " ")
        # clean = F.regexp_replace(clean, r"\[\[(?:[^\|\]]*\|)?([^\]]+)\]\]", r"\1")
        # clean = F.regexp_replace(clean, r"https?://\S+", " ")
        # clean = F.regexp_replace(clean, r"={2,}\s*(.*?)\s*={2,}", r" \1 ")
        # clean = F.regexp_replace(clean, r"<[^>]+>", " ")
        # clean = F.regexp_replace(clean, r"\s{2,}", " ")

        # wiki_cleaned = candidates.withColumn("clean_text", F.trim(clean))

        # # Enrich with description + infobox + categories
        # wiki_enriched = (
        #     wiki_cleaned
        #     .withColumn("infobox", extract_infobox(F.col("raw_text")))
        #     .withColumn("page_start_year", extract_start_year_from_infobox(F.col("infobox")))
        #     .withColumn("description", extract_first_paragraph(F.col("clean_text")))
        #     .withColumn("categories", extract_categories(F.col("raw_text")))
        # )

        # # Filter by allowed categories (efficiently via broadcast join)
        # if self.category_match_udf is not None:
        #     with_topic = (
        #         wiki_enriched
        #         .withColumn(
        #             "category_match",
        #             self.category_match_udf(F.col("categories"))
        #         )
        #         .filter(F.col("category_match"))
        #         .drop("category_match")
        #     )

        #     logger.log(
        #         f"Pages remaining in chunk after category keyword filter: "
        #         f"{with_topic.count()}"
        #     )
        # else:
        #     with_topic = wiki_enriched


        # # Structured fields from infobox
        # result = with_topic.select(
        #     F.col("page_id"),
        #     F.col("title"),
        #     F.col("norm_title"),
        #     F.col("entity_type"),
        #     F.col("start_year"),      # from your TSVs
        #     F.col("page_start_year"), # derived from infobox (for disambiguation)
        #     F.col("infobox")["_infobox_type"].alias("infobox_type"),

        #     # --- naming ---
        #     F.col("infobox")["alt_name"].alias("alt_title"),
        #     F.col("infobox")["native_name"].alias("native_title"),

        #     # --- origin / content ---
        #     F.col("infobox")["based_on"].alias("based_on"),
        #     F.col("infobox")["inspired_by"].alias("inspired_by"),

        #     # --- creators / people ---
        #     F.col("infobox")["creator"].alias("creator"),
        #     F.col("infobox")["developer"].alias("developer"),
        #     F.col("infobox")["showrunner"].alias("showrunner"),
        #     F.col("infobox")["writer"].alias("writer"),
        #     F.col("infobox")["screenplay"].alias("screenplay"),
        #     F.col("infobox")["teleplay"].alias("teleplay"),
        #     F.col("infobox")["story"].alias("story"),
        #     F.col("infobox")["director"].alias("director"),
        #     F.col("infobox")["creative_director"].alias("creative_director"),
        #     F.col("infobox")["presenter"].alias("presenter"),
        #     F.col("infobox")["starring"].alias("starring"),
        #     F.col("infobox")["judges"].alias("judges"),
        #     F.col("infobox")["voices"].alias("voices"),
        #     F.col("infobox")["narrator"].alias("narrator"),

        #     # --- music ---
        #     F.col("infobox")["theme_music_composer"].alias("theme_music_composer"),
        #     F.col("infobox")["open_theme"].alias("open_theme"),
        #     F.col("infobox")["end_theme"].alias("end_theme"),
        #     F.col("infobox")["composer"].alias("composer"),

        #     # --- language & counts ---
        #     F.col("infobox")["language"].alias("language"),
        #     F.col("infobox")["num_specials"].alias("num_specials"),

        #     # --- production ---
        #     F.col("infobox")["executive_producer"].alias("executive_producer"),
        #     F.col("infobox")["producer"].alias("producer"),
        #     F.col("infobox")["news_editor"].alias("news_editor"),
        #     F.col("infobox")["location"].alias("location"),
        #     F.col("infobox")["cinematography"].alias("cinematography"),
        #     F.col("infobox")["animator"].alias("animator"),
        #     F.col("infobox")["editor"].alias("editor"),
        #     F.col("infobox")["camera"].alias("camera"),
        #     F.col("infobox")["company"].alias("production_company"),
        #     F.col("infobox")["budget"].alias("budget"),

        #     # --- relations / links ---
        #     F.col("infobox")["list_episodes"].alias("list_episodes"),
        #     F.col("infobox")["related"].alias("related"),

        #     # --- description & categories ---
        #     F.col("description"),
        #     F.col("categories"),
        # )

        # logger.log(f"Extracted structured rows from chunk: {result.count()}")
        # return result

        return candidates

    def extract_all_dumps_and_save(self):
        whitelist = self.build_title_whitelist()

        dump_files = sorted(
            Path(DATA_DIR).glob("enwiki-latest-pages-articles-multistream*.xml-p*.bz2")
        )
        if not dump_files:
            logger.log(
                "No dump chunks found. Expected files like "
                "'enwiki-latest-pages-articles-multistreamXX.xml-pYYYYYpZZZZZ.bz2' in data/."
            )
            return

        logger.log(f"Found {len(dump_files)} dump chunk(s) to process.")
        logger.log(f"Already processed {len(self.processed_chunks)} chunk(s).")

        show_f, show_writer = self._open_tsv_writer(WIKI_SHOWS_TSV, SHOW_COLS)
        # ep_f,   ep_writer   = self._open_tsv_writer(WIKI_EPISODES_TSV, EPISODE_COLS)
        # act_f,  act_writer  = self._open_tsv_writer(WIKI_ACTORS_TSV, ACTOR_COLS)

        for dump_file in dump_files:
            dump_path = str(dump_file)

            if dump_path in self.processed_chunks:
                logger.log(f"Skipping already processed chunk: {dump_path}")
                continue

            logger.log(f"Starting chunk: {dump_path}")
            df_chunk = self.parse_wiki_dump_chunk(dump_path, whitelist)

            show_rows = episode_rows = actor_rows = 0

            for row in df_chunk.toLocalIterator():
                raw = row["raw_text"] or ""
                entity_type = row["entity_type"]
                start_year = row["start_year"]

                # categories, filter by keywords
                cats = extract_categories_py(raw)
                if self.category_match_fn is not None and not self.category_match_fn(cats):
                    continue

                info = extract_infobox_py(raw)
                if entity_type == "show" and info.get("_infobox_type", "").lower() != "television":
                    # skip character/books/etc. pages mis-matched on title
                    continue

                page_start_year = extract_start_year_from_infobox_py(info)
                show_year = row["start_year"]

                if entity_type == "show":
                    if page_start_year is not None and show_year is not None:
                        if abs(page_start_year - show_year) > 1:
                            # mismatch – this row is likely a wrong pairing
                            continue

                clean_txt = clean_text_py(raw)
                desc = extract_first_paragraph_py(clean_txt) or ""

                if not desc or len(desc) < 40:
                    desc = clean_txt[:2000]
                if len(desc) > 2000:
                    desc = desc[:2000]

                base = {
                    "page_id": row["page_id"],
                    "title": row["title"] or "",
                    "norm_title": row["norm_title"] or "",
                    "entity_type": entity_type,          # THIS is what was missing/wrong
                    "start_year": start_year,
                    "page_start_year": page_start_year,
                    "infobox_type": info.get("_infobox_type", ""),
                    "alt_title": info.get("alt_name", ""),
                    "native_title": info.get("native_name", ""),
                    "based_on": info.get("based_on", ""),
                    "inspired_by": info.get("inspired_by", ""),
                    "creator": info.get("creator", ""),
                    "developer": info.get("developer", ""),
                    "showrunner": info.get("showrunner", ""),
                    "writer": info.get("writer", ""),
                    "screenplay": info.get("screenplay", ""),
                    "teleplay": info.get("teleplay", ""),
                    "story": info.get("story", ""),
                    "director": info.get("director", ""),
                    "creative_director": info.get("creative_director", ""),
                    "presenter": info.get("presenter", ""),
                    "starring": info.get("starring", ""),
                    "judges": info.get("judges", ""),
                    "voices": info.get("voices", ""),
                    "narrator": info.get("narrator", ""),
                    "theme_music_composer": info.get("theme_music_composer", ""),
                    "open_theme": info.get("open_theme", ""),
                    "end_theme": info.get("end_theme", ""),
                    "composer": info.get("composer", ""),
                    "language": info.get("language", ""),
                    "num_specials": info.get("num_specials", ""),
                    "executive_producer": info.get("executive_producer", ""),
                    "producer": info.get("producer", ""),
                    "news_editor": info.get("news_editor", ""),
                    "location": info.get("location", ""),
                    "cinematography": info.get("cinematography", ""),
                    "animator": info.get("animator", ""),
                    "editor": info.get("editor", ""),
                    "camera": info.get("camera", ""),
                    "production_company": info.get("company", ""),
                    "budget": info.get("budget", ""),
                    "list_episodes": info.get("list_episodes", ""),
                    "related": info.get("related", ""),
                    "description": desc,           
                    "categories": cats or "",
                }

                if entity_type == "show":
                    self._write_row(show_writer, SHOW_COLS, base)
                    show_rows += 1
                # elif entity_type == "episode":
                #     data = {k: v for k, v in base.items() if k != "start_year"}
                #     self._write_row(ep_writer, EPISODE_COLS, data)
                #     episode_rows += 1
                # elif entity_type == "actor":
                #     data = {k: v for k, v in base.items() if k != "start_year"}
                #     self._write_row(act_writer, ACTOR_COLS, data)
                #     actor_rows += 1

            # # Shows keep start_year
            # df_shows = df_chunk.filter(F.col("entity_type") == "show")
            # logger.log(f"Shows in chunk: {df_shows.count()}")
            # logger.log(f"Show example rows:\n{df_shows.show(5, truncate=100)}")

            # # Episodes & actors DROP start_year (they don't have it conceptually)
            # df_episodes = (
            #     df_chunk
            #     .filter(F.col("entity_type") == "episode")
            #     .drop("start_year")
            # )
            # logger.log(f"Episodes in chunk: {df_episodes.count()}")
            # logger.log(f"Episode example rows:\n{df_episodes.show(5, truncate=100)}")

            # df_actors = (
            #     df_chunk
            #     .filter(F.col("entity_type") == "actor")
            #     .drop("start_year")
            # )
            # logger.log(f"Actors in chunk: {df_actors.count()}")
            # logger.log(f"Actor example rows:\n{df_actors.show(5, truncate=100)}")

            # show_rows    = self._append_chunk_to_tsv(df_shows,    WIKI_SHOWS_TSV)
            # episode_rows = self._append_chunk_to_tsv(df_episodes, WIKI_EPISODES_TSV)
            # actor_rows   = self._append_chunk_to_tsv(df_actors,   WIKI_ACTORS_TSV)

            self._append_chunk_mapping(dump_path, show_rows + episode_rows + actor_rows)

            logger.log(
                f"Finished chunk: {dump_path} "
                f"(shows={show_rows}, episodes={episode_rows}, actors={actor_rows})"
            )

        show_f.close()
        # ep_f.close()
        # act_f.close()

    def _open_tsv_writer(self, path: str, cols: list[str]):
        file_path = Path(path)
        _ensure_dir(file_path.parent)
        file_exists = file_path.exists()

        f = file_path.open("a", encoding="utf-8", newline="")
        writer = csv.writer(f, delimiter="\t")

        if not file_exists:
            writer.writerow(cols)

        return f, writer



    def _write_row(self, writer, cols, data: dict):
        row = []
        for c in cols:
            v = data.get(c, "")
            if v is None:
                v = ""
            s = str(v)
            if len(s) > 4000:  # optional safety truncation
                s = s[:4000]
            row.append(s)
        writer.writerow(row)

    def _append_chunk_to_tsv(self, df: DataFrame, path: str) -> int:
        """
        Append a chunk DataFrame to the main TSV (no Spark/Hadoop writers).
        Returns number of rows written.
        """
        _ensure_dir(Path(path).parent)

        if df.rdd.isEmpty():
            logger.log(f"No rows to append to {path}")
            return 0

        cols = df.columns
        df_str = df.select(
            *[
                F.substring(F.col(c).cast("string"), 1, 4000).alias(c)
                for c in cols
            ]
        )

        file_path = Path(path)
        file_exists = file_path.exists()

        row_count = 0

        with file_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="\t")

            if not file_exists:
                writer.writerow(cols)

            try:
                for r in df_str.toLocalIterator():
                    values = []
                    for c in cols:
                        v = r[c]
                        if v is None:
                            values.append("")
                        else:
                            # ensure it's a plain str and safe for utf-8
                            s = str(v)
                            # replace unencodable chars just in case
                            s = s.encode("utf-8", errors="replace").decode("utf-8")
                            values.append(s)

                    writer.writerow(values)
                    row_count += 1
            except Exception as e:
                logger.log(f"ERROR while streaming rows for {path}: {e}")
                raise

        logger.log(f"Appended {row_count} row(s) to {path}")
        return row_count

    def close(self):
        self.spark.stop()


if __name__ == "__main__":
    extractor = SparkWikiExtractor()
    try:
        extractor.extract_all_dumps_and_save()
    finally:
        extractor.close()
