from __future__ import annotations
from pathlib import Path
import os, sys, csv, re, math
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

WIKI_CATEGORIES_FILE   = f"{EXTRACTED_DIR}/television_categories.txt"
WIKI_CHUNK_MAP_TSV     = f"{EXTRACTED_DIR}/wiki_chunk_progress.tsv"

JOINED_OUTPUT_TSV    = f"{EXTRACTED_DIR}/joined_shows_wiki.tsv"
JOIN_WRITE_CHUNK_SIZE = 2000

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

medium_disambig = (
    r"(?i)\s*\("
    r"(?:(\d{4})\s*)?" 
    r"(?:(american|british|canadian|australian|indian|south korean|japanese)\s+)?" 
    r"(?:"
    r"(?:tv|television|web|streaming)\s+series"
    r"|series"
    r"|miniseries"
    r"|anime"
    r"|cartoon"
    r"|animated\s+series?"
    r"|drama\s+series?"
    r"|sitcom"
    r"|soap\s+opera"
    r"|telenovela"
    r"|game\s+show"
    r"|quiz\s+show"
    r"|reality\s+(?:show|series)"
    r"|talk\s+show"
    r"|children's\s+series"
    r"|documentary(?:\s+series)?"
    r")"
    r"\)\s*$"
)


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
    c = F.regexp_replace(c, medium_disambig, "")
    c = F.regexp_replace(c, r"\s*\(\d{4}.*?\)\s*$", "")
    c = F.regexp_replace(c, r"[-–—_:;/\.,!?]+", " ")
    c = F.regexp_replace(c, r"\s+", " ")
    return F.trim(c)

def has_television_infobox(raw_text: str) -> bool:
            if not raw_text:
                return False
            m = re.search(r"\{\{\s*infobox\s+television\b", raw_text, flags=re.IGNORECASE)
            return m is not None
has_television_infobox_udf = F.udf(has_television_infobox, T.BooleanType())


def _clean_wiki_value(value: str) -> str:
    """
    Clean a single infobox value
    """
    if not value:
        return ""

    v = value

    # --- 0) Special-case 'Collapsible list' and similar list templates ---
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

    # --- 3) Drop leftover '{{' and '}}' ---
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

    parts = re.split(r"\n\s*\n", text)
    first_block = ""
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if len(p) < 40:
            continue
        first_block = p
        break

    if not first_block:
        first_block = text

    sentences = re.split(r'(?<=[.!?])\s+', first_block)
    if sentences:
        candidate = " ".join(sentences[:3]).strip()
    else:
        candidate = first_block

    if not candidate:
        candidate = first_block

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
        logger.log("Loading extracted TSV files.")

        shows_raw = (
            self.spark.read
            .option("header", True)
            .option("sep", "\t")
            .csv(SHOWS_FILE)
        )

        self.shows = (
            shows_raw
            .withColumn(
                "norm_title",
                norm_title(F.col("title"))
            )
            .withColumn(
                "start_year",
                F.substring("date_start_iso", 1, 4).cast("int")
            )
            .distinct()
        )

        logger.log(
            f"Loaded TSVs: shows={self.shows.count()}, "
        )

    def parse_wiki_dump_chunk(self, dump_path: str) -> DataFrame:
        logger.log(f"Parsing Wikipedia dump chunk: {dump_path}")

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

        candidates = (
            wiki
            .filter(~F.lower(F.col("raw_text")).startswith("#redirect"))
            .select(
                "page_id",
                "title",
                "norm_title",
                "raw_text",
            )
        )

        candidates = candidates.filter(has_television_infobox_udf(F.col("raw_text")))

        logger.log(
            f"Normalized titles, filtered redirects, and filtered by television infobox. "
            f"Candidate pages in chunk after filter: {candidates.count()}"
        )

        return candidates

    def extract_all_dumps_and_save(self):

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

        for dump_file in dump_files:
            dump_path = str(dump_file)

            if dump_path in self.processed_chunks:
                logger.log(f"Skipping already processed chunk: {dump_path}")
                continue

            logger.log(f"Starting chunk: {dump_path}")
            df_chunk = self.parse_wiki_dump_chunk(dump_path)

            show_rows = 0

            for row in df_chunk.toLocalIterator():
                raw = row["raw_text"] or ""

                # categories, filter by allowed categories
                cats = extract_categories_py(raw)
                if self.category_match_fn is not None and not self.category_match_fn(cats):
                    continue

                info = extract_infobox_py(raw)

                page_start_year = extract_start_year_from_infobox_py(info)

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
                    "page_start_year": page_start_year,
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

                self._write_row(show_writer, SHOW_COLS, base)
                show_rows += 1
                
            self._append_chunk_mapping(dump_path, show_rows)

            logger.log(
                f"Finished chunk: {dump_path} "
                f"Found television infobox rows: {show_rows}"
            )

        show_f.close()

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
            if len(s) > 4000:
                s = s[:4000]
            row.append(s)
        writer.writerow(row)

    def _append_chunk_to_tsv(
        self,
        df: DataFrame,
        path: str,
        *,
        chunk_size: int = JOIN_WRITE_CHUNK_SIZE,
        known_count: int | None = None,
    ) -> int:
        """
        Append a chunk DataFrame to the main TSV (no Spark/Hadoop writers).
        Returns number of rows written.
        """
        _ensure_dir(Path(path).parent)

        total_rows = known_count if known_count is not None else df.count()
        if total_rows == 0:
            logger.log(f"No rows to append to {path}")
            return 0

        safe_chunk = max(1, chunk_size)
        partitions = max(1, math.ceil(total_rows / safe_chunk))
        logger.log(
            f"Preparing to append {total_rows} row(s) to {path} "
            f"in approximately {partitions} partition(s) "
            f"(~{safe_chunk} rows per chunk)."
        )

        cols = df.columns
        df_str = df.repartition(partitions).select(
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
    
    def join_wiki_with_extracted(self):
        logger.log("Joining extracted shows with Wikipedia data...")

        wiki_shows = (
            self.spark.read
            .option("header", True)
            .option("sep", "\t")
            .csv(WIKI_SHOWS_TSV)
        )

        joined = (
            self.shows.alias("ext")
            .join(
                wiki_shows.alias("wiki"),
                (F.col("ext.norm_title") == F.col("wiki.norm_title")) &
                (F.col("ext.start_year") == F.col("wiki.page_start_year")),
                how="full_outer",
            )
            .select(
                F.col("ext.*"),
                F.col("wiki.page_id").alias("wiki_page_id"),
                F.col("wiki.title").alias("wiki_title"),
                F.col("wiki.norm_title").alias("wiki_norm_title"),
                F.col("wiki.page_start_year").alias("wiki_page_start_year"),
                F.col("wiki.alt_title").alias("wiki_alt_title"),
                F.col("wiki.native_title").alias("wiki_native_title"),
                F.col("wiki.based_on").alias("wiki_based_on"),
                F.col("wiki.inspired_by").alias("wiki_inspired_by"),
                F.col("wiki.creator").alias("wiki_creator"),
                F.col("wiki.developer").alias("wiki_developer"),
                F.col("wiki.showrunner").alias("wiki_showrunner"),
                F.col("wiki.writer").alias("wiki_writer"),
                F.col("wiki.screenplay").alias("wiki_screenplay"),
                F.col("wiki.teleplay").alias("wiki_teleplay"),
                F.col("wiki.story").alias("wiki_story"),
                F.col("wiki.director").alias("wiki_director"),
                F.col("wiki.creative_director").alias("wiki_creative_director"),
                F.col("wiki.presenter").alias("wiki_presenter"),
                F.col("wiki.starring").alias("wiki_starring"),
                F.col("wiki.judges").alias("wiki_judges"),
                F.col("wiki.voices").alias("wiki_voices"),
                F.col("wiki.narrator").alias("wiki_narrator"),
                F.col("wiki.theme_music_composer").alias("wiki_theme_music_composer"),
                F.col("wiki.open_theme").alias("wiki_open_theme"),
                F.col("wiki.end_theme").alias("wiki_end_theme"),
                F.col("wiki.composer").alias("wiki_composer"),
                F.col("wiki.language").alias("wiki_language"),
                F.col("wiki.num_specials").alias("wiki_num_specials"),
                F.col("wiki.executive_producer").alias("wiki_executive_producer"),
                F.col("wiki.producer").alias("wiki_producer"),
                F.col("wiki.news_editor").alias("wiki_news_editor"),
                F.col("wiki.location").alias("wiki_location"),
                F.col("wiki.cinematography").alias("wiki_cinematography"),
                F.col("wiki.animator").alias("wiki_animator"),
                F.col("wiki.editor").alias("wiki_editor"),
                F.col("wiki.camera").alias("wiki_camera"),
                F.col("wiki.production_company").alias("wiki_production_company"),
                F.col("wiki.budget").alias("wiki_budget"),
                F.col("wiki.list_episodes").alias("wiki_list_episodes"),
                F.col("wiki.related").alias("wiki_related"),
                F.col("wiki.description").alias("wiki_description"),
                F.col("wiki.categories").alias("wiki_categories"),
            )
        )

        count = joined.count()
        logger.log(f"Joined shows count: {count}")

        self._append_chunk_to_tsv(
            joined,
            JOINED_OUTPUT_TSV,
            known_count=count,
        )

    def close(self):
        self.spark.stop()


if __name__ == "__main__":
    extractor = SparkWikiExtractor()
    try:
        extractor.extract_all_dumps_and_save()
        # load_extracted_data() 
        # make join on title + start year
        extractor.load_extracted_data()
        extractor.join_wiki_with_extracted()

    finally:
        extractor.close()
