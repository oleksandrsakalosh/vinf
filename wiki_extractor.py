from __future__ import annotations
from pathlib import Path
import os, sys, json, gzip, pickle, csv

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.sql import SparkSession, DataFrame, functions as F, types as T
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer, IDF, CountVectorizerModel, IDFModel

DATA_DIR = "data"
EXTRACTED_DIR = "extracted"
SPARK_DIR = "spark"

SHOWS_FILE    = f"{EXTRACTED_DIR}/extracted_shows.tsv"
SEASONS_FILE  = f"{EXTRACTED_DIR}/extracted_seasons.tsv"
EPISODES_FILE = f"{EXTRACTED_DIR}/extracted_episodes.tsv"
CREDITS_FILE  = f"{EXTRACTED_DIR}/extracted_credits.tsv"
ACTORS_FILE  = f"{EXTRACTED_DIR}/extracted_actors.tsv"

WIKI_DUMP = f"{DATA_DIR}/enwiki-latest-pages-articles-multistream18.xml-p23716198p25216197.bz2" # enwiki-latest-pages-articles.xml.bz2 // enwiki-latest-pages-articles-multistream18.xml-p23716198p25216197.bz2

from logger import WikiLogger
logger = WikiLogger()

def create_spark(app_name="ShowSearch") -> SparkSession:
    spark = (SparkSession.builder.appName(app_name)
        .master("local[12]") 
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.shuffle.partitions", "192")     
        .config("spark.sql.files.maxPartitionBytes", "134217728") 
        .config("spark.local.dir", "D:/spark_tmp")
        .getOrCreate()
    )
    return spark
    

class SparkWikiExtractor:

    def __init__(self):
        self.spark: SparkSession = create_spark()
        self.load_extracted_data()

    def load_extracted_data(self):
        logger.log("Loading extracted data files...")

        self.shows = (self.spark.read
                .option("header", True)
                .option("sep", "\t")
                .csv(SHOWS_FILE))

        self.seasons = (self.spark.read
                .option("header", True)
                .option("sep", "\t")
                .csv(SEASONS_FILE))

        self.episodes = (self.spark.read
                    .option("header", True)
                    .option("sep", "\t")
                    .csv(EPISODES_FILE))

        self.credits = (self.spark.read
                .option("header", True)
                .option("sep", "\t")
                .csv(CREDITS_FILE))
        
        self.actors = (self.spark.read
                .option("header", True)
                .option("sep", "\t")
                .csv(ACTORS_FILE))
        
        logger.log(f"Extracted data files loaded. Counts: shows={self.shows.count()}, seasons={self.seasons.count()}, episodes={self.episodes.count()}, credits={self.credits.count()}, actors={self.actors.count()}")
        
    def build_show_docs_df(self):
        logger.log("Building show documents DataFrame...")

        # Aggregate seasons into a text snippet per show_url
        seasons_agg = (self.seasons
            .groupBy("show_url")
            .agg(F.collect_list(
                F.concat_ws(" ",
                    F.lit("season"),
                    F.col("season_number"),
                    F.lit("has"),
                    F.col("episode_count"),
                    F.lit("episodes")
                )).alias("seasons_list"))
            .withColumn("seasons_text", F.concat_ws(" ", "seasons_list"))
            .select("show_url", "seasons_text"))
        
        logger.log(f"Seasons aggregated. Count: {seasons_agg.count()}")

        # Aggregate episodes into a text snippet per show_url
        episodes_agg = (self.episodes
            .groupBy("show_url")
            .agg(F.collect_list(
                F.concat_ws(" ",
                    F.lit("season"),
                    F.col("season_number"),
                    F.lit("episode"),
                    F.col("episode_number_in_season"),
                    F.coalesce(F.col("episode_title"), F.lit("")),
                    F.coalesce(F.col("airdate"), F.lit(""))
                )).alias("episodes_list"))
            .withColumn("episodes_text", F.concat_ws(" ", "episodes_list"))
            .select("show_url", "episodes_text"))
        
        logger.log(f"Episodes aggregated. Count: {episodes_agg.count()}")

        # Aggregate credits into a text snippet per show_url
        credits_agg = (self.credits
            .groupBy("show_url")
            .agg(F.collect_list(
                F.concat_ws(" ",
                    F.coalesce(F.col("actor_name"), F.lit("")),
                    F.coalesce(F.col("role"), F.lit(""))
                )).alias("credits_list"))
            .withColumn("credits_text", F.concat_ws(" ", "credits_list"))
            .select("show_url", "credits_text"))
        
        logger.log(f"Credits aggregated. Count: {credits_agg.count()}")
        logger.log("Joining all aggregated data to shows...")

        # Join all to shows
        show_docs = (self.shows
            .alias("s")
            .join(seasons_agg, self.shows.url == seasons_agg.show_url, "left")
            .join(episodes_agg, self.shows.url == episodes_agg.show_url, "left")
            .join(credits_agg, self.shows.url == credits_agg.show_url, "left")
            .select(
                F.monotonically_increasing_id().alias("doc_id"),
                F.col("s.url").alias("url"),
                F.col("s.title").alias("title"),
                F.col("s.genres"),
                F.col("s.description"),
                F.col("s.status"),
                F.col("s.network"),
                F.col("s.country"),
                F.col("s.runtime_minutes"),
                F.concat_ws(" ",
                    F.coalesce(F.col("date_start"), F.lit("")),
                    F.coalesce(F.col("date_end"), F.lit(""))
                ).alias("dates"),
                F.coalesce(F.col("seasons_text"), F.lit("")).alias("seasons_text"),
                F.coalesce(F.col("episodes_text"), F.lit("")).alias("episodes_text"),
                F.coalesce(F.col("credits_text"), F.lit("")).alias("credits_text"),
            ))
        
        show_docs = show_docs.withColumn("norm_title", norm_title(F.col("title")))

        # Build one big text field per show (weighted later in TF, if you want)
        show_docs = show_docs.withColumn(
            "full_text",
            F.concat_ws(" ",
                F.coalesce(F.col("title"), F.lit("")),
                F.coalesce(F.col("genres"), F.lit("")),
                F.coalesce(F.col("description"), F.lit("")),
                F.coalesce(F.col("status"), F.lit("")),
                F.coalesce(F.col("network"), F.lit("")),
                F.coalesce(F.col("country"), F.lit("")),
                F.coalesce(F.col("runtime_minutes"), F.lit("")),
                F.coalesce(F.col("dates"), F.lit("")),
                F.coalesce(F.col("seasons_text"), F.lit("")),
                F.coalesce(F.col("episodes_text"), F.lit("")),
                F.coalesce(F.col("credits_text"), F.lit("")),
            )
        )

        logger.log(f"Show documents DataFrame built. Count: {show_docs.count()}")

        return show_docs

    def parse_wiki_dump(self, show_whitelist: F.DataFrame) -> F.DataFrame:
        logger.log("Parsing Wikipedia dump and extracting relevant pages...")

        shows_nt = F.broadcast(show_whitelist)


        wiki = (self.spark.read.format("com.databricks.spark.xml")
            .option("rowTag", "page")
            .load(WIKI_DUMP)
            .select(
                F.col("id").cast("long").alias("page_id"),
                F.col("title").cast("string").alias("title"),
                F.col("revision.text._VALUE").cast("string").alias("raw_text"),
            )
           )
        
        logger.log(f"Loaded Wikipedia pages.")

        titles = (wiki
            .select("page_id", "title",norm_title(F.col("title")).alias("norm_title")))
        
        candidate_ids = (titles.join(shows_nt, on="norm_title", how="left_semi")
            .select("page_id").distinct())
        
        direct = (wiki.join(candidate_ids, on="page_id", how="inner")
            .select("page_id", "title", "raw_text"))

        logger.log(f"Direct title matches found: {direct.count()}")

        clean = F.col("raw_text")
        clean = F.regexp_replace(clean, r"\{\{[^}{]*\}\}", " ")
        clean = F.regexp_replace(clean, r"\[\[(File|Image):[^\]]+\]\]", " ")
        clean = F.regexp_replace(clean, r"\[\[([^|\]]+\|)?([^\]]+)\]\]", r"\2")
        clean = F.regexp_replace(clean, r"https?://\S+", " ")
        clean = F.regexp_replace(clean, r"={2,}\s*(.*?)\s*={2,}", r" \1 ")
        clean = F.regexp_replace(clean, r"<[^>]+>", " ")
        clean = F.regexp_replace(clean, r"\s{2,}", " ")

        wiki_cleaned = direct.withColumn("clean_text", F.trim(clean)).select("page_id", "title", "clean_text")

        logger.log("Wikipedia pages cleaned.")

        selected = wiki_cleaned.select("page_id", "title", "clean_text")

        return selected  # columns: page_id, title, clean_text
    
    def build_corpus(self):
        logger.log("Building combined corpus of show documents and Wikipedia articles...")

        show_docs = self.build_show_docs_df().select(
            F.col("doc_id"),
            F.col("url"),
            F.col("title"),
            F.col("norm_title"),
            F.col("full_text")
        )

        show_whitelist = show_docs.select("norm_title").distinct()

        wiki_docs = self.parse_wiki_dump(show_whitelist).select(
            F.col("norm_title"),
            F.col("clean_text")
        )

        wiki_best = (wiki_docs
            .groupBy("norm_title")
            .agg(F.concat_ws(" ", F.collect_list("clean_text")).alias("wiki_text"))
        )

        self.corpus = (show_docs
            .join(wiki_best, on="norm_title", how="left")
            .withColumn("wiki_text", F.coalesce(F.col("wiki_text"), F.lit("")))
            .withColumn("text", F.concat_ws(" ", F.col("full_text"), F.col("wiki_text")))
            .select("doc_id", "url", "title", "text"))
        
        logger.log(f"Combined corpus built. Total documents: {self.corpus.count()}")

    def close(self):
        self.spark.stop()

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def norm_title(col: str | None) -> str:
    c = F.lower(F.trim(col))
    c = F.regexp_replace(c, r"\s*\((american|british|canadian|australian|indian|south korean|japanese)\s+(tv|television)\s+series\)\s*$", "")
    c = F.regexp_replace(c, r"\s*\(\d{4}\s+tv\s+series\)\s*$", "")
    c = F.regexp_replace(c, r"\s*\(\d{4}.*?\)\s*$", "")
    c = F.regexp_replace(c, r"[-–—_:;/\.,!?]+", " ")
    c = F.regexp_replace(c, r"\s+", " ")
    return F.trim(c)

if __name__ == "__main__":
    extractor = SparkWikiExtractor()
    extractor.build_corpus()
    extractor.close()