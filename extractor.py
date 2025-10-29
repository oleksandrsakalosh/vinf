import sys
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Iterable
import re
import csv
from pathlib import Path
import html

from logger import ExtractorLogger

logger = ExtractorLogger()
LOGS_DIR = "logs"

FLAGS = re.DOTALL | re.IGNORECASE

patterns: Dict[str, Optional[re.Pattern]] = {
    # Metadata
    "keywords": re.compile( # <meta name="keywords" content="Comedy, Game/Quiz, television, series, show, episode guide">
        r'<meta\s+name="keywords"\s+content="(?P<keywords>[^"]+)"\s*/?>',
        FLAGS
    ),

    # Page-level
    "title": re.compile(
        r'<div\s+class="center\s+titleblock".*?>\s*'
        r'<h2>\s*<a[^>]*>(?P<title>[^<]+)</a>\s*</h2>',
        FLAGS
    ),

    "start_end_dates": re.compile(
        r'Start\s+date:\s*(?:<a[^>]*>)?(?P<date_start>[^<]+?)(?:</a>)?\s*<br\s*/?>\s*'
        r'End\s+date:\s*(?P<date_end>[^<]+?)\s*<br',
        FLAGS
    ),
    "status": re.compile(
        r'Status:\s*(?:<a[^>]*>)?(?P<status>[^<]+?)(?:</a>)?\s*<br',
        FLAGS
    ),
    "network_country": re.compile(
        r'Network\(s\):\s*<a[^>]*>(?P<network>[^<]+)</a>\s*'
        r'\(\s*<a[^>]*>(?P<country>[^<]+)</a>\s*\)\s*<br',
        FLAGS
    ),
    "runtime": re.compile(
        r'Run\s*time:\s*(?P<runtime>\d+)\s*min',
        FLAGS
    ),
    "episode_count": re.compile(
        r'Episodes:\s*(?P<episodes>[^<]+?)\s*<br',
        FLAGS
    ),
    "genres": re.compile(
        r'(?:<a[^>]*>)?Genre\(s\)(?:</a>)?\s*:\s*(?P<genres>[^<]+)\s*<br\s*/?>',
        re.IGNORECASE
    ),
    "cast_block": re.compile(
        r'<div\s+id="credits"[^>]*>\s*(?P<block>.*?)\s*</div>',
        FLAGS
    ),
    "cast_row": re.compile(
        r'<li(?:(?!class="lihd").)*?>\s*'
        r'(?:<a[^>]*>)?(?P<name>[^<]+?)(?:</a>)?\s*'
        r'\s+as\s+'
        r'(?P<role>[^<\[]+?)'
        r'(?:\s*\[[^\]]*\])?'     
        r'\s*</li>',
        FLAGS
    ),
    "description": re.compile(
        r'<div\s+id="blurb"[^>]*>\s*(?P<description>.*?)\s*(?:<br\s*/?>)?\s*</div>',
        FLAGS
    ),

    # Episodes
    "episode_block": re.compile(
        r'<div\s+id="eplist"[^>]*>\s*(?P<eplist>.*?</table>)\s*</div>',
        FLAGS
    ),
    "episode_row": re.compile(
        r"<tr>\s*"
        r"<td\s+class='epinfo\s+right'>\s*(?P<overall>\d+)\.\s*</td>\s*"
        r"<td\s+class='epinfo\s+left\s+pad'>\s*(?P<season_ep>\d+\s*-\s*\d+)\s*(?:&nbsp;)?\s*</td>\s*"
        r"<td\s+class='epinfo\s+right\s+pad'>\s*(?P<airdate>[^<]+?)\s*</td>\s*"
        r"<td\s+class='eptitle\s+left'>\s*(?:<a[^>]*>)?(?P<title>[^<]+)</a?>\s*</td>\s*"
        r"</tr>",
        FLAGS
    ),
    "overall_episode_number": re.compile(
        r'^\s*(?P<overall>\d+)\.\s*$',
        re.IGNORECASE
    ),
    "season_episode_number": re.compile(
        r'^\s*(?P<season>\d+)\s*-\s*(?P<ep>\d+)\s*',
        re.IGNORECASE
    ),
    "episode_airdate": re.compile(
        r'^\s*(?P<airdate>\d{1,2}\s+[A-Za-z]{3}\s+\d{2,4})\s*$',
        re.IGNORECASE
    ),
    "episode_title": re.compile(
        r'^\s*(?P<title>.+?)\s*$',
        re.IGNORECASE
    ),
}

@dataclass
class ShowInfo:
    keywords: Optional[str] = None
    title: Optional[str] = None
    date_start: Optional[str] = None
    date_start_iso: Optional[str] = None
    date_end: Optional[str] = None
    date_end_iso: Optional[str] = None
    status: Optional[str] = None
    network: Optional[str] = None
    country: Optional[str] = None
    runtime_minutes: Optional[int] = None
    episode_count: Optional[int] = None
    genres: Optional[List[str]] = field(default_factory=list)
    cast: Optional[Dict[str, List[str]]] = field(default_factory=dict)
    description: Optional[str] = None
    season_count: Optional[int] = None

@dataclass
class EpisodeInfo:
    season: Optional[int] = None
    title: Optional[str] = None
    airdate: Optional[str] = None
    airdate_iso: Optional[str] = None

@dataclass
class SeasonInfo:
    episodes: Optional[Dict[int, EpisodeInfo]] = field(default_factory=dict)

@dataclass
class ShowExtract:
    url: str
    show: ShowInfo = field(default_factory=ShowInfo)
    seasons: Optional[Dict[int, SeasonInfo]] = field(default_factory=dict)
    specials: Optional[SeasonInfo] = None

class Extractor:
    '''
    A class to handle extraction of series data from saved html pages.
    Reads lies in url_mapping.tsv one by one and extracts relevant information.
    Saves extracted data to structured file.
    '''
    def __init__(self):
        self.unique_actors = set()

    def run(self):
        for row in self._read_tsv():
            if not row:
                continue
            url = row["url"]
            path = row["path"]
            try:
                with open(path, "r", encoding="utf-8") as f:
                    html = f.read()
                show = self.extract_one(url, html)
                if show:
                    show.url = url
                    self._save_extracted_data(show)
                    self._mark_extracted(url, path, "success")
                    logger.log_extraction(url, "success")
                else:
                    self._mark_extracted(url, path, "no_data")
                    logger.log_warning(f"No data extracted for URL: {url}")
            except Exception as e:
                self._mark_extracted(url, path, "error")
                logger.log_error(url, e)

    def extract_one(self, url: str, text: str) -> Optional[ShowExtract]:
        show = ShowInfo()

        def g(m: re.Match[str] | None, name, default=None):
            if not m: return default
            if name in m.re.groupindex:
                v = m.group(name)
            elif m.groups():
                v = m.group(1)
            else:
                v = m.group(0)
            return clean(v)
        
        clean = lambda s: (None if s is None else re.sub(r"\s{2,}", " ", html.unescape(s).strip()))

        # Title
        show.title = g(patterns.get("title", None).search(text), "title")

        if not show.title:
            return None
        
        # Keywords (from meta tag)
        show.keywords = g(patterns.get("keywords").search(text), "keywords")

        # Dates (Start/End)
        date_m = patterns.get("start_end_dates")

        dm = date_m.search(text)
        ds = g(dm, "date_start")
        de = g(dm, "date_end")
        show.date_start = ds
        show.date_start_iso = self._date_to_iso(ds) if ds else None
        if de == "___ ____":
            de = None
            show.date_end_iso = None
        else:
            show.date_end = de
            show.date_end_iso = self._date_to_iso(de) if de else None

        # Status
        st_m = patterns.get("status")
        
        show.status = g(st_m.search(text), "status")

        # Network / Country
        nc_m = patterns.get("network_country")
        
        nm = nc_m.search(text)
        show.network = g(nm, "network")
        show.country = g(nm, "country")

        # Runtime (minutes)
        rt_m = patterns.get("runtime")
        
        rm = rt_m.search(text)
        rt = g(rm, "runtime")
        if rt and re.search(r"\d+", rt):
            show.runtime_minutes = int(re.search(r"\d+", rt).group(0))

        # Episode count (raw + try to parse leading int)
        ec_m = patterns.get("episode_count")
        
        em = ec_m.search(text)
        ec = g(em, "episodes")
        if ec:
            m = re.search(r"\d+", ec)
            show.episode_count = int(m.group(0)) if m else None

        # Genres (comma-separated)
        gn_m = patterns.get("genres")
        
        gm = gn_m.search(text)
        raw = g(gm, "genres")
        if raw:
            parts = [clean(p) for p in re.split(r"[,/|;]+", raw) if clean(p)]
            show.genres = parts

        # Description 
        desc_p = patterns.get("description")
        
        dm = desc_p.search(text)
        d = g(dm, "description")
        if d:
            # remove tags, collapse whitespace
            d = re.sub(r"<[^>]+>", " ", d)
            d = re.sub(r"\s{2,}", " ", d).strip()
            show.description = d or None

        # Cast (grouped by name -> role list)
        cast: dict[str, list[str]] = {}
        cb_pat = patterns.get("cast_block")
        cr_pat = patterns.get("cast_row")
        
        cbm = cb_pat.search(text)
        block = g(cbm, "block")
        if block:
            for m in cr_pat.finditer(block):
                name = g(m, "name")
                role = g(m, "role") or "cast"
                if not name:
                    continue
                key = name.strip().lower()
                if key not in cast:
                    cast[key] = []
                roles = [part.strip() for part in role.split(';') if part.strip()]
                for r in roles:
                    if r not in cast[key]:
                        cast[key].append(r)

        show.cast = cast

        # Episodes, seasons, specials 
        seasons: dict[int, SeasonInfo] = {}
        specials: Optional[SeasonInfo] = None
        seasons_count = 0

        ep_scope = text
        eb_pat = patterns.get("episode_block")
        
        ebm = eb_pat.search(text)
        if ebm:
            ep_scope = ebm.group(0)

        row_pat = patterns.get("episode_row")
        over_pat = patterns.get("overall_episode_number")
        ses_pat = patterns.get("season_episode_number")
        air_pat = patterns.get("episode_airdate")
        ttl_pat = patterns.get("episode_title")

        for row in row_pat.finditer(ep_scope):
            # Raw captures from row
            overall_raw = g(row, "overall")
            season_ep_raw = g(row, "season_ep")
            air_raw = g(row, "airdate")
            title_raw = g(row, "title")

            # Per-field refinement (optional patterns)
            if overall_raw:
                mm = over_pat.search(overall_raw)
                overall_raw = g(mm, "overall", overall_raw)

            season_no, ep_no = None, None
            if season_ep_raw:
                mm = ses_pat.search(season_ep_raw)
                if mm:
                    s = g(mm, "season")
                    e = g(mm, "ep")
                    season_no = int(s) if s and s.isdigit() else None
                    ep_no = int(e) if e and e.isdigit() else None
            # Fallback parse "1-10"
            if season_no is None and season_ep_raw:
                se_m = re.search(r"(\d+)\s*-\s*(\d+)", season_ep_raw)
                if se_m:
                    season_no = int(se_m.group(1))
                    ep_no = int(se_m.group(2))

            # Airdate refine + normalize
            if air_raw:
                mm = air_pat.search(air_raw)
                air_raw = g(mm, "airdate", air_raw)
            air_iso = air_raw if air_raw else None

            if title_raw:
                mm = ttl_pat.search(title_raw)
                title_raw = g(mm, "title", title_raw)

            # Build episode info
            epi = EpisodeInfo(
                season=season_no,
                title=title_raw,
                airdate=air_raw,
                airdate_iso=air_iso
            )

            # Route into seasons / specials
            if season_no == 0:
                if specials is None:
                    specials = SeasonInfo()
                # If episode number unknown, place at len+1; else keyed by ep number
                key = ep_no if ep_no is not None else (len(specials.episodes) + 1)
                specials.episodes[key] = epi
            else:
                if season_no is None:
                    # If truly unknown season, consider treating as season 1 fallback
                    season_no = 1
                if season_no not in seasons:
                    seasons[season_no] = SeasonInfo()
                    seasons_count += 1
                key = ep_no if ep_no is not None else (len(seasons[season_no].episodes) + 1)
                seasons[season_no].episodes[key] = epi

        show.season_count = seasons_count

        return ShowExtract(
            url=url,
            show=show,
            seasons=seasons,
            specials=specials
        )

    
    def _extract_title(self, html: str) -> Optional[str]:
        match = patterns["title"].search(html)
        if match:
            return match.group(1).strip()
        return None
    
    def _read_tsv(self) -> Iterable[Dict[str, str]]:
        path = Path(LOGS_DIR) / "url_mapping.tsv"
        if not path.exists():
            logger.log_warning(f"URL mapping file not found: {path}")
            return []
        with path.open("r", encoding="utf-8", newline="") as f:
            r = csv.DictReader(f, delimiter="\t")
            for row in r:
                yield row

    def _mark_extracted(self, url: str, path: str, status: str):
        extracted_fp = Path(LOGS_DIR) / "extracted_mapping.tsv"
        write_header = not extracted_fp.exists()
        with extracted_fp.open("a", encoding="utf-8", newline='') as f:
            writer = csv.writer(f, delimiter="\t")
            if write_header:
                writer.writerow(["url", "path", "status"])
            writer.writerow([url, path, status])

    def _save_extracted_data(self, data: ShowExtract):
        shows_fp = Path(LOGS_DIR) / "extracted_shows.tsv"
        write_header = not shows_fp.exists()
        with shows_fp.open("a", encoding="utf-8", newline='') as f:
            writer = csv.writer(f, delimiter="\t")
            if write_header:
                writer.writerow([
                    "url", "keywords", "title", "date_start", "date_start_iso",
                    "date_end", "date_end_iso", "status", "network", "country",
                    "runtime_minutes", "episode_count", "genres", "description",
                    "season_count"
                ])
            writer.writerow([
                data.url,
                data.show.keywords,
                data.show.title,
                data.show.date_start,
                data.show.date_start_iso,
                data.show.date_end,
                data.show.date_end_iso,
                data.show.status,
                data.show.network,
                data.show.country,
                data.show.runtime_minutes,
                data.show.episode_count,
                ",".join(data.show.genres) if data.show.genres else "",
                data.show.description,
                data.show.season_count
            ])
            
        actors_fp = Path(LOGS_DIR) / "extracted_actors.tsv"
        write_header_actors = not actors_fp.exists()
        with actors_fp.open("a", encoding="utf-8", newline='') as f:
            writer = csv.writer(f, delimiter="\t")
            if write_header_actors:
                writer.writerow(["actor_name"])
            for actor, _ in data.show.cast.items():
                if actor not in self.unique_actors:
                    writer.writerow([actor])
                    self.unique_actors.add(actor)
            
        credits_fp = Path(LOGS_DIR) / "extracted_credits.tsv"
        write_header_credits = not credits_fp.exists()
        with credits_fp.open("a", encoding="utf-8", newline='') as f:
            writer = csv.writer(f, delimiter="\t")
            if write_header_credits:
                writer.writerow(["show_url", "actor_name", "role"])
            for actor, roles in data.show.cast.items():
                for role in roles:
                    writer.writerow([data.url, actor, role])

        seasons_fp = Path(LOGS_DIR) / "extracted_seasons.tsv"
        write_header_seasons = not seasons_fp.exists()
        with seasons_fp.open("a", encoding="utf-8", newline='') as f:
            writer = csv.writer(f, delimiter="\t")
            if write_header_seasons:
                writer.writerow(["show_url", "season_number", "episode_count"])
            for season_num, season_info in data.seasons.items():
                writer.writerow([data.url, season_num, len(season_info.episodes)])

        episodes_fp = Path(LOGS_DIR) / "extracted_episodes.tsv"
        write_header_episodes = not episodes_fp.exists()
        with episodes_fp.open("a", encoding="utf-8", newline='') as f:
            writer = csv.writer(f, delimiter="\t")
            if write_header_episodes:
                writer.writerow([
                    "show_url", "season_number", "episode_number_in_season",
                    "episode_number_overall", "episode_title", "airdate",
                    "airdate_iso"
                ])
            number_overall = 1
            for season_num, season_info in data.seasons.items():
                for episode_num, episode_info in season_info.episodes.items():
                    writer.writerow([
                        data.url,
                        season_num,
                        episode_num,
                        number_overall,
                        episode_info.title,
                        episode_info.airdate,
                        episode_info.airdate_iso
                    ])
                    number_overall += 1
            if data.specials:
                for episode_num, episode_info in data.specials.episodes.items():
                    writer.writerow([
                        data.url,
                        0,
                        episode_num,
                        number_overall,
                        episode_info.title,
                        episode_info.airdate,
                        episode_info.airdate_iso
                    ])
                    number_overall += 1

    def _date_to_iso(self, date_str: str) -> Optional[str]:
        '''
        Convert date string like "12 Jan 20" or "Jan 20" to ISO format "2020-01-12".
        Returns None if input is invalid or cannot be parsed.
        '''
        if not date_str:
            return None
        date_str = date_str.strip()
        # Try full date first
        m = re.match(r"(\d{1,2})\s+([A-Za-z]{3,4})\s+(\d{2,4})", date_str)
        if m:
            day = int(m.group(1))
            month_str = m.group(2)
            year = int(m.group(3))
            if year < 30:
                year += 2000
            elif year < 100:
                year += 1900
            month = self._month_str_to_int(month_str)
            if month:
                return f"{year:04d}-{month:02d}-{day:02d}"
        # Try month-year only
        m = re.match(r"([A-Za-z]{3,4})\s+(\d{2,4})", date_str)
        if m:
            month_str = m.group(1)
            year = int(m.group(2))
            if year < 30:
                year += 2000
            elif year < 100:
                year += 1900
            month = self._month_str_to_int(month_str)
            if month:
                return f"{year:04d}-{month:02d}-01"
            
        return None
    
    def _month_str_to_int(self, month_str: str) -> Optional[int]:
        month_str = month_str.lower()
        months = {
            "jan": 1, "feb": 2, "mar": 3, "apr": 4, 
            "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7, "aug": 8, 
            "sep": 9, "oct": 10, "nov": 11, "dec": 12
        }
        return months.get(month_str, None)
    
    def validate_extraction(self):
        '''
        Basic validation of extracted data.
        Checks fields in extracted data from files and logs inconsistencies, missing data, etc.
        '''
        shows = set()

        shows_fp = Path(LOGS_DIR) / "extracted_shows.tsv"
        if not shows_fp.exists():
            logger.log_warning(f"Extracted shows file not found: {shows_fp}")
            return
        with shows_fp.open("r", encoding="utf-8", newline='') as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                url = row.get("url", "")
                shows.add(url)
                keywords = row.get("keywords", "")
                title = row.get("title", "")
                date_start = row.get("date_start", "")
                date_start_iso = row.get("date_start_iso", "")
                date_end = row.get("date_end", "")
                date_end_iso = row.get("date_end_iso", "")
                runtime = row.get("runtime_minutes", "")
                status = row.get("status", "")
                network = row.get("network", "")
                country = row.get("country", "")
                episode_count = row.get("episode_count", "")
                genres = row.get("genres", "")
                season_count = row.get("season_count", "")

                if not keywords:
                    logger.log_warning(f"Missing keywords for URL: {url}")
                if not title:
                    logger.log_warning(f"Missing title for URL: {url}")
                if not date_start or not date_start_iso:
                    logger.log_warning(f"Missing or invalid start date for URL: {url}")
                if (not date_end or not date_end_iso) and status and status.lower() in {"ended", "canceled", "cancelled/ended"}:
                    logger.log_warning(f"Missing or invalid end date for URL: {url}")
                if not runtime.isdigit():
                    logger.log_warning(f"Invalid runtime for URL: {url}")
                if not status:
                    logger.log_warning(f"Missing status for URL: {url}")
                if status and status.lower() not in {"current show", "ended", "on hiatus", "canceled", "cancelled/ended"}:
                    logger.log_warning(f"Unrecognized status '{status}' for URL: {url}")
                if not network:
                    logger.log_warning(f"Missing network for URL: {url}")
                if not country:
                    logger.log_warning(f"Missing country for URL: {url}")
                if not episode_count:
                    logger.log_warning(f"Missing episode count for URL: {url}")
                if not episode_count.isdigit():
                    logger.log_warning(f"Invalid episode count for URL: {url}")
                if not genres:
                    logger.log_warning(f"Missing genres for URL: {url}")
                if not season_count:
                    logger.log_warning(f"Missing season count for URL: {url}")
                if not season_count.isdigit():
                    logger.log_warning(f"Invalid season count for URL: {url}")

        seasons_fp = Path(LOGS_DIR) / "extracted_seasons.tsv"
        season_shows = set()
        if not seasons_fp.exists():
            logger.log_warning(f"Extracted seasons file not found: {seasons_fp}")
            return
        with seasons_fp.open("r", encoding="utf-8", newline='') as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                url = row.get("show_url", "")
                season_shows.add(url)
                season_number = row.get("season_number", "")
                episode_count = row.get("episode_count", "")

                if url not in shows:
                    logger.log_warning(f"Show URL in shows.tsv not found in seasons.tsv: {url}")
                if not season_number.isdigit():
                    logger.log_warning(f"Invalid season number for URL: {url}")
                if not episode_count.isdigit():
                    logger.log_warning(f"Invalid episode count for URL: {url}")

        fp_episodes = Path(LOGS_DIR) / "extracted_episodes.tsv"
        episode_shows = set()
        if not fp_episodes.exists():
            logger.log_warning(f"Extracted episodes file not found: {fp_episodes}")
            return
        with fp_episodes.open("r", encoding="utf-8", newline='') as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                url = row.get("show_url", "")
                episode_shows.add(url)
                season_number = row.get("season_number", "")
                episode_number_in_season = row.get("episode_number_in_season", "")
                episode_number_overall = row.get("episode_number_overall", "")
                episode_title = row.get("episode_title", "")
                airdate = row.get("airdate", "")
                airdate_iso = row.get("airdate_iso", "")

                if url not in shows:
                    logger.log_warning(f"Show URL in shows.tsv not found in episodes.tsv: {url}")
                if not season_number.isdigit():
                    logger.log_warning(f"Invalid season number for URL: {url}")
                if not episode_number_in_season.isdigit():
                    logger.log_warning(f"Invalid episode number in season for URL: {url}")
                if not episode_number_overall.isdigit():
                    logger.log_warning(f"Invalid overall episode number for URL: {url}")
                if not episode_title:
                    logger.log_warning(f"Missing episode title for URL: {url}")
                if not airdate or not airdate_iso:
                    logger.log_warning(f"Missing or invalid airdate for URL: {url}")

        fp_credits = Path(LOGS_DIR) / "extracted_credits.tsv"
        credits_shows = set()
        if not fp_credits.exists():
            logger.log_warning(f"Extracted credits file not found: {fp_credits}")
            return
        with fp_credits.open("r", encoding="utf-8", newline='') as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                url = row.get("show_url", "")
                credits_shows.add(url)
                actor_name = row.get("actor_name", "")
                role = row.get("role", "")

                if url not in shows:
                    logger.log_warning(f"Show URL in shows.tsv not found in credits.tsv: {url}")
                if not actor_name:
                    logger.log_warning(f"Missing actor name for URL: {url}")
                if not role:
                    logger.log_warning(f"Missing role for URL: {url}")

        for show in shows:
            if show not in season_shows:
                logger.log_warning(f"Show URL in shows.tsv not found in seasons.tsv: {show}")
            if show not in episode_shows:
                logger.log_warning(f"Show URL in shows.tsv not found in episodes.tsv: {show}")
            if show not in credits_shows:
                logger.log_warning(f"Show URL in shows.tsv not found in credits.tsv: {show}")

if __name__ == "__main__":
    extractor = Extractor()
    if len(sys.argv) > 1 and sys.argv[1] == "validate":
        extractor.validate_extraction()
    else:
        extractor.run()