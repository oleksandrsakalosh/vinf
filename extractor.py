from dataclasses import dataclass, field
from typing import List, Dict, Optional

PATH_TO_MAPPING = "data/url_mapping_extracted.tsv"

@dataclass
class EpisodeInfo:
    season: Optional[int] = None
    number_overall: Optional[int] = None
    number_in_season: Optional[int] = None
    title: Optional[str] = None
    airdate: Optional[str] = None
    airdate_iso: Optional[str] = None

@dataclass
class SeasonInfo:
    episode_count: Optional[int] = None
    episodes: Optional[Dict[int, EpisodeInfo]] = field(default_factory=dict)

@dataclass
class ShowInfo:
    title: Optional[str] = None
    dates: Optional[str] = None
    date_start: Optional[int] = None
    date_end: Optional[int] = None
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
class ShowExtract:
    url: Optional[str] = None
    show: ShowInfo
    seasons: Optional[Dict[int, SeasonInfo]] = field(default_factory=dict)
    specials: Optional[SeasonInfo] = None

class Extractor:
    '''
    A class to handle extraction of series data from saved html pages.
    Reads lies in data/url_mapping_extracted.tsv one by one and extracts relevant information.
    Saves extracted data to structured file.
    '''
    def __init__(self):
        pass
    