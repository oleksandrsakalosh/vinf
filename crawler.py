from pathlib import Path
import hashlib
import os, time, random, re, csv
import requests
import urllib.robotparser as urob
from urllib.parse import urljoin, urlsplit
from collections import deque

from logger import CrawlerLogger

BASE_URL = "https://epguides.com/"
UA = "UniSeriesCrawler/1.0 (+mailto:oleksandr.sakalosh0@gmail.com)"
BASE_DELAY = 2.0 
TIMEOUT = 20

logger = CrawlerLogger()
SAVE_DIR = "data" # .gitignore this directory

HREF_RE = re.compile(
    r'href=["\'](.*?)["\']', re.IGNORECASE
)

SKIP_SCHEMES = ('mailto:', 'javascript:', 'tel:', 'data:')

mode = "controlled" # "prod" | "dev" | "controlled"
LINKS = [
    "https://epguides.com/Simpsons/",
]

def load_robots(base_url):
    rp = urob.RobotFileParser()
    robots_url = urljoin(base_url, "robots.txt")
    try:
        rp.set_url(robots_url)
        rp.read()
        return rp
    except Exception as e:
        logger.log_error(robots_url, e)
        return None

class MyCrawler():
    '''
    A web crawler class that respects robots.txt and implements polite crawling.
    Saves fetched pages as HTML files in local storage.
    ''' 
    def __init__(self, base_url=BASE_URL):
        self.base_url = base_url
        self.rp = load_robots(base_url)
        if not self.rp:
            logger.log_error(base_url, "Failed to load robots.txt")
        self.visited = set()
        self.to_visit = deque([base_url])

    def allowed(self, url):
        if not self.rp:
            return True
        return self.rp.can_fetch(UA, url)
    
    def sleep_politely(self):
        delay = BASE_DELAY + random.uniform(0, BASE_DELAY)
        time.sleep(delay)

    def fetch(self, url):
        self.sleep_politely()

        headers = {
            "User-Agent": UA,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
        try:
            response = requests.get(url, headers=headers, timeout=TIMEOUT)
            response.raise_for_status()
            content_type = response.headers.get('Content-Type', '').lower()
            if 'text/html' in content_type:
                return response.text
            else:
                logger.log_skip(url, f"Unsupported Content-Type: {content_type}")
                return None
        except requests.HTTPError as e:
            logger.log_error(url, e, retry_count=0)
            return None
        except requests.RequestException as e:
            logger.log_error(url, e, retry_count=0)
            return None
        
    def normalize_url(self, link):
        return urljoin(self.base_url, link)

    def extract_links(self, html):
        links = []
        for match in HREF_RE.finditer(html):
            link = match.group(1)
            if link.startswith(SKIP_SCHEMES):
                continue
            links.append(link)
        return links

    def save_html(self, content, url):
        parts = urlsplit(url)
        path = parts.path
        if path.endswith("/") or not path:
            path = path + "index.html"
        if parts.query:
            h = hashlib.sha1(parts.query.encode("utf-8")).hexdigest()[:10]
            root, ext = os.path.splitext(path)
            path = f"{root}__q_{h}{ext or '.html'}"
        fp = Path(SAVE_DIR) / path.lstrip("/")
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content, encoding="utf-8")

        mapping_fp = Path(SAVE_DIR) / "url_mapping.tsv"
        write_header = not mapping_fp.exists()
        with mapping_fp.open("a", encoding="utf-8", newline='') as f:
            writer = csv.writer(f, delimiter="\t")
            if write_header:
                writer.writerow(["url", "path"])
            writer.writerow([url, str(fp)])

    def run(self):
        if mode == "prod":
            while self.to_visit:
                start = time.time()
                url = self.to_visit.popleft()
                logger.log_info(f"Fetching: {url}")
                if url in self.visited:
                    logger.log_skip(url, "Already visited")
                    continue

                html = self.fetch(url)
                if html:
                    self.save_html(html, url)
                    links = self.extract_links(html)
                    count = 0
                    for link in links:
                        new_url = self.normalize_url(link)
                        if not new_url: 
                            logger.log_skip(link, "Failed to normalize URL")
                            continue
                        if not new_url.startswith(self.base_url):
                            logger.log_skip(new_url, "Cross-origin link")
                            continue
                        if not self.allowed(new_url):
                            logger.log_skip(new_url, "Disallowed by robots.txt")
                            continue
                        if new_url in self.visited:
                            logger.log_skip(new_url, "Already visited")
                            continue

                        self.to_visit.append(new_url)
                        count += 1
                    response_time = time.time() - start
                    logger.log_page_crawled(url, count, response_time=response_time)
                else:
                    logger.log_skip(url, "No HTML content")
                self.visited.add(url)
        elif mode == "dev":
            max_depth = 2
            current_depth = 0

            while self.to_visit and current_depth < max_depth:
                logger.log_info(f"Current depth: {current_depth}")
                logger.log_info(f"URLs to visit: {len(self.to_visit)}")
                start = time.time()
                next_level = deque()
                for url in list(self.to_visit):
                    logger.log_info(f"Fetching: {url}")
                    if url in self.visited:
                        logger.log_skip(url, "Already visited")
                        continue

                    html = self.fetch(url)
                    if html:
                        self.save_html(html, url)
                        links = self.extract_links(html)
                        count = 0
                        for link in links:
                            new_url = self.normalize_url(link)
                            if not new_url: 
                                logger.log_skip(link, "Failed to normalize URL")
                                continue
                            if not new_url.startswith(self.base_url):
                                logger.log_skip(new_url, "Cross-origin link")
                                continue
                            if not self.allowed(new_url):
                                logger.log_skip(new_url, "Disallowed by robots.txt")
                                continue
                            if new_url in self.visited:
                                logger.log_skip(new_url, "Already visited")
                                continue
                            next_level.append(new_url)
                            count += 1
                        response_time = time.time() - start
                        logger.log_page_crawled(url, count, response_time)
                    else:
                        logger.log_skip(url, "No HTML content")
                    self.visited.add(url)
                self.to_visit = next_level
                current_depth += 1
        elif mode == "controlled":
            for url in LINKS:
                start = time.time()
                logger.log_info(f"Fetching: {url}")
                if url in self.visited:
                    logger.log_skip(url, "Already visited")
                    continue

                html = self.fetch(url)
                if html:
                    self.save_html(html, url)
                    response_time = time.time() - start
                    logger.log_page_crawled(url, 0, response_time)
                else:
                    logger.log_skip(url, "No HTML content")
                self.visited.add(url)

if __name__ == "__main__":
    crawler = MyCrawler()
    crawler.run()