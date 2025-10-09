import os, time, random, re
import requests
import urllib.robotparser as urob
from urllib.parse import urljoin

BASE_URL = "https://epguides.com/"
UA = "UniSeriesCrawler/1.0 (+mailto:oleksandr.sakalosh0@gmail.com)"
BASE_DELAY = 2.0 
TIMEOUT = 20
SAVE_DIR = "data"

def load_robots(base_url):
    rp = urob.RobotFileParser()
    robots_url = urljoin(base_url, "robots.txt")
    try:
        rp.set_url(robots_url)
        rp.read()
        return rp
    except Exception as e:
        print(f"Error loading robots.txt from {robots_url}: {e}")
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
            print("Warning: couldn't load robots.txt")
        self.visited = set()
        self.to_visit = set([base_url])

    def allowed(self, url):
        if not self.rp:
            return True
        return self.rp.can_fetch(UA, url)
    
    def sleep_politely(self):
        delay = BASE_DELAY + random.uniform(0, BASE_DELAY)
        time.sleep(delay)

    def fetch(self, url):
        if not self.allowed(url):
            raise ValueError(f"Fetching disallowed by robots.txt: {url}")
        self.sleep_politely()

        headers = {
            "User-Agent": UA,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
        try:
            response = requests.get(url, headers=headers, timeout=TIMEOUT)
            response.raise_for_status()
            return response.text

        except requests.HTTPError as e:
            print(f"HTTP error fetching {url}: {e}")
            return None
        except requests.RequestException as e:
            print(f"Error fetching {url}: {e}")
            return None
        
    def extract_links(self, html):
        return re.findall(r'href=["\'](.*?)["\']', html, re.IGNORECASE)
        
    def save_html(self, content, filename):
        os.makedirs(SAVE_DIR, exist_ok=True)
        filepath = os.path.join(SAVE_DIR, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

    def run(self):
        while self.to_visit:
            url = self.to_visit.pop()
            if url in self.visited:
                continue
            print(f"Fetching: {url}")
            html = self.fetch(url)
            if html:
                filename = re.sub(r'[^a-zA-Z0-9]', '_', url.replace(self.base_url, '')) + ".html"
                self.save_html(html, filename)
                links = self.extract_links(html)
                for link in links:
                    full_url = urljoin(self.base_url, link)
                    if full_url not in self.visited:
                        self.to_visit.add(full_url)
            self.visited.add(url)