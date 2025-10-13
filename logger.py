import logging
import json
from datetime import datetime

class CrawlerLogger:
    def __init__(self, log_file='crawler.log'):
        self.logger = logging.getLogger('CrawlerLogger')
        self.logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
        self.stats = {
            'pages_crawled': 0,
            'links_found': 0,
            'errors': 0,
            'start_time': datetime.now()
        }
    
    def log_page_crawled(self, url, links_found, response_time):
        self.stats['pages_crawled'] += 1
        self.stats['links_found'] += links_found
        
        self.logger.info(json.dumps({
            'event': 'page_crawled',
            'url': url,
            'links_found': links_found,
            'response_time_s': response_time,
            'total_pages': self.stats['pages_crawled']
        }))
    
    def log_error(self, url, error, retry_count=0):
        self.stats['errors'] += 1
        self.logger.error(json.dumps({
            'event': 'crawl_error',
            'url': url,
            'error': str(error),
            'retry_count': retry_count
        }))

    def log_skip(self, url, reason):
        self.logger.warning(json.dumps({
            'event': 'page_skipped',
            'url': url,
            'reason': reason
        }))

    def log_info(self, message):
        self.logger.info(json.dumps({
            'event': 'info',
            'message': message
        }))
    
    def print_stats(self):
        elapsed = datetime.now() - self.stats['start_time']
        rate = self.stats['pages_crawled'] / elapsed.total_seconds() if elapsed.total_seconds() > 0 else 0
        
        print(f"""
            Crawler Statistics:
            - Pages crawled: {self.stats['pages_crawled']}
            - Links found: {self.stats['links_found']}
            - Errors: {self.stats['errors']}
            - Runtime: {elapsed}
            - Crawl rate: {rate:.2f} pages/second
        """)