import scrapy
from urllib.parse import urlparse

class DocsSpider(scrapy.Spider):
    name = "docs"
    
    custom_settings = {
        "DEPTH_LIMIT": 8
    }
    
    # Class attributes
    start_urls = []
    allowed_domains = [
        "fastapi.tiangolo.com",
        "docs.python.org",
        "developer.mozilla.org",
        "docs.docker.com",
        "kubernetes.io",
        "k8s.io",
        "pytorch.org",
        "huggingface.co",
        "python.langchain.com",
        "docs.aws.amazon.com"
    ]
    
    # Predefined domains requiring Playwright
    PLAYWRIGHT_DOMAINS = {
        "react.dev",
        "nodejs.org",
        "developer.mozilla.org",
        "huggingface.co",
        "python.langchain.com"
    }

    # Sources configuration
    SOURCES_CONFIG = {
        "fastapi": {
            "start": "https://fastapi.tiangolo.com/tutorial/",
            "domain": "fastapi.tiangolo.com"
        },
        "python": {
            "start": "https://docs.python.org/3/",
            "domain": "docs.python.org"
        },
        "mdn": {
            "start": "https://developer.mozilla.org/en-US/docs/Web",
            "domain": "developer.mozilla.org"
        },
        "docker": {
            "start": "https://docs.docker.com/get-started/",
            "domain": "docs.docker.com"
        },
        "kubernetes": {
            "start": "https://kubernetes.io/docs/home/",
            "domain": "kubernetes.io"
        },
        "pytorch": {
            "start": "https://pytorch.org/docs/stable/index.html",
            "domain": "pytorch.org"
        },
        "huggingface": {
            "start": "https://huggingface.co/docs",
            "domain": "huggingface.co"
        },
        "langchain": {
            "start": "https://python.langchain.com/docs/get_started/introduction",
            "domain": "python.langchain.com"
        },
        "aws": {
            "start": "https://docs.aws.amazon.com/",
            "domain": "docs.aws.amazon.com"
        }
    }

    # Limits per source
    MAX_PAGES_PER_SOURCE = {
        "fastapi": 20000,
        "python": 20000,
        "docker": 20000,
        "kubernetes": 20000,
        "mdn": 20000,
        "pytorch": 20000,
        "huggingface": 20000,
        "langchain": 20000,
        "aws": 20000
    }

    # Ignore patterns to save bandwidth and noise
    IGNORE_KEYWORDS = [
        "/blog/", "/community/", "/events/", "/changelog/", 
        "/releases/", "/feed/", "/rss/", "/whatsnew/"
    ]
    
    # Non-English language subpaths to exclude
    EXCLUDE_LANGUAGES = [
        "/zh/", "/zh-cn/", "/fr/", "/de/", "/es/", "/ja/", 
        "/ko/", "/ru/", "/pt-br/", "/it/", "/nl/", "/vi/", 
        "/id/", "/pl/", "/tr/", "/ar/", "/hi/", "/uk/"
    ]

    def __init__(self, source="fastapi", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.source_arg = source.lower()
        self.scraped_counts = {k: 0 for k in self.SOURCES_CONFIG.keys()}
        
        # Determine start URLs dynamically
        targets = []
        if self.source_arg == "all":
            targets = list(self.SOURCES_CONFIG.keys())
        elif self.source_arg in self.SOURCES_CONFIG:
            targets = [self.source_arg]
        else:
            targets = ["fastapi"]
            
        self.start_urls = [self.SOURCES_CONFIG[src]["start"] for src in targets]

    def get_source_from_url(self, url):
        parsed = urlparse(url)
        domain = parsed.netloc
        
        if "fastapi.tiangolo" in domain:
            return "fastapi"
        elif "docs.python.org" in domain:
            return "python"
        elif "developer.mozilla" in domain:
            return "mdn"
        elif "docs.docker.com" in domain:
            return "docker"
        elif "kubernetes.io" in domain or "k8s.io" in domain:
            return "kubernetes"
        elif "pytorch.org" in domain:
            return "pytorch"
        elif "huggingface.co" in domain:
            return "huggingface"
        elif "python.langchain.com" in domain:
            return "langchain"
        elif "docs.aws.amazon.com" in domain:
            return "aws"
        return "other"

    def is_url_allowed(self, url, source):
        # Exclude non-English paths
        if source == "mdn" and "/en-US/" not in url:
            return False
            
        # General exclusion of non-English language subpaths
        for lang in self.EXCLUDE_LANGUAGES:
            if lang in url.lower():
                return False
                
        # Exclude blog / changelog / etc.
        for kw in self.IGNORE_KEYWORDS:
            if kw in url.lower():
                return False
                
        return True

    def parse(self, response):
        url = response.url
        meta_source = response.meta.get("source") or self.get_source_from_url(url)
        meta_domain = response.meta.get("domain") or urlparse(url).netloc
        
        # Determine if Playwright was used (or should be used on retry)
        is_playwright = response.meta.get("playwright", False) or meta_domain in self.PLAYWRIGHT_DOMAINS
        
        # Enforce page limit per source
        if meta_source in self.scraped_counts:
            if self.scraped_counts[meta_source] >= self.MAX_PAGES_PER_SOURCE.get(meta_source, 100):
                self.logger.info(f"Page limit reached for source '{meta_source}'. Stopping follows.")
                return

        # Check URL validity (language and path exclusions)
        if not self.is_url_allowed(url, meta_source):
            return

        # Fallback JS checks
        body_str = response.text
        has_js_app_mount = (
            '<div id="root">' in body_str or 
            '<div id="app">' in body_str or 
            '<div id="__next">' in body_str
        )
        
        # If it seems JS-rendered, and we haven't rendered it yet, retry with Playwright
        if not is_playwright and has_js_app_mount and len(body_str) < 15000:
            self.logger.info(f"Fallback: JS-rendered page detected at {url}. Retrying with Playwright...")
            new_meta = response.meta.copy()
            new_meta["playwright"] = True
            new_meta["source"] = meta_source
            new_meta["domain"] = meta_domain
            yield scrapy.Request(url, callback=self.parse, meta=new_meta, dont_filter=True)
            return

        # Increment scraped pages counter for this source
        if meta_source in self.scraped_counts:
            self.scraped_counts[meta_source] += 1

        yield {
            "url": url,
            "domain": meta_domain,
            "source": meta_source,
            "title": response.css("title::text").get() or "",
            "html": response.text
        }

        # Follow links
        for link in response.css("a::attr(href)").getall():
            next_page = response.urljoin(link)
            parsed_next = urlparse(next_page)
            
            # Check if domain is allowed
            domain_allowed = any(parsed_next.netloc.endswith(d) for d in self.allowed_domains)
            if domain_allowed and parsed_next.scheme in ["http", "https"]:
                next_source = self.get_source_from_url(next_page)
                
                # Only follow if it matches the current crawl scope
                if self.source_arg == "all" or next_source == meta_source:
                    if self.is_url_allowed(next_page, next_source):
                        use_playwright = parsed_next.netloc in self.PLAYWRIGHT_DOMAINS
                        yield response.follow(
                            next_page, 
                            callback=self.parse, 
                            meta={"playwright": use_playwright, "source": next_source, "domain": parsed_next.netloc}
                        )
