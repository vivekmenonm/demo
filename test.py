import json
import hashlib
import asyncio
import time
import tempfile

from datetime import datetime, timezone, timedelta
from urllib.parse import urlparse

import aiohttp
import boto3
import trafilatura
import feedparser

from bs4 import BeautifulSoup
from ddgs import DDGS

# -------------------------------------
# AWS CONFIG
# -------------------------------------

CONFIG_BUCKET = "news-crawler-config"
CONFIG_KEY = "config.json"

RESULT_BUCKET = "analytics-news"
RESULT_PREFIX = "country_news"

TMP_DIR = tempfile.gettempdir()
LOCAL_CONFIG = f"{TMP_DIR}/config.json"
OUTPUT_FILE = f"{TMP_DIR}/news_results.json"

s3 = boto3.client("s3")

# -------------------------------------
# LOAD CONFIG
# -------------------------------------

print("Downloading config...")
s3.download_file(CONFIG_BUCKET, CONFIG_KEY, LOCAL_CONFIG)

with open(LOCAL_CONFIG) as f:
    CONFIG = json.load(f)

COUNTRIES = CONFIG.get("countries", [])
KEYWORDS = CONFIG.get("keywords", [])
COUNTRY_WEBSITES = CONFIG.get("country_websites", {})
GLOBAL_WEBSITES = CONFIG.get("global_websites", [])

COUNTRY_CODE_MAP = CONFIG.get("country_code_map", {})

DAYS_BACK = CONFIG.get("days_back", 2)

MAX_RESULTS = 10
QUERY_CONCURRENCY = 15
ARTICLE_CONCURRENCY = 200

# -------------------------------------
# HELPERS
# -------------------------------------

def parse_date(value):
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except:
        return None


def iso2_for_country(country):
    return COUNTRY_CODE_MAP.get(country, country)


# -------------------------------------
# RELEVANCE FILTER (BALANCED)
# -------------------------------------

def is_relevant(article, country):
    text = (
        (article.get("title") or "") + " " +
        (article.get("snippet") or "")
    ).lower()

    country_lower = country.lower()
    domain = article.get("domain", "")

    local_sites = COUNTRY_WEBSITES.get(country, [])

    score = 0

    # mention of country
    if country_lower in text:
        score += 2

    # local trusted source
    if any(site in domain for site in local_sites):
        score += 2

    # domain relevance (your topic)
    if any(word in text for word in ["real estate", "housing", "infrastructure"]):
        score += 1

    return score >= 2


# -------------------------------------
# DEDUP
# -------------------------------------

seen = set()

def is_duplicate(url):
    key = hashlib.md5(url.encode()).hexdigest()
    if key in seen:
        return True
    seen.add(key)
    return False


# -------------------------------------
# SEARCH
# -------------------------------------

def ddg_search(query):
    results = []
    try:
        with DDGS() as ddgs:
            raw = ddgs.news(query, max_results=MAX_RESULTS)
            for r in raw:
                url = r.get("url")
                if not url:
                    continue
                results.append({
                    "title": r.get("title"),
                    "url": url,
                    "date": r.get("date"),
                    "snippet": r.get("body"),
                    "domain": urlparse(url).netloc
                })
    except:
        pass
    return results


def google_news(query):
    results = []
    url = f"https://news.google.com/rss/search?q={query}"

    try:
        feed = feedparser.parse(url)
        for e in feed.entries[:MAX_RESULTS]:
            link = e.link
            results.append({
                "title": e.title,
                "url": link,
                "date": getattr(e, "published", ""),
                "snippet": getattr(e, "summary", ""),
                "domain": urlparse(link).netloc
            })
    except:
        pass

    return results


# -------------------------------------
# BUILD QUERIES (FIXED)
# -------------------------------------

def build_queries():
    queries = []

    for c in COUNTRIES:

        # keyword-based queries
        for k in KEYWORDS:
            queries.append((c, f"{k} in {c}"))
            queries.append((c, f"{c} {k} news"))

        # local websites
        for site in COUNTRY_WEBSITES.get(c, []):
            queries.append((c, f"{c} site:{site}"))

        # global websites (fixed bug)
        for site in GLOBAL_WEBSITES:
            queries.append((c, f"{c} news site:{site}"))

    return queries


# -------------------------------------
# ASYNC FETCH
# -------------------------------------

async def fetch_html(session, url):
    try:
        async with session.get(url, timeout=10) as r:
            return await r.text()
    except:
        return ""


async def extract_article(session, url):
    html = await fetch_html(session, url)

    if not html:
        return ""

    try:
        text = trafilatura.extract(html)
        if text and len(text) > 200:
            return text
    except:
        pass

    try:
        soup = BeautifulSoup(html, "html.parser")
        paragraphs = soup.find_all("p")
        return " ".join(p.get_text() for p in paragraphs)
    except:
        return ""


async def process_article(session, article, country, query):
    url = article["url"]

    if is_duplicate(url):
        return None

    # relevance filter
    if not is_relevant(article, country):
        return None

    # date filter
    if DAYS_BACK:
        cutoff = datetime.now(timezone.utc) - timedelta(days=DAYS_BACK)
        parsed = parse_date(article.get("date") or "")
        if parsed and parsed < cutoff:
            return None

    content = await extract_article(session, url)

    return {
        "country": country,
        "query": query,
        "title": article["title"],
        "url": url,
        "domain": article["domain"],
        "date": article["date"],
        "snippet": article["snippet"],
        "article_content": content,
        "scraped_at": datetime.now(timezone.utc).isoformat()
    }


# -------------------------------------
# PROCESS ARTICLES
# -------------------------------------

async def process_articles(country, query, articles):
    connector = aiohttp.TCPConnector(limit=ARTICLE_CONCURRENCY)

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            process_article(session, a, country, query)
            for a in articles
        ]

        results = await asyncio.gather(*tasks)

    return [r for r in results if r]


# -------------------------------------
# RUN QUERY
# -------------------------------------

async def run_query(country, query):
    ddg = ddg_search(query)
    gnews = google_news(query)

    combined = ddg + gnews

    return await process_articles(country, query, combined)


# -------------------------------------
# RUN ALL
# -------------------------------------

async def run_all_queries(queries):
    semaphore = asyncio.Semaphore(QUERY_CONCURRENCY)

    async def sem_task(c, q):
        async with semaphore:
            return await run_query(c, q)

    tasks = [sem_task(c, q) for c, q in queries]

    results = await asyncio.gather(*tasks)

    articles = []
    for r in results:
        articles.extend(r)

    return articles


# -------------------------------------
# MAIN
# -------------------------------------

def main():
    start = time.time()

    print("Run started:", datetime.now(timezone.utc))

    queries = build_queries()
    print("Total queries:", len(queries))

    articles = asyncio.run(run_all_queries(queries))

    print("Articles collected:", len(articles))

    with open(OUTPUT_FILE, "w") as f:
        json.dump({"articles": articles}, f, indent=2)

    s3.upload_file(OUTPUT_FILE, RESULT_BUCKET, f"{RESULT_PREFIX}/news.json")

    print("Uploaded to S3")


if __name__ == "__main__":
    main()
