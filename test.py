# Start 2
import json
import hashlib
import asyncio
import time
import tempfile
import requests

from datetime import datetime, timezone, timedelta
from urllib.parse import urlparse, quote, parse_qs

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

print("⬇️ Downloading config...")
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
QUERY_CONCURRENCY = 8
ARTICLE_CONCURRENCY = 30

# 🔥 Normalize keywords once
KEYWORDS_LOWER = [k.lower() for k in KEYWORDS]

print(
    "✅ Config loaded:",
    f"countries={len(COUNTRIES)}",
    f"keywords={len(KEYWORDS)}",
    f"country_sites={sum(len(v) for v in COUNTRY_WEBSITES.values())}",
    f"global_sites={len(GLOBAL_WEBSITES)}"
)

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
# GOOGLE NEWS REAL URL FIX
# -------------------------------------

def extract_real_url(link):
    try:
        parsed = urlparse(link)
        query = parse_qs(parsed.query)

        if "url" in query:
            return query["url"][0]

        html = requests.get(link, timeout=5).text
        soup = BeautifulSoup(html, "html.parser")
        a = soup.find("a", href=True)

        return a["href"] if a else link

    except:
        return link


# -------------------------------------
# 🔥 UPDATED RELEVANCE FILTER
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

    # ✅ Country match
    if country_lower in text:
        score += 2

    # ✅ Trusted source boost
    if any(site in domain for site in local_sites):
        score += 2

    # 🔥 Keyword matching (ALL keywords)
    keyword_hits = 0

    for k in KEYWORDS_LOWER:
        words = k.split()
        if any(word in text for word in words):
            keyword_hits += 1

    if keyword_hits >= 1:
        score += 1

    if keyword_hits >= 2:
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
        query = quote(query)

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

    except Exception as e:
        print("⚠️ DDG error:", e)

    return results


def google_news(query):
    results = []

    try:
        encoded_query = quote(query)
        url = f"https://news.google.com/rss/search?q={encoded_query}"

        feed = feedparser.parse(url)

        for e in feed.entries[:MAX_RESULTS]:
            real_url = extract_real_url(e.link)

            results.append({
                "title": e.title,
                "url": real_url,
                "date": getattr(e, "published", ""),
                "snippet": getattr(e, "summary", ""),
                "domain": urlparse(real_url).netloc
            })

    except Exception as e:
        print("⚠️ Google News error:", e)

    return results


# -------------------------------------
# BUILD QUERIES
# -------------------------------------

def build_queries():
    queries = []

    for c in COUNTRIES:

        for k in KEYWORDS:
            queries.append((c, f"{k} in {c}"))
            queries.append((c, f"{c} {k} news"))

        for site in COUNTRY_WEBSITES.get(c, []):
            queries.append((c, f"{c} site:{site}"))

        for site in GLOBAL_WEBSITES:
            queries.append((c, f"{c} news site:{site}"))

    return queries


# -------------------------------------
# ASYNC FETCH
# -------------------------------------

async def fetch_html(session, url):
    try:
        async with session.get(url, timeout=10) as r:
            return await r.text(errors="ignore")
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

    if not is_relevant(article, country):
        return None

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
# PROCESS ARTICLES (with progress)
# -------------------------------------

async def process_articles(country, query, articles):
    connector = aiohttp.TCPConnector(limit=ARTICLE_CONCURRENCY)

    total = len(articles)
    completed = 0

    async with aiohttp.ClientSession(connector=connector) as session:

        async def wrapped(a):
            nonlocal completed
            result = await process_article(session, a, country, query)
            completed += 1

            if completed % 10 == 0:
                print(f"📄 {country}: {completed}/{total}")

            return result

        tasks = [wrapped(a) for a in articles]

        results = await asyncio.gather(*tasks)

    return [r for r in results if r]


# -------------------------------------
# RUN QUERY
# -------------------------------------

async def run_query(country, query):
    print(f"🔎 {country} → {query}")

    ddg = ddg_search(query)
    gnews = google_news(query)

    combined = ddg + gnews

    return await process_articles(country, query, combined)


# -------------------------------------
# RUN ALL QUERIES (with progress)
# -------------------------------------

async def run_all_queries(queries):
    semaphore = asyncio.Semaphore(QUERY_CONCURRENCY)

    total = len(queries)
    completed = 0

    async def sem_task(c, q):
        nonlocal completed
        async with semaphore:
            result = await run_query(c, q)
            completed += 1
            print(f"✅ Query {completed}/{total}")
            return result

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

    print("🚀 Started:", datetime.now(timezone.utc))

    queries = build_queries()
    print("📊 Total queries:", len(queries))

    articles = asyncio.run(run_all_queries(queries))

    print("📰 Articles collected:", len(articles))

    with open(OUTPUT_FILE, "w") as f:
        json.dump({"articles": articles}, f, indent=2)

    s3.upload_file(OUTPUT_FILE, RESULT_BUCKET, f"{RESULT_PREFIX}/news.json")

    print("☁️ Uploaded to S3")

    runtime = round(time.time() - start, 2)
    print(f"⏱ Finished in {runtime}s")


if __name__ == "__main__":
    main()
