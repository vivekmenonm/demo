import json
import hashlib
import asyncio
import time
import os

from urllib.parse import urlparse, quote

import aiohttp
import trafilatura
import feedparser
from bs4 import BeautifulSoup
from ddgs import DDGS

from sentence_transformers import SentenceTransformer, util

# -------------------------------------
# LOAD MODEL
# -------------------------------------

model = SentenceTransformer('all-MiniLM-L6-v2')

# -------------------------------------
# CONFIG
# -------------------------------------

CONFIG_FILE = "config.json"
OUTPUT_DIR = "./output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_RESULTS = 10

with open(CONFIG_FILE) as f:
    CONFIG = json.load(f)

COUNTRIES = CONFIG["countries"]
KEYWORDS = CONFIG["keywords"]

GLOBAL_SITES = CONFIG["global_websites"]
COUNTRY_SITES = CONFIG["country_websites"]

print("Config loaded:", COUNTRIES)

# -------------------------------------
# DYNAMIC COUNTRY CONTEXT (🔥 IMPORTANT)
# -------------------------------------

def build_country_context(countries, keywords):
    context = {}
    unique_keywords = list(set(k.lower() for k in keywords))

    for c in countries:
        context[c] = c + " " + " ".join(unique_keywords)

    return context

COUNTRY_CONTEXT = build_country_context(COUNTRIES, KEYWORDS)

# -------------------------------------
# CONSTANTS
# -------------------------------------

ALL_COUNTRIES = [c.lower() for c in COUNTRIES] + ["dubai", "uae", "china", "us"]

# -------------------------------------
# DEDUP
# -------------------------------------

seen = set()

def is_duplicate(url):
    h = hashlib.md5(url.encode()).hexdigest()
    if h in seen:
        return True
    seen.add(h)
    return False

# -------------------------------------
# SEARCH
# -------------------------------------

def google_news(query):
    results = []
    url = f"https://news.google.com/rss/search?q={quote(query)}&hl=en-IN&gl=IN&ceid=IN:en"
    feed = feedparser.parse(url)

    for e in feed.entries[:MAX_RESULTS]:
        results.append({
            "title": e.title,
            "url": e.link,
            "snippet": getattr(e, "summary", ""),
            "domain": urlparse(e.link).netloc
        })

    return results


def ddg_search(query):
    results = []
    try:
        with DDGS() as ddgs:
            data = list(ddgs.news(query, max_results=MAX_RESULTS))
            for r in data:
                results.append({
                    "title": r.get("title"),
                    "url": r.get("url"),
                    "snippet": r.get("body"),
                    "domain": urlparse(r.get("url")).netloc
                })
    except:
        pass
    return results

# -------------------------------------
# QUERY BUILDER
# -------------------------------------

def build_queries():
    queries = []

    for country in COUNTRIES:
        for keyword in KEYWORDS:

            # 1. Broad query (all sources)
            queries.append((country, f"{country} {keyword}"))

            # 2. Country-specific site
            for site in COUNTRY_SITES.get(country, [])[:1]:
                queries.append((country, f"{country} {keyword} site:{site}"))

            # 🔥 3. Global site (IMPORTANT)
            for site in GLOBAL_SITES[:2]:   # limit to top 2 to avoid overload
                queries.append((country, f"{country} {keyword} site:{site}"))

    return queries

# -------------------------------------
# EXTRACTION
# -------------------------------------

async def fetch_html(session, url):
    try:
        async with session.get(url, timeout=10) as r:
            return await r.text()
    except:
        return ""

async def extract_article(session, url, snippet):
    html = await fetch_html(session, url)

    if not html:
        return snippet or ""

    # 1. trafilatura
    text = trafilatura.extract(html)
    if text and len(text) > 100:
        return text

    # 2. fallback BS4
    soup = BeautifulSoup(html, "html.parser")
    paragraphs = soup.find_all("p")
    text = " ".join(p.get_text() for p in paragraphs)

    if text and len(text) > 80:
        return text

    # 3. final fallback
    return snippet or ""

# -------------------------------------
# FILTERING LOGIC
# -------------------------------------

def semantic_score(content, country):
    doc_emb = model.encode(content, convert_to_tensor=True)
    country_emb = model.encode(COUNTRY_CONTEXT[country], convert_to_tensor=True)
    return util.cos_sim(doc_emb, country_emb).item()


def keyword_score(content):
    text = content.lower()
    matches = sum(1 for k in KEYWORDS if k.lower() in text)
    return matches / max(len(KEYWORDS), 1)


def country_ratio(content, country):
    text = content.lower()
    words = text.split()
    count = sum(1 for w in words if country.lower() in w)
    return count / max(len(words), 1)


def dominant_country(content):
    text = content.lower()
    counts = {c: text.count(c) for c in ALL_COUNTRIES}
    return max(counts, key=counts.get), counts


def is_country_involved(content, country):
    text = content.lower()

    keywords = [
        "government", "policy", "economy", "market",
        "investment", "trade", "growth",
        "housing", "infrastructure", "real estate"
    ]

    return any(country.lower() in text and kw in text for kw in keywords)

# -------------------------------------
# PROCESS ARTICLE
# -------------------------------------

async def process_article(session, article, country):
    url = article["url"]

    if not url or is_duplicate(url):
        return None

    content = await extract_article(session, url, article.get("snippet"))

    if not content or len(content) < 80:
        return None

    # -------- SCORING --------
    sem = semantic_score(content, country)
    kw = keyword_score(content)
    ratio = country_ratio(content, country)
    main_country, counts = dominant_country(content)
    involved = is_country_involved(content, country)

    # -------- FILTER RULES --------

    if kw < 0.05:
        return None

    if ratio < 0.002:
        return None

    if main_country != country.lower() and not involved:
        return None

    if sem < 0.35:
        return None

    score = round(sem * 10 + kw * 5 + ratio * 100, 2)

    return {
        "country": country,
        "title": article["title"],
        "url": url,
        "domain": article["domain"],
        "content": content,
        "score": score
    }

# -------------------------------------
# RUN QUERY
# -------------------------------------

counter = 0

async def run_query(country, query, total):
    global counter
    counter += 1

    print(f"[{counter}/{total}] 🔍 {query}")

    g = google_news(query)
    d = ddg_search(query)

    combined = g + d

    async with aiohttp.ClientSession() as session:
        tasks = [process_article(session, a, country) for a in combined]
        results = await asyncio.gather(*tasks)

    valid = [r for r in results if r]

    print(f"   → Valid: {len(valid)}")

    return valid

# -------------------------------------
# MAIN
# -------------------------------------

async def run_all():
    queries = build_queries()
    total = len(queries)

    print(f"\n🚀 Total Queries: {total}\n")

    tasks = [run_query(c, q, total) for c, q in queries]
    results = await asyncio.gather(*tasks)

    articles = []
    for r in results:
        articles.extend(r)

    articles = sorted(articles, key=lambda x: x["score"], reverse=True)

    return articles


def main():
    start = time.time()

    articles = asyncio.run(run_all())

    print("\n✅ TOTAL ARTICLES:", len(articles))

    # -------------------------------------
    # GROUP BY COUNTRY
    # -------------------------------------

    country_map = {}

    for article in articles:
        c = article["country"]

        if c not in country_map:
            country_map[c] = []

        country_map[c].append(article)

    # -------------------------------------
    # SORT + SAVE PER COUNTRY
    # -------------------------------------

    for country, items in country_map.items():

        # 🔥 sort descending by score
        items = sorted(items, key=lambda x: x["score"], reverse=True)

        # create folder
        country_dir = os.path.join(OUTPUT_DIR, country)
        os.makedirs(country_dir, exist_ok=True)

        # save file
        file_path = os.path.join(country_dir, "news.json")

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(items, f, indent=2, ensure_ascii=False)

        print(f"📁 Saved {len(items)} articles → {country}/news.json")

    print("⏱ Runtime:", round(time.time() - start, 2), "seconds")


if __name__ == "__main__":
    main()