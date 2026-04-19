import json
import hashlib
import asyncio
import time
import os
import boto3

from datetime import datetime, timezone
from urllib.parse import urlparse

import aiohttp
import trafilatura
from bs4 import BeautifulSoup
from ddgs import DDGS
from email.utils import parsedate_to_datetime

from sentence_transformers import SentenceTransformer, util
from collections import defaultdict

# -------------------------------------
# MODEL
# -------------------------------------

model = SentenceTransformer('all-MiniLM-L6-v2')

# -------------------------------------
# CONFIG
# -------------------------------------

CONFIG = json.load(open("config.json"))

COUNTRIES = CONFIG["countries"]
KEYWORDS = CONFIG["keywords"]

GLOBAL_SITES = CONFIG["global_websites"]
COUNTRY_SITES = CONFIG["country_websites"]

ALLOWED_WEBSITES = CONFIG.get("allowed_websites", [])
DAYS_BACK = CONFIG["days_back"]

AWS_ENABLED = CONFIG.get("aws_enabled", False)
RESULT_BUCKET = CONFIG.get("result_bucket")
RESULT_PREFIX = CONFIG.get("result_prefix", "country_news")
OUTPUT_DIR = CONFIG.get("output_dir", "country_news")
COUNTRY_CODE_MAP = CONFIG.get("country_code_map", {})

os.makedirs(OUTPUT_DIR, exist_ok=True)

s3 = boto3.client("s3") if AWS_ENABLED else None

MAX_RESULTS = 10
BATCH_SIZE = 20
ARTICLE_CONCURRENCY = 20

article_semaphore = asyncio.Semaphore(ARTICLE_CONCURRENCY)

# -------------------------------------
# EMBEDDING CACHE
# -------------------------------------

embedding_cache = {}

def get_embedding(text):
    key = text[:300]
    if key not in embedding_cache:
        embedding_cache[key] = model.encode(text, convert_to_tensor=True)
    return embedding_cache[key]

# -------------------------------------
# CATEGORY LABELS
# -------------------------------------

CATEGORY_LABELS = [
    "real estate housing property",
    "infrastructure transport rail airport road",
    "macro economy inflation gdp interest rates",
    "construction materials labor costs",
    "investment foreign investment policy tax",
    "automobile ev industry tesla byd volkswagen"
]

category_embeddings = model.encode(CATEGORY_LABELS, convert_to_tensor=True)

# -------------------------------------
# COUNTRY EMBEDDING (FIX)
# -------------------------------------

country_cache = {}
keyword_cache = {}

def get_country_emb(country):
    if country not in country_cache:
        country_cache[country] = get_embedding(
            f"{country} economy business market major cities"
        )
    return country_cache[country]

def get_keyword_emb(country, keyword):
    key = f"{country}-{keyword}"
    if key not in keyword_cache:
        keyword_cache[key] = get_embedding(f"{country} {keyword}")
    return keyword_cache[key]

# -------------------------------------
# HELPERS
# -------------------------------------

def parse_date(value):
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except:
        try:
            return parsedate_to_datetime(value)
        except:
            return datetime.now(timezone.utc)

def is_recent(d):
    return (datetime.now(timezone.utc) - d).days <= DAYS_BACK

def is_duplicate(url, seen):
    h = hashlib.md5(url.encode()).hexdigest()
    if h in seen:
        return True
    seen.add(h)
    return False

def is_valid(article):
    url = article["url"]
    if any(x in url for x in ["/opinion/", "/author/", "video"]):
        return False
    return len(article.get("title", "")) > 15

# -------------------------------------
# SEARCH
# -------------------------------------

def ddg_search(query):
    out = []
    try:
        with DDGS() as ddgs:
            data = list(ddgs.news(query, max_results=MAX_RESULTS))
            for r in data:
                out.append({
                    "title": r.get("title"),
                    "url": r.get("url"),
                    "date": r.get("date"),
                    "snippet": r.get("body"),
                    "domain": urlparse(r.get("url")).netloc
                })
    except:
        pass
    return out

# -------------------------------------
# QUERY BUILDER
# -------------------------------------

def build_queries():
    queries = []

    for country in COUNTRIES:
        c_sites = COUNTRY_SITES.get(country, [])
        g_sites = GLOBAL_SITES

        for cat, kw_map in KEYWORDS.items():
            for base, vars in kw_map.items():
                for kw in [base] + vars:

                    queries.append((country, base, f"{country} {kw} news", "global"))
                    queries.append((country, base, f"{kw} in {country}", "global"))

                    for s in g_sites:
                        queries.append((country, base, f"{country} {kw} site:{s}", "global"))

                    queries.append((country, base, f"{country} {kw}", "country"))

                    for s in c_sites:
                        queries.append((country, base, f"{country} {kw} site:{s}", "country"))

    return queries

# -------------------------------------
# EXTRACTION
# -------------------------------------

async def fetch(session, url):
    try:
        print(f"🌐 Fetching: {url}")
        async with session.get(url, timeout=10) as r:
            return await r.text()
    except:
        print(f"❌ Fetch failed: {url}")
        return ""

async def extract(session, url, snippet):
    html = await fetch(session, url)

    if not html:
        return snippet or ""

    text = trafilatura.extract(html)
    if text and len(text) > 100:
        return text

    soup = BeautifulSoup(html, "html.parser")
    return " ".join(p.get_text() for p in soup.find_all("p"))

# -------------------------------------
# PROCESS
# -------------------------------------

async def process(session, art, country, keyword, domains, seen, source_type):
    async with article_semaphore:

        url = art["url"]

        if not url or is_duplicate(url, seen):
            return None

        if not is_valid(art):
            return None

        domain = art["domain"]

        if source_type == "global":
            if not any(g in domain for g in GLOBAL_SITES):
                return None

        if ALLOWED_WEBSITES:
            if not any(d in domain for d in domains):
                return None

        d = parse_date(art.get("date"))
        if not is_recent(d):
            return None

        content = await extract(session, url, art.get("snippet"))

        if not content or len(content.split()) < 80:
            print(f"❌ Content fail: {url}")
            return None

        emb = get_embedding(content[:500])

        c_score = util.cos_sim(get_country_emb(country), emb).item()
        if c_score < (0.15 if source_type == "global" else 0.20):
            return None

        score = util.cos_sim(get_keyword_emb(country, keyword), emb).item()

        # fallback boost
        if score < 0.30 and c_score > 0.25:
            score += 0.05

        if score < 0.25:
            return None

        if any(g in domain for g in GLOBAL_SITES):
            score += 0.07

        if country.lower() in art["title"].lower():
            score += 0.05

        sims = util.cos_sim(emb, category_embeddings)[0]
        idx = sims.argmax().item()

        return {
            "country": country,
            "country_code": COUNTRY_CODE_MAP.get(country),
            "keyword": keyword,
            "category": CATEGORY_LABELS[idx],
            "title": art["title"],
            "url": url,
            "domain": domain,
            "date": d.isoformat(),
            "article_content": content,
            "score": score,
            "embedding": emb
        }

# -------------------------------------
# DEDUP
# -------------------------------------

def dedup(arts):
    out = []
    for a in arts:
        if not any(util.cos_sim(a["embedding"], b["embedding"]).item() > 0.8 for b in out):
            out.append(a)

    for a in out:
        del a["embedding"]

    return out

# -------------------------------------
# SAVE
# -------------------------------------

def save_results(articles):
    grouped = defaultdict(list)

    for a in articles:
        grouped[a["country"]].append(a)

    for country, items in grouped.items():
        items = sorted(items, key=lambda x: x["score"], reverse=True)

        iso = COUNTRY_CODE_MAP.get(country, "XX")

        folder = os.path.join(OUTPUT_DIR, iso)
        os.makedirs(folder, exist_ok=True)

        file_path = os.path.join(folder, "news.json")

        with open(file_path, "w") as f:
            json.dump(items, f, indent=2)

        print(f"📁 Saved {country} → {file_path}")

        if AWS_ENABLED and s3:
            key = f"{RESULT_PREFIX}/{iso}/news.json"
            try:
                s3.upload_file(file_path, RESULT_BUCKET, key)
                print(f"☁️ Uploaded → s3://{RESULT_BUCKET}/{key}")
            except Exception as e:
                print(f"❌ Upload failed: {e}")

def save_master(articles):
    history_dir = os.path.join(OUTPUT_DIR, "history")
    os.makedirs(history_dir, exist_ok=True)

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M")
    file_path = os.path.join(history_dir, f"news_{ts}.json")

    with open(file_path, "w") as f:
        json.dump(articles, f, indent=2)

    print(f"🧾 Master saved → {file_path}")

    if AWS_ENABLED and s3:
        key = f"{RESULT_PREFIX}/history/news_{ts}.json"
        try:
            s3.upload_file(file_path, RESULT_BUCKET, key)
            print(f"☁️ Master uploaded")
        except Exception as e:
            print(f"❌ Master upload failed: {e}")

# -------------------------------------
# MAIN
# -------------------------------------

async def run_all():
    queries = build_queries()
    seen = set()
    results = []

    print(f"🚀 Total Queries: {len(queries)}")

    async with aiohttp.ClientSession() as session:
        for i in range(0, len(queries), BATCH_SIZE):
            batch = queries[i:i+BATCH_SIZE]
            tasks = []

            print(f"🔄 Batch {i//BATCH_SIZE + 1}")

            for country, keyword, query, source_type in batch:
                domains = GLOBAL_SITES + COUNTRY_SITES.get(country, [])

                print(f"🔍 {query}")

                # 🔍 search
                res = await asyncio.to_thread(ddg_search, query)

                # 🧠 process articles
                for a in res:
                    tasks.append(
                        process(session, a, country, keyword, domains, seen, source_type)
                    )

            # ⚡ run async tasks
            out = await asyncio.gather(*tasks)

            valid = [x for x in out if x]
            results.extend(valid)

            print(f"   → Batch Valid: {len(valid)} | Total: {len(results)}\n")

    # ✅ IMPORTANT: keep ALL results, only deduplicate
    results = dedup(results)

    return results

def main():
    start = time.time()

    articles = asyncio.run(run_all())

    print("✅ FINAL:", len(articles))

    save_results(articles)
    save_master(articles)

    print("⏱ Time:", round(time.time() - start, 2), "sec")

if __name__ == "__main__":
    main()
