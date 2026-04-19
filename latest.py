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
ARTICLE_CONCURRENCY = 20

article_semaphore = asyncio.Semaphore(ARTICLE_CONCURRENCY)

# -------------------------------------
# EMBEDDINGS
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
    "investment foreign investment policy tax"
]

category_embeddings = model.encode(CATEGORY_LABELS, convert_to_tensor=True)

# -------------------------------------
# EMBEDDING HELPERS
# -------------------------------------

country_cache = {}
keyword_cache = {}

def get_country_emb(country):
    if country not in country_cache:
        country_cache[country] = get_embedding(f"{country} economy")
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
    if any(x in url for x in ["/opinion/", "/author/"]):
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
        for keyword in KEYWORDS:

            # GLOBAL
            queries.append((country, keyword, f"{country} {keyword} news", "global"))
            queries.append((country, keyword, f"{keyword} in {country}", "global"))

            for s in GLOBAL_SITES:
                queries.append((country, keyword, f"{country} {keyword} site:{s}", "global"))

            # COUNTRY
            for s in COUNTRY_SITES.get(country, []):
                queries.append((country, keyword, f"{country} {keyword} site:{s}", "country"))

    return queries

# -------------------------------------
# EXTRACTION
# -------------------------------------

async def fetch(session, url):
    try:
        async with session.get(url, timeout=10) as r:
            return await r.text()
    except:
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
# PROCESS (STRICT + SEMANTIC)
# -------------------------------------

async def process(session, art, country, keyword, seen, source_type):
    async with article_semaphore:

        url = art["url"]

        if not url or is_duplicate(url, seen):
            return None

        if not is_valid(art):
            return None

        domain = art["domain"]

        # 🔥 STRICT DOMAIN CONTROL
        if source_type == "global":
            if not any(g in domain for g in GLOBAL_SITES):
                return None

        if source_type == "country":
            if not any(c in domain for c in COUNTRY_SITES.get(country, [])):
                return None

        # DATE FILTER
        d = parse_date(art.get("date"))
        if not is_recent(d):
            return None

        content = await extract(session, url, art.get("snippet"))

        if not content or len(content.split()) < 120:
            return None

        emb = get_embedding(content[:500])

        # COUNTRY CHECK
        c_score = util.cos_sim(get_country_emb(country), emb).item()
        if c_score < 0.20:
            return None

        # KEYWORD CHECK
        score = util.cos_sim(get_keyword_emb(country, keyword), emb).item()
        if score < 0.30:
            return None

        # CATEGORY
        sims = util.cos_sim(emb, category_embeddings)[0]
        idx = sims.argmax().item()

        return {
            "country": country,
            "country_code": COUNTRY_CODE_MAP.get(country),
            "keyword": keyword,
            "category": CATEGORY_LABELS[idx],
            "title": art["title"],
            "url": url,
            "source": domain,
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
# SAVE (LOCAL + AWS)
# -------------------------------------

def save_results(articles):
    from collections import defaultdict
    import os
    import json

    grouped = defaultdict(list)

    # 🔹 Group articles by country
    for a in articles:
        grouped[a["country"]].append(a)

    # 🔹 Process each country separately
    for country, items in grouped.items():

        # ✅ Sort by score (best first)
        items = sorted(items, key=lambda x: x["score"], reverse=True)

        iso = COUNTRY_CODE_MAP.get(country, "XX")

        # 📁 Create folder: country_news/IN/
        folder = os.path.join(OUTPUT_DIR, iso)
        os.makedirs(folder, exist_ok=True)

        file_path = os.path.join(folder, "news.json")

        # 💾 Save JSON
        with open(file_path, "w") as f:
            json.dump(items, f, indent=2)

        print(f"📁 Saved {country} ({len(items)} articles) -> {file_path}")

        # ☁️ Upload to S3 (if enabled)
        if AWS_ENABLED and s3:
            key = f"{RESULT_PREFIX}/{iso}/news.json"
            try:
                s3.upload_file(file_path, RESULT_BUCKET, key)
                print(f"☁️ Uploaded to s3://{RESULT_BUCKET}/{key}")
            except Exception as e:
                print(f"❌ Upload failed for {country}: {e}")
                

def save_master(articles):
    import os
    import json
    from datetime import datetime

    # 📁 Create history folder
    history_dir = os.path.join(OUTPUT_DIR, "history")
    os.makedirs(history_dir, exist_ok=True)

    # 🕒 Timestamp
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M")

    file_name = f"news_{ts}.json"
    file_path = os.path.join(history_dir, file_name)

    # 💾 Save locally
    with open(file_path, "w") as f:
        json.dump(articles, f, indent=2)

    print(f"🧾 Master saved -> {file_path}")

    # ☁️ Upload to AWS (if enabled)
    if AWS_ENABLED and s3:
        s3_key = f"{RESULT_PREFIX}/history/{file_name}"

        try:
            s3.upload_file(file_path, RESULT_BUCKET, s3_key)
            print(f"☁️ Uploaded master to s3://{RESULT_BUCKET}/{s3_key}")
        except Exception as e:
            print(f"❌ Master upload failed: {e}")

# -------------------------------------
# MAIN
# -------------------------------------

async def run_all():
    queries = build_queries()
    seen = set()
    results = []

    async with aiohttp.ClientSession() as session:
        for country, keyword, query, source_type in queries:
            res = await asyncio.to_thread(ddg_search, query)

            tasks = [
                process(session, a, country, keyword, seen, source_type)
                for a in res
            ]

            out = await asyncio.gather(*tasks)
            results.extend([x for x in out if x])

    results = sorted(results, key=lambda x: x["score"], reverse=True)
    results = dedup(results)

    return dedup(results)

def main():
    start = time.time()

    articles = asyncio.run(run_all())

    print("✅ FINAL:", len(articles))

    save_results(articles)

    save_master(articles)

    print("⏱ Time:", round(time.time() - start, 2), "sec")

if __name__ == "__main__":
    main()
