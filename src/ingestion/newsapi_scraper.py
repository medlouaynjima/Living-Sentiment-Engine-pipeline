"""
newsapi_scraper.py
──────────────────
Fetches financial / tech news headlines via NewsAPI and saves them to
data/raw/YYYY-MM-DD.csv.  Idempotent: skips a date if the file already
exists unless --force is passed.

Usage:
    python src/ingestion/newsapi_scraper.py               # today
    python src/ingestion/newsapi_scraper.py --date 2024-05-01
    python src/ingestion/newsapi_scraper.py --force       # overwrite
"""

import argparse
import csv
import logging
import os
import sys
from datetime import date
from pathlib import Path

import yaml
from newsapi import NewsApiClient

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Helpers ─────────────────────────────────────────────────────────────────

def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Resolve env-var placeholder for API key
    api_key = cfg["newsapi"]["api_key"]
    if api_key.startswith("${") and api_key.endswith("}"):
        env_var = api_key[2:-1]
        api_key = os.environ.get(env_var, "")
        if not api_key:
            raise EnvironmentError(
                f"Environment variable '{env_var}' is not set. "
                "Export it before running the scraper."
            )
        cfg["newsapi"]["api_key"] = api_key

    return cfg


def fetch_top_headlines(client: NewsApiClient, keyword: str, cfg: dict) -> list[dict]:
    """
    Use get_top_headlines — works in real-time on the free NewsAPI tier.
    No date filter required; returns the most current headlines.
    """
    try:
        response = client.get_top_headlines(
            q=keyword,
            language=cfg["newsapi"]["language"],
            page_size=min(cfg["newsapi"]["page_size"], 100),
        )
        articles = response.get("articles", [])
        log.info("  [top-headlines] keyword=%-20s  articles=%d", keyword, len(articles))
        return articles
    except Exception as exc:
        log.warning("  [top-headlines] keyword=%-20s  ERROR: %s", keyword, exc)
        return []


def fetch_everything_window(client: NewsApiClient, keyword: str, cfg: dict, days_back: int = 7) -> list[dict]:
    """
    Fallback: get_everything with a rolling 7-day window.
    Free tier allows up to 1 month old articles via this endpoint.
    """
    from datetime import timedelta
    from_date = (date.today() - timedelta(days=days_back)).isoformat()
    try:
        response = client.get_everything(
            q=keyword,
            from_param=from_date,
            language=cfg["newsapi"]["language"],
            sort_by="publishedAt",
            page_size=cfg["newsapi"]["page_size"],
        )
        articles = response.get("articles", [])
        log.info("  [everything]    keyword=%-20s  articles=%d", keyword, len(articles))
        return articles
    except Exception as exc:
        log.warning("  [everything]    keyword=%-20s  ERROR: %s", keyword, exc)
        return []


def deduplicate(articles: list[dict]) -> list[dict]:
    seen_titles = set()
    unique = []
    for art in articles:
        title = (art.get("title") or "").strip()
        if title and title not in seen_titles:
            seen_titles.add(title)
            unique.append(art)
    return unique


def save_to_csv(articles: list[dict], output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "date", "source", "author", "title", "description",
        "url", "publishedAt",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for art in articles:
            writer.writerow(
                {
                    "date": output_path.stem,          # YYYY-MM-DD from filename
                    "source": (art.get("source") or {}).get("name", ""),
                    "author": art.get("author", ""),
                    "title": art.get("title", ""),
                    "description": art.get("description", ""),
                    "url": art.get("url", ""),
                    "publishedAt": art.get("publishedAt", ""),
                }
            )
    return len(articles)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Scrape NewsAPI headlines.")
    parser.add_argument("--date", default=str(date.today()), help="Target date YYYY-MM-DD (used for output filename)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing file")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config.yaml")
    parser.add_argument("--days-back", type=int, default=7, help="Days back for get_everything window (default: 7)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    target_date = args.date

    raw_dir = Path(cfg["data"]["raw_dir"])
    output_path = raw_dir / f"{target_date}.csv"

    if output_path.exists() and not args.force:
        log.info("File already exists: %s  (use --force to overwrite)", output_path)
        sys.exit(0)

    log.info("Scraping headlines for %s …", target_date)
    client = NewsApiClient(api_key=cfg["newsapi"]["api_key"])

    all_articles: list[dict] = []
    keywords = cfg["newsapi"]["keywords"]

    # ── Phase 1: Top Headlines (real-time, free tier) ─────────────────────────
    log.info("Phase 1: get_top_headlines (real-time) …")
    for keyword in keywords:
        articles = fetch_top_headlines(client, keyword, cfg)
        all_articles.extend(articles)

    # ── Phase 2: Everything with rolling window (broader coverage) ────────────
    log.info("Phase 2: get_everything (last %d days) …", args.days_back)
    for keyword in keywords:
        articles = fetch_everything_window(client, keyword, cfg, days_back=args.days_back)
        all_articles.extend(articles)

    unique_articles = deduplicate(all_articles)
    log.info("Total unique headlines after dedup: %d", len(unique_articles))

    if not unique_articles:
        log.warning(
            "No articles found. Check your NEWSAPI_KEY or try --days-back 30 for a wider window."
        )
        sys.exit(0)

    count = save_to_csv(unique_articles, output_path)
    log.info("Saved %d headlines → %s", count, output_path)


if __name__ == "__main__":
    main()

