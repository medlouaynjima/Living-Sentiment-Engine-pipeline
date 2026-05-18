"""
yfinance_scraper.py
───────────────────
Fetches real-time financial news for major tickers using Yahoo Finance.
Bypasses NewsAPI rate limits and provides high-quality market news.
Appends to data/raw/YYYY-MM-DD.csv.
"""

import argparse
import csv
import logging
from datetime import date, datetime
from pathlib import Path

import yfinance as yf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META"]


def fetch_yfinance_news(tickers: list[str]) -> list[dict]:
    articles = []
    seen_titles = set()
    for t in tickers:
        log.info("Fetching Yahoo Finance news for %s", t)
        try:
            ticker = yf.Ticker(t)
            news = ticker.news
            for item in news:
                title = item.get("title", "").strip()
                if not title or title in seen_titles:
                    continue
                seen_titles.add(title)
                
                pub_time = item.get("providerPublishTime", 0)
                pub_iso = datetime.fromtimestamp(pub_time).isoformat() if pub_time else ""
                
                articles.append({
                    "date": date.today().isoformat(),
                    "source": item.get("publisher", "Yahoo Finance"),
                    "author": "",
                    "title": title,
                    "description": item.get("summary", ""),
                    "url": item.get("link", ""),
                    "publishedAt": pub_iso,
                })
        except Exception as exc:
            log.warning("Could not fetch news for %s: %s", t, exc)
    return articles


def append_to_csv(articles: list[dict], output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["date", "source", "author", "title", "description", "url", "publishedAt"]
    
    file_exists = output_path.exists()
    
    existing_titles = set()
    if file_exists:
        with open(output_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_titles.add(row.get("title", ""))
                
    new_articles = [a for a in articles if a["title"] not in existing_titles]
    
    if not new_articles:
        return 0

    with open(output_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerows(new_articles)
        
    return len(new_articles)


def main():
    parser = argparse.ArgumentParser(description="Scrape Yahoo Finance news.")
    parser.add_argument("--date", default=str(date.today()))
    args = parser.parse_args()

    raw_dir = Path("data/raw")
    output_path = raw_dir / f"{args.date}.csv"

    articles = fetch_yfinance_news(TICKERS)
    if not articles:
        log.warning("No articles found via Yahoo Finance.")
        return

    count = append_to_csv(articles, output_path)
    log.info("Appended %d new Yahoo Finance headlines → %s", count, output_path)


if __name__ == "__main__":
    main()
