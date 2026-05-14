"""
test_scraper.py — Unit tests for newsapi_scraper.py
"""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.ingestion.newsapi_scraper import deduplicate, fetch_headlines, save_to_csv


# ── Fixtures ──────────────────────────────────────────────────────────────────

MOCK_ARTICLES = [
    {
        "source": {"name": "Reuters"},
        "author": "Jane Doe",
        "title": "Apple beats Q2 earnings expectations",
        "description": "Apple reported strong Q2 results...",
        "url": "https://reuters.com/apple",
        "publishedAt": "2024-05-01T10:00:00Z",
    },
    {
        "source": {"name": "Bloomberg"},
        "author": "John Smith",
        "title": "Fed raises rates by 25 basis points",
        "description": "The Federal Reserve raised interest rates...",
        "url": "https://bloomberg.com/fed",
        "publishedAt": "2024-05-01T11:00:00Z",
    },
    # Duplicate
    {
        "source": {"name": "Reuters"},
        "author": "Jane Doe",
        "title": "Apple beats Q2 earnings expectations",
        "description": "Duplicate article",
        "url": "https://reuters.com/apple-dup",
        "publishedAt": "2024-05-01T10:30:00Z",
    },
]


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestDeduplicate:
    def test_removes_duplicate_titles(self):
        unique = deduplicate(MOCK_ARTICLES)
        assert len(unique) == 2

    def test_empty_list(self):
        assert deduplicate([]) == []

    def test_all_unique(self):
        articles = [
            {"title": f"Headline {i}"} for i in range(5)
        ]
        assert len(deduplicate(articles)) == 5

    def test_filters_empty_titles(self):
        articles = [{"title": ""}, {"title": "   "}, {"title": "Valid headline"}]
        result = deduplicate(articles)
        assert len(result) == 1
        assert result[0]["title"] == "Valid headline"


class TestFetchHeadlines:
    def test_returns_articles_on_success(self):
        mock_client = MagicMock()
        mock_client.get_everything.return_value = {
            "articles": MOCK_ARTICLES[:2],
            "status": "ok",
        }
        cfg = {"newsapi": {"language": "en", "page_size": 100}}
        result = fetch_headlines(mock_client, "Apple", cfg, "2024-05-01")
        assert len(result) == 2

    def test_returns_empty_on_error(self):
        mock_client = MagicMock()
        mock_client.get_everything.side_effect = Exception("API error")
        cfg = {"newsapi": {"language": "en", "page_size": 100}}
        result = fetch_headlines(mock_client, "Apple", cfg, "2024-05-01")
        assert result == []


class TestSaveToCsv:
    def test_creates_file_with_correct_schema(self, tmp_path):
        output_path = tmp_path / "2024-05-01.csv"
        count = save_to_csv(MOCK_ARTICLES[:2], output_path)
        assert output_path.exists()
        df = pd.read_csv(output_path)
        assert set(df.columns) == {"date", "source", "author", "title", "description", "url", "publishedAt"}
        assert len(df) == 2
        assert count == 2

    def test_creates_parent_directory(self, tmp_path):
        output_path = tmp_path / "nested" / "dir" / "2024-05-01.csv"
        save_to_csv(MOCK_ARTICLES[:1], output_path)
        assert output_path.exists()

    def test_date_is_filename_stem(self, tmp_path):
        output_path = tmp_path / "2024-05-01.csv"
        save_to_csv(MOCK_ARTICLES[:1], output_path)
        df = pd.read_csv(output_path)
        assert df["date"].iloc[0] == "2024-05-01"
