

"""
newsAnalyzer.py — Lightweight company news scraper (no site APIs)

Features
- Scrapes multiple popular news sites via public web pages (Reuters, CNBC, Yahoo Finance, The Verge)
- Uses requests + BeautifulSoup (no APIs) with retry/backoff and polite rate limiting
- Flexible search per company across sites
- Extracts canonical metadata (title, description, published time) and falls back to page text
- Simple sentiment analysis with VADER (vaderSentiment)
- Deduplicates, normalizes dates (dateparser), and outputs JSON/CSV
- CLI usage examples at bottom

Notes
- Respect each site's Terms of Use and robots.txt. This code is for educational/demo purposes.
- HTML can change; site-specific selectors may need updates over time.
- Some sites heavily use JS; this script targets pages that are server-rendered enough to parse.

Dependencies
    pip install requests beautifulsoup4 lxml dateparser pandas vaderSentiment

Python 3.9+
"""
from __future__ import annotations

import time
import re
import json
import hashlib
import random
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Iterable
from urllib.parse import urlencode, urljoin

import requests
from bs4 import BeautifulSoup
import dateparser
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DEFAULT_HEADERS_POOL = [
    {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Connection": "keep-alive",
    },
    {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:128.0) Gecko/20100101 Firefox/128.0",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Connection": "keep-alive",
    },
]


def build_session(timeout: int = 15) -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=0.6,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    s.mount("http://", HTTPAdapter(max_retries=retries))
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.request = _wrap_request_with_timeout(s.request, timeout)
    return s


def _wrap_request_with_timeout(orig_request, timeout):
    def wrapped(method, url, **kwargs):
        headers = kwargs.pop("headers", None) or random.choice(DEFAULT_HEADERS_POOL)
        kwargs["headers"] = headers
        kwargs.setdefault("timeout", timeout)
        return orig_request(method, url, **kwargs)
    return wrapped


# Data Model
@dataclass
class Article:
    source: str
    company: str
    title: str
    url: str
    published: Optional[str]
    description: Optional[str]
    snippet: Optional[str]
    sentiment: Optional[float]

    def key(self) -> str:
        return hashlib.sha256(self.url.encode("utf-8")).hexdigest()


META_TIME_PROPS = (
    "article:published_time",
    "og:pubdate",
    "pubdate",
    "og:updated_time",
    "timestamp",
)


def extract_metadata(soup: BeautifulSoup) -> Dict[str, Optional[str]]:
    """Extract common metadata from a generic article page."""
    def meta(prop=None, name=None):
        if prop:
            tag = soup.find("meta", attrs={"property": prop})
            if tag and tag.get("content"):
                return tag["content"].strip()
        if name:
            tag = soup.find("meta", attrs={"name": name})
            if tag and tag.get("content"):
                return tag["content"].strip()
        return None

    title = (
        meta(prop="og:title")
        or meta(name="twitter:title")
        or (soup.title.get_text(strip=True) if soup.title else None)
    )
    desc = meta(prop="og:description") or meta(name="description") or meta(name="twitter:description")

    raw_time = None
    for p in META_TIME_PROPS:
        raw_time = meta(prop=p) or meta(name=p)
        if raw_time:
            break

    published_iso = None
    if raw_time:
        dt = dateparser.parse(raw_time)
        if dt:
            published_iso = dt.isoformat()

    first_p = None
    p_tag = soup.find("p")
    if p_tag:
        first_p = p_tag.get_text(" ", strip=True)

    return {
        "title": title,
        "description": desc,
        "published": published_iso,
        "snippet": first_p,
    }


# Web Scrappers
class BaseSite:
    BASE_URL: str = ""
    NAME: str = "base"

    def __init__(self, session: requests.Session, pause: float = 1.2):
        self.s = session
        self.pause = pause

    def search(self, company: str) -> List[str]:
        """Return list of article URLs for the company."""
        raise NotImplementedError

    def parse_article(self, url: str) -> Dict[str, Optional[str]]:
        r = self.s.get(url)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "lxml")
        return extract_metadata(soup)

    def polite_pause(self):
        time.sleep(self.pause + random.random() * 0.6)


class ReutersSite(BaseSite):
    BASE_URL = "https://www.reuters.com"
    NAME = "Reuters"

    def search(self, company: str) -> List[str]:
        # Reuters site search
        qs = urlencode({"query": company})
        url = f"{self.BASE_URL}/site-search/?{qs}"
        r = self.s.get(url)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "lxml")
        links = []
        for a in soup.select("a[href]"):
            href = a.get("href")
            if not href:
                continue
            # Normalize relative links
            if href.startswith("/"):
                href = urljoin(self.BASE_URL, href)
            # Filter obvious article paths
            if "/business" in href or "/markets" in href or "/technology" in href:
                if href not in links:
                    links.append(href)
        self.polite_pause()
        return links[:15]


class CNBCSite(BaseSite):
    BASE_URL = "https://www.cnbc.com"
    NAME = "CNBC"

    def search(self, company: str) -> List[str]:
        qs = urlencode({"query": company, "department": "news"})
        url = f"{self.BASE_URL}/search/?{qs}"
        r = self.s.get(url)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "lxml")
        links = []
        # Results anchors often have data-test or article classes
        for a in soup.select("a.Card-title, a[href*='/202']"):
            href = a.get("href")
            if not href:
                continue
            if href.startswith("/"):
                href = urljoin(self.BASE_URL, href)
            if href not in links and "/video/" not in href:
                links.append(href)
        self.polite_pause()
        return links[:15]


class YahooFinanceSite(BaseSite):
    BASE_URL = "https://finance.yahoo.com"
    NAME = "Yahoo Finance"

    def search(self, company: str) -> List[str]:
        # If the user passes a ticker (e.g., AAPL), we can hit the ticker news tab; else fallback to site search
        # Heuristic: 1-5 uppercase letters/digits considered ticker-like
        is_ticker = bool(re.fullmatch(r"[A-Z]{1,5}\d?", company.strip()))
        if is_ticker:
            url = f"{self.BASE_URL}/quote/{company}/news?p={company}"
        else:
            q = urlencode({"q": company})
            url = f"{self.BASE_URL}/screener/news?{q}"
        r = self.s.get(url)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "lxml")
        links = []
        for a in soup.select("a[href]"):
            href = a.get("href")
            if not href:
                continue
            if href.startswith("/"):
                href = urljoin(self.BASE_URL, href)
            # Yahoo aggregates many sources; accept plausible news paths
            if any(seg in href for seg in ["/news/", "/finance/news/"]):
                if href not in links:
                    links.append(href)
        self.polite_pause()
        return links[:20]


class TheVergeSite(BaseSite):
    BASE_URL = "https://www.theverge.com"
    NAME = "The Verge"

    def search(self, company: str) -> List[str]:
        qs = urlencode({"q": company})
        url = f"{self.BASE_URL}/search?{qs}"
        r = self.s.get(url)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "lxml")
        links = []
        for a in soup.select("a[href*='/']"):
            href = a.get("href")
            if not href:
                continue
            if href.startswith("/"):
                href = urljoin(self.BASE_URL, href)
            # Filter to recent-ish article paths that include year
            if re.search(r"/20\d{2}/\d{1,2}/\d{1,2}/", href):
                if href not in links:
                    links.append(href)
        self.polite_pause()
        return links[:15]


# ----------- Additional Sites: MarketWatch, WSJ, Barron's -----------
class MarketWatchSite(BaseSite):
    BASE_URL = "https://www.marketwatch.com"
    NAME = "MarketWatch"

    def search(self, company: str) -> List[str]:
        qs = urlencode({"q": company})
        url = f"{self.BASE_URL}/search/?{qs}"
        r = self.s.get(url)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "lxml")
        links: List[str] = []
        for a in soup.select("a[href]"):
            href = a.get("href")
            if not href:
                continue
            if href.startswith("/"):
                href = urljoin(self.BASE_URL, href)
            if "/story/" in href and href not in links:
                links.append(href)
        self.polite_pause()
        return links[:15]


class WSJSite(BaseSite):
    BASE_URL = "https://www.wsj.com"
    NAME = "WSJ"

    def search(self, company: str) -> List[str]:
        qs = urlencode({"query": company})
        url = f"{self.BASE_URL}/search?{qs}"
        r = self.s.get(url)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "lxml")
        links: List[str] = []
        for a in soup.select("a[href]"):
            href = a.get("href")
            if not href:
                continue
            if href.startswith("/"):
                href = urljoin(self.BASE_URL, href)
            if href.startswith(self.BASE_URL) and "/articles/" in href and href not in links:
                links.append(href)
        self.polite_pause()
        return links[:12]


class BarronsSite(BaseSite):
    BASE_URL = "https://www.barrons.com"
    NAME = "Barron's"

    def search(self, company: str) -> List[str]:
        qs = urlencode({"keyword": company})
        url = f"{self.BASE_URL}/search?{qs}"
        r = self.s.get(url)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "lxml")
        links: List[str] = []
        for a in soup.select("a[href]"):
            href = a.get("href")
            if not href:
                continue
            if href.startswith("/"):
                href = urljoin(self.BASE_URL, href)
            if href.startswith(self.BASE_URL) and "/articles/" in href and href not in links:
                links.append(href)
        self.polite_pause()
        return links[:12]


# ------------------------- Analyzer ------------------------- #

# Helper to identify likely article URLs (not category/home pages)
def _looks_like_article_url(url: str) -> bool:
    if re.search(r"/20\d{2}/\d{1,2}/\d{1,2}/", url):
        return True
    keywords = ["/article/", "/articles/", "/story/", "/business/", "/markets/"]
    return any(k in url for k in keywords)


class NewsAnalyzer:
    def __init__(self, sites: Optional[List[BaseSite]] = None, pause: float = 1.2):
        self.session = build_session()
        self.pause = pause
        self.analyzer = SentimentIntensityAnalyzer()
        self.sites: List[BaseSite] = sites or [
            ReutersSite(self.session, pause),
            CNBCSite(self.session, pause),
            YahooFinanceSite(self.session, pause),
            MarketWatchSite(self.session, pause),
            WSJSite(self.session, pause),
            BarronsSite(self.session, pause),
            TheVergeSite(self.session, pause),
        ]

    def _score_sentiment(self, text: str) -> float:
        if not text:
            return 0.0
        return float(self.analyzer.polarity_scores(text)["compound"])

    def _dedupe_urls(self, articles: Iterable[Article]) -> List[Article]:
        seen = set()
        out: List[Article] = []
        for a in articles:
            if a.url in seen:
                continue
            seen.add(a.url)
            out.append(a)
        return out

    def search_company(self, company: str, per_site_limit: int = 8) -> List[Article]:
        collected: List[Article] = []
        for site in self.sites:
            try:
                urls = site.search(company)[:per_site_limit]
            except Exception as e:
                print(f"[warn] {site.NAME} search failed for '{company}': {e}")
                continue

            for url in urls:
                try:
                    meta = site.parse_article(url)
                    if not _looks_like_article_url(url):
                        continue
                    title = meta.get("title") or ""
                    desc = meta.get("description")
                    snippet = meta.get("snippet")
                    combined = " ".join([t for t in [title, desc, snippet] if t])
                    sent = self._score_sentiment(combined)
                    art = Article(
                        source=site.NAME,
                        company=company,
                        title=title,
                        url=url,
                        published=meta.get("published"),
                        description=desc,
                        snippet=snippet,
                        sentiment=sent,
                    )
                    collected.append(art)
                except Exception as e:
                    print(f"[warn] Failed to parse article: {url} — {e}")
                finally:
                    time.sleep(0.8 + random.random() * 0.5)
        return self._dedupe_urls(collected)

    @staticmethod
    def to_dataframe(articles: List[Article]) -> pd.DataFrame:
        df = pd.DataFrame([asdict(a) for a in articles])
        # Sort by published (desc) when available
        def _to_ts(x):
            try:
                return pd.to_datetime(x)
            except Exception:
                return pd.NaT
        if not df.empty:
            df["published_ts"] = df["published"].apply(_to_ts)
            df = df.sort_values(["published_ts"], ascending=[False])
        return df

    @staticmethod
    def save_json(articles: List[Article], path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump([asdict(a) for a in articles], f, ensure_ascii=False, indent=2)

    @staticmethod
    def save_csv(articles: List[Article], path: str) -> None:
        df = NewsAnalyzer.to_dataframe(articles)
        df.drop(columns=["published_ts"], errors="ignore").to_csv(path, index=False)


# ------------------------- CLI ------------------------- #

# ------------------------- CLI ------------------------- #

def print_company_summary(company: str, articles: List[Article]) -> None:
    print(f"\n===== Summary for {company} =====")
    if not articles:
        print("No articles found.\n")
        return
    df = NewsAnalyzer.to_dataframe(articles)
    # Working Sentiment Analyase (do not touch)
    avg_sent = round(float(df["sentiment"].mean()), 3) if not df["sentiment"].isna().all() else 0.0
    min_sent = round(float(df["sentiment"].min()), 3) if not df["sentiment"].isna().all() else 0.0
    max_sent = round(float(df["sentiment"].max()), 3) if not df["sentiment"].isna().all() else 0.0
    most_recent = df["published"].dropna().iloc[0] if df["published"].dropna().shape[0] else "Unknown"
    by_source = df.groupby("source")["url"].count().sort_values(ascending=False)

    print(f"Articles: {len(df)}")
    print(f"Most recent publish time: {most_recent}")
    print(f"Sentiment — avg: {avg_sent}, min: {min_sent}, max: {max_sent}")
    print("Sources:")
    for s, c in by_source.items():
        print(f"  - {s}: {c}")

    # Show top 10 headlines (simplicity purposes)
    print("\nTop headlines:")
    show = df[["published", "source", "title", "url", "sentiment"]].head(10)
    with pd.option_context('display.max_colwidth', 100):
        print(show.to_string(index=False))
    print()


def prompt_for_companies_if_empty(cli_companies: List[str]) -> List[str]:
    if cli_companies:
        return cli_companies
    try:
        raw = input("Enter a ticker or company name (comma-separated for multiple): ").strip()
    except EOFError:
        raw = ""
    comps = [c.strip() for c in raw.split(',') if c.strip()]
    return comps

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Scrape company news from several public sites (no APIs)")
    parser.add_argument("companies", nargs="*", help="Company names or tickers, e.g., AAPL MSFT 'NVIDIA' ")
    parser.add_argument("--limit", type=int, default=8, help="Max articles per site per company")
    parser.add_argument("--out-json", default=None, help="Path to write JSON results")
    parser.add_argument("--out-csv", default=None, help="Path to write CSV results")
    parser.add_argument("--pause", type=float, default=1.2, help="Polite per-site pause seconds")

    args = parser.parse_args()

    analyzer = NewsAnalyzer(pause=args.pause)
    companies = prompt_for_companies_if_empty(args.companies)
    if not companies:
        print("No companies provided. Exiting.")
        raise SystemExit(0)

    all_articles: List[Article] = []

    for comp in companies:
        print(f"\n[info] Searching: {comp}")
        arts = analyzer.search_company(comp, per_site_limit=args.limit)
        print(f"[info] Found {len(arts)} items for {comp}")
        print_company_summary(comp, arts)
        all_articles.extend(arts)

    # Overall table (across all companies) for quick glance
    df = NewsAnalyzer.to_dataframe(all_articles)
    if not df.empty:
        print("\nAll results (first 20):")
        with pd.option_context('display.max_colwidth', 100):
            print(df[["published", "source", "company", "title", "url", "sentiment"]].head(20).to_string(index=False))

    if args.out_json:
        NewsAnalyzer.save_json(all_articles, args.out_json)
        print(f"[info] Wrote JSON -> {args.out_json}")
    if args.out_csv:
        NewsAnalyzer.save_csv(all_articles, args.out_csv)
        print(f"[info] Wrote CSV  -> {args.out_csv}")

    print("\nDone.")