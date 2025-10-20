import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Any
import requests

# Third‑party: snscrape (no official X API required)
try:
    import snscrape.modules.twitter as sntwitter  # still lives under 'twitter' in snscrape
except Exception as e:
    print("snscrape is not installed. Install with: pip install snscrape", file=sys.stderr)
    raise


# -----------------------------
# Data models
# -----------------------------
@dataclass
class TweetBrief:
    date: str
    username: str
    content: str
    like_count: int
    retweet_count: int
    reply_count: int
    url: str

@dataclass
class QueryTrendSummary:
    query: str
    window_hours: int
    total_tweets: int
    unique_authors: int
    total_likes: int
    total_retweets: int
    total_replies: int
    top_tweets: List[TweetBrief]


# -----------------------------
# Helpers
# -----------------------------
_DEF_MAX_TWEETS = 500
_DEF_TOP = 10

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)

def _build_since_str(hours: int) -> tuple[datetime, str]:
    since_dt = _now_utc() - timedelta(hours=hours)
    # snscrape's `since:` filter is date‑only; we still hard‑filter by dt afterward
    since_str = since_dt.strftime('%Y-%m-%d')
    return since_dt, since_str


def _pick_search_scraper():
    """Return the appropriate search scraper class depending on snscrape version."""
    # Older versions
    if hasattr(sntwitter, 'TwitterSearchScraper'):
        return sntwitter.TwitterSearchScraper
    # Some builds may expose XSearchScraper in the future
    if hasattr(sntwitter, 'XSearchScraper'):
        return getattr(sntwitter, 'XSearchScraper')
    # Fallback: raise with guidance
    raise RuntimeError(
        "Your snscrape build doesn't have a search scraper. Try: \n"
        "  pip install --upgrade snscrape\n"
        "or to get the latest fixes directly: \n"
        "  pip install --upgrade git+https://github.com/JustAnotherArchivist/snscrape.git"
    )


def _x_api_recent_search(query: str, start_time: datetime, end_time: datetime, max_results_total: int, top_n: int, bearer: str) -> QueryTrendSummary:
    """Fallback using the official X API v2 Recent Search.
    Requires a Bearer token (set env X_BEARER_TOKEN or pass via --bearer).
    """
    if bearer is None or not bearer.strip():
        raise RuntimeError("X API bearer token not provided.")

    # API endpoints (try api.x.com first; fallback to api.twitter.com)
    endpoints = [
        "https://api.x.com/2/tweets/search/recent",
        "https://api.twitter.com/2/tweets/search/recent",
    ]

    headers = {"Authorization": f"Bearer {bearer}"}
    params = {
        "query": query,
        "start_time": start_time.isoformat().replace("+00:00", "Z"),
        "end_time": end_time.isoformat().replace("+00:00", "Z"),
        "max_results": 100,  # per-request cap
        "tweet.fields": "created_at,public_metrics,entities",
        "expansions": "author_id",
        "user.fields": "username",
        # Note: We could add `sort_order=recency` if/when supported
    }

    collected = []
    users_map: Dict[str, str] = {}
    next_token = None

    while len(collected) < max_results_total:
        if next_token:
            params["next_token"] = next_token
        else:
            params.pop("next_token", None)

        last_err = None
        resp = None
        for url in endpoints:
            try:
                resp = requests.get(url, headers=headers, params=params, timeout=30)
                # Some orgs proxy api.x.com to api.twitter.com; accept 200 only
                if resp.status_code == 200:
                    break
            except requests.RequestException as re:
                last_err = re
                resp = None
        if resp is None or resp.status_code != 200:
            detail = resp.text if resp is not None else str(last_err)
            raise RuntimeError(f"X API recent search failed: {detail}")

        data = resp.json()
        batch = data.get("data", [])
        includes = data.get("includes", {})
        users = includes.get("users", [])
        for u in users:
            if "id" in u and "username" in u:
                users_map[u["id"]] = u["username"]

        for tw in batch:
            pm = tw.get("public_metrics", {})
            created_at = tw.get("created_at")
            # Parse ISO and enforce window even if API returns extras
            try:
                tdate = datetime.fromisoformat(created_at.replace("Z", "+00:00")) if created_at else None
            except Exception:
                tdate = None
            if tdate is None or tdate < start_time or tdate > end_time:
                continue

            username = users_map.get(tw.get("author_id", ""), "unknown")
            url = f"https://x.com/{username}/status/{tw.get('id')}"
            collected.append(
                TweetBrief(
                    date=tdate.isoformat(),
                    username=username,
                    content=(tw.get("text", "")[:500]),
                    like_count=int(pm.get("like_count", 0) or 0),
                    retweet_count=int(pm.get("retweet_count", 0) or 0),
                    reply_count=int(pm.get("reply_count", 0) or 0),
                    url=url,
                )
            )

        meta = data.get("meta", {})
        next_token = meta.get("next_token")
        if not next_token:
            break

    # Aggregate
    authors = {t.username for t in collected}
    total_likes = sum(t.like_count for t in collected)
    total_retweets = sum(t.retweet_count for t in collected)
    total_replies = sum(t.reply_count for t in collected)
    top = sorted(collected, key=lambda t: t.like_count, reverse=True)[: top_n]

    return QueryTrendSummary(
        query=query,
        window_hours=int((end_time - start_time).total_seconds() // 3600) or 1,
        total_tweets=len(collected),
        unique_authors=len(authors),
        total_likes=total_likes,
        total_retweets=total_retweets,
        total_replies=total_replies,
        top_tweets=top,
    )


# -----------------------------
# Core scraping
# -----------------------------

def get_recent_x_mentions(query: str, window_hours: int = 24, max_tweets: int = _DEF_MAX_TWEETS, top_n: int = _DEF_TOP, use_api: bool = False, bearer: Optional[str] = None) -> QueryTrendSummary:
    """
    Fetch recent X posts for a query (cashtag or quoted name) within the last `window_hours`.
    Aggregates engagement and returns the top `top_n` posts by likes.
    """
    SearchScraper = _pick_search_scraper()
    since_dt, since_str = _build_since_str(window_hours)

    search = f"{query} since:{since_str}"

    tweets: List[TweetBrief] = []
    authors: set[str] = set()
    total_likes = total_retweets = total_replies = 0

    try:
        it = SearchScraper(search).get_items()
        for i, tw in enumerate(it):
            if i >= max_tweets:
                break
            # Ensure datetime and time window
            tdate = getattr(tw, 'date', None)
            if tdate is None:
                continue
            if tdate.tzinfo is None:
                tdate = tdate.replace(tzinfo=timezone.utc)
            if tdate < since_dt:
                continue

            username = getattr(tw.user, 'username', 'unknown') if getattr(tw, 'user', None) else 'unknown'
            content = getattr(tw, 'rawContent', '') or getattr(tw, 'content', '') or ''
            url = getattr(tw, 'url', '') or ''
            like_count = int(getattr(tw, 'likeCount', 0) or 0)
            retweet_count = int(getattr(tw, 'retweetCount', 0) or 0)
            reply_count = int(getattr(tw, 'replyCount', 0) or 0)

            authors.add(username)
            total_likes += like_count
            total_retweets += retweet_count
            total_replies += reply_count

            tweets.append(
                TweetBrief(
                    date=tdate.isoformat(),
                    username=username,
                    content=content[:500],
                    like_count=like_count,
                    retweet_count=retweet_count,
                    reply_count=reply_count,
                    url=url,
                )
            )

    except Exception as e:
        # If snscrape fails and API fallback is enabled, try the official API
        if use_api:
            start_dt = _now_utc() - timedelta(hours=window_hours)
            end_dt = _now_utc()
            return _x_api_recent_search(query, start_dt, end_dt, max_tweets, top_n, bearer or os.getenv("X_BEARER_TOKEN", ""))
        _msg = (
            f"snscrape failed for query '{query}': {e}\n\n"
            "This often happens when X/Twitter changes their internal endpoints. Try:\n"
            "  1) pip install --upgrade snscrape\n"
            "  2) pip install --upgrade git+https://github.com/JustAnotherArchivist/snscrape.git\n"
            "  3) If scraping remains blocked, pass --use-api and provide X API bearer via --bearer or env X_BEARER_TOKEN."
        )
        raise RuntimeError(_msg)

    # Rank top tweets by likes
    top = sorted(tweets, key=lambda t: t.like_count, reverse=True)[: top_n]

    return QueryTrendSummary(
        query=query,
        window_hours=window_hours,
        total_tweets=len(tweets),
        unique_authors=len(authors),
        total_likes=total_likes,
        total_retweets=total_retweets,
        total_replies=total_replies,
        top_tweets=top,
    )


def social_trend_for_company(
    ticker: str,
    company_name: Optional[str] = None,
    window_hours: int = 24,
    max_tweets_per_query: int = _DEF_MAX_TWEETS,
    top_n: int = _DEF_TOP,
    use_api: bool = False,
    bearer: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build a trend signal for a company by querying:
      1) its cashtag (e.g., $AAPL)
      2) its quoted company name (e.g., "Apple") — optional, if provided
    """
    cashtag = f"${ticker.upper()}"
    queries = [cashtag]
    if company_name:
        queries.append(f'"{company_name}"')

    summaries: List[QueryTrendSummary] = []
    agg = {"total_tweets": 0, "total_likes": 0, "total_retweets": 0, "total_replies": 0}

    for q in queries:
        s = get_recent_x_mentions(q, window_hours=window_hours, max_tweets=max_tweets_per_query, top_n=top_n, use_api=use_api, bearer=bearer)
        summaries.append(s)
        agg["total_tweets"] += s.total_tweets
        agg["total_likes"] += s.total_likes
        agg["total_retweets"] += s.total_retweets
        agg["total_replies"] += s.total_replies

    return {
        "ticker": ticker.upper(),
        "window_hours": window_hours,
        "queries": [asdict(s) for s in summaries],
        "aggregate": agg,
        "generated_at": _now_utc().isoformat(),
        "source": "x/snscrape",
    }


# -----------------------------
# Presentation
# -----------------------------

def pretty_print_trend(trend: Dict[str, Any]) -> str:
    lines = []
    lines.append(f"SOCIAL TREND (X) • {trend['ticker']} • last {trend['window_hours']}h")
    agg = trend["aggregate"]
    lines.append(
        f"Aggregate: tweets={agg['total_tweets']:,}  likes={agg['total_likes']:,}  "
        f"retweets={agg['total_retweets']:,}  replies={agg['total_replies']:,}"
    )
    for q in trend["queries"]:
        lines.append(
            f"\nQuery: {q['query']}  |  tweets={q['total_tweets']:,}  authors≈{q['unique_authors']:,}  "
            f"likes={q['total_likes']:,}  retweets={q['total_retweets']:,}  replies={q['total_replies']:,}"
        )
        tops = q.get("top_tweets", [])
        if tops:
            lines.append("Top posts by likes:")
            for t in tops[:5]:
                lines.append(
                    f"  • @{t['username']}  ♥{t['like_count']} ↻{t['retweet_count']} 💬{t['reply_count']}  {t['url']}"
                )
    return "\n".join(lines)


# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Fetch recent X (Twitter) chatter for a ticker/company using snscrape.")
    ap.add_argument("--ticker", required=True, help="Ticker symbol, e.g., AAPL")
    ap.add_argument("--name", default=None, help="Optional company name in quotes, e.g., Apple")
    ap.add_argument("--hours", type=int, default=24, help="Lookback window in hours (default: 24)")
    ap.add_argument("--max", type=int, default=_DEF_MAX_TWEETS, help="Max tweets per query (default: 500)")
    ap.add_argument("--top", type=int, default=_DEF_TOP, help="Top-N posts to keep (default: 10)")
    ap.add_argument("--use-api", action="store_true", help="Use official X API v2 Recent Search as fallback if snscrape fails (requires bearer)")
    ap.add_argument("--bearer", default=None, help="X API Bearer token; if omitted, uses env X_BEARER_TOKEN")

    args = ap.parse_args()

    try:
        trend = social_trend_for_company(
            ticker=args.ticker,
            company_name=args.name,
            window_hours=args.hours,
            max_tweets_per_query=args.max,
            top_n=args.top,
            use_api=args.use_api,
            bearer=(args.bearer or os.getenv("X_BEARER_TOKEN")),
        )
        print(pretty_print_trend(trend))
    except Exception as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

# --- END ---
