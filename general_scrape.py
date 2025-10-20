"""
General social scrape → Perplexity triage

Pulls recent posts for arbitrary queries using your local scrapper.py (snscrape or X API fallback),
then asks Perplexity to highlight potentially market‑moving items and implicated tickers.

Usage examples:
  export PERPLEXITY_API_KEY="YOUR_PPLX_KEY"
  export X_BEARER_TOKEN="YOUR_X_BEARER"   # only needed with --use-api

  python general_scrape.py --queries "$AAPL,$NVDA,Apple,NVIDIA" --hours 6 --max 200 --top 10 --use-api
  python general_scrape.py --tickers "AAPL,NVDA" --names "Apple,NVIDIA" --hours 6 --top 12 --use-api

This script expects scrapper.py to be in the same directory.
"""
from __future__ import annotations

import os
import json
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional

import requests
import importlib.util

# --- Load local scraper module (scrapper.py) ---
SCRAPER_PATH = Path(__file__).with_name("scrapper.py")
if not SCRAPER_PATH.exists():
    raise RuntimeError("scrapper.py not found in this directory. Place general_scrape.py next to scrapper.py.")

spec = importlib.util.spec_from_file_location("scrapper", str(SCRAPER_PATH))
scraper = importlib.util.module_from_spec(spec)  # type: ignore
assert spec.loader is not None
spec.loader.exec_module(scraper)  # type: ignore


# --- Collection layer ---
def collect_posts_for_queries(
    queries: List[str],
    hours: int = 6,
    max_per_query: int = 300,
    top_n: int = 10,
    use_api: bool = False,
    bearer: Optional[str] = None,
) -> Dict[str, Any]:
    """Fetch recent posts for each query via scrapper.get_recent_x_mentions."""
    results_structs = []
    clean_queries = [q.strip() for q in queries if q and q.strip()]
    for q in clean_queries:
        summary = scraper.get_recent_x_mentions(
            q,
            window_hours=hours,
            max_tweets=max_per_query,
            top_n=top_n,
            use_api=use_api,
            bearer=bearer,
        )
        # summary is a dataclass (QueryTrendSummary) from scrapper.py
        results_structs.append(asdict(summary))

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "window_hours": hours,
        "queries": clean_queries,
        "results": results_structs,
    }


def _format_posts_for_prompt(collected: Dict[str, Any], per_query_cap: int = 10) -> str:
    """Compact the collected posts into a readable block for the LLM prompt."""
    lines: List[str] = []
    for idx, qres in enumerate(collected.get("results", [])):
        q = qres.get("query", "?")
        tops = (qres.get("top_tweets") or [])[:per_query_cap]
        lines.append(f"\n[Query {idx+1}] {q} — top {len(tops)} posts")
        for t in tops:
            date = (t.get("date", "")[:19]).replace("T", " ")
            user = t.get("username", "unknown")
            likes = t.get("like_count", 0)
            rts = t.get("retweet_count", 0)
            reps = t.get("reply_count", 0)
            url = t.get("url", "")
            content = (t.get("content", "") or "").replace("\n", " ").strip()
            if len(content) > 300:
                content = content[:297] + "..."
            lines.append(f"- [{date}] @{user} ♥{likes} ↻{rts} 💬{reps} — {content}\n  {url}")
    return "\n".join(lines)


def build_perplexity_prompt(collected: Dict[str, Any]) -> str:
    body = _format_posts_for_prompt(collected, per_query_cap=10)
    prompt = f"""
You are a finance-savvy analyst. Given recent public social posts, identify any items that could be **market-moving** or relevant to **public companies**.

For each distinct topic or company you detect, provide:
1) A concise **headline** (≤12 words)
2) The **companies/tickers** likely impacted (be explicit; use cashtags if possible)
3) A one-sentence **why it matters** (earnings, guidance, product issue, litigation, regulation, macro, etc.)
4) A **credibility flag**: High / Medium / Low (based on source quality, corroboration, and specificity)
5) A **risk note** if the content seems rumor/speculative or unclear
6) **Links** to the source posts that support the claim

Then finish with a **3-sentence takeaway** focused on the most actionable items.

Recent posts (window: {collected["window_hours"]}h, generated_at: {collected["generated_at"]}):
{body}
""".strip()
    return prompt


def analyze_with_perplexity(
    prompt: str,
    pplx_key: str,
    model: str = "sonar-reasoning",
    max_tokens: int = 1200,
    temperature: float = 0.2,
    endpoint: str = "https://api.perplexity.ai/chat/completions",
) -> str:
    headers = {"Authorization": f"Bearer {pplx_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    resp = requests.post(endpoint, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if data.get("choices"):
        return data["choices"][0]["message"]["content"]
    return "No analysis returned."


def parse_list_arg(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


def main():
    ap = argparse.ArgumentParser(description="General social scrape → Perplexity analysis (market impact).")
    ap.add_argument("--queries", help='Comma-separated queries (e.g., "$AAPL,$NVDA,Apple,NVIDIA")')
    ap.add_argument("--tickers", help='Comma-separated tickers (e.g., "AAPL,NVDA")')
    ap.add_argument("--names", help='Comma-separated company names aligned with --tickers (e.g., "Apple,NVIDIA")')
    ap.add_argument("--hours", type=int, default=6, help="Lookback window in hours (default: 6)")
    ap.add_argument("--max", type=int, default=300, help="Max posts per query to fetch (default: 300)")
    ap.add_argument("--top", type=int, default=10, help="Top-N posts per query to include (default: 10)")
    ap.add_argument("--use-api", action="store_true", help="Use X official API (requires bearer token)")
    ap.add_argument("--bearer", default=None, help="X API bearer (or set env X_BEARER_TOKEN)")
    ap.add_argument("--pplx-key", default=None, help="Perplexity API key (or set env PERPLEXITY_API_KEY)")
    ap.add_argument("--out", default="general_scrape_output.json", help="Path to save collected+analysis JSON")
    args = ap.parse_args()

    # Build the query list
    queries: List[str] = []
    queries += parse_list_arg(args.queries)

    tickers = parse_list_arg(args.tickers)
    names = parse_list_arg(args.names)
    for i, t in enumerate(tickers):
        queries.append(f"${t.upper()}")
        if i < len(names):
            queries.append(f'"{names[i]}"')

    if not queries:
        raise SystemExit("Provide --queries or --tickers/--names")

    bearer = args.bearer or os.getenv("X_BEARER_TOKEN")
    pplx_key = args.pplx_key or os.getenv("PERPLEXITY_API_KEY")
    if not pplx_key:
        raise SystemExit("Perplexity API key missing. Set PERPLEXITY_API_KEY or pass --pplx-key.")

    # 1) Collect
    collected = collect_posts_for_queries(
        queries=queries,
        hours=args.hours,
        max_per_query=args.max,
        top_n=args.top,
        use_api=args.use_api,
        bearer=bearer,
    )

    # 2) Build prompt
    prompt = build_perplexity_prompt(collected)

    # 3) Analyze
    try:
        analysis = analyze_with_perplexity(prompt, pplx_key)
    except Exception as e:
        analysis = f"Perplexity API error: {e}"

    # 4) Save artifacts
    artifact = {
        "inputs": {
            "queries": queries,
            "hours": args.hours,
            "max": args.max,
            "top": args.top,
            "use_api": args.use_api,
        },
        "collected": collected,
        "prompt_preview": prompt[:2000],  # keep JSON light
        "analysis": analysis,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(args.out, "w") as f:
        json.dump(artifact, f, indent=2)

    print("\n=== PERPLEXITY ANALYSIS ===\n")
    print(analysis)
    print(f"\nSaved output → {args.out}")


if __name__ == "__main__":
    main()
