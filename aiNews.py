# ai_payload.py
import json
import pandas as pd

def build_payload(company: str, df: pd.DataFrame, max_headlines: int = 12) -> dict:
    # Clean up first
    df = df.copy()
    df["published"] = pd.to_datetime(df["published"], errors="coerce")
    # Drop section/category pages with no publish time
    df = df.dropna(subset=["published"])
    # Keep most recent first
    df = df.sort_values("published", ascending=False)

    # Basic stats
    stats = {
        "company": company,
        "articles": int(len(df)),
        "most_recent": df["published"].max().isoformat() if not df.empty else None,
        "sentiment": {
            "avg": round(float(df["sentiment"].mean()), 3) if not df.empty else None,
            "min": round(float(df["sentiment"].min()), 3) if not df.empty else None,
            "max": round(float(df["sentiment"].max()), 3) if not df.empty else None,
        },
        "by_source": (
            df.groupby("source")["url"].count().sort_values(ascending=False).to_dict()
            if not df.empty else {}
        ),
    }

    # Top N headlines (recent first)
    headlines = df[["published", "source", "title", "url", "sentiment"]] \
                    .head(max_headlines) \
                    .to_dict(orient="records")

    return {"stats": stats, "headlines": headlines}


def build_prompt(company: str, payload: dict) -> str:
    """
    A compact instruction set for the LLM—short, structured, and grounded in the payload.
    """
    return f"""
You are a financial research assistant. Analyze recent third-party news scraped for {company}.
Use only the provided JSON facts. Do not invent sources.

Return:
1) A neutral, bullet-point summary of key themes & catalysts.
2) Sentiment read (overall & by source), note outliers.
3) Data quality flags (e.g., undated pages, duplicates).
4) Actionable next steps (what to verify, what to watch).
5) One-sentence TL;DR.

JSON:
{json.dumps(payload, ensure_ascii=False)}
    """.strip()