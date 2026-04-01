#!/usr/bin/env python3
"""
Fetch resolved multi-outcome forecasting questions from Kalshi's public API.

Usage:
    python data/kalshi/fetch_kalshi.py \
        --output_dir /fast/nchandak/forecast-sim/data/kalshi \
        --months_back 3
"""

import argparse
import json
import re
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
REQUEST_DELAY = 0.5  # seconds between API calls


def api_get(endpoint: str, params: dict = None) -> dict:
    """Make a GET request to the Kalshi API."""
    url = f"{BASE_URL}/{endpoint}"
    if params:
        query = "&".join(f"{k}={v}" for k, v in params.items() if v is not None)
        url += f"?{query}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def looks_recent(ticker: str, years: list[str]) -> bool:
    """Check if an event ticker contains a recent year indicator (e.g., '26', '25')."""
    # Tickers like KXDENMARK3RD-26MAR24-3 have year encoded
    for y in years:
        if f"-{y}" in ticker:
            return True
    return False


def fetch_settled_events(cutoff_years: list[str], max_pages: int = 100) -> list[dict]:
    """Paginate through settled events, return mutually_exclusive ones with recent tickers."""
    events = []
    cursor = ""
    for page in range(max_pages):
        params = {"status": "settled", "limit": "200"}
        if cursor:
            params["cursor"] = cursor
        time.sleep(REQUEST_DELAY)
        data = api_get("events", params)
        batch = data.get("events", [])
        for e in batch:
            if e.get("mutually_exclusive"):
                events.append(e)
        cursor = data.get("cursor", "")
        if not cursor or not batch:
            break
        if (page + 1) % 5 == 0:
            print(f"  Scanned {(page+1)*200} events, found {len(events)} ME events so far...", flush=True)
    return events


def fetch_event_markets(event_ticker: str) -> tuple[dict, list[dict]]:
    """Fetch event detail with nested markets."""
    time.sleep(REQUEST_DELAY)
    data = api_get(f"events/{event_ticker}")
    return data.get("event", {}), data.get("markets", [])


def parse_date(ts: str) -> str:
    """Extract YYYY-MM-DD from an ISO timestamp string."""
    if not ts or ts.startswith("0001"):
        return ""
    return ts[:10]


def has_date_reference(text: str) -> bool:
    """Check if text contains a date-like reference."""
    date_patterns = [
        r"\b\d{4}\b",                          # year like 2026
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d",  # "March 3"
        r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",        # MM/DD/YYYY
    ]
    return any(re.search(p, text, re.IGNORECASE) for p in date_patterns)


def build_question(event: dict, markets: list[dict]) -> dict | None:
    """Build a question record from an event and its markets. Returns None if invalid."""
    if len(markets) < 2:
        return None

    # Determine winner
    winners = [m for m in markets if m.get("result") == "yes"]
    if winners:
        answer = winners[0].get("yes_sub_title", "")
    else:
        # All "no" — winner is the "Other" catch-all via expiration_value
        exp_vals = [m.get("expiration_value", "") for m in markets
                    if m.get("expiration_value") and m["expiration_value"].lower() not in ("no", "yes", "")]
        answer = exp_vals[0] if exp_vals else ""

    if not answer:
        return None

    # Dates
    open_times = [m["open_time"] for m in markets if m.get("open_time") and not m["open_time"].startswith("0001")]
    settlement_times = [m["settlement_ts"] for m in markets if m.get("settlement_ts") and not m["settlement_ts"].startswith("0001")]
    close_times = [m["close_time"] for m in markets if m.get("close_time") and not m["close_time"].startswith("0001")]

    question_start_date = parse_date(min(open_times)) if open_times else ""
    resolution_date = parse_date(min(settlement_times)) if settlement_times else (parse_date(max(close_times)) if close_times else "")

    if not resolution_date:
        return None

    # Resolution criteria — generalize by replacing the outcome name with a placeholder
    # so we don't leak a specific candidate in the criteria
    sample_market = markets[0]
    outcome_name = sample_market.get("yes_sub_title", "")
    rules = sample_market.get("rules_primary", "")
    if outcome_name and outcome_name in rules:
        rules = rules.replace(outcome_name, "the selected outcome", 1)
    if sample_market.get("rules_secondary"):
        secondary = sample_market["rules_secondary"]
        if outcome_name and outcome_name in secondary:
            secondary = secondary.replace(outcome_name, "the selected outcome", 1)
        rules += "\n" + secondary

    # Append resolution date if not already mentioned in title or rules
    title = event.get("title", "")
    if not has_date_reference(title) and not has_date_reference(rules):
        rules += f"\nThis question resolved on {resolution_date}."

    # Outcomes and prices
    outcomes = []
    final_prices = {}
    for m in markets:
        sub = m.get("yes_sub_title", "")
        if sub:
            outcomes.append(sub)
            final_prices[sub] = m.get("last_price_dollars", "0")

    # Aggregate volume and open interest
    total_volume = sum(float(m.get("volume_fp", 0)) for m in markets)
    total_open_interest = sum(float(m.get("open_interest_fp", 0)) for m in markets)

    return {
        "question_title": title,
        "background": "",
        "resolution_criteria": rules.strip(),
        "answer_type": "multiple_choice",
        "answer": answer,
        "resolution_date": resolution_date,
        "question_start_date": question_start_date,
        "url": f"https://kalshi.com/events/{event['event_ticker']}",
        "data_source": "kalshi",
        "news_source": "kalshi",
        "resolution": 1,
        "outcomes": outcomes,
        "category": event.get("category", ""),
        "event_ticker": event.get("event_ticker", ""),
        "total_volume": int(total_volume),
        "total_open_interest": int(total_open_interest),
        "final_prices": final_prices,
        "num_outcomes": len(markets),
    }


def main():
    parser = argparse.ArgumentParser(description="Fetch resolved multi-outcome questions from Kalshi.")
    parser.add_argument("--output_dir", default="/fast/nchandak/forecast-sim/data/kalshi",
                        help="Directory to write output JSONL")
    parser.add_argument("--months_back", type=int, default=3,
                        help="Only include questions resolved within this many months")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cutoff = datetime.now(timezone.utc) - timedelta(days=args.months_back * 30)
    cutoff_str = cutoff.strftime("%Y-%m-%d")
    print(f"Fetching multi-outcome events settled since {cutoff_str}")

    # Step 1: Get settled mutually-exclusive events (limit pagination to avoid 100K+ events)
    # Derive which year prefixes to look for in tickers
    now = datetime.now(timezone.utc)
    cutoff_years = list({(now - timedelta(days=30 * i)).strftime("%y") for i in range(args.months_back + 1)})
    # Scale pages with time window — ~30 pages per 3 months covers the data well
    max_pages = max(30, args.months_back * 10)
    print(f"Fetching settled events (scanning up to {max_pages} pages)...")
    events = fetch_settled_events(cutoff_years, max_pages=max_pages)
    print(f"Found {len(events)} mutually-exclusive settled events")

    # Step 2: Fetch markets for each, build questions
    questions = []
    skipped = {"no_markets": 0, "no_answer": 0, "too_old": 0}
    for i, event in enumerate(events):
        ticker = event["event_ticker"]
        if (i + 1) % 50 == 0:
            print(f"  Processing {i+1}/{len(events)}...", flush=True)

        try:
            _, markets = fetch_event_markets(ticker)
        except Exception as e:
            print(f"  Error fetching {ticker}: {e}")
            continue

        if len(markets) < 2:
            skipped["no_markets"] += 1
            continue

        q = build_question(event, markets)
        if q is None:
            skipped["no_answer"] += 1
            continue

        if q["resolution_date"] < cutoff_str:
            skipped["too_old"] += 1
            continue

        questions.append(q)

    print(f"\nCollected {len(questions)} questions")
    print(f"Skipped: {skipped}")

    # Step 3: Write output
    output_path = output_dir / "kalshi_resolved.jsonl"
    with open(output_path, "w") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    print(f"Wrote {len(questions)} questions to {output_path}")

    # Print sample
    if questions:
        print("\nSample question:")
        print(json.dumps(questions[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
