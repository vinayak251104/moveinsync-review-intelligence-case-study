# pip install google-play-scraper pandas pyarrow
"""
MoveInSync Play Store review scraper.

First run:   python scrape_reviews.py --months 18
Later runs: python scrape_reviews.py          (fetches only new reviews)
"""
import argparse
import os
import time
from datetime import datetime, timedelta

import pandas as pd
from google_play_scraper import Sort, app, reviews

APPS = {
    "employee":   "com.moveinsync.ets",
    "driver":     "com.moveinsync.driver.dco",
    "field":      "field.moveinsync.ets",
    "transit":    "com.moveinsync.publictransit",
    "workplace":  "com.workinsync.meeting",
}
DEFAULT_APPS = ["employee", "driver"]   # primary scope; add others if needed

DATA_DIR = "data"
DROP_COLS = ["userName", "userImage"]   # not needed, avoids storing personal data
PAGE_SIZE = 200
OVERLAP_DAYS = 7                        # re-check recent days to catch edited reviews


def fetch_with_retry(app_id, token, lang, country, tries=5):
    for attempt in range(tries):
        try:
            return reviews(
                app_id, lang=lang, country=country,
                sort=Sort.NEWEST, count=PAGE_SIZE,
                continuation_token=token,
            )
        except Exception as e:
            wait = 2 ** attempt
            print(f"  error ({e}); retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError("Too many failures, try again later")


def scrape_app(app_id, stop_date, lang="en", country="in"):
    rows, token = [], None
    while True:
        batch, token = fetch_with_retry(app_id, token, lang, country)
        if not batch:
            break
        rows.extend(batch)
        oldest = min(r["at"] for r in batch)
        print(f"  {len(rows)} fetched, oldest so far: {oldest:%Y-%m-%d}")
        if oldest < stop_date or token is None:
            break
        time.sleep(1)                   # be polite
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df[df["at"] >= stop_date]
    return df.drop(columns=[c for c in DROP_COLS if c in df.columns])


def update_master(name, app_id, months, lang, country):
    path = os.path.join(DATA_DIR, f"reviews_{name}_{lang}_{country}.parquet")
    master = pd.read_parquet(path) if os.path.exists(path) else None

    if master is not None and len(master):
        stop_date = master["at"].max() - timedelta(days=OVERLAP_DAYS)
        print(f"[{name}] incremental run, fetching since {stop_date:%Y-%m-%d}")
    else:
        stop_date = datetime.now() - timedelta(days=30 * months)
        print(f"[{name}] first run, fetching since {stop_date:%Y-%m-%d}")

    new = scrape_app(app_id, stop_date, lang, country)
    if new.empty:
        print(f"[{name}] no new reviews")
        return
    new["app_name"] = name
    new["scraped_at"] = datetime.now()

    combined = new if master is None else pd.concat([master, new])
    # keep the latest copy of each review (picks up edits and new dev replies)
    combined = (combined.sort_values("scraped_at")
                        .drop_duplicates("reviewId", keep="last")
                        .sort_values("at", ascending=False))
    combined.to_parquet(path, index=False)
    combined.to_csv(path.replace(".parquet", ".csv"), index=False)
    print(f"[{name}] saved {len(combined)} total reviews -> {path}")


def snapshot_ratings(names, lang, country):
    """Append today's overall rating + star histogram for each app."""
    rows = []
    for name in names:
        info = app(APPS[name], lang=lang, country=country)
        h = info.get("histogram") or [None] * 5
        rows.append({
            "date": datetime.now().date(), "app_name": name,
            "score": info["score"], "ratings": info["ratings"],
            "reviews": info["reviews"], "version": info["version"],
            "updated": info["updated"],
            "star1": h[0], "star2": h[1], "star3": h[2],
            "star4": h[3], "star5": h[4],
        })
    path = os.path.join(DATA_DIR, "rating_snapshots.csv")
    df = pd.DataFrame(rows)
    if os.path.exists(path):
        old = pd.read_csv(path, parse_dates=["date"])
        old["date"] = old["date"].dt.date
        df = pd.concat([old, df]).drop_duplicates(["date", "app_name"], keep="last")
    df.to_csv(path, index=False)
    print(df.tail(len(names)).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--apps", nargs="+", default=DEFAULT_APPS, choices=APPS.keys())
    ap.add_argument("--months", type=int, default=18, help="history to fetch on first run")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--country", default="in")
    args = ap.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    for name in args.apps:
        update_master(name, APPS[name], args.months, args.lang, args.country)
    snapshot_ratings(args.apps, args.lang, args.country)