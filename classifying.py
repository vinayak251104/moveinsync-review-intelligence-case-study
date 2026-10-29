# py -m pip install anthropic pandas
import os, json, time
import pandas as pd
from anthropic import Anthropic

TEST = True          # True = first 100 reviews only. Set False for the full run.
MODEL = "claude-haiku-5-5"
BATCH = 20
IN_CSV = "data/reviews_employee_en_in.csv"
OUT_CSV = "data/reviews_classified.csv"
PROGRESS = "data/classify_progress.jsonl"   # lets you resume if it stops

client = Anthropic()   # reads ANTHROPIC_API_KEY from the environment

SYSTEM = """You are analysing Google Play reviews of MoveInSync, an app employees use to book
office cabs/shuttles, track them live, and raise issues. Classify each review.

Categories: APP_ACCESS (crashes, won't open, login/OTP failures, slowness),
BOOKING (failed/impossible booking or cancelling, roster issues, ad-hoc cabs),
TRACKING (wrong/laggy GPS, wrong pickup time, vehicle not visible, bad notifications),
NO_CAB (booked but no vehicle assigned or arrived, vendor failure),
LATE_CAB (cab late, long waits, wrong route, detours),
POOLING (badly grouped riders, long rides from clubbing),
NO_SHOW (marked no-show unfairly, check-in problems),
DRIVER (rude, refuses rides, driving quality),
SAFETY (women's safety, night travel, OTP/driver verification, SOS),
SUPPORT (help desk unreachable, no resolution),
UX_CHANGE (redesign complaints, confusing screens, missing features, permissions),
OTHER_PRODUCT (desk booking, meeting rooms, parking - not commute),
PRAISE (positive, no issue), UNCLEAR (too short or vague).

Rules:
- primary: the single main reason for the rating
- secondary: up to 2 other categories clearly present, else []
- owner: "app" if the app itself failed; "ops_vendor" if cab/driver/vendor behaviour;
  "client_company" if it depends on the employer's transport desk; "none" for PRAISE/UNCLEAR
- severity: 1-3 (3 = user could not complete their commute or felt unsafe)
- issue: max 12 words, specific (e.g. "OTP not received on login"); "" for PRAISE/UNCLEAR
- Judge by the review text, not the star rating. A 5-star review can contain a complaint.
- Return ONLY a JSON list of {"id","primary","secondary","owner","severity","issue"}."""


def classify(batch):
    payload = [{"id": r.reviewId, "score": int(r.score), "text": str(r.content)[:600]}
               for r in batch.itertuples()]
    for attempt in range(5):
        try:
            msg = client.messages.create(
                model=MODEL, max_tokens=4000, system=SYSTEM,
                messages=[{"role": "user", "content": json.dumps(payload)}],
            )
            text = msg.content[0].text.strip().replace("```json", "").replace("```", "")
            return json.loads(text)
        except Exception as e:
            print("  retry:", e)
            time.sleep(2 ** attempt)
    return []


df = pd.read_csv(IN_CSV)
if TEST:
    df = df.head(100)

# load anything already classified (resume support)
done = {}
if os.path.exists(PROGRESS):
    with open(PROGRESS, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            done[row["id"]] = row

todo = df[~df["reviewId"].isin(done)]
print(f"{len(done)} already done, {len(todo)} to go")

with open(PROGRESS, "a", encoding="utf-8") as f:
    for i in range(0, len(todo), BATCH):
        for row in classify(todo.iloc[i:i + BATCH]):
            done[row["id"]] = row
            f.write(json.dumps(row) + "\n")
        f.flush()
        if (i // BATCH) % 10 == 0:
            print(f"  {i + BATCH}/{len(todo)} batches processed")

labels = pd.DataFrame(done.values())
labels["secondary"] = labels["secondary"].apply(
    lambda x: "|".join(x) if isinstance(x, list) else "")
result = df.merge(labels, left_on="reviewId", right_on="id", how="left").drop(columns="id")
result.to_csv(OUT_CSV, index=False)

print(f"\nLabelled {result['primary'].notna().sum()} of {len(result)} reviews")
print(result["primary"].value_counts())