"""
MoveInSync Employee App Review Intelligence Dashboard
Run:
    pip install streamlit pandas plotly
    streamlit run app.py

Place reviews_employee_en_in.csv next to app.py or in ./data/.
Optional: rating_snapshots.csv and reviews_classified.csv in the same locations.
"""
from pathlib import Path
import re
from collections import Counter

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="MoveInSync | Review Intelligence",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Styling ----------
st.markdown("""
<style>
.block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
[data-testid="stMetric"] {
    background: rgba(120, 120, 120, 0.07);
    border: 1px solid rgba(120, 120, 120, 0.18);
    padding: 14px 16px; border-radius: 12px;
}
.small-note {font-size: 0.88rem; color: #777;}
</style>
""", unsafe_allow_html=True)

# ---------- File discovery ----------
SEARCH_DIRS = [Path.cwd(), Path.cwd() / "data", Path("/mnt/data")]

def find_file(filename, required=False):
    for folder in SEARCH_DIRS:
        candidate = folder / filename
        if candidate.exists():
            return candidate
    if required:
        raise FileNotFoundError(
            f"Cannot find {filename}. Put it beside app.py or inside a data/ folder."
        )
    return None

RAW_PATH = find_file("reviews_employee_en_in.csv", required=True)
SNAPSHOT_PATH = find_file("rating_snapshots.csv")
CLASSIFIED_PATH = find_file("reviews_classified.csv")

@st.cache_data(show_spinner="Loading and preparing review data...")
def load_data(path_string):
    raw = pd.read_csv(path_string)
    df = raw.copy()

    required = {"content", "score", "at"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Raw CSV is missing required columns: {sorted(missing)}")

    for col in ["at", "repliedAt", "scraped_at"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
        else:
            df[col] = pd.NaT

    for col in ["score", "thumbsUpCount"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = 0

    for col in ["replyContent", "content", "reviewCreatedVersion", "appVersion", "app_name"]:
        if col not in df.columns:
            df[col] = ""

    df["content_clean"] = (
        df["content"].fillna("").astype(str)
        .str.replace(r"\s+", " ", regex=True).str.strip()
    )
    df["reply_clean"] = (
        df["replyContent"].fillna("").astype(str)
        .str.replace(r"\s+", " ", regex=True).str.strip()
    )
    df["has_text"] = df["content_clean"].ne("")
    df["has_reply"] = df["reply_clean"].ne("")
    df["review_word_count"] = df["content_clean"].str.findall(r"\b\w+\b").str.len()
    df["review_char_count"] = df["content_clean"].str.len()
    df["reply_delay_hours"] = (df["repliedAt"] - df["at"]).dt.total_seconds() / 3600
    df["month"] = df["at"].dt.to_period("M").astype("string")
    df["weekday"] = df["at"].dt.day_name()
    return raw, df

raw, df_all = load_data(str(RAW_PATH))

@st.cache_data(show_spinner=False)
def load_optional_csv(path_string):
    return pd.read_csv(path_string)

# ---------- Sidebar filters ----------
st.sidebar.title("Dashboard controls")
st.sidebar.caption("Filter the review population used in most charts.")
valid_dates = df_all["at"].dropna()
if valid_dates.empty:
    st.error("No valid review timestamps were found in the source CSV.")
    st.stop()

min_date = valid_dates.min().date()
max_date = valid_dates.max().date()
date_range = st.sidebar.date_input(
    "Review date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

score_options = [1, 2, 3, 4, 5]
selected_scores = st.sidebar.multiselect(
    "Star ratings",
    options=score_options,
    default=score_options,
    format_func=lambda x: f"{x} star" if x == 1 else f"{x} stars",
)
if not selected_scores:
    st.sidebar.warning("Select at least one rating to display results.")
    st.stop()

min_words = st.sidebar.slider("Minimum review word count", 0, 100, 0, 1)
reply_filter = st.sidebar.selectbox(
    "Developer reply",
    ["All reviews", "Replied only", "No reply"],
)
search_text = st.sidebar.text_input("Search review text", placeholder="e.g. login, driver, booking")

# ---------- Create filtered frame ----------
df = df_all.copy()
df = df.loc[
    df["at"].notna()
    & (df["at"].dt.date >= start_date)
    & (df["at"].dt.date <= end_date)
    & df["score"].isin(selected_scores)
    & (df["review_word_count"] >= min_words)
].copy()

if reply_filter == "Replied only":
    df = df[df["has_reply"]]
elif reply_filter == "No reply":
    df = df[~df["has_reply"]]

if search_text.strip():
    df = df[df["content_clean"].str.contains(re.escape(search_text.strip()), case=False, na=False)]

if df.empty:
    st.warning("No reviews match these filters. Widen the date range or relax the filters.")
    st.stop()

# ---------- Helpers ----------
def pct(numerator, denominator):
    return 100 * numerator / denominator if denominator else 0.0

def fmt_pct(value):
    return f"{value:.1f}%"

def metric_delta(current, baseline, suffix=""):
    if pd.isna(current) or pd.isna(baseline):
        return None
    return f"{current - baseline:+.2f}{suffix}"

def tokenize(text):
    stopwords = set("""
    a an the and or but if while with to of for in on at by from is are was were be been
    being it its this that these those i we you they he she them my our your their me us
    as am do does did have has had not no so very really just app application good great
    nice best worst bad please kindly sir madam
    """.split())
    words = re.findall(r"[a-z]+", str(text).lower())
    return [w for w in words if len(w) > 2 and w not in stopwords]

KEYWORD_GROUPS = {
    "App access / login / crash": [
        "login", "log in", "otp", "crash", "crashes", "crashing", "open", "loading",
        "load", "error", "stuck", "password", "signin", "sign in", "update"
    ],
    "Booking / schedule": [
        "book", "booking", "schedule", "scheduled", "cancel", "cancellation",
        "slot", "shift", "availability", "reschedule"
    ],
    "Tracking / GPS": [
        "track", "tracking", "location", "gps", "map", "live location", "route"
    ],
    "Cab availability / no-show": [
        "no cab", "cab not", "didn't come", "did not come", "no show", "noshow",
        "vehicle not", "not assigned", "unavailable"
    ],
    "Delay / late cab": [
        "late", "delay", "delayed", "waiting", "waited", "arrived late"
    ],
    "Driver / safety": [
        "driver", "driving", "unsafe", "safety", "rash", "behaviour", "behavior",
        "rude", "speeding"
    ],
    "Support / response": [
        "support", "customer care", "helpdesk", "complaint", "helpline", "no response",
        "respond", "contact"
    ],
    "UX / usability": [
        "confusing", "unintuitive", "difficult to use", "hard to use", "user friendly",
        "interface", "design", "complicated", "easy to use"
    ],
}

def matches_phrase(text, phrases):
    text = str(text).lower()
    return any(phrase in text for phrase in phrases)

def style_plot(fig, height=None):
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=10, r=10, t=55, b=10),
        legend_title_text="",
        font=dict(family="Arial, sans-serif", size=12),
    )
    if height:
        fig.update_layout(height=height)
    return fig

def show_download(df_to_download, filename, label):
    st.download_button(
        label=label,
        data=df_to_download.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
    )

# ---------- Header ----------
st.title("🚕 MoveInSync Employee App")
st.subheader("Review Intelligence Dashboard")
st.markdown(
    "A review-led view of employee commute app experience, combining rating trends, "
    "developer-response patterns, text signals, app-version comparisons, and the AI-classification pilot."
)
st.caption(
    f"Source: {RAW_PATH.name} · Dataset date range: {min_date} to {max_date} · "
    "Filters apply to charts unless a section explicitly says otherwise."
)

# ---------- KPI strip ----------
n = len(df)
mean_score = df["score"].mean()
one_star_pct = pct((df["score"] == 1).sum(), n)
low_pct = pct(df["score"].isin([1, 2]).sum(), n)
reply_pct = pct(df["has_reply"].sum(), n)
median_words = df["review_word_count"].median()

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Reviews in view", f"{n:,}")
k2.metric("Mean written-review score", f"{mean_score:.2f}/5")
k3.metric("1-star share", fmt_pct(one_star_pct))
k4.metric("1–2 star share", fmt_pct(low_pct))
k5.metric("Developer reply coverage", fmt_pct(reply_pct))

st.info(
    "Interpretation note: these metrics describe the filtered written reviews, not every app user. "
    "The Play Store's overall app rating is a separate metric."
)

# ---------- Tabs ----------
tabs = st.tabs([
    "Executive overview",
    "Rating & time trends",
    "Response operations",
    "Text & issue signals",
    "App versions",
    "Classifier pilot",
    "Data quality & conclusions",
])

# TAB 1: Executive overview
with tabs[0]:
    left, right = st.columns([1.1, 1])
    with left:
        rating_counts = df["score"].value_counts().reindex([1, 2, 3, 4, 5], fill_value=0).rename_axis("Rating").reset_index(name="Reviews")
        fig = px.bar(rating_counts, x="Rating", y="Reviews", text="Reviews", title="Written-review rating distribution")
        fig.update_traces(textposition="outside")
        fig.update_xaxes(type="category")
        st.plotly_chart(style_plot(fig, 370), use_container_width=True)

    with right:
        mix = pd.DataFrame({
            "Sentiment bucket": ["Low (1–2 stars)", "Mid (3 stars)", "High (4–5 stars)"],
            "Reviews": [
                int(df["score"].isin([1, 2]).sum()),
                int((df["score"] == 3).sum()),
                int(df["score"].isin([4, 5]).sum()),
            ],
        })
        fig = px.pie(mix, names="Sentiment bucket", values="Reviews", hole=0.55, title="Rating mix")
        st.plotly_chart(style_plot(fig, 370), use_container_width=True)

    st.markdown("### Quick read")
    st.write(
        f"In the current filtered view, **{low_pct:.1f}%** of written reviews are 1–2 stars, "
        f"while **{pct(df['score'].isin([4, 5]).sum(), n):.1f}%** are 4–5 stars. "
        f"The mean written-review score is **{mean_score:.2f}/5**. "
        "This mix is useful for prioritization, but the written-review population is self-selected."
    )

    st.markdown("### Reviews with the most thumbs-up")
    top_engaged = df.sort_values(["thumbsUpCount", "review_word_count"], ascending=False).head(10)
    display_cols = [c for c in ["at", "score", "thumbsUpCount", "content", "replyContent", "reviewCreatedVersion"] if c in top_engaged.columns]
    st.dataframe(top_engaged[display_cols], use_container_width=True, hide_index=True)
    show_download(df, "movesync_filtered_reviews.csv", "Download filtered review rows")

# TAB 2: Rating trends
with tabs[1]:
    trend = df.copy()
    trend["month_period"] = trend["at"].dt.to_period("M")
    monthly = trend.groupby("month_period").agg(
        review_count=("score", "size"),
        mean_score=("score", "mean"),
        one_star_pct=("score", lambda s: (s == 1).mean() * 100),
        low_rating_pct=("score", lambda s: s.isin([1, 2]).mean() * 100),
    ).reset_index()
    monthly["Month"] = monthly["month_period"].astype(str)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly["Month"], y=monthly["mean_score"], name="Average score",
        mode="lines+markers", yaxis="y"
    ))
    fig.add_trace(go.Bar(
        x=monthly["Month"], y=monthly["review_count"], name="Review volume",
        opacity=0.22, yaxis="y2"
    ))
    fig.update_layout(
        title="Monthly average score and review volume",
        xaxis_title="Month",
        yaxis=dict(title="Average written-review score", range=[1, 5]),
        yaxis2=dict(title="Review count", overlaying="y", side="right", showgrid=False),
    )
    st.plotly_chart(style_plot(fig, 470), use_container_width=True)

    fig = px.line(
        monthly, x="Month", y=["one_star_pct", "low_rating_pct"],
        markers=True, title="Monthly low-rating share",
        labels={"value": "Reviews (%)", "variable": "Measure", "Month": "Month"},
    )
    fig.update_layout(yaxis_title="Share of monthly reviews (%)")
    st.plotly_chart(style_plot(fig, 390), use_container_width=True)

    st.markdown("### Months with the lowest average scores")
    min_month_volume = st.slider("Minimum reviews per month for ranking", 1, 100, 20)
    ranked = monthly[monthly["review_count"] >= min_month_volume].sort_values("mean_score")
    st.dataframe(
        ranked[["Month", "review_count", "mean_score", "one_star_pct", "low_rating_pct"]]
        .rename(columns={"review_count": "Reviews", "mean_score": "Mean score", "one_star_pct": "1-star (%)", "low_rating_pct": "1–2 star (%)"})
        .round(2).head(15),
        use_container_width=True, hide_index=True,
    )
    st.caption("A monthly dip is a signal to investigate, not evidence of a cause. Review counts affect volatility.")

    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday = df.groupby("weekday").agg(
        reviews=("score", "size"),
        mean_score=("score", "mean"),
        one_star_pct=("score", lambda s: (s == 1).mean() * 100),
    ).reindex(weekday_order).reset_index()
    fig = px.bar(weekday, x="weekday", y="mean_score", hover_data=["reviews", "one_star_pct"], title="Average score by weekday")
    fig.update_yaxes(range=[1, 5])
    st.plotly_chart(style_plot(fig, 350), use_container_width=True)

# TAB 3: Response operations
with tabs[2]:
    replied = int(df["has_reply"].sum())
    unreplied = n - replied
    a, b, c = st.columns(3)
    a.metric("Replies present", f"{replied:,}")
    b.metric("No reply", f"{unreplied:,}")
    c.metric("Reply coverage", fmt_pct(reply_pct))

    response_by_score = df.groupby("score").agg(
        reviews=("score", "size"),
        replied=("has_reply", "sum"),
        mean_score=("score", "mean"),
    ).reset_index()
    response_by_score["reply_rate_pct"] = response_by_score["replied"] / response_by_score["reviews"] * 100
    fig = px.bar(response_by_score, x="score", y="reply_rate_pct", text=response_by_score["reply_rate_pct"].round(1),
                 hover_data=["reviews", "replied"], title="Developer reply rate by star rating")
    fig.update_traces(texttemplate="%{text}%", textposition="outside")
    fig.update_xaxes(type="category", title="Star rating")
    fig.update_yaxes(title="Reviews with reply (%)")
    st.plotly_chart(style_plot(fig, 380), use_container_width=True)

    valid_delay = df.loc[df["has_reply"] & df["reply_delay_hours"].between(0, 24 * 90), "reply_delay_hours"].dropna()
    if len(valid_delay):
        fig = px.histogram(
            pd.DataFrame({"Reply delay (hours)": valid_delay}),
            x="Reply delay (hours)", nbins=35,
            title="Reply delay distribution (0–90 days only)",
        )
        st.plotly_chart(style_plot(fig, 380), use_container_width=True)
        st.metric("Median reply delay", f"{valid_delay.median():.1f} hours")
    else:
        st.info("No usable non-negative reply delays in the current filtered view.")

    by_reply = df.groupby("has_reply").agg(
        reviews=("score", "size"),
        mean_score=("score", "mean"),
        one_star_pct=("score", lambda s: (s == 1).mean() * 100),
    ).reset_index()
    by_reply["Reply status"] = by_reply["has_reply"].map({True: "Replied", False: "No reply"})
    fig = px.bar(by_reply, x="Reply status", y="mean_score", text=by_reply["mean_score"].round(2),
                 hover_data=["reviews", "one_star_pct"], title="Average score by reply status")
    fig.update_yaxes(range=[1, 5], title="Average score")
    st.plotly_chart(style_plot(fig, 360), use_container_width=True)

    st.warning(
        "Reply coverage is not resolution rate. A reply can be templated or may not resolve the issue. "
        "Comparisons between replied and unreplied reviews are observational and affected by selection bias."
    )

# TAB 4: Text and issue signals
with tabs[3]:
    text_reviews = df.loc[df["has_text"]].copy()
    token_counts = Counter(token for text in text_reviews["content_clean"] for token in tokenize(text))
    top_n = st.slider("Number of common terms", 10, 50, 25)
    terms = pd.DataFrame(token_counts.most_common(top_n), columns=["Term", "Count"])
    if not terms.empty:
        fig = px.bar(terms.sort_values("Count"), x="Count", y="Term", orientation="h", title=f"Top {top_n} normalized terms")
        st.plotly_chart(style_plot(fig, max(420, top_n * 20)), use_container_width=True)

    st.markdown("### Overlapping keyword issue groups")
    group_rows = []
    for group, phrases in KEYWORD_GROUPS.items():
        mask = text_reviews["content_clean"].map(lambda x: matches_phrase(x, phrases))
        matched = text_reviews.loc[mask]
        group_rows.append({
            "Issue signal": group,
            "Matched reviews": len(matched),
            "Share of text reviews (%)": pct(len(matched), len(text_reviews)),
            "1–2 star matches": int(matched["score"].isin([1, 2]).sum()),
            "Low-rating share among matches (%)": pct(matched["score"].isin([1, 2]).sum(), len(matched)),
        })
    keyword_summary = pd.DataFrame(group_rows).sort_values("Matched reviews", ascending=False)
    st.dataframe(keyword_summary.round(2), use_container_width=True, hide_index=True)
    fig = px.bar(
        keyword_summary.sort_values("Matched reviews"),
        x="Matched reviews", y="Issue signal", orientation="h",
        title="Reviews matching at least one phrase per issue group",
        hover_data=["Share of text reviews (%)", "1–2 star matches"],
    )
    st.plotly_chart(style_plot(fig, 430), use_container_width=True)

    st.markdown("### Inspect examples")
    selected_group = st.selectbox("Choose an issue signal", list(KEYWORD_GROUPS.keys()))
    phrases = KEYWORD_GROUPS[selected_group]
    mask = text_reviews["content_clean"].map(lambda x: matches_phrase(x, phrases))
    examples = text_reviews.loc[mask].sort_values(["score", "thumbsUpCount"], ascending=[True, False])
    show_cols = [c for c in ["at", "score", "content", "thumbsUpCount", "replyContent", "reviewCreatedVersion"] if c in examples.columns]
    st.dataframe(examples[show_cols].head(30), use_container_width=True, hide_index=True)
    st.caption(
        "Keyword groups overlap and use simple phrase matching. They are discovery aids, not validated labels; "
        "manually inspect examples before reporting issue counts."
    )
    show_download(keyword_summary, "movesync_keyword_signals.csv", "Download keyword summary")

# TAB 5: App versions
with tabs[4]:
    version_col = "reviewCreatedVersion"
    version_df = df.loc[df[version_col].notna() & df[version_col].astype(str).ne("")].copy()
    min_version_reviews = st.slider("Minimum reviews per version", 3, 50, 10)
    versions = version_df.groupby(version_col).agg(
        reviews=("score", "size"),
        mean_score=("score", "mean"),
        one_star_pct=("score", lambda s: (s == 1).mean() * 100),
        low_rating_pct=("score", lambda s: s.isin([1, 2]).mean() * 100),
    ).reset_index()
    versions = versions[versions["reviews"] >= min_version_reviews].sort_values("reviews", ascending=False)

    if not versions.empty:
        left, right = st.columns(2)
        with left:
            fig = px.bar(versions.head(20).sort_values("reviews"), x="reviews", y=version_col, orientation="h",
                         title="Most-reviewed app versions")
            st.plotly_chart(style_plot(fig, 500), use_container_width=True)
        with right:
            fig = px.scatter(
                versions, x="reviews", y="mean_score", size="low_rating_pct",
                hover_name=version_col,
                hover_data=["one_star_pct", "low_rating_pct"],
                title="Version volume vs average score",
                labels={"reviews": "Review count", "mean_score": "Average score", "low_rating_pct": "1–2 star share (%)"},
            )
            fig.update_yaxes(range=[1, 5])
            st.plotly_chart(style_plot(fig, 500), use_container_width=True)
        st.dataframe(versions.rename(columns={
            version_col: "Version", "reviews": "Reviews", "mean_score": "Mean score",
            "one_star_pct": "1-star (%)", "low_rating_pct": "1–2 star (%)"
        }).round(2), use_container_width=True, hide_index=True)
        show_download(versions, "movesync_version_summary.csv", "Download version summary")
    else:
        st.info("No app versions meet the minimum review threshold for this filter.")

    st.warning(
        "Version-level differences are observational. Review-created version may be missing, and versions have "
        "different exposure periods. Validate release notes and review examples before attributing changes to a release."
    )

# TAB 6: Classifier pilot
with tabs[5]:
    if CLASSIFIED_PATH is None:
        st.info("Optional file not found. Add reviews_classified.csv beside app.py or in data/ to enable this section.")
    else:
        classified = load_optional_csv(str(CLASSIFIED_PATH))
        st.markdown(f"**Loaded:** `{CLASSIFIED_PATH.name}` · **Rows:** {len(classified):,}")
        st.caption(
            "This is the classifier pilot, not a representative sample of all app reviews. "
            "If it contains the first 100 date-ordered rows, do not extrapolate its label distribution to the full dataset."
        )
        display_cols = [c for c in ["reviewId", "primary", "secondary", "owner", "severity", "issue"] if c in classified.columns]
        st.dataframe(classified[display_cols].head(20), use_container_width=True, hide_index=True)

        if "primary" in classified.columns:
            counts = classified["primary"].fillna("MISSING").value_counts().rename_axis("Primary category").reset_index(name="Reviews")
            fig = px.bar(counts.sort_values("Reviews"), x="Reviews", y="Primary category", orientation="h",
                         title="AI-classified pilot: primary labels")
            st.plotly_chart(style_plot(fig, max(400, len(counts) * 24)), use_container_width=True)

        if "owner" in classified.columns:
            owners = classified["owner"].fillna("MISSING").value_counts().rename_axis("Owner").reset_index(name="Reviews")
            fig = px.pie(owners, names="Owner", values="Reviews", hole=0.5, title="Suggested ownership in classified pilot")
            st.plotly_chart(style_plot(fig, 350), use_container_width=True)

        if {"primary", "score"}.issubset(classified.columns):
            cross = pd.crosstab(classified["primary"].fillna("MISSING"), classified["score"])
            st.markdown("**Category × rating cross-tabulation (counts)**")
            st.dataframe(cross, use_container_width=True)

        if "severity" in classified.columns:
            severity = pd.to_numeric(classified["severity"], errors="coerce")
            st.metric("Rows with severity assigned", f"{severity.notna().sum():,}")
            sev_counts = severity.value_counts().sort_index().rename_axis("Severity").reset_index(name="Reviews")
            fig = px.bar(sev_counts, x="Severity", y="Reviews", title="Severity distribution in classifier pilot")
            st.plotly_chart(style_plot(fig, 320), use_container_width=True)

        st.warning(
            "The pilot's category counts depend on sample order and model judgment. "
            "Assess label quality against a manually reviewed reference set before claiming model accuracy."
        )

# TAB 7: Data quality and conclusions
with tabs[6]:
    st.markdown("### Data quality")
    quality = pd.DataFrame([
        ["Rows in source file", len(df_all)],
        ["Unique review IDs", df_all["reviewId"].nunique() if "reviewId" in df_all else np.nan],
        ["Duplicate review IDs", df_all["reviewId"].duplicated().sum() if "reviewId" in df_all else np.nan],
        ["Missing review text", df_all["content"].isna().sum()],
        ["Blank review text", (~df_all["has_text"]).sum()],
        ["Missing scores", df_all["score"].isna().sum()],
        ["Scores outside 1–5", int((~df_all["score"].between(1, 5) & df_all["score"].notna()).sum())],
        ["Missing review timestamps", df_all["at"].isna().sum()],
        ["Developer replies", int(df_all["has_reply"].sum())],
        ["Replies with missing reply timestamp", int((df_all["has_reply"] & df_all["repliedAt"].isna()).sum())],
        ["Negative reply delays", int((df_all["reply_delay_hours"] < 0).sum())],
    ], columns=["Check", "Value"])
    st.dataframe(quality, use_container_width=True, hide_index=True)

    st.markdown("### Automatically generated interpretation for current filters")
    positive_pct = pct(df["score"].isin([4, 5]).sum(), n)
    if low_pct >= 30:
        burden = "The low-rating share is elevated in this filtered view and deserves a focused review of complaint themes."
    elif low_pct >= 20:
        burden = "Low ratings form a meaningful minority of the filtered reviews; prioritize recurring issues rather than isolated anecdotes."
    else:
        burden = "Low ratings are a smaller share of this filtered view, but high-severity complaints should still be reviewed individually."

    if reply_pct < 20:
        response_text = "Developer reply coverage is low; consider prioritizing response coverage for actionable low-rated reviews."
    elif reply_pct < 50:
        response_text = "Developer replies cover a minority of reviews; response coverage and follow-through could be improved."
    else:
        response_text = "Developer reply coverage is relatively broad, but coverage alone cannot establish whether issues were resolved."

    st.write(
        f"- **Experience signal:** Mean written-review score is **{mean_score:.2f}/5**; "
        f"**{positive_pct:.1f}%** are 4–5 stars and **{low_pct:.1f}%** are 1–2 stars. {burden}\n"
        f"- **Response operations:** Reply coverage is **{reply_pct:.1f}%** in this filtered view. {response_text}\n"
        "- **Issue discovery:** Use keyword groups to shortlist recurring themes, then validate with manual reading or the classifier pilot.\n"
        "- **Release monitoring:** Investigate months and app versions with unusually weak scores, but do not infer causality without release and operational evidence."
    )

    st.markdown("### Source-level conclusions (unfiltered raw dataset)")
    full = df_all.loc[df_all["score"].between(1, 5)].copy()
    full_n = len(full)
    full_mean = full["score"].mean()
    full_one = pct((full["score"] == 1).sum(), full_n)
    full_low = pct(full["score"].isin([1, 2]).sum(), full_n)
    full_high = pct(full["score"].isin([4, 5]).sum(), full_n)
    full_reply = pct(full["has_reply"].sum(), len(df_all))

    st.markdown(
        f"""
        **Baseline facts from the raw file (all rows, independent of sidebar filters):**

        - **{len(df_all):,}** written reviews, dated **{df_all['at'].min().date()} to {df_all['at'].max().date()}**.
        - Mean written-review score: **{full_mean:.2f}/5**; median: **{full['score'].median():.0f}/5**.
        - **{full_one:.1f}%** are 1-star and **{full_low:.1f}%** are 1–2 stars; **{full_high:.1f}%** are 4–5 stars.
        - Developer reply coverage: **{full_reply:.1f}%**.
        - Treat this as self-selected review evidence, not a representative survey of all employees or app sessions.
        """
    )

    if SNAPSHOT_PATH is not None:
        snapshots = load_optional_csv(str(SNAPSHOT_PATH))
        st.markdown("### Play Store snapshot (separate metric)")
        st.dataframe(snapshots, use_container_width=True, hide_index=True)
        st.caption(
            "The app-level Play Store rating is based on a different population and denominator than the mean of scraped written reviews."
        )

    show_download(quality, "movesync_data_quality_checks.csv", "Download data-quality checks")

st.divider()
st.caption(
    "MoveInSync review intelligence · Exploratory analytics, not causal inference. "
    "Keyword matching is heuristic; classifier results are a separate pilot."
)
