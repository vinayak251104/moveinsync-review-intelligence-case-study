
# MoveInSync Employee Commute App — Review Intelligence

An exploratory analytics and review-triage case study built around Google Play written reviews for the MoveInSync Employee commute app. The project combines descriptive analysis, interactive visualizations, heuristic text exploration, app-version comparisons, developer-response analysis, and an optional AI-classifier pilot.

**Live Demo:** [Launch the Streamlit Dashboard](https://moveinsync-review-intelligence-case-study-vinayak251104.streamlit.app/)

**GitHub Repository:** [moveinsync-review-intelligence-case-study](https://github.com/vinayak251104/moveinsync-review-intelligence-case-study)

> **Important:** This project analyzes user-written reviews. Reviews are self-selected feedback and are not a representative survey of every employee, app session, or commute. Findings are descriptive unless independently validated.

## Table of Contents

- [Project Objectives](#project-objectives)
- [Repository Structure](#repository-structure)
- [Technology Stack](#technology-stack)
- [Data Sources](#data-sources)
- [Workflow](#workflow)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Dashboard Features](#dashboard-features)
- [Metrics and Definitions](#metrics-and-definitions)
- [AI Tools and Their Purpose](#ai-tools-and-their-purpose)
- [AI Classification Pilot](#ai-classification-pilot)
- [How to Run the Project](#how-to-run-the-project)
- [Outputs](#outputs)
- [Limitations](#limitations)
- [Recommended Next Steps](#recommended-next-steps)

## Project Objectives

The goal is to turn a noisy collection of public app reviews into a repeatable workflow for understanding user experience and identifying areas that deserve investigation.

The project is designed to:

1. Profile the dataset and assess basic data quality.
2. Understand the distribution of written-review ratings.
3. Track monthly review volume and rating changes.
4. Examine developer reply coverage and reply timing.
5. Explore review text to surface recurring product and operational signals.
6. Compare ratings across app versions where sufficient reviews are available.
7. Use AI-assisted classification to organize a limited sample of reviews into issue categories.
8. Translate descriptive signals into investigation questions and potential operational actions.

This is an analytics and prioritization aid, not a production incident detector or a definitive measure of customer satisfaction.

## Repository Structure

```text
moveinsync-review-intelligence-case-study/
├── app.py
├── scraper.py
├── classifying.py
├── classifying_gemini.py
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── reviews_employee_en_in.csv
│   ├── reviews_employee_en_in.parquet
│   ├── rating_snapshots.csv
│   ├── reviews_classified.csv
│   └── reviews_classified_recovered.csv
├── outputs/
│   ├── classified_pilot_copy.csv
│   └── top_engaged_reviews.csv
└── exploratory_data_analysis/
    └── MoveInSync_Exploratory_Analysis.ipynb
```

*Note: The exact files in your repository may differ depending on which data and generated outputs you commit.*

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Data processing, analysis, and application logic |
| pandas | Data cleaning, grouping, aggregation, and CSV handling |
| NumPy | Numerical operations and missing-value handling |
| Matplotlib | Static charts in the EDA notebook |
| Plotly | Interactive charts and visualizations |
| Streamlit | Interactive dashboard and user controls |
| Jupyter Notebook | Reproducible exploratory data analysis |
| Regular expressions | Basic text tokenization and phrase matching |
| Google Gemini API (`google-genai`) | AI-assisted classification of a limited review sample |
| Git and GitHub | Version control and project hosting |
| Streamlit Community Cloud | Dashboard deployment |

## Data Sources

### 1. Written App Reviews

The primary dataset is `reviews_employee_en_in.csv`, containing Google Play written reviews collected for the MoveInSync Employee commute app.

The dataset used during development contained **8,355 written reviews**, spanning approximately **14 November 2023 to 28 October 2026**. Recheck these figures against the exact source file before using them in a formal report.

Relevant fields include:

| Field | Purpose |
|---|---|
| `reviewId` | Review identity and duplicate checks |
| `content` | User-written review text |
| `score` | Star rating from 1 to 5 |
| `at` | Review timestamp |
| `thumbsUpCount` | Public engagement signal |
| `replyContent` | Developer response text |
| `repliedAt` | Reply timestamp |
| `reviewCreatedVersion` | App-version comparisons |
| `appVersion` | Optional app-version metadata |
| `scraped_at` | Data extraction metadata, when available |

### 2. Rating Metadata

`rating_snapshots.csv` optionally stores app-level rating metadata captured at a point in time.

**The Play Store's displayed app rating is not the same as the mean of scraped written-review ratings.** The populations, denominators, and aggregation methods may differ.

### 3. AI-Classified Reviews

`reviews_classified.csv` contains the optional output of the separate AI classification pipeline.

Expected fields may include:

- `reviewId`
- `primary`
- `secondary`
- `owner`
- `severity`
- `issue`
- `score`, if included in the classifier output

The dashboard treats this file as an optional input. AI-generated labels are not assumed to be correct or representative of the full dataset.

## Workflow

### Step 1 — Data Loading and Quality Checks

- Load the raw review dataset.
- Validate required columns and data types.
- Parse review and developer-reply timestamps.
- Convert rating and engagement fields to numeric values.
- Inspect missing values, blank review text, duplicate IDs, invalid ratings, and timestamp anomalies.

### Step 2 — Data Preparation

The analysis derives additional fields, including:

- `content_clean`: whitespace-normalized review text.
- `reply_clean`: whitespace-normalized developer response.
- `has_text`: whether a review contains non-empty text.
- `has_reply`: whether a developer response exists.
- `review_word_count`: number of word-like tokens.
- `review_char_count`: review character count.
- `reply_delay_hours`: elapsed time between review and reply.
- `month`: calendar month of the review.
- `weekday`: day of the week of the review.

The original review text is retained. Normalization supports analysis rather than changing the review's meaning.

### Step 3 — Rating Analysis

- Count reviews by star rating.
- Calculate mean, median, and standard deviation.
- Calculate one-star, low-rating, and high-rating shares.
- Compare monthly average ratings with review volume.
- Track changes in one-star and low-rating shares over time.

### Step 4 — Developer Response Analysis

- Calculate developer reply coverage.
- Compare reply coverage across star ratings.
- Calculate reply delays where timestamps are usable.
- Compare ratings for reviews with and without developer replies.

Reply presence does not establish that a problem was resolved. Developers may also choose which reviews to answer, creating selection bias.

### Step 5 — Text and Engagement Exploration

- Tokenize review text using regular expressions.
- Remove a small list of common words.
- Identify frequent terms.
- Examine review length and thumbs-up patterns.
- Inspect highly engaged reviews and examples matching keyword groups.

This is lightweight text exploration, not a complete natural-language-processing pipeline. The EDA stage does not rely on embeddings, topic models, or a trained sentiment model.

### Step 6 — Keyword-Based Issue Signals

The notebook and dashboard use phrase lists to find reviews mentioning broad areas such as:

- App access, login, OTP, loading, and crashes
- Booking and scheduling
- Tracking, GPS, maps, and location
- Cab availability and no-shows
- Delays and waiting
- Driver behaviour and safety
- Support and responses
- User experience and usability

A review can match multiple groups. Keyword matches are exploratory signals, not confirmed issue counts.

### Step 7 — App-Version Comparisons

For versions with sufficient review volume, the dashboard reports review counts, average ratings, one-star shares, and low-rating shares.

These comparisons are observational. A lower average rating does not establish that a particular release caused a problem.

### Step 8 — AI-Assisted Classification

A separate Gemini-powered workflow assigns structured labels to a limited sample of reviews. The output can include an issue category, secondary category, suggested owner, severity, and short issue description.

The classification pilot is kept separate from full-dataset EDA.

## Exploratory Data Analysis

### Rating Distribution

Ratings are treated as ordinal scores from 1 to 5.

- **One-star share:** 1-star reviews divided by reviews with valid ratings.
- **Low-rating share:** 1- and 2-star reviews divided by reviews with valid ratings.
- **High-rating share:** 4- and 5-star reviews divided by reviews with valid ratings.

The denominator should always be stated when reporting percentages. Dashboard filters may change the population used in the calculations.

### Monthly Trends

Reviews are grouped by calendar month to calculate:

- Review count
- Mean and median rating
- One-star count and percentage
- 1–2-star count and percentage
- 4–5-star count and percentage

Review volume is considered alongside rating trends because small monthly samples can be volatile.

### Developer Response Analysis

- **Reply coverage:** Reviews with non-empty developer reply text divided by reviews in scope.
- **Reply delay:** Reply timestamp minus review timestamp.
- Negative reply delays are treated as timestamp anomalies.
- Extremely long delays may be excluded from selected visualizations to keep charts interpretable.

Response coverage does not measure reply quality, helpfulness, or resolution.

### Review Text and Engagement

Word count and character count are basic proxies for review length. `thumbsUpCount` is used as a public engagement indicator.

Neither metric directly measures issue severity or how representative a review is of the wider employee population.

### Keyword Matching

Each issue group consists of a list of phrases. A review is counted once per group if it contains any phrase in that group.

Consequently:

- Group counts can overlap.
- Percentages across groups do not necessarily sum to 100%.
- Phrases can occur in positive as well as negative reviews.
- Synonyms and misspellings may be missed.
- Manual review is required before presenting a keyword group as a confirmed product finding.

### App-Version Analysis

Version summaries use `reviewCreatedVersion` when available and report rating metrics for versions meeting a minimum-volume threshold.

Versions can differ in exposure duration, rollout status, user population, and metadata completeness. These differences limit direct comparisons.

## Dashboard Features

The Streamlit dashboard contains the following sections.

### Global Filters

- Review date range
- Selected star ratings
- Minimum review word count
- Developer-reply status
- Phrase search within review text

Most charts use the filtered review population. Data-quality views can help distinguish filtered results from source-level figures.

### Executive Overview

- Number of reviews in view
- Mean written-review rating
- One-star share
- 1–2-star share
- Developer reply coverage
- Star-rating distribution
- Low-, medium-, and high-rating mix
- Reviews with the most thumbs-up
- Downloadable filtered reviews

### Rating and Time Trends

- Monthly average rating and review volume
- Monthly one-star and low-rating shares
- Lowest-scoring months subject to a minimum review threshold
- Average score by weekday

### Response Operations

- Reply count and coverage
- Reply rate by star rating
- Reply-delay distribution and median delay
- Mean rating by reply status
- Guidance on interpreting response metrics

### Text and Issue Signals

- Common normalized terms
- Keyword-group counts and shares
- Low-rating matches within each keyword group
- Example reviews for a selected group
- Downloadable keyword summaries

### App Versions

- Most-reviewed versions
- Review volume versus mean rating
- One-star and low-rating shares
- Configurable minimum review threshold
- Downloadable version summaries

### Classifier Pilot

When the optional classifier CSV is available:

- Primary-category counts
- Suggested ownership distribution
- Category-by-rating cross-tabulation
- Severity distribution
- Sample classified reviews

### Data Quality and Conclusions

- Duplicate-ID and missing-data checks
- Invalid-score and timestamp checks
- Source-level summary statistics
- Interpretation for the current filters
- Optional rating snapshot display

## Metrics and Definitions

| Metric | Definition | Important caveat |
|---|---|---|
| Mean written-review rating | Arithmetic mean of valid review scores | Not the same as the Play Store's displayed app rating |
| One-star share | 1-star reviews divided by valid scored reviews | Sensitive to self-selection |
| Low-rating share | 1–2-star reviews divided by valid scored reviews | Useful for triage, not a complete satisfaction measure |
| High-rating share | 4–5-star reviews divided by valid scored reviews | High ratings can coexist with serious issues for a minority |
| Reply coverage | Reviews with non-empty developer replies divided by reviews in scope | Does not measure resolution |
| Reply delay | `repliedAt` minus `at` | Depends on timestamp quality |
| Review length | Word-like tokens or characters | Not a severity measure |
| Thumbs-up count | Public helpfulness votes captured in the source | Older reviews may have had more time to accumulate votes |
| Keyword matches | Reviews containing one or more phrases in a group | Groups overlap and matching is context-insensitive |
| Version mean score | Average review score for a version | Does not establish release quality causality |
| Classifier category count | Pilot rows assigned to a category | Depends on sample selection and label quality |

## AI Tools and Their Purpose

AI supported parts of the workflow, but it was not treated as a substitute for data checks or human review.

| Tool | Purpose | Scope and caveats |
|---|---|---|
| **Google Gemini API (`google-genai`)** | Applied the review taxonomy to a small pilot sample and generated structured labels such as primary/secondary category, suggested owner, severity, and concise issue summary. | The pilot contains 100 reviews selected from the first rows in date order. It is not representative, so its category shares must not be generalized to all reviews. Labels should be manually checked before operational use. |
| **ChatGPT** | Assisted with project planning, Python and Streamlit troubleshooting, analytical framing, and documentation refinement. | Suggestions and generated code should be tested against the source data and running application. Findings should be supported by the documented data and methods. |
| **Claude** |  Assisted with developing the Google Play review scraping and processing workflow, as well as the LLM-based classification pipeline. The classifier processed reviews in batches of 20 and generated structured outputs, including primary and secondary issue categories, ownership, severity, and concise issue descriptions | The pipeline incorporated retry handling, JSONL progress tracking, resume support, validation, and CSV output. |

AI-assisted outputs were reviewed as part of the workflow. The final analysis and dashboard were built around the collected review dataset.

**Note:** Review scraping, LLM-based classification, and dashboard analytics are separate stages. The deployed Streamlit dashboard does not call the Gemini API at runtime.  |


The dashboard itself does **not** call an AI model at runtime. Gemini classification is a separate workflow, and the dashboard reads any available classified-review CSV as an optional input.

Keep API keys and other credentials out of source control. Configure secrets locally or through the deployment platform's secrets settings.

## AI Classification Pilot

The classification taxonomy includes categories such as:

- `APP_ACCESS`
- `BOOKING`
- `TRACKING`
- `NO_CAB`
- `LATE_CAB`
- `POOLING`
- `NO_SHOW`
- `DRIVER`
- `SAFETY`
- `SUPPORT`
- `UX_CHANGE`
- `OTHER_PRODUCT`
- `PRAISE`
- `UNCLEAR`

The classifier can also assign:

- **Owner:** Suggested responsibility area, such as app, operations/vendor, client company, or none.
- **Severity:** An exploratory severity rating.
- **Issue summary:** A short description of the reported issue.

### Pilot Scope

The pilot covers **100 reviews selected from the first rows in date order**, not a random or stratified sample. This choice limits its ability to represent the full review dataset.

The label distribution must not be interpreted as the true distribution of all user issues. Manually inspect a subset of labels before using them in decisions or presenting them as validated outputs.

Classifier accuracy should only be reported after comparison with a manually annotated reference set. Useful evaluation measures include per-class precision, recall, F1 score, a confusion matrix, and agreement on ownership and severity.

## How to Run the Project

### Requirements

- Python 3.10 or later is recommended.
- The raw review CSV must be available locally.
- Internet access is not required to run the dashboard after dependencies are installed, unless additional online services are used.

### 1. Open a Terminal in the Project Folder

In PowerShell, use quotes around paths containing spaces:

```powershell
cd "C:\Users\YourName\Downloads\MoveInSync"
```

### 2. Install Dependencies

If the repository includes `requirements.txt`, install the listed dependencies:

```powershell
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
```

If the `py` launcher is unavailable, replace `py` with `python`.

### 3. Start the Dashboard

```powershell
py -m streamlit run app.py
```

Using `py -m streamlit` can avoid issues where the Streamlit executable is not on PATH.

### 4. Run the EDA Notebook

Launch Jupyter:

```powershell
py -m jupyter notebook
```

Open `MoveInSync_Exploratory_Analysis.ipynb` and run the cells from top to bottom. The notebook writes generated summaries to `outputs/`.

### 5. Run the Gemini Classification Workflow

The Gemini classification script is separate from the dashboard. To use it, install the dependency and configure your API key securely according to the script's requirements.

Do not hard-code API keys in Python files or commit them to GitHub.

### Common Setup Issues

**`streamlit` is not recognized**

```powershell
py -m pip install streamlit pandas plotly numpy
py -m streamlit run app.py
```

**File not found**

Confirm `app.py` and the raw CSV are in the folder from which you launched Streamlit, or put the CSV in a `data/` folder. Check spelling and file extensions.

**Different Python environments**

Use the same Python launcher for installation and execution:

```powershell
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

## Outputs

The EDA notebook writes CSV summaries into `outputs/`, including:

- Data-quality summary
- Overall rating summary and distribution
- Monthly trends
- Developer-response summary
- Response rates by rating
- Score by reply status
- Review length and engagement by rating
- Top terms
- Keyword-group summary
- App-version summary
- Optional classifier pilot copy
- Top-engaged reviews

The dashboard also offers downloadable tables for filtered reviews, keyword signals, app-version summaries, and data-quality checks.

Regenerate outputs from the same source data before publishing final numbers. Record the analysis date, filters, and source-file version alongside reported results.

## Limitations

### 1. Self-Selection Bias

People who write app-store reviews may differ from those who do not. Review data cannot directly estimate satisfaction across all employees or all commutes.

### 2. Written-Review Mean Versus App Rating

The average of scraped written reviews is not the Play Store's aggregate rating. They may have different denominators and aggregation methods.

### 3. No Causal Inference

Monthly rating dips, app-version differences, reply status, and keyword counts are descriptive patterns. They do not prove that a release, vendor, driver, or operational process caused an outcome.

### 4. Keyword Rules Are Simplistic

Phrase matching cannot reliably understand context, negation, sarcasm, spelling errors, or synonyms. Categories overlap and should not be summed as though they were mutually exclusive.

### 5. Timestamp Quality

Missing, malformed, or inconsistent timestamps can affect trend and reply-delay metrics. Negative reply delays are anomalies that require investigation.

### 6. Version Metadata May Be Incomplete

Some reviews may not contain version information. Versions may also differ in exposure duration and user populations.

### 7. Classifier Pilot Is Limited

The 100-review date-ordered pilot is not a random or representative sample. Its category proportions must not be extrapolated to all reviews. AI-generated ownership and severity labels may be wrong and should be reviewed.

### 8. No Validated Resolution Metric

The dataset shows whether a developer reply exists, not whether the user's problem was fixed. Measuring resolution would require linked support or operational records.

### 9. Privacy and Responsible Reporting

Review text can contain personal information. Avoid exposing unnecessary identifiers or sensitive text in public reports. Use only data you are permitted to process, and do not commit secrets or API keys.

### 10. Dashboard Interpretation

Global filters change the population used by most charts. Record the filters when quoting results. Pair exploratory findings with manual validation and domain knowledge before making business decisions.

## Recommended Next Steps

1. **Validate issue themes:** Manually review examples from keyword groups and classifier categories.
2. **Create a labelled reference set:** Annotate a stratified sample across ratings and issue types, then evaluate classifier precision and recall.
3. **Prioritize by impact:** Combine frequency with severity, employee impact, operational risk, and confidence—not raw keyword counts alone.
4. **Investigate time anomalies:** Compare low-scoring months with release notes, commute incidents, holidays, shift changes, and service logs.
5. **Measure resolution:** If access is available, connect review themes to support tickets, incident data, or service-level metrics.
6. **Improve monitoring:** Track low-rating share, validated recurring themes, reply coverage, and time-to-resolution.
7. **Separate evidence from hypotheses:** Label findings as observed, inferred, or requiring validation.

## Summary

This project provides a reproducible path from employee commute app reviews to descriptive metrics, issue-discovery signals, and an interactive dashboard. Its value lies in helping a product or operations team decide **what to investigate next**, not in claiming that text heuristics or a small AI pilot have already established definitive root causes.

Use the full dataset for descriptive rating and response analysis, validate text-derived signals manually, and treat the classifier as a separate pilot until it has been evaluated against human-labelled data.

