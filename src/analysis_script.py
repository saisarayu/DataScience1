"""
============================================================
  Municipal Grievance System – Data Pipeline & Analysis
  File: src/analysis_script.py

  PIPELINE STAGES
  ───────────────
  Stage 1 │ Load raw data
  Stage 2 │ Clean  (missing values, duplicates, dates)
  Stage 3 │ Enrich (derived columns)
  Stage 4 │ Categorise (keyword-based tagging)
  Stage 5 │ Trend analysis (daily / weekly / monthly)
  Stage 6 │ Priority scoring (frequency + recency)
  Stage 7 │ Export processed dataset

Run:  python src/analysis_script.py
============================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────
RAW_PATH  = Path("data/raw/grievances.csv")
OUT_PATH  = Path("data/processed/grievances_clean.csv")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("  MUNICIPAL GRIEVANCE ANALYSIS PIPELINE")
print("=" * 60)

# ══════════════════════════════════════════════════════════
# STAGE 1 – LOAD RAW DATA
# ══════════════════════════════════════════════════════════
print("\n[Stage 1] Loading raw data …")

df = pd.read_csv(RAW_PATH)
print(f"  Shape : {df.shape}")
print(f"  Dtypes:\n{df.dtypes.to_string()}")

# ══════════════════════════════════════════════════════════
# STAGE 2 – DATA CLEANING
# ══════════════════════════════════════════════════════════
print("\n[Stage 2] Cleaning data …")

# ── 2a. Date columns → datetime ───────────────────────────
for col in ["filed_on", "resolved_on"]:
    df[col] = pd.to_datetime(df[col], errors="coerce")  # bad values → NaT

# ── 2b. Report missing values before imputation ───────────
missing_report = df.isnull().sum()
missing_pct    =  (df.isnull().mean() * 100).round(2)
print("\n  Missing values (before cleaning):")
print(pd.DataFrame({"count": missing_report, "pct": missing_pct})
      .query("count > 0").to_string())

# ── 2c. Impute / fill missing values ─────────────────────
#   ward     → fill with mode (most common ward)
df["ward"]    = df["ward"].fillna(df["ward"].mode()[0])

#   channel  → fill with "Unknown"
df["channel"] = df["channel"].fillna("Unknown")

#   citizen_age → fill with median age (ignore outliers)
median_age = np.nanmedian(df["citizen_age"].dropna().values)
df["citizen_age"] = df["citizen_age"].fillna(median_age).astype(int)

#   satisfaction_score → only valid when resolved; leave NaT for open cases
# (no imputation needed here)

# ── 2d. Remove duplicate complaint IDs ────────────────────
before = len(df)
df.drop_duplicates(subset=["complaint_id"], keep="first", inplace=True)
after  = len(df)
print(f"\n  Duplicates removed : {before - after}")

# ── 2e. Strip whitespace from all string columns ──────────
str_cols = df.select_dtypes("object").columns
df[str_cols] = df[str_cols].apply(lambda s: s.str.strip())

# ── 2f. Standardise category & status casing ─────────────
df["category"] = df["category"].str.title()
df["status"]   = df["status"].str.title()
df["priority"] = df["priority"].str.title()

print(f"\n  Data shape after cleaning : {df.shape}")

# ══════════════════════════════════════════════════════════
# STAGE 3 – FEATURE ENGINEERING / ENRICHMENT
# ══════════════════════════════════════════════════════════
print("\n[Stage 3] Feature engineering …")

# Resolution time (in days)
df["resolution_days"] = (df["resolved_on"] - df["filed_on"]).dt.days

# Time-based features from filed_on
df["filed_year"]    = df["filed_on"].dt.year
df["filed_month"]   = df["filed_on"].dt.month
df["filed_month_name"] = df["filed_on"].dt.strftime("%b %Y")   # e.g. "Jan 2025"
df["filed_week"]    = df["filed_on"].dt.isocalendar().week.astype(int)
df["filed_weekday"] = df["filed_on"].dt.day_name()             # Monday, Tuesday …
df["filed_date"]    = df["filed_on"].dt.date                   # date-only

# Is complaint still open?
df["is_open"] = df["status"].isin(["Open", "In Progress"])

print("  New columns:", ["resolution_days", "filed_month_name",
                         "filed_weekday", "is_open"])

# ══════════════════════════════════════════════════════════
# STAGE 4 – KEYWORD-BASED CATEGORISATION (fallback tagger)
# ══════════════════════════════════════════════════════════
print("\n[Stage 4] Keyword categorisation …")

KEYWORD_MAP = {
    "Water Supply"         : ["water", "pipeline", "tap", "meter", "pressure", "leakage"],
    "Waste Management"     : ["garbage", "waste", "dustbin", "dumping", "trash", "litter"],
    "Road & Infrastructure": ["road", "pothole", "footpath", "highway", "construction", "waterlogged"],
    "Street Lighting"      : ["light", "electric", "voltage", "power", "street light", "short circuit"],
    "Sewage & Drainage"    : ["sewage", "drain", "drainage", "manhole", "sewer", "flood"],
    "Public Property"      : ["park", "tree", "bench", "toilet", "stray", "encroachment"],
    "Noise & Pollution"    : ["noise", "smoke", "dust", "pollution", "burning", "foul smell"],
}


def categorise_by_keyword(text: str) -> str:
    """
    Returns the best-matching category based on keyword hits.
    Falls back to 'Other' if no match found.
    """
    text = str(text).lower()
    for cat, keywords in KEYWORD_MAP.items():
        if any(kw in text for kw in keywords):
            return cat
    return "Other"


# Apply only where category is missing (in this dataset it shouldn't be,
# but in real ingestion it often will be)
mask = df["category"].isna() | (df["category"].str.strip() == "")
df.loc[mask, "category"] = df.loc[mask, "description"].apply(categorise_by_keyword)

# Also create a 'verified_category' by cross-checking description keywords
df["verified_category"] = df["description"].apply(categorise_by_keyword)

print(f"  Category distribution:\n{df['category'].value_counts().to_string()}")

# ══════════════════════════════════════════════════════════
# STAGE 5 – TREND ANALYSIS
# ══════════════════════════════════════════════════════════
print("\n[Stage 5] Trend analysis …")

# ── 5a. Daily complaint volume ────────────────────────────
daily_trend = (
    df.groupby("filed_date")
    .size()
    .reset_index(name="count")
    .sort_values("filed_date")
)
print(f"\n  Daily average complaints : {daily_trend['count'].mean():.1f}")
print(f"  Peak day               : {daily_trend.loc[daily_trend['count'].idxmax(), 'filed_date']}"
      f" ({daily_trend['count'].max()} complaints)")

# ── 5b. Weekly trend ──────────────────────────────────────
weekly_trend = (
    df.groupby(["filed_year", "filed_week"])
    .size()
    .reset_index(name="count")
)

# ── 5c. Monthly trend ─────────────────────────────────────
monthly_trend = (
    df.groupby("filed_month_name")
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
)
print(f"\n  Top 3 busiest months:\n{monthly_trend.head(3).to_string(index=False)}")

# ── 5d. Category trend (monthly) ─────────────────────────
category_monthly = (
    df.groupby(["filed_month_name", "category"])
    .size()
    .reset_index(name="count")
)

# ── 5e. Complaints by weekday (helps with staffing) ───────
weekday_order = ["Monday", "Tuesday", "Wednesday",
                 "Thursday", "Friday", "Saturday", "Sunday"]
weekday_trend = (
    df.groupby("filed_weekday")
    .size()
    .reindex(weekday_order)
    .reset_index(name="count")
)
print(f"\n  Complaints by weekday:\n{weekday_trend.to_string(index=False)}")

# ── 5f. Ward-level complaint volume ──────────────────────
ward_trend = (
    df.groupby("ward")
    .agg(
        total     = ("complaint_id", "count"),
        open_count= ("is_open",       "sum"),
        avg_resolution = ("resolution_days", "mean"),
    )
    .round(1)
    .sort_values("total", ascending=False)
    .reset_index()
)
print(f"\n  Top 5 wards by complaint volume:\n{ward_trend.head(5).to_string(index=False)}")

# ── 5g. Resolution time by category ──────────────────────
resolution_by_cat = (
    df[df["resolution_days"].notna()]
    .groupby("category")["resolution_days"]
    .agg(["mean", "median", "max"])
    .round(1)
    .rename(columns={"mean": "avg_days", "median": "median_days", "max": "max_days"})
    .sort_values("avg_days", ascending=False)
)
print(f"\n  Resolution time by category:\n{resolution_by_cat.to_string()}")

# ══════════════════════════════════════════════════════════
# STAGE 6 – PRIORITY SCORING
#   Score = (frequency_score × 0.5) + (recency_score × 0.3)
#                                    + (open_ratio × 0.2)
# ══════════════════════════════════════════════════════════
print("\n[Stage 6] Priority scoring …")

REF_DATE = df["filed_on"].max()   # latest date in dataset = "today"

# ── Per-complaint recency score (0-100, higher = more recent) ─────────
max_age_days = (REF_DATE - df["filed_on"]).dt.days.max()
df["recency_score"] = (
    1 - (REF_DATE - df["filed_on"]).dt.days / max_age_days
) * 100

# ── Category-level priority summary ──────────────────────
cat_stats = (
    df.groupby("category")
    .agg(
        frequency       = ("complaint_id",   "count"),
        open_count      = ("is_open",         "sum"),
        avg_recency     = ("recency_score",   "mean"),
    )
    .reset_index()
)

# Normalise frequency (0-100)
cat_stats["freq_norm"] = (
    (cat_stats["frequency"] - cat_stats["frequency"].min()) /
    (cat_stats["frequency"].max() - cat_stats["frequency"].min()) * 100
)

# Open ratio (%)
cat_stats["open_ratio"] = (
    cat_stats["open_count"] / cat_stats["frequency"] * 100
).round(1)

# Composite priority score
cat_stats["priority_score"] = (
    cat_stats["freq_norm"]  * 0.50 +
    cat_stats["avg_recency"]* 0.30 +
    cat_stats["open_ratio"] * 0.20
).round(2)

cat_stats = cat_stats.sort_values("priority_score", ascending=False)

print("\n  Category Priority Scores (higher = needs more attention):")
print(cat_stats[["category", "frequency", "open_count",
                  "open_ratio", "priority_score"]].to_string(index=False))

# ── Ward-level priority ───────────────────────────────────
ward_stats = (
    df.groupby("ward")
    .agg(
        frequency   = ("complaint_id", "count"),
        open_count  = ("is_open",       "sum"),
        avg_recency = ("recency_score", "mean"),
    )
    .reset_index()
)
ward_stats["freq_norm"] = (
    (ward_stats["frequency"] - ward_stats["frequency"].min()) /
    (ward_stats["frequency"].max() - ward_stats["frequency"].min()) * 100
)
ward_stats["open_ratio"] = (
    ward_stats["open_count"] / ward_stats["frequency"] * 100
).round(1)
ward_stats["priority_score"] = (
    ward_stats["freq_norm"]  * 0.50 +
    ward_stats["avg_recency"]* 0.30 +
    ward_stats["open_ratio"] * 0.20
).round(2)
ward_stats = ward_stats.sort_values("priority_score", ascending=False)

print("\n  Top 5 Priority Wards:")
print(ward_stats.head(5)[["ward", "frequency", "open_count",
                           "open_ratio", "priority_score"]].to_string(index=False))

# ── Flag critical open complaints (open > 30 days, priority=High/Critical) ──
df["days_open"] = np.where(
    df["is_open"],
    (REF_DATE - df["filed_on"]).dt.days,
    np.nan
)

critical_backlog = df[
    (df["is_open"]) &
    (df["days_open"] >= 30) &
    (df["priority"].isin(["High", "Critical"]))
].sort_values("days_open", ascending=False)

print(f"\n  ⚠️  Critical open complaints (≥30 days, High/Critical priority): "
      f"{len(critical_backlog)}")

# ══════════════════════════════════════════════════════════
# STAGE 7 – EXPORT PROCESSED DATASET
# ══════════════════════════════════════════════════════════
print("\n[Stage 7] Saving processed data …")

df.to_csv(OUT_PATH, index=False)

# Also save summary tables for dashboard use
cat_stats.to_csv("data/processed/category_priority.csv", index=False)
ward_stats.to_csv("data/processed/ward_priority.csv", index=False)
monthly_trend.to_csv("data/processed/monthly_trend.csv", index=False)
weekday_trend.to_csv("data/processed/weekday_trend.csv", index=False)
resolution_by_cat.to_csv("data/processed/resolution_by_category.csv")
critical_backlog.to_csv("data/processed/critical_backlog.csv", index=False)

print(f"  ✅ Processed dataset → {OUT_PATH}")
print(f"  ✅ Summary tables    → data/processed/")
print(f"\n{'='*60}")
print("  PIPELINE COMPLETE")
print(f"{'='*60}")

# ── Quick KPI summary for reference ──────────────────────
total        = len(df)
open_pct     = df["is_open"].mean() * 100
resolved_pct = (df["status"] == "Resolved").mean() * 100
avg_res_days = df["resolution_days"].mean()

print(f"""
  KEY METRICS
  ───────────────────────────────────────────
  Total complaints      : {total:,}
  Open / In Progress    : {open_pct:.1f}%
  Resolved              : {resolved_pct:.1f}%
  Avg resolution time   : {avg_res_days:.1f} days
  Critical backlog      : {len(critical_backlog)} complaints
  ───────────────────────────────────────────
""")