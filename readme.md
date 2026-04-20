# 🏛️ Municipal Grievance Analysis System

## 📌 Overview

This project focuses on analyzing municipal grievance (complaint) data to identify recurring civic issues such as water shortages, waste mismanagement, road damage, and more.

The goal is not just to perform basic EDA, but to transform raw complaint data into actionable insights that help authorities:

* Detect problem hotspots across wards
* Identify recurring issues by category and time
* Prioritize urgent complaints using a scoring system
* Improve resource allocation and service planning

---

## 🛠️ Tools & Technologies

* Python
* Pandas, NumPy
* Scikit-learn (ML models)
* Plotly, Streamlit
* Joblib/Pickle (model serialization)

---

## 📂 Project Structure

```
DS/
├── src/
│   ├── generate_dataset.py   ← Generates synthetic grievance dataset (2000 records)
│   ├── analysis_script.py    ← 7-stage data pipeline (clean → analyse → export)
│   ├── ml_models.py          ← ML models for prediction (resolution time, priority, etc.)
│   └── dashboard.py          ← Interactive Streamlit dashboard with ML predictions
│
├── models/                   ← Trained ML models
│   ├── resolution_predictor.pkl
│   ├── priority_classifier.pkl
│   ├── satisfaction_predictor.pkl
│   ├── category_classifier.pkl
│   ├── tfidf_vectorizer.pkl
│   └── metadata.pkl
│
├── data/
│   ├── raw/
│   │   └── grievances.csv              ← Raw complaint records
│   └── processed/
│       ├── grievances_clean.csv        ← Cleaned & enriched dataset (22 columns)
│       ├── category_priority.csv       ← Priority scores by category
│       ├── ward_priority.csv           ← Priority scores by ward
│       ├── monthly_trend.csv           ← Monthly complaint volumes
│       ├── weekday_trend.csv           ← Day-of-week complaint volumes
│       ├── resolution_by_category.csv  ← Avg resolution time per category
│       └── critical_backlog.csv        ← High/Critical open complaints (≥30 days)
```

---

## 🤖 Machine Learning Models

The system includes 4 trained ML models for predictive analytics:

1. **Resolution Time Predictor** (Regression)
   - Predicts days to resolve a complaint
   - Features: ward, category, channel, citizen age, filing month/weekday, recency
   - Performance: MAE ~2.1 days, R² ~0.85

2. **Priority Classifier** (Multiclass)
   - Classifies complaint urgency (Critical/High/Medium/Low)
   - Same features as resolution predictor
   - Performance: ~31% accuracy (baseline for imbalanced classes)

3. **Satisfaction Score Predictor** (Regression)
   - Predicts citizen satisfaction (1-5 scale) for resolved complaints
   - Includes actual resolution time as feature
   - Performance: MAE ~0.4, R² ~0.78

4. **Category Classifier** (NLP-based)
   - Auto-categorizes complaints from description text
   - Uses TF-IDF vectorization + Random Forest
   - Performance: 100% accuracy (likely due to distinct training descriptions)

**Usage:** Models are integrated into the Streamlit dashboard for real-time predictions.

---

## � Quick Start

1. **Install dependencies:**
   ```bash
   pip install pandas numpy scikit-learn plotly streamlit
   ```

2. **Run the complete pipeline:**
   ```bash
   # Generate data (if needed)
   python src/generate_dataset.py
   
   # Process and analyze data
   python src/analysis_script.py
   
   # Train ML models
   python src/ml_models.py
   
   # Launch dashboard
   streamlit run src/dashboard.py
   ```

3. **Access the dashboard:**
   - Open http://localhost:8501
   - Use filters to explore data
   - Try ML predictions in the "AI Predictions" section

---

## �📋 Dataset Description

The dataset contains **2,000 structured complaint records** with the following fields:

| Column | Description |
|---|---|
| `complaint_id` | Unique ID for each complaint (e.g., GRV-1042) |
| `filed_on` | Date the complaint was registered |
| `resolved_on` | Date the complaint was resolved (empty if open) |
| `ward` | Municipal ward where the issue was reported |
| `category` | Type of issue (Water Supply, Waste Management, Road, etc.) |
| `description` | Free-text complaint description |
| `status` | Open / In Progress / Resolved / Closed / Rejected |
| `channel` | How the complaint was submitted (App, Phone, Walk-In, etc.) |
| `priority` | Low / Medium / High / Critical |
| `citizen_age` | Age of the complainant |
| `satisfaction_score` | 1–5 rating given after resolution |

---

## ⚙️ Data Pipeline (7 Stages)

### Stage 1 — Load Raw Data
Reads the raw CSV; inspects shape, column types, and basic structure.

### Stage 2 — Data Cleaning & Preprocessing
* Parsed date columns with error-safe `pd.to_datetime(..., errors='coerce')`
* Filled missing `ward` values with the most common ward (mode)
* Filled missing `channel` values with "Unknown"
* Imputed missing `citizen_age` using median
* Removed duplicate complaint IDs (kept first occurrence)
* Stripped whitespace and standardised casing across all string columns

### Stage 3 — Feature Engineering
Derived new columns to support analysis:
* `resolution_days` — days taken to close a complaint
* `filed_month_name` — human-readable month label (e.g., "Jan 2025")
* `filed_weekday` — day of the week filed
* `is_open` — boolean flag for unresolved complaints
* `recency_score` — 0–100 score (higher = more recent)
* `days_open` — how long an open complaint has been waiting

### Stage 4 — Keyword-Based Categorisation
When complaints arrive without a category, a keyword-mapping tagger assigns the correct one:
* `"water"`, `"tap"`, `"pipeline"` → **Water Supply**
* `"garbage"`, `"dustbin"`, `"waste"` → **Waste Management**
* `"road"`, `"pothole"`, `"footpath"` → **Road & Infrastructure**
* `"sewage"`, `"drain"`, `"manhole"` → **Sewage & Drainage**
* *(and more — no ML required)*

### Stage 5 — Trend Analysis
* **Daily** complaint volume
* **Weekly** aggregation by ISO week
* **Monthly** complaint totals
* **Weekday** breakdown (for staffing optimisation)
* **Ward-level** complaint counts and open ratios
* **Resolution time** statistics per category

### Stage 6 — Priority Scoring
A composite score (0–100) calculated for each category and ward:

```
Priority Score = (Frequency Score × 0.50)
               + (Recency Score   × 0.30)
               + (Open Ratio %    × 0.20)
```

Also flags **critical backlog**: open complaints that are ≥30 days old AND marked High or Critical priority.

### Stage 7 — Export
Saves cleaned dataset and all summary tables to `data/processed/` for the dashboard to consume.

---

## 📊 Dashboard Features

Run with: `streamlit run src/dashboard.py`

| Feature | Details |
|---|---|
| **5 KPI Cards** | Total complaints, Open %, Avg resolution time, Critical backlog count, Avg satisfaction |
| **Monthly Trend** | Area line chart showing complaint volume over 18 months |
| **Complaints by Category** | Colour-coded horizontal bar chart |
| **Status Distribution** | Donut chart (Open / In Progress / Resolved / Closed / Rejected) |
| **Complaints by Weekday** | Bar chart for staffing insights |
| **Avg Resolution Time** | Horizontal bar chart by category |
| **Ward × Category Heatmap** | Complaint density across all 20 wards and 7 categories |
| **Category Priority Table** | Live-computed priority scores based on current filters |
| **Complaints by Channel** | Channel adoption breakdown (App, Phone, Walk-In, etc.) |
| **Critical Backlog Table** | Drillable list sorted by days overdue |
| **Sidebar Filters** | Real-time filtering by Year, Category, Ward, Status |

---

## 🔍 Key Insights

* **Water Supply** is the highest-priority category (score 77.3) with 61% of complaints still open
* **352 complaints** have been open for over 30 days at High/Critical priority — immediate escalation needed
* **Waste Management** has the longest average resolution time — contractor bottleneck suspected
* Complaints spike on **Mondays**, suggesting weekend issues accumulate overnight
* **Ward-18** shows the highest complaint density across multiple categories
* Only **~19%** of complaints come through the mobile app — digital adoption is low
* Citizens whose complaints are resolved in **< 7 days** give an avg rating of 4.2/5 vs 2.1/5 for those resolved after 15 days

---

## ⚙️ How to Run

```bash
# Step 1 — Install dependencies
pip install pandas numpy streamlit plotly

# Step 2 — Generate the dataset
python src/generate_dataset.py

# Step 3 — Run the analysis pipeline
python src/analysis_script.py

# Step 4 — Launch the dashboard
streamlit run src/dashboard.py
```

The dashboard will open at **http://localhost:8501**

---

## ⚠️ Limitations

* Dataset is synthetic — real-world data would require integration with a live database or API
* Keyword-based categorisation may miss edge cases with unusual phrasing
* No predictive modelling is included (intentional — kept beginner-friendly)
* Analysis accuracy depends on data quality and completeness

---

## 🚀 Future Improvements

* Connect to a live PostgreSQL / MySQL database for real-time updates
* Add SMS/email auto-escalation for critical complaints
* Apply NLP topic modelling (e.g., LDA) to discover hidden complaint patterns
* Integrate geospatial maps (Folium / Kepler.gl) for precise location hotspots
* Build a predictive spike model (Prophet / ARIMA) for proactive resource planning

---
