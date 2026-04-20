"""
============================================================
  Municipal Grievance System – Streamlit Dashboard v2.0
  File: src/dashboard.py
  Run:  streamlit run src/dashboard.py
============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import pickle
import sys
import os

# Add src to path for imports
sys.path.append(os.path.dirname(__file__))

# ── Load ML Models ─────────────────────────────────────────
MODEL_DIR = Path("models")
try:
    with open(MODEL_DIR / 'metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    with open(MODEL_DIR / 'priority_classifier.pkl', 'rb') as f:
        pri_model = pickle.load(f)
    with open(MODEL_DIR / 'resolution_predictor.pkl', 'rb') as f:
        res_model = pickle.load(f)
    with open(MODEL_DIR / 'satisfaction_predictor.pkl', 'rb') as f:
        sat_model = pickle.load(f)
    with open(MODEL_DIR / 'category_classifier.pkl', 'rb') as f:
        cat_model = pickle.load(f)
    with open(MODEL_DIR / 'tfidf_vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    encoders = metadata['encoders']
    models_loaded = True
except:
    models_loaded = False
    st.warning("⚠️ ML models not found. Run `python src/ml_models.py` to train models first.")

# ── ML Prediction Functions ────────────────────────────────
def predict_priority(complaint_data):
    """Predict priority level for a new complaint"""
    if not models_loaded:
        return "Model not loaded"
    
    input_data = pd.DataFrame([complaint_data])
    input_data['ward_encoded'] = encoders['ward'].transform([input_data['ward'].iloc[0]])
    input_data['category_encoded'] = encoders['category'].transform([input_data['category'].iloc[0]])
    input_data['channel_encoded'] = encoders['channel'].transform([input_data['channel'].iloc[0]])
    input_data['weekday_encoded'] = encoders['weekday'].transform([str(input_data['filed_weekday'].iloc[0])])

    input_features = input_data[['ward_encoded', 'category_encoded', 'channel_encoded', 'weekday_encoded',
                                'citizen_age', 'filed_month', 'recency_score']].fillna(
        input_data[['ward_encoded', 'category_encoded', 'channel_encoded', 'weekday_encoded',
                   'citizen_age', 'filed_month', 'recency_score']].mean()
    )

    pred_encoded = pri_model.predict(input_features)[0]
    return encoders['priority'].inverse_transform([pred_encoded])[0]

def predict_resolution_time(complaint_data):
    """Predict resolution time for a new complaint"""
    if not models_loaded or res_model is None:
        return "Model not available"
    
    input_data = pd.DataFrame([complaint_data])
    input_data['ward_encoded'] = encoders['ward'].transform([input_data['ward'].iloc[0]])
    input_data['category_encoded'] = encoders['category'].transform([input_data['category'].iloc[0]])
    input_data['channel_encoded'] = encoders['channel'].transform([input_data['channel'].iloc[0]])
    input_data['weekday_encoded'] = encoders['weekday'].transform([str(input_data['filed_weekday'].iloc[0])])

    input_features = input_data[['ward_encoded', 'category_encoded', 'channel_encoded', 'weekday_encoded',
                                'citizen_age', 'filed_month', 'recency_score']].fillna(
        input_data[['ward_encoded', 'category_encoded', 'channel_encoded', 'weekday_encoded',
                   'citizen_age', 'filed_month', 'recency_score']].mean()
    )

    return f"{res_model.predict(input_features)[0]:.1f} days"

def predict_category(description):
    """Predict category from complaint description"""
    if not models_loaded or cat_model is None or tfidf is None:
        return "Model not available"

    text_vec = tfidf.transform([description])
    pred_encoded = cat_model.predict(text_vec)[0]
    return encoders['category'].inverse_transform([pred_encoded])[0]

# ── Page configuration ─────────────────────────────────────
st.set_page_config(
    page_title  = "Municipal Grievance Dashboard",
    page_icon   = "🏛️",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ── Custom CSS ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', sans-serif;
    background-color: #0f172a;
    color: #e2e8f0;
}
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }

/* ── KPI Cards ── */
.kpi-card {
    border-radius: 14px;
    padding: 1.3rem 1.5rem;
    text-align: center;
    box-shadow: 0 6px 24px rgba(0,0,0,0.35);
    backdrop-filter: blur(10px);
    margin-bottom: 4px;
}
.kpi-card-blue  { background: linear-gradient(135deg,#1e3a5f,#0f172a); border:1px solid rgba(99,102,241,0.4); }
.kpi-card-orange{ background: linear-gradient(135deg,#431407,#0f172a); border:1px solid rgba(251,146,60,0.5); }
.kpi-card-green { background: linear-gradient(135deg,#052e16,#0f172a); border:1px solid rgba(52,211,153,0.4); }
.kpi-card-red   { background: linear-gradient(135deg,#450a0a,#0f172a); border:1px solid rgba(248,113,113,0.5); }
.kpi-card-teal  { background: linear-gradient(135deg,#042f2e,#0f172a); border:1px solid rgba(45,212,191,0.4); }
.kpi-value { font-size: 2rem; font-weight: 700; line-height: 1.15; }
.kpi-label { font-size: 0.78rem; color: #94a3b8; margin-top: 5px; letter-spacing: 0.05em; text-transform: uppercase; }

/* ── Key Insights Banner ── */
.insight-banner {
    background: linear-gradient(135deg, rgba(30,27,75,0.95), rgba(15,23,42,0.98));
    border: 1px solid rgba(99,102,241,0.5);
    border-left: 5px solid #6366f1;
    border-radius: 14px;
    padding: 1.2rem 1.6rem;
    margin-bottom: 1.2rem;
}
.insight-banner h3 { margin:0 0 0.7rem 0; color:#c7d2fe; font-size:1.05rem; }
.insight-item { 
    padding: 0.35rem 0; 
    color:#e2e8f0; 
    font-size: 0.92rem;
    border-bottom: 1px solid rgba(255,255,255,0.05);
}
.insight-item:last-child { border-bottom: none; }

/* ── Top Problems ── */
.top-problem {
    background: rgba(30,41,59,0.8);
    border-left: 4px solid;
    border-radius: 8px;
    padding: 0.7rem 1rem;
    margin: 6px 0;
    font-size: 0.9rem;
}
.rank-1 { border-color: #f87171; }
.rank-2 { border-color: #fb923c; }
.rank-3 { border-color: #facc15; }

/* ── Section Headers ── */
.section-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #c7d2fe;
    border-left: 4px solid #6366f1;
    padding-left: 10px;
    margin: 1.5rem 0 0.8rem 0;
    letter-spacing: 0.02em;
}

/* ── Recommendation Box ── */
.rec-box {
    background: linear-gradient(135deg, rgba(5,46,22,0.8), rgba(15,23,42,0.95));
    border: 1px solid rgba(52,211,153,0.35);
    border-left: 5px solid #34d399;
    border-radius: 14px;
    padding: 1.2rem 1.6rem;
    margin-top: 0.5rem;
}
.rec-box h3 { color:#6ee7b7; margin:0 0 0.7rem 0; font-size:1rem; }
.rec-item { padding: 0.3rem 0; color:#d1fae5; font-size:0.9rem; }

/* ── Worst Ward Box ── */
.ward-box {
    background: linear-gradient(135deg, rgba(69,10,10,0.8), rgba(15,23,42,0.95));
    border: 1px solid rgba(248,113,113,0.4);
    border-radius: 12px;
    padding: 1rem 1.4rem;
    text-align: center;
}
.ward-name { font-size:1.6rem; font-weight:700; color:#f87171; }
.ward-sub  { font-size:0.8rem; color:#fca5a5; margin-top:4px; text-transform:uppercase; letter-spacing:0.05em; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#1e1b4b 0%,#0f172a 100%);
}
section[data-testid="stSidebar"] .stMultiSelect [data-baseweb="tag"] {
    background-color: rgba(99,102,241,0.3) !important;
}

/* ── Divider spacing ── */
hr { border-color: rgba(255,255,255,0.07); margin: 1.2rem 0; }
</style>
""", unsafe_allow_html=True)

PLOTLY_THEME = "plotly_dark"

# ══════════════════════════════════════════════════════════
# DATA LOADER
# ══════════════════════════════════════════════════════════
@st.cache_data(show_spinner="Loading grievance data …")
def load_data():
    df = pd.read_csv("data/processed/grievances_clean.csv",
                     parse_dates=["filed_on", "resolved_on"])
    df["filed_date"]     = pd.to_datetime(df["filed_date"])
    df["is_open"]        = df["is_open"].astype(bool)
    df["filed_month_dt"] = df["filed_on"].dt.to_period("M").dt.to_timestamp()
    return df

try:
    df_full = load_data()
except FileNotFoundError:
    st.error("⚠️  Processed data not found. Run `generate_dataset.py` then `analysis_script.py` first.")
    st.stop()

# ══════════════════════════════════════════════════════════
# SIDEBAR — FILTERS
# ══════════════════════════════════════════════════════════
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/city-hall.png", width=64)
    st.markdown("### 🏛️ Grievance Monitor")
    st.markdown("---")

    years = sorted(df_full["filed_year"].dropna().unique().astype(int), reverse=True)
    sel_years = st.multiselect("📅 Year", years, default=years)

    all_cats = sorted(df_full["category"].unique())
    sel_cats = st.multiselect("📂 Category", all_cats, default=all_cats)

    all_wards = sorted(df_full["ward"].unique())
    sel_wards = st.multiselect("📍 Ward", all_wards, default=all_wards)

    all_status = sorted(df_full["status"].unique())
    sel_status = st.multiselect("🔖 Status", all_status, default=all_status)

    st.markdown("---")
    st.caption("Municipal Grievance Dashboard v2.0")

# ── Apply filters ──────────────────────────────────────────
df = df_full[
    df_full["filed_year"].isin(sel_years) &
    df_full["category"].isin(sel_cats) &
    df_full["ward"].isin(sel_wards) &
    df_full["status"].isin(sel_status)
].copy()

# ── Pre-compute key metrics ────────────────────────────────
total          = len(df)
open_pct       = df["is_open"].mean() * 100 if total else 0
avg_res        = df["resolution_days"].mean() if total else 0
avg_sat        = df["satisfaction_score"].mean()

backlog_df = df[
    df["is_open"] &
    df.get("days_open", pd.Series(dtype=float)).fillna(0).ge(30) &
    df["priority"].isin(["High", "Critical"])
] if "days_open" in df.columns else df[
    df["is_open"] & df["priority"].isin(["High", "Critical"])
]
critical_count = len(backlog_df)

top_cats = df["category"].value_counts().reset_index(name="count").rename(
    columns={"index": "category"})

worst_ward = (
    df.groupby("ward").size().idxmax() if total else "N/A"
)
worst_ward_count = df["ward"].value_counts().iloc[0] if total else 0

avg_res_by_cat = (
    df[df["resolution_days"].notna()]
    .groupby("category")["resolution_days"].mean()
)
slowest_cat     = avg_res_by_cat.idxmax() if not avg_res_by_cat.empty else "N/A"
slowest_days    = avg_res_by_cat.max() if not avg_res_by_cat.empty else 0
overall_avg_res = avg_res_by_cat.mean() if not avg_res_by_cat.empty else 0

# ══════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════
st.markdown("## 🏛️ Municipal Grievance Analysis Dashboard")
st.markdown(
    f"Showing **{total:,}** complaints &nbsp;·&nbsp; "
    f"**{len(sel_wards)}** Wards &nbsp;·&nbsp; "
    f"**{len(sel_cats)}** Categories"
)
st.markdown("---")

# ══════════════════════════════════════════════════════════
# SECTION 1 — 🚨 KEY INSIGHTS (top of page)
# ══════════════════════════════════════════════════════════
top1_cat   = top_cats.iloc[0]["category"] if len(top_cats) > 0 else "N/A"
top1_count = int(top_cats.iloc[0]["count"])  if len(top_cats) > 0 else 0
top2_cat   = top_cats.iloc[1]["category"] if len(top_cats) > 1 else "N/A"
top2_count = int(top_cats.iloc[1]["count"])  if len(top_cats) > 1 else 0

slowest_ratio = (slowest_days / overall_avg_res) if overall_avg_res > 0 else 1

st.markdown(f"""
<div class="insight-banner">
    <h3>🚨 Key Insights — What the Data is Telling You</h3>
    <div class="insight-item">
        💧 <b>{top1_cat}</b> has the highest complaints — <b>{top1_count} cases</b> and counting
    </div>
    <div class="insight-item">
        📍 <b>{worst_ward}</b> is the top complaint hotspot with <b>{worst_ward_count} issues</b> — needs immediate ward-level attention
    </div>
    <div class="insight-item">
        🔴 <b>{critical_count} complaints</b> are pending for more than 30 days — these require urgent escalation
    </div>
    <div class="insight-item">
        🐢 <b>{slowest_cat}</b> has the highest resolution time at <b>{slowest_days:.1f} days</b> — 
        that's <b>{slowest_days/overall_avg_res:.1f}×</b> slower than the overall average ({overall_avg_res:.1f} days)
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# SECTION 2 — KPI CARDS (meaningful labels + color logic)
# ══════════════════════════════════════════════════════════
k1, k2, k3, k4, k5 = st.columns(5)

with k1:
    st.markdown(f"""
    <div class="kpi-card kpi-card-blue">
        <div class="kpi-value" style="color:#818cf8">{total:,}</div>
        <div class="kpi-label">Total Complaints Recorded</div>
    </div>""", unsafe_allow_html=True)

with k2:
    st.markdown(f"""
    <div class="kpi-card kpi-card-orange">
        <div class="kpi-value" style="color:#fb923c">{open_pct:.1f}%</div>
        <div class="kpi-label">Complaints Still Unresolved ⚠️</div>
    </div>""", unsafe_allow_html=True)

with k3:
    res_color = "#34d399" if avg_res <= 10 else "#fb923c" if avg_res <= 18 else "#f87171"
    st.markdown(f"""
    <div class="kpi-card kpi-card-green">
        <div class="kpi-value" style="color:{res_color}">{avg_res:.1f} days</div>
        <div class="kpi-label">Avg Time to Resolve a Complaint</div>
    </div>""", unsafe_allow_html=True)

with k4:
    st.markdown(f"""
    <div class="kpi-card kpi-card-red">
        <div class="kpi-value" style="color:#f87171">{critical_count}</div>
        <div class="kpi-label">Critical Cases — Action Required 🚨</div>
    </div>""", unsafe_allow_html=True)

with k5:
    sat_color = "#34d399" if not np.isnan(avg_sat) and avg_sat >= 3.5 else "#fb923c"
    sat_text  = f"⭐ {avg_sat:.1f} / 5" if not np.isnan(avg_sat) else "N/A"
    st.markdown(f"""
    <div class="kpi-card kpi-card-teal">
        <div class="kpi-value" style="color:{sat_color}">{sat_text}</div>
        <div class="kpi-label">Citizen Satisfaction Score</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# SECTION 2.5 — 🤖 ML PREDICTIONS
# ══════════════════════════════════════════════════════════
if models_loaded:
    st.markdown('<div class="section-title">🤖 AI Predictions — What Will Happen Next?</div>', unsafe_allow_html=True)
    
    col_ml1, col_ml2 = st.columns(2)
    
    with col_ml1:
        st.markdown("#### 📝 Predict Category from Description")
        desc_input = st.text_area(
            "Enter complaint description:",
            "Water supply is not working in my area for 3 days",
            height=100
        )
        if st.button("🔍 Predict Category"):
            predicted_cat = predict_category(desc_input)
            st.success(f"**Predicted Category:** {predicted_cat}")
    
    with col_ml2:
        st.markdown("#### 🎯 Predict Complaint Priority")
        with st.form("priority_form"):
            ward_options = sorted(encoders['ward'].classes_)
            cat_options = sorted(encoders['category'].classes_)
            channel_options = sorted(encoders['channel'].classes_)
            weekday_options = sorted(encoders['weekday'].classes_)
            
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                ward_sel = st.selectbox("Ward", ward_options, index=0)
                category_sel = st.selectbox("Category", cat_options, index=0)
                age_sel = st.number_input("Citizen Age", min_value=18, max_value=100, value=35)
            
            with col_p2:
                channel_sel = st.selectbox("Channel", channel_options, index=0)
                weekday_sel = st.selectbox("Filed Weekday", weekday_options, index=2)  # Wednesday
                month_sel = st.number_input("Filed Month", min_value=1, max_value=12, value=4)
                recency_sel = st.number_input("Recency Score", min_value=0.0, max_value=100.0, value=50.0)
            
            submitted = st.form_submit_button("🔮 Predict Priority & Resolution Time")
            
            if submitted:
                complaint_data = {
                    'ward': ward_sel,
                    'category': category_sel,
                    'channel': channel_sel,
                    'citizen_age': age_sel,
                    'filed_month': month_sel,
                    'filed_weekday': weekday_sel,
                    'recency_score': recency_sel
                }
                
                pred_priority = predict_priority(complaint_data)
                pred_resolution = predict_resolution_time(complaint_data)
                
                st.success(f"**Predicted Priority:** {pred_priority}")
                st.info(f"**Estimated Resolution Time:** {pred_resolution}")

# ══════════════════════════════════════════════════════════
# SECTION 3 — TOP 3 PROBLEMS + WORST WARD
# ══════════════════════════════════════════════════════════
st.markdown('<div class="section-title">🏆 Top Issues & Worst Affected Area</div>', unsafe_allow_html=True)

col_top, col_ward = st.columns([3, 1])

with col_top:
    rank_classes = ["rank-1", "rank-2", "rank-3"]
    rank_medals  = ["🥇", "🥈", "🥉"]
    html_blocks  = ""
    for i, row in top_cats.head(3).iterrows():
        idx = list(top_cats.head(3).index).index(i)
        pct = row['count'] / total * 100
        html_blocks += f"""
        <div class="top-problem {rank_classes[idx]}">
            {rank_medals[idx]} <b>{row['category']}</b>
            &nbsp;—&nbsp; <b>{int(row['count'])} complaints</b>
            <span style="color:#94a3b8; font-size:0.82rem;"> ({pct:.1f}% of total)</span>
        </div>"""
    st.markdown(html_blocks, unsafe_allow_html=True)

with col_ward:
    st.markdown(f"""
    <div class="ward-box">
        <div style="color:#fca5a5; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:4px;">
            📍 Worst Affected Area
        </div>
        <div class="ward-name">{worst_ward}</div>
        <div class="ward-sub">{worst_ward_count} Complaints</div>
        <div style="color:#f87171; font-size:0.78rem; margin-top:8px;">🔴 Highest Complaint Density</div>
    </div>""", unsafe_allow_html=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════
# SECTION 4 — COMPLAINT TRENDS
# ══════════════════════════════════════════════════════════
st.markdown('<div class="section-title">📈 Complaint Trends</div>', unsafe_allow_html=True)
col_l, col_r = st.columns([3, 2])

with col_l:
    monthly = (
        df.groupby("filed_month_dt").size()
        .reset_index(name="count").sort_values("filed_month_dt")
    )
    fig_monthly = px.area(
        monthly, x="filed_month_dt", y="count",
        title="Monthly Complaint Volume",
        labels={"filed_month_dt": "Month", "count": "Complaints"},
        template=PLOTLY_THEME, color_discrete_sequence=["#6366f1"],
    )
    fig_monthly.update_traces(line_width=2.5, fillcolor="rgba(99,102,241,0.12)")
    fig_monthly.update_layout(margin=dict(l=0,r=0,t=36,b=0), height=300)
    st.plotly_chart(fig_monthly, use_container_width=True)

with col_r:
    fig_cat = px.bar(
        top_cats.head(8), x="count", y="category",
        orientation="h", title="Complaints by Category",
        labels={"count": "Complaints", "category": ""},
        template=PLOTLY_THEME,
        color="count", color_continuous_scale="Viridis",
    )
    fig_cat.update_layout(margin=dict(l=0,r=0,t=36,b=0), height=300,
                          yaxis=dict(autorange="reversed"),
                          coloraxis_showscale=False)
    st.plotly_chart(fig_cat, use_container_width=True)

# ══════════════════════════════════════════════════════════
# SECTION 5 — OPERATIONAL BREAKDOWN
# ══════════════════════════════════════════════════════════
st.markdown('<div class="section-title">📊 Operational Breakdown</div>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

with col1:
    status_counts = df["status"].value_counts().reset_index(name="count")
    fig_status = px.pie(
        status_counts, names="status", values="count",
        title="Status Distribution",
        hole=0.55, template=PLOTLY_THEME,
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    fig_status.update_layout(margin=dict(l=0,r=0,t=36,b=0), height=300)
    st.plotly_chart(fig_status, use_container_width=True)

with col2:
    WDAY_ORDER = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    wday_counts = (
        df.groupby("filed_weekday").size()
        .reindex(WDAY_ORDER).reset_index(name="count")
    )
    fig_wday = px.bar(
        wday_counts, x="filed_weekday", y="count",
        title="Complaints by Day of Week",
        labels={"filed_weekday": "", "count": "Complaints"},
        template=PLOTLY_THEME,
        color="count", color_continuous_scale="Bluyl",
    )
    fig_wday.update_layout(margin=dict(l=0,r=0,t=36,b=0), height=300,
                           coloraxis_showscale=False)
    st.plotly_chart(fig_wday, use_container_width=True)

with col3:
    res_data = (
        df[df["resolution_days"].notna()]
        .groupby("category")["resolution_days"].mean()
        .reset_index().rename(columns={"resolution_days": "avg_days"})
        .sort_values("avg_days", ascending=True)
    )
    fig_res = px.bar(
        res_data, x="avg_days", y="category",
        orientation="h",
        title="Avg Resolution Time (days)",
        labels={"avg_days": "Days", "category": ""},
        template=PLOTLY_THEME,
        color="avg_days", color_continuous_scale="RdYlGn_r",
    )
    fig_res.update_layout(margin=dict(l=0,r=0,t=36,b=0), height=300,
                          coloraxis_showscale=False)
    st.plotly_chart(fig_res, use_container_width=True)

# ══════════════════════════════════════════════════════════
# SECTION 6 — WARD × CATEGORY HEATMAP
# ══════════════════════════════════════════════════════════
st.markdown('<div class="section-title">🗺️ Ward × Category Heatmap</div>', unsafe_allow_html=True)

heatmap_data = (
    df.groupby(["ward", "category"]).size()
    .reset_index(name="count")
    .pivot(index="ward", columns="category", values="count")
    .fillna(0)
)
fig_heat = px.imshow(
    heatmap_data,
    title="Complaint Density — Ward × Category",
    labels=dict(x="Category", y="Ward", color="Complaints"),
    color_continuous_scale="Blues",
    template=PLOTLY_THEME, aspect="auto",
)
fig_heat.update_layout(margin=dict(l=0,r=0,t=44,b=0), height=430)
st.plotly_chart(fig_heat, use_container_width=True)

# ══════════════════════════════════════════════════════════
# SECTION 7 — PRIORITY INTELLIGENCE
# ══════════════════════════════════════════════════════════
st.markdown('<div class="section-title">🎯 Priority Intelligence</div>', unsafe_allow_html=True)
p1, p2 = st.columns([3, 2])

with p1:
    cat_p = (
        df.groupby("category")
        .agg(frequency=("complaint_id","count"),
             open_count=("is_open","sum"),
             avg_recency=("recency_score","mean"))
        .reset_index()
    )
    cat_p["freq_norm"]  = (
        (cat_p["frequency"] - cat_p["frequency"].min()) /
        (cat_p["frequency"].max() - cat_p["frequency"].min()) * 100
    ).fillna(0)
    cat_p["open_ratio"] = (cat_p["open_count"] / cat_p["frequency"] * 100).round(1)
    cat_p["priority_score"] = (
        cat_p["freq_norm"] * 0.50 +
        cat_p["avg_recency"] * 0.30 +
        cat_p["open_ratio"] * 0.20
    ).round(1)
    cat_p = cat_p.sort_values("priority_score", ascending=False)

    st.markdown("**Category Priority Scores** *(higher = needs more attention)*")
    st.dataframe(
        cat_p[["category","frequency","open_count","open_ratio","priority_score"]]
        .rename(columns={
            "category":"Category","frequency":"Total",
            "open_count":"Open","open_ratio":"Open %",
            "priority_score":"Priority Score ↑"
        }),
        use_container_width=True, hide_index=True,
    )

with p2:
    ch_counts = df["channel"].value_counts().reset_index(name="count")
    fig_ch = px.pie(
        ch_counts, names="channel", values="count",
        title="Complaints by Channel",
        hole=0.4, template=PLOTLY_THEME,
        color_discrete_sequence=px.colors.qualitative.Safe,
    )
    fig_ch.update_layout(margin=dict(l=0,r=0,t=36,b=0), height=320)
    st.plotly_chart(fig_ch, use_container_width=True)

# ══════════════════════════════════════════════════════════
# SECTION 8 — CRITICAL BACKLOG (highlighted)
# ══════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("""
<div style="
    background: linear-gradient(90deg, rgba(127,29,29,0.6), rgba(15,23,42,0));
    border-left: 5px solid #ef4444;
    border-radius: 10px;
    padding: 0.7rem 1.2rem;
    margin-bottom: 0.8rem;
">
    <span style="font-size:1.05rem; font-weight:700; color:#fca5a5;">
        🚨 Critical Complaints — Immediate Attention Required
    </span><br>
    <span style="font-size:0.82rem; color:#f87171;">
        Open complaints ≥ 30 days · Marked High or Critical priority · Sorted by days waiting
    </span>
</div>
""", unsafe_allow_html=True)

backlog_cols = ["complaint_id","ward","category","description",
                "priority","filed_on","days_open","status"]
available    = [c for c in backlog_cols if c in df.columns]

if "days_open" in df.columns:
    backlog = (
        df[df["is_open"] &
           df["priority"].isin(["High","Critical"]) &
           (df["days_open"].fillna(0) >= 30)]
        .sort_values("days_open", ascending=False)[available]
        .copy()
    )
else:
    backlog = df[
        df["is_open"] & df["priority"].isin(["High","Critical"])
    ][available].copy()

if backlog.empty:
    st.success("✅ No critical backlog items match the current filters.")
else:
    # Color-code rows: >60 days = dark red label, 30-60 = red
    def highlight_backlog(row):
        days = row.get("days_open", 0) or 0
        if days >= 60:
            return ["background-color: rgba(127,29,29,0.45)"] * len(row)
        elif days >= 30:
            return ["background-color: rgba(127,29,29,0.20)"] * len(row)
        return [""] * len(row)

    styled = backlog.style.apply(highlight_backlog, axis=1)
    st.dataframe(styled, use_container_width=True, hide_index=True)
    st.caption(f"🔴 Dark red = waiting 60+ days  |  🟥 Light red = waiting 30–60 days  |  Total: {len(backlog)} complaints")

# ══════════════════════════════════════════════════════════
# SECTION 9 — RECOMMENDED ACTIONS
# ══════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(f"""
<div class="rec-box">
    <h3>📌 Recommended Actions for Authorities</h3>
    <div class="rec-item">🔺 1. &nbsp;<b>Increase workforce in {worst_ward}</b> — it has the highest complaint density in the system</div>
    <div class="rec-item">💧 2. &nbsp;<b>Prioritize {top1_cat} issues</b> — {top1_count} complaints logged, {int(df[df['category']==top1_cat]['is_open'].sum())} still unresolved</div>
    <div class="rec-item">🚨 3. &nbsp;<b>Immediately address the {critical_count} critical backlog complaints</b> — citizens waiting 30+ days will not return to digital channels</div>
    <div class="rec-item">🐢 4. &nbsp;<b>Audit the {slowest_cat} resolution process</b> — averaging {slowest_days:.1f} days vs {overall_avg_res:.1f} days overall; review vendor SLAs</div>
    <div class="rec-item">📱 5. &nbsp;<b>Run a mobile app adoption campaign</b> — most complaints still arrive via phone/walk-in, inflating manual intake costs</div>
</div>
""", unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────
st.markdown("---")
st.caption("🏛️ Municipal Grievance Decision-Support System · Built with Streamlit & Plotly · v2.0")
