"""
============================================================
  Municipal Grievance System – Machine Learning Models
  File: src/ml_models.py

  MODELS INCLUDED
  ───────────────
  1. Resolution Time Predictor (Regression)
  2. Priority Classifier (Multiclass)
  3. Satisfaction Score Predictor (Regression)
  4. Category Classifier (NLP-based)

Run:  python src/ml_models.py
============================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

# ── Paths ──────────────────────────────────────────────────
DATA_PATH = Path("data/processed/grievances_clean.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("  MUNICIPAL GRIEVANCE ML MODELS")
print("=" * 60)

# ══════════════════════════════════════════════════════════
# LOAD AND PREPARE DATA
# ══════════════════════════════════════════════════════════
print("\n[1] Loading data …")

df = pd.read_csv(DATA_PATH)
print(f"  Loaded {len(df)} records")

# ── Feature Engineering ───────────────────────────────────
def prepare_features(df):
    """Prepare features for ML models"""
    # Create copy
    data = df.copy()

    # Encode categorical variables
    le_ward = LabelEncoder()
    le_category = LabelEncoder()
    le_channel = LabelEncoder()
    le_priority = LabelEncoder()
    le_weekday = LabelEncoder()

    data['ward_encoded'] = le_ward.fit_transform(data['ward'])
    data['category_encoded'] = le_category.fit_transform(data['category'])
    data['channel_encoded'] = le_channel.fit_transform(data['channel'])
    data['priority_encoded'] = le_priority.fit_transform(data['priority'])
    data['weekday_encoded'] = le_weekday.fit_transform(data['filed_weekday'].astype(str))

    # Save encoders for later use
    encoders = {
        'ward': le_ward,
        'category': le_category,
        'channel': le_channel,
        'priority': le_priority,
        'weekday': le_weekday
    }

    # Numeric features
    features = [
        'ward_encoded', 'category_encoded', 'channel_encoded', 'weekday_encoded',
        'citizen_age', 'filed_month', 'recency_score'
    ]

    return data, features, encoders

data, features, encoders = prepare_features(df)

# ══════════════════════════════════════════════════════════
# MODEL 1: RESOLUTION TIME PREDICTOR (REGRESSION)
# ══════════════════════════════════════════════════════════
print("\n[2] Training Resolution Time Predictor …")

# Filter resolved complaints only
resolved = data[data['status'] == 'Resolved'].dropna(subset=['resolution_days'])

if len(resolved) > 0:
    X_res = resolved[features]
    y_res = resolved['resolution_days']

    # Handle missing values
    X_res = X_res.fillna(X_res.mean())

    X_train_res, X_test_res, y_train_res, y_test_res = train_test_split(
        X_res, y_res, test_size=0.2, random_state=42
    )

    # Train model
    res_model = RandomForestRegressor(n_estimators=100, random_state=42)
    res_model.fit(X_train_res, y_train_res)

    # Evaluate
    y_pred_res = res_model.predict(X_test_res)
    print(".2f")
    print(".2f")
    print(".2f")

    # Save model
    with open(MODEL_DIR / 'resolution_predictor.pkl', 'wb') as f:
        pickle.dump(res_model, f)

else:
    print("  No resolved complaints with resolution days found")
    res_model = None

# ══════════════════════════════════════════════════════════
# MODEL 2: PRIORITY CLASSIFIER (MULTICLASS)
# ══════════════════════════════════════════════════════════
print("\n[3] Training Priority Classifier …")

X_pri = data[features]
y_pri = data['priority_encoded']

# Handle missing values
X_pri = X_pri.fillna(X_pri.mean())

X_train_pri, X_test_pri, y_train_pri, y_test_pri = train_test_split(
    X_pri, y_pri, test_size=0.2, random_state=42, stratify=y_pri
)

# Train model
pri_model = RandomForestClassifier(n_estimators=100, random_state=42)
pri_model.fit(X_train_pri, y_train_pri)

# Evaluate
y_pred_pri = pri_model.predict(X_test_pri)
print(".3f")
print(classification_report(y_test_pri, y_pred_pri,
                          target_names=encoders['priority'].classes_))

# Save model
with open(MODEL_DIR / 'priority_classifier.pkl', 'wb') as f:
    pickle.dump(pri_model, f)

# ══════════════════════════════════════════════════════════
# MODEL 3: SATISFACTION SCORE PREDICTOR (REGRESSION)
# ══════════════════════════════════════════════════════════
print("\n[4] Training Satisfaction Score Predictor …")

# Filter resolved complaints with satisfaction scores
satisfied = data.dropna(subset=['satisfaction_score'])

if len(satisfied) > 0:
    X_sat = satisfied[features + ['resolution_days']]
    y_sat = satisfied['satisfaction_score']

    # Handle missing values
    X_sat = X_sat.fillna(X_sat.mean())

    X_train_sat, X_test_sat, y_train_sat, y_test_sat = train_test_split(
        X_sat, y_sat, test_size=0.2, random_state=42
    )

    # Train model
    sat_model = RandomForestRegressor(n_estimators=100, random_state=42)
    sat_model.fit(X_train_sat, y_train_sat)

    # Evaluate
    y_pred_sat = sat_model.predict(X_test_sat)
    print(".2f")
    print(".2f")
    print(".2f")

    # Save model
    with open(MODEL_DIR / 'satisfaction_predictor.pkl', 'wb') as f:
        pickle.dump(sat_model, f)

else:
    print("  No satisfaction scores found")
    sat_model = None

# ══════════════════════════════════════════════════════════
# MODEL 4: CATEGORY CLASSIFIER (NLP-BASED)
# ══════════════════════════════════════════════════════════
print("\n[5] Training Category Classifier (NLP) …")

# Prepare text data
text_data = data.dropna(subset=['description'])
X_text = text_data['description']
y_cat = text_data['category_encoded']

if len(text_data) > 0:
    # TF-IDF vectorization
    tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
    X_text_vec = tfidf.fit_transform(X_text)

    X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(
        X_text_vec, y_cat, test_size=0.2, random_state=42, stratify=y_cat
    )

    # Train model
    cat_model = RandomForestClassifier(n_estimators=100, random_state=42)
    cat_model.fit(X_train_cat, y_train_cat)

    # Evaluate
    y_pred_cat = cat_model.predict(X_test_cat)
    print(".3f")
    print(classification_report(y_test_cat, y_pred_cat,
                              target_names=encoders['category'].classes_))

    # Save model and vectorizer
    with open(MODEL_DIR / 'category_classifier.pkl', 'wb') as f:
        pickle.dump(cat_model, f)
    with open(MODEL_DIR / 'tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(tfidf, f)

else:
    print("  No description data found")
    cat_model = None
    tfidf = None

# ══════════════════════════════════════════════════════════
# SAVE ENCODERS AND FEATURES
# ══════════════════════════════════════════════════════════
print("\n[6] Saving encoders and metadata …")

metadata = {
    'features': features,
    'encoders': encoders,
    'models_trained': {
        'resolution_predictor': res_model is not None,
        'priority_classifier': True,
        'satisfaction_predictor': sat_model is not None,
        'category_classifier': cat_model is not None
    }
}

with open(MODEL_DIR / 'metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)

print("\n✅ ML Models training completed!")
print(f"   Models saved in: {MODEL_DIR}")
print("\n📊 Model Performance Summary:")
print("   • Resolution Time: Predicts days to resolve complaints")
print("   • Priority Classification: Classifies complaint urgency")
print("   • Satisfaction Prediction: Forecasts citizen satisfaction")
print("   • Category Classification: Auto-categorizes from descriptions")

# ══════════════════════════════════════════════════════════
# PREDICTION FUNCTIONS
# ══════════════════════════════════════════════════════════
def predict_resolution_time(complaint_data):
    """Predict resolution time for a new complaint"""
    if res_model is None:
        return None

    # Prepare input using pre-fitted encoders
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

    return res_model.predict(input_features)[0]

def predict_priority(complaint_data):
    """Predict priority level for a new complaint"""
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

def predict_satisfaction(complaint_data):
    """Predict satisfaction score for a resolved complaint"""
    if sat_model is None:
        return None

    input_data = pd.DataFrame([complaint_data])
    input_data['ward_encoded'] = encoders['ward'].transform([input_data['ward'].iloc[0]])
    input_data['category_encoded'] = encoders['category'].transform([input_data['category'].iloc[0]])
    input_data['channel_encoded'] = encoders['channel'].transform([input_data['channel'].iloc[0]])
    input_data['weekday_encoded'] = encoders['weekday'].transform([str(input_data['filed_weekday'].iloc[0])])

    input_features = input_data[['ward_encoded', 'category_encoded', 'channel_encoded', 'weekday_encoded',
                                'citizen_age', 'filed_month', 'recency_score', 'resolution_days']].fillna(
        input_data[['ward_encoded', 'category_encoded', 'channel_encoded', 'weekday_encoded',
                   'citizen_age', 'filed_month', 'recency_score', 'resolution_days']].mean()
    )

    return sat_model.predict(input_features)[0]

def predict_category(description):
    """Predict category from complaint description"""
    if cat_model is None or tfidf is None:
        return None

    text_vec = tfidf.transform([description])
    pred_encoded = cat_model.predict(text_vec)[0]
    return encoders['category'].inverse_transform([pred_encoded])[0]

# Example usage
if __name__ == "__main__":
    print("\n🔍 Example Predictions:")

    # Sample complaint
    sample = {
        'ward': 'Ward-01',
        'category': 'Water Supply',
        'channel': 'Mobile App',
        'citizen_age': 35,
        'filed_month': 4,
        'filed_weekday': 'Wednesday',
        'recency_score': 50.0,
        'resolution_days': 15.0
    }

    print(f"   Sample complaint priority: {predict_priority(sample)}")
    print(f"   Predicted resolution time: {predict_resolution_time(sample):.1f} days")
    print(f"   Predicted satisfaction: {predict_satisfaction(sample):.1f}/5")

    sample_desc = "No water supply for 3 days in our area"
    print(f"   Description category: {predict_category(sample_desc)}")