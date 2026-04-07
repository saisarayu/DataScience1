"""
============================================================
  Municipal Grievance System – Dataset Generator
  File: src/generate_dataset.py
  Purpose: Create a realistic synthetic dataset of ~2000 complaints
           and save it to data/raw/grievances.csv
============================================================
Run:  python src/generate_dataset.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import random

# ── Seed for reproducibility ───────────────────────────────
np.random.seed(42)
random.seed(42)

# ── Constants ──────────────────────────────────────────────
N = 2000                          # Total number of complaint records
OUTPUT_PATH = Path("data/raw/grievances.csv")

# ── Reference data ─────────────────────────────────────────
WARDS = [f"Ward-{i:02d}" for i in range(1, 21)]   # 20 municipal wards

CATEGORIES = {
    "Water Supply": [
        "No water supply for 3 days",
        "Low water pressure in pipes",
        "Dirty/contaminated water coming from tap",
        "Water pipeline leakage on main road",
        "Water meter not working properly",
    ],
    "Waste Management": [
        "Garbage not collected for a week",
        "Overflowing dustbin near bus stop",
        "Illegal dumping of waste on street",
        "No waste bins available in the area",
        "Garbage truck does not come on schedule",
    ],
    "Road & Infrastructure": [
        "Large pothole causing accidents on highway",
        "Road completely waterlogged after rain",
        "Broken footpath causing injury to pedestrians",
        "Speed breaker removed without notice",
        "Road construction debris not cleared",
    ],
    "Street Lighting": [
        "Street lights not working for 2 weeks",
        "Frequent power fluctuation in area",
        "Electric pole fallen on road – dangerous",
        "Short circuit sparks from street light",
        "Broken street light reported multiple times",
    ],
    "Sewage & Drainage": [
        "Sewage overflowing onto road",
        "Blocked drain causing flooding in colony",
        "Foul smell from open drainage channel",
        "Manhole cover missing – safety hazard",
        "Sewage mixing with drinking water pipeline",
    ],
    "Public Property": [
        "Park benches broken and vandalised",
        "Public toilet locked / not accessible",
        "Tree fallen on parked vehicles",
        "Stray dogs attacking residents near park",
        "Encroachment on footpath by shop owners",
    ],
    "Noise & Pollution": [
        "Construction noise beyond permitted hours",
        "Factory emitting foul-smelling smoke",
        "Burning of garbage creating air pollution",
        "Loud music from nearby event venue",
        "Dust from road construction affecting residents",
    ],
}

STATUSES   = ["Open", "In Progress", "Resolved", "Closed", "Rejected"]
STATUS_W   = [0.35, 0.25, 0.25, 0.10, 0.05]     # Weighted prob of each status

CHANNELS   = ["Mobile App", "Walk-In", "Phone Call", "Website", "WhatsApp"]
PRIORITIES = ["Low", "Medium", "High", "Critical"]
PRIORITY_W = [0.30, 0.40, 0.20, 0.10]

# ── Helper: generate complaint date (last 18 months) ───────
start_date = pd.Timestamp("2024-10-01")
end_date   = pd.Timestamp("2026-04-01")
date_range_days = (end_date - start_date).days


def random_date():
    """Return a random pandas Timestamp within the last 18 months."""
    offset = np.random.randint(0, date_range_days)
    return start_date + pd.Timedelta(days=int(offset))


def resolved_date(filed: pd.Timestamp, status: str) -> pd.Timestamp | None:
    """If resolved/closed, generate a resolution date 1-30 days after filed."""
    if status in ("Resolved", "Closed"):
        lag = np.random.randint(1, 31)
        return filed + pd.Timedelta(days=int(lag))
    return pd.NaT


# ── Build records ──────────────────────────────────────────
records = []
complaint_id = 1000

for _ in range(N):
    category = random.choice(list(CATEGORIES.keys()))
    description = random.choice(CATEGORIES[category])
    status = random.choices(STATUSES, weights=STATUS_W)[0]
    filed_on = random_date()

    # Inject ~8% missing values across key fields (realistic data quality)
    ward     = random.choice(WARDS) if np.random.rand() > 0.05 else np.nan
    channel  = random.choice(CHANNELS) if np.random.rand() > 0.03 else np.nan
    priority = random.choices(PRIORITIES, weights=PRIORITY_W)[0]

    records.append({
        "complaint_id"    : f"GRV-{complaint_id}",
        "filed_on"        : filed_on.strftime("%Y-%m-%d"),
        "resolved_on"     : resolved_date(filed_on, status),
        "ward"            : ward,
        "category"        : category,
        "description"     : description,
        "status"          : status,
        "channel"         : channel,
        "priority"        : priority,
        "citizen_age"     : np.random.choice([np.nan, np.random.randint(18, 75)],
                                              p=[0.10, 0.90]),
        "satisfaction_score": np.where(
            status in ("Resolved", "Closed"),
            np.random.randint(1, 6),          # 1-5 rating
            np.nan
        ),
    })
    complaint_id += 1

# ── Save ───────────────────────────────────────────────────
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df = pd.DataFrame(records)
df.to_csv(OUTPUT_PATH, index=False)

print(f"✅ Dataset generated → {OUTPUT_PATH}")
print(f"   Shape  : {df.shape}")
print(f"   Columns: {list(df.columns)}")
print(f"\nSample rows:\n{df.head(3).to_string()}")
