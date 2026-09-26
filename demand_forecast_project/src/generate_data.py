import os
import numpy as np
import pandas as pd

np.random.seed(42)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# ---------------------------------------------------------------------
# 1. Weekly demand data across product categories
# ---------------------------------------------------------------------

N_WEEKS = 104  # 2 years
start_date = pd.Timestamp("2024-01-07")  # first Sunday of 2024
dates = pd.date_range(start=start_date, periods=N_WEEKS, freq="W")

categories = {
    "Beverages":        {"base": 1200, "trend": 3.0,  "amp": 250, "phase": 0.0,  "noise": 60},
    "Snacks":           {"base": 950,  "trend": 2.0,  "amp": 180, "phase": 0.5,  "noise": 50},
    "Dairy":            {"base": 1400, "trend": 1.0,  "amp": 120, "phase": 1.0,  "noise": 70},
    "Frozen Foods":     {"base": 800,  "trend": 1.5,  "amp": 220, "phase": 2.5,  "noise": 45},
    "Bakery":           {"base": 700,  "trend": -0.5, "amp": 150, "phase": 0.2,  "noise": 40},
    "Household Care":   {"base": 600,  "trend": 2.5,  "amp": 60,  "phase": 3.0,  "noise": 30},
    "Personal Care":    {"base": 550,  "trend": 1.8,  "amp": 80,  "phase": 1.8,  "noise": 28},
    "Produce":          {"base": 1100, "trend": 0.5,  "amp": 300, "phase": 3.4,  "noise": 80},
    "Meat & Seafood":   {"base": 900,  "trend": 1.2,  "amp": 200, "phase": 2.9,  "noise": 55},
    "Bread & Grains":   {"base": 750,  "trend": 0.8,  "amp": 100, "phase": 0.9,  "noise": 35},
    "Confectionery":    {"base": 500,  "trend": 3.5,  "amp": 260, "phase": 4.2,  "noise": 40},
    "Health Supplements": {"base": 400, "trend": 4.0, "amp": 90,  "phase": 5.0,  "noise": 22},
}

records = []
t = np.arange(N_WEEKS)

for cat, p in categories.items():
    trend_component = p["trend"] * t
    seasonal_component = p["amp"] * np.sin(2 * np.pi * t / 52 + p["phase"])
    # slight extra bump around weeks 46-52 and 98-104 (holiday season)
    holiday_boost = np.where(((t % 52) >= 45) & ((t % 52) <= 51), p["base"] * 0.18, 0)
    # Add mild random-walk drift on top of the deterministic trend so the
    # series isn't perfectly regular (more realistic + harder to forecast).
    drift = np.cumsum(np.random.normal(0, p["noise"] * 0.06, size=N_WEEKS))
    noise = np.random.normal(0, p["noise"] * 1.6, size=N_WEEKS) + drift
    values = p["base"] + trend_component + seasonal_component + holiday_boost + noise
    values = np.clip(values, a_min=10, a_max=None).round().astype(int)

    for d, v in zip(dates, values):
        records.append({"week_start": d, "category": cat, "units_sold": v})

df = pd.DataFrame(records)
df = df.sort_values(["category", "week_start"]).reset_index(drop=True)
df.to_csv(os.path.join(DATA_DIR, "weekly_sales.csv"), index=False)
print(f"weekly_sales.csv written: {df.shape[0]} rows, {df['category'].nunique()} categories, "
      f"{df['week_start'].min().date()} to {df['week_start'].max().date()}")

# ---------------------------------------------------------------------
# 2. A/B test data: pricing/promotion change -> conversion rate
# ---------------------------------------------------------------------
# Control: existing pricing/promo. Treatment: new pricing/promo.
# True conversion rates chosen so the treatment shows ~8% relative lift.

n_control = 24000
n_treatment = 24000

p_control = 0.115          # 11.5% baseline conversion
p_treatment = p_control * 1.08  # ~8% relative lift -> ~12.42%

control_conversions = np.random.binomial(1, p_control, n_control)
treatment_conversions = np.random.binomial(1, p_treatment, n_treatment)

ab_df = pd.DataFrame({
    "user_id": np.arange(1, n_control + n_treatment + 1),
    "group": ["control"] * n_control + ["treatment"] * n_treatment,
    "converted": np.concatenate([control_conversions, treatment_conversions]),
})
ab_df = ab_df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
ab_df.to_csv(os.path.join(DATA_DIR, "ab_test_conversions.csv"), index=False)

print(f"ab_test_conversions.csv written: {ab_df.shape[0]} rows")
print(ab_df.groupby("group")["converted"].mean())
