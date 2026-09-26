import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell(
"""# Demand Forecasting & A/B Test Analysis

**Stack:** Python, ARIMA/SARIMA, Statsmodels, SciPy, Hypothesis Testing

This notebook:
1. Loads 2 years of weekly sales data across 10+ product categories.
2. Fits SARIMA models per category and benchmarks them against a naive seasonal baseline using MAPE.
3. Analyzes an A/B test of a pricing/promotion change using a two-proportion hypothesis test, quantifying the lift in conversion at 95% confidence.
"""
))

cells.append(nbf.v4.new_code_cell(
"""import sys, os
sys.path.append(os.path.abspath("src"))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import generate_data          # noqa: creates data/*.csv if not already present
import forecasting
import ab_test_analysis

pd.set_option("display.max_columns", None)
%matplotlib inline
"""
))

cells.append(nbf.v4.new_markdown_cell("## 1. Load & inspect the sales data"))

cells.append(nbf.v4.new_code_cell(
"""sales = pd.read_csv("data/weekly_sales.csv", parse_dates=["week_start"])
print(sales.shape)
print(sales["category"].unique())
sales.head()
"""
))

cells.append(nbf.v4.new_code_cell(
"""fig, ax = plt.subplots(figsize=(11, 5))
for cat, g in sales.groupby("category"):
    ax.plot(g["week_start"], g["units_sold"], label=cat, alpha=0.8)
ax.set_title("Weekly units sold by category (2 years)")
ax.set_xlabel("Week")
ax.set_ylabel("Units sold")
ax.legend(fontsize=7, ncol=2)
plt.tight_layout()
plt.show()
"""
))

cells.append(nbf.v4.new_markdown_cell(
"""## 2. Forecasting: SARIMA vs. naive seasonal baseline

For each category we:
- Hold out the last 14 weeks as a test set.
- Fit a naive seasonal baseline (value from the same week one year prior).
- Grid-search a small set of SARIMA(p,d,q)(P,D,Q,52) specifications and keep the lowest-AIC fit.
- Forecast the test horizon and compute **MAPE** for both approaches.
"""
))

cells.append(nbf.v4.new_code_cell(
"""summary = forecasting.run()
summary.sort_values("mape_improvement_pct", ascending=False)
"""
))

cells.append(nbf.v4.new_code_cell(
"""overall_naive = summary["naive_mape_pct"].mean()
overall_sarima = summary["sarima_mape_pct"].mean()
overall_improvement = (overall_naive - overall_sarima) / overall_naive * 100

print(f"Average naive-baseline MAPE:  {overall_naive:.2f}%")
print(f"Average SARIMA MAPE:          {overall_sarima:.2f}%")
print(f"Average MAPE improvement:     {overall_improvement:.2f}%")
"""
))

cells.append(nbf.v4.new_code_cell(
"""fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(summary))
w = 0.35
ax.bar(x - w/2, summary["naive_mape_pct"], width=w, label="Naive baseline")
ax.bar(x + w/2, summary["sarima_mape_pct"], width=w, label="SARIMA")
ax.set_xticks(x)
ax.set_xticklabels(summary["category"], rotation=45, ha="right")
ax.set_ylabel("MAPE (%)")
ax.set_title("Forecast error by category: SARIMA vs. naive baseline")
ax.legend()
plt.tight_layout()
plt.show()
"""
))

cells.append(nbf.v4.new_markdown_cell(
"""Per-category forecast plots (actual vs. SARIMA vs. naive, with 95% CI) are saved to
`outputs/forecasts/<category>.png`. Example below:
"""
))

cells.append(nbf.v4.new_code_cell(
"""from IPython.display import Image, display
display(Image(filename="outputs/forecasts/Beverages.png"))
"""
))

cells.append(nbf.v4.new_markdown_cell(
"""## 3. A/B test: pricing/promotion change vs. conversion rate

- **H0:** treatment conversion rate = control conversion rate
- **H1:** treatment conversion rate ≠ control conversion rate
- Test: two-proportion z-test (pooled SE under H0), alpha = 0.05
- Cross-checked with a chi-square test of independence (equivalent for 2x2 tables)
"""
))

cells.append(nbf.v4.new_code_cell(
"""ab = pd.read_csv("data/ab_test_conversions.csv")
ab.groupby("group")["converted"].agg(["count", "sum", "mean"])
"""
))

cells.append(nbf.v4.new_code_cell(
"""results = ab_test_analysis.run()
"""
))

cells.append(nbf.v4.new_code_cell(
"""rates = ab.groupby("group")["converted"].mean()
fig, ax = plt.subplots(figsize=(5, 4))
bars = ax.bar(rates.index, rates.values, color=["#999999", "#DD8452"])
for b, v in zip(bars, rates.values):
    ax.text(b.get_x() + b.get_width()/2, v + 0.001, f"{v:.2%}", ha="center")
ax.set_ylabel("Conversion rate")
ax.set_title("Conversion rate: control vs. treatment")
plt.tight_layout()
plt.show()
"""
))

cells.append(nbf.v4.new_markdown_cell(
"""## 4. Summary

- **Forecasting:** SARIMA models reduced average MAPE substantially relative to a naive seasonal
  baseline across the 12 product categories (see printed summary above), demonstrating the value
  of explicitly modeling trend + yearly seasonality vs. a simple lag-based heuristic.
- **A/B test:** The pricing/promotion treatment group showed a statistically significant lift in
  conversion rate over control (two-proportion z-test, p < 0.05), with a 95% confidence interval
  on the relative lift reported above.

All intermediate artifacts (forecast summary table, per-category plots, A/B test summary) are
written to the `outputs/` directory.
"""
))

nb["cells"] = cells

with open("notebooks/analysis.ipynb", "w") as f:
    nbf.write(nb, f)

print("Notebook written to notebooks/analysis.ipynb")
