# Demand-Intelligence-Experimentation

# Demand Forecasting & A/B Test Analysis

**Python · SARIMA · Statsmodels · SciPy · Time Series · Hypothesis Testing**

An end-to-end data science project combining **time-series demand forecasting** and **A/B test analysis** using synthetic, reproducible datasets.

The project demonstrates how statistical and forecasting methods can be applied to common business problems such as **demand planning, inventory forecasting, pricing experiments, and conversion optimization**.

---

## Project Overview

### 1. Demand Forecasting

Builds **SARIMA time-series models** for 12 product categories using two years of weekly sales data.

The forecasting pipeline:

* Models each product category independently
* Uses the final 14 weeks as a held-out test set
* Compares SARIMA against a seasonal-naive baseline
* Evaluates forecasts using **MAPE**
* Generates 95% confidence intervals
* Saves category-level and overall forecasting results

### 2. A/B Test Analysis

Analyzes a simulated pricing/promotion experiment to measure its effect on conversion rate.

The analysis:

* Compares control and treatment conversion rates
* Performs a **two-proportion z-test**
* Calculates absolute and relative conversion lift
* Computes 95% confidence intervals
* Uses a chi-square test as an independent cross-check
* Determines statistical significance at `α = 0.05`

---

## Key Technologies

* **Python**
* **Pandas**
* **NumPy**
* **Statsmodels**
* **SciPy**
* **Matplotlib**
* **Jupyter Notebook**
* **SARIMA**
* **Hypothesis Testing**
* **A/B Testing**
* **Statistical Analysis**

---

## Project Structure

```text
demand-forecast-project/
│
├── data/
│   ├── weekly_sales.csv
│   │   └── Synthetic 2-year weekly sales data
│   │
│   └── ab_test_conversions.csv
│       └── Synthetic A/B test data
│
├── src/
│   ├── generate_data.py
│   │   └── Generates reproducible synthetic datasets
│   │
│   ├── forecasting.py
│   │   └── SARIMA forecasting and baseline comparison
│   │
│   └── ab_test_analysis.py
│       └── A/B test statistical analysis
│
├── notebooks/
│   └── analysis.ipynb
│       └── End-to-end analysis with charts and results
│
├── outputs/
│   ├── forecasts/
│   │   └── <category>.png
│   │       └── Forecast plots with 95% confidence intervals
│   │
│   ├── forecast_summary.csv
│   ├── forecast_overall_summary.txt
│   ├── ab_test_summary.csv
│   └── ab_test_summary.txt
│
├── build_notebook.py
│   └── Rebuilds the analysis notebook
│
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

On Windows:

```bash
venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Generate the datasets

```bash
python src/generate_data.py
```

This creates both synthetic datasets in the `data/` directory.

### 4. Run demand forecasting

```bash
python src/forecasting.py
```

This runs SARIMA forecasting for each product category and compares the results against the seasonal-naive baseline.

### 5. Run the A/B test

```bash
python src/ab_test_analysis.py
```

This performs the hypothesis test and generates conversion-rate, lift, p-value, and confidence-interval results.

### 6. Rebuild the notebook (optional)

```bash
python build_notebook.py
jupyter nbconvert --to notebook --execute --inplace notebooks/analysis.ipynb
```

Alternatively, open the existing notebook:

```text
notebooks/analysis.ipynb
```

The notebook contains the complete analysis, visualizations, and output.

---

# Methodology

## 1. Demand Forecasting

### Dataset

`data/weekly_sales.csv`

The dataset contains:

* **104 weeks** of historical sales data
* **12 product categories**
* Weekly units sold
* Category-specific trends
* Yearly seasonality
* Holiday-season demand patterns
* Random noise and drift

Schema:

| Column       | Description                  |
| ------------ | ---------------------------- |
| `week_start` | Start date of the sales week |
| `category`   | Product category             |
| `units_sold` | Number of units sold         |

### Train/Test Split

The final **14 weeks** of data are held out for testing for each category.

```text
Training Data                    Test Data
<------------------------------>|------------>
          90 weeks                  14 weeks
```

This allows the forecasting models to be evaluated on future observations that were not used during training.

### Seasonal-Naive Baseline

A seasonal-naive model is used as the baseline.

The forecast for a given week is based on the actual sales from the same week one year earlier.

If sufficient historical data is unavailable, the model falls back to the most recently observed value.

### SARIMA Model

A small grid of SARIMA configurations is evaluated for each product category:

```text
SARIMA(p,d,q)(P,D,Q,52)
```

The seasonal period is set to **52 weeks** to capture yearly seasonality.

Models are implemented using:

```python
statsmodels.tsa.statespace.sarimax.SARIMAX
```

The configuration with the lowest **AIC (Akaike Information Criterion)** is selected for each category.

The selected model is then used to forecast the 14-week test horizon with a **95% confidence interval**.

### Evaluation

Forecast performance is measured using:

**MAPE (Mean Absolute Percentage Error)**

The project compares:

* Seasonal-naive MAPE
* SARIMA MAPE
* MAPE improvement by category
* Overall average performance

Results are saved to:

```text
outputs/forecast_summary.csv
outputs/forecast_overall_summary.txt
```

Forecast visualizations are saved to:

```text
outputs/forecasts/
```

---

# 2. A/B Test Analysis

### Dataset

`data/ab_test_conversions.csv`

The dataset contains **48,000 simulated users** divided between:

* Control group
* Treatment group

Schema:

| Column      | Description                  |
| ----------- | ---------------------------- |
| `user_id`   | Unique user identifier       |
| `group`     | Control or treatment         |
| `converted` | Conversion indicator: 0 or 1 |

The synthetic data is generated with a predefined treatment effect to make the experiment reproducible.

### Hypothesis Test

The analysis uses a **two-proportion z-test**.

**Null hypothesis:**

```text
H₀: p_control = p_treatment
```

**Alternative hypothesis:**

```text
H₁: p_control ≠ p_treatment
```

A significance level of:

```text
α = 0.05
```

is used for the two-sided test.

### Metrics

The analysis calculates:

* Control conversion rate
* Treatment conversion rate
* Absolute conversion lift
* Relative conversion lift
* Z-statistic
* P-value
* 95% confidence interval
* Statistical significance

### Confidence Interval

A 95% confidence interval is calculated for the difference between treatment and control conversion rates using an unpooled standard error.

The interval is also propagated to estimate the confidence interval for relative lift.

### Statistical Cross-Check

A **chi-square test of independence** is performed on the corresponding 2×2 contingency table using:

```python
scipy.stats.chi2_contingency
```

For a two-group experiment with a binary outcome, the chi-square test provides a useful cross-check of the two-proportion z-test.

Results are saved to:

```text
outputs/ab_test_summary.csv
outputs/ab_test_summary.txt
```

---

# Results

The forecasting analysis produces category-level and overall MAPE comparisons between SARIMA and the seasonal-naive baseline.

The A/B test produces conversion rates, lift estimates, confidence intervals, and statistical significance results.

All generated results are reproducible because the synthetic datasets use fixed random seeds.

---

# Reproducible Synthetic Data

This project does **not require an external dataset**.

The datasets are generated locally using:

```bash
python src/generate_data.py
```

The generated data includes realistic patterns such as:

* Product-level demand differences
* Long-term trends
* Annual seasonality
* Holiday effects
* Random variation
* Treatment/control conversion differences

This makes the repository completely self-contained while allowing the forecasting and statistical analysis pipelines to be reproduced.

---

# Using Real Data

The project can be adapted to real-world datasets with minimal changes.

### Demand Forecasting

Replace:

```text
data/weekly_sales.csv
```

with a dataset containing:

```text
week_start
category
units_sold
```

The existing forecasting pipeline can then be reused.

### A/B Testing

Replace:

```text
data/ab_test_conversions.csv
```

with an experiment dataset containing at least:

```text
group
converted
```

If the experiment measures a continuous outcome such as revenue per user, the analysis can be extended to use an appropriate statistical test, such as a two-sample t-test or another suitable method.

---

# Future Improvements

Potential extensions include:

* Rolling-origin / walk-forward cross-validation
* Automated SARIMA order selection
* `auto_arima` experimentation
* Seasonal decomposition diagnostics
* Additional forecasting models such as Prophet or gradient boosting
* Forecast error analysis by category
* Bayesian A/B testing
* Power analysis and sample-size estimation
* Sequential testing considerations
* Experiment segmentation and heterogeneous treatment effects
* Interactive dashboards for forecasting and experiment results

---

## Resume Alignment

This project demonstrates experience with:

* Time-series forecasting
* SARIMA modeling
* Forecast evaluation
* Statistical hypothesis testing
* A/B testing
* Confidence intervals
* Experiment analysis
* Python-based data analysis
* Business-oriented analytical problem solving
