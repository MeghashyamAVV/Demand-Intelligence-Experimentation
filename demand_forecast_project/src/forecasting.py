import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
PLOT_DIR = os.path.join(OUT_DIR, "forecasts")
os.makedirs(PLOT_DIR, exist_ok=True)

TEST_HORIZON = 14  

CANDIDATE_ORDERS = [
    ((1, 1, 1), (0, 1, 1, 52)),
    ((1, 1, 0), (1, 1, 0, 52)),
    ((2, 1, 1), (0, 1, 1, 52)),
    ((1, 0, 1), (1, 1, 0, 52)),
]

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true, dtype=float), np.array(y_pred, dtype=float)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def naive_seasonal_forecast(train, horizon):
    """Forecast = value from 52 weeks prior; falls back to last value if
    insufficient history for a full seasonal lag."""
    if len(train) >= 52:
        lagged = train[-52:-52 + horizon]
        if len(lagged) < horizon:
            lagged = np.concatenate([lagged, np.repeat(train[-1], horizon - len(lagged))])
        return lagged
    else:
        return np.repeat(train[-1], horizon)


def fit_best_sarima(train_series):
    """Try a few candidate SARIMA specs, keep the one with lowest AIC."""
    best_aic = np.inf
    best_res = None
    best_spec = None
    for order, seasonal_order in CANDIDATE_ORDERS:
        try:
            model = SARIMAX(
                train_series,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            res = model.fit(disp=False)
            if res.aic < best_aic:
                best_aic = res.aic
                best_res = res
                best_spec = (order, seasonal_order)
        except Exception:
            continue
    return best_res, best_spec, best_aic


def run():
    df = pd.read_csv(os.path.join(DATA_DIR, "weekly_sales.csv"), parse_dates=["week_start"])
    categories = sorted(df["category"].unique())

    summary_rows = []

    for cat in categories:
        cat_df = df[df["category"] == cat].sort_values("week_start").reset_index(drop=True)
        y = cat_df["units_sold"].values
        dates = cat_df["week_start"].values

        train, test = y[:-TEST_HORIZON], y[-TEST_HORIZON:]
        train_dates, test_dates = dates[:-TEST_HORIZON], dates[-TEST_HORIZON:]

        # --- Naive baseline ---
        naive_preds = naive_seasonal_forecast(train, TEST_HORIZON)
        naive_mape = mape(test, naive_preds)

        # --- SARIMA ---
        res, spec, aic = fit_best_sarima(train)
        if res is not None:
            forecast = res.get_forecast(steps=TEST_HORIZON)
            sarima_preds = forecast.predicted_mean
            ci = forecast.conf_int(alpha=0.05)
            sarima_mape = mape(test, sarima_preds)
        else:
            sarima_preds = naive_preds
            ci = None
            sarima_mape = naive_mape
            spec = None

        improvement_pct = (naive_mape - sarima_mape) / naive_mape * 100 if naive_mape else 0

        summary_rows.append({
            "category": cat,
            "sarima_order": str(spec[0]) if spec else "n/a",
            "sarima_seasonal_order": str(spec[1]) if spec else "n/a",
            "aic": round(aic, 1) if np.isfinite(aic) else None,
            "naive_mape_pct": round(naive_mape, 2),
            "sarima_mape_pct": round(sarima_mape, 2),
            "mape_improvement_pct": round(improvement_pct, 2),
        })

        # --- Plot ---
        plt.figure(figsize=(9, 4.5))
        plt.plot(train_dates[-30:], train[-30:], label="Train (recent)", color="#4C72B0")
        plt.plot(test_dates, test, label="Actual", color="black", marker="o", ms=3)
        plt.plot(test_dates, sarima_preds, label="SARIMA forecast", color="#DD8452", linestyle="--")
        plt.plot(test_dates, naive_preds, label="Naive baseline", color="#999999", linestyle=":")
        if ci is not None:
            ci_arr = ci.values if hasattr(ci, "values") else np.asarray(ci)
            plt.fill_between(test_dates, ci_arr[:, 0], ci_arr[:, 1], color="#DD8452", alpha=0.15,
                              label="95% CI")
        plt.title(f"{cat}: SARIMA vs Naive Forecast (MAPE {sarima_mape:.1f}% vs {naive_mape:.1f}%)")
        plt.xlabel("Week")
        plt.ylabel("Units sold")
        plt.legend(fontsize=8)
        plt.xticks(rotation=45)
        plt.tight_layout()
        safe_name = cat.replace(" ", "_").replace("&", "and")
        plt.savefig(os.path.join(PLOT_DIR, f"{safe_name}.png"), dpi=120)
        plt.close()

        print(f"{cat:22s} naive MAPE={naive_mape:6.2f}%  sarima MAPE={sarima_mape:6.2f}%  "
              f"improvement={improvement_pct:6.2f}%")

    summary_df = pd.DataFrame(summary_rows)
    overall_naive = summary_df["naive_mape_pct"].mean()
    overall_sarima = summary_df["sarima_mape_pct"].mean()
    overall_improvement = (overall_naive - overall_sarima) / overall_naive * 100

    summary_df.to_csv(os.path.join(OUT_DIR, "forecast_summary.csv"), index=False)

    print("\n=== OVERALL ===")
    print(f"Avg naive MAPE:  {overall_naive:.2f}%")
    print(f"Avg SARIMA MAPE: {overall_sarima:.2f}%")
    print(f"Avg improvement: {overall_improvement:.2f}%")

    with open(os.path.join(OUT_DIR, "forecast_overall_summary.txt"), "w") as f:
        f.write(f"Average naive-baseline MAPE across {len(categories)} categories: {overall_naive:.2f}%\n")
        f.write(f"Average SARIMA MAPE across {len(categories)} categories: {overall_sarima:.2f}%\n")
        f.write(f"Average MAPE improvement vs naive baseline: {overall_improvement:.2f}%\n")

    return summary_df


if __name__ == "__main__":
    run()
