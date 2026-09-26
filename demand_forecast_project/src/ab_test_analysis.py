import os

import numpy as np
import pandas as pd
from scipy import stats

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

ALPHA = 0.05

def two_proportion_ztest(conv_c, n_c, conv_t, n_t, alpha=0.05):
    p_c = conv_c / n_c
    p_t = conv_t / n_t

    p_pool = (conv_c + conv_t) / (n_c + n_t)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n_c + 1 / n_t))
    z_stat = (p_t - p_c) / se_pool
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))  # two-sided

    se_diff = np.sqrt(p_c * (1 - p_c) / n_c + p_t * (1 - p_t) / n_t)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    diff = p_t - p_c
    ci_low, ci_high = diff - z_crit * se_diff, diff + z_crit * se_diff

    relative_lift = diff / p_c

    rel_ci_low, rel_ci_high = ci_low / p_c, ci_high / p_c

    return {
        "p_control": p_c,
        "p_treatment": p_t,
        "abs_diff": diff,
        "relative_lift_pct": relative_lift * 100,
        "z_stat": z_stat,
        "p_value": p_value,
        "ci_diff_low": ci_low,
        "ci_diff_high": ci_high,
        "rel_lift_ci_low_pct": rel_ci_low * 100,
        "rel_lift_ci_high_pct": rel_ci_high * 100,
        "significant_at_alpha": p_value < alpha,
    }


def run():
    df = pd.read_csv(os.path.join(DATA_DIR, "ab_test_conversions.csv"))

    grp = df.groupby("group")["converted"].agg(["sum", "count", "mean"])
    conv_c, n_c = grp.loc["control", "sum"], grp.loc["control", "count"]
    conv_t, n_t = grp.loc["treatment", "sum"], grp.loc["treatment", "count"]

    results = two_proportion_ztest(conv_c, n_c, conv_t, n_t, alpha=ALPHA)

    contingency = np.array([
        [conv_c, n_c - conv_c],
        [conv_t, n_t - conv_t],
    ])
    chi2, chi2_p, dof, expected = stats.chi2_contingency(contingency, correction=False)

    lines = []
    lines.append("A/B TEST ANALYSIS: Pricing / Promotion Change -> Conversion Rate")
    lines.append("=" * 65)
    lines.append(f"Control:   n={n_c:,}   conversions={conv_c:,}   rate={results['p_control']:.4%}")
    lines.append(f"Treatment: n={n_t:,}   conversions={conv_t:,}   rate={results['p_treatment']:.4%}")
    lines.append("")
    lines.append(f"Absolute lift:            {results['abs_diff']:.4%}")
    lines.append(f"Relative lift:            {results['relative_lift_pct']:.2f}%")
    lines.append(f"95% CI (absolute diff):   [{results['ci_diff_low']:.4%}, {results['ci_diff_high']:.4%}]")
    lines.append(f"95% CI (relative lift):   [{results['rel_lift_ci_low_pct']:.2f}%, "
                  f"{results['rel_lift_ci_high_pct']:.2f}%]")
    lines.append("")
    lines.append(f"Two-proportion z-test:    z = {results['z_stat']:.3f},  p-value = {results['p_value']:.5f}")
    lines.append(f"Chi-square test (cross-check): chi2 = {chi2:.3f}, p-value = {chi2_p:.5f}")
    lines.append("")
    verdict = "STATISTICALLY SIGNIFICANT" if results["significant_at_alpha"] else "NOT statistically significant"
    lines.append(f"Result at alpha = {ALPHA}: {verdict}")
    if results["significant_at_alpha"]:
        lines.append(f"=> Reject H0. The promotion/pricing change increased conversion by "
                      f"~{results['relative_lift_pct']:.1f}% (relative), with 95% confidence "
                      f"the true relative lift is between {results['rel_lift_ci_low_pct']:.1f}% "
                      f"and {results['rel_lift_ci_high_pct']:.1f}%.")

    summary_text = "\n".join(lines)
    print(summary_text)

    with open(os.path.join(OUT_DIR, "ab_test_summary.txt"), "w") as f:
        f.write(summary_text + "\n")

    pd.DataFrame([results]).to_csv(os.path.join(OUT_DIR, "ab_test_summary.csv"), index=False)

    return results


if __name__ == "__main__":
    run()
