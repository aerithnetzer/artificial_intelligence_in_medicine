"""
Statistical visualizations and hypothesis tests.

Functions for Jaccard distance vs citations scatter plots,
Fisher r-to-z correlation comparison, Pearson correlation analysis,
funding-citation impact tests, and cross-field structural comparisons.
"""

import json
from itertools import combinations
from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import (
    ks_2samp,
    kruskal,
    mannwhitneyu,
    norm as stats_norm,
    spearmanr,
)

from artificial_intelligence_in_medicine.config import PROCESSED_DATA_DIR
from artificial_intelligence_in_medicine.visualizations.utils import (
    MODE_COLORS,
    MODE_LABELS,
    MODES,
    add_citation_count_column,
    add_grant_columns,
    clean_year_column,
    ensure_comparative_dir,
    ensure_output_dir,
    load_features,
    save_plot,
)


# ---------------------------------------------------------------------------
# Fisher r-to-z helpers
# ---------------------------------------------------------------------------
def fisher_r_to_z(r: float) -> float:
    """Fisher r-to-z transformation."""
    r = max(min(r, 0.999999), -0.999999)
    return 0.5 * np.log((1 + r) / (1 - r))


def compare_independent_correlations(r1: float, n1: int, r2: float, n2: int):
    """
    Fisher z test for two independent Pearson correlations.
    Returns (z_stat, p_value_two_tailed).
    """
    if (
        any(v is None for v in [r1, r2, n1, n2])
        or any(np.isnan(v) for v in [r1, r2])
        or n1 < 4
        or n2 < 4
    ):
        return None, None
    z1 = fisher_r_to_z(r1)
    z2 = fisher_r_to_z(r2)
    se = np.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))
    if se == 0:
        return None, None
    z = (z1 - z2) / se
    p = 2 * (1 - stats_norm.cdf(abs(z)))
    return float(z), float(p)


def test_pearson_correlation_diff(
    ai_pearson: float,
    ge_pearson: float,
    n_a: int,
    n_b: int,
    alpha: float = 0.05,
) -> dict:
    """
    Test whether two Pearson correlations differ significantly
    using the Fisher r-to-z transformation.
    """
    z_score, p_value = compare_independent_correlations(ai_pearson, n_a, ge_pearson, n_b)
    if z_score is None:
        return {"z_score": None, "p_value": None, "significant": False}
    return {
        "z_score": z_score,
        "p_value": p_value,
        "significant": p_value < alpha,
    }


# ---------------------------------------------------------------------------
# Pearson cleaning helper
# ---------------------------------------------------------------------------
def _compute_pearson_after_cleaning(csv_path: Path):
    """Compute Pearson r between jaccard_distance and num_cited_by after IQR outlier removal."""
    required = ["jaccard_distance", "num_cited_by"]
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        logger.error(f"File not found: {csv_path}")
        return None, None
    if not all(col in df.columns for col in required):
        logger.error(f"Missing required columns in {csv_path}")
        return None, None
    df_corr = df[required].dropna()
    if df_corr.empty:
        return None, None

    q1 = df_corr.quantile(0.25)
    q3 = df_corr.quantile(0.75)
    iqr = q3 - q1
    mask = (
        (df_corr["jaccard_distance"] >= q1["jaccard_distance"] - 1.5 * iqr["jaccard_distance"])
        & (df_corr["jaccard_distance"] <= q3["jaccard_distance"] + 1.5 * iqr["jaccard_distance"])
        & (df_corr["num_cited_by"] >= q1["num_cited_by"] - 1.5 * iqr["num_cited_by"])
        & (df_corr["num_cited_by"] <= q3["num_cited_by"] + 1.5 * iqr["num_cited_by"])
    )
    df_corr = df_corr[mask]
    n = len(df_corr)
    if n < 4 or df_corr["jaccard_distance"].nunique() <= 1:
        return None, n
    pearson = df_corr["jaccard_distance"].corr(df_corr["num_cited_by"], method="pearson")
    logger.info(f"Pearson ({csv_path.name}): {pearson:.4f} (n={n})")
    return pearson, n


# ---------------------------------------------------------------------------
# Scatter plot with line of best fit
# ---------------------------------------------------------------------------
def scatterplot_with_line_of_best_fit(input_path: Path, mode: str):
    """
    Scatter plot of Jaccard distance vs number of citations
    with IQR outlier removal and a line of best fit.
    """
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        logger.error(f"Input file not found at {input_path}")
        return

    required = ["jaccard_distance", "num_cited_by"]
    if not all(col in df.columns for col in required):
        logger.error(f"Input CSV must contain {required} columns.")
        return

    df_corr = df[required].dropna()

    # IQR outlier removal
    q1 = df_corr.quantile(0.25)
    q3 = df_corr.quantile(0.75)
    iqr = q3 - q1
    df_corr = df_corr[
        (df_corr["jaccard_distance"] >= q1["jaccard_distance"] - 1.5 * iqr["jaccard_distance"])
        & (df_corr["jaccard_distance"] <= q3["jaccard_distance"] + 1.5 * iqr["jaccard_distance"])
        & (df_corr["num_cited_by"] >= q1["num_cited_by"] - 1.5 * iqr["num_cited_by"])
        & (df_corr["num_cited_by"] <= q3["num_cited_by"] + 1.5 * iqr["num_cited_by"])
    ]

    if df_corr.empty or df_corr["jaccard_distance"].nunique() <= 1:
        logger.warning("Insufficient data for scatter plot.")
        return

    pearson = df_corr["jaccard_distance"].corr(df_corr["num_cited_by"], method="pearson")
    spearman = df_corr["jaccard_distance"].corr(df_corr["num_cited_by"], method="spearman")
    logger.info(f"Pearson: {pearson:.4f}, Spearman: {spearman:.4f}")

    x = df_corr["jaccard_distance"]
    y = df_corr["num_cited_by"]
    m, b = np.polyfit(x, y, 1)
    best_fit_y = m * x + b

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="Data", marker=dict(opacity=0.6)))
    fig.add_trace(
        go.Scatter(
            x=x,
            y=best_fit_y,
            mode="lines",
            name="Best fit",
            line=dict(color="red", width=2),
        )
    )
    fig.update_layout(
        xaxis_title="Jaccard Distance",
        yaxis_title="Number of Citations",
        title=f"Jaccard Distance vs Citations ({mode})<br>r={pearson:.3f}",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        width=1200,
        height=800,
        yaxis=dict(range=[0, max(y) * 1.05]),
    )

    figures_dir = ensure_output_dir(mode)
    save_plot(fig, figures_dir / "jaccard_vs_citations_scatter")


# ---------------------------------------------------------------------------
# Compare mode correlations
# ---------------------------------------------------------------------------
def compare_mode_correlations(
    ai_path: Path | None = None,
    ge_path: Path | None = None,
):
    """
    Compare Pearson correlations (Jaccard distance vs citations) between
    AI and GE fields using Fisher r-to-z test. Logs results.
    """
    ai_path = (
        ai_path or PROCESSED_DATA_DIR / "ARTIFICIAL_INTELLIGENCE" / "interdisciplinary_nodes.csv"
    )
    ge_path = ge_path or PROCESSED_DATA_DIR / "GENE_EXPRESSION" / "interdisciplinary_nodes.csv"

    ai_pearson, n_a = _compute_pearson_after_cleaning(ai_path)
    ge_pearson, n_b = _compute_pearson_after_cleaning(ge_path)

    if ai_pearson is None or ge_pearson is None:
        logger.error("Could not compute both correlations.")
        return

    z_stat, p_val = compare_independent_correlations(ai_pearson, n_a, ge_pearson, n_b)
    if p_val is None:
        logger.error("Failed to compute z / p.")
        return

    diff = ai_pearson - ge_pearson
    logger.info(
        f"AI r={ai_pearson:.4f} (n={n_a}) | GE r={ge_pearson:.4f} (n={n_b}) | "
        f"diff={diff:.4f} | z={z_stat:.3f} | p={p_val:.4g}"
    )
    if p_val < 0.05:
        logger.success(f"Difference significant (p={p_val:.4g})")
    else:
        logger.info(f"Difference not significant (p={p_val:.4g})")


# ===========================================================================
# STREAM 3: FUNDING-CITATION STATISTICAL TESTS
# ===========================================================================


def _rank_biserial(u_stat: float, n1: int, n2: int) -> float:
    """Compute rank-biserial correlation (effect size for Mann-Whitney U)."""
    return 1.0 - (2.0 * u_stat) / (n1 * n2)


def funding_citation_tests():
    """
    For each mode, run formal statistical tests on funded vs. unfunded articles:

    1. Mann-Whitney U test: funded vs. unfunded citation counts
    2. Kruskal-Wallis H test: citation counts across funding tiers (0, 1, 2, 3+)
    3. Spearman correlation: number of funding sources vs. citation count

    Prints structured results table and saves JSON.
    """
    logger.info("Running funding-citation statistical tests...")
    out = ensure_comparative_dir()
    results = {}

    for mode in MODES:
        label = MODE_LABELS[mode]
        df = load_features(mode)
        df = add_citation_count_column(df)
        df = add_grant_columns(df)

        funded = df[df["has_funding"]]["citation_count"].values
        unfunded = df[~df["has_funding"]]["citation_count"].values

        mode_results = {"mode": label, "n_funded": len(funded), "n_unfunded": len(unfunded)}

        # --- Mann-Whitney U: funded vs unfunded ---
        if len(funded) > 0 and len(unfunded) > 0:
            u_stat, p_val = mannwhitneyu(funded, unfunded, alternative="two-sided")
            rb = _rank_biserial(u_stat, len(funded), len(unfunded))
            mode_results["mw_u_stat"] = float(u_stat)
            mode_results["mw_p_value"] = float(p_val)
            mode_results["mw_rank_biserial"] = float(rb)
            mode_results["funded_median"] = float(np.median(funded))
            mode_results["unfunded_median"] = float(np.median(unfunded))
            logger.info(
                f"[{label}] Mann-Whitney U: U={u_stat:.0f}, p={p_val:.4g}, "
                f"r_rb={rb:.4f}, funded_med={np.median(funded):.1f}, "
                f"unfunded_med={np.median(unfunded):.1f}"
            )
        else:
            logger.warning(f"[{label}] Insufficient funded/unfunded data for Mann-Whitney.")

        # --- Kruskal-Wallis: across funding tiers ---
        tiers = {}
        for _, row in df.iterrows():
            n = row["num_funding_sources"]
            bucket = str(min(n, 3)) if n < 3 else "3+"
            tiers.setdefault(bucket, []).append(row["citation_count"])

        tier_groups = [np.array(tiers[k]) for k in sorted(tiers.keys()) if len(tiers[k]) > 0]
        if len(tier_groups) >= 2:
            h_stat, kw_p = kruskal(*tier_groups)
            mode_results["kw_h_stat"] = float(h_stat)
            mode_results["kw_p_value"] = float(kw_p)
            mode_results["kw_n_tiers"] = len(tier_groups)
            tier_medians = {
                k: float(np.median(tiers[k])) for k in sorted(tiers.keys()) if len(tiers[k]) > 0
            }
            mode_results["tier_medians"] = tier_medians
            logger.info(
                f"[{label}] Kruskal-Wallis: H={h_stat:.2f}, p={kw_p:.4g}, medians={tier_medians}"
            )

        # --- Spearman: num_funding_sources vs citation_count ---
        valid = df[["num_funding_sources", "citation_count"]].dropna()
        if len(valid) > 10:
            rho, sp_p = spearmanr(valid["num_funding_sources"], valid["citation_count"])
            mode_results["spearman_rho"] = float(rho)
            mode_results["spearman_p"] = float(sp_p)
            logger.info(f"[{label}] Spearman(funding, citations): rho={rho:.4f}, p={sp_p:.4g}")

        results[mode] = mode_results

    # Save results
    json_path = out / "funding_citation_tests.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.success(f"Saved funding-citation test results to {json_path}")

    return results


# ===========================================================================
# STREAM 5: CROSS-FIELD STATISTICAL TESTS
# ===========================================================================


def cross_field_citation_tests():
    """
    Pairwise Mann-Whitney U tests comparing citation count distributions
    across all 3 fields. Tests whether AI articles receive significantly
    different citation counts than GE or NULL.
    """
    logger.info("Running cross-field citation distribution tests...")
    out = ensure_comparative_dir()

    citation_data = {}
    for mode in MODES:
        df = load_features(mode)
        df = add_citation_count_column(df)
        citation_data[mode] = df["citation_count"].values

    results = {}
    for m1, m2 in combinations(MODES, 2):
        a, b = citation_data[m1], citation_data[m2]
        u_stat, p_val = mannwhitneyu(a, b, alternative="two-sided")
        rb = _rank_biserial(u_stat, len(a), len(b))
        key = f"{MODE_LABELS[m1]} vs {MODE_LABELS[m2]}"
        results[key] = {
            "u_stat": float(u_stat),
            "p_value": float(p_val),
            "rank_biserial": float(rb),
            f"median_{MODE_LABELS[m1]}": float(np.median(a)),
            f"median_{MODE_LABELS[m2]}": float(np.median(b)),
            f"n_{MODE_LABELS[m1]}": len(a),
            f"n_{MODE_LABELS[m2]}": len(b),
        }
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        logger.info(f"  {key}: U={u_stat:.0f}, p={p_val:.4g} {sig}, r_rb={rb:.4f}")

    json_path = out / "cross_field_citation_tests.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.success(f"Saved cross-field citation tests to {json_path}")

    return results


def cross_field_geographic_tests():
    """
    Pairwise 2-sample Kolmogorov-Smirnov tests comparing latitude and
    longitude distributions across all 3 fields. Tests whether geographic
    distributions of publications differ significantly.
    """
    logger.info("Running cross-field geographic distribution tests...")
    out = ensure_comparative_dir()

    geo_data = {}
    for mode in MODES:
        df = load_features(mode)
        df = df.dropna(subset=["matched_lat", "matched_lon"])
        geo_data[mode] = {
            "lat": df["matched_lat"].values,
            "lon": df["matched_lon"].values,
        }

    results = {}
    for m1, m2 in combinations(MODES, 2):
        key = f"{MODE_LABELS[m1]} vs {MODE_LABELS[m2]}"

        lat_stat, lat_p = ks_2samp(geo_data[m1]["lat"], geo_data[m2]["lat"])
        lon_stat, lon_p = ks_2samp(geo_data[m1]["lon"], geo_data[m2]["lon"])

        results[key] = {
            "lat_ks_stat": float(lat_stat),
            "lat_ks_p": float(lat_p),
            "lat_reject_null_005": bool(lat_p < 0.05),
            "lon_ks_stat": float(lon_stat),
            "lon_ks_p": float(lon_p),
            "lon_reject_null_005": bool(lon_p < 0.05),
            f"n_{MODE_LABELS[m1]}": len(geo_data[m1]["lat"]),
            f"n_{MODE_LABELS[m2]}": len(geo_data[m2]["lat"]),
        }
        logger.info(
            f"  {key}: lat KS={lat_stat:.4f} (p={lat_p:.4g}), "
            f"lon KS={lon_stat:.4f} (p={lon_p:.4g})"
        )

    json_path = out / "cross_field_geographic_tests.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.success(f"Saved cross-field geographic tests to {json_path}")

    return results


def cross_field_growth_rate_tests():
    """
    Compare year-over-year growth rates across fields using Mann-Whitney U.
    Computes annual publication growth rates and tests for pairwise
    differences in growth dynamics.
    """
    logger.info("Running cross-field growth rate tests...")
    out = ensure_comparative_dir()

    growth_rates = {}
    for mode in MODES:
        df = load_features(mode)
        df = clean_year_column(df)
        counts = df["year"].value_counts().sort_index()
        # Filter to years with meaningful counts
        counts = counts[counts >= 10]
        rates = counts.pct_change().dropna()
        growth_rates[mode] = rates.values

    results = {}
    for m1, m2 in combinations(MODES, 2):
        a, b = growth_rates[m1], growth_rates[m2]
        key = f"{MODE_LABELS[m1]} vs {MODE_LABELS[m2]}"
        if len(a) > 2 and len(b) > 2:
            u_stat, p_val = mannwhitneyu(a, b, alternative="two-sided")
            results[key] = {
                "u_stat": float(u_stat),
                "p_value": float(p_val),
                f"mean_growth_{MODE_LABELS[m1]}": float(np.mean(a)),
                f"mean_growth_{MODE_LABELS[m2]}": float(np.mean(b)),
                f"median_growth_{MODE_LABELS[m1]}": float(np.median(a)),
                f"median_growth_{MODE_LABELS[m2]}": float(np.median(b)),
            }
            sig = (
                "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            )
            logger.info(
                f"  {key}: U={u_stat:.1f}, p={p_val:.4g} {sig}, "
                f"mean_growth=({np.mean(a):.3f}, {np.mean(b):.3f})"
            )
        else:
            logger.warning(f"  {key}: Not enough data points for growth rate test.")

    json_path = out / "cross_field_growth_rate_tests.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.success(f"Saved cross-field growth rate tests to {json_path}")

    return results


def run_all_statistical_tests():
    """
    Master function to run all statistical tests (Streams 3 + 5) and produce
    a consolidated summary. Prints a formatted results table to the logger.
    """
    logger.info("=" * 70)
    logger.info("RUNNING ALL STATISTICAL TESTS")
    logger.info("=" * 70)

    results = {}

    # Stream 3
    logger.info("\n--- STREAM 3: Funding-Citation Impact ---")
    results["funding_citation"] = funding_citation_tests()

    # Stream 5
    logger.info("\n--- STREAM 5a: Cross-Field Citation Distributions ---")
    results["cross_field_citations"] = cross_field_citation_tests()

    logger.info("\n--- STREAM 5b: Cross-Field Geographic Distributions ---")
    results["cross_field_geographic"] = cross_field_geographic_tests()

    logger.info("\n--- STREAM 5c: Cross-Field Growth Rates ---")
    results["cross_field_growth"] = cross_field_growth_rate_tests()

    # Save consolidated results
    out = ensure_comparative_dir()
    json_path = out / "all_statistical_tests.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.success(f"\nAll statistical test results saved to {json_path}")

    # Print summary table
    logger.info("\n" + "=" * 70)
    logger.info("STATISTICAL TESTS SUMMARY")
    logger.info("=" * 70)

    # Funding summary
    for mode, res in results.get("funding_citation", {}).items():
        if isinstance(res, dict) and "mw_p_value" in res:
            sig = "***" if res["mw_p_value"] < 0.001 else "ns"
            logger.info(
                f"  {res.get('mode', mode)}: Funded vs Unfunded MW p={res['mw_p_value']:.4g} {sig}"
            )

    # Cross-field summary
    for comparison, res in results.get("cross_field_citations", {}).items():
        if isinstance(res, dict):
            sig = "***" if res["p_value"] < 0.001 else "ns"
            logger.info(f"  Citations {comparison}: MW p={res['p_value']:.4g} {sig}")

    for comparison, res in results.get("cross_field_geographic", {}).items():
        if isinstance(res, dict):
            logger.info(
                f"  Geo {comparison}: lat KS p={res['lat_ks_p']:.4g}, "
                f"lon KS p={res['lon_ks_p']:.4g}"
            )

    logger.info("=" * 70)

    return results
