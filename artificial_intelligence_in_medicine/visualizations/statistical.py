"""
Statistical visualizations and hypothesis tests.

Functions for Jaccard distance vs citations scatter plots,
Fisher r-to-z correlation comparison, and Pearson correlation analysis.
"""

from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm as stats_norm

from artificial_intelligence_in_medicine.config import PROCESSED_DATA_DIR
from artificial_intelligence_in_medicine.visualizations.utils import (
    ensure_output_dir,
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
