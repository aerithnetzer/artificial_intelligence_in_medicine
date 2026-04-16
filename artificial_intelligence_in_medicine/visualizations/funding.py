"""
Funding agency visualizations.

Unified functions for analyzing and plotting funding agency impact
on citation graphs -- by citations and by number of publications.
"""

from loguru import logger
import pandas as pd
import plotly.graph_objects as go

from artificial_intelligence_in_medicine.config import FIGURES_DIR
from artificial_intelligence_in_medicine.visualizations.utils import (
    ensure_output_dir,
    load_features,
    save_plot,
)


def _prepare_grants_data(df: pd.DataFrame) -> pd.DataFrame | None:
    """
    Shared helper: filter to records with non-empty grant_list,
    explode grants, extract agency name, clean year.
    Returns prepared DataFrame or None if no valid data.
    """
    df_with_grants = df.dropna(subset=["grant_list"])
    df_with_grants = df_with_grants[
        df_with_grants["grant_list"].apply(lambda x: isinstance(x, list) and len(x) > 0)
    ]

    if len(df_with_grants) == 0:
        logger.warning("No records with non-empty grant lists found.")
        return None

    grants_df = df_with_grants.explode("grant_list")
    grants_df["agency"] = grants_df["grant_list"].apply(
        lambda x: x if isinstance(x, str) else None
    )

    # Clean year
    grants_df["year_clean"] = grants_df["year"].apply(
        lambda x: str(x).split("-")[0] if pd.notna(x) else None
    )
    grants_df["year_clean"] = pd.to_numeric(grants_df["year_clean"], errors="coerce")

    # Add citation count
    if "citation_count" in grants_df.columns:
        grants_df["citations"] = grants_df["citation_count"].fillna(0)
    elif "cited_by" in grants_df.columns:
        grants_df["citations"] = grants_df["cited_by"].apply(
            lambda x: len(x) if isinstance(x, list) else 0
        )
    else:
        grants_df["citations"] = 0

    valid = grants_df.dropna(subset=["agency", "year_clean"])
    if len(valid) == 0:
        logger.warning("No valid records with both agency and year data.")
        return None

    return valid


def funding_agency(mode: str, metric: str = "citations"):
    """
    Top 5 funding agencies per year as a stacked bar chart.

    Parameters
    ----------
    mode : str
        One of ARTIFICIAL_INTELLIGENCE, GENE_EXPRESSION, NULL
    metric : str
        "citations" -- rank agencies by total citations per year
        "papers" -- rank agencies by number of publications funded per year
    """
    df = load_features(mode)
    valid = _prepare_grants_data(df)
    if valid is None:
        return

    if metric == "citations":
        agg_df = valid.groupby(["year_clean", "agency"])["citations"].sum().reset_index()
        agg_df.rename(columns={"year_clean": "year", "citations": "value"}, inplace=True)
        value_label = "Total Citations"
        suffix = "top_5_agencies_by_year"
    else:
        agg_df = valid.groupby(["year_clean", "agency"]).size().reset_index(name="value")
        agg_df.rename(columns={"year_clean": "year"}, inplace=True)
        value_label = "Number of Publications Funded"
        suffix = "top_5_agencies_by_year_number_of_papers"

    # Top 5 per year
    top = agg_df.sort_values(["year", "value"], ascending=[True, False]).groupby("year").head(5)

    if len(top) == 0:
        logger.warning("No data to plot.")
        return

    pivot_df = top.pivot(index="year", columns="agency", values="value").fillna(0).sort_index()
    years = pivot_df.index.astype(str)

    fig = go.Figure()
    for agency in pivot_df.columns:
        fig.add_trace(go.Bar(x=years, y=pivot_df[agency], name=agency))

    title_metric = "Total Citations" if metric == "citations" else "Publications Funded"
    fig.update_layout(
        barmode="stack",
        title=f"Top 5 Funding Agencies by {title_metric} per Year ({mode})",
        xaxis_title="Year",
        yaxis_title=value_label,
        legend_title="Agency",
        xaxis_tickangle=-45,
        width=1200,
        height=700,
        margin=dict(l=40, r=40, t=80, b=120),
    )

    output_path = FIGURES_DIR / mode / suffix
    ensure_output_dir(mode)
    save_plot(fig, output_path)


# Backward-compatible aliases
def funding_agency_number_of_papers(mode: str):
    """Alias for funding_agency(mode, metric='papers')."""
    return funding_agency(mode, metric="papers")
