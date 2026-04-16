"""
Temporal visualizations.

Functions for analyzing and plotting how citation graph structure
changes over time: article counts, citation rates, constraint
evolution, community composition dynamics.
"""

from collections import defaultdict
from pathlib import Path

from loguru import logger
import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from artificial_intelligence_in_medicine.config import FIGURES_DIR, INTERIM_DATA_DIR
from artificial_intelligence_in_medicine.visualizations.utils import (
    COMMUNITY_PALETTE,
    add_citation_count_column,
    clean_year_column,
    save_plot,
)


def normalized_citations_over_time(mode: str):
    """
    Plot proportion of total citations contributed by publications of each year.
    """
    input_path = INTERIM_DATA_DIR / mode / "features_with_ror.json"
    logger.info(f"Loading data from {input_path}...")

    df = pd.read_json(input_path)
    df = add_citation_count_column(df)
    df = clean_year_column(df)

    if df.empty:
        logger.warning("No valid data after cleaning.")
        return

    citations_per_year = df.groupby("year")["citation_count"].sum().sort_index()
    total_citations = citations_per_year.sum()
    if total_citations == 0:
        logger.warning("Total citations equal zero; nothing to plot.")
        return

    normalized = citations_per_year / total_citations

    fig = px.line(
        x=normalized.index,
        y=normalized.values,
        markers=True,
        labels={"x": "Year", "y": "Proportion of Total Citations"},
        title=f"{mode} Normalized Citations Over Time "
        f"(CV={normalized.std() / normalized.mean():.2f})",
        width=1200,
        height=800,
    )
    fig.update_yaxes(tickformat=".2%")

    output_path = FIGURES_DIR / mode / "normalized_citations_over_time"
    save_plot(fig, output_path)


def plot_normalized_articles_over_time(mode: str):
    """
    Dual-axis plot: min-max scaled article counts (left) and raw counts (right) per year.
    """
    input_path = INTERIM_DATA_DIR / mode / "features_with_ror.json"
    df = pd.read_json(input_path)
    df = clean_year_column(df)

    if df.empty:
        logger.warning("Empty dataframe.")
        return

    year_counts = df["year"].value_counts().sort_index()
    min_count, max_count = year_counts.min(), year_counts.max()

    if min_count == max_count:
        scaled_counts = year_counts * 0.0
    else:
        scaled_counts = (year_counts - min_count) / (max_count - min_count)

    cv = scaled_counts.std() / scaled_counts.mean() if scaled_counts.mean() != 0 else 0

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=scaled_counts.index,
            y=scaled_counts.values,
            name="Min-Max Scaled Count",
            mode="lines+markers",
            yaxis="y1",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=year_counts.index,
            y=year_counts.values,
            name="Raw Article Count",
            mode="lines+markers",
            yaxis="y2",
        )
    )
    fig.update_layout(
        title=f"{mode} Articles Over Time (Scaled CV={cv:.2f})",
        xaxis=dict(title="Year"),
        yaxis=dict(title="Min-Max Scaled Article Count", range=[0, 1]),
        yaxis2=dict(title="Raw Article Count", overlaying="y", side="right"),
        legend=dict(x=0.01, y=0.99),
    )

    save_plot(fig, FIGURES_DIR / mode / "articles_over_time_raw_and_scaled")


def _extract_community_year_data(G: nx.Graph) -> pd.DataFrame:
    """
    Shared helper: extract year and community label from graph nodes,
    filtering out singleton-community isolates.
    """
    data = {"year": [], "community": []}

    # Pre-compute community sizes
    community_sizes = defaultdict(int)
    for node_id in G.nodes():
        community = G.nodes[node_id].get("community_label", G.nodes[node_id].get("community"))
        community_sizes[community] += 1

    for node_id in G.nodes():
        node_attrs = G.nodes[node_id]
        degree = G.degree(node_id)
        community = node_attrs.get("community_label", node_attrs.get("community"))

        # Filter out isolated nodes in singleton communities
        if not (degree == 0 and community_sizes[community] == 1):
            year = node_attrs.get("year")
            if year is not None:
                data["year"].append(year)
                data["community"].append(str(community))

    return pd.DataFrame(data).sort_values("year")


def plot_horizontal_timeline(G: nx.Graph, mode: str):
    """
    Scatter-style horizontal timeline: community presence over years,
    marker size = publication count.
    """
    df = _extract_community_year_data(G)
    community_counts = df.groupby(["year", "community"]).size().reset_index(name="count")
    unique_communities = sorted(community_counts["community"].unique())
    color_map = {
        c: COMMUNITY_PALETTE[i % len(COMMUNITY_PALETTE)] for i, c in enumerate(unique_communities)
    }

    timeline_data = []
    for i, community in enumerate(unique_communities):
        cd = community_counts[community_counts["community"] == community]
        for _, row in cd.iterrows():
            timeline_data.append(
                {
                    "year": row["year"],
                    "community": community,
                    "community_y": i,
                    "count": row["count"],
                }
            )
    timeline_df = pd.DataFrame(timeline_data)

    fig = px.scatter(
        timeline_df,
        x="year",
        y="community_y",
        size="count",
        color="community",
        title=f"Community Timeline ({mode})",
        labels={"year": "Year", "community_y": "Community", "count": "Publications"},
        color_discrete_map=color_map,
        size_max=30,
    )
    fig.update_layout(
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(len(unique_communities))),
            ticktext=unique_communities,
            title="Community",
        ),
        xaxis_title="Year",
        showlegend=False,
        height=max(400, len(unique_communities) * 50),
    )
    for i in range(len(unique_communities)):
        fig.add_hline(y=i, line_dash="dot", line_color="lightgray", opacity=0.5)

    save_plot(fig, FIGURES_DIR / mode / "horizontal_timeline")


def plot_communities_vertical_barchart(G: nx.Graph, mode: str):
    """
    Stacked bar chart showing community composition over time.
    """
    df = _extract_community_year_data(G)
    community_counts = df.groupby(["year", "community"]).size().reset_index(name="count")
    unique_communities = sorted(community_counts["community"].unique())
    color_map = {
        c: COMMUNITY_PALETTE[i % len(COMMUNITY_PALETTE)] for i, c in enumerate(unique_communities)
    }

    fig = px.bar(
        community_counts,
        x="year",
        y="count",
        color="community",
        title=f"Community Composition Over Time ({mode})",
        labels={"year": "Year", "count": "Publications", "community": "Community"},
        category_orders={"community": unique_communities},
        color_discrete_map=color_map,
    )
    fig.update_layout(
        xaxis_type="linear",
        xaxis_title="Year",
        yaxis_title="Number of Publications",
        legend_title="Community",
        bargap=0.2,
    )

    save_plot(fig, FIGURES_DIR / mode / "community_composition_over_time")


def horizontal_timeline(graph_path: Path, features_path: Path, output_path: Path):
    """
    Backward-compatible wrapper for plot_horizontal_timeline.
    Loads a pickled graph and calls the core function.
    """
    import pickle

    with open(graph_path, "rb") as f:
        G = pickle.load(f)

    mode = (
        output_path.parent.name
        if output_path.parent.name in ("ARTIFICIAL_INTELLIGENCE", "GENE_EXPRESSION", "NULL")
        else "UNKNOWN"
    )

    plot_horizontal_timeline(G, mode)


# Backward-compatible aliases
normalized_articles_over_time = plot_normalized_articles_over_time
