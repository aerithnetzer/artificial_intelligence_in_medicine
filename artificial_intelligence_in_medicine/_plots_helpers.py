from pathlib import Path
import pickle
import networkx as nx
import igraph as ig
from loguru import logger
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import norm as stats_norm
from tqdm import tqdm
import typer
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from artificial_intelligence_in_medicine.config import (
    FIGURES_DIR,
    INTERIM_DATA_DIR,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)
from datetime import datetime


def plot_communities(G: nx.Graph, MODE: str):
    output_path: Path = FIGURES_DIR / MODE / "communities.html"
    communities = nx.get_node_attributes(G, "community")
    community_names = {}
    if nx.get_node_attributes(G, "title"):
        community_titles = defaultdict(list)
        for n, data in G.nodes(data=True):
            community_id = data.get("community")
            title = data.get("title")
            if title:
                community_titles[community_id].append(title)

        for community_id, titles in community_titles.items():
            if titles:
                try:
                    vectorizer = TfidfVectorizer(stop_words="english", max_features=5)
                    vectorizer.fit_transform(titles)
                    top_terms = vectorizer.get_feature_names_out()
                    community_names[community_id] = ", ".join(top_terms)
                except ValueError:
                    community_names[community_id] = f"Community {community_id}"
            else:
                community_names[community_id] = f"Community {community_id}"
    else:
        community_names = {
            i: f"""Community {i}
            """
            for i in set(communities.values())
        }

    # Save community labels
    nx.set_node_attributes(
        G, {n: community_names[communities[n]] for n in G.nodes()}, "community_label"
    )

    print("Graph and labels saved. Visualizing...")

    # Visualization
    try:
        pos = nx.spring_layout(G, seed=42, k=0.15 if G.number_of_nodes() < 1000 else None)
        x_coords = [pos[n][0] for n in G.nodes()]
        y_coords = [pos[n][1] for n in G.nodes()]

        node_titles = [data.get("title", f"Node {n}") for n, data in G.nodes(data=True)]

        colors_discrete = px.colors.qualitative.Set1 + px.colors.qualitative.Set3
        node_colors = [
            colors_discrete[data["community"] % len(colors_discrete)]
            for n, data in G.nodes(data=True)
        ]

        # Edges
        edge_x, edge_y = [], []
        edge_shapes = []
        for u, v in G.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_shapes.append(
                dict(
                    type="line",
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    line=dict(color="#888", width=0.5),
                    opacity=0.7,
                    layer="below",
                )
            )

        # Node sizes
        citation_counts = [
            len(data.get("cited_by", [])) if data.get("cited_by") else 0
            for _, data in G.nodes(data=True)
        ]
        node_sizes = [10 + 2 * np.log(c + 1) for c in citation_counts]
        print(len(node_sizes), "node sizes calculated")

        node_trace = go.Scatter(
            x=x_coords,
            y=y_coords,
            mode="markers",
            hoverinfo="text",
            text=[
                f"Title: {title}<br>"
                f"Community: {data['community_label']}<br>"
                f"Cited by: {len(data.get('cited_by', []))}"
                for (n, data), title in zip(G.nodes(data=True), node_titles)
            ],
            marker=dict(size=node_sizes, color=node_colors, line=dict(width=2, color="white")),
        )

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=0.5, color="#888"),
            hoverinfo="none",
            mode="lines",
        )

        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=(
                    f"Interactive Graph Visualization<br>"
                    f"{len(set(communities.values()))} communities, "
                ),
                showlegend=False,
                hovermode="closest",
                margin=dict(b=20, l=5, r=5, t=40),
                annotations=[
                    dict(
                        text="Hover over nodes to see titles",
                        showarrow=False,
                        xref="paper",
                        yref="paper",
                        x=0.005,
                        y=-0.002,
                        xanchor="left",
                        yanchor="bottom",
                        font=dict(color="gray", size=12),
                    )
                ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                width=1800,
                height=1200,
                shapes=edge_shapes,
            ),
        )

        fig.write_html(output_path)

        png_output_path = output_path.with_suffix(".png")
        fig.write_image(png_output_path, width=1200, height=800, scale=5)

    except Exception as e:
        print(f"Could not plot communities: {e}")


def plot_horizontal_timeline(G, MODE: str):
    features_path: Path = INTERIM_DATA_DIR / MODE / "features_with_ror.json"
    fig_output_path: Path = FIGURES_DIR / MODE / "horizontal_timeline.html"
    """Generate horizontal timeline showing community presence over years."""
    features_df = pd.read_json(features_path)
    features_df["pmid"] = features_df["pmid"].astype(str)

    data = {
        "year": [],
        "community": [],
    }

    for node_id in G.nodes():
        node_attrs = G.nodes[node_id]  # Access node attributes
        degree = G.degree(node_id)  # Get degree of specific node

        # Get community from node attributes
        community = node_attrs.get("community_label", node_attrs.get("community"))

        # Count nodes with same community
        same_community_count = sum(
            1
            for other_node in G.nodes()
            if G.nodes[other_node].get("community_label", G.nodes[other_node].get("community"))
            == community
        )

        # Filter out isolated nodes in singleton communities
        if not (degree == 0 and same_community_count == 1):
            data["year"].append(node_attrs["year"])
            data["community"].append(str(community))

    df = pd.DataFrame(data)
    df = df.sort_values("year")
    community_counts_by_year = df.groupby(["year", "community"]).size().reset_index(name="count")

    # Create color mapping using same logic as original
    colors_discrete = px.colors.qualitative.Set1 + px.colors.qualitative.Set3
    unique_communities = sorted(community_counts_by_year["community"].unique())
    color_map = {
        community: colors_discrete[i % len(colors_discrete)]
        for i, community in enumerate(unique_communities)
    }

    print("Generating horizontal timeline of communities...")

    # Create timeline data - each community gets a horizontal bar
    timeline_data = []
    for i, community in enumerate(unique_communities):
        community_data = community_counts_by_year[
            community_counts_by_year["community"] == community
        ]
        for _, row in community_data.iterrows():
            timeline_data.append(
                {
                    "year": row["year"],
                    "community": community,
                    "community_y": i,  # Y position for horizontal layout
                    "count": row["count"],
                    "color": color_map[community],
                }
            )

    timeline_df = pd.DataFrame(timeline_data)

    # Create scatter plot with sized markers for timeline effect
    fig = px.scatter(
        timeline_df,
        x="year",
        y="community_y",
        size="count",
        color="community",
        title="Community Timeline - Horizontal View",
        labels={"year": "Year", "community_y": "Community", "count": "Publications"},
        color_discrete_map=color_map,
        size_max=30,
    )

    # Update layout for timeline appearance
    fig.update_layout(
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(len(unique_communities))),
            ticktext=unique_communities,
            title="Community",
        ),
        xaxis_title="Year",
        showlegend=False,  # Remove legend since y-axis shows communities
        # Scale height with number of communities
        height=max(400, len(unique_communities) * 50),
        bargap=0.2,
    )

    # Add horizontal lines to separate communities
    for i in range(len(unique_communities)):
        fig.add_hline(y=i, line_dash="dot", line_color="lightgray", opacity=0.5)

    # Fixed: create parent directory
    fig_output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(fig_output_path)
    logger.success(f"""Horizontal community timeline saved as '{fig_output_path}'""")


def make_scatterplot_visualization():
    # Make a scatterplot where the x axis is year and node radius is citation count
    return None


def fisher_r_to_z(r: float) -> float:
    # Clamp to avoid infinities
    r = max(min(r, 0.999999), -0.999999)
    return 0.5 * np.log((1 + r) / (1 - r))


def plot_communities_vertical_barchart(MODE: str, G: nx.Graph):
    features_path: Path = INTERIM_DATA_DIR / MODE / "features_with_ror.json"
    output_path: Path = FIGURES_DIR / MODE / "horizontal_timeline.html"
    """Generate horizontal timeline showing community presence over years."""
    features_df = pd.read_json(features_path)
    features_df["pmid"] = features_df["pmid"].astype(str)

    data = {
        "year": [],
        "community": [],
    }

    for node_id in G.nodes():
        node_attrs = G.nodes[node_id]  # Access node attributes
        degree = G.degree(node_id)  # Get degree of specific node

        # Get community from node attributes
        community = node_attrs.get("community_label", node_attrs.get("community"))

        # Count nodes with same community
        same_community_count = sum(
            1
            for other_node in G.nodes()
            if G.nodes[other_node].get("community_label", G.nodes[other_node].get("community"))
            == community
        )

        # Filter out isolated nodes in singleton communities
        if not (degree == 0 and same_community_count == 1):
            data["year"].append(node_attrs["year"])
            data["community"].append(str(community))

    df = pd.DataFrame(data)
    df = df.sort_values("year")
    community_counts_by_year = df.groupby(["year", "community"]).size().reset_index(name="count")

    # Create color mapping using same logic as original
    colors_discrete = px.colors.qualitative.Set1 + px.colors.qualitative.Set3
    unique_communities = sorted(community_counts_by_year["community"].unique())
    color_map = {
        community: colors_discrete[i % len(colors_discrete)]
        for i, community in enumerate(unique_communities)
    }

    print("Generating horizontal timeline of communities...")

    # Create timeline data - each community gets a horizontal bar
    timeline_data = []
    for i, community in enumerate(unique_communities):
        community_data = community_counts_by_year[
            community_counts_by_year["community"] == community
        ]
        for _, row in community_data.iterrows():
            timeline_data.append(
                {
                    "year": row["year"],
                    "community": community,
                    "community_y": i,  # Y position for horizontal layout
                    "count": row["count"],
                    "color": color_map[community],
                }
            )

    timeline_df = pd.DataFrame(timeline_data)

    print("Generating stacked bar chart of communities over time...")

    colors_discrete = px.colors.qualitative.Set1 + px.colors.qualitative.Set3
    unique_communities = sorted(community_counts_by_year["community"].unique())
    color_map = {
        community: colors_discrete[i % len(colors_discrete)]
        for i, community in enumerate(unique_communities)
    }

    fig = px.bar(
        community_counts_by_year,
        x="year",
        y="count",
        color="community",
        title="Community Composition Over Time",
        labels={"year": "Year", "count": "Number of Publications", "community": "Community"},
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

    fig.write_html(output_path)
    print(f"Community evolution plot saved as '{output_path}'")


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


def test_pearson_correlation_diff_statistically_significant(
    AI_pearson: float,
    GE_pearson: float,
    n_A: int,
    n_B: int,
    alpha: float = 0.05,
):
    """
    Wrapper returning dict using Fisher r-to-z (kept for backward compatibility).
    """
    z_score, p_value = compare_independent_correlations(AI_pearson, n_A, GE_pearson, n_B)
    if z_score is None:
        return {"z_score": None, "p_value": None, "significant": False}
    return {
        "z_score": z_score,
        "p_value": p_value,
        "significant": p_value < alpha,
    }


def normalized_citations_over_time(MODE: str):
    input_path: Path = (INTERIM_DATA_DIR / MODE / "features_with_ror.json",)
    """
    Plot proportion of total citations contributed by publications of each year.
    (Total citations in that year) / (Total citations across all years).
    """
    logger.info(f"Loading data from {input_path}...")
    try:
        df = pd.read_json(input_path)
    except FileNotFoundError:
        logger.error(f"Input file not found at {input_path}")
        raise typer.Exit(code=1)

    if "year" not in df.columns:
        logger.error("Input JSON must contain 'year' column.")
        raise typer.Exit(code=1)

    if df.empty:
        logger.warning("Input dataframe is empty.")
        return

    # Determine citation counts per row
    if "citation_count" in df.columns:
        df["__citations"] = df["citation_count"].fillna(0)
    elif "cited_by" in df.columns:
        df["__citations"] = df["cited_by"].apply(lambda x: len(x) if isinstance(x, list) else 0)
    else:
        logger.error("No citation field found ('citation_count' or 'cited_by').")
        raise typer.Exit(code=1)

    # Clean year values (ensure numeric)
    df_year = df[pd.notna(df["year"])].copy()
    df_year["year"] = pd.to_numeric(df_year["year"], errors="coerce")
    df_year = df_year.dropna(subset=["year"])
    if df_year.empty:
        logger.warning("No valid year data after cleaning.")
        return
    df_year["year"] = df_year["year"].astype(int)

    citations_per_year = df_year.groupby("year")["__citations"].sum().sort_index()
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
        title=f"""{MODE} Normalized Citations Over Time (CV={
            normalized.std() / normalized.mean():.2f})""",
        width=1200,
        height=800,
    )
    fig.update_yaxes(tickformat=".2%")
    fig.write_image(
        FIGURES_DIR / MODE / "normalized_citations_over_time.png", scale=5, width=1200, height=800
    )
    logger.success("Displayed normalized citations over time plot.")


def plot_normalized_articles_over_time(MODE: str):
    input_path: Path = INTERIM_DATA_DIR / MODE / "features_with_ror.json"
    """
    Generates a line plot showing:
    - Min–max scaled article counts per year (left y-axis)
    - Raw article counts per year (right y-axis)
    """
    logger.info(f"Loading data from {input_path}...")
    try:
        df = pd.read_json(input_path)
    except FileNotFoundError:
        logger.error(f"Input file not found at {input_path}")
        logger.error(
            "Please run the 'identify-jumped-articles.py' script first to generate the data."
        )
        raise typer.Exit(code=1)

    if "year" not in df.columns:
        logger.error("Input JSON must contain 'year' column.")
        raise typer.Exit(code=1)

    # Count articles per year
    year_counts = df["year"].value_counts().sort_index()

    min_count = year_counts.min()
    max_count = year_counts.max()

    if min_count == max_count:
        logger.warning("All years have the same article count; min–max scaling will be zero.")
        scaled_counts = year_counts * 0.0
    else:
        scaled_counts = (year_counts - min_count) / (max_count - min_count)

    cv = scaled_counts.std() / scaled_counts.mean()

    # Create figure
    fig = go.Figure()

    # Scaled counts (left y-axis)
    fig.add_trace(
        go.Scatter(
            x=scaled_counts.index,
            y=scaled_counts.values,
            name="Min–Max Scaled Count",
            mode="lines+markers",
            yaxis="y1",
        )
    )

    # Raw counts (right y-axis)
    fig.add_trace(
        go.Scatter(
            x=year_counts.index,
            y=year_counts.values,
            name="Raw Article Count",
            mode="lines+markers",
            yaxis="y2",
        )
    )

    # Layout with dual y-axes
    fig.update_layout(
        title=f"{MODE} Articles Over Time (Scaled CV={cv:.2f})",
        xaxis=dict(title="Year"),
        yaxis=dict(
            title="Min–Max Scaled Article Count",
            range=[0, 1],
        ),
        yaxis2=dict(
            title="Raw Article Count",
            overlaying="y",
            side="right",
        ),
        legend=dict(x=0.01, y=0.99),
    )

    fig.write_html(str(FIGURES_DIR / MODE / "articles_over_time_raw_and_scaled.html"))


def scatterplot_with_line_of_best_fit(
    input_path: Path,
    mode: str,
):
    """
    Calculates and prints the correlation between Jaccard distance and the number of citations,
    removes outliers, and saves a scatterplot with a line of best fit.
    """
    logger.info(f"Loading data from {input_path}...")
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        logger.error(f"Input file not found at {input_path}")
        logger.error(
            "Please run the 'identify-jumped-articles.py' script first to generate the data."
        )
        raise typer.Exit(code=1)

    required_columns = ["jaccard_distance", "num_cited_by"]
    if not all(col in df.columns for col in required_columns):
        logger.error(f"Input CSV must contain {required_columns} columns.")
        raise typer.Exit(code=1)

    logger.info("Calculating correlation between 'jaccard_distance' and 'num_cited_by'...")

    # Drop rows with missing values in the relevant columns
    df_corr = df[required_columns].dropna()

    # Remove outliers using the IQR method for both columns
    Q1 = df_corr.quantile(0.25)
    Q3 = df_corr.quantile(0.75)
    IQR = Q3 - Q1
    df_corr = df_corr[
        (
            (
                df_corr["jaccard_distance"]
                >= (Q1["jaccard_distance"] - 1.5 * IQR["jaccard_distance"])
            )
            & (
                df_corr["jaccard_distance"]
                <= (Q3["jaccard_distance"] + 1.5 * IQR["jaccard_distance"])
            )
            & (df_corr["num_cited_by"] >= (Q1["num_cited_by"] - 1.5 * IQR["num_cited_by"]))
            & (df_corr["num_cited_by"] <= (Q3["num_cited_by"] + 1.5 * IQR["num_cited_by"]))
        )
    ]

    if df_corr.empty or (df_corr["jaccard_distance"].nunique() <= 1):
        logger.warning(
            "No data available to calculate correlation after dropping NaNs and removing outliers, or all Jaccard distances are identical."
        )
        return

    # Calculate Pearson correlation
    pearson_corr = df_corr["jaccard_distance"].corr(df_corr["num_cited_by"], method="pearson")
    logger.info(
        f"""Pearson correlation between Jaccard distance and number of citations: {
            pearson_corr:.4f}"""
    )

    # Calculate Spearman correlation
    spearman_corr = df_corr["jaccard_distance"].corr(df_corr["num_cited_by"], method="spearman")
    logger.info(
        f"""Spearman correlation between Jaccard distance and number of citations: {
            spearman_corr:.4f}"""
    )

    figures_dir = FIGURES_DIR / mode
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig_path_html = figures_dir / "jaccard_vs_citations_scatter.html"
    fig_path_png = figures_dir / "jaccard_vs_citations_scatter.png"
    x = df_corr["jaccard_distance"]
    y = df_corr["num_cited_by"]

    # Line of best fit
    m, b = np.polyfit(x, y, 1)
    best_fit_y = m * x + b

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="Data", marker=dict(opacity=0.6)))
    fig.add_trace(
        go.Scatter(
            x=x, y=best_fit_y, mode="lines", name="Best fit line", line=dict(color="red", width=2)
        )
    )
    fig.update_layout(
        xaxis_title="Jaccard Distance",
        yaxis_title="Number of Citations",
        title="Jaccard Distance vs Number of Citations",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        width=2000,
        height=1200,
        # Extend y-axis to slightly above max citations
        yaxis=dict(range=[0, max(y) * 1.05]),
    )
    fig.write_html(str(fig_path_html))
    fig.write_image(str(fig_path_png), scale=10)
    logger.info(f"Scatterplot saved to {fig_path_html} and {fig_path_png}")


# --- ADDED CODE: helper + comparison command ---
def _compute_pearson_after_cleaning(csv_path: Path):
    required_columns = ["jaccard_distance", "num_cited_by"]
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        logger.error(f"File not found: {csv_path}")
        return None, None
    if not all(col in df.columns for col in required_columns):
        logger.error(f"Missing required columns in {csv_path}")
        return None, None
    df_corr = df[required_columns].dropna()
    if df_corr.empty:
        logger.warning(f"No rows after dropna for {csv_path}")
        return None, None

    Q1 = df_corr.quantile(0.25)
    Q3 = df_corr.quantile(0.75)
    IQR = Q3 - Q1
    mask = (
        (df_corr["jaccard_distance"] >= Q1["jaccard_distance"] - 1.5 * IQR["jaccard_distance"])
        & (df_corr["jaccard_distance"] <= Q3["jaccard_distance"] + 1.5 * IQR["jaccard_distance"])
        & (df_corr["num_cited_by"] >= Q1["num_cited_by"] - 1.5 * IQR["num_cited_by"])
        & (df_corr["num_cited_by"] <= Q3["num_cited_by"] + 1.5 * IQR["num_cited_by"])
    )
    df_corr = df_corr[mask]
    n = len(df_corr)
    if n < 4 or df_corr["jaccard_distance"].nunique() <= 1:
        logger.warning(f"Insufficient variance / size after filtering for {csv_path}")
        return None, n
    pearson = df_corr["jaccard_distance"].corr(df_corr["num_cited_by"], method="pearson")
    logger.info(f"Pearson ({csv_path.name}): {pearson:.4f} (n={n})")
    return pearson, n


def compare_mode_correlations(
    ai_path: Path = PROCESSED_DATA_DIR / "ARTIFICIAL_INTELLIGENCE" / "interdisciplinary_nodes.csv",
    ge_path: Path = PROCESSED_DATA_DIR / "GENE_EXPRESSION" / "interdisciplinary_nodes.csv",
):
    """
    Compare Pearson correlations (Jaccard distance vs citations) between modes and report exact p-value.
    """
    ai_pearson, n_A = _compute_pearson_after_cleaning(ai_path)
    ge_pearson, n_B = _compute_pearson_after_cleaning(ge_path)

    if ai_pearson is None or ge_pearson is None:
        logger.error("Could not compute both correlations. Abort.")
        raise typer.Exit(code=1)

    z_stat, p_val = compare_independent_correlations(ai_pearson, n_A, ge_pearson, n_B)
    if p_val is None:
        logger.error("Failed to compute z / p.")
        raise typer.Exit(code=1)

    diff = ai_pearson - ge_pearson
    logger.info(
        f"""AI r={ai_pearson:.4f} (n={n_A}) | GE r={ge_pearson:.4f} (n={n_B}) | diff={diff:.4f} "
        f"| z={z_stat:.3f} | p={p_val:.4g}"""
    )
    if p_val < 0.05:
        logger.success(f"Difference significant (p={p_val:.4g})")
    else:
        logger.info(f"Difference not significant (p={p_val:.4g})")


# --- END ADDED CODE ---


def funding_agency(MODE: str):
    input_path: Path = INTERIM_DATA_DIR / MODE / "features_with_ror.json"
    output_path: Path = FIGURES_DIR / MODE / "top_5_agencies_by_year.html"
    """
    This script reads publication data, identifies the top 5 funding agencies
    for each year based on total citations, and generates a stacked bar chart
    to visualize the results using Plotly.
    """
    import plotly.graph_objects as go
    import plotly.io as pio

    df = pd.read_json(input_path)
    print(f"Total records in dataset: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")

    # Check grant_list column
    print("\nGrant list info:")
    print(f"Non-null grant_list count: {df['grant_list'].notna().sum()}")
    print(f"Null grant_list count: {df['grant_list'].isna().sum()}")

    # Sample some grant_list values
    non_null_grants = df[df["grant_list"].notna()]
    if len(non_null_grants) > 0:
        print("\nSample grant_list values:")
        for i in range(min(3, len(non_null_grants))):
            print(f"Record {i}: {non_null_grants.iloc[i]['grant_list']}")
            print(f"Type: {type(non_null_grants.iloc[i]['grant_list'])}")

    # Check year column
    print("\nYear info:")
    print(f"Non-null year count: {df['year'].notna().sum()}")
    if "year" in df.columns:
        print(f"Year range: {df['year'].min()} to {df['year'].max()}")
        print(f"Year value types: {df['year'].apply(type).value_counts()}")

    # Filter out rows with no grant data
    df_with_grants = df.dropna(subset=["grant_list"])
    print(f"\nRecords with grant data: {len(df_with_grants)}")

    if len(df_with_grants) == 0:
        print("No records with grant data found. Cannot generate plot.")
        return

    # Filter out empty grant lists
    df_with_grants = df_with_grants[
        df_with_grants["grant_list"].apply(lambda x: isinstance(x, list) and len(x) > 0)
    ]
    print(f"Records with non-empty grant lists: {len(df_with_grants)}")

    if len(df_with_grants) == 0:
        print("No records with non-empty grant lists found. Cannot generate plot.")
        return

    # Explode the list of grants into separate rows
    grants_df = df_with_grants.explode("grant_list")
    print(f"\nAfter explode: {len(grants_df)} records")

    if len(grants_df) > 0:
        print("Sample exploded grant_list values:")
        for i in range(min(3, len(grants_df))):
            print(f"Record {i}: {grants_df.iloc[i]['grant_list']}")
            print(f"Type: {type(grants_df.iloc[i]['grant_list'])}")

    # Extract agency from each grant dictionary
    grants_df["agency"] = grants_df["grant_list"].apply(
        lambda x: x if isinstance(x, str) else None
    )

    # Check extracted agencies
    print("\nAgency extraction:")
    print(f"Non-null agencies: {grants_df['agency'].notna().sum()}")
    print(f"Unique agencies: {grants_df['agency'].nunique()}")
    if grants_df["agency"].notna().sum() > 0:
        print(f"""Sample agencies: {grants_df["agency"].dropna().head().tolist()}""")

    # Clean and validate year data
    if "year" not in grants_df.columns:
        print("Error: 'year' column not found in the data.")
        return

    # Handle different year formats
    grants_df["year_clean"] = grants_df["year"].apply(
        lambda x: str(x).split("-")[0] if pd.notna(x) else None
    )
    grants_df["year_clean"] = pd.to_numeric(grants_df["year_clean"], errors="coerce")

    print("\nYear processing:")
    print(f"Valid years: {grants_df['year_clean'].notna().sum()}")
    if grants_df["year_clean"].notna().sum() > 0:
        print(
            f"""Year range: {grants_df["year_clean"].min()} to {grants_df["year_clean"].max()}"""
        )

    # Use citation_count if available, otherwise calculate from cited_by
    if "citation_count" in grants_df.columns:
        grants_df["citations"] = grants_df["citation_count"].fillna(0)
    else:
        grants_df["citations"] = grants_df["cited_by"].apply(
            lambda x: len(x) if isinstance(x, list) else 0
        )

    print("\nCitation data:")
    print(f"Total citations: {grants_df['citations'].sum()}")
    print(f"Mean citations per record: {grants_df['citations'].mean():.2f}")

    # Filter for valid data
    valid_data = grants_df.dropna(subset=["agency", "year_clean"])
    print(f"\nValid records (with agency and year): {len(valid_data)}")

    if len(valid_data) == 0:
        print("No valid records with both agency and year data. Cannot generate plot.")
        return

    # Sum citations for each agency by year
    agency_citations_by_year = (
        valid_data.groupby(["year_clean", "agency"])["citations"].sum().reset_index()
    )
    agency_citations_by_year.rename(
        columns={"year_clean": "year", "citations": "citation_count"}, inplace=True
    )

    print(f"""\nAggregated data: {len(agency_citations_by_year)} agency-year combinations""")

    # Get the top 5 agencies for each year based on citations
    top_agencies_per_year = (
        agency_citations_by_year.sort_values(["year", "citation_count"], ascending=[True, False])
        .groupby("year")
        .head(5)
    )

    print("Top 5 most cited agencies by year (based on total citations):")
    print(top_agencies_per_year)

    if len(top_agencies_per_year) == 0:
        print("No data to plot.")
        return

    # Plotting the results as a stacked bar chart with Plotly
    try:
        # Pivot data for stacked bar chart
        pivot_df = top_agencies_per_year.pivot(
            index="year", columns="agency", values="citation_count"
        ).fillna(0)
        pivot_df = pivot_df.sort_index()
        years = pivot_df.index.astype(str)
        fig = go.Figure()
        for agency in pivot_df.columns:
            fig.add_trace(go.Bar(x=years, y=pivot_df[agency], name=agency))
        fig.update_layout(
            barmode="stack",
            title="Top 5 Funding Agencies by Total Citations per Year (Stacked)",
            xaxis_title="Year",
            yaxis_title="Total Number of Citations",
            legend_title="Agency",
            xaxis_tickangle=-45,
            autosize=False,
            width=1200,
            height=700,
            margin=dict(l=40, r=40, t=80, b=120),
        )
        print("\nPlotting stacked bar chart with Plotly. Opening in browser or saving as HTML.")
        pio.write_html(fig, file=output_path, auto_open=True)
    except ImportError:
        print("\nPlease install plotly to see the plot:")
        print("pip install plotly")


def funding_agency_number_of_papers(MODE: str):
    input_path: Path = INTERIM_DATA_DIR / MODE / "features_with_ror.json"
    output_path: Path = FIGURES_DIR / MODE / "top_5_agencies_by_year_number_of_papers.html"
    """
    This script reads publication data, identifies the top 5 funding agencies
    for each year based on number of publications funded, and generates a stacked bar chart
    to visualize the results using Plotly.
    """
    import plotly.graph_objects as go
    import plotly.io as pio

    df = pd.read_json(input_path)
    print(f"Total records in dataset: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")

    # Check grant_list column
    print("\nGrant list info:")
    print(f"Non-null grant_list count: {df['grant_list'].notna().sum()}")
    print(f"Null grant_list count: {df['grant_list'].isna().sum()}")

    # Sample some grant_list values
    non_null_grants = df[df["grant_list"].notna()]
    if len(non_null_grants) > 0:
        print("\nSample grant_list values:")
        for i in range(min(3, len(non_null_grants))):
            print(f"Record {i}: {non_null_grants.iloc[i]['grant_list']}")
            print(f"Type: {type(non_null_grants.iloc[i]['grant_list'])}")

    # Check year column
    print("\nYear info:")
    print(f"Non-null year count: {df['year'].notna().sum()}")
    if "year" in df.columns:
        print(f"Year range: {df['year'].min()} to {df['year'].max()}")
        print(f"Year value types: {df['year'].apply(type).value_counts()}")

    # Filter out rows with no grant data
    df_with_grants = df.dropna(subset=["grant_list"])
    print(f"\nRecords with grant data: {len(df_with_grants)}")

    if len(df_with_grants) == 0:
        print("No records with grant data found. Cannot generate plot.")
        return

    # Filter out empty grant lists
    df_with_grants = df_with_grants[
        df_with_grants["grant_list"].apply(lambda x: isinstance(x, list) and len(x) > 0)
    ]
    print(f"Records with non-empty grant lists: {len(df_with_grants)}")

    if len(df_with_grants) == 0:
        print("No records with non-empty grant lists found. Cannot generate plot.")
        return

    # Explode the list of grants into separate rows
    grants_df = df_with_grants.explode("grant_list")
    print(f"\nAfter explode: {len(grants_df)} records")

    if len(grants_df) > 0:
        print("Sample exploded grant_list values:")
        for i in range(min(3, len(grants_df))):
            print(f"Record {i}: {grants_df.iloc[i]['grant_list']}")
            print(f"Type: {type(grants_df.iloc[i]['grant_list'])}")

    # Extract agency from each grant dictionary
    grants_df["agency"] = grants_df["grant_list"].apply(
        lambda x: x if isinstance(x, str) else None
    )

    # Check extracted agencies
    print("\nAgency extraction:")
    print(f"Non-null agencies: {grants_df['agency'].notna().sum()}")
    print(f"Unique agencies: {grants_df['agency'].nunique()}")
    if grants_df["agency"].notna().sum() > 0:
        print(f"""Sample agencies: {grants_df["agency"].dropna().head().tolist()}""")

    # Clean and validate year data
    if "year" not in grants_df.columns:
        print("Error: 'year' column not found in the data.")
        return

    # Handle different year formats
    grants_df["year_clean"] = grants_df["year"].apply(
        lambda x: str(x).split("-")[0] if pd.notna(x) else None
    )
    grants_df["year_clean"] = pd.to_numeric(grants_df["year_clean"], errors="coerce")

    print("\nYear processing:")
    print(f"Valid years: {grants_df['year_clean'].notna().sum()}")
    if grants_df["year_clean"].notna().sum() > 0:
        print(
            f"""Year range: {grants_df["year_clean"].min()} to {grants_df["year_clean"].max()}"""
        )

    # Use publication count instead of citations
    # Each row is a publication-agency-year combination after explode

    # Filter for valid data
    valid_data = grants_df.dropna(subset=["agency", "year_clean"])
    print(f"\nValid records (with agency and year): {len(valid_data)}")

    if len(valid_data) == 0:
        print("No valid records with both agency and year data. Cannot generate plot.")
        return

    # Count publications for each agency by year
    agency_pubs_by_year = (
        valid_data.groupby(["year_clean", "agency"]).size().reset_index(name="publication_count")
    )
    agency_pubs_by_year.rename(columns={"year_clean": "year"}, inplace=True)

    print(f"""\nAggregated data: {len(agency_pubs_by_year)} agency-year combinations""")

    # Get the top 5 agencies for each year based on publication count
    top_agencies_per_year = (
        agency_pubs_by_year.sort_values(["year", "publication_count"], ascending=[True, False])
        .groupby("year")
        .head(5)
    )

    print("Top 5 agencies by number of publications funded per year:")
    print(top_agencies_per_year)

    if len(top_agencies_per_year) == 0:
        print("No data to plot.")
        return

    # Plotting the results as a stacked bar chart with Plotly
    try:
        # Pivot data for stacked bar chart
        pivot_df = top_agencies_per_year.pivot(
            index="year", columns="agency", values="publication_count"
        ).fillna(0)
        pivot_df = pivot_df.sort_index()
        years = pivot_df.index.astype(str)
        fig = go.Figure()
        for agency in pivot_df.columns:
            fig.add_trace(go.Bar(x=years, y=pivot_df[agency], name=agency))
        fig.update_layout(
            barmode="stack",
            title="Top 5 Funding Agencies by Number of Publications Funded per Year (Stacked)",
            xaxis_title="Year",
            yaxis_title="Number of Publications Funded",
            legend_title="Agency",
            xaxis_tickangle=-45,
            autosize=False,
            width=1200,
            height=700,
            margin=dict(l=40, r=40, t=80, b=120),
        )
        print("\nPlotting stacked bar chart with Plotly. Opening in browser or saving as HTML.")
        pio.write_html(fig, file=output_path, auto_open=True)
    except ImportError:
        print("\nPlease install plotly to see the plot:")
        print("pip install plotly")


def mesh_headings_network(MODE: str):
    features_path: Path = INTERIM_DATA_DIR / MODE / "features_with_ror.json"
    graph_path: Path = MODELS_DIR / MODE / "citation_model_with_communities.pkl"
    mesh_xml_path: Path = RAW_DATA_DIR / "desc2025.xml"
    output_csv: Path = PROCESSED_DATA_DIR / MODE / "edge_mesh_distances.csv"
    """
    Compute MeSH-based distances for all edges in the network and plot the graph with Plotly.

    For each edge (u, v) in citation_model_with_communities.pkl:
      - Fetch MeSH headings for the two endpoint publications from features_with_ror.json
      - Compute the mean pairwise MeSH tree distance between the two heading sets
      - Save results to CSV
      - Plot the graph with Plotly, coloring nodes by community and grouping edges by MeSH distance bins
    """
    from itertools import product
    import xml.etree.ElementTree as ET

    import igraph as ig

    def build_mesh_map(xml_path: Path) -> dict:
        tree = ET.parse(str(xml_path))
        root = tree.getroot()
        mesh_map_local = {}
        for record in root.findall(".//DescriptorRecord"):
            heading = record.findtext(".//DescriptorName/String")
            tree_numbers = [tn.text for tn in record.findall(".//TreeNumberList/TreeNumber")]
            if heading and tree_numbers:
                mesh_map_local[heading] = tree_numbers
        return mesh_map_local

    def mesh_tree_distance(tree_num_a: str, tree_num_b: str) -> int:
        a_parts = tree_num_a.split(".")
        b_parts = tree_num_b.split(".")
        common_len = 0
        for x, y in zip(a_parts, b_parts):
            if x == y:
                common_len += 1
            else:
                break
        return (len(a_parts) - common_len) + (len(b_parts) - common_len)

    def mesh_heading_mean_distance(headings_a, headings_b, mesh_map) -> float | None:
        distances = []
        for h1, h2 in product(headings_a, headings_b):
            trees_a = mesh_map.get(h1, [])
            trees_b = mesh_map.get(h2, [])
            for ta, tb in product(trees_a, trees_b):
                distances.append(mesh_tree_distance(ta, tb))
        if not distances:
            return None
        return float(np.mean(distances))

    logger.info(f"Reading features: {features_path}")
    df = pd.read_json(features_path)

    # Normalize identifiers and mesh_headings
    if "pmid" not in df.columns:
        raise ValueError("features_with_ror.json must contain a 'pmid' column.")
    df["pmid"] = df["pmid"].astype(str)

    def _normalize_headings(x):
        if isinstance(x, list):
            out = []
            for item in x:
                if isinstance(item, str):
                    out.append(item)
                elif isinstance(item, dict):
                    for key in ("heading", "DescriptorName", "name", "text"):
                        if key in item and isinstance(item[key], str):
                            out.append(item[key])
                            break
            return out
        return []

    if "mesh_headings" in df.columns:
        df["mesh_headings"] = df["mesh_headings"].apply(_normalize_headings)
    else:
        logger.warning("No 'mesh_headings' column found; all distances will be None.")
        df["mesh_headings"] = [[] for _ in range(len(df))]

    # Build identifier -> headings maps for several possible id columns
    id_columns = [c for c in ["pmid", "pmcid", "id", "name", "doi"] if c in df.columns]
    id_maps = {col: dict(zip(df[col].astype(str), df["mesh_headings"])) for col in id_columns}

    # Load graph
    logger.info(f"Loading graph: {graph_path}")
    with open(graph_path, "rb") as f:
        graph: ig.Graph = pickle.load(f)

    v_attrs = set(graph.vs.attributes())

    # Try to choose a vertex attribute that we can map to a df column
    vid_attr = None
    df_key_for_mapping = None

    # 1) Prefer 'pmid' if present on graph
    if "pmid" in v_attrs:
        vid_attr = "pmid"
        df_key_for_mapping = "pmid"
    else:
        # 2) Try to find best-overlap between any graph vertex attribute and available df id columns
        best_overlap = -1
        for g_attr in v_attrs:
            try:
                g_vals = set(map(str, graph.vs[g_attr]))
            except Exception:
                continue
            for df_col, mapping in id_maps.items():
                overlap = len(g_vals & set(mapping.keys()))
                if overlap > best_overlap:
                    best_overlap = overlap
                    vid_attr = g_attr
                    df_key_for_mapping = df_col

        # 3) If nothing overlaps, but sizes match, assume vertex index aligns to df order (use pmid)
        if (best_overlap <= 0) and (graph.vcount() == len(df)):
            vid_attr = "__index__"
            df_key_for_mapping = "pmid"
            index_to_id = list(df["pmid"].astype(str))
            logger.warning(
                "No matching vertex attribute found. Assuming vertex index aligns with df order (using pmid)."
            )

    if vid_attr is None or df_key_for_mapping is None:
        raise ValueError(
            "Could not resolve a vertex identifier attribute to map MeSH headings. "
            "Ensure the graph has a 'pmid' vertex attribute or another attribute that overlaps with a column in features_with_ror.json."
        )

    # Build MeSH map
    logger.info(f"Loading MeSH descriptors: {mesh_xml_path}")
    mesh_map = build_mesh_map(mesh_xml_path)

    # Compute distances per edge
    records = []
    logger.info(
        f"""Computing MeSH mean distances for all edges using graph attr '{
            vid_attr
        }' mapped to df column '{df_key_for_mapping}'..."""
    )

    id_to_headings = id_maps[df_key_for_mapping] if df_key_for_mapping in id_maps else {}
    for e in tqdm(graph.es, total=len(graph.es)):
        u_idx, v_idx = e.tuple
        if vid_attr == "__index__":
            u_id = index_to_id[u_idx]
            v_id = index_to_id[v_idx]
        else:
            u_id = str(graph.vs[u_idx][vid_attr])
            v_id = str(graph.vs[v_idx][vid_attr])

        h_u = id_to_headings.get(u_id, [])
        h_v = id_to_headings.get(v_id, [])

        mean_d = mesh_heading_mean_distance(h_u, h_v, mesh_map) if h_u and h_v else None
        records.append(
            {
                "source": u_id,
                "target": v_id,
                "mesh_mean_distance": mean_d,
                "has_headings_source": bool(h_u),
                "has_headings_target": bool(h_v),
            }
        )

    out_df = pd.DataFrame(records)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    logger.success(f"""Saved MeSH edge distances to {output_csv} (n_edges={len(out_df)})""")

    # Basic summary
    available = out_df["mesh_mean_distance"].dropna()
    if not available.empty:
        logger.info(
            f"""Distance stats | count={len(available)} mean={available.mean():.3f}"""
            f"""median={available.median():.3f} std={available.std():.3f}"""
        )
    else:
        logger.warning("No edges had both endpoints with MeSH headings; no distances computed.")

    # ---- Plot with Plotly ----
    logger.info("Building interactive network plot with Plotly...")

    # Node IDs helper
    def _node_id(idx: int) -> str:
        if vid_attr == "__index__":
            return index_to_id[idx]
        return str(graph.vs[idx][vid_attr])

    # Layout
    try:
        layout = graph.layout("fr")
    except Exception:
        layout = graph.layout("kk")
    coords = np.array(layout.coords if hasattr(layout, "coords") else layout)

    # Node attributes
    node_ids = [_node_id(i) for i in range(graph.vcount())]
    degrees = graph.degree()
    deg_min, deg_max = (min(degrees), max(degrees)) if degrees else (0, 1)
    sizes = [8 + 12 * ((d - deg_min) / (deg_max - deg_min + 1e-9)) for d in degrees]

    # Community/color
    comm_attr = (
        "community_label"
        if "community_label" in v_attrs
        else ("community" if "community" in v_attrs else None)
    )
    if comm_attr:
        comm_values = [str(graph.vs[i][comm_attr]) for i in range(graph.vcount())]
    else:
        comm_values = ["deg" if d > np.median(degrees) else "low" for d in degrees]

    unique_comms = sorted(set(comm_values))
    color_map = {
        c: (px.colors.qualitative.Set1 + px.colors.qualitative.Set3)[
            i % (len(px.colors.qualitative.Set1) + len(px.colors.qualitative.Set3))
        ]
        for i, c in enumerate(unique_comms)
    }
    node_colors = [color_map[c] for c in comm_values]

    # Hover: enrich with title/year if present
    df_titles = {}
    if "title" in df.columns:
        df_titles = dict(zip(df[df_key_for_mapping].astype(str), df["title"].astype(str)))
    df_years = {}
    if "year" in df.columns:
        try:
            df_years = dict(zip(df[df_key_for_mapping].astype(str), df["year"]))
        except Exception:
            df_years = {}

    hover_text = []
    for nid in node_ids:
        title = df_titles.get(nid, "")
        year = df_years.get(nid, "")
        comm = comm_values[node_ids.index(nid)]
        hover_text.append(f"""id: {nid}<br>community: {comm}<br>year: {year}<br>{title}""")

    # Edge distance bins (quantiles)
    edge_dist_map = {}
    for _, row in out_df.iterrows():
        key = frozenset((str(row["source"]), str(row["target"])))
        edge_dist_map[key] = row["mesh_mean_distance"]

    finite_dists = out_df["mesh_mean_distance"].dropna().astype(float)
    if len(finite_dists) >= 3:
        q1, q2 = np.quantile(finite_dists, [0.33, 0.66])

        def bin_label(d):
            if pd.isna(d):
                return "unknown"
            if d <= q1:
                return "low"
            if d <= q2:
                return "medium"
            return "high"

        bins = ["low", "medium", "high", "unknown"]
        bin_styles = {
            "low": {"color": "rgba(44,160,44,0.35)", "width": 1.5},
            "medium": {"color": "rgba(31,119,180,0.35)", "width": 1.5},
            "high": {"color": "rgba(214,39,40,0.35)", "width": 2.0},
            "unknown": {"color": "rgba(150,150,150,0.20)", "width": 1.0},
        }
    else:

        def bin_label(d):  # fallback: single bin
            return "unknown" if pd.isna(d) else "all"

        bins = ["all"]
        bin_styles = {"all": {"color": "rgba(150,150,150,0.25)", "width": 1.0}}

    # Build edge traces by bin
    edge_traces = []
    for b in bins:
        xs, ys, hover = [], [], []
        for e in graph.es:
            u, v = e.tuple
            uid, vid = _node_id(u), _node_id(v)
            d = edge_dist_map.get(frozenset((uid, vid)), np.nan)
            if bin_label(d) != b:
                continue
            xs += [coords[u, 0], coords[v, 0], None]
            ys += [coords[u, 1], coords[v, 1], None]
            hover.append(
                f"""{uid} — {vid}<br>MeSH mean distance: {d if not pd.isna(d) else "NA"}"""
            )
        if not xs:
            continue
        edge_traces.append(
            go.Scattergl(
                x=xs,
                y=ys,
                mode="lines",
                line=dict(color=bin_styles[b]["color"], width=bin_styles[b]["width"]),
                hoverinfo="skip",
                name=f"edges: {b}",
            )
        )

    # Node trace
    node_trace = go.Scattergl(
        x=coords[:, 0],
        y=coords[:, 1],
        mode="markers",
        text=hover_text,
        hoverinfo="text",
        marker=dict(size=sizes, color=node_colors, line=dict(width=0.5, color="#333")),
        name="nodes",
    )

    fig = go.Figure(data=edge_traces + [node_trace])
    # Legend helpers
    legend_items = [
        go.Scattergl(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(size=10, color=color_map[c]),
            name=f"community: {c}",
            hoverinfo="none",
            showlegend=True,
        )
        for c in unique_comms
    ]
    for li in legend_items:
        fig.add_trace(li)

    fig.update_layout(
        title=f"{MODE} citation graph with MeSH distance edges",
        showlegend=True,
        template="plotly_white",
        margin=dict(l=20, r=20, t=60, b=20),
        width=1200,
        height=900,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )

    out_html = FIGURES_DIR / MODE / "mesh_network_plot.html"
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_html)
    logger.success(f"Interactive network saved to {out_html}")


def plot_constraint(G: nx.Graph, MODE: str, constraints: dict):
    import pandas as pd

    output_path: Path = FIGURES_DIR / MODE / "constraints.html"

    # Save community labels
    nx.set_node_attributes(G, {n: constraints[n] for n in G.nodes()}, "constraint")

    logger.info("Constraints calculated")

    df = pd.DataFrame(
        {
            "pmid": nx.get_node_attributes(G, "title").keys(),
            "title": nx.get_node_attributes(G, "title").values(),
            "constraint": nx.get_node_attributes(G, "constraint").values(),
        }
    )

    df.to_csv(FIGURES_DIR / MODE / "constraint_table.csv")

    # Visualization
    pos = nx.spring_layout(G, seed=42, k=0.15 if G.number_of_nodes() < 1000 else None)
    x_coords = [pos[n][0] for n in G.nodes()]
    y_coords = [pos[n][1] for n in G.nodes()]

    node_titles = [data.get("title", f"Node {n}") for n, data in G.nodes(data=True)]

    # Edges
    edge_x, edge_y = [], []
    edge_shapes = []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_shapes.append(
            dict(
                type="line",
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                line=dict(color="#888", width=0.5),
                opacity=0.7,
                layer="below",
            )
        )
    node_trace = go.Scatter(
        x=x_coords,
        y=y_coords,
        mode="markers",
        hoverinfo="text",
        text=[
            f"Title: {title}<br>Constraint: {constraints[n]}"
            for (n, data), title in zip(G.nodes(data=True), node_titles)
        ],
        marker=dict(
            line=dict(width=2, color="white"),
        ),
    )
    node_constraints = []
    for n in constraints.keys():
        node_constraints.append(constraints[n])
    node_trace.marker.color = node_constraints
    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.5, color="#888"),
        hoverinfo="none",
        mode="lines",
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=(f"Interactive Graph Visualization (Colored by constraints)<br>"),
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[
                dict(
                    text="Hover over nodes to see titles",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.005,
                    y=-0.002,
                    xanchor="left",
                    yanchor="bottom",
                    font=dict(color="gray", size=12),
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            width=1800,
            height=1200,
            shapes=edge_shapes,
        ),
    )

    fig.write_html(output_path)

    png_output_path = output_path.with_suffix(".png")
    fig.write_image(png_output_path, width=1200, height=800, scale=5)
    logger.success(f"Successfully saved output images to path: {output_path}")


def plot_cartographic_density(MODE) -> None:
    import pandas as pd

    df = pd.read_json(INTERIM_DATA_DIR / MODE / "features_with_ror.json")
    print(df.head())
    output_path: Path = FIGURES_DIR / MODE / "geo_density.html"
    import plotly.express as px

    fig = px.density_map(
        df, lat="matched_lat", lon="matched_lon", radius=5, map_style="open-street-map", zoom=0
    )
    fig.update_layout(map_style="open-street-map", map_center_lon=0)
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.write_html(output_path)
    fig.write_image(output_path.with_suffix(".png"), scale=4)
    return None
