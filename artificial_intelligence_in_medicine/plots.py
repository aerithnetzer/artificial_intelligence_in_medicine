import math
from pathlib import Path
import pickle

from kneed import DataGenerator, KneeLocator
from loguru import logger
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.linalg import norm
from scipy.stats import norm as stats_norm
from tqdm import tqdm
import typer

from artificial_intelligence_in_medicine.config import (
    FIGURES_DIR,
    INTERIM_DATA_DIR,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)

app = typer.Typer()
DEFAULT_MODE = "ARTIFICIAL_INTELLIGENCE"
MODE = "GENE_EXPRESSION"
if MODE == "GENE_EXPRESSION":
    INFLECTION_POINT = 25
else:  # Default value for other modes
    INFLECTION_POINT = 23


# --- ADDED: Fisher r-to-z utilities ---
def fisher_r_to_z(r: float) -> float:
    # Clamp to avoid infinities
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


# --- END ADDED ---


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


def normalized_citations_over_time(
    input_path: Path = INTERIM_DATA_DIR / MODE / "features_with_ror.json",
):
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
        title=f"{MODE} Normalized Citations Over Time",
        width=1200,
        height=800,
    )
    fig.update_yaxes(tickformat=".2%")
    fig.write_image(
        FIGURES_DIR / MODE / "normalized_citations_over_time.png", scale=5, width=1200, height=800
    )
    logger.success("Displayed normalized citations over time plot.")


def normalized_articles_over_time(
    input_path: Path = INTERIM_DATA_DIR / MODE / "features_with_ror.json",
):
    """
    Generates a line plot showing the number of articles per year, normalized by the maximum number of articles in any year.
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
    max_count = year_counts.max()
    normalized_counts = year_counts / max_count

    # Create line plot
    fig = px.line(
        x=normalized_counts.index,
        y=normalized_counts.values,
        labels={"x": "Year", "y": "Normalized Article Count"},
        title="Normalized Articles Over Time",
    )
    fig.show()


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
        f"Pearson correlation between Jaccard distance and number of citations: {pearson_corr:.4f}"
    )

    # Calculate Spearman correlation
    spearman_corr = df_corr["jaccard_distance"].corr(df_corr["num_cited_by"], method="spearman")
    logger.info(
        f"Spearman correlation between Jaccard distance and number of citations: {spearman_corr:.4f}"
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
        yaxis=dict(range=[0, max(y) * 1.05]),  # Extend y-axis to slightly above max citations
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


@app.command("compare-correlations")
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
        f"AI r={ai_pearson:.4f} (n={n_A}) | GE r={ge_pearson:.4f} (n={n_B}) | diff={diff:.4f} "
        f"| z={z_stat:.3f} | p={p_val:.4g}"
    )
    if p_val < 0.05:
        logger.success(f"Difference significant (p={p_val:.4g})")
    else:
        logger.info(f"Difference not significant (p={p_val:.4g})")


# --- END ADDED CODE ---


def funding_agency(
    input_path: Path = INTERIM_DATA_DIR / MODE / "features_with_ror.json",
    output_path: Path = FIGURES_DIR / MODE / "top_5_agencies_by_year.html",
):
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
    print(f"\nGrant list info:")
    print(f"Non-null grant_list count: {df['grant_list'].notna().sum()}")
    print(f"Null grant_list count: {df['grant_list'].isna().sum()}")

    # Sample some grant_list values
    non_null_grants = df[df["grant_list"].notna()]
    if len(non_null_grants) > 0:
        print(f"\nSample grant_list values:")
        for i in range(min(3, len(non_null_grants))):
            print(f"Record {i}: {non_null_grants.iloc[i]['grant_list']}")
            print(f"Type: {type(non_null_grants.iloc[i]['grant_list'])}")

    # Check year column
    print(f"\nYear info:")
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
        print(f"Sample exploded grant_list values:")
        for i in range(min(3, len(grants_df))):
            print(f"Record {i}: {grants_df.iloc[i]['grant_list']}")
            print(f"Type: {type(grants_df.iloc[i]['grant_list'])}")

    # Extract agency from each grant dictionary
    grants_df["agency"] = grants_df["grant_list"].apply(
        lambda x: x if isinstance(x, str) else None
    )

    # Check extracted agencies
    print(f"\nAgency extraction:")
    print(f"Non-null agencies: {grants_df['agency'].notna().sum()}")
    print(f"Unique agencies: {grants_df['agency'].nunique()}")
    if grants_df["agency"].notna().sum() > 0:
        print(f"Sample agencies: {grants_df['agency'].dropna().head().tolist()}")

    # Clean and validate year data
    if "year" not in grants_df.columns:
        print("Error: 'year' column not found in the data.")
        return

    # Handle different year formats
    grants_df["year_clean"] = grants_df["year"].apply(
        lambda x: str(x).split("-")[0] if pd.notna(x) else None
    )
    grants_df["year_clean"] = pd.to_numeric(grants_df["year_clean"], errors="coerce")

    print(f"\nYear processing:")
    print(f"Valid years: {grants_df['year_clean'].notna().sum()}")
    if grants_df["year_clean"].notna().sum() > 0:
        print(f"Year range: {grants_df['year_clean'].min()} to {grants_df['year_clean'].max()}")

    # Use citation_count if available, otherwise calculate from cited_by
    if "citation_count" in grants_df.columns:
        grants_df["citations"] = grants_df["citation_count"].fillna(0)
    else:
        grants_df["citations"] = grants_df["cited_by"].apply(
            lambda x: len(x) if isinstance(x, list) else 0
        )

    print(f"\nCitation data:")
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

    print(f"\nAggregated data: {len(agency_citations_by_year)} agency-year combinations")

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


def horizontal_timeline(
    graph_path: Path = MODELS_DIR / MODE / "citation_model_with_communities.pkl",
    features_path: Path = INTERIM_DATA_DIR / MODE / "features_with_ror.json",
    output_path: Path = FIGURES_DIR / MODE,
):
    import igraph as ig

    """Generate horizontal timeline showing community presence over years."""
    with open(graph_path, "rb") as f:
        graph: ig.Graph = pickle.load(f)

    graph = graph.simplify()
    graph = graph.as_undirected(combine_edges="first")

    features_df = pd.read_json(features_path)
    features_df["pmid"] = features_df["pmid"].astype(str)

    if "year" not in graph.vs.attributes():
        raise ValueError("Graph vertices are missing the 'year' attribute.")

    data = {
        "year": [],
        "community": [],
    }
    for v in graph.vs:
        degree = graph.degree(v.index)
        community = (
            v["community_label"] if "community_label" in graph.vs.attributes() else v["community"]
        )
        if not (
            degree == 0
            and sum(
                1
                for vv in graph.vs
                if (
                    (
                        vv["community_label"]
                        if "community_label" in graph.vs.attributes()
                        else vv["community"]
                    )
                    == community
                )
            )
            == 1
        ):
            data["year"].append(v["year"])
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
        height=max(400, len(unique_communities) * 50),  # Scale height with number of communities
        bargap=0.2,
    )

    # Add horizontal lines to separate communities
    for i in range(len(unique_communities)):
        fig.add_hline(y=i, line_dash="dot", line_color="lightgray", opacity=0.5)

    output_path.mkdir(parents=True, exist_ok=True)
    plot_file = output_path / "community_timeline_horizontal.html"
    fig.write_html(plot_file)
    logger.success(f"Horizontal community timeline saved as '{plot_file}'")


@app.command()
def main(
    mode: str = typer.Option(
        DEFAULT_MODE,
        "--mode",
        help="Dataset MODE (e.g. ARTIFICIAL_INTELLIGENCE or GENE_EXPRESSION)",
    ),
    input_path: Path = typer.Option(None, help="Override features_with_ror.json path"),
    model_path: Path = typer.Option(None, help="Override citation_model.pkl path"),
    output_path: Path = typer.Option(None, help="Elbow curve output HTML path"),
):
    """
    Generate core plots (scatter, median jaccard over time, timeline, elbow curve).
    """
    # Resolve paths based on selected mode if not provided
    if input_path is None:
        input_path = INTERIM_DATA_DIR / mode / "features_with_ror.json"
    if model_path is None:
        model_path = MODELS_DIR / mode / "citation_model.pkl"
    if output_path is None:
        output_path = FIGURES_DIR / mode / "elbow_curve_with_inflection_point.html"

    # Use mode-specific scatter & line plots
    scatterplot_with_line_of_best_fit(
        input_path=PROCESSED_DATA_DIR / mode / "interdisciplinary_nodes.csv",
        mode=mode,
    )
    horizontal_timeline(
        graph_path=MODELS_DIR / mode / "citation_model_with_communities.pkl",
        features_path=INTERIM_DATA_DIR / mode / "features_with_ror.json",
        output_path=FIGURES_DIR / mode,
    )

    # Existing elbow logic
    with open(model_path, "rb") as f:
        G = pickle.load(f)

    df = pd.read_json(input_path)
    logger.info(f"Loaded {len(df)} rows from {input_path}")
    df = df.dropna(subset=["year"])
    df["year"] = pd.to_datetime(df["year"].astype(int), format="%Y")
    year_counts_df = df["year"].value_counts().sort_index().reset_index()
    year_counts_df.columns = ["year", "count"]

    write_path: Path = FIGURES_DIR / mode / "rows_per_year.html"
    write_path.parent.mkdir(parents=True, exist_ok=True)
    fig = px.line(
        year_counts_df,
        x="year",
        y="count",
        markers=True,
        title=f"Number of Rows per Year {mode}",
        labels={"year": "Year", "count": "Row Count"},
    )
    fig.write_html(write_path)

    degrees_raw = [val for (_, val) in G.degree()]
    processed_degrees = []
    for item in degrees_raw:
        if (
            isinstance(item, (list, tuple))
            and len(item) == 1
            and isinstance(item[0], (int, float))
        ):
            processed_degrees.append(int(item[0]))
        elif isinstance(item, (int, float)):
            processed_degrees.append(int(item))
    if not processed_degrees:
        logger.warning("No degree data to plot.")
        return

    sorted_degrees_values = sorted(processed_degrees, reverse=True)
    p1 = np.array([0, sorted_degrees_values[0]])
    p_last_idx = len(sorted_degrees_values) - 1
    p_last = np.array([p_last_idx, sorted_degrees_values[p_last_idx]])

    distances = []
    for i, deg in enumerate(sorted_degrees_values):
        pi = np.array([i, deg])
        dist = (
            0
            if np.all(p_last == p1)
            else np.abs(np.cross(p_last - p1, p1 - pi)) / np.linalg.norm(p_last - p1)
        )
        distances.append(dist)

    if distances:
        elbow_index = int(np.argmax(distances))
        inflection_degree_threshold = sorted_degrees_values[elbow_index]
        fig2 = go.Figure()
        fig2.add_trace(
            go.Scatter(
                x=list(range(len(sorted_degrees_values))),
                y=sorted_degrees_values,
                mode="lines+markers",
                name="Degree Distribution",
            )
        )
        fig2.add_trace(
            go.Scatter(
                x=[p1[0], p_last[0]],
                y=[p1[1], p_last[1]],
                mode="lines",
                name="Line endpoints",
                line=dict(dash="dash", color="red"),
            )
        )
        fig2.add_trace(
            go.Scatter(
                x=[elbow_index],
                y=[inflection_degree_threshold],
                mode="markers",
                name="Elbow",
                marker=dict(color="red", size=10, symbol="x"),
            )
        )
        fig2.update_layout(
            title=f"Elbow Curve of Node Degrees ({mode})",
            xaxis_title="Node Rank (Sorted by Degree)",
            yaxis_title="Degree",
            height=900,
            width=900,
        )
        fig2.write_html(output_path)
        fig2.write_image(Path(str(output_path)).with_suffix(".png"))
        logger.success(f"Elbow curve saved to {output_path}")


def make_scatterplot_visualization():
    # Make a scatterplot where the x axis is year and node radius is citation count
    return None


if __name__ == "__main__":
    app()
    # normalized_articles_over_time()
    # normalized_citations_over_time()
