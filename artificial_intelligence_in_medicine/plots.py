from pathlib import Path
import pickle

import igraph as ig
from loguru import logger
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import norm as stats_norm
from tqdm import tqdm
import typer
from artificial_intelligence_in_medicine._plots_helpers import (
    horizontal_timeline,
    normalized_articles_over_time,
    normalized_citations_over_time,
    scatterplot_with_line_of_best_fit,
)
from artificial_intelligence_in_medicine.config import (
    FIGURES_DIR,
    INTERIM_DATA_DIR,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)

app = typer.Typer()
DEFAULT_MODE = "ARTIFICIAL_INTELLIGENCE"
MODE = "GENE_EXPRESSION"  # or "GENE_EXPRESSION"
if MODE == "GENE_EXPRESSION":
    INFLECTION_POINT = 25
else:  # Default value for other modes
    INFLECTION_POINT = 23


def run_plots(
    mode="GENE_EXPRESSION",
    input_path: Path = typer.Option(None, help="Override features_with_ror.json path"),
    model_path: Path = typer.Option(None, help="Override citation_model.pkl path"),
    output_path: Path = typer.Option(None, help="Elbow curve output HTML path"),
):
    """
    Generate core plots (scatter, median jaccard over time, timeline, elbow curve).
    """
    # mesh_headings_network()
    # Resolve paths based on selected mode if not provided
    input_path = INTERIM_DATA_DIR / mode / "features_with_ror.json"
    model_path = MODELS_DIR / mode / "citation_model.pkl"
    output_path = FIGURES_DIR / mode / "elbow_curve_with_inflection_point.html"
    normalized_citations_over_time()
    # Use mode-specific scatter & line plots
    scatterplot_with_line_of_best_fit(
        input_path=(PROCESSED_DATA_DIR / mode / "interdisciplinary_nodes.csv"),
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
        title=f"Number of Rows per Year {mode} (CV={year_counts_df['count'].std() / year_counts_df['count'].mean():.2f})",
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
        num_vertices_at_elbow = np.sum(
            np.array(sorted_degrees_values) == inflection_degree_threshold
        )
        print(
            f"Number of vertices with degree = {inflection_degree_threshold}: {num_vertices_at_elbow}"
        )
        fig2.write_html(output_path)
        fig2.write_image(Path(str(output_path)).with_suffix(".png"))
        logger.success(f"Elbow curve saved to {output_path}")


if __name__ == "__main__":
    run_plots()
