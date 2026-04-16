"""
Core plot generation script.

Generates elbow curve, rows-per-year, scatter plots, and timeline visualizations.
"""

from pathlib import Path
import pickle

from loguru import logger
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import typer

from artificial_intelligence_in_medicine.config import (
    FIGURES_DIR,
    INTERIM_DATA_DIR,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
)
from artificial_intelligence_in_medicine.visualizations import (
    horizontal_timeline,
    normalized_citations_over_time,
    scatterplot_with_line_of_best_fit,
)

app = typer.Typer()


def run_plots(mode: str = "GENE_EXPRESSION"):
    """
    Generate core plots (scatter, normalized citations, timeline, elbow curve).
    """
    input_path = INTERIM_DATA_DIR / mode / "features_with_ror.json"
    model_path = MODELS_DIR / mode / "citation_model.pkl"
    output_path = FIGURES_DIR / mode / "elbow_curve_with_inflection_point.html"

    # Normalized citations
    normalized_citations_over_time(mode)

    # Scatter plot
    scatter_path = PROCESSED_DATA_DIR / mode / "interdisciplinary_nodes.csv"
    if scatter_path.exists():
        scatterplot_with_line_of_best_fit(input_path=scatter_path, mode=mode)

    # Timeline
    communities_path = MODELS_DIR / mode / "citation_model_with_communities.pkl"
    if communities_path.exists():
        horizontal_timeline(
            graph_path=communities_path,
            features_path=input_path,
            output_path=FIGURES_DIR / mode,
        )

    # Elbow curve
    with open(model_path, "rb") as f:
        G = pickle.load(f)

    df = pd.read_json(input_path)
    logger.info(f"Loaded {len(df)} rows from {input_path}")
    df = df.dropna(subset=["year"])
    df["year"] = pd.to_datetime(df["year"].astype(int), format="%Y")
    year_counts_df = df["year"].value_counts().sort_index().reset_index()
    year_counts_df.columns = ["year", "count"]

    write_path = FIGURES_DIR / mode / "rows_per_year.html"
    write_path.parent.mkdir(parents=True, exist_ok=True)
    fig = px.line(
        year_counts_df,
        x="year",
        y="count",
        markers=True,
        title=f"Number of Rows per Year {mode} "
        f"(CV={year_counts_df['count'].std() / year_counts_df['count'].mean():.2f})",
        labels={"year": "Year", "count": "Row Count"},
    )
    fig.write_html(write_path)

    # Elbow curve computation
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

    sorted_degrees = sorted(processed_degrees, reverse=True)
    p1 = np.array([0, sorted_degrees[0]])
    p_last_idx = len(sorted_degrees) - 1
    p_last = np.array([p_last_idx, sorted_degrees[p_last_idx]])

    distances = []
    for i, deg in enumerate(sorted_degrees):
        pi = np.array([i, deg])
        dist = (
            0
            if np.all(p_last == p1)
            else np.abs(np.cross(p_last - p1, p1 - pi)) / np.linalg.norm(p_last - p1)
        )
        distances.append(dist)

    if distances:
        elbow_index = int(np.argmax(distances))
        inflection_threshold = sorted_degrees[elbow_index]

        fig2 = go.Figure()
        fig2.add_trace(
            go.Scatter(
                x=list(range(len(sorted_degrees))),
                y=sorted_degrees,
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
                y=[inflection_threshold],
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


if __name__ == "__main__":
    run_plots()
