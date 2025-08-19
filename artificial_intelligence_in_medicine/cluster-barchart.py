from pathlib import Path
import pickle

import igraph as ig
import pandas as pd
import plotly.express as px
import typer

from artificial_intelligence_in_medicine.config import (
    FIGURES_DIR,
    INTERIM_DATA_DIR,
    MODELS_DIR,
)

app = typer.Typer()
MODE = "GENE_EXPRESSION"


@app.command()
def vertical_barchart(
    graph_path: Path = MODELS_DIR / MODE / "citation_model_with_communities.pkl",
    features_path: Path = INTERIM_DATA_DIR / MODE / "features_with_ror.json",
    output_path: Path = FIGURES_DIR / MODE,
):
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
    community_counts_by_year = community_counts_by_year.sort_values("year")

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

    output_path.mkdir(parents=True, exist_ok=True)
    plot_file = output_path / "community_evolution_by_year.html"
    fig.write_html(plot_file)
    print(f"Community evolution plot saved as '{plot_file}'")


@app.command()
def horizontal_timeline(
    graph_path: Path = MODELS_DIR / MODE / "citation_model_with_communities.pkl",
    features_path: Path = INTERIM_DATA_DIR / MODE / "features_with_ror.json",
    output_path: Path = FIGURES_DIR / MODE,
):
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
    fig.write_image(plot_file.with_suffix(".png"), width=1200, height=800, scale=5)
    print(f"Horizontal community timeline saved as '{plot_file}'")


if __name__ == "__main__":
    app()
