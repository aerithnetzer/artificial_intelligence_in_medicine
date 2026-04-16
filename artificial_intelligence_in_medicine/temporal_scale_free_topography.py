from collections import defaultdict
from pathlib import Path
import pickle

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import igraph as ig
from loguru import logger
import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import seaborn as sns
import typer

from artificial_intelligence_in_medicine.config import (
    FIGURES_DIR,
    INTERIM_DATA_DIR,
    MODELS_DIR,
)

app = typer.Typer()


def get_node_locations(features_path):
    """
    Extract node locations from the features JSON file.

    Returns
    -------
    dict
        Mapping of node identifier to (lat, lon) tuple
    """
    logger.info(f"Loading node locations from {features_path}")
    df = pd.read_json(features_path)

    # Create a mapping from PMID or node identifier to lat/lon
    location_map = {}

    for idx, row in df.iterrows():
        pmid = row.get("pmid") or row.get("id")
        lat = row.get("matched_lat")
        lon = row.get("matched_lon")

        if pmid and lat is not None and lon is not None:
            if not pd.isna(lat) and not pd.isna(lon):
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    location_map[str(pmid)] = (lat, lon)

    logger.info(f"Loaded {len(location_map)} node locations")
    return location_map


def calculate_preferential_attachment_scores(graph):
    """
    Calculate a preferential attachment score for each node.

    The score is based on how much a node's degree growth correlates
    with its existing degree over time. Nodes following preferential
    attachment will gain edges proportional to their current degree.

    Parameters
    ----------
    graph : ig.Graph
        Citation network graph with year attributes

    Returns
    -------
    dict
        Mapping of node index to preferential attachment score
    """
    logger.info("Calculating preferential attachment scores for each node...")

    # Get all unique years
    years = sorted(
        set(v["year"] for v in graph.vs if "year" in v.attributes() and v["year"] is not None)
    )

    if len(years) < 2:
        logger.error("Need at least 2 years of data")
        return {}

    # For each node, track its degree over time
    node_degree_history = defaultdict(list)

    for year in years:
        # Get subgraph up to this year
        nodes_up_to_year = [
            v.index
            for v in graph.vs
            if "year" in v.attributes() and v["year"] is not None and v["year"] <= year
        ]

        if len(nodes_up_to_year) < 10:
            continue

        subgraph = graph.subgraph(nodes_up_to_year)

        # Get degree for each node in this time slice
        for i, node_idx in enumerate(nodes_up_to_year):
            degree = subgraph.degree(i)
            node_degree_history[node_idx].append((year, degree))

    # Calculate preferential attachment score for each node
    pa_scores = {}

    for node_idx, history in node_degree_history.items():
        if len(history) < 3:  # Need at least 3 time points
            continue

        years_data = [h[0] for h in history]
        degrees = [h[1] for h in history]

        # Calculate degree growth between time steps
        degree_growth = [degrees[i + 1] - degrees[i] for i in range(len(degrees) - 1)]
        prev_degrees = degrees[:-1]

        # Filter out cases where degree doesn't change
        valid_indices = [
            i for i in range(len(degree_growth)) if prev_degrees[i] > 0 and degree_growth[i] >= 0
        ]

        if len(valid_indices) < 2:
            continue

        filtered_growth = [degree_growth[i] for i in valid_indices]
        filtered_prev = [prev_degrees[i] for i in valid_indices]

        # Preferential attachment score: correlation between previous degree and degree growth
        if len(set(filtered_prev)) > 1 and len(set(filtered_growth)) > 1:
            corr, pval = spearmanr(filtered_prev, filtered_growth)
            if not np.isnan(corr):
                pa_scores[node_idx] = corr

    logger.info(f"Calculated PA scores for {len(pa_scores)} nodes")
    return pa_scores


def create_kde_heatmap_by_year(graph, location_map, pa_scores, output_path):
    """
    Create animated KDE heatmap showing median preferential attachment scores by location.

    Parameters
    ----------
    graph : ig.Graph
        Citation network graph
    location_map : dict
        Mapping of node ID to (lat, lon)
    pa_scores : dict
        Mapping of node index to PA score
    output_path : Path
        Output path for the animation
    """
    logger.info("Creating animated KDE heatmap...")

    # Debug: Check what attributes nodes have
    sample_node = graph.vs[0]
    logger.debug(f"Sample node attributes: {sample_node.attributes()}")
    logger.debug(f"Sample location map keys: {list(location_map.keys())[:5]}")

    # Group nodes by year and collect their locations and PA scores
    year_data = defaultdict(lambda: {"lats": [], "lons": [], "scores": []})

    matched_count = 0
    unmatched_count = 0

    for v in graph.vs:
        if v.index not in pa_scores:
            continue

        # Try multiple ways to get node ID
        node_id = None
        if "_nx_name" in v.attributes():
            node_id = str(v["_nx_name"])
        elif "name" in v.attributes():
            node_id = str(v["name"])
        elif "pmid" in v.attributes():
            node_id = str(v["pmid"])
        elif "id" in v.attributes():
            node_id = str(v["id"])
        else:
            node_id = str(v.index)

        year = v["year"] if "year" in v.attributes() else None

        if year is None:
            continue

        if node_id not in location_map:
            unmatched_count += 1
            continue

        matched_count += 1
        lat, lon = location_map[node_id]
        score = pa_scores[v.index]

        year_data[year]["lats"].append(lat)
        year_data[year]["lons"].append(lon)
        year_data[year]["scores"].append(score)

    logger.info(f"Matched {matched_count} nodes with locations, {unmatched_count} unmatched")

    years = sorted(year_data.keys())

    if not years:
        logger.error("No data to plot")
        return

    logger.info(f"Plotting {len(years)} years")

    # Setup figure
    fig = plt.figure(figsize=(16, 9))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Add geographic features
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, alpha=0.5)
    ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.3)
    ax.add_feature(cfeature.OCEAN, facecolor="white")
    ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())

    # Color scale for PA scores (-1 to 1)
    vmin, vmax = -0.5, 0.5
    cmap = plt.get_cmap("RdBu_r")

    # Initialize colorbar
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation="vertical", fraction=0.02, pad=0.04)
    cbar.set_label("Preferential Attachment Score", rotation=270, labelpad=20)

    title = ax.text(0.5, 1.05, "", transform=ax.transAxes, ha="center", fontsize=16, weight="bold")

    contours = []

    def init():
        title.set_text("")
        return [title]

    def update(frame):
        # Clear previous contours
        for coll in contours:
            coll.remove()
        contours.clear()

        # Redraw geographic features
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, alpha=0.5)
        ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.3)
        ax.add_feature(cfeature.OCEAN, facecolor="white")

        year = years[frame]
        data = year_data[year]

        if len(data["lats"]) < 10:
            title.set_text(
                f"Preferential Attachment by Location - Year {int(year)} (insufficient data)"
            )
            return [title]

        # Create DataFrame for seaborn
        df = pd.DataFrame({"lon": data["lons"], "lat": data["lats"], "score": data["scores"]})

        # Normalize scores to [0, 1] for better visualization
        # Map [-0.5, 0.5] to color scale
        normalized_scores = np.clip(df["score"], vmin, vmax)

        # Plot KDE with color mapped to median PA score
        try:
            # Create multiple KDE plots colored by score bins
            score_bins = np.linspace(vmin, vmax, 5)

            for i in range(len(score_bins) - 1):
                bin_min, bin_max = score_bins[i], score_bins[i + 1]
                mask = (df["score"] >= bin_min) & (df["score"] < bin_max)

                if mask.sum() < 5:
                    continue

                bin_df = df[mask]
                color = cmap(norm((bin_min + bin_max) / 2))

                contour = sns.kdeplot(
                    data=bin_df,
                    x="lon",
                    y="lat",
                    fill=True,
                    color=color,
                    bw_adjust=0.5,
                    thresh=0.05,
                    levels=5,
                    alpha=0.4,
                    ax=ax,
                    transform=ccrs.PlateCarree(),
                )

                # Save contours for next frame cleanup
                for coll in ax.collections[len(contours) :]:
                    contours.append(coll)

        except Exception as e:
            logger.warning(f"Could not plot KDE for year {year}: {e}")

        title.set_text(
            f"Preferential Attachment by Location - Year {int(year)} (n={len(data['lats'])})"
        )

        return [title] + contours

    # Create animation
    anim = animation.FuncAnimation(
        fig, update, init_func=init, frames=len(years), interval=500, blit=False, repeat=True
    )

    # Save animation
    logger.info(f"Saving animation to {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix == ".gif":
        anim.save(output_path, writer="pillow", fps=2, dpi=150)
    else:
        anim.save(output_path, writer="ffmpeg", fps=2, dpi=150)

    plt.close()


@app.command()
def main(
    mode: str = typer.Option(
        "ARTIFICIAL_INTELLIGENCE", help="Mode: ARTIFICIAL_INTELLIGENCE, GENE_EXPRESSION, or NULL"
    ),
    graph_path: Path = typer.Option(None, help="Path to citation graph pickle"),
    features_path: Path = typer.Option(None, help="Path to features_with_ror.json"),
    output_path: Path = typer.Option(None, help="Output path for animation"),
):
    """
    Generate animated KDE heatmap showing preferential attachment scores by location.
    """
    graph_path = graph_path or MODELS_DIR / mode / "citation_model.pkl"
    features_path = features_path or INTERIM_DATA_DIR / mode / "features_with_ror.json"
    output_path = output_path or FIGURES_DIR / mode / "geographic_preferential_attachment_kde.gif"

    if mode == "ARTIFICIAL_INTELLIGENCE":
        min_degree = 25
    else:
        min_degree = 23

    # Load graph
    logger.info(f"Loading graph from {graph_path}")
    with open(graph_path, "rb") as f:
        graph = pickle.load(f)

    if isinstance(graph, nx.Graph):
        graph = ig.Graph.from_networkx(graph)
        vertices_to_keep = [v.index for v in graph.vs if v.degree() >= min_degree]
        graph = graph.induced_subgraph(vertices_to_keep)
        print(graph)

    logger.info(f"Graph has {graph.vcount()} nodes and {graph.ecount()} edges")

    # Load locations
    location_map = get_node_locations(features_path)

    # Calculate preferential attachment scores
    pa_scores = calculate_preferential_attachment_scores(graph)

    # Create animated KDE heatmap
    create_kde_heatmap_by_year(graph, location_map, pa_scores, output_path)

    logger.info("Done!")


if __name__ == "__main__":
    app()
