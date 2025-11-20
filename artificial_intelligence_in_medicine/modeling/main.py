from pathlib import Path
import pickle
import json
from scipy.stats import ks_2samp, ttest_ind
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from loguru import logger
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import powerlaw
import seaborn as sns
from tqdm import tqdm
import typer

from artificial_intelligence_in_medicine.config import FIGURES_DIR, INTERIM_DATA_DIR, MODELS_DIR

app = typer.Typer()

MODE = "null"
features_path: Path = INTERIM_DATA_DIR / MODE / "features_with_ror.json"
model_path: Path = MODELS_DIR / MODE / "citation_model.pkl"


def calculate_scale_free_inflection(g: nx.Graph):
    degrees = np.array([g.degree[n] for n in g.nodes])
    degrees = degrees[degrees > 0]  # Exclude zero-degree nodes
    fit = powerlaw.Fit(degrees)
    k_min = fit.xmin  # This is the inflection point for scale-free behavior
    return k_min


def get_lat_lon(row):
    """
    Extracts latitude and longitude from the matched ROR data.
    """
    lat = row.get("matched_lat")
    lon = row.get("matched_lon")

    if lat is not None and lon is not None and not pd.isna(lat) and not pd.isna(lon):
        return lat, lon
    return None, None


def plot_degree_distribution_with_inflection(g: nx.Graph, inflection_degree):
    degrees = np.array([g.degree[n] for n in g.nodes])
    degrees_sorted = np.sort(degrees)
    x = np.arange(len(degrees_sorted))
    colors = ["red" if deg == inflection_degree else "blue" for deg in degrees_sorted]

    fig = go.Figure(
        data=go.Scatter(x=x, y=degrees_sorted, mode="markers", marker=dict(color=colors, size=8))
    )
    fig.update_layout(
        title="Degree Distribution with Inflection Point",
        xaxis_title="Sorted Node Index",
        yaxis_title="Degree",
    )
    fig.show()


def plot_lat_lon_scatter(
    df,
    output_path,
    lat_col="matched_lat",
    lon_col="matched_lon",
    title="Lat/Lon Scatterplot",
):
    """
    Plots latitude and longitude as a scatterplot on a world map.
    """
    plt.figure(figsize=(16, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor="lightgray")
    ax.add_feature(cfeature.OCEAN, facecolor="white")
    ax.scatter(
        df[lon_col],
        df[lat_col],
        color="red",
        s=10,
        alpha=0.6,
        transform=ccrs.PlateCarree(),
        label="Locations",
    )
    ax.set_title(title, fontsize=16)
    ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)


def geo_heatmap(
    input_path: Path = INTERIM_DATA_DIR / MODE / "features_with_ror.json",
    output_path: Path = FIGURES_DIR / MODE / "global_kde_heatmap.png",
    inflection_point: int = 10,
):
    """
    Generates a KDE density plot of author affiliations (only articles above inflection point).
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import numpy as np
    import pandas as pd

    logger.info("Processing dataset...")
    df = pd.read_json(input_path)
    df["matched_lat"], df["matched_lon"] = zip(*df.apply(get_lat_lon, axis=1))
    df = df.explode(["matched_lat", "matched_lon"])
    df = df.dropna(subset=["matched_lat", "matched_lon"])
    df = df[(df["matched_lat"].between(-90, 90)) & (df["matched_lon"].between(-180, 180))]
    # Filter articles above the provided inflection point
    if "cited_by" in df.columns:
        degrees = df["cited_by"].apply(lambda x: len(x) if isinstance(x, (list, set)) else 0)
        df = df[degrees > inflection_point]
    else:
        logger.warning("No 'cited_by' column found, using all articles.")

    # Plot KDE heatmap
    plt.figure(figsize=(16, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor="lightgray")
    ax.add_feature(cfeature.OCEAN, facecolor="white")

    if not df.empty:
        sns.kdeplot(
            x=df["matched_lon"],
            y=df["matched_lat"],
            fill=True,
            cmap="viridis",
            bw_adjust=0.5,
            thresh=0.05,
            levels=20,
            alpha=0.7,
            ax=ax,
            transform=ccrs.PlateCarree(),
        )
    else:
        logger.warning("No articles above inflection point to plot.")

    ax.set_title("Global KDE Heatmap of Author Locations (Above Inflection Point)", fontsize=16)
    ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
    plt.tight_layout()
    logger.info("Saving figure to {}", output_path)
    plt.savefig(output_path, dpi=300)


def calculate_communities(
    graph_path: Path = MODELS_DIR / MODE / "citation_model.pkl",
    output_path: Path = FIGURES_DIR / MODE / "community_detection.html",
):
    import numpy as np
    import plotly.graph_objects as go
    from sklearn.feature_extraction.text import TfidfVectorizer

    with open(graph_path, "rb") as f:
        G: nx.Graph = pickle.load(f)

    # Only keep the largest connected component
    if G.is_directed():
        largest_cc = max(nx.weakly_connected_components(G), key=len)
    else:
        largest_cc = max(nx.connected_components(G), key=len)
    G = G.subgraph(largest_cc).copy()
    original_node_count = len(G.nodes)

    # Calculate inflection point dynamically
    inflection_point = calculate_scale_free_inflection(G)
    if inflection_point is None:
        print("Could not determine inflection point, using minimum degree = 1")
        inflection_point = 1

    # Filter nodes by degree
    high_degree_nodes = [n for n in G.nodes if G.degree[n] > inflection_point]
    G = G.subgraph(high_degree_nodes).copy()

    # Log after filtering
    high_degree_count = len(G.nodes)
    percent_high_degree = (
        100 * high_degree_count / original_node_count if original_node_count > 0 else 0
    )
    print(
        f"Inflection point degree: {inflection_point}. "
        f"Nodes above inflection: {high_degree_count} "
        f"({percent_high_degree:.2f}% of total nodes: {original_node_count})"
    )

    # Remove isolated nodes
    isolated = [n for n in G.nodes if G.degree[n] == 0]
    if isolated:
        G.remove_nodes_from(isolated)

    # Community detection (Louvain)
    communities = []
    if len(G.nodes) > 0 and len(G.edges) > 0:
        try:
            communities = nx.community.louvain_communities(G)
        except Exception as e:
            print(f"Error during community detection: {e}")
            communities = [set([n]) for n in G.nodes]
    elif len(G.nodes) > 0:
        communities = [set([n]) for n in G.nodes]

    # Assign community membership
    node_community = {}
    for i, comm in enumerate(communities):
        for n in comm:
            node_community[n] = i
    nx.set_node_attributes(G, node_community, "community")

    # Generate community names
    community_names = {}
    titles = nx.get_node_attributes(G, "title")
    if titles:
        from collections import defaultdict

        community_titles = defaultdict(list)
        for n, title in titles.items():
            cid = node_community[n]
            community_titles[cid].append(title)
        for cid, titles_list in community_titles.items():
            if titles_list:
                try:
                    vectorizer = TfidfVectorizer(stop_words="english", max_features=5)
                    vectorizer.fit_transform(titles_list)
                    top_terms = vectorizer.get_feature_names_out()
                    community_names[cid] = ", ".join(top_terms)
                except ValueError:
                    community_names[cid] = f"Community {cid}"
            else:
                community_names[cid] = f"Community {cid}"
    else:
        community_names = {i: f"Community {i}" for i in range(len(communities))}

    # Save community labels to node attributes
    for n in G.nodes:
        cid = node_community[n]
        G.nodes[n]["community_label"] = community_names.get(cid, f"Community {cid}")

    # Save graph with communities
    output_graph_path = graph_path.with_name(f"{graph_path.stem}_with_communities.pkl")
    with open(output_graph_path, "wb") as f:
        pickle.dump(G, f)

    # Save community name mapping
    with open(output_path.with_name("community_labels.pkl"), "wb") as f:
        pickle.dump(community_names, f)

    print("Graph and labels saved. Visualizing...")

    # Visualization
    try:
        pos = nx.spring_layout(G, seed=42)
        node_titles = [G.nodes[n].get("title", f"Node {n}") for n in G.nodes]
        colors_discrete = px.colors.qualitative.Set1 + px.colors.qualitative.Set3
        node_colors = [colors_discrete[node_community[n] % len(colors_discrete)] for n in G.nodes]
        citation_counts = [len(G.nodes[n].get("cited_by", [])) for n in G.nodes]
        node_sizes = [10 + 2 * np.log(c + 1) for c in citation_counts]

        edge_x, edge_y = [], []
        edge_shapes = []
        for u, v in G.edges:
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
        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=0.5, color="#888"),
            hoverinfo="none",
            mode="lines",
        )

        node_x = [pos[n][0] for n in G.nodes]
        node_y = [pos[n][1] for n in G.nodes]
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers",
            hoverinfo="text",
            text=[
                f"Title: {title}<br>Community: {G.nodes[n]['community_label']}<br>Cited by: {len(G.nodes[n].get('cited_by', []))}"
                for n, title in zip(G.nodes, node_titles)
            ],
            marker=dict(size=node_sizes, color=node_colors, line=dict(width=2, color="white")),
        )

        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=f"Interactive Directed Graph Visualization<br>{len(communities)} communities",
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

        # Save static PNG (requires `pip install -U kaleido`)
        png_output_path = output_path.with_suffix(".png")
        fig.write_image(png_output_path, width=1200, height=800, scale=5)

    except Exception as e:
        print(f"Visualization error: {e}")


def init_graph(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = INTERIM_DATA_DIR / MODE / "features_with_ror.json",
    model_path: Path = MODELS_DIR / MODE / "citation_model.pkl",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Initializing graph for citation modeling.")
    G = nx.DiGraph()
    df = pd.read_json(features_path)
    # Check available columns
    logger.info(f"Available columns: {df.columns.tolist()}")

    # Use consistent column name for PMID
    pmid_column = "pmid" if "pmid" in df.columns else "_id"
    if pmid_column not in df.columns:
        raise KeyError(
            f"Neither 'pmid' nor '_id' found in DataFrame columns: {df.columns.tolist()}"
        )
    logger.info(f"Using '{pmid_column}' as PMID column")

    print(str(df["cited_by"]))
    logger.info("Now adding nodes to the graph.")
    for _, row in tqdm(df.iterrows(), desc="Adding nodes"):
        G.add_node(
            str(row[pmid_column]),
            title=row["title"],
            cited_by=row["cited_by"],
            mesh_headings=row["mesh_headings"],
            year=row["year"],
        )

    logger.info("Now adding edges to the graph.")
    pmid_set = set(df[pmid_column].astype(str))
    for _, row in tqdm(df.iterrows(), desc="Adding edges"):
        cited_by_list = row["cited_by"]
        if hasattr(cited_by_list, "__iter__") and not isinstance(cited_by_list, str):
            source_pmid = str(row[pmid_column])
            for cited_pmid in cited_by_list:
                cited_pmid_str = str(cited_pmid)
                if cited_pmid_str in pmid_set:
                    G.add_edge(source_pmid, cited_pmid_str)
        else:
            continue

    logger.info(
        "Saved graph. Number of nodes: {}, number of edges: {}".format(
            G.number_of_nodes(), G.number_of_edges()
        )
    )
    with open(model_path, "wb") as f:
        pickle.dump(G, f)
    return G


def get_or_init_graph(features_path, model_path):
    if Path(model_path).exists():
        with open(model_path, "rb") as f:
            G = pickle.load(f)
    else:
        G = init_graph(features_path=features_path, model_path=model_path)
    return G


def compare_modes_geo_and_topology(
    geo_paths: dict,  # {"gene_expression": Path, "artificial_intelligence": Path, "null": Path}
    graph_paths: dict,  # {"gene_expression": Path, "artificial_intelligence": Path, "null": Path}
):
    results = {}

    # Load data
    dfs = {mode: pd.read_json(path) for mode, path in geo_paths.items()}
    graphs = {mode: pickle.load(open(path, "rb")) for mode, path in graph_paths.items()}
    print("Data and graphs loaded.")
    print({mode: (len(df), len(graphs[mode].nodes)) for mode, df in dfs.items()})
    # Geographical distribution comparison (lat/lon)
    for mode in ["gene_expression", "artificial_intelligence"]:
        lat_null = dfs["null"]["matched_lat"].dropna()
        lon_null = dfs["null"]["matched_lon"].dropna()
        lat_mode = dfs[mode]["matched_lat"].dropna()
        lon_mode = dfs[mode]["matched_lon"].dropna()

        lat_stat, lat_p = ks_2samp(lat_mode, lat_null)
        lon_stat, lon_p = ks_2samp(lon_mode, lon_null)
        results[f"{mode}_vs_null_geo"] = {
            "lat_ks_stat": lat_stat,
            "lat_ks_p": lat_p,
            "lat_reject_null": lat_p < 0.05,
            "lon_ks_stat": lon_stat,
            "lon_ks_p": lon_p,
            "lon_reject_null": lon_p < 0.05,
        }

    # Network topology comparison (degree distribution and more)
    for mode in ["gene_expression", "artificial_intelligence"]:
        print(f"Comparing {mode} to null topology...")
        G_mode = graphs[mode]
        G_null = graphs["null"]

        deg_null = np.array([d for n, d in G_null.degree()])
        deg_mode = np.array([d for n, d in G_mode.degree()])
        deg_stat, deg_p = ks_2samp(deg_mode, deg_null)
        deg_reject_null = deg_p < 0.05

        # Clustering coefficient
        clustering_mode = nx.average_clustering(G_mode)
        clustering_null = nx.average_clustering(G_null)
        clustering_stat, clustering_p = ttest_ind(
            list(nx.clustering(G_mode).values()),
            list(nx.clustering(G_null).values()),
            equal_var=False,
        )
        clustering_reject_null = clustering_p < 0.05

        # Modularity (using Louvain communities)
        try:
            comm_mode = nx.community.louvain_communities(G_mode)
            comm_null = nx.community.louvain_communities(G_null)
            modularity_mode = nx.community.modularity(G_mode, comm_mode)
            modularity_null = nx.community.modularity(G_null, comm_null)
            modularity_diff = modularity_mode - modularity_null
            modularity_reject_null = abs(modularity_diff) > 0.05
        except Exception:
            comm_mode = comm_null = None
            modularity_mode = modularity_null = modularity_diff = modularity_reject_null = None

        # Average shortest path length (only for connected graphs)
        try:
            if nx.is_connected(G_mode):
                avg_path_mode = nx.average_shortest_path_length(G_mode)
            else:
                largest_cc_mode = max(nx.connected_components(G_mode), key=len)
                sub_mode = G_mode.subgraph(largest_cc_mode)
                avg_path_mode = nx.average_shortest_path_length(sub_mode)
        except Exception:
            avg_path_mode = None

        try:
            if nx.is_connected(G_null):
                avg_path_null = nx.average_shortest_path_length(G_null)
            else:
                largest_cc_null = max(nx.connected_components(G_null), key=len)
                sub_null = G_null.subgraph(largest_cc_null)
                avg_path_null = nx.average_shortest_path_length(sub_null)
        except Exception:
            avg_path_null = None

        if avg_path_mode is not None and avg_path_null is not None:
            path_diff = avg_path_mode - avg_path_null
            avg_shortest_path_reject_null = abs(path_diff) > 0.1
        else:
            path_diff = None
            avg_shortest_path_reject_null = False

        results[f"{mode}_vs_null_topology"] = {
            "degree_ks_stat": deg_stat,
            "degree_ks_p": deg_p,
            "degree_reject_null": deg_reject_null,
            "degree_mean_mode": deg_mode.mean(),
            "degree_mean_null": deg_null.mean(),
            "clustering_mean_mode": clustering_mode,
            "clustering_mean_null": clustering_null,
            "clustering_t_stat": clustering_stat,
            "clustering_t_p": clustering_p,
            "clustering_reject_null": clustering_reject_null,
            "modularity_mode": modularity_mode,
            "modularity_null": modularity_null,
            "modularity_diff": modularity_diff,
            "modularity_reject_null": modularity_reject_null,
            "avg_shortest_path_mode": avg_path_mode,
            "avg_shortest_path_null": avg_path_null,
            "avg_shortest_path_diff": path_diff,
            "avg_shortest_path_reject_null": avg_shortest_path_reject_null,
        }
        print(results[f"{mode}_vs_null_topology"])
        print("Now performing cluster-level tests...")
        # Cluster-level hypothesis testing
        cluster_results = []
        if modularity_mode is not None and comm_mode is not None:
            # For each community in mode, compare its metrics to null
            for i, community in tqdm(
                enumerate(comm_mode), desc="Cluster tests", total=len(comm_mode)
            ):
                community_nodes = list(community)
                # Degree distribution
                degrees_community = np.array([G_mode.degree[n] for n in community_nodes])
                degrees_null = np.array(
                    [G_null.degree[n] for n in G_null.nodes if n in community_nodes]
                )
                if len(degrees_community) > 0 and len(degrees_null) > 0:
                    deg_stat, deg_p = ks_2samp(degrees_community, degrees_null)
                    deg_reject_null = deg_p < 0.05
                else:
                    deg_stat, deg_p, deg_reject_null = None, None, False

                # Clustering
                clustering_community = [nx.clustering(G_mode, n) for n in community_nodes]
                clustering_null = [
                    nx.clustering(G_null, n) for n in community_nodes if n in G_null.nodes
                ]
                if len(clustering_community) > 1 and len(clustering_null) > 1:
                    cluster_stat, cluster_p = ttest_ind(
                        clustering_community, clustering_null, equal_var=False
                    )
                    cluster_reject_null = cluster_p < 0.05
                else:
                    cluster_stat, cluster_p, cluster_reject_null = None, None, False

                # Shortest path length (within community)
                try:
                    sub_mode = G_mode.subgraph(community_nodes)
                    if nx.is_connected(sub_mode):
                        avg_path_mode = nx.average_shortest_path_length(sub_mode)
                    else:
                        largest_cc_mode = max(nx.connected_components(sub_mode), key=len)
                        sub_mode_cc = sub_mode.subgraph(largest_cc_mode)
                        avg_path_mode = nx.average_shortest_path_length(sub_mode_cc)
                except Exception:
                    avg_path_mode = None

                try:
                    sub_null = G_null.subgraph([n for n in community_nodes if n in G_null.nodes])
                    if len(sub_null.nodes) > 1:
                        if nx.is_connected(sub_null):
                            avg_path_null = nx.average_shortest_path_length(sub_null)
                        else:
                            largest_cc_null = max(nx.connected_components(sub_null), key=len)
                            sub_null_cc = sub_null.subgraph(largest_cc_null)
                            avg_path_null = nx.average_shortest_path_length(sub_null_cc)
                    else:
                        avg_path_null = None
                except Exception:
                    avg_path_null = None

                # Test for significant difference in average shortest path length
                if avg_path_mode is not None and avg_path_null is not None:
                    path_diff = avg_path_mode - avg_path_null
                    path_reject_null = abs(path_diff) > 0.1
                else:
                    path_diff, path_reject_null = None, False

                cluster_results.append(
                    {
                        "community_index": i,
                        "community_size": len(community_nodes),
                        "degree_ks_stat": deg_stat,
                        "degree_ks_p": deg_p,
                        "degree_reject_null": deg_reject_null,
                        "clustering_t_stat": cluster_stat,
                        "clustering_t_p": cluster_p,
                        "clustering_reject_null": cluster_reject_null,
                        "avg_shortest_path_mode": avg_path_mode,
                        "avg_shortest_path_null": avg_path_null,
                        "avg_shortest_path_diff": path_diff,
                        "avg_shortest_path_reject_null": path_reject_null,
                    }
                )
        results[f"{mode}_vs_null_cluster_tests"] = cluster_results

    return results


def save_results_to_json(results, output_path):
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    G = get_or_init_graph(features_path, model_path)
    inflection = calculate_scale_free_inflection(G)
    # plot_degree_distribution_with_inflection(G, inflection)
    # calculate_communities()
    geo_paths = {
        "gene_expression": Path(INTERIM_DATA_DIR / "gene_expression" / "features_with_ror.json"),
        "artificial_intelligence": Path(
            INTERIM_DATA_DIR / "artificial_intelligence" / "features_with_ror.json"
        ),
        "null": Path(INTERIM_DATA_DIR / "null" / "features_with_ror.json"),
    }
    graph_paths = {
        "gene_expression": Path(MODELS_DIR / "gene_expression" / "citation_model.pkl"),
        "artificial_intelligence": Path(
            MODELS_DIR / "artificial_intelligence" / "citation_model.pkl"
        ),
        "null": Path(MODELS_DIR / "null" / "citation_model.pkl"),
    }
    results = compare_modes_geo_and_topology(geo_paths, graph_paths)
    results = compare_modes_geo_and_topology(geo_paths, graph_paths)
    save_results_to_json(results, "comparison_results.json")
