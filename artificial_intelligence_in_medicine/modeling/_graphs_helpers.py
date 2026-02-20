from pathlib import Path
import pickle
from typing import Any, Counter
import numpy as np
from loguru import logger
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
from networkx.algorithms.structuralholes import constraint
from networkx.classes import Graph
import pandas as pd
from tqdm import tqdm
import typer
import matplotlib.pyplot as plt
from artificial_intelligence_in_medicine.config import (
    FIGURES_DIR,
    INTERIM_DATA_DIR,
    MODELS_DIR,
)
from sentence_transformers import SentenceTransformer


def load_graph(graph_path):
    with open(graph_path, "rb") as f:
        graph = pickle.load(f)
    return graph


def citation_community_graph(G: nx.Graph):
    nx.community.louvain_communities(G, seed=42)


def plot_normalized_constraints_over_time(G, constraints, MODE):
    """
    Plot normalized constraint distributions over time.

    Normalization is implicit: each year's distribution is based only on
    nodes that have a valid constraint value, so years with fewer articles
    do not dominate the visualization.

    Parameters
    ----------
    G : networkx.Graph or DiGraph
        Graph with node attribute 'year'
    constraints : dict
        Output of nx.constraint(G)
    MODE : str
        Mode name used for titles and output path

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Interactive violin plot
    """
    import plotly.graph_objects as go
    from collections import defaultdict

    # Collect constraint values by year
    year_to_constraints = defaultdict(list)

    for node, c in constraints.items():
        year = G.nodes[node].get("year")
        if year is not None:
            year_to_constraints[year].append(c)

    # Sort years
    years = sorted(year_to_constraints.keys())

    fig = go.Figure()

    for year in years:
        values = year_to_constraints[year]

        if len(values) < 2:
            # Skip degenerate distributions
            continue

        fig.add_trace(
            go.Violin(
                x=[year] * len(values),
                y=values,
                name=str(year),
                box_visible=True,
                meanline_visible=True,
                showlegend=False,
                hovertemplate=(
                    "<b>Year:</b> %{x}<br><b>Constraint:</b> %{y:.4f}<br><extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=f"Normalized Constraint Distributions Over Time ({MODE})",
        xaxis_title="Year",
        yaxis_title="Constraint",
        template="plotly_white",
        width=1100,
        height=600,
    )

    # Save output
    output_path = FIGURES_DIR / MODE / "normalized_constraints_over_time.html"
    fig.write_html(output_path)
    logger.info(f"Saved normalized constraint distribution plot to {output_path}")

    return fig


def plot_constraints_over_time(G, constraints, MODE):
    """
    Plot the median constraint value for every year over time using Plotly.

    Parameters:
    -----------
    G : networkx.DiGraph
        A NetworkX directed graph where nodes have a 'year' attribute
    constraints : dict
        Dictionary mapping node IDs to their constraint values (from nx.constraint)
    MODE : str
        Description of the constraint type (used for plot title/label)

    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Interactive Plotly figure
    """
    # Extract year and constraint data for nodes in the graph
    year_constraint_data = {}

    import plotly.graph_objects as go
    import numpy as np

    for node in G.nodes():
        if node in constraints:
            year = G.nodes[node].get("year")
            if year is not None:
                if year not in year_constraint_data:
                    year_constraint_data[year] = []
                year_constraint_data[year].append(constraints[node])
        else:
            continue

    # Calculate median constraint for each year
    years = sorted(year_constraint_data.keys())
    medians = [np.median(year_constraint_data[year]) for year in years]

    # Create the Plotly figure
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=years,
            y=medians,
            mode="lines+markers",
            name=f"Median Constraint",
            line=dict(width=2, color="#1f77b4"),
            marker=dict(size=8, color="#1f77b4"),
            hovertemplate="<b>Year:</b> %{x}<br>"
            + "<b>Median Constraint:</b> %{y:.4f}<br>"
            + "<extra></extra>",
        )
    )

    fig.update_layout(
        title=f"Median Network Constraint Over Time ({MODE})",
        xaxis_title="Year",
        yaxis_title="Median Constraint",
        hovermode="closest",
        template="plotly_white",
        width=1000,
        height=600,
        showlegend=False,
    )

    # Save the figure
    output_path = FIGURES_DIR / MODE / "constraints_over_time.html"
    fig.write_html(output_path)
    logger.info(f"Saved constraint over time plot to {output_path}")

    return fig


def largest_component_subgraph(graph: nx.Graph):
    components = nx.connected_components(graph)
    print(components)
    largest = max(components, key=len)
    return graph.subgraph(largest)


def compare_brokerage(node_brokerage, community_brokerage):
    max_node = max(node_brokerage.values())
    max_community = max(community_brokerage.values())
    if max_community > max_node:
        return "Communities have greater brokerage."
    else:
        return "Individual articles have greater brokerage."


def print_top_community_attributes(graph: nx.Graph, community_brokerage):
    logger.debug(f"Type of community_brokerage: {type(community_brokerage)}")
    top_community = max(community_brokerage, key=community_brokerage.get)
    print(f"Top community (ID: {top_community}) attributes:")

    # Get all nodes in the top community
    for v in graph.nodes:
        if graph.nodes[v].get("community") == top_community:
            print(graph.nodes[v])


def assign_community_labels(MODE, G):
    from collections import defaultdict
    from sklearn.feature_extraction.text import TfidfVectorizer

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
        community_names = {i: f"Community {i}" for i in set(communities.values())}

    # Save community labels
    nx.set_node_attributes(
        G, {n: community_names[communities[n]] for n in G.nodes()}, "community_label"
    )

    return G


def calculate_constraint(g: Path | nx.Graph):
    if isinstance(g, Path):
        with open(g, "rb") as f:
            G: nx.Graph
            G = pickle.load(f)
    else:
        G = g
    return constraint(G)


def visualize_constraints(g):
    pass


def initialize_graph(mode: str):
    MODE = mode
    features_path: Path = INTERIM_DATA_DIR / MODE / "dataset_with_citation_data.json"
    model_path: Path = MODELS_DIR / MODE / "citation_model.pkl"
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Initializing graph for citation modeling.")
    G = nx.DiGraph()
    df = pd.read_json(features_path)
    # Check available columns
    logger.info(f"Available columns: {df.columns.tolist()}")
    logger.info(f"Number of articles in {MODE}: {len(df)}")
    # Use consistent column name for PMID
    pmid_column = "pmid" if "pmid" in df.columns else "_id"
    if pmid_column not in df.columns:
        raise KeyError(
            f"""Neither 'pmid' nor '_id' found in DataFrame columns: {df.columns.tolist()}"""
        )
    logger.info(f"Using '{pmid_column}' as PMID column")

    logger.info("Now adding nodes to the graph.")
    logger.info("Now adding nodes to the graph.")
    for _, row in tqdm(df.iterrows(), desc="Adding nodes"):
        G.add_node(
            str(row[pmid_column]),
            title=row["title"],
            cited_by=row["cited_by"],
            mesh_headings=row["mesh_headings"],
            year=row["year"],
            matched_lat=row.get("matched_lat"),
            matched_lon=row.get("matched_lon"),
            matched_country=row.get("matched_country"),
            matched_ror_id=row.get("matched_ror_id"),
            matched_name=row.get("matched_name"),
            matched_raw_text=row.get("matched_raw_text"),
        )

    logger.info("Now adding edges to the graph.")
    pmid_set = set(df[pmid_column].astype(str))
    for _, row in tqdm(df.iterrows(), desc="Adding edges"):
        cited_by_list = row["cited_by"]
        if hasattr(cited_by_list, "__iter__") and not isinstance(cited_by_list, str):
            source_pmid = str(row[pmid_column])
            for cited_pmid in cited_by_list:
                cited_pmid_str = str(cited_pmid)
                if cited_pmid_str in pmid_set and row["pmid"] != cited_pmid_str:
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


def graph_visualization(G: nx.Graph):
    return None


def find_central_nodes(mode: str = "ARTIFICIAL_INTELLIGENCE"):
    graph: Path | nx.Graph = MODELS_DIR / mode / "citation_model.pkl"
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    if isinstance(graph, Path):
        logger.info("Initializing graph for citation modeling.")
        with open(graph, "rb") as f:
            G = pickle.load(f)
    else:
        G = graph
    logger.info("Filtering out nodes with degree 0.")
    nodes_to_remove = [node for node, degree in dict(G.degree()).items() if degree == 0]
    G.remove_nodes_from(nodes_to_remove)
    logger.info(f"Removed {len(nodes_to_remove)} nodes with degree 0.")

    logger.info("Calculating degree centrality for all nodes.")
    centrality = nx.degree_centrality(G)

    logger.info("Finding top 5 central nodes.")
    sorted_nodes = sorted(centrality.items(), key=lambda item: item[1], reverse=True)

    top_5_nodes = sorted_nodes[:5]

    logger.info("Top 5 most central nodes:")
    for node_id, score in top_5_nodes:
        title = G.nodes[node_id].get("title", "No Title")
        logger.info(f"  - Title: {title}, Centrality: {score:.4f}")


def community_detection(mode: str, G: nx.Graph | Path, inflection_point: int) -> nx.Graph:
    from collections import defaultdict
    from pathlib import Path
    import pickle

    import networkx as nx
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    from sklearn.feature_extraction.text import TfidfVectorizer

    g = G
    original_node_count = g.number_of_nodes()

    # Filter to high-degree nodes
    g = g.to_undirected()
    g = g.subgraph([n for n, d in g.degree() if d > inflection_point]).copy()

    high_degree_nodes = g.number_of_nodes()
    percent_high_degree = (
        100 * high_degree_nodes / original_node_count if original_node_count > 0 else 0
    )

    print(
        f"Inflection point degree: {inflection_point}. "
        f"Nodes above inflection: {high_degree_nodes} "
        f"({percent_high_degree:.2f}% of total nodes: {original_node_count})"
    )

    # Remove isolated nodes
    isolated = list(nx.isolates(g))
    if isolated:
        g.remove_nodes_from(isolated)

    # Preserve graph metadata
    if hasattr(g, "graph") and "year" in g.graph:
        g.graph["year"] = g.graph["year"]

    # COMMUNITY DETECTION
    communities = None
    if g.number_of_nodes() > 0 and g.number_of_edges() > 0:
        try:
            # Using greedy modularity as an analog to Leiden (works in NetworkX)
            communities_list = list(greedy_modularity_communities(g))
            # Build membership map
            logger.success("Sucessfully completed community detection.")
            membership = {}
            for i, comm in enumerate(communities_list):
                for node in comm:
                    membership[node] = i
            communities = membership
            modularity_score = nx.algorithms.community.modularity(g, communities_list)
        except Exception as e:
            logger.error(f"Error during community detection: {e}")
            communities = {n: i for i, n in enumerate(g.nodes())}
            modularity_score = 0.0
    else:
        communities = {n: i for i, n in enumerate(g.nodes())}
        modularity_score = 0.0
    # Assign community membership
    nx.set_node_attributes(g, communities, "community")
    return g
    # Generate community names
    #


def calculate_inflection_point(G: nx.Graph, MODE: str) -> int:
    import numpy as np

    mode = MODE
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
        inflection_degree_threshold: int = sorted_degrees_values[elbow_index]

    return inflection_degree_threshold


def assign_countries_from_latlon(G):
    """
    Adds a 'country' attribute to each node using matched_lat/matched_lon.
    Uses Natural Earth (offline, reproducible).
    """
    import geopandas as gpd
    from shapely.geometry import Point

    # Load world country polygons
    world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    world = world[["name", "geometry"]].rename(columns={"name": "country"})

    # Build GeoDataFrame of nodes
    rows = []
    for node, data in G.nodes(data=True):
        lat = data.get("matched_lat")
        lon = data.get("matched_lon")
        if lat is not None and lon is not None:
            rows.append(
                {
                    "node": node,
                    "geometry": Point(lon, lat),
                }
            )

    nodes_gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")

    # Spatial join
    joined = gpd.sjoin(nodes_gdf, world, how="left", predicate="within")

    # Attach country back to graph
    country_map = dict(zip(joined["node"], joined["country"]))
    nx.set_node_attributes(G, country_map, "country")

    return G


def assign_countries_from_latlon(G):
    """
    Adds a 'country' attribute to each node using matched_lat/matched_lon.
    Compatible with GeoPandas >= 1.0.
    """
    import geopandas as gpd
    import networkx as nx
    from shapely.geometry import Point
    from pathlib import Path
    import zipfile
    import urllib.request

    # Where to cache Natural Earth
    data_dir = Path.home() / ".cache" / "natural_earth"
    data_dir.mkdir(parents=True, exist_ok=True)

    shp_path = data_dir / "ne_110m_admin_0_countries.shp"

    if not shp_path.exists():
        url = "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
        zip_path = data_dir / "countries.zip"
        urllib.request.urlretrieve(url, zip_path)

        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(data_dir)

    # Load country polygons
    world = gpd.read_file(shp_path)[["ADMIN", "geometry"]]
    world = world.rename(columns={"ADMIN": "country"})

    # Build GeoDataFrame of nodes
    rows = []
    for node, data in G.nodes(data=True):
        lat = data.get("matched_lat")
        lon = data.get("matched_lon")
        if lat is not None and lon is not None:
            rows.append(
                {
                    "node": node,
                    "geometry": Point(lon, lat),
                }
            )

    if not rows:
        raise ValueError("No nodes had matched_lat/matched_lon")

    nodes_gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")

    # Spatial join
    joined = gpd.sjoin(nodes_gdf, world, how="left", predicate="within")

    # Attach to graph
    country_map = dict(zip(joined["node"], joined["country"]))
    nx.set_node_attributes(G, country_map, "country")

    return G


def plot_constraints_by_community(G, constraints, MODE, min_articles=5):
    """
    Plot constraint distributions by country using violins.

    Parameters
    ----------
    G : networkx.Graph
        Graph with 'country' attribute on nodes
    constraints : dict
        Output of nx.constraint(G)
    MODE : str
        Mode name for labeling
    """
    import plotly.graph_objects as go
    from collections import defaultdict

    community_constraints = defaultdict(list)

    for node, c in constraints.items():
        community = G.nodes[node].get("community_label")
        if community is not None:
            community_constraints[community].append(c)

    # Filter small-N communities
    community_constraints = {
        c: vals for c, vals in community_constraints.items() if len(vals) >= min_articles
    }

    # Sort by median constraint
    communities = sorted(
        community_constraints.keys(),
        key=lambda c: sum(community_constraints[c]) / len(community_constraints[c]),
    )

    fig = go.Figure()

    for community in communities:
        fig.add_trace(
            go.Violin(
                x=[str(community)] * len(community_constraints[community]),
                y=community_constraints[community],
                name=str(community),
                box_visible=True,
                meanline_visible=True,
                showlegend=False,
            )
        )

    fig.update_layout(
        title=f"Constraint Distributions by Community ({MODE})",
        xaxis_title="Community",
        yaxis_title="Constraint",
        template="plotly_white",
        width=1200,
        height=600,
    )

    output_path = FIGURES_DIR / MODE / "constraints_by_community.html"
    fig.write_html(output_path)
    logger.info(f"Saved constraints-by-community plot to {output_path}")

    return fig


def plot_constraints_by_country(G, constraints, MODE, min_articles=20):
    """
    Plot constraint distributions by country using violins.

    Parameters
    ----------
    G : networkx.Graph
        Graph with 'country' attribute on nodes
    constraints : dict
        Output of nx.constraint(G)
    MODE : str
        Mode name for labeling
    min_articles : int
        Minimum number of articles required to include a country
    """
    import plotly.graph_objects as go
    from collections import defaultdict

    country_constraints = defaultdict(list)

    for node, c in constraints.items():
        country = G.nodes[node].get("country")
        if country is not None:
            country_constraints[country].append(c)

    # Filter small-N countries
    country_constraints = {
        c: vals for c, vals in country_constraints.items() if len(vals) >= min_articles
    }

    # Sort by median constraint
    countries = sorted(
        country_constraints.keys(),
        key=lambda c: sum(country_constraints[c]) / len(country_constraints[c]),
    )

    fig = go.Figure()

    for country in countries:
        fig.add_trace(
            go.Violin(
                x=[country] * len(country_constraints[country]),
                y=country_constraints[country],
                name=country,
                box_visible=True,
                meanline_visible=True,
                showlegend=False,
            )
        )

    fig.update_layout(
        title=f"Constraint Distributions by Country ({MODE})",
        xaxis_title="Country",
        yaxis_title="Constraint",
        template="plotly_white",
        width=1200,
        height=600,
    )

    output_path = FIGURES_DIR / MODE / "constraints_by_country.html"
    fig.write_html(output_path)
    logger.info(f"Saved constraints-by-country plot to {output_path}")

    return fig


def generate_embeddings(
    g: nx.Graph,
    text_attr: str,
    embedding_attr: str = "embedding",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 64,
    normalize: bool = True,
):
    """
    Generate sentence-transformer embeddings for node text attributes
    and store them on the graph.

    Args:
        g: NetworkX graph
        text_attr: node attribute containing text (e.g. "title")
        embedding_attr: node attribute name to store embeddings
        model_name: sentence-transformers model name
        batch_size: encoding batch size
        normalize: whether to L2-normalize embeddings
    """

    model = SentenceTransformer(model_name)

    # --- collect nodes with text ---
    nodes = []
    texts = []
    for n, data in g.nodes(data=True):
        text = data.get(text_attr)
        if isinstance(text, str) and text.strip():
            nodes.append(n)
            texts.append(text)

    if not nodes:
        raise ValueError(f"No nodes found with text attribute '{text_attr}'")

    # --- encode in batches ---
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Batches"):
        batch = texts[i : i + batch_size]
        emb = model.encode(
            batch,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )
        embeddings.append(emb)

    embeddings = np.vstack(embeddings)  # (N, d)

    # --- attach to graph ---
    emb_dict = {node: emb for node, emb in zip(nodes, embeddings)}

    nx.set_node_attributes(g, emb_dict, embedding_attr)

    return g
