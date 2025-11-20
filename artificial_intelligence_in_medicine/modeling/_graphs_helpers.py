from pathlib import Path
import pickle
from typing import Counter

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


def load_graph(graph_path):
    with open(graph_path, "rb") as f:
        graph = pickle.load(f)
    return graph


def largest_component_subgraph(graph: nx.Graph):
    components = nx.connected_components(graph)
    print(components)
    largest = max(components, key=len)
    return graph.subgraph(largest)


def compute_brokerage(graph):
    # Node-level brokerage: edges to nodes in other communities
    node_brokerage = []
    community_brokerage = Counter()
    membership = graph.vs["community"]
    for v in graph.vs:
        v_comm = membership[v.index]
        count = 0
        for neighbor in graph.neighbors(v.index):
            n_comm = membership[neighbor]
            if n_comm != v_comm:
                count += 1
                community_brokerage[v_comm] += 1
        node_brokerage.append(count)
    # Each edge counted twice, so halve community brokerage
    for k in community_brokerage:
        community_brokerage[k] //= 2
    return node_brokerage, community_brokerage


def compare_brokerage(node_brokerage, community_brokerage):
    max_node = max(node_brokerage)
    max_community = max(community_brokerage.values())
    if max_community > max_node:
        return "Communities have greater brokerage."
    else:
        return "Individual articles have greater brokerage."


def print_top_community_attributes(graph, community_brokerage):
    top_community = max(community_brokerage, key=community_brokerage.get)
    print(f"Top community (ID: {top_community}) attributes:")
    for v in graph.vs.select(community=top_community):
        print(v.attributes())


def analyze_brokerage(graph: nx.Graph | Path):
    if isinstance(graph, Path):
        graph = load_graph(graph)
    else:
        graph = nx.to_undirected(graph)
        graph = largest_component_subgraph(graph)
    node_brokerage, community_brokerage = compute_brokerage(graph)
    result = compare_brokerage(node_brokerage, community_brokerage)
    print(f"Node brokerage: {node_brokerage}")
    print(f"Community brokerage: {community_brokerage}")
    print(result)
    print_top_community_attributes(graph, community_brokerage)
    return node_brokerage, community_brokerage, result


def calculate_constraint(g: Path | nx.Graph):
    if isinstance(g, Path):
        with open(g, "rb") as f:
            G: nx.Graph
            G = pickle.load(f)
    else:
        G = g
    return constraint(G)


def initialize_graph(mode: str = "ARTIFICIAL_INTELLIGENCE"):
    MODE = mode
    features_path: Path = INTERIM_DATA_DIR / MODE / "features_with_ror.json"
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


def community_detection(mode: str, g: nx.Graph | Path):
    from collections import defaultdict
    from pathlib import Path
    import pickle

    import networkx as nx
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    from sklearn.feature_extraction.text import TfidfVectorizer

    MODE = mode
    graph_path: Path = MODELS_DIR / MODE / "citation_model.pkl"
    output_path: Path = FIGURES_DIR / MODE / "community_detection.html"
    if isinstance(g, Path):
        with open(graph_path, "rb") as f:
            G: nx.Graph = pickle.load(f)
    else:
        G = g
    original_node_count = G.number_of_nodes()

    # Choose inflection point
    if MODE == "GENE_EXPRESSION":
        inflection_point = 25
    else:
        inflection_point = 23

    # Filter to high-degree nodes
    graph = G.to_undirected()
    graph = graph.subgraph([n for n, d in graph.degree() if d > inflection_point]).copy()

    high_degree_nodes = graph.number_of_nodes()
    percent_high_degree = (
        100 * high_degree_nodes / original_node_count if original_node_count > 0 else 0
    )
    print(
        f"Inflection point degree: {inflection_point}. "
        f"Nodes above inflection: {high_degree_nodes} "
        f"({percent_high_degree:.2f}% of total nodes: {original_node_count})"
    )

    # Remove isolated nodes
    isolated = list(nx.isolates(graph))
    if isolated:
        graph.remove_nodes_from(isolated)

    # Preserve graph metadata
    if hasattr(G, "graph") and "year" in G.graph:
        graph.graph["year"] = G.graph["year"]

    # COMMUNITY DETECTION
    communities = None
    if graph.number_of_nodes() > 0 and graph.number_of_edges() > 0:
        try:
            # Using greedy modularity as an analog to Leiden (works in NetworkX)
            communities_list = list(greedy_modularity_communities(graph))
            # Build membership map
            membership = {}
            for i, comm in enumerate(communities_list):
                for node in comm:
                    membership[node] = i
            communities = membership
            modularity_score = nx.algorithms.community.modularity(graph, communities_list)
        except Exception as e:
            print(f"Error during community detection: {e}")
            communities = {n: i for i, n in enumerate(graph.nodes())}
            modularity_score = 0.0
    else:
        communities = {n: i for i, n in enumerate(graph.nodes())}
        modularity_score = 0.0

    # Assign community membership
    nx.set_node_attributes(graph, communities, "community")

    # Generate community names
    community_names = {}
    if nx.get_node_attributes(graph, "title"):
        community_titles = defaultdict(list)
        for n, data in graph.nodes(data=True):
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
        graph, {n: community_names[communities[n]] for n in graph.nodes()}, "community_label"
    )

    # Save updated graph and labels
    output_graph_path = graph_path.with_name(f"{graph_path.stem}_with_communities.pkl")
    with open(output_graph_path, "wb") as f:
        pickle.dump(graph, f)

    with open(output_path.with_name("community_labels.pkl"), "wb") as f:
        pickle.dump(community_names, f)

    print("Graph and labels saved. Visualizing...")

    # Visualization
    try:
        pos = nx.spring_layout(graph, seed=42, k=0.15 if graph.number_of_nodes() < 1000 else None)
        x_coords = [pos[n][0] for n in graph.nodes()]
        y_coords = [pos[n][1] for n in graph.nodes()]

        node_titles = [data.get("title", f"Node {n}") for n, data in graph.nodes(data=True)]

        colors_discrete = px.colors.qualitative.Set1 + px.colors.qualitative.Set3
        node_colors = [
            colors_discrete[data["community"] % len(colors_discrete)]
            for n, data in graph.nodes(data=True)
        ]

        # Edges
        edge_x, edge_y = [], []
        edge_shapes = []
        for u, v in graph.edges():
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
            for _, data in graph.nodes(data=True)
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
                for (n, data), title in zip(graph.nodes(data=True), node_titles)
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
                    f"Modularity: {modularity_score:.4f}"
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
