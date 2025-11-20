import pickle
import math
import networkx as nx
import igraph as ig
import pandas as pd
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from loguru import logger
from artificial_intelligence_in_medicine.config import MODELS_DIR, RESULTS_DATA_DIR

MODE = "GENE_EXPRESSION"
graph_path = MODELS_DIR / MODE / "citation_model_with_communities.pkl"

logger.info(f"Loading graph: {graph_path}")
with open(graph_path, "rb") as f:
    graph = pickle.load(f)

logger.info(f"Loaded graph type: {type(graph)}")


# -------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------


def is_valid_number(v):
    return v is not None and not (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))


def get_top_percent(df, column, percent=0.1):
    cutoff = max(1, int(len(df) * percent))
    return df.sort_values(by=column, ascending=False).head(cutoff)


def nx_to_igraph(nx_graph):
    """Convert a NetworkX graph to an igraph.Graph, preserving key attributes."""
    logger.info("Converting NetworkX graph to igraph...")
    G = nx_graph.to_undirected()
    g = ig.Graph()
    g.add_vertices(list(G.nodes()))
    g.add_edges(list(G.edges()))

    # copy node attributes
    for attr_name in next(iter(G.nodes(data=True)))[1].keys():
        g.vs[attr_name] = [G.nodes[n].get(attr_name) for n in G.nodes()]

    # copy edge attributes
    for attr_name in next(iter(G.edges(data=True)))[2].keys():
        g.es[attr_name] = [G.edges[e].get(attr_name) for e in G.edges()]

    return g


# -------------------------------------------------------------------
# 1. Compute node-level constraint
# -------------------------------------------------------------------


def compute_node_constraints(graph, titles=None):
    """Compute Burt's constraint for all nodes in the largest connected component."""
    weights = graph.es["cited_by"] if "cited_by" in graph.es.attribute_names() else None
    components = graph.components()
    largest_component_id = max(range(len(components)), key=lambda i: len(components[i]))
    vertices_in_largest_component = components[largest_component_id]
    largest_component = graph.induced_subgraph(vertices_in_largest_component)

    constraints = largest_component.constraint(weights=weights)

    has_name = "name" in largest_component.vs.attribute_names()
    has_title = "title" in largest_component.vs.attribute_names()
    has_community = "community" in largest_component.vs.attribute_names()

    labels = [v["name"] if has_name else v.index for v in largest_component.vs]
    community = [v["community"] if has_community else None for v in largest_component.vs]
    title_vals = (
        [v["title"] if has_title else None for v in largest_component.vs]
        if titles is None
        else [titles.get(v["name"] if has_name else v.index) for v in largest_component.vs]
    )

    df_nodes = pd.DataFrame(
        {"node": labels, "constraint": constraints, "community": community, "title": title_vals}
    )

    return df_nodes


# -------------------------------------------------------------------
# 2. Compute community-level constraint
# -------------------------------------------------------------------


def compute_community_constraints(graph):
    """Aggregate graph into a community-level graph and compute Burt’s constraint for each community."""
    community_nodes = defaultdict(list)
    community_edges = defaultdict(int)

    has_community = "community" in graph.vs.attribute_names()
    if not has_community:
        raise ValueError(
            "Graph vertices must have a 'community' attribute to compute community constraints."
        )

    for v in graph.vs:
        community_nodes[v["community"]].append(v.index)

    for e in graph.es:
        s_comm = graph.vs[e.source]["community"]
        t_comm = graph.vs[e.target]["community"]
        if s_comm != t_comm:
            community_edges[(s_comm, t_comm)] += 1

    community_graph = ig.Graph()
    community_graph.add_vertices(list(community_nodes.keys()))
    community_graph.add_edges(list(community_edges.keys()))

    constraints = community_graph.constraint()
    df_communities = pd.DataFrame(
        {"community": list(community_nodes.keys()), "constraint": constraints}
    )
    return df_communities


# -------------------------------------------------------------------
# 3. Compute community labels (TF-IDF)
# -------------------------------------------------------------------


def compute_community_labels(node_community, titles):
    """Compute TF-IDF labels for each community from article titles."""
    community_titles = defaultdict(list)
    for n, title in titles.items():
        cid = node_community[n]
        community_titles[cid].append(title)

    community_names = {}
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
    return community_names


# -------------------------------------------------------------------
# 4. Run everything
# -------------------------------------------------------------------

# Convert to igraph if needed
if isinstance(graph, (nx.Graph, nx.DiGraph)):
    graph = nx_to_igraph(graph)

# Safely extract attributes
has_name = "name" in graph.vs.attribute_names()
has_title = "title" in graph.vs.attribute_names()
has_community = "community" in graph.vs.attribute_names()

node_community = {
    (v["name"] if has_name else v.index): v["community"] for v in graph.vs if has_community
}
titles = {(v["name"] if has_name else v.index): v["title"] for v in graph.vs if has_title}

# Compute metrics
df_nodes = compute_node_constraints(graph, titles)
df_communities = compute_community_constraints(graph)
community_names = compute_community_labels(node_community, titles)

# Add labels
df_communities["label"] = df_communities["community"].map(community_names)
df_nodes["community_label"] = df_nodes["community"].map(community_names)

# -------------------------------------------------------------------
# 5. Extract top 10%
# -------------------------------------------------------------------

top_nodes = get_top_percent(df_nodes, "constraint", percent=0.1)
top_communities = get_top_percent(df_communities, "constraint", percent=0.1)

# -------------------------------------------------------------------
# 6. Display clean tables
# -------------------------------------------------------------------

print("\n=== Top 10% Nodes by Burt's Constraint ===")
print(
    top_nodes[["node", "title", "community", "community_label", "constraint"]].to_string(
        index=False
    )
)

print("\n=== Top 10% Communities by Burt's Constraint ===")
print(top_communities[["community", "label", "constraint"]].to_string(index=False))


# Optional: export
top_nodes.to_html(RESULTS_DATA_DIR / MODE / "top_nodes_constraint.html", index=False)
top_communities.to_html(RESULTS_DATA_DIR / MODE / "top_communities_constraint.csv", index=False)

# 7. Display the basic characteristics of the graph
print(graph.vcount())
