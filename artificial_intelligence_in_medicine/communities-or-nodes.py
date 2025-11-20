import pickle
from artificial_intelligence_in_medicine.config import MODELS_DIR
from igraph import Graph
from collections import Counter, defaultdict

MODE = "GENE_EXPRESSION"


def load_graph(graph_path):
    with open(graph_path, "rb") as f:
        graph = pickle.load(f)
    return graph


def largest_component_subgraph(graph):
    components = graph.components()
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


def analyze_brokerage(graph_path):
    graph = load_graph(graph_path)
    graph = largest_component_subgraph(graph)
    node_brokerage, community_brokerage = compute_brokerage(graph)
    result = compare_brokerage(node_brokerage, community_brokerage)
    print(f"Node brokerage: {node_brokerage}")
    print(f"Community brokerage: {community_brokerage}")
    print(result)
    print_top_community_attributes(graph, community_brokerage)
    return node_brokerage, community_brokerage, result


analyze_brokerage(MODELS_DIR / MODE / "citation_model_with_communities.pkl")
