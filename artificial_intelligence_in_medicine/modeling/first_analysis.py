from collections import defaultdict
import time

import igraph as ig
from loguru import logger
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

from _graphs_helpers import initialize_graph
from artificial_intelligence_in_medicine.config import FIGURES_DIR, RESULTS_DATA_DIR

MODES = ["ARTIFICIAL_INTELLIGENCE", "GENE_EXPRESSION", "NULL"]


def clean_graph(G_nx):
    logger.info("Removing self-loops")
    G_nx.remove_edges_from(nx.selfloop_edges(G_nx))

    logger.info("Keeping largest connected component")

    if G_nx.number_of_nodes() == 0:
        return G_nx

    if G_nx.is_directed():
        components = nx.weakly_connected_components(G_nx)
    else:
        components = nx.connected_components(G_nx)

    largest_cc = max(components, key=len)
    G_nx = G_nx.subgraph(largest_cc).copy()

    logger.info("Removing isolates")
    G_nx.remove_nodes_from(list(nx.isolates(G_nx)))

    logger.success(
        f"Cleaned graph: {G_nx.number_of_nodes()} nodes, {G_nx.number_of_edges()} edges"
    )

    return G_nx


def power_law_fit(G_ig) -> ig.FittedPowerLaw:
    pass


def generate_cluster_graph(G_ig, MODE):
    # Generate community clusters
    communities = G_ig.community_leiden()

    for i, community in enumerate(communities):
        with open(RESULTS_DATA_DIR / MODE / "community_list_{i:05d}.txt", "w") as f:
            f.write(f"Community {i}")
            for v in community:
                print(f"\t{G_ig.vs[v]['title']}---{G_ig.vs[v]['name']}")
    num_communities = len(communities)
    palette1 = ig.RainbowPalette(n=num_communities)
    for i, community in enumerate(communities):
        G_ig.vs[community]["color"] = i
        community_edges = G_ig.es.select(_within=community)
        community_edges["color"] = G_ig.vs["title"] = [
            "\n\n" + label for label in G_ig.vs["title"]
        ]
        fig1, ax1 = plt.subplots()
    ig.plot(
        communities,
        target=ax1,
        mark_groups=True,
        palette=palette1,
        vertex_size=15,
        edge_width=0.5,
    )
    fig1.set_size_inches(20, 20)
    fig1.savefig(RESULTS_DATA_DIR / MODE / "community_leiden_cluastergraph.png", dpi=400)
    plt.close(fig1)


def nx_to_igraph(G_nx):
    mapping = {node: i for i, node in enumerate(G_nx.nodes())}
    reverse_mapping = {i: node for node, i in mapping.items()}
    edges = [(mapping[u], mapping[v]) for u, v in G_nx.edges()]

    G_ig = ig.Graph(directed=False)
    G_ig.add_vertices(len(mapping))
    G_ig.add_edges(edges)

    G_ig.vs["name"] = [reverse_mapping[i] for i in range(len(mapping))]

    attr_keys = [
        "title",
        "year",
        "matched_country",
        "matched_name",
        "matched_lat",
        "matched_lon",
    ]

    for attr in attr_keys:
        G_ig.vs[attr] = [G_nx.nodes[reverse_mapping[i]].get(attr) for i in range(len(mapping))]

    return G_ig


def compute_constraint_bins(G_ig):
    logger.info(f"Computing structural constraint for {G_ig.vcount()} nodes")

    start = time.time()
    constraint_vals = G_ig.constraint()
    elapsed = time.time() - start
    logger.success(f"Constraint computed in {elapsed:.2f} seconds")

    bins = np.arange(0.0, 1.01, 0.2)
    bin_labels = ["0.0–0.2", "0.2–0.4", "0.4–0.6", "0.6–0.8", "0.8–1.0"]

    node_bins = {}
    for i, value in enumerate(constraint_vals):
        bin_index = np.digitize(value, bins, right=True) - 1
        bin_index = max(0, min(bin_index, len(bin_labels) - 1))
        node_bins[i] = bin_index

    cmap = plt.get_cmap("Set2")
    colors = [cmap(i) for i in range(len(bin_labels))]
    node_colors = [colors[node_bins[i]] for i in range(G_ig.vcount())]

    legend_patches = [
        mpatches.Patch(color=colors[i], label=bin_labels[i]) for i in range(len(bin_labels))
    ]

    constraint_dict = {G_ig.vs[i]["name"]: constraint_vals[i] for i in range(G_ig.vcount())}

    return constraint_dict, node_bins, node_colors, legend_patches, bin_labels


def main():
    for MODE in tqdm(MODES, desc="Modes"):
        logger.info(f"Processing mode: {MODE}")

        G_nx = initialize_graph(MODE)
        G_nx = clean_graph(G_nx)
        G_ig = nx_to_igraph(G_nx)
        generate_cluster_graph(G_ig, MODE)
        figures_path = FIGURES_DIR / MODE
        figures_path.mkdir(parents=True, exist_ok=True)

        results_path = RESULTS_DATA_DIR / MODE
        results_path.mkdir(parents=True, exist_ok=True)

        constraint_dict, node_bins, node_colors, legend_patches, bin_labels = (
            compute_constraint_bins(G_ig)
        )

        df_original = pd.DataFrame(
            {
                "node_id": list(constraint_dict.keys()),
                "constraint": list(constraint_dict.values()),
                "constraint_bin": [bin_labels[node_bins[i]] for i in range(len(node_bins))],
            }
        )

        df_original.to_csv(
            results_path / "original_graph_constraint.csv",
            index=False,
        )

        layout = G_ig.layout("drl")

        fig, ax = plt.subplots(figsize=(20, 16))
        ig.plot(
            G_ig,
            target=ax,
            layout=layout,
            vertex_size=5,
            vertex_color=node_colors,
            edge_width=0.3,
        )

        plt.title("Original Graph (Constraint Binned)")
        plt.legend(handles=legend_patches, title="Constraint", loc="best")
        plt.tight_layout()
        plt.savefig(figures_path / "original_graph.svg")
        plt.close()

        # --------------------------------------------
        # Louvain community detection
        # --------------------------------------------
        logger.info("Running Louvain community detection...")
        partition = G_ig.community_multilevel()
        membership = partition.membership
        logger.success(f"Detected {len(partition)} communities")

        communities = defaultdict(list)
        for i, comm_id in enumerate(membership):
            communities[comm_id].append(G_ig.vs[i]["name"])

        # --------------------------------------------
        # Build community meta-graph
        # --------------------------------------------
        logger.info("Building community meta-graph")

        comm_edges = defaultdict(int)

        for e in G_ig.es:
            u, v = e.tuple
            cu = membership[u]
            cv = membership[v]
            if cu != cv:
                edge = tuple(sorted((cu, cv)))
                comm_edges[edge] += 1

        G_comm = ig.Graph(directed=False)
        G_comm.add_vertices(len(communities))
        G_comm.add_edges(list(comm_edges.keys()))

        G_comm.vs["name"] = list(range(len(communities)))
        G_comm.vs["num_members"] = [len(communities[i]) for i in range(len(communities))]
        G_comm.es["weight"] = [comm_edges[e.tuple] for e in G_comm.es]

        # --------------------------------------------
        # Save community meta-graph
        # --------------------------------------------
        logger.info("Saving community meta-graph")

        G_comm.write_graphml(str(results_path / "community_metagraph.graphml"))

        edge_df = pd.DataFrame(
            {
                "source": [G_comm.vs[e.tuple[0]]["name"] for e in G_comm.es],
                "target": [G_comm.vs[e.tuple[1]]["name"] for e in G_comm.es],
                "weight": G_comm.es["weight"],
            }
        )

        edge_df.to_csv(
            results_path / "community_metagraph_edges.csv",
            index=False,
        )

        node_df = pd.DataFrame(
            {
                "community_id": G_comm.vs["name"],
                "num_members": G_comm.vs["num_members"],
                "member_node_ids": [
                    ";".join(map(str, communities[i])) for i in range(len(communities))
                ],
            }
        )

        node_df.to_csv(
            results_path / "community_metagraph_nodes.csv",
            index=False,
        )

        # --------------------------------------------
        # Constraint on meta graph
        # --------------------------------------------
        constraint_meta, node_bins_meta, _, _, bin_labels_meta = compute_constraint_bins(G_comm)

        df_meta = pd.DataFrame(
            {
                "community_id": list(constraint_meta.keys()),
                "constraint": list(constraint_meta.values()),
                "constraint_bin": [
                    bin_labels_meta[node_bins_meta[i]] for i in range(len(node_bins_meta))
                ],
                "num_members": G_comm.vs["num_members"],
            }
        )

        df_meta.to_csv(
            results_path / "community_metagraph_constraint.csv",
            index=False,
        )

        logger.info("Plotting community meta-graph")

        cmap = plt.get_cmap("Set2")
        colors_meta = [cmap(i) for i in range(len(bin_labels_meta))]
        node_colors_meta = [colors_meta[node_bins_meta[i]] for i in range(G_comm.vcount())]

        import math

        vertex_sizes = [3 + 1.5 * math.log1p(n) for n in G_comm.vs["num_members"]]
        layout_comm = G_comm.layout("drl")

        fig, ax = plt.subplots(figsize=(20, 16))
        ig.plot(
            G_comm,
            target=ax,
            layout=layout_comm,
            vertex_size=vertex_sizes,
            vertex_color=node_colors_meta,
            edge_width=[0.2 + 1.0 * (w - w_min) / (w_max - w_min) for w in weights],
        )

        plt.title("Community Meta-Graph (Constraint Binned)")
        plt.legend(handles=legend_patches, title="Constraint", loc="best")
        plt.tight_layout()
        plt.savefig(figures_path / "community_metagraph.svg")
        plt.close()

        logger.success(f"Finished mode: {MODE}")


if __name__ == "__main__":
    main()
