from loguru import logger
import pandas as pd
from artificial_intelligence_in_medicine.config import FIGURES_DIR, RESULTS_DATA_DIR
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from collections import defaultdict
from _graphs_helpers import initialize_graph
import igraph as ig
import networkx as nx
from tqdm import tqdm
import time

MODES = ["ARTIFICIAL_INTELLIGENCE", "GENE_EXPRESSION", "NULL"]


# ------------------------------------------------------------
# Convert NetworkX → igraph
# ------------------------------------------------------------
def nx_to_igraph(G_nx):
    mapping = {node: i for i, node in enumerate(G_nx.nodes())}
    reverse_mapping = {i: node for node, i in mapping.items()}
    edges = [(mapping[u], mapping[v]) for u, v in G_nx.edges()]

    G_ig = ig.Graph()
    G_ig.add_vertices(len(mapping))
    G_ig.add_edges(edges)

    # Node name = PMID string
    G_ig.vs["name"] = [reverse_mapping[i] for i in range(len(mapping))]

    # Copy all NetworkX node attributes
    attr_keys = ["title", "year", "matched_country", "matched_name", "matched_lat", "matched_lon"]
    for attr in attr_keys:
        G_ig.vs[attr] = [G_nx.nodes[reverse_mapping[i]].get(attr) for i in range(len(mapping))]

    return G_ig


# ------------------------------------------------------------
# Constraint + binning
# ------------------------------------------------------------
def compute_constraint_bins(G_ig):
    logger.info(f"Computing structural constraint for {G_ig.vcount()} nodes")

    start = time.time()
    constraint_vals = G_ig.constraint()
    elapsed = time.time() - start

    logger.success(f"Constraint computed in {elapsed:.2f} seconds")

    bins = np.arange(0.0, 1.01, 0.2)
    bin_labels = ["0.0–0.2", "0.2–0.4", "0.4–0.6", "0.6–0.8", "0.8–1.0"]

    node_bins = {}
    for i, value in tqdm(enumerate(constraint_vals), desc="Computing bins"):
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


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    for MODE in tqdm(MODES, desc="Modes"):
        logger.info(f"Processing mode: {MODE}")

        # --------------------------------------------
        # Load graph
        # --------------------------------------------
        G_nx = initialize_graph(MODE)
        G_nx.remove_nodes_from(list(nx.isolates(G_nx)))
        G_ig = nx_to_igraph(G_nx)

        figures_path = FIGURES_DIR / MODE
        figures_path.mkdir(parents=True, exist_ok=True)

        results_path = RESULTS_DATA_DIR / MODE
        results_path.mkdir(parents=True, exist_ok=True)

        # --------------------------------------------
        # Constraint on original graph
        # --------------------------------------------
        constraint_dict, node_bins, node_colors, legend_patches, bin_labels = (
            compute_constraint_bins(G_ig)
        )
        logger.info("Finished computing constraints")

        logger.info("Building DF")
        df_original = pd.DataFrame(
            {
                "node_id": list(constraint_dict.keys()),
                "constraint": list(constraint_dict.values()),
                "constraint_bin": [bin_labels[node_bins[i]] for i in range(len(node_bins))],
            }
        )

        df_original.to_csv(results_path / "original_graph_constraint.csv", index=False)

        logger.info("Saved df original")
        # --------------------------------------------
        # Plot original graph
        # --------------------------------------------
        logger.info("Plotting layout")
        layout = G_ig.layout("fr")

        fig, ax = plt.subplots(figsize=(10, 8))
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

        logger.info("Finished plotting")
        # --------------------------------------------
        # Louvain community detection
        # --------------------------------------------
        logger.info("Running Louvain community detection...")
        start = time.time()
        partition = G_ig.community_multilevel()
        elapsed = time.time() - start
        logger.success(f"Louvain detected {len(partition)} communities in {elapsed:.2f}s")

        membership = partition.membership

        communities = defaultdict(list)
        for i, comm_id in tqdm(
            enumerate(membership),
            total=len(membership),
            desc="Assigning communities",
            leave=False,
        ):
            communities[comm_id].append(G_ig.vs[i]["name"])

        # --------------------------------------------
        # Build community meta-graph
        # --------------------------------------------
        logger.info("Building community meta-graph")

        comm_edges = defaultdict(int)

        for e in tqdm(G_ig.es, desc="Meta edges", leave=False):
            u, v = e.tuple
            cu = membership[u]
            cv = membership[v]
            if cu != cv:
                edge = tuple(sorted((cu, cv)))
                comm_edges[edge] += 1

        G_comm = ig.Graph()
        G_comm.add_vertices(len(communities))
        G_comm.add_edges(list(comm_edges.keys()))

        # --------------------------------------------
        # Constraint on meta graph
        # --------------------------------------------
        constraint_meta, node_bins_meta, node_colors_meta, legend_patches_meta, bin_labels_meta = (
            compute_constraint_bins(G_comm)
        )

        df_meta = pd.DataFrame(
            {
                "community_id": list(range(len(communities))),
                "constraint": list(constraint_meta.values()),
                "constraint_bin": [
                    bin_labels_meta[node_bins_meta[i]] for i in range(len(node_bins_meta))
                ],
                "num_members": [len(communities[i]) for i in range(len(communities))],
                "member_node_ids": [
                    ";".join(map(str, communities[i])) for i in range(len(communities))
                ],
            }
        )

        df_meta.to_csv(
            results_path / "community_metagraph_constraint.csv",
            index=False,
        )

        # --------------------------------------------
        # Plot meta graph
        # --------------------------------------------
        layout_meta = G_comm.layout("fr")

        fig, ax = plt.subplots(figsize=(8, 6))
        ig.plot(
            G_comm,
            target=ax,
            layout=layout_meta,
            vertex_size=[len(communities[i]) * 10 for i in range(len(communities))],
            vertex_color=node_colors_meta,
            edge_width=[comm_edges[e.tuple] for e in G_comm.es],
        )

        plt.title("Community Meta-Graph (Constraint Binned)")
        plt.legend(handles=legend_patches_meta, title="Constraint", loc="best")
        plt.tight_layout()
        plt.savefig(figures_path / "community_metagraph.svg")
        plt.close()

        key_stats = {
            "length_of_communities": len(communities),
            "num_meta_edges": len(comm_edges),
            "num_original_nodes": G_ig.vcount(),
            "num_original_edges": G_ig.ecount(),
        }

        with open(results_path / "summary.txt", "w") as f:
            for k, v in key_stats.items():
                f.write(f"{k}: {v}\n")

        logger.success(f"Finished mode: {MODE}")


if __name__ == "__main__":
    main()
