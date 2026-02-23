import networkx as nx
from artificial_intelligence_in_medicine.config import (
    DATA_DIR,
    FIGURES_DIR,
    INTERIM_DATA_DIR,
    RESULTS_DATA_DIR,
)
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from collections import defaultdict
import pandas as pd
from _graphs_helpers import initialize_graph

MODES = ["ARTIFICIAL_INTELLIGENCE", "GENE_EXPRESSION", "NULL"]


def compute_constraint_bins(G):
    constraint_dict = nx.constraint(G)

    bins = np.arange(0.0, 1.01, 0.2)
    bin_labels = [
        "0.0–0.2",
        "0.2–0.4",
        "0.4–0.6",
        "0.6–0.8",
        "0.8–1.0",
    ]

    node_bins = {}
    for node, value in constraint_dict.items():
        bin_index = np.digitize(value, bins, right=True) - 1
        bin_index = max(0, min(bin_index, len(bin_labels) - 1))
        node_bins[node] = bin_index

    cmap = plt.get_cmap("Set2")
    colors = [cmap(i) for i in range(len(bin_labels))]
    node_colors = [colors[node_bins[n]] for n in G.nodes()]

    legend_patches = [
        mpatches.Patch(color=colors[i], label=bin_labels[i]) for i in range(len(bin_labels))
    ]

    return constraint_dict, node_bins, node_colors, legend_patches, bin_labels


def main():
    for MODE in MODES:
        G: nx.DiGraph[str] = initialize_graph(mode=MODE)
        G.remove_nodes_from(list(nx.isolates(G)))

        figures_path = FIGURES_DIR / MODE
        figures_path.mkdir(parents=True, exist_ok=True)

        results_path = RESULTS_DATA_DIR / MODE
        results_path.mkdir(parents=True, exist_ok=True)

        constraint_dict, node_bins, node_colors, legend_patches, bin_labels = (
            compute_constraint_bins(G)
        )

        df_original = pd.DataFrame(
            {
                "node_id": list(constraint_dict.keys()),
                "constraint": list(constraint_dict.values()),
                "constraint_bin": [bin_labels[node_bins[n]] for n in constraint_dict.keys()],
            }
        )
        df_original.to_csv(results_path / "original_graph_constraints.csv", index=False)

        pos_original = nx.spring_layout(G, seed=42)

        plt.figure(figsize=(10, 8))
        nx.draw_networkx_nodes(G, pos_original, node_size=20, node_color=node_colors)
        nx.draw_networkx_edges(G, pos_original, alpha=0.3, width=0.5)
        plt.axis("off")
        plt.title("Original Graph (Constraint Binned)")
        plt.legend(handles=legend_patches, title="Constraint Score", loc="best")
        plt.tight_layout()
        plt.savefig(figures_path / "original_graph.svg")
        plt.close()

        communities = nx.community.louvain_communities(G)

        node_to_community = {}
        for i, community in enumerate(communities):
            for node in community:
                node_to_community[node] = i

        G_COMMUNITY = nx.Graph()

        for i, community in enumerate(communities):
            G_COMMUNITY.add_node(i, num_members=len(community), members=community)

        edge_weights = defaultdict(int)
        for u, v in G.edges():
            cu = node_to_community[u]
            cv = node_to_community[v]
            if cu != cv:
                edge = tuple(sorted((cu, cv)))
                edge_weights[edge] += 1

        for (cu, cv), weight in edge_weights.items():
            G_COMMUNITY.add_edge(cu, cv, weight=weight)

        constraint_meta, node_bins_meta, node_colors_meta, legend_patches_meta, bin_labels_meta = (
            compute_constraint_bins(G_COMMUNITY)
        )

        df_meta = pd.DataFrame(
            {
                "community_id": list(constraint_meta.keys()),
                "constraint": list(constraint_meta.values()),
                "constraint_bin": [
                    bin_labels_meta[node_bins_meta[n]] for n in constraint_meta.keys()
                ],
                "num_members": [
                    G_COMMUNITY.nodes[n]["num_members"] for n in constraint_meta.keys()
                ],
                "member_node_ids": [
                    ";".join(map(str, G_COMMUNITY.nodes[n]["members"]))
                    for n in constraint_meta.keys()
                ],
            }
        )
        df_meta.to_csv(results_path / "community_metagraph_constraints.csv", index=False)

        node_sizes = [G_COMMUNITY.nodes[n]["num_members"] * 30 for n in G_COMMUNITY.nodes()]

        edge_widths = [G_COMMUNITY[u][v]["weight"] for u, v in G_COMMUNITY.edges()]

        pos_meta = nx.spring_layout(G_COMMUNITY, seed=42)

        plt.figure(figsize=(8, 6))
        nx.draw_networkx_nodes(
            G_COMMUNITY,
            pos_meta,
            node_size=node_sizes,
            node_color=node_colors_meta,
        )
        nx.draw_networkx_edges(
            G_COMMUNITY,
            pos_meta,
            width=edge_widths,
            alpha=0.6,
        )
        nx.draw_networkx_labels(G_COMMUNITY, pos_meta, font_size=8)
        plt.axis("off")
        plt.title("Community Meta-Graph (Constraint Binned)")
        plt.legend(handles=legend_patches_meta, title="Constraint Score", loc="best")
        plt.tight_layout()
        plt.savefig(figures_path / "community_metagraph.svg")
        plt.close()

        key_stats = {
            "length_of_communities": len(communities),
            "num_meta_edges": G_COMMUNITY.number_of_edges(),
            "num_original_nodes": G.number_of_nodes(),
            "num_original_edges": G.number_of_edges(),
        }

        with open(results_path / "test.result", "w") as f:
            for k, v in key_stats.items():
                f.write(f"{k}: {v}\n")

        print(key_stats)


if __name__ == "__main__":
    main()
