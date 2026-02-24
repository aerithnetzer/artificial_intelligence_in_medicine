from loguru import logger
import cudf
import cugraph
from artificial_intelligence_in_medicine.config import DATA_DIR, FIGURES_DIR, RESULTS_DATA_DIR
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from collections import defaultdict
from _graphs_helpers import initialize_graph
import networkx as nx

MODES = ["ARTIFICIAL_INTELLIGENCE", "GENE_EXPRESSION", "NULL"]


def compute_betweenness_bins_cugraph(G_cu, batch_size=500):
    logger.info("Computing betweenness centrality on GPU (batched)...")
    all_vertices = G_cu.nodes().to_pandas().tolist()

    accumulated = {v: 0.0 for v in all_vertices}
    num_batches = -(-len(all_vertices) // batch_size)

    for i in range(0, len(all_vertices), batch_size):
        batch = all_vertices[i:i + batch_size]
        logger.info(f"Processing batch {i // batch_size + 1} / {num_batches}")
        bc_df = cugraph.betweenness_centrality(G_cu, normalized=True, k=batch)
        for vertex, score in zip(bc_df["vertex"].to_pandas(), bc_df["betweenness_centrality"].to_pandas()):
            accumulated[vertex] += score

    bc_dict = {v: score / num_batches for v, score in accumulated.items()}
    logger.success("Betweenness centrality computed.")

    bins = np.arange(0.0, 1.01, 0.2)
    bin_labels = ["0.0–0.2", "0.2–0.4", "0.4–0.6", "0.6–0.8", "0.8–1.0"]

    node_bins = {}
    for node in bc_dict.keys():
        value = bc_dict[node]
        bin_index = np.digitize(value, bins, right=True) - 1
        bin_index = max(0, min(bin_index, len(bin_labels) - 1))
        node_bins[node] = bin_index

    legend_patches = [
        mpatches.Patch(color=plt.get_cmap("Set2")(i), label=bin_labels[i])
        for i in range(len(bin_labels))
    ]

    return bc_dict, node_bins, legend_patches, bin_labels


def compute_force_atlas2_positions(G_cu):
    fa2_df = cugraph.force_atlas2(G_cu).to_pandas()
    pos = {row.vertex: (row.x, row.y) for row in fa2_df.itertuples(index=False)}
    return pos


def main():
    for MODE in MODES:
        G_nx = initialize_graph(MODE)
        G_nx.remove_nodes_from(list(nx.isolates(G_nx)))

        # Build explicit integer mapping to handle string node IDs
        nodes_list = list(G_nx.nodes())
        node_to_int = {n: i for i, n in enumerate(nodes_list)}
        int_to_node = {i: n for i, n in enumerate(nodes_list)}

        df_edges = cudf.DataFrame({
            "src": [node_to_int[u] for u, v in G_nx.edges()],
            "dst": [node_to_int[v] for u, v in G_nx.edges()],
        })
        G_cu = cugraph.Graph()
        G_cu.from_cudf_edgelist(df_edges, source="src", destination="dst", renumber=False)

        figures_path = FIGURES_DIR / MODE
        figures_path.mkdir(parents=True, exist_ok=True)

        results_path = RESULTS_DATA_DIR / MODE
        results_path.mkdir(parents=True, exist_ok=True)

        centrality_dict, node_bins, legend_patches, bin_labels = (
            compute_betweenness_bins_cugraph(G_cu)
        )

        # Remap centrality results back to original node IDs
        centrality_dict_orig = {int_to_node[k]: v for k, v in centrality_dict.items()}
        node_bins_orig = {int_to_node[k]: v for k, v in node_bins.items()}

        df_original = cudf.DataFrame(
            {
                "node_id": list(centrality_dict_orig.keys()),
                "betweenness_centrality": list(centrality_dict_orig.values()),
                "centrality_bin": [bin_labels[node_bins_orig[n]] for n in centrality_dict_orig.keys()],
            }
        )
        df_original.to_csv(results_path / "original_graph_betweenness.csv", index=False)

        # GPU ForceAtlas2 layout, remapped to original node IDs
        pos_original_cu = compute_force_atlas2_positions(G_cu)
        pos_original = {int_to_node[k]: v for k, v in pos_original_cu.items()}

        # Build node colors ordered by G_nx.nodes()
        cmap = plt.get_cmap("Set2")
        colors = [cmap(i) for i in range(len(bin_labels))]
        node_colors = [colors[node_bins_orig[n]] for n in G_nx.nodes()]

        plt.figure(figsize=(10, 8))
        nx.draw_networkx_nodes(G_nx, pos_original, node_size=20, node_color=node_colors)
        nx.draw_networkx_edges(G_nx, pos_original, alpha=0.3, width=0.5)
        plt.axis("off")
        plt.title("Original Graph (Betweenness Binned)")
        plt.legend(handles=legend_patches, title="Betweenness Centrality", loc="best")
        plt.tight_layout()
        plt.savefig(figures_path / "original_graph.svg")
        plt.close()

        # Louvain community detection (GPU)
        logger.info("Computing Louvain communities...")
        parts, _modularity = cugraph.louvain(G_cu)
        logger.success("Louvain communities computed.")

        # Map integer nodes to communities, then remap to original node IDs
        int_node_to_community = dict(zip(parts["vertex"].to_pandas(), parts["partition"].to_pandas()))
        node_to_community = {int_to_node[k]: v for k, v in int_node_to_community.items()}

        # Build community metagraph
        communities = defaultdict(list)
        for node, comm in node_to_community.items():
            communities[comm].append(node)

        comm_edges = defaultdict(int)
        for u, v in G_nx.edges():
            cu = node_to_community[u]
            cv = node_to_community[v]
            if cu != cv:
                edge = tuple(sorted((cu, cv)))
                comm_edges[edge] += 1

        # Build explicit integer mapping for community metagraph
        comm_nodes = list(communities.keys())
        comm_to_int = {c: i for i, c in enumerate(comm_nodes)}
        int_to_comm = {i: c for i, c in enumerate(comm_nodes)}

        df_comm_edges = cudf.DataFrame(
            {
                "src": [comm_to_int[e[0]] for e in comm_edges.keys()],
                "dst": [comm_to_int[e[1]] for e in comm_edges.keys()],
                "weight": list(comm_edges.values()),
            }
        )
        G_COMMUNITY = cugraph.Graph(directed=False)
        G_COMMUNITY.from_cudf_edgelist(
            df_comm_edges, source="src", destination="dst", edge_attr="weight", renumber=False
        )

        centrality_meta, node_bins_meta, legend_patches_meta, bin_labels_meta = (
            compute_betweenness_bins_cugraph(G_COMMUNITY)
        )

        # Remap back to original community IDs
        centrality_meta_orig = {int_to_comm[k]: v for k, v in centrality_meta.items()}
        node_bins_meta_orig = {int_to_comm[k]: v for k, v in node_bins_meta.items()}

        df_meta = cudf.DataFrame(
            {
                "community_id": list(centrality_meta_orig.keys()),
                "betweenness_centrality": list(centrality_meta_orig.values()),
                "centrality_bin": [
                    bin_labels_meta[node_bins_meta_orig[n]] for n in centrality_meta_orig.keys()
                ],
                "num_members": [len(communities[n]) for n in centrality_meta_orig.keys()],
                "member_node_ids": [
                    ";".join(map(str, communities[n])) for n in centrality_meta_orig.keys()
                ],
            }
        )
        df_meta.to_csv(results_path / "community_metagraph_betweenness.csv", index=False)

        # Community visualization, remapped to original community IDs
        pos_meta_cu = compute_force_atlas2_positions(G_COMMUNITY)
        pos_meta = {int_to_comm[k]: v for k, v in pos_meta_cu.items()}

        # Build community NetworkX graph
        G_COMM_NX = nx.Graph()
        for comm_id in comm_nodes:
            G_COMM_NX.add_node(comm_id)
        for e, w in comm_edges.items():
            G_COMM_NX.add_edge(e[0], e[1], weight=w)

        # Build node colors ordered by G_COMM_NX.nodes()
        cmap_meta = plt.get_cmap("Set2")
        colors_meta = [cmap_meta(i) for i in range(len(bin_labels_meta))]
        node_colors_meta = [colors_meta[node_bins_meta_orig[n]] for n in G_COMM_NX.nodes()]

        node_sizes = [len(communities[n]) * 30 for n in G_COMM_NX.nodes()]
        edge_widths = [comm_edges[tuple(sorted((u, v)))] for u, v in G_COMM_NX.edges()]

        plt.figure(figsize=(8, 6))
        nx.draw_networkx_nodes(
            G_COMM_NX, pos_meta, node_size=node_sizes, node_color=node_colors_meta
        )
        nx.draw_networkx_edges(G_COMM_NX, pos_meta, width=edge_widths, alpha=0.6)
        nx.draw_networkx_labels(G_COMM_NX, pos_meta, font_size=8)
        plt.axis("off")
        plt.title("Community Meta-Graph (Betweenness Binned)")
        plt.legend(handles=legend_patches_meta, title="Betweenness Centrality", loc="best")
        plt.tight_layout()
        plt.savefig(figures_path / "community_metagraph.svg")
        plt.close()

        key_stats = {
            "length_of_communities": len(communities),
            "num_meta_edges": len(comm_edges),
            "num_original_nodes": len(G_nx.nodes()),
            "num_original_edges": len(G_nx.edges()),
        }
        with open(results_path / "test.result", "w") as f:
            for k, v in key_stats.items():
                f.write(f"{k}: {v}\n")

        print(key_stats)


if __name__ == "__main__":
    main()
