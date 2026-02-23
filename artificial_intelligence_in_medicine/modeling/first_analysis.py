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


def compute_betweenness_bins_cugraph(G_cu, nodes, batch_size=500):
    logger.info("Computing betweenness centrality on GPU (batched)...")
    all_vertices = G_cu.nodes().to_pandas().tolist()

    accumulated = {v: 0.0 for v in all_vertices}
    num_batches = -(-len(all_vertices) // batch_size)

    for i in range(0, len(all_vertices), batch_size):
        batch = all_vertices[i : i + batch_size]
        logger.info(f"Processing batch {i // batch_size + 1} / {num_batches}")
        bc_df = cugraph.betweenness_centrality(G_cu, normalized=True, k=batch)
        for vertex, score in zip(
            bc_df["vertex"].to_pandas(), bc_df["betweenness_centrality"].to_pandas()
        ):
            accumulated[vertex] += score

    bc_dict = {v: score / num_batches for v, score in accumulated.items()}
    logger.success("Betweenness centrality computed.")

    bins = np.arange(0.0, 1.01, 0.2)
    bin_labels = ["0.0–0.2", "0.2–0.4", "0.4–0.6", "0.6–0.8", "0.8–1.0"]

    node_bins = {}
    for node in bc_dict.keys():
        value = bc_dict.get(node, 0.0)
        bin_index = np.digitize(value, bins, right=True) - 1
        bin_index = max(0, min(bin_index, len(bin_labels) - 1))
        node_bins[node] = bin_index

    cmap = plt.get_cmap("Set2")
    colors = [cmap(i) for i in range(len(bin_labels))]

    # Only include nodes that exist in node_bins (i.e. were present in cuGraph)
    cu_vertex_set = set(all_vertices)
    valid_nodes = [n for n in nodes if n in cu_vertex_set]
    if len(valid_nodes) < len(nodes):
        logger.warning(
            f"{len(nodes) - len(valid_nodes)} nodes from NetworkX were not found in cuGraph "
            f"and will be skipped for coloring."
        )

    node_colors = [colors[node_bins.get(n, 0)] for n in valid_nodes]

    legend_patches = [
        mpatches.Patch(color=colors[i], label=bin_labels[i]) for i in range(len(bin_labels))
    ]

    return bc_dict, node_bins, node_colors, legend_patches, bin_labels, valid_nodes


def compute_force_atlas2_positions(G_cu, iterations=2000):
    """
    Compute ForceAtlas2 layout for cuGraph.
    Returns a dictionary {node_id: (x, y)} for plotting in matplotlib.
    """
    fa2_df = cugraph.force_atlas2(G_cu, iterations=iterations)
    pos = {
        row["vertex"]: (row["x"], row["y"]) for row in fa2_df.to_pandas().itertuples(index=False)
    }
    return pos


def main():
    for MODE in MODES:
        G_nx = initialize_graph(MODE)
        G_nx.remove_nodes_from(list(nx.isolates(G_nx)))

        # Convert to cuGraph
        df_edges = cudf.DataFrame(
            {"src": [int(u) for u, v in G_nx.edges()], "dst": [int(v) for u, v in G_nx.edges()]}
        )
        G_cu = cugraph.Graph()
        G_cu.from_cudf_edgelist(df_edges, source="src", destination="dst", renumber=True)

        figures_path = FIGURES_DIR / MODE
        figures_path.mkdir(parents=True, exist_ok=True)

        results_path = RESULTS_DATA_DIR / MODE
        results_path.mkdir(parents=True, exist_ok=True)

        # Cast nodes to int to match cuGraph's renumbered integer vertices
        nodes = [int(n) for n in G_nx.nodes()]

        centrality_dict, node_bins, node_colors, legend_patches, bin_labels, valid_nodes = (
            compute_betweenness_bins_cugraph(G_cu, nodes)
        )

        # Save original node betweenness
        df_original = cudf.DataFrame(
            {
                "node_id": list(centrality_dict.keys()),
                "betweenness_centrality": list(centrality_dict.values()),
                "centrality_bin": [bin_labels[node_bins[n]] for n in centrality_dict.keys()],
            }
        )
        df_original.to_csv(results_path / "original_graph_betweenness.csv", index=False)

        # GPU ForceAtlas2 layout
        pos_original = compute_force_atlas2_positions(G_cu)

        # Build a NetworkX subgraph with only valid_nodes for drawing
        G_nx_valid = G_nx.subgraph([str(n) for n in valid_nodes]).copy()

        plt.figure(figsize=(10, 8))
        nx.draw_networkx_nodes(G_nx_valid, pos_original, node_size=20, node_color=node_colors)
        nx.draw_networkx_edges(G_nx_valid, pos_original, alpha=0.3, width=0.5)
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

        # Map nodes to communities
        node_to_community = dict(zip(parts["vertex"].to_pandas(), parts["partition"].to_pandas()))

        # Build community metagraph
        communities = defaultdict(list)
        for node, comm in node_to_community.items():
            communities[comm].append(node)

        G_COMMUNITY = cugraph.Graph(directed=False)
        comm_edges = defaultdict(int)
        for u, v in G_nx.edges():
            cu = node_to_community.get(int(u))
            cv = node_to_community.get(int(v))
            if cu is None or cv is None:
                continue
            if cu != cv:
                edge = tuple(sorted((cu, cv)))
                comm_edges[edge] += 1

        df_comm_edges = cudf.DataFrame(
            {
                "src": [e[0] for e in comm_edges.keys()],
                "dst": [e[1] for e in comm_edges.keys()],
                "weight": list(comm_edges.values()),
            }
        )
        G_COMMUNITY.from_cudf_edgelist(
            df_comm_edges, source="src", destination="dst", edge_attr="weight"
        )

        # Compute betweenness centrality on community metagraph
        comm_nodes = list(communities.keys())
        (
            centrality_meta,
            node_bins_meta,
            node_colors_meta,
            legend_patches_meta,
            bin_labels_meta,
            valid_comm_nodes,
        ) = compute_betweenness_bins_cugraph(G_COMMUNITY, comm_nodes)

        # Save community metagraph
        df_meta = cudf.DataFrame(
            {
                "community_id": list(centrality_meta.keys()),
                "betweenness_centrality": list(centrality_meta.values()),
                "centrality_bin": [
                    bin_labels_meta[node_bins_meta[n]] for n in centrality_meta.keys()
                ],
                "num_members": [len(communities[n]) for n in centrality_meta.keys()],
                "member_node_ids": [
                    ";".join(map(str, communities[n])) for n in centrality_meta.keys()
                ],
            }
        )
        df_meta.to_csv(results_path / "community_metagraph_betweenness.csv", index=False)

        # Community visualization with GPU layout
        pos_meta = compute_force_atlas2_positions(G_COMMUNITY)

        node_sizes = [len(communities[n]) * 30 for n in valid_comm_nodes]
        edge_widths = [
            comm_edges.get((e[0], e[1]), comm_edges.get((e[1], e[0]), 1))
            for e in df_comm_edges.to_pandas()[["src", "dst"]].itertuples(index=False, name=None)
        ]

        # Build a small NetworkX graph for drawing
        G_COMM_NX = nx.Graph()
        for comm_id in valid_comm_nodes:
            G_COMM_NX.add_node(comm_id)
        for u, v, w in df_comm_edges.to_pandas().itertuples(index=False):
            G_COMM_NX.add_edge(u, v, weight=w)

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

        # Key stats
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
