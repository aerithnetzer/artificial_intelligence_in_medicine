from loguru import logger
import typer
import pickle
import artificial_intelligence_in_medicine
from artificial_intelligence_in_medicine._plots_helpers import (
    plot_communities,
    plot_semantic_graph,
    plot_communities_vertical_barchart,
    plot_horizontal_timeline,
    plot_constraint,
    plot_cartographic_density,
    plot_normalized_articles_over_time,
    visualize_graph,
    visualize_communities,
)
from artificial_intelligence_in_medicine.config import FIGURES_DIR, GRAPHS_DIR
from artificial_intelligence_in_medicine.modeling._graphs_helpers import (
    assign_community_labels,
    assign_countries_from_latlon,
    plot_constraints_by_community,
    plot_constraints_by_country,
    assign_countries_from_latlon,
    calculate_constraint,
    plot_normalized_constraints_over_time,
    community_detection,
    initialize_graph,
    calculate_inflection_point,
    plot_constraints_over_time,
    generate_embeddings,
)

app = typer.Typer()

MODES = ["NULL", "ARTIFICIAL_INTELLIGENCE", "GENE_EXPRESSION"]


def main():
    import networkx as nx

    for MODE in MODES:
        # 1. Initialize base graph (contains year, metadata, etc.)
        G: nx.DiGraph[str] = initialize_graph(MODE)
        logger.info(f"Length of nodes: {len(G.nodes())}")

        # Find nodes with degree < 2
        nodes_to_remove = [n for n, d in G.degree() if d < 2]

        # Remove in place
        G.remove_nodes_from(nodes_to_remove)

        logger.info(f"Length after removing low-degree nodes: {len(G.nodes())}")
        fig, _ = visualize_communities(G, mode="citation")
        with open(FIGURES_DIR / MODE / "aggregated_communities_citation.png", "wb") as f:
            pickle.dump(fig, f)
        G_sem = generate_embeddings(
            G,
            text_attr="title",
        )
        fig, _ = visualize_communities(G_sem, mode="semantic")
        with open(FIGURES_DIR / MODE / "aggregated_communities_semantic.png", "wb") as f:
            pickle.dump(fig, f)
        all_keys = set()

if __name__ == "__main__":
    main()
