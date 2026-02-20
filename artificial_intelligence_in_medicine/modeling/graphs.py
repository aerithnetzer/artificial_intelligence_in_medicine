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
        # plot_semantic_graph(G, MODE, "embedding", top_k=4, min_degree=1)
        with open(FIGURES_DIR / MODE / "aggregated_communities_semantic.png", "wb") as f:
            pickle.dump(fig, f)
        all_keys = set()

        for _, attrs in G.nodes(data=True):
            all_keys.update(attrs.keys())

        # 2. Compute inflection point on the same graph
        inflection_point = calculate_inflection_point(G, MODE=MODE)

        # plot_cartographic_density(MODE)
        # 3. Community detection
        # IMPORTANT: treat the returned graph as the canonical graph going forward
        g: nx.Graph = community_detection(
            mode=MODE,
            G=G,
            inflection_point=inflection_point,
        )
        plot_normalized_articles_over_time(MODE)
        # _ = assign_countries_from_latlon(g)
        try:
            _ = plot_communities(g, MODE)
        except Exception as e:
            logger.critical(f"You are bad at coding. \n{e}")
        # 4. Sanity check: communities exist
        communities = nx.get_node_attributes(g, "community")

        # 5. Compute constraints ON THE SAME GRAPH that has year attributes
        constraints = nx.constraint(g)

        plot_constraints_by_community(g, constraints, MODE)
        plot_constraints_by_country(g, constraints, MODE)
        # 6. Plot constraints using THE SAME GRAPH
        plot_constraint(g, MODE, constraints)
        plot_constraints_over_time(g, constraints, MODE)
        plot_normalized_constraints_over_time(g, constraints, MODE)

        logger.info("Now plotting communities")

        # 7. Label communities (mutates g)
        g = assign_community_labels(MODE, g)

        # 8. All downstream plots use g
        plot_communities(g, MODE=MODE)
        plot_horizontal_timeline(g, MODE)
        plot_communities_vertical_barchart(
            G=g,
            MODE=MODE,
        )
        nx.write_graphml_xml(g, GRAPHS_DIR / MODE / "graph.gml")


if __name__ == "__main__":
    main()
