from loguru import logger
import typer

import artificial_intelligence_in_medicine
from artificial_intelligence_in_medicine._plots_helpers import (
    plot_communities,
    plot_semantic_graph,
    plot_communities_vertical_barchart,
    plot_horizontal_timeline,
    plot_constraint,
    plot_cartographic_density,
    plot_normalized_articles_over_time,
)
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

MODES = ["ARTIFICIAL_INTELLIGENCE", "GENE_EXPRESSION", "NULL"]


def main():
    import networkx as nx

    for MODE in MODES:
        # 1. Initialize base graph (contains year, metadata, etc.)
        G: nx.DiGraph[str] = initialize_graph(MODE)
        G = generate_embeddings(G, text_attr= "title",)
        if MODE == "GENE_EXPRESSION":
            plot_semantic_graph(G, MODE, "embedding", min_degree=10)
        else:
            plot_semantic_graph(G, MODE, "embedding")
        all_keys = set()

        for _, attrs in G.nodes(data=True):
            all_keys.update(attrs.keys())

        # 2. Compute inflection point on the same graph
        inflection_point = calculate_inflection_point(G, MODE=MODE)

        plot_cartographic_density(MODE)
        # 3. Community detection
        # IMPORTANT: treat the returned graph as the canonical graph going forward
        g: nx.Graph = community_detection(
            mode=MODE,
            G=G,
            inflection_point=inflection_point,
        )
        plot_normalized_articles_over_time(MODE)
        _ = assign_countries_from_latlon(g)
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


if __name__ == "__main__":
    main()
