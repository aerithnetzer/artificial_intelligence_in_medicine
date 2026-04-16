"""
Graph analysis orchestrator.

For each MODE (AI, Gene Expression, NULL): initializes the citation graph,
runs community detection, computes Burt's constraint, generates embeddings,
and produces all per-mode visualizations.
"""

from loguru import logger
import networkx as nx
import typer

from artificial_intelligence_in_medicine.config import FIGURES_DIR, GRAPHS_DIR
from artificial_intelligence_in_medicine.modeling._graphs_helpers import (
    assign_community_labels,
    community_detection,
    generate_embeddings,
    initialize_graph,
    plot_constraints_by_community,
    plot_constraints_by_country,
    plot_constraints_over_time,
    plot_normalized_constraints_over_time,
)
from artificial_intelligence_in_medicine.visualizations import (
    funding_agency,
    plot_cartographic_density,
    plot_communities,
    plot_communities_vertical_barchart,
    plot_constraint,
    plot_horizontal_timeline,
    plot_normalized_articles_over_time,
    visualize_communities,
)

app = typer.Typer()

MODES = ["NULL", "ARTIFICIAL_INTELLIGENCE", "GENE_EXPRESSION"]


def main():
    for MODE in MODES:
        logger.info(f"=== Processing {MODE} ===")

        # 1. Initialize base graph
        G: nx.DiGraph = initialize_graph(MODE)
        logger.info(f"Graph nodes: {len(G.nodes())}")

        # 2. Visualize raw community structure
        try:
            visualize_communities(
                G,
                mode="citation",
                output_path=str(FIGURES_DIR / MODE / "aggregated_communities_citation.png"),
            )
        except Exception as e:
            logger.warning(f"Could not visualize citation communities: {e}")

        # 3. Remove low-degree nodes
        nodes_to_remove = [n for n, d in G.degree() if d < 2]
        G.remove_nodes_from(nodes_to_remove)
        logger.info(f"After removing low-degree nodes: {len(G.nodes())}")

        # 4. Generate semantic embeddings
        try:
            G_sem = generate_embeddings(G, text_attr="title")
            visualize_communities(
                G_sem,
                mode="semantic",
                output_path=str(FIGURES_DIR / MODE / "aggregated_communities_semantic.png"),
            )
        except Exception as e:
            logger.warning(f"Could not generate embeddings: {e}")

        # 5. Per-mode temporal and geographic plots
        plot_normalized_articles_over_time(MODE)
        try:
            plot_cartographic_density(MODE)
        except Exception as e:
            logger.warning(f"Could not plot cartographic density: {e}")

        # 6. Community detection
        g: nx.Graph = community_detection(mode=MODE, G=G, inflection_point=1)

        # 7. Plot communities
        try:
            plot_communities(g, MODE)
        except Exception as e:
            logger.error(f"Could not plot communities: {e}")

        # 8. Compute and plot constraints
        constraints = nx.constraint(g)
        plot_constraints_by_community(g, constraints, MODE)
        plot_constraints_by_country(g, constraints, MODE)
        plot_constraint(g, MODE, constraints)
        plot_constraints_over_time(g, constraints, MODE)
        plot_normalized_constraints_over_time(g, constraints, MODE)

        # 9. Label communities and generate timeline plots
        g = assign_community_labels(MODE, g)
        plot_communities(g, mode=MODE)
        plot_horizontal_timeline(g, MODE)
        plot_communities_vertical_barchart(g, MODE)

        # 10. Funding
        try:
            funding_agency(MODE, metric="citations")
            funding_agency(MODE, metric="papers")
        except Exception as e:
            logger.warning(f"Could not plot funding for {MODE}: {e}")

        # 11. Save graph
        graphs_dir = GRAPHS_DIR / MODE
        graphs_dir.mkdir(parents=True, exist_ok=True)
        nx.write_graphml_xml(g, graphs_dir / "graph.gml")
        logger.success(f"Finished {MODE}")


if __name__ == "__main__":
    main()
