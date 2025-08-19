from pathlib import Path
import pickle

from loguru import logger
import networkx as nx
import pandas as pd
from tqdm import tqdm
import typer

from artificial_intelligence_in_medicine.config import (
    INTERIM_DATA_DIR,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
)

app = typer.Typer()

MODE = "GENE_EXPRESSION"


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    graph: Path = MODELS_DIR / MODE / "citation_model.pkl",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Initializing graph for citation modeling.")
    with open(graph, "rb") as f:
        G = pickle.load(f)

    logger.info("Filtering out nodes with degree 0.")
    nodes_to_remove = [node for node, degree in dict(G.degree()).items() if degree == 0]
    G.remove_nodes_from(nodes_to_remove)
    logger.info(f"Removed {len(nodes_to_remove)} nodes with degree 0.")

    logger.info("Calculating degree centrality for all nodes.")
    centrality = nx.degree_centrality(G)

    logger.info("Finding top 5 central nodes.")
    sorted_nodes = sorted(centrality.items(), key=lambda item: item[1], reverse=True)

    top_5_nodes = sorted_nodes[:5]

    logger.info("Top 5 most central nodes:")
    for node_id, score in top_5_nodes:
        title = G.nodes[node_id].get("title", "No Title")
        logger.info(f"  - Title: {title}, Centrality: {score:.4f}")


if __name__ == "__main__":
    app()
