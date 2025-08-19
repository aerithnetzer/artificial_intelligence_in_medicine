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
)

app = typer.Typer()

MODE = "GENE_EXPRESSION"


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = INTERIM_DATA_DIR / MODE / "features_with_ror.json",
    model_path: Path = MODELS_DIR / MODE / "citation_model.pkl",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Initializing graph for citation modeling.")
    G = nx.DiGraph()
    df = pd.read_json(features_path)
    # Check available columns
    logger.info(f"Available columns: {df.columns.tolist()}")

    # Use consistent column name for PMID
    pmid_column = "pmid" if "pmid" in df.columns else "_id"
    if pmid_column not in df.columns:
        raise KeyError(
            f"Neither 'pmid' nor '_id' found in DataFrame columns: {df.columns.tolist()}"
        )
    logger.info(f"Using '{pmid_column}' as PMID column")

    print(str(df["cited_by"]))
    logger.info("Now adding nodes to the graph.")
    for _, row in tqdm(df.iterrows(), desc="Adding nodes"):
        G.add_node(
            str(row[pmid_column]),
            title=row["title"],
            cited_by=row["cited_by"],
            mesh_headings=row["mesh_headings"],
            year=row["year"],
        )

    logger.info("Now adding edges to the graph.")
    pmid_set = set(df[pmid_column].astype(str))
    for _, row in tqdm(df.iterrows(), desc="Adding edges"):
        cited_by_list = row["cited_by"]
        if hasattr(cited_by_list, "__iter__") and not isinstance(cited_by_list, str):
            source_pmid = str(row[pmid_column])
            for cited_pmid in cited_by_list:
                cited_pmid_str = str(cited_pmid)
                if cited_pmid_str in pmid_set:
                    G.add_edge(source_pmid, cited_pmid_str)
        else:
            continue

    logger.info(
        "Saved graph. Number of nodes: {}, number of edges: {}".format(
            G.number_of_nodes(), G.number_of_edges()
        )
    )
    with open(model_path, "wb") as f:
        pickle.dump(G, f)
    # -----------------------------------------


if __name__ == "__main__":
    app()
