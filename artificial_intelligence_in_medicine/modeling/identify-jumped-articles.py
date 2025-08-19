from pathlib import Path
import pickle

from loguru import logger
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import typer

from artificial_intelligence_in_medicine.config import (
    FIGURES_DIR,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
)

MODE = "ARTIFICIAL_INTELLIGENCE"
app = typer.Typer()


def jaccard_distance(set1, set2):
    """Calculates Jaccard distance between two sets."""
    set1 = set(set1)
    set2 = set(set2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    if union == 0:
        return 0.0
    return 1 - intersection / union
    # {1, 2, 3}, {1, 2, 3, 4} = 1 - 3/4 = .25, very similar


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    graph_path: Path = MODELS_DIR / MODE / "citation_model.pkl",
    output_path: Path = PROCESSED_DATA_DIR / MODE / "interdisciplinary_nodes.csv",
    plot_path: Path = FIGURES_DIR / MODE / "interdisciplinary_nodes_plot.png",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Loading graph.")
    with open(graph_path, "rb") as f:
        G = pickle.load(f)

    logger.info("Calculating interdisciplinary scores for each node.")
    results = []
    for node in tqdm(G.nodes(), desc="Processing nodes"):
        node_attrs = G.nodes[node]
        node_mesh = node_attrs.get("mesh_headings")

        successors = list(G.successors(node))
        if len(successors) <= 1:
            continue

        cited_by_mesh_union = set()
        for succ in successors:
            succ_attrs = G.nodes[succ]
            succ_mesh = succ_attrs.get("mesh_headings")
            cited_by_mesh_union.update(succ_mesh)
        node_attrs.get("year")
        distance = jaccard_distance(node_mesh, cited_by_mesh_union)
        results.append(
            {
                "pmid": node,
                "title": node_attrs.get("title"),
                "pub_date": node_attrs.get("year"),
                "jaccard_distance": distance,
                "num_cited_by": len(successors),
            }
        )
    max_successors = (
        max([len(list(G.successors(node))) for node in G.nodes()])
        if G.number_of_nodes() > 0
        else 0
    )
    print(f"Maximum successors: {max_successors}")
    max_cited_by = max([r["num_cited_by"] for r in results]) if results else 0
    print(f"Maximum num_cited_by: {max_cited_by}")
    logger.info("Saving results.")
    df_results = pd.DataFrame(results)
    print(df_results["pub_date"].head())
    df_results = df_results.sort_values(by="jaccard_distance", ascending=False)
    df_results.to_csv(output_path, index=False)

    logger.info(f"Saved top interdisciplinary nodes to {output_path}")

    # Create and save the plot
    logger.info("Generating plot.")
    df_results["pub_date"] = pd.to_datetime(df_results["pub_date"], errors="coerce", format="%Y")
    df_filtered = df_results.dropna(subset=["pub_date", "jaccard_distance"])

    _, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(
        df_filtered["pub_date"],
        df_filtered["jaccard_distance"],
        s=df_filtered["num_cited_by"],  # Node radius based on number of citations
        alpha=0.6,
        edgecolors="w",
        linewidth=0.5,
    )
    ax.set_title("Interdisciplinary Articles: Jaccard Distance vs. Year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Jaccard Distance")
    ax.grid(True)

    # Create a legend for bubble sizes
    handles, labels = scatter.legend_elements(prop="sizes", alpha=0.6, num=5)
    ax.legend(
        handles,
        labels,
        loc="upper left",
        title="Citations",
        bbox_to_anchor=(1.04, 1),
        frameon=False,
        handletextpad=1.5,
        labelspacing=1.5,
    )

    plt.subplots_adjust(right=0.85)
    plt.savefig(plot_path, dpi=300)
    logger.info(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    app()
