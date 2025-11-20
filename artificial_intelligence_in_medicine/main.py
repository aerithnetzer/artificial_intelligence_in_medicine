from pathlib import Path
from loguru import logger
import pandas as pd
import igraph as ig

from artificial_intelligence_in_medicine.config import INTERIM_DATA_DIR

MODE = "ARTIFICIAL_INTELLIGENCE"


def initialize_graph(feature_path: Path = INTERIM_DATA_DIR / MODE / "features_with_ror.json"):
    df = pd.read_json(feature_path)
    df = df[
        [
            "pmid",
            "cited_by",
            "title",
            "year",
            "mesh_headings",
            "author_list",
            "grant_list",
            "affiliation",
            "matched_name",
            "matched_ror_id",
            "matched_lat",
            "matched_lon",
            "matched_country",
            "matched_raw_text",
        ]
    ]
    logger.info(df.columns)

    # Get set of valid PMIDs (only those in the dataframe)
    valid_pmids = set(df["pmid"])

    # Create edge list: only include edges where both vertices exist
    edges = []
    skipped_edges = 0
    for _, row in df.iterrows():
        cited_pmid = row["pmid"]
        citing_pmids = row["cited_by"]
        if citing_pmids:  # Check if the list is not None/empty
            for citing_pmid in citing_pmids:
                # Only add edge if citing PMID is also in our dataframe
                if citing_pmid in valid_pmids:
                    edges.append((citing_pmid, cited_pmid))
                else:
                    skipped_edges += 1

    logger.info(
        f"Created {len(edges)} edges, skipped {skipped_edges} edges with out-of-dataframe vertices"
    )

    # Create graph with only PMIDs from the dataframe
    g = ig.Graph(directed=True)
    g.add_vertices(list(valid_pmids))
    g.add_edges(edges)

    return g


def main():
    initialize_graph()


if __name__ == "__main__":
    main()
