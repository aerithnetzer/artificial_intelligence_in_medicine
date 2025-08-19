import json
from pathlib import Path

from loguru import logger
import pandas as pd
from tqdm import tqdm
import typer

from artificial_intelligence_in_medicine.config import (
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
)

app = typer.Typer()

MODE = "GENE_EXPRESSION"  # Change this to "ARTIFICIAL_INTELLIGENCE" if needed


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = INTERIM_DATA_DIR / f"{MODE}/dataset_with_citation_data.json",
    output_path: Path = INTERIM_DATA_DIR / MODE / "features-flattened-citation-data.parquet",
    # -----------------------------------------
):
    logger.info("Flattening dataset to CSV for easier analysis...")
    X = json.load(input_path.open("r"))
    print(X[0].keys())
    pmids = []
    titles = []
    authors = []
    cited_by = []
    year = []
    doi = []
    mesh_headings = []
    for x in tqdm(X):
        pmids.append(x["pmid"])
        titles.append(x["title"])
        try:
            cited_by.append(x["cited_by"])
        except KeyError:
            cited_by.append(None)
        authors.append(x["author_list"])
        year.append(x["year"])
        doi.append(x["doi"])
        mesh_headings.append(x["mesh_headings"])
    pd.DataFrame(
        {
            "pmid": pmids,
            "title": titles,
            "authors": authors,
            "cited_by": cited_by,
            "year": year,
            "doi": doi,
            "mesh_headings": mesh_headings,
        }
    ).to_parquet(output_path, index=False)


if __name__ == "__main__":
    app()
