import json
import os
from pathlib import Path

from Bio import Entrez
from loguru import logger
from tqdm import tqdm
import typer

from artificial_intelligence_in_medicine._dataset_helpers import (
    citation_data,
    flatten_data_to_parquet,
    get_affiliation_coordinate_data,
    get_citation_data,
    get_institution_coordinates,
    get_institution_location_data,
)
from artificial_intelligence_in_medicine.config import (
    INTERIM_DATA_DIR,
    RAW_DATA_DIR,
)

app = typer.Typer()

# This file takes as input a list of PMIDs, fetches their metadata from PubMed,
# enriches it with citation data from the iCite API, and matches affiliations to ROR entries.
# Final enriched data is saved as JSON files.

Entrez.email = os.getenv("NCBI_EMAIL")
Entrez.api_key = os.getenv("NCBI_API_KEY")

batch_size = 10_000


@app.command()
def main(
    MODE: str = "ARTIFICIAL_INTELLIGENCE",
    input_path: Path = typer.Option(None, help="Override input PMIDs path"),
    output_path: Path = typer.Option(None, help="Override output directory"),
):
    logger.info("Processing dataset...")
    input_path = input_path or RAW_DATA_DIR / MODE / "pmids.txt"
    output_path = output_path or INTERIM_DATA_DIR / MODE
    with open(input_path, "r") as file:  # Load PMIDS from text file
        pmids = file.read().splitlines()

    logger.info(f"Fetching data for {len(pmids)} PMIDs...")
    logger.info(f"Output will be saved to {output_path}")
    logger.info(f"Batch size: {batch_size}")

    batch_number = 0
    for i in tqdm(range(0, len(pmids), batch_size), desc="Fetching data in batches"):
        batch = pmids[i : i + batch_size]
        if os.path.exists(output_path / f"batch_{batch_number:05d}.json"):
            logger.info(
                f"{output_path / f'batch_{batch_number:05d}.json'} already exists, skipping..."
            )
            batch_number += 1
            continue
        try:
            handle = Entrez.efetch(db="pubmed", id=batch, retmode="xml")
            records = Entrez.read(handle)
            with open(output_path / f"batch_{batch_number:05d}.json", "w") as f:
                f.write(json.dumps(records, indent=2))
        except Exception as e:
            print(f"Error fetching batch {batch_number}: {e}")
        batch_number += 1
    citation_data(
        input_path=f"{str(INTERIM_DATA_DIR)}/{MODE}/batch*.json",
        output_path=f"{str(INTERIM_DATA_DIR)}/{MODE}/dataset_with_citation_data.json",
    )
    get_affiliation_coordinate_data(mode=MODE)
    get_citation_data(mode=MODE)
    flatten_data_to_parquet(mode=MODE)
    get_institution_location_data(mode=MODE)
    get_institution_coordinates(mode=MODE)


if __name__ == "__main__":
    app()
