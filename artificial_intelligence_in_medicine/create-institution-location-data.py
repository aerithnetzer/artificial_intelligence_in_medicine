import ast
import json
from pathlib import Path

from elasticsearch import Elasticsearch
from loguru import logger
import pandas as pd
from tqdm import tqdm
import typer

from artificial_intelligence_in_medicine.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR

app = typer.Typer()
es = Elasticsearch("http://localhost:9200", verify_certs=False)  # change if using remote
MODE = "GENE_EXPRESSION"


def get_first_author_affiliation(authors):
    print(authors)
    if isinstance(authors, str):
        try:
            authors = json.loads(authors)
        except json.JSONDecodeError:
            authors = ast.literal_eval(authors)
            print(authors)
            first_author = authors[0] if isinstance(authors, list) else authors
            if len(authors) == 0:
                return None
            else:
                if len(first_author["AffiliationInfo"]) != 0:
                    return first_author["AffiliationInfo"][0]["Affiliation"]


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = INTERIM_DATA_DIR / MODE / "features-flattened-citation-data.parquet",
    output_path: Path = INTERIM_DATA_DIR / MODE / "dataset_with_affiliations.parquet",
):
    df = pd.read_parquet(input_path)
    df["affiliation"] = df["authors"].apply(get_first_author_affiliation)
    print(df["affiliation"].isna().sum(), "missing affiliations")
    print(df["affiliation"].nunique(), "unique affiliations")
    print(df["affiliation"].head())
    df.to_parquet(output_path, index=False)


if __name__ == "__main__":
    app()
