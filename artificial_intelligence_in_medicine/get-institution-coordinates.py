from pathlib import Path

from elasticsearch import Elasticsearch
from loguru import logger
import pandas as pd
from tqdm import tqdm
import typer

from artificial_intelligence_in_medicine.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR

app = typer.Typer()
MODE = "GENE_EXPRESSION"
es = Elasticsearch("http://localhost:9200", verify_certs=False)  # change if using remote


def search_ror(affiliation_text):
    """
    Search for an organization in the ROR database based on affiliation text.
    Returns the best match with relevant details or None if no match found.
    """
    if not affiliation_text:
        return None

    try:
        res = es.search(
            index="ror",
            body={
                "query": {
                    "bool": {
                        "must": [
                            {
                                "multi_match": {
                                    "query": affiliation_text,
                                    "fields": ["names.types.ror_display", "locations"],
                                    "type": "best_fields",
                                }
                            }
                        ],
                        "should": [
                            {
                                "match_phrase": {
                                    "names.types.ror_display": {
                                        "query": affiliation_text,
                                        "slop": 5,
                                        "boost": 4,
                                    }
                                }
                            },
                            {"match": {"locations": {"query": affiliation_text, "boost": 2}}},
                        ],
                    }
                }
            },
        )

        matches = res.get("hits", {}).get("hits", [])
        if matches:
            top_match = matches[0]["_source"]

            # Extract organization name and coordinates
            matched_org = {
                "matched_name": top_match.get("names.types.ror_display", ""),
                "ror_id": top_match.get("id", ""),
                "lat": top_match.get("locations.geonames_details.lat"),
                "lon": top_match.get("locations.geonames_details.lng"),
                "country": top_match.get("locations.geonames_details.country_name", ""),
                "raw_text": affiliation_text,
            }

            return (
                matched_org["matched_name"],
                matched_org["ror_id"],
                matched_org["lat"],
                matched_org["lon"],
                matched_org["country"],
                matched_org["raw_text"],
            )
        else:
            return None, None, None, None, None
    except Exception as e:
        print(f"Error searching for affiliation: {e}")
        return {"matched_name": "", "raw_text": affiliation_text}


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = INTERIM_DATA_DIR / MODE / "dataset_with_affiliations.parquet",
    output_path: Path = INTERIM_DATA_DIR / MODE / "features_with_ror.parquet",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Generating features from dataset...")
    df = pd.read_parquet(input_path)
    print(len(df))

    def get_ror_data(affiliation):
        print(affiliation)
        if pd.isna(affiliation):
            return None, None, None, None, None, None
        try:
            data = search_ror(affiliation)
            if data:
                return data[0], data[1], data[2], data[3], data[4], data[5]
        except Exception as e:
            logger.error(f"Error processing affiliation '{affiliation}': {e}")
        return None, None, None, None, None, None

    tqdm.pandas(desc="Fetching ROR data")
    results = df["affiliation"].progress_apply(get_ror_data)

    df[
        [
            "matched_name",
            "matched_ror_id",
            "matched_lat",
            "matched_lon",
            "matched_country",
            "matched_raw_text",
        ]
    ] = pd.DataFrame(results.tolist(), index=df.index)

    df.to_parquet(output_path, index=False)

    # -----------------------------------------


if __name__ == "__main__":
    app()
