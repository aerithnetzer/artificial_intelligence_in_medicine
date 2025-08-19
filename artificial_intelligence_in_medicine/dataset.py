import ast
import glob
import json
import os
from pathlib import Path

from Bio import Entrez
from elasticsearch import Elasticsearch
from loguru import logger
import pandas as pd
import requests
from tqdm import tqdm
import typer

from artificial_intelligence_in_medicine.config import INTERIM_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()

Entrez.email = os.getenv("NCBI_EMAIL")
Entrez.api_key = os.getenv("NCBI_API_KEY")

icite_baseurl = "https://icite.od.nih.gov/api/pubs?pmids="
MODE = "GENE_EXPRESSION"
batch_size = 10_000

es = Elasticsearch("http://localhost:9200", verify_certs=False)  # change if using remote


def chunker(seq, size):
    """Yield successive n-sized chunks from seq."""
    return (seq[pos : pos + size] for pos in range(0, len(seq), size))


def citation_data(input_path: str, output_path: str):
    """
    Processes a directory of JSON files to extract citation data,
    queries the iCite API for citation counts, and saves the enriched data.
    """

    json_files = glob.glob(input_path)
    if not json_files:
        logger.error(f"No JSON files found at path: {input_path}")
        return

    all_data = []
    for f in tqdm(json_files, desc="Processing batch files"):
        with open(f, "r") as file:
            data = json.load(file)
            articles = data  # Assuming the JSON file root is the articles dict/list
            pmids = []
            article_metadata = {}

            for article_list in articles.values():
                if not isinstance(article_list, list):
                    continue

                for article in article_list:
                    if not isinstance(article, dict):
                        continue

                    medline_citation = article.get("MedlineCitation", {})

                    if not medline_citation:
                        continue

                    pmid = medline_citation.get("PMID", None)
                    if not pmid:
                        continue

                    pmids.append(str(pmid))
                    article_info = medline_citation.get("Article", {})
                    title = article_info.get("ArticleTitle", None)

                    date_completed = medline_citation.get("DateCompleted", {})
                    year = date_completed.get("Year", None)

                    mesh_headings = [
                        mh.get("DescriptorName")
                        for mh in medline_citation.get("MeshHeadingList", [])
                        if mh.get("DescriptorName")
                    ]

                    author_list = [author for author in article_info.get("AuthorList", [])]

                    grant_list = [
                        grant.get("Agency")
                        for grant in article_info.get("GrantList", [])
                        if grant.get("Agency")
                    ]
                    print(grant_list)
                    affiliation = (
                        author_list[0].get("AffiliationInfo", [{}])[0].get("Affiliation")
                        if author_list and author_list[0].get("AffiliationInfo")
                        else None
                    )
                    article_metadata[str(pmid)] = {
                        "title": title,
                        "year": year,
                        "mesh_headings": mesh_headings,
                        "author_list": author_list,
                        "grant_list": grant_list,
                        "cited_by": [],  # Placeholder, to be filled by iCite API
                        "affiliation": affiliation,
                    }
            # Query iCite API in chunks
            for i in tqdm(range(0, len(pmids), 500), desc="Querying iCite API", leave=False):
                pmid_chunk = pmids[i : i + 100]
                if not pmid_chunk:
                    continue
                pmid_string = ",".join(pmid_chunk)
                url = f"https://icite.od.nih.gov/api/pubs?pmids={pmid_string}"
                try:
                    r = requests.get(url)
                    r.raise_for_status()  # Raises HTTPError for bad responses
                    icite_data = r.json()
                    for item in icite_data.get("data", []):
                        item_pmid = str(item.get("pmid"))
                        if item_pmid in article_metadata:
                            article_metadata[item_pmid]["cited_by"] = item.get("cited_by", [])
                except requests.exceptions.RequestException as e:
                    logger.error(f"Error fetching from iCite API: {e}")

            # Append processed data to all_data
            for pmid, metadata in article_metadata.items():
                all_data.append({"pmid": pmid, **metadata})

    # Save all processed data to a single file
    with open(output_path, "w") as f:
        json.dump(all_data, f, indent=4)

    logger.info(f"Enriched dataset saved to {output_path}")


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


def get_affiliation_coordinate_data(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = INTERIM_DATA_DIR / MODE / "dataset_with_affiliations.json",
    output_path: Path = INTERIM_DATA_DIR / MODE / "features_with_ror.json",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Getting affiliation coordinate data...")
    df = pd.read_json(input_path)
    print(df["affiliation"].nunique())

    def get_ror_data(affiliation):
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
    df.to_json(output_path, index=False)
    logger.info(f"Data with ROR features saved to {output_path}")
    logger.info(f"Data has columns: {df.columns.tolist()}")


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / MODE / "pmids.txt",
    output_path: Path = INTERIM_DATA_DIR / MODE,
):
    logger.info("Processing dataset...")
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
    get_affiliation_coordinate_data(
        input_path=INTERIM_DATA_DIR / MODE / "dataset_with_citation_data.json"
    )


if __name__ == "__main__":
    app()
