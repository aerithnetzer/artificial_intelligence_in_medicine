import glob
import json
import os

from loguru import logger
import requests
from tqdm import tqdm
import typer

from artificial_intelligence_in_medicine.config import INTERIM_DATA_DIR

app = typer.Typer()
icite_baseurl = "https://icite.od.nih.gov/api/pubs?pmids="


def chunker(seq, size):
    """Yield successive n-sized chunks from seq."""
    return (seq[pos : pos + size] for pos in range(0, len(seq), size))


MODE = "GENE_EXPRESSION"


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: str = f"{str(INTERIM_DATA_DIR)}/{MODE}/*.json",
    output_path: str = f"{str(INTERIM_DATA_DIR)}/{MODE}/dataset_with_citation_data.json",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    json_files = sorted(glob.glob(input_path))
    json_files = json_files[:1]  # Limit to first 100 files for testing
    logger.info(f"Number of JSON files found: {len(json_files)}")
    batch = 0
    all_data = []
    for f in tqdm(json_files):
        with open(f, "r") as data:
            data = json.load(data)
            articles = data["PubmedArticle"]
            pmids = []
            article_metadata = {}
            for article in articles:
                article: dict
                pmid = article["MedlineCitation"].get("PMID")
                pmids.append(str(pmid))
                title = article["MedlineCitation"]["Article"].get("ArticleTitle")
                try:
                    year = article["MedlineCitation"]["Article"]["Journal"]["JournalIssue"][
                        "PubDate"
                    ]["Year"]
                except KeyError:
                    year = article["MedlineCitation"]["Article"]["Journal"]["JournalIssue"][
                        "PubDate"
                    ].get("MedlineDate")

                mesh_headings = article["MedlineCitation"].get("MeshHeadingList")
                author_list = article["MedlineCitation"]["Article"].get("AuthorList")

                article_metadata[pmid] = {
                    "title": title,
                    "year": year,
                    "mesh_headings": mesh_headings,
                    "author_list": author_list,
                }
            for pmid_chunk in chunker(pmids, 100):
                pmid_string = ",".join(pmid_chunk)
                url = f"{icite_baseurl}{pmid_string}"
                r = requests.get(url)
                if r.ok:
                    citation_data = r.json()["data"]
                    for d in citation_data:
                        pmid = str(d["pmid"])
                        d.update(article_metadata[pmid])
                        all_data.append(d)
        if batch % 10 == 0:
            logger.info(f"Processed {batch} files, {len(all_data)} records collected.")
            with open(output_path, "w") as f:
                json.dump(all_data, f, indent=4)
        batch += 1
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    else:
        with open(output_path, "w") as f:
            json.dump(all_data, f, indent=4)

    logger.success(f"Features generation complete. Saved to {output_path}")


if __name__ == "__main__":
    app()
