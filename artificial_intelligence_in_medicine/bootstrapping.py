import requests
import random
import time
import pandas as pd
from tqdm import tqdm
import os

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
EMAIL = ""  # required by NCBI policy


def get_total_count():
    """Get total number of PubMed records available."""
    params = {"db": "pubmed", "retmode": "json", "term": "all[filter]"}
    r = requests.get(ESEARCH_URL, params=params)
    r.raise_for_status()
    return int(r.json()["esearchresult"]["count"])


def get_random_pmids(total_count, n=5000):
    """Get n random unique PMIDs."""
    random_ids = random.sample(range(1, total_count), n)
    return [str(rid) for rid in random_ids]


def fetch_pubmed_records(pmids):
    """Fetch title and year for a batch of PMIDs."""
    ids_str = ",".join(pmids)
    data = {
        "db": "pubmed",
        "id": ids_str,
        "retmode": "json",
        "email": EMAIL,
    }
    r = requests.post(ESUMMARY_URL, data=data)
    r.raise_for_status()
    data = r.json().get("result", {})
    records = []
    for pid in pmids:
        if pid in data:
            rec = data[pid]
            title = rec.get("title", "")
            year = rec.get("pubdate", "").split(" ")[0]
            records.append({"pmid": pid, "year": year, "title": title})
    return records


def get_random_pubmed_sample(output_file, n_total=150000, batch_size=500):
    """
    Fetches a large sample of random PubMed records and saves them to a CSV file
    in batches to ensure fault tolerance.
    """
    total_count = get_total_count()
    header_written = os.path.exists(output_file)

    for _ in tqdm(range(n_total // batch_size)):
        pmids = get_random_pmids(total_count, n=batch_size)
        try:
            batch_records = fetch_pubmed_records(pmids)
            if not batch_records:
                continue

            df_batch = pd.DataFrame(batch_records)
            df_batch.to_csv(output_file, mode="a", header=not header_written, index=False)
            header_written = True  # Ensure header is only written once

        except requests.exceptions.RequestException:
            print("Error fetching batch; retrying after 2s...")
            time.sleep(2)
            continue
        time.sleep(0.34)  # ~3 requests per second = safe under NCBI rate limits


# Example usage
if __name__ == "__main__":
    output_filename = "random_pubmed_sample.csv"
    get_random_pubmed_sample(output_filename, n_total=150_000, batch_size=500)
    print(f"Data saved to {output_filename}")
