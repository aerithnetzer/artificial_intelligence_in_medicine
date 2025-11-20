import pandas as pd


def save_pmids_to_txt(csv_path, txt_path):
    df = pd.read_csv(csv_path)
    pmids = df["pmid"].astype(str)
    with open(txt_path, "w") as f:
        for pmid in pmids:
            f.write(f"{pmid}\n")


save_pmids_to_txt("random_pubmed_sample.csv", "pmids.txt")
