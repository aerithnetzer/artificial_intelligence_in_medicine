from pathlib import Path

import pandas as pd
import typer

from artificial_intelligence_in_medicine.config import (
    FIGURES_DIR,
    INTERIM_DATA_DIR,
    RAW_DATA_DIR,
)

MODE = "ARTIFICIAL_INTELLIGENCE"

app = typer.Typer()


def read_txt_file(path: Path) -> set:
    with open(path, "r") as f:
        return set(f.readlines())


def read_json_file(path: Path) -> set:
    df = pd.read_json(path)
    print(df.columns)
    return set(df["pmid"].to_list())


def find_disintersection_of_sets(set1, set2) -> list:
    """
    This function finds the disintersection between two sets. That is, the elements that are in set1 but not in set2.
    """
    pmids = []
    for i in set1:
        if i not in set2:
            pmids.append(i.strip())

    return pmids


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / MODE / "pmids.txt",
    features_path: Path = INTERIM_DATA_DIR / MODE / "features_with_ror.json",
):
    pmids = []
    set1 = read_txt_file(input_path)
    set2 = read_json_file(features_path)

    pmids = find_disintersection_of_sets(set1, set2)
    print(len(pmids))
    return pmids


if __name__ == "__main__":
    app()
