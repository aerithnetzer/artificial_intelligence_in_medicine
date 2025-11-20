from pathlib import Path
import pickle

from loguru import logger
import networkx as nx
import pandas as pd
from tqdm import tqdm
import typer
import matplotlib.pyplot as plt
from artificial_intelligence_in_medicine.config import (
    INTERIM_DATA_DIR,
    MODELS_DIR,
)
from artificial_intelligence_in_medicine.modeling._graphs_helpers import (
    analyze_brokerage,
    calculate_constraint,
    community_detection,
    initialize_graph,
)

app = typer.Typer()

MODE = "ARTIFICIAL_INTELLIGENCE"


def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = INTERIM_DATA_DIR / MODE / "features_with_ror.json",
    model_path: Path = MODELS_DIR / MODE / "citation_model.pkl",
    # -----------------------------------------
):
    G = initialize_graph()
    analyze_brokerage(G)
    community_detection(mode=MODE, g=G)
    print(calculate_constraint(G))


if __name__ == "__main__":
    main()
