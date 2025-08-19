import pandas as pd
from artificial_intelligence_in_medicine.config import PROCESSED_DATA_DIR

modes = ["ARTIFICIAL_INTELLIGENCE", "GENE_EXPRESSION"]

for mode in modes:
    csv_path = PROCESSED_DATA_DIR / mode / "interdisciplinary_nodes.csv"
    df = pd.read_csv(csv_path)
    avg_distance = df["jaccard_distance"].mean()
    print(f"Average Jaccard distance for {mode}: {avg_distance:.4f}")
