from glob import glob
import json
import pandas as pd
from artificial_intelligence_in_medicine.config import INTERIM_DATA_DIR
from pathlib import Path

features_path = "./data/interim/GENE_EXPRESSION/features_with_ror.json"

path = "./data/interim/GENE_EXPRESSION/batch_00000.json"


df1 = pd.read_json(features_path)
with open(path, "r") as f:
    batch_data = json.load(f)
print(df1.head())
for i in batch_data["PubmedArticle"]:
    print(i["PubmedData"])
    exit()
