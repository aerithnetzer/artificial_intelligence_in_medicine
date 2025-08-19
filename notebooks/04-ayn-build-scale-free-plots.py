# %%
import pickle

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

from artificial_intelligence_in_medicine.config import MODELS_DIR

MODE = "artificial_intelligence"

# %%
# Load the graph
with open(MODELS_DIR / MODE / "citation_model.pkl", "rb") as f:
    G: nx.DiGraph
    G = pickle.load(f)

# %%
nodes = []
degrees = []
for node, degree in G.degree():
    nodes.append(node)
    degrees.append(degree)


# %%
df = pd.DataFrame(
    {
        "nodes": nodes,
        "degrees": degrees,
    }
)
log_degrees = np.log10(degrees)
sns.displot(
    degrees,
    kde=True,  # overlay KDE
    bins=50,  # adjust bins as needed
)
plt.xlabel("Degree")
plt.ylabel("Count")
plt.show()
plt.show()
