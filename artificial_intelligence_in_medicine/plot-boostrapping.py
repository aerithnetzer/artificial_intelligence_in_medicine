import pandas as pd
import plotly.express as px

# Read the CSV
df = pd.read_csv("random_pubmed_sample.csv")

# Extract year and count publications per year
df["year"] = pd.to_numeric(df["year"], errors="coerce")
year_counts = df["year"].value_counts().sort_index()

# Calculate coefficient of variation (CV)
mean_count = year_counts.mean()
std_count = year_counts.std()
cv = std_count / mean_count if mean_count else 0

# Plot with Plotly
fig = px.line(
    x=year_counts.index,
    y=year_counts.values,
    labels={"x": "Year", "y": "Number of Publications"},
    title=f"PubMed Publications Over Time (CV={cv:.2f})",
)
fig.show()
