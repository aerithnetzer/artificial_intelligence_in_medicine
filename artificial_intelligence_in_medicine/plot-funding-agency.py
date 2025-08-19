import glob
import json
import os
from pathlib import Path

from loguru import logger
import pandas as pd
import typer

from artificial_intelligence_in_medicine.config import (
    FIGURES_DIR,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
)

app = typer.Typer()
icite_baseurl = "https://icite.od.nih.gov/api/pubs?pmids="


def chunker(seq, size):
    """Yield successive n-sized chunks from seq."""
    return (seq[pos : pos + size] for pos in range(0, len(seq), size))


MODE = "GENE_EXPRESSION"


@app.command()
def main(
    input_path: Path = INTERIM_DATA_DIR / MODE / "features_with_ror.json",
    output_path: Path = FIGURES_DIR / MODE / "top_5_agencies_by_year.html",
):
    """
    This script reads publication data, identifies the top 5 funding agencies
    for each year based on total citations, and generates a stacked bar chart
    to visualize the results using Plotly.
    """
    import plotly.graph_objects as go
    import plotly.io as pio

    df = pd.read_json(input_path)
    print(f"Total records in dataset: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")

    # Check grant_list column
    print(f"\nGrant list info:")
    print(f"Non-null grant_list count: {df['grant_list'].notna().sum()}")
    print(f"Null grant_list count: {df['grant_list'].isna().sum()}")

    # Sample some grant_list values
    non_null_grants = df[df["grant_list"].notna()]
    if len(non_null_grants) > 0:
        print(f"\nSample grant_list values:")
        for i in range(min(3, len(non_null_grants))):
            print(f"Record {i}: {non_null_grants.iloc[i]['grant_list']}")
            print(f"Type: {type(non_null_grants.iloc[i]['grant_list'])}")

    # Check year column
    print(f"\nYear info:")
    print(f"Non-null year count: {df['year'].notna().sum()}")
    if "year" in df.columns:
        print(f"Year range: {df['year'].min()} to {df['year'].max()}")
        print(f"Year value types: {df['year'].apply(type).value_counts()}")

    # Filter out rows with no grant data
    df_with_grants = df.dropna(subset=["grant_list"])
    print(f"\nRecords with grant data: {len(df_with_grants)}")

    if len(df_with_grants) == 0:
        print("No records with grant data found. Cannot generate plot.")
        return

    # Filter out empty grant lists
    df_with_grants = df_with_grants[
        df_with_grants["grant_list"].apply(lambda x: isinstance(x, list) and len(x) > 0)
    ]
    print(f"Records with non-empty grant lists: {len(df_with_grants)}")

    if len(df_with_grants) == 0:
        print("No records with non-empty grant lists found. Cannot generate plot.")
        return

    # Explode the list of grants into separate rows
    grants_df = df_with_grants.explode("grant_list")
    print(f"\nAfter explode: {len(grants_df)} records")

    if len(grants_df) > 0:
        print(f"Sample exploded grant_list values:")
        for i in range(min(3, len(grants_df))):
            print(f"Record {i}: {grants_df.iloc[i]['grant_list']}")
            print(f"Type: {type(grants_df.iloc[i]['grant_list'])}")

    # Extract agency from each grant dictionary
    grants_df["agency"] = grants_df["grant_list"].apply(
        lambda x: x if isinstance(x, str) else None
    )

    # Check extracted agencies
    print(f"\nAgency extraction:")
    print(f"Non-null agencies: {grants_df['agency'].notna().sum()}")
    print(f"Unique agencies: {grants_df['agency'].nunique()}")
    if grants_df["agency"].notna().sum() > 0:
        print(f"Sample agencies: {grants_df['agency'].dropna().head().tolist()}")

    # Clean and validate year data
    if "year" not in grants_df.columns:
        print("Error: 'year' column not found in the data.")
        return

    # Handle different year formats
    grants_df["year_clean"] = grants_df["year"].apply(
        lambda x: str(x).split("-")[0] if pd.notna(x) else None
    )
    grants_df["year_clean"] = pd.to_numeric(grants_df["year_clean"], errors="coerce")

    print(f"\nYear processing:")
    print(f"Valid years: {grants_df['year_clean'].notna().sum()}")
    if grants_df["year_clean"].notna().sum() > 0:
        print(f"Year range: {grants_df['year_clean'].min()} to {grants_df['year_clean'].max()}")

    # Use citation_count if available, otherwise calculate from cited_by
    if "citation_count" in grants_df.columns:
        grants_df["citations"] = grants_df["citation_count"].fillna(0)
    else:
        grants_df["citations"] = grants_df["cited_by"].apply(
            lambda x: len(x) if isinstance(x, list) else 0
        )

    print(f"\nCitation data:")
    print(f"Total citations: {grants_df['citations'].sum()}")
    print(f"Mean citations per record: {grants_df['citations'].mean():.2f}")

    # Filter for valid data
    valid_data = grants_df.dropna(subset=["agency", "year_clean"])
    print(f"\nValid records (with agency and year): {len(valid_data)}")

    if len(valid_data) == 0:
        print("No valid records with both agency and year data. Cannot generate plot.")
        return

    # Sum citations for each agency by year
    agency_citations_by_year = (
        valid_data.groupby(["year_clean", "agency"])["citations"].sum().reset_index()
    )
    agency_citations_by_year.rename(
        columns={"year_clean": "year", "citations": "citation_count"}, inplace=True
    )

    print(f"\nAggregated data: {len(agency_citations_by_year)} agency-year combinations")

    # Get the top 5 agencies for each year based on citations
    top_agencies_per_year = (
        agency_citations_by_year.sort_values(["year", "citation_count"], ascending=[True, False])
        .groupby("year")
        .head(5)
    )

    print("Top 5 most cited agencies by year (based on total citations):")
    print(top_agencies_per_year)

    if len(top_agencies_per_year) == 0:
        print("No data to plot.")
        return

    # Plotting the results as a stacked bar chart with Plotly
    try:
        # Pivot data for stacked bar chart
        pivot_df = top_agencies_per_year.pivot(
            index="year", columns="agency", values="citation_count"
        ).fillna(0)
        pivot_df = pivot_df.sort_index()
        years = pivot_df.index.astype(str)
        fig = go.Figure()
        for agency in pivot_df.columns:
            fig.add_trace(go.Bar(x=years, y=pivot_df[agency], name=agency))
        fig.update_layout(
            barmode="stack",
            title="Top 5 Funding Agencies by Total Citations per Year (Stacked)",
            xaxis_title="Year",
            yaxis_title="Total Number of Citations",
            legend_title="Agency",
            xaxis_tickangle=-45,
            autosize=False,
            width=1200,
            height=700,
            margin=dict(l=40, r=40, t=80, b=120),
        )
        print("\nPlotting stacked bar chart with Plotly. Opening in browser or saving as HTML.")
        pio.write_html(fig, file=output_path, auto_open=True)
    except ImportError:
        print("\nPlease install plotly to see the plot:")
        print("pip install plotly")


if __name__ == "__main__":
    app()
