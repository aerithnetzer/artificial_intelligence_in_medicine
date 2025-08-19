import json
import os
from pathlib import Path

from Bio import Entrez
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from loguru import logger
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from tqdm import tqdm
import typer

from artificial_intelligence_in_medicine.config import (
    FIGURES_DIR,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
)

app = typer.Typer()

Entrez.email = os.getenv("NCBI_EMAIL")
Entrez.api_key = os.getenv("NCBI_API_KEY")


batch_size = 1_000

MODE = "GENE_EXPRESSION"


def plot_lat_lon_scatter(
    df,
    output_path,
    lat_col="matched_lat",
    lon_col="matched_lon",
    title="Lat/Lon Scatterplot",
):
    """
    Plots latitude and longitude as a scatterplot on a world map.
    """
    plt.figure(figsize=(16, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor="lightgray")
    ax.add_feature(cfeature.OCEAN, facecolor="white")
    ax.scatter(
        df[lon_col],
        df[lat_col],
        color="red",
        s=10,
        alpha=0.6,
        transform=ccrs.PlateCarree(),
        label="Locations",
    )
    ax.set_title(title, fontsize=16)
    ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)


def get_lat_lon(row):
    """
    Extracts latitude and longitude from the matched ROR data.
    """
    lat = row.get("matched_lat")
    lon = row.get("matched_lon")

    if lat is not None and lon is not None and not pd.isna(lat) and not pd.isna(lon):
        return lat, lon
    return None, None


@app.command()
def main(
    input_path: Path = INTERIM_DATA_DIR / MODE / "features_with_ror.json",
    output_path: Path = FIGURES_DIR / MODE / "global_kde_heatmap_by_year.png",
):
    """
    Generates a time-dependent KDE density plot of author affiliations.
    The color gradient represents the year of publication.
    """
    logger.info("Processing dataset...")
    df = pd.read_json(input_path)
    scatter_output_path: Path = FIGURES_DIR / MODE / "map_scatter_output.png"
    plot_lat_lon_scatter(df, scatter_output_path)
    print(df["affiliation"])
    print(df.columns)
    df["matched_lat"], df["matched_lon"] = zip(*df.apply(get_lat_lon, axis=1))
    # Drop NaNs and filter lat/lon
    df = df.explode(["matched_lat", "matched_lon"])
    df = df.dropna(subset=["matched_lat", "matched_lon", "year"])
    df = df[(df["matched_lat"].between(-90, 90)) & (df["matched_lon"].between(-180, 180))]

    df["lat"], df["lon"] = zip(*df.apply(get_lat_lon, axis=1))
    # Setup plot with PlateCarree projection
    plt.figure(figsize=(16, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Add geographic features
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor="lightgray")
    ax.add_feature(cfeature.OCEAN, facecolor="white")

    # Create a colormap and a normalizer for the years
    unique_years = sorted(df["year"].unique())
    min_year, max_year = min(unique_years), max(unique_years)
    norm = mcolors.Normalize(vmin=min_year, vmax=max_year)
    cmap = plt.get_cmap("viridis")

    # Plot KDE for each year
    for year in unique_years:
        year_df = df[df["year"] == year]
        sns.kdeplot(
            x=year_df["matched_lon"],
            y=year_df["matched_lat"],
            fill=True,
            color=cmap(norm(year)),
            bw_adjust=0.5,
            thresh=0.05,
            levels=10,
            alpha=0.5,
            ax=ax,
            transform=ccrs.PlateCarree(),
        )

    # Add a colorbar as a gradient legend
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Dummy array for the mappable
    cbar = plt.colorbar(sm, ax=ax, orientation="vertical", fraction=0.02, pad=0.04)
    cbar.set_label("Year", rotation=270, labelpad=15)

    # Title and layout
    ax.set_title("Global KDE Heatmap of Author Locations by Year", fontsize=16)
    ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
    plt.tight_layout()
    logger.info("Saving figure to {}", output_path)
    plt.savefig(output_path, dpi=300)


if __name__ == "__main__":
    app()
