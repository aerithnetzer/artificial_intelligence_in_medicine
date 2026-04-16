"""
Geographic visualizations.

Functions for plotting geographic distribution of publications,
KDE heatmaps, country-level analyses, and geographic constraint maps.
"""

from loguru import logger
import plotly.express as px

from artificial_intelligence_in_medicine.config import FIGURES_DIR
from artificial_intelligence_in_medicine.visualizations.utils import (
    clean_year_column,
    ensure_output_dir,
    load_features,
    save_plot,
)


def plot_cartographic_density(mode: str):
    """
    Interactive Plotly density map of author affiliation locations.
    """
    df = load_features(mode)
    output_path = FIGURES_DIR / mode / "geo_density"
    ensure_output_dir(mode)

    fig = px.density_map(
        df.dropna(subset=["matched_lat", "matched_lon"]),
        lat="matched_lat",
        lon="matched_lon",
        radius=5,
        map_style="open-street-map",
        zoom=0,
    )
    fig.update_layout(map_style="open-street-map", map_center_lon=0)
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    save_plot(fig, output_path, width=1400, height=800)


def plot_geographic_kde_by_year(mode: str):
    """
    Static matplotlib KDE density plot of author locations,
    with color gradient representing publication year.
    Uses cartopy for map projection.
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.cm as mcm
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    import seaborn as sns

    df = load_features(mode)
    df = clean_year_column(df)
    df = df.dropna(subset=["matched_lat", "matched_lon"])
    df = df[(df["matched_lat"].between(-90, 90)) & (df["matched_lon"].between(-180, 180))]

    if df.empty:
        logger.warning(f"No geographic data for {mode}")
        return

    unique_years = sorted(df["year"].unique())
    min_year, max_year = min(unique_years), max(unique_years)
    norm = mcolors.Normalize(vmin=min_year, vmax=max_year)
    cmap = plt.get_cmap("viridis")

    plt.figure(figsize=(16, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor="lightgray")
    ax.add_feature(cfeature.OCEAN, facecolor="white")

    for year in unique_years:
        year_df = df[df["year"] == year]
        if len(year_df) < 10:
            continue
        try:
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
        except Exception:
            pass

    sm = mcm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation="vertical", fraction=0.02, pad=0.04)
    cbar.set_label("Year", rotation=270, labelpad=15)

    ax.set_title(f"Global KDE Heatmap of Author Locations by Year ({mode})", fontsize=16)
    ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
    plt.tight_layout()

    output_path = FIGURES_DIR / mode / "global_kde_heatmap_by_year.png"
    ensure_output_dir(mode)
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.success(f"Saved geographic KDE heatmap to {output_path}")


def plot_lat_lon_scatter(mode: str):
    """
    Simple scatterplot of author lat/lon on a world map using cartopy.
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt

    df = load_features(mode)
    df = df.dropna(subset=["matched_lat", "matched_lon"])

    plt.figure(figsize=(16, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor="lightgray")
    ax.add_feature(cfeature.OCEAN, facecolor="white")
    ax.scatter(
        df["matched_lon"],
        df["matched_lat"],
        color="red",
        s=10,
        alpha=0.6,
        transform=ccrs.PlateCarree(),
    )
    ax.set_title(f"Author Locations ({mode})", fontsize=16)
    ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
    plt.tight_layout()

    output_path = FIGURES_DIR / mode / "map_scatter_output.png"
    ensure_output_dir(mode)
    plt.savefig(output_path, dpi=200)
    plt.close()
    logger.success(f"Saved scatter map to {output_path}")
