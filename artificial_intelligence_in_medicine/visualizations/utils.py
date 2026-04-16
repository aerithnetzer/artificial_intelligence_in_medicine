"""
Shared utilities for visualization modules.

Provides consistent data loading, color palettes, output helpers,
and MODE constants used across all visualization functions.
"""

from pathlib import Path

from loguru import logger
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px

from artificial_intelligence_in_medicine.config import (
    FIGURES_DIR,
    INTERIM_DATA_DIR,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODES = ["ARTIFICIAL_INTELLIGENCE", "GENE_EXPRESSION", "NULL"]

MODE_LABELS = {
    "ARTIFICIAL_INTELLIGENCE": "Artificial Intelligence",
    "GENE_EXPRESSION": "Gene Expression",
    "NULL": "Random PubMed Sample",
}

MODE_COLORS = {
    "ARTIFICIAL_INTELLIGENCE": "#1f77b4",  # blue
    "GENE_EXPRESSION": "#ff7f0e",  # orange
    "NULL": "#2ca02c",  # green
}

MODE_COLORS_LIGHT = {
    "ARTIFICIAL_INTELLIGENCE": "rgba(31,119,180,0.15)",
    "GENE_EXPRESSION": "rgba(255,127,14,0.15)",
    "NULL": "rgba(44,160,44,0.15)",
}

COMMUNITY_PALETTE = px.colors.qualitative.Set1 + px.colors.qualitative.Set3


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
def load_features(mode: str) -> pd.DataFrame:
    """Load the features_with_ror.json for a given mode as a DataFrame."""
    path = INTERIM_DATA_DIR / mode / "features_with_ror.json"
    logger.info(f"Loading features for {mode} from {path}")
    df = pd.read_json(path)
    logger.info(f"  Loaded {len(df)} records with columns: {df.columns.tolist()}")
    return df


def load_features_all() -> dict[str, pd.DataFrame]:
    """Load features for all three modes. Returns dict keyed by mode name."""
    return {mode: load_features(mode) for mode in MODES}


def load_graph(mode: str) -> nx.DiGraph:
    """Load the citation_model.pkl graph for a given mode."""
    import pickle

    path = MODELS_DIR / mode / "citation_model.pkl"
    logger.info(f"Loading graph for {mode} from {path}")
    with open(path, "rb") as f:
        G = pickle.load(f)
    logger.info(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def load_interdisciplinary_nodes(mode: str) -> pd.DataFrame | None:
    """Load the interdisciplinary_nodes.csv for a mode, or None if missing."""
    path = PROCESSED_DATA_DIR / mode / "interdisciplinary_nodes.csv"
    if not path.exists():
        logger.warning(f"No interdisciplinary_nodes.csv for {mode}")
        return None
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Year cleaning
# ---------------------------------------------------------------------------
def clean_year_column(df: pd.DataFrame, col: str = "year") -> pd.DataFrame:
    """
    Ensure the year column is numeric integer, dropping NaN rows.
    Returns a copy with cleaned year.
    """
    df = df.copy()
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[col])
    df[col] = df[col].astype(int)
    return df


# ---------------------------------------------------------------------------
# Citation count helper
# ---------------------------------------------------------------------------
def citation_count(row_or_series):
    """Compute citation count from a cited_by field (list or scalar)."""
    val = row_or_series
    if isinstance(val, list):
        return len(val)
    return 0


def add_citation_count_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'citation_count' column derived from 'cited_by'."""
    df = df.copy()
    if "citation_count" not in df.columns:
        if "cited_by" in df.columns:
            df["citation_count"] = df["cited_by"].apply(
                lambda x: len(x) if isinstance(x, list) else 0
            )
        else:
            df["citation_count"] = 0
    return df


# ---------------------------------------------------------------------------
# Grant / funding helpers
# ---------------------------------------------------------------------------
def add_grant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add helper columns for funding analysis:
      - has_funding: bool
      - num_funding_sources: int
    """
    df = df.copy()
    df["has_funding"] = df["grant_list"].apply(lambda x: isinstance(x, list) and len(x) > 0)
    df["num_funding_sources"] = df["grant_list"].apply(
        lambda x: len(x) if isinstance(x, list) else 0
    )
    return df


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------
def community_color_map(communities: list | set) -> dict:
    """Build a color map for a set of community labels."""
    unique = sorted(set(str(c) for c in communities))
    return {c: COMMUNITY_PALETTE[i % len(COMMUNITY_PALETTE)] for i, c in enumerate(unique)}


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
def ensure_output_dir(mode: str) -> Path:
    """Ensure the figures directory for a mode exists and return it."""
    out = FIGURES_DIR / mode
    out.mkdir(parents=True, exist_ok=True)
    return out


def ensure_comparative_dir() -> Path:
    """Ensure the comparative figures directory exists and return it."""
    out = FIGURES_DIR / "comparative"
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_plot(fig, output_path: Path, width: int = 1200, height: int = 800, scale: int = 3):
    """
    Save a Plotly figure as both interactive HTML and static PNG.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    html_path = output_path.with_suffix(".html")
    png_path = output_path.with_suffix(".png")

    fig.write_html(str(html_path))
    try:
        fig.write_image(str(png_path), width=width, height=height, scale=scale)
    except Exception as e:
        logger.warning(f"Could not write PNG ({png_path}): {e}")

    logger.success(f"Saved: {html_path}")
    return html_path


# ---------------------------------------------------------------------------
# Graph constraint helpers
# ---------------------------------------------------------------------------
def compute_constraint_dict(G: nx.Graph) -> dict:
    """
    Compute Burt's structural constraint for all nodes.
    Works with directed or undirected graphs.
    """
    if isinstance(G, nx.DiGraph):
        g_undirected = G.to_undirected()
    else:
        g_undirected = G
    return nx.constraint(g_undirected)


def constraint_by_year(G: nx.Graph, constraints: dict) -> dict[int, list[float]]:
    """Group constraint values by the 'year' node attribute."""
    from collections import defaultdict

    year_constraints = defaultdict(list)
    for node, c in constraints.items():
        if np.isnan(c):
            continue
        year = G.nodes[node].get("year")
        if year is not None:
            try:
                year = int(float(year))
            except (ValueError, TypeError):
                continue
            year_constraints[year].append(c)
    return dict(year_constraints)
