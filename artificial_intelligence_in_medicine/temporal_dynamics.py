from collections import defaultdict
from pathlib import Path
import pickle

import igraph as ig
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import powerlaw
from scipy.stats import kstest

from artificial_intelligence_in_medicine.config import MODELS_DIR


def analyze_scale_free_by_year_ks(graph_path):
    """
    Loads a pickled graph and evaluates scale-free characteristics by year
    using the Kolmogorov–Smirnov (KS) test between the degree distribution
    and a fitted power-law model.

    Parameters
    ----------
    graph_path : str or Path
        Path to the pickled graph object (igraph or networkx).

    Returns
    -------
    dict
        A dictionary keyed by year containing:
            - alpha: fitted power-law exponent
            - ks_stat: Kolmogorov–Smirnov statistic
            - p_value: KS p-value (higher = closer to power law)
            - normalized_score: p_value / n_nodes
    """
    # Load the graph
    with open(graph_path, "rb") as f:
        graph = pickle.load(f)

    # Convert to igraph if it's a NetworkX graph
    if isinstance(graph, nx.Graph):
        g = ig.Graph.from_networkx(graph)
    elif isinstance(graph, ig.Graph):
        g = graph
    else:
        raise TypeError("Graph must be a NetworkX or igraph object")

    if "year" not in g.vs.attributes():
        raise ValueError("Graph vertices must have a 'year' attribute")

    year_nodes = defaultdict(list)
    for v in g.vs:
        year = v["year"]
        if year is not None:
            year_nodes[year].append(v.index)

    results = {}

    for year, nodes in sorted(year_nodes.items()):
        subgraph = g.subgraph(nodes)
        degrees = np.array(subgraph.degree())
        degrees = degrees[degrees > 0]  # remove isolates

        if len(degrees) < 10 or len(set(degrees)) < 2:
            results[year] = {
                "alpha": np.nan,
                "ks_stat": np.nan,
                "p_value": np.nan,
                "normalized_score": np.nan,
            }
            continue

        try:
            # Fit power law
            fit = ig.power_law_fit(degrees)
            alpha = fit.alpha
            xmin = fit.xmin

            # Generate theoretical power-law distribution
            def power_law_cdf(x):
                return 1 - (x / xmin) ** (1 - alpha)

            # Empirical CDF
            sorted_deg = np.sort(degrees)
            empirical_cdf = np.arange(1, len(sorted_deg) + 1) / len(sorted_deg)

            # Theoretical CDF for observed x values
            theoretical_cdf = power_law_cdf(sorted_deg)

            # KS statistic and p-value
            ks_stat, p_value = kstest(sorted_deg, power_law_cdf)

            n_nodes = len(nodes)
            normalized_score = p_value / n_nodes

            results[year] = {
                "alpha": alpha,
                "ks_stat": ks_stat,
                "p_value": p_value,
                "normalized_score": normalized_score,
            }
        except Exception:
            results[year] = {
                "alpha": np.nan,
                "ks_stat": np.nan,
                "p_value": np.nan,
                "normalized_score": np.nan,
            }

    return results


def analyze_scale_free_by_year(graph_path):
    """
    Loads a pickled graph and computes a normalized measure of
    scale-free characteristics by year.

    Parameters
    ----------
    graph_path : str or Path
        Path to the pickled graph object (igraph or networkx).

    Returns
    -------
    dict
        A dictionary keyed by year, where each value is a normalized
        scale-free score (alpha from power-law fit divided by number of nodes).
    """
    # Load the graph
    with open(graph_path, "rb") as f:
        graph = pickle.load(f)

    # Convert to igraph if needed
    if isinstance(graph, nx.Graph):
        g = ig.Graph.from_networkx(graph)
    elif isinstance(graph, ig.Graph):
        g = graph
    else:
        raise TypeError("Graph must be a NetworkX or igraph object")

    # Ensure nodes have a 'year' attribute
    if "year" not in g.vs.attributes():
        raise ValueError("Graph vertices must have a 'year' attribute")

    # Group nodes by year
    year_nodes = defaultdict(list)
    for v in g.vs:
        year = v["year"]
        if year is not None:
            year_nodes[year].append(v.index)

    scale_free_results = {}

    # Compute normalized power-law fit by year
    for year, nodes in sorted(year_nodes.items()):
        subgraph = g.subgraph(nodes)
        degrees = subgraph.degree()

        if len(set(degrees)) < 2:  # skip trivial degree distributions
            scale_free_results[year] = np.nan
            continue

        try:
            fit = ig.power_law_fit(degrees)
            alpha = fit.alpha
            n_nodes = len(nodes)
            normalized_score = alpha / n_nodes
            scale_free_results[year] = normalized_score
        except Exception:
            scale_free_results[year] = np.nan

    return scale_free_results


def analyze_scale_free_characteristics(graph_path: str | Path) -> pd.DataFrame:
    """
    Load a graph (NetworkX or igraph) and analyze its scale-free characteristics
    by modeling how the degree distribution fits a power law, optionally over time
    using the node attribute `year`.

    Parameters
    ----------
    graph_path : str or Path
        Path to the graph file. Supports GraphML, GML, or Pickle for NetworkX.

    Returns
    -------
    pd.DataFrame
        Columns:
        - year: Year (if available)
        - n_nodes: Number of nodes
        - n_edges: Number of edges
        - alpha: Estimated power-law exponent
        - sigma: Power-law fit standard error
        - R: Log-likelihood ratio (vs. exponential)
        - p_value: Significance of power-law fit
    """
    graph_path = Path(graph_path)

    # --- Load graph ---
    if graph_path.suffix in [".graphml", ".gml"]:
        try:
            graph = ig.read(str(graph_path))
        except Exception:
            g_nx = (
                nx.read_graphml(graph_path)
                if graph_path.suffix == ".graphml"
                else nx.read_gml(graph_path)
            )
            graph = ig.Graph.from_networkx(g_nx)

    elif graph_path.suffix in [".pkl", ".pickle"]:
        with open(graph_path, "rb") as f:
            loaded = pickle.load(f)
        if isinstance(loaded, nx.Graph):
            graph = ig.Graph.from_networkx(loaded)
        elif isinstance(loaded, ig.Graph):
            graph = loaded
        else:
            raise TypeError("Pickled object is not a NetworkX or igraph graph.")

    else:
        raise ValueError(f"Unsupported graph format: {graph_path.suffix}")

    # --- Collect node years if available ---
    years = graph.vs["year"] if "year" in graph.vs.attributes() else None

    # --- Prepare analysis ---
    results = []

    if years:
        unique_years = sorted(set(years))
        for yr in unique_years:
            sub_v = [v.index for v in graph.vs if v["year"] == yr]
            if len(sub_v) < 20:
                continue  # too few nodes to analyze reliably
            subgraph = graph.subgraph(sub_v)
            degrees = np.array(subgraph.degree())
            if np.all(degrees == 0):
                continue
            fit = powerlaw.Fit(degrees, verbose=False)
            R, p = fit.distribution_compare("power_law", "exponential")
            results.append(
                {
                    "year": yr,
                    "n_nodes": len(sub_v),
                    "n_edges": len(subgraph.es),
                    "alpha": fit.alpha,
                    "sigma": fit.sigma,
                    "R": R,
                    "p_value": p,
                }
            )
    else:
        degrees = np.array(graph.degree())
        fit = powerlaw.Fit(degrees, verbose=False)
        R, p = fit.distribution_compare("power_law", "exponential")
        results.append(
            {
                "year": None,
                "n_nodes": graph.vcount(),
                "n_edges": graph.ecount(),
                "alpha": fit.alpha,
                "sigma": fit.sigma,
                "R": R,
                "p_value": p,
            }
        )

    return pd.DataFrame(results)


def main(mode: str = "GENE_EXPRESSION"):
    """Run all temporal dynamics analyses for a given mode."""
    graph_path = MODELS_DIR / mode / "citation_model.pkl"

    results = analyze_scale_free_by_year_ks(graph_path)
    for year, metrics in results.items():
        print(
            f"{year:.0f}: alpha={metrics['alpha']:.2f}, KS={metrics['ks_stat']:.3f}, "
            f"p={metrics['p_value']:.3f}, normalized={metrics['normalized_score']:.5f}"
        )

    df_scale_free = analyze_scale_free_characteristics(graph_path)
    print(df_scale_free)
    df_valid = df_scale_free.dropna(subset=["alpha"])
    plt.figure(figsize=(10, 6))
    plt.plot(df_valid["year"], df_valid["alpha"], marker="o")
    plt.title(f"Power-law Exponent (alpha) Over Time ({mode})")
    plt.xlabel("Year")
    plt.ylabel("Alpha")
    plt.grid(True)

    results_norm = analyze_scale_free_by_year(graph_path)
    for year, score in results_norm.items():
        print(f"{year}: {score:.4f}")
    years = [y for y in results_norm if not np.isnan(y)]
    scores = [results_norm[y] for y in years]

    plt.figure(figsize=(10, 6))
    plt.plot(years, scores, marker="o")
    plt.xlabel("Year")
    plt.ylabel("Normalized Scale-Free Score (alpha / N)")
    plt.title(f"Scale-Free Structure Over Time ({mode})")
    plt.show()


if __name__ == "__main__":
    import typer

    typer.run(main)
