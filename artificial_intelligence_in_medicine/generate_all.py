"""
Generate All Visualizations.

Orchestrator script that generates all per-mode and cross-field comparative
visualizations in a single run.

Usage:
    uv run python -m artificial_intelligence_in_medicine.generate_all all
    uv run python -m artificial_intelligence_in_medicine.generate_all temporal
    uv run python -m artificial_intelligence_in_medicine.generate_all geographic
    uv run python -m artificial_intelligence_in_medicine.generate_all funding
    uv run python -m artificial_intelligence_in_medicine.generate_all comparative
    uv run python -m artificial_intelligence_in_medicine.generate_all per-mode --mode ARTIFICIAL_INTELLIGENCE
"""

from loguru import logger
import typer

from artificial_intelligence_in_medicine.visualizations.utils import MODE_LABELS, MODES

app = typer.Typer(help="Generate all visualizations for the AI in Medicine project.")


# -----------------------------------------------------------------------
# Per-mode visualizations (run for each of the 3 modes)
# -----------------------------------------------------------------------
def _run_per_mode_temporal(mode: str):
    from artificial_intelligence_in_medicine.visualizations.temporal import (
        normalized_citations_over_time,
        plot_normalized_articles_over_time,
    )

    logger.info(f"[{mode}] Generating temporal visualizations...")
    normalized_citations_over_time(mode)
    plot_normalized_articles_over_time(mode)


def _run_per_mode_geographic(mode: str):
    from artificial_intelligence_in_medicine.visualizations.geographic import (
        plot_cartographic_density,
    )

    logger.info(f"[{mode}] Generating geographic visualizations...")
    plot_cartographic_density(mode)


def _run_per_mode_funding(mode: str):
    from artificial_intelligence_in_medicine.visualizations.funding import funding_agency

    logger.info(f"[{mode}] Generating funding visualizations...")
    funding_agency(mode, metric="citations")
    funding_agency(mode, metric="papers")


def _run_per_mode_statistical(mode: str):
    from artificial_intelligence_in_medicine.config import PROCESSED_DATA_DIR
    from artificial_intelligence_in_medicine.visualizations.statistical import (
        scatterplot_with_line_of_best_fit,
    )

    path = PROCESSED_DATA_DIR / mode / "interdisciplinary_nodes.csv"
    if path.exists():
        logger.info(f"[{mode}] Generating statistical scatter plot...")
        scatterplot_with_line_of_best_fit(input_path=path, mode=mode)
    else:
        logger.warning(f"[{mode}] No interdisciplinary_nodes.csv found, skipping scatter.")


# -----------------------------------------------------------------------
# Cross-field comparative visualizations
# -----------------------------------------------------------------------
def _run_comparative_temporal():
    from artificial_intelligence_in_medicine.visualizations.comparative import (
        comparative_article_growth,
        comparative_citation_velocity,
        comparative_constraint_distributions,
        comparative_constraint_over_time,
        comparative_cumulative_growth,
        comparative_degree_distribution,
    )

    logger.info("Generating comparative temporal visualizations...")
    comparative_article_growth()
    comparative_cumulative_growth()
    comparative_citation_velocity()
    comparative_constraint_over_time()
    comparative_constraint_distributions()
    comparative_degree_distribution()


def _run_comparative_geographic():
    from artificial_intelligence_in_medicine.visualizations.comparative import (
        comparative_country_bars,
        comparative_geographic_density,
        country_temporal_growth,
        geographic_constraint_map,
        regional_constraint_heatmap,
    )

    logger.info("Generating comparative geographic visualizations...")
    comparative_country_bars()
    comparative_geographic_density()
    country_temporal_growth()
    geographic_constraint_map()
    regional_constraint_heatmap()


def _run_comparative_funding():
    from artificial_intelligence_in_medicine.visualizations.comparative import (
        comparative_top_agencies,
        funding_diversity_over_time,
        funding_geography_heatmap,
        funding_vs_constraint,
        multi_agency_constraint,
    )

    logger.info("Generating comparative funding visualizations...")
    comparative_top_agencies()
    funding_diversity_over_time()
    funding_vs_constraint()
    funding_geography_heatmap()
    multi_agency_constraint()


def _run_comparative_summary():
    from artificial_intelligence_in_medicine.visualizations.comparative import summary_dashboard

    logger.info("Generating summary dashboard...")
    summary_dashboard()


def _run_graph_visualizations():
    from artificial_intelligence_in_medicine.visualizations.comparative import (
        comparative_graph_statistics,
        per_mode_graph_communities,
        per_mode_graph_structure,
    )

    logger.info("Generating graph structure visualizations...")
    for m in MODES:
        logger.info(f"[{m}] Graph structure + communities...")
        try:
            per_mode_graph_structure(m)
        except Exception as e:
            logger.warning(f"Could not visualize graph structure for {m}: {e}")
        try:
            per_mode_graph_communities(m)
        except Exception as e:
            logger.warning(f"Could not visualize graph communities for {m}: {e}")

    try:
        comparative_graph_statistics()
    except Exception as e:
        logger.warning(f"Could not generate comparative graph statistics: {e}")


# -----------------------------------------------------------------------
# CLI commands
# -----------------------------------------------------------------------
@app.command()
def per_mode(
    mode: str = typer.Option(None, help="Run for a specific mode only"),
):
    """Generate per-mode visualizations (temporal, geographic, funding, statistical)."""
    modes = [mode] if mode else MODES
    for m in modes:
        logger.info(f"=== Per-mode visualizations for {MODE_LABELS.get(m, m)} ===")
        _run_per_mode_temporal(m)
        _run_per_mode_geographic(m)
        _run_per_mode_funding(m)
        _run_per_mode_statistical(m)
    logger.success("Per-mode visualizations complete.")


@app.command()
def temporal():
    """Generate all temporal visualizations (per-mode + comparative)."""
    for m in MODES:
        _run_per_mode_temporal(m)
    _run_comparative_temporal()
    logger.success("Temporal visualizations complete.")


@app.command()
def geographic():
    """Generate all geographic visualizations (per-mode + comparative)."""
    for m in MODES:
        _run_per_mode_geographic(m)
    _run_comparative_geographic()
    logger.success("Geographic visualizations complete.")


@app.command()
def funding():
    """Generate all funding visualizations (per-mode + comparative)."""
    for m in MODES:
        _run_per_mode_funding(m)
    _run_comparative_funding()
    logger.success("Funding visualizations complete.")


@app.command()
def comparative():
    """Generate all cross-field comparative visualizations."""
    _run_comparative_temporal()
    _run_comparative_geographic()
    _run_comparative_funding()
    _run_comparative_summary()
    _run_graph_visualizations()
    logger.success("All comparative visualizations complete.")


@app.command()
def graphs():
    """Generate graph structure visualizations for all modes."""
    _run_graph_visualizations()
    logger.success("Graph visualizations complete.")


@app.command()
def all():
    """Generate ALL visualizations: per-mode + comparative + graphs + summary."""
    logger.info("=== Generating ALL visualizations ===")

    # Per-mode
    for m in MODES:
        logger.info(f"--- {MODE_LABELS.get(m, m)} ---")
        _run_per_mode_temporal(m)
        _run_per_mode_geographic(m)
        _run_per_mode_funding(m)
        _run_per_mode_statistical(m)

    # Graph structure visualizations
    _run_graph_visualizations()

    # Comparative
    _run_comparative_temporal()
    _run_comparative_geographic()
    _run_comparative_funding()

    # Summary
    _run_comparative_summary()

    # Statistical comparison
    from artificial_intelligence_in_medicine.visualizations.statistical import (
        compare_mode_correlations,
    )

    try:
        compare_mode_correlations()
    except Exception as e:
        logger.warning(f"Could not compare mode correlations: {e}")

    logger.success("=== All visualizations generated successfully ===")


if __name__ == "__main__":
    app()
