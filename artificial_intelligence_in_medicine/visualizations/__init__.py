"""
Visualizations package for artificial_intelligence_in_medicine.

Re-exports all public visualization functions for backward compatibility
and convenient imports.
"""

# Network visualizations
# Comparative (cross-field) visualizations
from artificial_intelligence_in_medicine.visualizations.comparative import (
    comparative_article_growth,
    comparative_citation_velocity,
    comparative_constraint_distributions,
    comparative_constraint_over_time,
    comparative_country_bars,
    comparative_cumulative_growth,
    comparative_degree_distribution,
    comparative_geographic_concentration,
    comparative_geographic_density,
    comparative_graph_statistics,
    comparative_institutional_concentration,
    comparative_mesh_composition_shifts,
    comparative_mesh_entropy_over_time,
    comparative_top_agencies,
    comparative_top_institutions,
    country_temporal_growth,
    funding_diversity_over_time,
    funding_geography_heatmap,
    funding_vs_constraint,
    geographic_constraint_map,
    multi_agency_constraint,
    per_mode_graph_communities,
    per_mode_graph_structure,
    regional_constraint_heatmap,
    summary_dashboard,
)

# Funding visualizations
from artificial_intelligence_in_medicine.visualizations.funding import (
    funding_agency,
    funding_agency_number_of_papers,
)

# Geographic visualizations
from artificial_intelligence_in_medicine.visualizations.geographic import (
    plot_cartographic_density,
    plot_geographic_kde_by_year,
    plot_lat_lon_scatter,
)
from artificial_intelligence_in_medicine.visualizations.network import (
    plot_communities,
    plot_constraint,
    plot_semantic_graph,
    visualize_communities,
    visualize_graph,
)

# Statistical visualizations
from artificial_intelligence_in_medicine.visualizations.statistical import (
    compare_independent_correlations,
    compare_mode_correlations,
    cross_field_citation_tests,
    cross_field_geographic_tests,
    cross_field_growth_rate_tests,
    fisher_r_to_z,
    funding_citation_tests,
    run_all_statistical_tests,
    scatterplot_with_line_of_best_fit,
    test_pearson_correlation_diff,
)

# Temporal visualizations
from artificial_intelligence_in_medicine.visualizations.temporal import (
    horizontal_timeline,
    normalized_articles_over_time,
    normalized_citations_over_time,
    plot_communities_vertical_barchart,
    plot_horizontal_timeline,
    plot_normalized_articles_over_time,
)

# Utility functions
from artificial_intelligence_in_medicine.visualizations.utils import (
    MODE_COLORS,
    MODE_LABELS,
    MODES,
    load_features,
    load_features_all,
    load_graph,
    save_plot,
)

__all__ = [
    # Network
    "plot_communities",
    "plot_constraint",
    "plot_semantic_graph",
    "visualize_communities",
    "visualize_graph",
    # Temporal
    "normalized_citations_over_time",
    "plot_normalized_articles_over_time",
    "normalized_articles_over_time",
    "plot_horizontal_timeline",
    "plot_communities_vertical_barchart",
    "horizontal_timeline",
    # Geographic
    "plot_cartographic_density",
    "plot_geographic_kde_by_year",
    "plot_lat_lon_scatter",
    # Funding
    "funding_agency",
    "funding_agency_number_of_papers",
    # Statistical
    "scatterplot_with_line_of_best_fit",
    "compare_mode_correlations",
    "compare_independent_correlations",
    "fisher_r_to_z",
    "test_pearson_correlation_diff",
    "funding_citation_tests",
    "cross_field_citation_tests",
    "cross_field_geographic_tests",
    "cross_field_growth_rate_tests",
    "run_all_statistical_tests",
    # Comparative
    "comparative_article_growth",
    "comparative_cumulative_growth",
    "comparative_citation_velocity",
    "comparative_constraint_over_time",
    "comparative_constraint_distributions",
    "comparative_degree_distribution",
    "comparative_country_bars",
    "comparative_geographic_density",
    "country_temporal_growth",
    "geographic_constraint_map",
    "regional_constraint_heatmap",
    "comparative_top_agencies",
    "funding_diversity_over_time",
    "funding_vs_constraint",
    "funding_geography_heatmap",
    "multi_agency_constraint",
    "summary_dashboard",
    # New comparative analyses
    "comparative_mesh_entropy_over_time",
    "comparative_mesh_composition_shifts",
    "comparative_top_institutions",
    "comparative_institutional_concentration",
    "comparative_geographic_concentration",
    # Graph structure
    "per_mode_graph_structure",
    "per_mode_graph_communities",
    "comparative_graph_statistics",
    # Utils
    "MODES",
    "MODE_COLORS",
    "MODE_LABELS",
    "load_features",
    "load_features_all",
    "load_graph",
    "save_plot",
]
