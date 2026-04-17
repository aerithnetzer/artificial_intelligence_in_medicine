"""
Cross-field comparative visualizations.

All functions overlay data from AI, Gene Expression, and NULL (random PubMed)
on the same axes to reveal structural differences in citation graph behavior
across temporal, geographic, and funding dimensions.

Every function produces both interactive Plotly HTML and static PNG.
"""

from collections import defaultdict

from loguru import logger
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from scipy import stats as sp_stats

from artificial_intelligence_in_medicine.visualizations.utils import (
    MODE_COLORS,
    MODE_COLORS_LIGHT,
    MODE_LABELS,
    MODES,
    add_citation_count_column,
    add_grant_columns,
    clean_year_column,
    compute_constraint_dict,
    constraint_by_year,
    ensure_comparative_dir,
    load_features,
    load_graph,
    save_plot,
)

# =========================================================================
# TEMPORAL COMPARISONS
# =========================================================================


def comparative_article_growth():
    """
    Overlaid line chart: articles per year for AI, GE, NULL.
    Both raw counts and min-max normalized on same figure (dual y-axis).
    """
    logger.info("Generating comparative article growth...")
    out = ensure_comparative_dir()
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    for mode in MODES:
        df = load_features(mode)
        df = clean_year_column(df)
        counts = df["year"].value_counts().sort_index()

        # Normalized
        mn, mx = counts.min(), counts.max()
        norm = (counts - mn) / (mx - mn) if mx != mn else counts * 0

        fig.add_trace(
            go.Scatter(
                x=counts.index,
                y=counts.values,
                mode="lines+markers",
                name=f"{MODE_LABELS[mode]} (raw)",
                line=dict(color=MODE_COLORS[mode], width=2),
                legendgroup=mode,
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=norm.index,
                y=norm.values,
                mode="lines",
                name=f"{MODE_LABELS[mode]} (normalized)",
                line=dict(color=MODE_COLORS[mode], width=1, dash="dash"),
                legendgroup=mode,
                showlegend=False,
            ),
            secondary_y=True,
        )

    fig.update_layout(
        title="Comparative Article Growth Over Time",
        width=1200,
        height=700,
        template="plotly_white",
    )
    fig.update_xaxes(title_text="Year")
    fig.update_yaxes(title_text="Raw Article Count", secondary_y=False)
    fig.update_yaxes(title_text="Min-Max Normalized", secondary_y=True, range=[0, 1.05])

    save_plot(fig, out / "comparative_article_growth")


def comparative_cumulative_growth():
    """
    Overlaid cumulative article count S-curves, normalized to [0,1].
    """
    logger.info("Generating comparative cumulative growth...")
    out = ensure_comparative_dir()
    fig = go.Figure()

    for mode in MODES:
        df = load_features(mode)
        df = clean_year_column(df)
        counts = df["year"].value_counts().sort_index()
        cumulative = counts.cumsum()
        norm_cum = cumulative / cumulative.iloc[-1]

        fig.add_trace(
            go.Scatter(
                x=norm_cum.index,
                y=norm_cum.values,
                mode="lines",
                name=MODE_LABELS[mode],
                line=dict(color=MODE_COLORS[mode], width=3),
            )
        )

    fig.update_layout(
        title="Cumulative Growth Curves (Normalized)",
        xaxis_title="Year",
        yaxis_title="Fraction of Total Articles",
        width=1100,
        height=700,
        template="plotly_white",
    )

    save_plot(fig, out / "comparative_cumulative_growth")


def comparative_citation_velocity():
    """
    Overlaid line: median citations per article by publication year.
    """
    logger.info("Generating comparative citation velocity...")
    out = ensure_comparative_dir()
    fig = go.Figure()

    for mode in MODES:
        df = load_features(mode)
        df = add_citation_count_column(df)
        df = clean_year_column(df)

        median_per_year = df.groupby("year")["citation_count"].median().sort_index()
        # Only plot years with at least 20 articles
        year_counts = df["year"].value_counts()
        valid_years = year_counts[year_counts >= 20].index
        median_per_year = median_per_year[median_per_year.index.isin(valid_years)]

        fig.add_trace(
            go.Scatter(
                x=median_per_year.index,
                y=median_per_year.values,
                mode="lines+markers",
                name=MODE_LABELS[mode],
                line=dict(color=MODE_COLORS[mode], width=2),
            )
        )

    fig.update_layout(
        title="Median Citations per Article Over Time",
        xaxis_title="Publication Year",
        yaxis_title="Median Citation Count",
        width=1100,
        height=700,
        template="plotly_white",
    )

    save_plot(fig, out / "comparative_citation_velocity")


def comparative_constraint_over_time():
    """
    Overlaid line: median Burt's constraint by year for all 3 fields.
    """
    logger.info("Generating comparative constraint over time...")
    out = ensure_comparative_dir()
    fig = go.Figure()

    for mode in MODES:
        try:
            G = load_graph(mode)
            constraints = compute_constraint_dict(G)
            by_year = constraint_by_year(G, constraints)

            years = sorted(by_year.keys())
            medians = [float(np.median(by_year[y])) for y in years]

            fig.add_trace(
                go.Scatter(
                    x=years,
                    y=medians,
                    mode="lines+markers",
                    name=MODE_LABELS[mode],
                    line=dict(color=MODE_COLORS[mode], width=2),
                )
            )
        except Exception as e:
            logger.warning(f"Could not compute constraint for {mode}: {e}")

    fig.update_layout(
        title="Median Burt's Constraint Over Time",
        xaxis_title="Year",
        yaxis_title="Median Constraint",
        width=1100,
        height=700,
        template="plotly_white",
    )

    save_plot(fig, out / "comparative_constraint_over_time")


def comparative_constraint_distributions():
    """
    Overlaid violin/box plots: constraint distribution per field.
    """
    logger.info("Generating comparative constraint distributions...")
    out = ensure_comparative_dir()

    all_data = []
    for mode in MODES:
        try:
            G = load_graph(mode)
            constraints = compute_constraint_dict(G)
            for node, c in constraints.items():
                if not np.isnan(c):
                    all_data.append({"field": MODE_LABELS[mode], "constraint": c})
        except Exception as e:
            logger.warning(f"Could not load graph for {mode}: {e}")

    if not all_data:
        logger.warning("No constraint data available.")
        return

    df = pd.DataFrame(all_data)

    fig = go.Figure()
    for mode in MODES:
        label = MODE_LABELS[mode]
        subset = df[df["field"] == label]["constraint"]
        if len(subset) == 0:
            continue
        fig.add_trace(
            go.Violin(
                y=subset,
                name=label,
                line_color=MODE_COLORS[mode],
                box_visible=True,
                meanline_visible=True,
            )
        )

    fig.update_layout(
        title="Constraint Distribution by Field",
        yaxis_title="Burt's Constraint",
        width=900,
        height=700,
        template="plotly_white",
    )

    save_plot(fig, out / "comparative_constraint_distributions")


def comparative_degree_distribution():
    """
    Overlaid log-log degree distribution (CCDF) for all 3 fields.
    """
    logger.info("Generating comparative degree distribution...")
    out = ensure_comparative_dir()
    fig = go.Figure()

    for mode in MODES:
        try:
            G = load_graph(mode)
            degrees = sorted([d for _, d in G.degree()], reverse=True)
            if not degrees:
                continue
            n = len(degrees)
            ccdf_x = sorted(set(degrees))
            ccdf_y = [sum(1 for d in degrees if d >= k) / n for k in ccdf_x]

            fig.add_trace(
                go.Scatter(
                    x=ccdf_x,
                    y=ccdf_y,
                    mode="lines",
                    name=MODE_LABELS[mode],
                    line=dict(color=MODE_COLORS[mode], width=2),
                )
            )
        except Exception as e:
            logger.warning(f"Could not compute degree dist for {mode}: {e}")

    fig.update_layout(
        title="Complementary Cumulative Degree Distribution (CCDF)",
        xaxis_title="Degree (k)",
        yaxis_title="P(X >= k)",
        xaxis_type="log",
        yaxis_type="log",
        width=1000,
        height=700,
        template="plotly_white",
    )

    save_plot(fig, out / "comparative_degree_distribution")


# =========================================================================
# GEOGRAPHIC COMPARISONS
# =========================================================================


def comparative_country_bars():
    """
    Grouped horizontal bar: top 20 countries across all fields.
    """
    logger.info("Generating comparative country bars...")
    out = ensure_comparative_dir()

    all_country_data = []
    for mode in MODES:
        df = load_features(mode)
        counts = df["matched_country"].value_counts()
        for country, count in counts.items():
            all_country_data.append(
                {
                    "country": country,
                    "count": count,
                    "field": MODE_LABELS[mode],
                }
            )

    df_all = pd.DataFrame(all_country_data)

    # Get top 20 countries by total count across all fields
    total_by_country = df_all.groupby("country")["count"].sum().nlargest(20).index
    df_top = df_all[df_all["country"].isin(total_by_country)]

    # Sort by total count
    country_order = (
        df_top.groupby("country")["count"].sum().sort_values(ascending=True).index.tolist()
    )

    fig = px.bar(
        df_top,
        x="count",
        y="country",
        color="field",
        orientation="h",
        barmode="group",
        category_orders={"country": country_order},
        color_discrete_map={MODE_LABELS[m]: MODE_COLORS[m] for m in MODES},
        title="Top 20 Countries by Publication Count",
        labels={"count": "Number of Publications", "country": "Country"},
    )
    fig.update_layout(width=1100, height=800, template="plotly_white")

    save_plot(fig, out / "comparative_country_bars")


def comparative_geographic_density():
    """
    Side-by-side density maps for AI, GE, NULL on world projection.
    """
    logger.info("Generating comparative geographic density...")
    out = ensure_comparative_dir()

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[MODE_LABELS[m] for m in MODES],
        specs=[[{"type": "map"}, {"type": "map"}, {"type": "map"}]],
    )

    for i, mode in enumerate(MODES, 1):
        df = load_features(mode)
        df = df.dropna(subset=["matched_lat", "matched_lon"])

        fig.add_trace(
            go.Densitymap(
                lat=df["matched_lat"],
                lon=df["matched_lon"],
                radius=5,
                showscale=(i == 3),
                colorscale="Viridis",
            ),
            row=1,
            col=i,
        )

    fig.update_layout(
        title_text="Geographic Distribution of Publications by Field",
        width=2000,
        height=600,
        map=dict(style="open-street-map", center=dict(lat=20, lon=0), zoom=0.8),
        map2=dict(style="open-street-map", center=dict(lat=20, lon=0), zoom=0.8),
        map3=dict(style="open-street-map", center=dict(lat=20, lon=0), zoom=0.8),
    )

    save_plot(fig, out / "comparative_geographic_density", width=2000, height=600)


def country_temporal_growth():
    """
    Faceted line chart: top 10 countries' publication trajectories by field.
    """
    logger.info("Generating country temporal growth...")
    out = ensure_comparative_dir()

    all_data = []
    for mode in MODES:
        df = load_features(mode)
        df = clean_year_column(df)
        df = df.dropna(subset=["matched_country"])
        year_country = df.groupby(["year", "matched_country"]).size().reset_index(name="count")
        year_country["field"] = MODE_LABELS[mode]
        all_data.append(year_country)

    df_all = pd.concat(all_data, ignore_index=True)

    # Top 10 countries overall
    top_countries = df_all.groupby("matched_country")["count"].sum().nlargest(10).index.tolist()
    df_top = df_all[df_all["matched_country"].isin(top_countries)]

    fig = px.line(
        df_top,
        x="year",
        y="count",
        color="field",
        facet_col="matched_country",
        facet_col_wrap=5,
        color_discrete_map={MODE_LABELS[m]: MODE_COLORS[m] for m in MODES},
        title="Publication Growth by Country and Field (Top 10 Countries)",
        labels={"count": "Articles", "year": "Year", "matched_country": "Country"},
    )
    fig.update_layout(width=1600, height=800, template="plotly_white")
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

    save_plot(fig, out / "country_temporal_growth", width=1600, height=800)


def geographic_constraint_map():
    """
    Choropleth: median constraint by country, faceted by field.
    """
    logger.info("Generating geographic constraint map...")
    out = ensure_comparative_dir()

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[MODE_LABELS[m] for m in MODES],
        specs=[[{"type": "choropleth"}, {"type": "choropleth"}, {"type": "choropleth"}]],
    )

    for i, mode in enumerate(MODES, 1):
        try:
            G = load_graph(mode)
            constraints = compute_constraint_dict(G)

            country_constraints = defaultdict(list)
            for node, c in constraints.items():
                if np.isnan(c):
                    continue
                country = G.nodes[node].get("matched_country")
                if country:
                    country_constraints[country].append(c)

            countries = list(country_constraints.keys())
            medians = [float(np.median(country_constraints[c])) for c in countries]

            fig.add_trace(
                go.Choropleth(
                    locationmode="country names",
                    locations=countries,
                    z=medians,
                    colorscale="Viridis",
                    showscale=(i == 3),
                    colorbar=dict(title="Median<br>Constraint") if i == 3 else None,
                ),
                row=1,
                col=i,
            )
        except Exception as e:
            logger.warning(f"Could not compute for {mode}: {e}")

    fig.update_layout(
        title_text="Median Burt's Constraint by Country",
        width=2000,
        height=500,
    )
    fig.update_geos(
        showframe=False,
        showcoastlines=True,
        projection_type="natural earth",
    )

    save_plot(fig, out / "geographic_constraint_map", width=2000, height=500)


def regional_constraint_heatmap():
    """
    Heatmap: year x top-15-country, cell = median constraint.
    One heatmap per field, arranged vertically.
    """
    logger.info("Generating regional constraint heatmap...")
    out = ensure_comparative_dir()

    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=[MODE_LABELS[m] for m in MODES],
        vertical_spacing=0.08,
    )

    for row, mode in enumerate(MODES, 1):
        try:
            G = load_graph(mode)
            constraints = compute_constraint_dict(G)

            records = []
            for node, c in constraints.items():
                if np.isnan(c):
                    continue
                country = G.nodes[node].get("matched_country")
                year = G.nodes[node].get("year")
                if country and year:
                    try:
                        records.append(
                            {
                                "country": country,
                                "year": int(float(year)),
                                "constraint": c,
                            }
                        )
                    except (ValueError, TypeError):
                        pass

            df = pd.DataFrame(records)
            if df.empty:
                continue

            # Top 15 countries by count
            top = df["country"].value_counts().nlargest(15).index
            df = df[df["country"].isin(top)]

            pivot = df.pivot_table(
                values="constraint",
                index="country",
                columns="year",
                aggfunc="median",
            ).sort_index()

            fig.add_trace(
                go.Heatmap(
                    z=pivot.values,
                    x=pivot.columns.astype(str),
                    y=pivot.index,
                    colorscale="Viridis",
                    showscale=(row == 1),
                    colorbar=dict(title="Median<br>Constraint") if row == 1 else None,
                ),
                row=row,
                col=1,
            )
        except Exception as e:
            logger.warning(f"Could not compute for {mode}: {e}")

    fig.update_layout(
        title_text="Regional Constraint Evolution (Year x Country)",
        width=1400,
        height=1200,
        template="plotly_white",
    )

    save_plot(fig, out / "regional_constraint_heatmap", width=1400, height=1200)


# =========================================================================
# FUNDING COMPARISONS
# =========================================================================


def comparative_top_agencies():
    """
    Grouped bar: top 10 agencies across AI/GE/NULL by total citation count.
    """
    logger.info("Generating comparative top agencies...")
    out = ensure_comparative_dir()

    all_data = []
    for mode in MODES:
        df = load_features(mode)
        df = add_citation_count_column(df)
        df = df[df["grant_list"].apply(lambda x: isinstance(x, list) and len(x) > 0)]
        if df.empty:
            continue
        grants = df.explode("grant_list")
        grants["agency"] = grants["grant_list"].apply(lambda x: x if isinstance(x, str) else None)
        grants = grants.dropna(subset=["agency"])
        agg = grants.groupby("agency")["citation_count"].sum().reset_index()
        agg["field"] = MODE_LABELS[mode]
        all_data.append(agg)

    if not all_data:
        logger.warning("No funding data available.")
        return

    df_all = pd.concat(all_data, ignore_index=True)
    top_agencies = df_all.groupby("agency")["citation_count"].sum().nlargest(10).index
    df_top = df_all[df_all["agency"].isin(top_agencies)]

    # Sort
    agency_order = (
        df_top.groupby("agency")["citation_count"].sum().sort_values(ascending=True).index.tolist()
    )

    fig = px.bar(
        df_top,
        x="citation_count",
        y="agency",
        color="field",
        orientation="h",
        barmode="group",
        category_orders={"agency": agency_order},
        color_discrete_map={MODE_LABELS[m]: MODE_COLORS[m] for m in MODES},
        title="Top 10 Funding Agencies by Total Citations Across Fields",
        labels={"citation_count": "Total Citations", "agency": "Agency"},
    )
    fig.update_layout(width=1200, height=700, template="plotly_white")

    save_plot(fig, out / "comparative_top_agencies")


def funding_diversity_over_time():
    """
    Overlaid line: number of unique funding agencies per year, per field.
    """
    logger.info("Generating funding diversity over time...")
    out = ensure_comparative_dir()
    fig = go.Figure()

    for mode in MODES:
        df = load_features(mode)
        df = clean_year_column(df)
        df = df[df["grant_list"].apply(lambda x: isinstance(x, list) and len(x) > 0)]
        if df.empty:
            continue
        grants = df.explode("grant_list")
        grants["agency"] = grants["grant_list"].apply(lambda x: x if isinstance(x, str) else None)
        grants = grants.dropna(subset=["agency"])
        diversity = grants.groupby("year")["agency"].nunique().sort_index()

        fig.add_trace(
            go.Scatter(
                x=diversity.index,
                y=diversity.values,
                mode="lines+markers",
                name=MODE_LABELS[mode],
                line=dict(color=MODE_COLORS[mode], width=2),
            )
        )

    fig.update_layout(
        title="Funding Diversity Over Time (Unique Agencies per Year)",
        xaxis_title="Year",
        yaxis_title="Number of Unique Agencies",
        width=1100,
        height=700,
        template="plotly_white",
    )

    save_plot(fig, out / "funding_diversity_over_time")


def funding_vs_constraint():
    """
    Violin: constraint distribution for funded vs unfunded articles, per field.
    """
    logger.info("Generating funding vs constraint...")
    out = ensure_comparative_dir()

    all_data = []
    for mode in MODES:
        try:
            G = load_graph(mode)
            constraints = compute_constraint_dict(G)
            df = load_features(mode)
            df = add_grant_columns(df)
            df["pmid"] = df["pmid"].astype(str)

            for _, row in df.iterrows():
                pmid = str(row["pmid"])
                c = constraints.get(pmid)
                if c is None or np.isnan(c):
                    continue
                all_data.append(
                    {
                        "field": MODE_LABELS[mode],
                        "funded": "Funded" if row["has_funding"] else "Unfunded",
                        "constraint": c,
                    }
                )
        except Exception as e:
            logger.warning(f"Could not process {mode}: {e}")

    if not all_data:
        logger.warning("No data for funding vs constraint.")
        return

    df_all = pd.DataFrame(all_data)

    fig = px.violin(
        df_all,
        x="field",
        y="constraint",
        color="funded",
        box=True,
        title="Constraint Distribution: Funded vs Unfunded Articles",
        labels={"constraint": "Burt's Constraint", "field": "Field"},
    )
    fig.update_layout(width=1100, height=700, template="plotly_white")

    save_plot(fig, out / "funding_vs_constraint")


def funding_geography_heatmap():
    """
    Heatmap: top 10 agencies x top 10 countries, cell = article count.
    One per field, arranged vertically.
    """
    logger.info("Generating funding-geography heatmap...")
    out = ensure_comparative_dir()

    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=[MODE_LABELS[m] for m in MODES],
        vertical_spacing=0.08,
    )

    for row, mode in enumerate(MODES, 1):
        df = load_features(mode)
        df = df[df["grant_list"].apply(lambda x: isinstance(x, list) and len(x) > 0)]
        df = df.dropna(subset=["matched_country"])
        if df.empty:
            continue

        grants = df.explode("grant_list")
        grants["agency"] = grants["grant_list"].apply(lambda x: x if isinstance(x, str) else None)
        grants = grants.dropna(subset=["agency"])

        top_agencies = grants["agency"].value_counts().nlargest(10).index
        top_countries = grants["matched_country"].value_counts().nlargest(10).index

        filtered = grants[
            grants["agency"].isin(top_agencies) & grants["matched_country"].isin(top_countries)
        ]
        pivot = filtered.pivot_table(
            values="pmid",
            index="agency",
            columns="matched_country",
            aggfunc="count",
            fill_value=0,
        )

        fig.add_trace(
            go.Heatmap(
                z=pivot.values,
                x=pivot.columns.tolist(),
                y=pivot.index.tolist(),
                colorscale="Blues",
                showscale=(row == 1),
                colorbar=dict(title="Article<br>Count") if row == 1 else None,
            ),
            row=row,
            col=1,
        )

    fig.update_layout(
        title_text="Funding Agency x Country (Top 10 Each)",
        width=1200,
        height=1400,
        template="plotly_white",
    )

    save_plot(fig, out / "funding_geography_heatmap", width=1200, height=1400)


def multi_agency_constraint():
    """
    Violin: constraint distribution grouped by number of funding sources (0, 1, 2, 3+).
    One panel per field.
    """
    logger.info("Generating multi-agency constraint analysis...")
    out = ensure_comparative_dir()

    all_data = []
    for mode in MODES:
        try:
            G = load_graph(mode)
            constraints = compute_constraint_dict(G)
            df = load_features(mode)
            df = add_grant_columns(df)
            df["pmid"] = df["pmid"].astype(str)

            for _, row in df.iterrows():
                pmid = str(row["pmid"])
                c = constraints.get(pmid)
                if c is None or np.isnan(c):
                    continue
                n_sources = row["num_funding_sources"]
                bucket = str(n_sources) if n_sources < 3 else "3+"
                all_data.append(
                    {
                        "field": MODE_LABELS[mode],
                        "funding_sources": bucket,
                        "constraint": c,
                    }
                )
        except Exception as e:
            logger.warning(f"Could not process {mode}: {e}")

    if not all_data:
        logger.warning("No data for multi-agency constraint.")
        return

    df_all = pd.DataFrame(all_data)

    fig = px.violin(
        df_all,
        x="funding_sources",
        y="constraint",
        color="field",
        box=True,
        color_discrete_map={MODE_LABELS[m]: MODE_COLORS[m] for m in MODES},
        title="Constraint by Number of Funding Sources",
        labels={
            "constraint": "Burt's Constraint",
            "funding_sources": "Number of Funding Agencies",
        },
        category_orders={"funding_sources": ["0", "1", "2", "3+"]},
    )
    fig.update_layout(width=1200, height=700, template="plotly_white")

    save_plot(fig, out / "multi_agency_constraint")


# =========================================================================
# SUMMARY DASHBOARD
# =========================================================================


def summary_dashboard():
    """
    Multi-panel summary figure combining key metrics across all fields.
    4 panels: growth curves, citation velocity, constraint over time, country bars.
    """
    logger.info("Generating summary dashboard...")
    out = ensure_comparative_dir()

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "Cumulative Growth (Normalized)",
            "Median Citations per Article",
            "Median Constraint Over Time",
            "Top 10 Countries by Publications",
        ],
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "bar"}],
        ],
    )

    # Panel 1: Cumulative growth
    for mode in MODES:
        df = load_features(mode)
        df = clean_year_column(df)
        counts = df["year"].value_counts().sort_index()
        cumulative = counts.cumsum()
        norm_cum = cumulative / cumulative.iloc[-1]

        fig.add_trace(
            go.Scatter(
                x=norm_cum.index,
                y=norm_cum.values,
                mode="lines",
                name=MODE_LABELS[mode],
                line=dict(color=MODE_COLORS[mode], width=2),
                legendgroup=mode,
            ),
            row=1,
            col=1,
        )

    # Panel 2: Citation velocity
    for mode in MODES:
        df = load_features(mode)
        df = add_citation_count_column(df)
        df = clean_year_column(df)
        median_cit = df.groupby("year")["citation_count"].median().sort_index()
        year_n = df["year"].value_counts()
        valid = year_n[year_n >= 20].index
        median_cit = median_cit[median_cit.index.isin(valid)]

        fig.add_trace(
            go.Scatter(
                x=median_cit.index,
                y=median_cit.values,
                mode="lines",
                name=MODE_LABELS[mode],
                line=dict(color=MODE_COLORS[mode], width=2),
                legendgroup=mode,
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    # Panel 3: Constraint over time
    for mode in MODES:
        try:
            G = load_graph(mode)
            constraints = compute_constraint_dict(G)
            by_year = constraint_by_year(G, constraints)
            years = sorted(by_year.keys())
            medians = [float(np.median(by_year[y])) for y in years]

            fig.add_trace(
                go.Scatter(
                    x=years,
                    y=medians,
                    mode="lines",
                    name=MODE_LABELS[mode],
                    line=dict(color=MODE_COLORS[mode], width=2),
                    legendgroup=mode,
                    showlegend=False,
                ),
                row=2,
                col=1,
            )
        except Exception:
            pass

    # Panel 4: Top 10 countries (total across all fields)
    all_countries = {}
    for mode in MODES:
        df = load_features(mode)
        counts = df["matched_country"].value_counts()
        for country, count in counts.items():
            all_countries[country] = all_countries.get(country, 0) + count

    top10 = sorted(all_countries.items(), key=lambda x: -x[1])[:10]
    top10_names = [c[0] for c in top10]

    for mode in MODES:
        df = load_features(mode)
        counts = df["matched_country"].value_counts()
        vals = [counts.get(c, 0) for c in top10_names]

        fig.add_trace(
            go.Bar(
                x=top10_names,
                y=vals,
                name=MODE_LABELS[mode],
                marker_color=MODE_COLORS[mode],
                legendgroup=mode,
                showlegend=False,
            ),
            row=2,
            col=2,
        )

    fig.update_layout(
        title_text="Citation Graph Structure: Cross-Field Summary",
        width=1600,
        height=1000,
        template="plotly_white",
        barmode="group",
    )

    save_plot(fig, out / "summary_dashboard", width=1600, height=1000)


# =========================================================================
# GRAPH STRUCTURE VISUALIZATIONS
# =========================================================================


def per_mode_graph_structure(mode: str, max_nodes: int = 5000):
    """
    Visualize the citation graph structure for a single mode.

    Renders the network with:
    - Spring layout positioning
    - Nodes colored by Burt's constraint (quantile bins)
    - Node size proportional to degree
    - Edges drawn as thin lines

    For large graphs, samples to max_nodes highest-degree nodes.
    """
    import networkx as nx

    from artificial_intelligence_in_medicine.config import FIGURES_DIR

    logger.info(f"Generating graph structure visualization for {mode}...")
    out = FIGURES_DIR / mode
    out.mkdir(parents=True, exist_ok=True)

    G = load_graph(mode)

    # For very large graphs, keep only the top-degree nodes for readability
    if G.number_of_nodes() > max_nodes:
        logger.info(f"Graph has {G.number_of_nodes()} nodes, sampling top {max_nodes} by degree")
        top_nodes = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:max_nodes]
        top_ids = [n for n, _ in top_nodes]
        G = G.subgraph(top_ids).copy()

    # Remove isolates for cleaner layout
    isolates = list(nx.isolates(G))
    G.remove_nodes_from(isolates)
    logger.info(f"Visualizing {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Compute constraint
    if isinstance(G, nx.DiGraph):
        g_und = G.to_undirected()
    else:
        g_und = G
    constraints = nx.constraint(g_und)

    # Layout
    logger.info("Computing spring layout...")
    pos = nx.spring_layout(G, seed=42, iterations=50, k=1.0 / (G.number_of_nodes() ** 0.3))

    # Edges
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=0.15, color="rgba(150,150,150,0.3)"),
        hoverinfo="none",
        showlegend=False,
    )

    # Nodes: color by constraint, size by degree
    nodes = list(G.nodes())
    degrees = dict(G.degree())
    node_x = [pos[n][0] for n in nodes]
    node_y = [pos[n][1] for n in nodes]
    node_color = [constraints.get(n, 0.0) for n in nodes]
    # Clamp NaN to 0
    node_color = [0.0 if np.isnan(c) else c for c in node_color]
    node_size = [3 + 2 * np.log1p(degrees[n]) for n in nodes]
    node_hover = [
        f"<b>{G.nodes[n].get('title', n)}</b><br>"
        f"Degree: {degrees[n]}<br>"
        f"Constraint: {constraints.get(n, 0.0):.4f}<br>"
        f"Year: {G.nodes[n].get('year', '?')}"
        for n in nodes
    ]

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        hoverinfo="text",
        text=node_hover,
        marker=dict(
            size=node_size,
            color=node_color,
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Constraint", thickness=15),
            line=dict(width=0.3, color="white"),
        ),
        showlegend=False,
    )

    label = MODE_LABELS[mode]
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=f"{label} Citation Graph ({G.number_of_nodes()} nodes, "
            f"{G.number_of_edges()} edges)",
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=50),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            width=1400,
            height=1000,
            plot_bgcolor="white",
        ),
    )

    save_plot(fig, out / "graph_structure", width=1400, height=1000, scale=3)
    return fig


def per_mode_graph_communities(mode: str, max_nodes: int = 5000):
    """
    Visualize graph structure colored by Louvain community membership.

    Nodes colored by community, sized by degree.
    """
    import networkx as nx
    from networkx.algorithms.community import greedy_modularity_communities

    from artificial_intelligence_in_medicine.config import FIGURES_DIR

    logger.info(f"Generating community-colored graph for {mode}...")
    out = FIGURES_DIR / mode
    out.mkdir(parents=True, exist_ok=True)

    G = load_graph(mode)

    # Sample if large
    if G.number_of_nodes() > max_nodes:
        top_nodes = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:max_nodes]
        top_ids = [n for n, _ in top_nodes]
        G = G.subgraph(top_ids).copy()

    # Remove isolates and convert to undirected for community detection
    isolates = list(nx.isolates(G))
    G.remove_nodes_from(isolates)

    if isinstance(G, nx.DiGraph):
        g_und = G.to_undirected()
    else:
        g_und = G.copy()

    # Community detection
    logger.info("Running community detection...")
    communities_list = list(greedy_modularity_communities(g_und))
    membership = {}
    for i, comm in enumerate(communities_list):
        for node in comm:
            membership[node] = i
    n_communities = len(communities_list)
    modularity = nx.algorithms.community.modularity(g_und, communities_list)
    logger.info(f"Found {n_communities} communities, Q={modularity:.4f}")

    # Color palette
    import plotly.express as px_colors

    palette = px_colors.colors.qualitative.Set1 + px_colors.colors.qualitative.Set3
    comm_colors = {i: palette[i % len(palette)] for i in range(n_communities)}

    # Layout
    logger.info("Computing spring layout...")
    pos = nx.spring_layout(G, seed=42, iterations=50, k=1.0 / (G.number_of_nodes() ** 0.3))

    # Edges
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=0.15, color="rgba(150,150,150,0.25)"),
        hoverinfo="none",
        showlegend=False,
    )

    # Group nodes by community for separate traces (gives legend)
    degrees = dict(G.degree())
    traces = [edge_trace]

    # Only show top 15 communities in legend, rest grouped as "Other"
    comm_sizes = {}
    for node in G.nodes():
        c = membership.get(node, -1)
        comm_sizes[c] = comm_sizes.get(c, 0) + 1
    top_comms = sorted(comm_sizes.keys(), key=lambda c: -comm_sizes[c])[:15]
    top_set = set(top_comms)

    for comm_id in top_comms:
        c_nodes = [n for n in G.nodes() if membership.get(n, -1) == comm_id]
        if not c_nodes:
            continue
        traces.append(
            go.Scatter(
                x=[pos[n][0] for n in c_nodes],
                y=[pos[n][1] for n in c_nodes],
                mode="markers",
                hoverinfo="text",
                text=[
                    f"<b>{G.nodes[n].get('title', n)}</b><br>"
                    f"Community: {comm_id} (n={comm_sizes[comm_id]})<br>"
                    f"Degree: {degrees[n]}"
                    for n in c_nodes
                ],
                marker=dict(
                    size=[3 + 2 * np.log1p(degrees[n]) for n in c_nodes],
                    color=comm_colors[comm_id],
                    line=dict(width=0.3, color="white"),
                ),
                name=f"C{comm_id} (n={comm_sizes[comm_id]})",
                legendgroup=str(comm_id),
            )
        )

    # "Other" communities
    other_nodes = [n for n in G.nodes() if membership.get(n, -1) not in top_set]
    if other_nodes:
        traces.append(
            go.Scatter(
                x=[pos[n][0] for n in other_nodes],
                y=[pos[n][1] for n in other_nodes],
                mode="markers",
                hoverinfo="text",
                text=[
                    f"<b>{G.nodes[n].get('title', n)}</b><br>"
                    f"Community: {membership.get(n, '?')}<br>"
                    f"Degree: {degrees[n]}"
                    for n in other_nodes
                ],
                marker=dict(
                    size=[3 + 2 * np.log1p(degrees[n]) for n in other_nodes],
                    color="#cccccc",
                    line=dict(width=0.3, color="white"),
                ),
                name=f"Other ({len(other_nodes)} nodes)",
            )
        )

    label = MODE_LABELS[mode]
    fig = go.Figure(
        data=traces,
        layout=go.Layout(
            title=f"{label} Citation Graph: Community Structure "
            f"({n_communities} communities, Q={modularity:.3f})",
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=50),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            width=1400,
            height=1000,
            plot_bgcolor="white",
            legend=dict(
                title="Community",
                x=1.02,
                y=1,
                font=dict(size=9),
            ),
        ),
    )

    save_plot(fig, out / "graph_communities", width=1400, height=1000, scale=3)
    return fig


def comparative_graph_statistics():
    """
    Side-by-side summary table + bar chart of key graph metrics for all 3 fields:
    nodes, edges, density, avg degree, clustering coefficient, # components.
    """
    import networkx as nx

    logger.info("Generating comparative graph statistics...")
    out = ensure_comparative_dir()

    metrics = []
    for mode in MODES:
        try:
            G = load_graph(mode)
            # Convert to undirected for certain metrics
            g_und = G.to_undirected() if isinstance(G, nx.DiGraph) else G

            n_nodes = G.number_of_nodes()
            n_edges = G.number_of_edges()
            density = nx.density(G)
            avg_degree = sum(d for _, d in G.degree()) / max(n_nodes, 1)
            n_components = (
                nx.number_weakly_connected_components(G)
                if G.is_directed()
                else nx.number_connected_components(G)
            )

            # Clustering on undirected (can be slow on large graphs; sample if needed)
            if n_nodes > 50000:
                sample_nodes = list(g_und.nodes())[:10000]
                avg_clustering = nx.average_clustering(g_und, nodes=sample_nodes)
            else:
                avg_clustering = nx.average_clustering(g_und)

            metrics.append(
                {
                    "Field": MODE_LABELS[mode],
                    "Nodes": n_nodes,
                    "Edges": n_edges,
                    "Density": density,
                    "Avg Degree": avg_degree,
                    "Avg Clustering": avg_clustering,
                    "Components": n_components,
                }
            )
        except Exception as e:
            logger.warning(f"Could not compute metrics for {mode}: {e}")

    if not metrics:
        logger.warning("No graph metrics computed.")
        return

    df = pd.DataFrame(metrics)

    # Create multi-panel figure
    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=["Nodes", "Edges", "Density", "Avg Degree", "Avg Clustering", "Components"],
    )

    metric_cols = ["Nodes", "Edges", "Density", "Avg Degree", "Avg Clustering", "Components"]
    positions = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3)]

    for (r, c), col in zip(positions, metric_cols):
        fig.add_trace(
            go.Bar(
                x=df["Field"],
                y=df[col],
                marker_color=[
                    MODE_COLORS[m] for m in MODES if MODE_LABELS[m] in df["Field"].values
                ],
                showlegend=False,
                text=[f"{v:.4g}" for v in df[col]],
                textposition="auto",
            ),
            row=r,
            col=c,
        )

    fig.update_layout(
        title_text="Comparative Graph Statistics",
        width=1400,
        height=800,
        template="plotly_white",
    )

    save_plot(fig, out / "comparative_graph_statistics", width=1400, height=800)

    # Also save as CSV
    csv_path = out / "graph_statistics.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved graph statistics CSV to {csv_path}")

    return df


# =========================================================================
# STREAM 1: MeSH TOPIC EVOLUTION
# =========================================================================


def _extract_mesh_terms(mesh_headings) -> list[str]:
    """
    Extract flat list of MeSH descriptor names from the mesh_headings field.
    Handles both list-of-dicts (with 'DescriptorName' key) and list-of-strings.
    """
    if not isinstance(mesh_headings, list):
        return []
    terms = []
    for item in mesh_headings:
        if isinstance(item, dict):
            name = item.get("DescriptorName", "")
            if isinstance(name, str) and name:
                terms.append(name)
        elif isinstance(item, str) and item:
            terms.append(item)
    return terms


def comparative_mesh_entropy_over_time():
    """
    Shannon entropy of MeSH term frequency distribution per year, overlaid
    for all 3 fields.

    Higher entropy = more diverse research topics in that year.
    Shows whether AI-in-medicine is broadening its topical scope compared
    to the mature Gene Expression field.
    """
    logger.info("Generating comparative MeSH entropy over time...")
    out = ensure_comparative_dir()
    fig = go.Figure()

    for mode in MODES:
        df = load_features(mode)
        df = clean_year_column(df)

        # Flatten MeSH headings per article
        df["_mesh_flat"] = df["mesh_headings"].apply(_extract_mesh_terms)

        entropy_by_year = {}
        for year, group in df.groupby("year"):
            all_terms = []
            for terms in group["_mesh_flat"]:
                all_terms.extend(terms)
            if not all_terms:
                continue
            # Frequency distribution
            term_counts = pd.Series(all_terms).value_counts(normalize=True)
            # Shannon entropy: H = -sum(p * log2(p))
            entropy = float(-(term_counts * np.log2(term_counts)).sum())
            entropy_by_year[year] = entropy

        if not entropy_by_year:
            continue

        years = sorted(entropy_by_year.keys())
        entropies = [entropy_by_year[y] for y in years]

        fig.add_trace(
            go.Scatter(
                x=years,
                y=entropies,
                mode="lines+markers",
                name=MODE_LABELS[mode],
                line=dict(color=MODE_COLORS[mode], width=2),
            )
        )

    fig.update_layout(
        title="MeSH Topic Diversity Over Time (Shannon Entropy)",
        xaxis_title="Year",
        yaxis_title="Shannon Entropy (bits)",
        width=1100,
        height=700,
        template="plotly_white",
    )

    save_plot(fig, out / "comparative_mesh_entropy")


def comparative_mesh_composition_shifts():
    """
    Stacked area chart showing proportion of top-10 MeSH terms over time
    for each field. Reveals dominant topic transitions (e.g., the shift from
    'Neural Networks' to 'Deep Learning').

    Produces one subplot per field (3 rows).
    """
    logger.info("Generating comparative MeSH composition shifts...")
    out = ensure_comparative_dir()

    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=[MODE_LABELS[m] for m in MODES],
        vertical_spacing=0.08,
    )

    for row, mode in enumerate(MODES, 1):
        df = load_features(mode)
        df = clean_year_column(df)
        df["_mesh_flat"] = df["mesh_headings"].apply(_extract_mesh_terms)

        # Find overall top 10 MeSH terms for this mode
        all_terms = []
        for terms in df["_mesh_flat"]:
            all_terms.extend(terms)
        top_terms = pd.Series(all_terms).value_counts().head(10).index.tolist()

        # Compute yearly proportions for top terms
        year_range = sorted(df["year"].unique())
        term_proportions = {term: [] for term in top_terms}
        for year in year_range:
            year_terms = []
            for terms in df[df["year"] == year]["_mesh_flat"]:
                year_terms.extend(terms)
            total = len(year_terms) if year_terms else 1
            for term in top_terms:
                count = sum(1 for t in year_terms if t == term)
                term_proportions[term].append(count / total)

        for term in top_terms:
            fig.add_trace(
                go.Scatter(
                    x=year_range,
                    y=term_proportions[term],
                    mode="lines",
                    name=term if row == 1 else None,
                    stackgroup="one",
                    showlegend=(row == 1),
                    legendgroup=term,
                ),
                row=row,
                col=1,
            )

    fig.update_layout(
        title_text="MeSH Term Composition Over Time (Top 10 Terms per Field)",
        width=1200,
        height=1200,
        template="plotly_white",
    )
    fig.update_yaxes(title_text="Proportion", row=2, col=1)

    save_plot(fig, out / "comparative_mesh_composition", width=1200, height=1200)


# =========================================================================
# STREAM 2: INSTITUTION-LEVEL ANALYSIS
# =========================================================================


def comparative_top_institutions():
    """
    Grouped horizontal bar chart: top-15 institutions by article count
    across all 3 fields. Uses matched_name from ROR matching.
    """
    logger.info("Generating comparative top institutions...")
    out = ensure_comparative_dir()

    all_data = []
    for mode in MODES:
        df = load_features(mode)
        df = df.dropna(subset=["matched_name"])
        counts = df["matched_name"].value_counts()
        for inst, count in counts.items():
            all_data.append({"institution": inst, "count": count, "field": MODE_LABELS[mode]})

    df_all = pd.DataFrame(all_data)
    if df_all.empty:
        logger.warning("No institution data available.")
        return

    # Top 15 institutions by total count
    total_by_inst = df_all.groupby("institution")["count"].sum().nlargest(15).index
    df_top = df_all[df_all["institution"].isin(total_by_inst)]
    inst_order = (
        df_top.groupby("institution")["count"].sum().sort_values(ascending=True).index.tolist()
    )

    fig = px.bar(
        df_top,
        x="count",
        y="institution",
        color="field",
        orientation="h",
        barmode="group",
        category_orders={"institution": inst_order},
        color_discrete_map={MODE_LABELS[m]: MODE_COLORS[m] for m in MODES},
        title="Top 15 Institutions by Publication Count",
        labels={"count": "Number of Publications", "institution": "Institution"},
    )
    fig.update_layout(width=1200, height=800, template="plotly_white")

    save_plot(fig, out / "comparative_top_institutions", width=1200, height=800)


def comparative_institutional_concentration():
    """
    Gini coefficient of institution publication counts per year, overlaid
    for all 3 fields. Higher Gini = more concentrated in fewer institutions.

    Shows whether AI research is becoming more or less institutionally
    concentrated over time.
    """
    logger.info("Generating comparative institutional concentration...")
    out = ensure_comparative_dir()
    fig = go.Figure()

    def _gini(values):
        """Compute Gini coefficient for a 1D array of non-negative values."""
        v = np.sort(np.asarray(values, dtype=float))
        n = len(v)
        if n < 2 or v.sum() == 0:
            return np.nan
        index = np.arange(1, n + 1)
        return float((2 * (index * v).sum()) / (n * v.sum()) - (n + 1) / n)

    for mode in MODES:
        df = load_features(mode)
        df = clean_year_column(df)
        df = df.dropna(subset=["matched_name"])

        gini_by_year = {}
        for year, group in df.groupby("year"):
            inst_counts = group["matched_name"].value_counts().values
            if len(inst_counts) < 5:
                continue
            gini_by_year[year] = _gini(inst_counts)

        if not gini_by_year:
            continue

        years = sorted(gini_by_year.keys())
        ginis = [gini_by_year[y] for y in years]

        fig.add_trace(
            go.Scatter(
                x=years,
                y=ginis,
                mode="lines+markers",
                name=MODE_LABELS[mode],
                line=dict(color=MODE_COLORS[mode], width=2),
            )
        )

    fig.update_layout(
        title="Institutional Concentration Over Time (Gini Coefficient)",
        xaxis_title="Year",
        yaxis_title="Gini Coefficient",
        yaxis_range=[0, 1],
        width=1100,
        height=700,
        template="plotly_white",
    )

    save_plot(fig, out / "comparative_institutional_concentration")


# =========================================================================
# STREAM 4: GEOGRAPHIC CONCENTRATION
# =========================================================================


def comparative_geographic_concentration():
    """
    Herfindahl-Hirschman Index (HHI) of country-level publication shares
    per year, overlaid for all 3 fields.

    Lower HHI = more geographically dispersed research output.
    """
    logger.info("Generating comparative geographic concentration (HHI)...")
    out = ensure_comparative_dir()
    fig = go.Figure()

    for mode in MODES:
        df = load_features(mode)
        df = clean_year_column(df)
        df = df.dropna(subset=["matched_country"])

        hhi_by_year = {}
        for year, group in df.groupby("year"):
            counts = group["matched_country"].value_counts()
            total = counts.sum()
            if total < 10:
                continue
            shares = counts / total
            hhi_by_year[year] = float((shares**2).sum())

        if not hhi_by_year:
            continue

        years = sorted(hhi_by_year.keys())
        hhis = [hhi_by_year[y] for y in years]

        fig.add_trace(
            go.Scatter(
                x=years,
                y=hhis,
                mode="lines+markers",
                name=MODE_LABELS[mode],
                line=dict(color=MODE_COLORS[mode], width=2),
            )
        )

    fig.update_layout(
        title="Geographic Concentration Over Time (Herfindahl-Hirschman Index)",
        xaxis_title="Year",
        yaxis_title="HHI (lower = more dispersed)",
        width=1100,
        height=700,
        template="plotly_white",
    )

    save_plot(fig, out / "comparative_geographic_concentration")
