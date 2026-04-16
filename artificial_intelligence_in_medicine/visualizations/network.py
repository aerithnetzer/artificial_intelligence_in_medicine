"""
Network graph visualizations.

Functions for plotting citation graphs, community detection results,
constraint-colored networks, and semantic similarity graphs.
"""

from collections import defaultdict
from pathlib import Path

from loguru import logger
import matplotlib.cm as cm
from matplotlib.colors import to_rgba
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from artificial_intelligence_in_medicine.config import FIGURES_DIR
from artificial_intelligence_in_medicine.visualizations.utils import (
    COMMUNITY_PALETTE,
    save_plot,
)


def plot_communities(G: nx.Graph, mode: str):
    """
    Interactive Plotly network visualization colored by community,
    with TF-IDF-derived community labels and citation-scaled node sizes.
    """
    output_path = FIGURES_DIR / mode / "communities.html"
    communities = nx.get_node_attributes(G, "community")
    community_names = {}

    # Build community labels using top TF-IDF terms
    titles_attr = nx.get_node_attributes(G, "title")
    if titles_attr:
        community_titles = defaultdict(list)
        for n, data in G.nodes(data=True):
            community_id = data.get("community")
            title = data.get("title")
            if community_id is not None and title:
                community_titles[community_id].append(title)

        for community_id, titles in community_titles.items():
            if titles:
                try:
                    vectorizer = TfidfVectorizer(stop_words="english", max_features=5)
                    vectorizer.fit_transform(titles)
                    top_terms = vectorizer.get_feature_names_out()
                    community_names[community_id] = ", ".join(top_terms)
                except ValueError:
                    community_names[community_id] = f"Community {community_id}"
            else:
                community_names[community_id] = f"Community {community_id}"
    else:
        community_names = {i: f"Community {i}" for i in set(communities.values())}

    # Assign community labels to nodes
    nx.set_node_attributes(
        G,
        {
            n: community_names.get(communities.get(n), f"Community {communities.get(n)}")
            for n in G.nodes()
        },
        "community_label",
    )

    # Layout
    pos = nx.spring_layout(G, seed=42, k=0.15 if G.number_of_nodes() < 1000 else None)
    x_coords = [pos[n][0] for n in G.nodes()]
    y_coords = [pos[n][1] for n in G.nodes()]

    # Node properties
    node_titles = [data.get("title", f"Node {n}") for n, data in G.nodes(data=True)]
    node_colors = [
        COMMUNITY_PALETTE[data.get("community", 0) % len(COMMUNITY_PALETTE)]
        for _, data in G.nodes(data=True)
    ]

    # Edge positions
    edge_x, edge_y = [], []
    edge_shapes = []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_shapes.append(
            dict(
                type="line",
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                line=dict(color="#888", width=0.5),
                opacity=0.7,
                layer="below",
            )
        )

    # Node sizes based on citations
    citation_counts = [len(data.get("cited_by") or []) for _, data in G.nodes(data=True)]
    node_sizes = [10 + 2 * np.log(c + 1) for c in citation_counts]

    # Hover text
    node_hover_texts = [
        f"Title: {title}<br>"
        f"Community: {data.get('community_label', 'Unknown')}<br>"
        f"Cited by: {len(data.get('cited_by') or [])}"
        for (n, data), title in zip(G.nodes(data=True), node_titles)
    ]

    node_trace = go.Scatter(
        x=x_coords,
        y=y_coords,
        mode="markers",
        hoverinfo="text",
        text=node_hover_texts,
        marker=dict(size=node_sizes, color=node_colors, line=dict(width=2, color="white")),
    )
    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.5, color="#888"),
        hoverinfo="none",
        mode="lines",
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=f"Citation Network Communities ({mode})<br>"
            f"{len(set(communities.values()))} communities",
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            width=1800,
            height=1200,
            shapes=edge_shapes,
        ),
    )

    save_plot(fig, output_path, width=1200, height=800, scale=5)
    return G


def plot_constraint(G: nx.Graph, mode: str, constraints: dict):
    """
    Interactive network visualization colored by Burt's structural constraint.
    Also saves a constraint_table.csv.
    """
    import pandas as pd

    output_path = FIGURES_DIR / mode / "constraints.html"
    nx.set_node_attributes(G, {n: constraints[n] for n in G.nodes()}, "constraint")
    logger.info("Constraints calculated")

    df = pd.DataFrame(
        {
            "pmid": list(nx.get_node_attributes(G, "title").keys()),
            "title": list(nx.get_node_attributes(G, "title").values()),
            "constraint": list(nx.get_node_attributes(G, "constraint").values()),
        }
    )
    df.to_csv(FIGURES_DIR / mode / "constraint_table.csv")

    # Layout
    pos = nx.spring_layout(G, seed=42, k=0.15 if G.number_of_nodes() < 1000 else None)
    x_coords = [pos[n][0] for n in G.nodes()]
    y_coords = [pos[n][1] for n in G.nodes()]
    node_titles = [data.get("title", f"Node {n}") for n, data in G.nodes(data=True)]

    # Edges
    edge_x, edge_y, edge_shapes = [], [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_shapes.append(
            dict(
                type="line",
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                line=dict(color="#888", width=0.5),
                opacity=0.7,
                layer="below",
            )
        )

    node_trace = go.Scatter(
        x=x_coords,
        y=y_coords,
        mode="markers",
        hoverinfo="text",
        text=[
            f"Title: {title}<br>Constraint: {constraints[n]:.4f}"
            for (n, _), title in zip(G.nodes(data=True), node_titles)
        ],
        marker=dict(
            color=[constraints[n] for n in G.nodes()],
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Constraint"),
            line=dict(width=2, color="white"),
        ),
    )
    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.5, color="#888"),
        hoverinfo="none",
        mode="lines",
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title="Citation Network (Colored by Burt's Constraint)",
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            width=1800,
            height=1200,
            shapes=edge_shapes,
        ),
    )

    save_plot(fig, output_path, width=1200, height=800, scale=5)


def plot_semantic_graph(
    g: nx.Graph,
    mode: str,
    embedding_attr: str = "embedding",
    title_attr: str = "title",
    min_degree: int = 23,
    top_k: int = 4,
    min_similarity: float = 0.6,
    layout: str = "spring",
    seed: int = 42,
):
    """
    Build and plot a semantic similarity graph from node embeddings.
    Nodes are colored by normalized structural constraint.
    """
    from networkx.algorithms.structuralholes import constraint

    # Filter nodes by connectivity
    eligible_nodes = {n for n in g.nodes() if g.degree(n) > min_degree}
    if len(eligible_nodes) < 2:
        raise ValueError(
            f"Need at least 2 nodes with degree > {min_degree} to build a semantic graph."
        )

    # Collect embeddings
    nodes, embs = [], []
    for n in eligible_nodes:
        emb = g.nodes[n].get(embedding_attr)
        if emb is None:
            continue
        nodes.append(n)
        embs.append(np.asarray(emb, dtype=np.float32))

    if len(nodes) < 2:
        raise ValueError("Need at least 2 eligible nodes with embeddings.")

    X = np.vstack(embs)

    # Cosine similarity
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    Xn = X / norms
    S = (Xn @ Xn.T).astype(np.float32)
    np.fill_diagonal(S, -1.0)

    # Build semantic graph
    sg = nx.DiGraph()
    sg.add_nodes_from(nodes)
    for i in range(len(nodes)):
        idx = np.argpartition(S[i], -top_k)[-top_k:]
        idx = idx[np.argsort(S[i, idx])[::-1]]
        for j in idx:
            sim = float(S[i, j])
            if sim >= min_similarity:
                sg.add_edge(nodes[i], nodes[j], weight=sim)

    # Drop isolates
    sg.remove_nodes_from([n for n in sg.nodes() if sg.in_degree(n) == 0 and sg.out_degree(n) == 0])
    if sg.number_of_nodes() == 0:
        raise ValueError("No nodes left after filtering isolates.")

    # Layout
    if layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(sg)
    else:
        pos = nx.spring_layout(sg, seed=seed, iterations=200)

    constraints = constraint(sg)

    # Edge traces
    weights = np.array([d["weight"] for _, _, d in sg.edges(data=True)])
    w_min, w_max = weights.min(), weights.max()
    widths = 0.5 + 2.5 * (weights - w_min) / (w_max - w_min + 1e-8)

    edge_traces = []
    for (u, v, d), w in zip(sg.edges(data=True), widths):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_traces.append(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line=dict(width=w, color="rgba(150,150,150,0.6)"),
                hoverinfo="none",
                showlegend=False,
            )
        )

    # Node trace with constraint coloring
    c_vals = np.array([constraints.get(n, 0) for n in sg.nodes()])
    c_min, c_max = c_vals.min(), c_vals.max()
    c_norm = (c_vals - c_min) / (c_max - c_min + 1e-8)

    node_x, node_y, hovertext = [], [], []
    for n in sg.nodes():
        x, y = pos[n]
        node_x.append(x)
        node_y.append(y)
        hovertext.append(g.nodes[n].get(title_attr, str(n)))

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        hovertext=hovertext,
        hoverinfo="text",
        marker=dict(
            size=6,
            color=c_norm.tolist(),
            colorscale="Greys",
            reversescale=True,
            line=dict(width=0.5, color="black"),
            colorbar=dict(title="Constraint", thickness=10),
        ),
        name="nodes",
    )

    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        title=f"Semantic Graph (deg>{min_degree}, top_k={top_k}, min_sim={min_similarity})",
        showlegend=False,
        hovermode="closest",
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
    )

    save_plot(fig, FIGURES_DIR / mode / "semanticgraph")
    return fig, sg


def visualize_graph(G, output_path, n_bins: int = 5):
    """
    ForceAtlas2 layout with quantile-binned structural constraint coloring.
    """
    logger.info("Computing layout")
    pos = nx.spring_layout(G, seed=42, iterations=100)

    # Structural constraint
    logger.info("Computing structural constraint")
    if isinstance(G, nx.DiGraph):
        constraint_dict = nx.constraint(G.to_undirected())
    else:
        constraint_dict = nx.constraint(G)

    nodes = list(G.nodes())
    constraint_values = np.array([constraint_dict.get(n, np.nan) for n in nodes])
    valid_mask = ~np.isnan(constraint_values)
    valid_values = constraint_values[valid_mask]

    if len(valid_values) == 0:
        raise ValueError("All constraint values are NaN. Graph too sparse.")

    # Quantile binning
    bins = np.quantile(valid_values, np.linspace(0, 1, n_bins + 1))
    binned = np.full(len(nodes), -1)
    binned[valid_mask] = np.digitize(valid_values, bins[1:-1])

    categorical_palette = [
        "#e41a1c",
        "#377eb8",
        "#4daf4a",
        "#984ea3",
        "#ff7f00",
        "#ffff33",
        "#a65628",
        "#f781bf",
    ]

    # Edge trace
    edge_x, edge_y = [], []
    for u, v in tqdm(G.edges(), desc="Building edge trace"):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=0.4, color="#bbbbbb"),
        hoverinfo="none",
        showlegend=False,
    )

    # Node traces per bin
    node_traces = []
    for bin_idx in range(n_bins):
        bin_nodes = [nodes[i] for i in range(len(nodes)) if binned[i] == bin_idx]
        if not bin_nodes:
            continue
        x_vals = [pos[n][0] for n in bin_nodes]
        y_vals = [pos[n][1] for n in bin_nodes]
        hover_text = [
            f"<b>{G.nodes[n].get('title', n)}</b><br>Constraint: {constraint_dict.get(n, 0):.4f}"
            for n in bin_nodes
        ]
        lower, upper = bins[bin_idx], bins[bin_idx + 1]
        label = f"{lower:.3f} - {upper:.3f}"
        node_traces.append(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="markers",
                text=hover_text,
                hoverinfo="text",
                marker=dict(
                    size=9,
                    color=categorical_palette[bin_idx % len(categorical_palette)],
                    line=dict(width=1, color="black"),
                ),
                name=label,
                showlegend=True,
            )
        )

    # NaN nodes
    nan_nodes = [nodes[i] for i in range(len(nodes)) if binned[i] == -1]
    if nan_nodes:
        node_traces.append(
            go.Scatter(
                x=[pos[n][0] for n in nan_nodes],
                y=[pos[n][1] for n in nan_nodes],
                mode="markers",
                hoverinfo="text",
                text=[
                    f"<b>{G.nodes[n].get('title', n)}</b><br>Constraint: Undefined"
                    for n in nan_nodes
                ],
                marker=dict(size=9, color="#999999", line=dict(width=1, color="black")),
                name="Undefined (deg < 2)",
                showlegend=True,
            )
        )

    fig = go.Figure(
        data=[edge_trace] + node_traces,
        layout=go.Layout(
            title="Graph Colored by Structural Constraint (Quantile Bins)",
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            width=1800,
            height=1200,
            legend=dict(title="Constraint Range", x=1.02, y=1),
        ),
    )

    save_plot(fig, output_path, width=1200, height=800, scale=5)


def visualize_communities(
    G: nx.Graph,
    mode: str = "semantic",
    resolution: float = 1.0,
    random_state: int = 42,
    node_scale: float = 1.0,
    edge_scale: float = 1.0,
    title: str | None = None,
    figsize: tuple = (14, 10),
    output_path: str = "communities.png",
    dpi: int = 150,
) -> dict:
    """
    Detect Louvain communities, build a weighted community metagraph,
    and save a matplotlib visualization on a dark background.
    """
    if mode not in ("semantic", "citation"):
        raise ValueError(f"mode must be 'semantic' or 'citation', got {mode!r}")

    # Detect communities
    communities = nx.community.louvain_communities(
        G, weight="weight", resolution=resolution, seed=random_state
    )
    node_community: dict = {}
    partition: dict[int, list] = {}
    for comm_id, members in enumerate(communities):
        partition[comm_id] = list(members)
        for node in members:
            node_community[node] = comm_id

    community_ids = sorted(partition.keys())
    n_communities = len(community_ids)
    modularity = nx.community.modularity(G, communities, weight="weight")

    # Build the community graph
    CG = nx.Graph()
    CG.add_nodes_from(community_ids)
    for u, v, data in G.edges(data=True):
        cu, cv = node_community[u], node_community[v]
        if cu == cv:
            continue
        w = data.get("weight", 1.0)
        if CG.has_edge(cu, cv):
            CG[cu][cv]["weight"] += w
        else:
            CG.add_edge(cu, cv, weight=w)

    # Per-community stats
    comm_sizes = {c: len(members) for c, members in partition.items()}
    max_size = max(comm_sizes.values()) if comm_sizes else 1
    intra_stats: dict[int, float] = {}
    for comm, members in partition.items():
        subg = G.subgraph(members)
        n = len(members)
        if mode == "semantic":
            weights = [d.get("weight", 0.0) for _, _, d in subg.edges(data=True)]
            intra_stats[comm] = float(np.mean(weights)) if weights else 0.0
        else:
            possible = n * (n - 1) / 2 if n > 1 else 1
            intra_stats[comm] = subg.number_of_edges() / possible

    # Layout & draw
    fig, ax = plt.subplots(figsize=figsize, facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    pos = nx.spring_layout(CG, weight="weight", seed=random_state, k=2.5)
    cmap_mpl = cm.get_cmap("plasma", n_communities)
    node_colors = {c: cmap_mpl(i) for i, c in enumerate(community_ids)}

    # Edges
    edge_weights = nx.get_edge_attributes(CG, "weight")
    if edge_weights:
        max_ew = max(edge_weights.values())
        min_ew = min(edge_weights.values())
        ew_range = max_ew - min_ew if max_ew != min_ew else 1.0
    else:
        min_ew, ew_range = 1.0, 1.0

    for (u, v), w in edge_weights.items():
        norm_w = (w - min_ew) / ew_range
        lw = (0.5 + norm_w * 6.0) * edge_scale
        alpha = 0.15 + norm_w * 0.65
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        ax.plot([x0, x1], [y0, y1], color="#c0c0c0", linewidth=lw, alpha=alpha, zorder=1)
        if norm_w > 0.6:
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            label = f"{w:.2f}" if mode == "semantic" else f"{int(w)}"
            ax.text(
                mx,
                my,
                label,
                fontsize=7,
                color="white",
                ha="center",
                va="center",
                zorder=5,
                bbox=dict(boxstyle="round,pad=0.15", fc="#0d1117", alpha=0.6, ec="none"),
            )

    # Nodes
    for comm in community_ids:
        x, y = pos[comm]
        size = 400 + (comm_sizes[comm] / max_size) * 2200 * node_scale
        color = node_colors[comm]
        stat = intra_stats[comm]
        glow_color = list(to_rgba(color))
        glow_color[3] = 0.18
        ax.scatter(x, y, s=size * 2.2, color=[glow_color], zorder=2)
        ax.scatter(x, y, s=size, color=[color], edgecolors="white", linewidths=0.8, zorder=3)
        label_lines = [f"C{comm}", f"n={comm_sizes[comm]}"]
        label_lines.append(f"sim={stat:.2f}" if mode == "semantic" else f"dens={stat:.2f}")
        ax.text(
            x,
            y,
            "\n".join(label_lines),
            fontsize=8,
            color="white",
            ha="center",
            va="center",
            fontweight="bold",
            zorder=4,
            multialignment="center",
        )

    # Title & legend
    if title is None:
        mode_label = "Semantic Similarity" if mode == "semantic" else "Citation"
        title = (
            f"{mode_label} Community Graph  |  {n_communities} communities  |  Q={modularity:.4f}"
        )
    ax.set_title(title, color="white", fontsize=13, pad=14, fontweight="bold")
    ax.axis("off")

    legend_patches = [
        mpatches.Patch(
            color=node_colors[c],
            label=(
                f"C{c}: {comm_sizes[c]} nodes | "
                + (
                    f"avg sim {intra_stats[c]:.2f}"
                    if mode == "semantic"
                    else f"density {intra_stats[c]:.2f}"
                )
            ),
        )
        for c in community_ids
    ]
    ax.legend(
        handles=legend_patches,
        loc="lower left",
        fontsize=8,
        framealpha=0.3,
        facecolor="#1c2128",
        edgecolor="#444",
        labelcolor="white",
        title=(
            "Edge weight = sum cosine sim" if mode == "semantic" else "Edge weight = sum citations"
        ),
        title_fontsize=8,
    )

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())

    # Also save interactive HTML version
    html_path = Path(output_path).with_suffix(".html")
    _community_graph_to_plotly(
        CG, pos, comm_sizes, intra_stats, node_colors, community_ids, mode, modularity, html_path
    )

    plt.close(fig)

    return {
        "communities": node_community,
        "community_graph": CG,
        "partition": partition,
        "modularity": modularity,
        "intra_stats": intra_stats,
    }


def _community_graph_to_plotly(
    CG, pos, comm_sizes, intra_stats, node_colors, community_ids, mode, modularity, output_path
):
    """Helper to save an interactive Plotly version of the community metagraph."""
    edge_x, edge_y = [], []
    for u, v in CG.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=1, color="rgba(150,150,150,0.5)"),
        hoverinfo="none",
        showlegend=False,
    )

    node_x = [pos[c][0] for c in community_ids]
    node_y = [pos[c][1] for c in community_ids]
    node_size = [10 + 30 * (comm_sizes[c] / max(comm_sizes.values())) for c in community_ids]
    hover = [
        f"Community {c}<br>Size: {comm_sizes[c]}<br>"
        f"{'Avg sim' if mode == 'semantic' else 'Density'}: {intra_stats[c]:.3f}"
        for c in community_ids
    ]

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        hovertext=hover,
        hoverinfo="text",
        text=[f"C{c}" for c in community_ids],
        textposition="top center",
        marker=dict(
            size=node_size,
            color=[f"rgba{to_rgba(node_colors[c])}" for c in community_ids],
            line=dict(width=1, color="white"),
        ),
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=f"Community Metagraph ({mode}) | Q={modularity:.4f}",
            showlegend=False,
            hovermode="closest",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            width=1000,
            height=800,
        ),
    )
    fig.write_html(str(output_path))
