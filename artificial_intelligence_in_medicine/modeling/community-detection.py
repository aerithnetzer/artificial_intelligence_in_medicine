from pathlib import Path
import pickle

import igraph
from igraph import Graph
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
import typer

from artificial_intelligence_in_medicine.config import FIGURES_DIR, MODELS_DIR

app = typer.Typer()

MODE = "ARTIFICIAL_INTELLIGENCE"


@app.command()
def main(
    graph_path: Path = MODELS_DIR / MODE / "citation_model.pkl",
    output_path: Path = FIGURES_DIR / MODE / "community_detection.html",
):
    with open(graph_path, "rb") as f:
        G = pickle.load(f)
    g = Graph.from_networkx(G)

    original_node_count = g.vcount()

    if MODE == "GENE_EXPRESSION":
        inflection_point = 25
    else:
        inflection_point = 23
    graph = g.simplify()
    graph = graph.as_undirected(combine_edges="first")
    graph = graph.subgraph([v for v in graph.vs if v.degree() > inflection_point])

    # Log after filtering
    high_degree_nodes = graph.vcount()
    percent_high_degree = (
        100 * high_degree_nodes / original_node_count if original_node_count > 0 else 0
    )
    print(
        f"Inflection point degree: {inflection_point}. "
        f"Nodes above inflection: {high_degree_nodes} "
        f"({percent_high_degree:.2f}% of total nodes: {original_node_count})"
    )
    # Remove isolated nodes produced by the high-degree filtering
    if graph.vcount() > 0:
        isolated = [v.index for v in graph.vs if v.degree() == 0]
        if isolated:
            graph.delete_vertices(isolated)

    if hasattr(G, "graph") and "year" in G.graph:
        graph["year"] = G.graph["year"]

    communities = None

    if graph.vcount() > 0 and graph.ecount() > 0:
        try:
            communities = graph.community_leiden(objective_function="modularity")
        except Exception as e:
            print(f"Error during community detection: {e}")
    elif graph.vcount() > 0:
        communities = igraph.VertexClustering(graph, membership=list(range(graph.vcount())))

    if communities and graph.vcount() > 0:
        graph.vs["community"] = communities.membership

        # Generate community names
        community_names = {}
        if "title" in graph.vs.attributes():
            from collections import defaultdict

            community_titles = defaultdict(list)
            for i, title in enumerate(graph.vs["title"]):
                community_id = communities.membership[i]
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
            community_names = {i: f"Community {i}" for i in range(len(communities))}

        # Save community labels to graph
        graph.vs["community_label"] = [
            community_names.get(cid, f"Community {cid}") for cid in graph.vs["community"]
        ]

        # Save graph
        output_graph_path = graph_path.with_name(f"{graph_path.stem}_with_communities.pkl")
        with open(output_graph_path, "wb") as f:
            pickle.dump(graph, f)

        # Also save community name mapping
        with open(output_path.with_name("community_labels.pkl"), "wb") as f:
            pickle.dump(community_names, f)

        print("Graph and labels saved. Visualizing...")

        try:
            layout = (
                graph.layout_fruchterman_reingold()
                if graph.vcount() <= 1000
                else graph.layout_auto()
            )

            x_coords = [pos[0] for pos in layout]
            y_coords = [pos[1] for pos in layout]
            node_titles = (
                graph.vs["title"]
                if "title" in graph.vs.attributes()
                else [f"Node {i}" for i in range(graph.vcount())]
            )

            colors_discrete = px.colors.qualitative.Set1 + px.colors.qualitative.Set3
            node_colors = [
                colors_discrete[communities.membership[i] % len(colors_discrete)]
                for i in range(graph.vcount())
            ]

            edge_x, edge_y = [], []
            edge_shapes = []
            for edge in graph.es:
                x0, y0 = layout[edge.source]
                x1, y1 = layout[edge.target]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                # Add arrow annotation for direction
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
            edge_trace = go.Scatter(
                x=edge_x,
                y=edge_y,
                line=dict(width=0.5, color="#888"),
                hoverinfo="none",
                mode="lines",
            )

            citations_attr = graph.vs["cited_by"]
            for v in graph.vs:
                if v["cited_by"] is None:
                    v["cited_by"] = []
            # Ensure citations_attr is updated after fixing None values
            citations_attr = graph.vs["cited_by"]
            citation_counts = [(len(c)) for c in citations_attr]
            node_sizes = [10 + 2 * np.log(c + 1) for c in citation_counts]
            print(len(node_sizes), "node sizes calculated")
            node_trace = go.Scatter(
                x=x_coords,
                y=y_coords,
                mode="markers",
                hoverinfo="text",
                text=[
                    f"Title: {title}<br>Community: {graph.vs[i]['community_label']}<br>Cited by: {len(graph.vs[i]['cited_by'])}"
                    for i, title in enumerate(node_titles)
                ],
                marker=dict(size=node_sizes, color=node_colors, line=dict(width=2, color="white")),
            )

            fig = go.Figure(
                data=[edge_trace, node_trace],
                layout=go.Layout(
                    title=f"Interactive Directed Graph Visualization<br>{len(communities)} communities, Modularity: {communities.modularity:.4f}",
                    showlegend=False,
                    hovermode="closest",
                    margin=dict(b=20, l=5, r=5, t=40),
                    annotations=[
                        dict(
                            text="Hover over nodes to see titles",
                            showarrow=False,
                            xref="paper",
                            yref="paper",
                            x=0.005,
                            y=-0.002,
                            xanchor="left",
                            yanchor="bottom",
                            font=dict(color="gray", size=12),
                        )
                    ],
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    width=1800,
                    height=1200,
                    shapes=edge_shapes,
                ),
            )

            fig.write_html(output_path)

            # Save static PNG (requires `pip install -U kaleido`)
            png_output_path = output_path.with_suffix(".png")
            fig.write_image(png_output_path, width=1200, height=800, scale=5)

        except Exception as e:
            print(f"Could not plot communities: {e}")


if __name__ == "__main__":
    app()
