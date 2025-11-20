from loguru import logger
import networkx as nx
import igraph as ig


def nx_to_igraph(nx_graph):
    """Convert a NetworkX graph to an igraph.Graph, preserving key attributes."""
    logger.info("Converting NetworkX graph to igraph...")
    G = nx_graph.to_undirected()
    g = ig.Graph()
    g.add_vertices(list(G.nodes()))
    g.add_edges(list(G.edges()))

    # copy node attributes
    for attr_name in next(iter(G.nodes(data=True)))[1].keys():
        g.vs[attr_name] = [G.nodes[n].get(attr_name) for n in G.nodes()]

    # copy edge attributes
    for attr_name in next(iter(G.edges(data=True)))[2].keys():
        g.es[attr_name] = [G.edges[e].get(attr_name) for e in G.edges()]

    return g
