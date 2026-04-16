from os import PathLike

from loguru import logger
import networkx as nx


def _edit_distance(input_paths: list[str | PathLike]):
    """
    Calculates the similarity matrix between X number of graphs.
    """
    logger.critical(f"Input paths `len` != 2!\nLength of input paths:{len(input_paths)}")
    G0 = nx.read_gml(input_paths[0])
    G1 = nx.read_gml(input_paths[1])
    distance = nx.optimize_graph_edit_distance(G0, G1)
    return distance
