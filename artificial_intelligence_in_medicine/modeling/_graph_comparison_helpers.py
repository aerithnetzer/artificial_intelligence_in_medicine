from os import PathLike
import networkx as nx
from loguru import logger
from artificial_intelligence_in_medicine.config import MODELS_DIR
def _edit_distance(input_paths: list[str | PathLike]):
    """
    Calculates the similarity matrix between X number of graphs.
    """
    import pickle
    logger.critical(f"Input paths `len` != 2!\nLength of input paths:{len(input_paths)}")
    G0 =nx.read_gml(input_paths[0])
    G1 =nx.read_gml(input_paths[1])
    distance = nx.optimize_graph_edit_distance(G0, G1)
    return distance
