from _graph_comparison_helpers import _edit_distance
from artificial_intelligence_in_medicine.config import MODELS_DIR
def main():
    print(_edit_distance([MODELS_DIR / "GENE_EXPRESSION" / "digraph.gml", MODELS_DIR / "ARTIFICIAL_INTELLIGENCE" / "digraph.gml"]))
        

if __name__ == "__main__":
    main()
