import os
import sys

sys.path.append(os.getcwd())
from generator import heterograph_from_vvg_generator

segmentation_path = "../../OCTA_500_RELEVANT_fix/segs"
json_path = "../../OCTA_500_RELEVANT_fix/json"
void_graph_save_path = "../../OCTA_500_RELEVANT_fix/void_graph_faz_more_features"
hetero_edges_save_path = "../../OCTA_500_RELEVANT_fix/hetero_edges"
image_path = "../../OCTA_500_RELEVANT_fix/images"

assert len(os.listdir(segmentation_path)) > 0
assert len(os.listdir(json_path)) > 0
assert len(os.listdir(image_path)) > 0


Gernerator_from_VVG = heterograph_from_vvg_generator.HeterographFromVVGGenerator(
    seg_path=segmentation_path,
    vvg_path=json_path,
    faz_node=True,
    void_graph_save_path=void_graph_save_path,
    hetero_edges_save_path=False,
    image_path=image_path,
    debug=False,
)
Gernerator_from_VVG.save_region_graphs(no_edges=True)
