from generator import heterograph_from_vvg_generator
import pandas as pd
import os 


datatype = "DVC"

segmentation_path = f"../../OCTA_500_RELEVANT_fix/segs"
json_path = f"../../OCTA_500_RELEVANT_fix/json"
void_graph_save_path = f"../../OCTA_500_RELEVANT_fix/void_graph_faz"
hetero_edges_save_path = f"../../OCTA_500_RELEVANT_fix/hetero_edges_faz"
image_path = f"../../OCTA_500_RELEVANT_fix/images"

faz_node_save_path = f"../../OCTA_500_RELEVANT_fix/faz_node"
faz_region_edges_save_path = f"../../OCTA_500_RELEVANT_fix/faz_region_edges"
faz_vessel_edges_save_path = f"../../OCTA_500_RELEVANT_fix/faz_vessel_edges"

Gernerator_from_VVG = heterograph_from_vvg_generator.HeterographFromVVGGenerator(
                                                            seg_path= segmentation_path,
                                                            vvg_path= json_path,
                                                            faz_node= True,
                                                            void_graph_save_path= void_graph_save_path,
                                                            hetero_edges_save_path= hetero_edges_save_path,
                                                            image_path= image_path,
                                                            faz_node_save_path= faz_node_save_path,
                                                            faz_region_edges_save_path= faz_region_edges_save_path,
                                                            faz_vessel_edges_save_path= faz_vessel_edges_save_path,
                                                            debug=False)
Gernerator_from_VVG.save_region_graphs()