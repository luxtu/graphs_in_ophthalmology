from generator import heterograph_from_vvg_generator
import pandas as pd
import os 


datatype = "SVC"

segmentation_path = f"../../TumNeuro/checked_seg_{datatype}"
json_path = f"../../TumNeuro/checked_seg_{datatype}_graph/json"
void_graph_save_path = f"../../TumNeuro/checked_seg_{datatype}_graph/void_graph_faz"
hetero_edges_save_path = f"../../TumNeuro/checked_seg_{datatype}_graph/hetero_edges_faz"
image_path = f"../../TumNeuro/checked_images_{datatype}/"

faz_node_save_path = f"../../TumNeuro/checked_seg_{datatype}_graph/faz_node"
faz_region_edges_save_path = f"../../TumNeuro/checked_seg_{datatype}_graph/faz_region_edges"
faz_vessel_edges_save_path = f"../../TumNeuro/checked_seg_{datatype}_graph/faz_vessel_edges"

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