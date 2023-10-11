from generator import heterograph_from_vvg_generator



data_type = "DCP"

segmentation_path = f"/media/data/alex_johannes/octa_data/Cairo/{data_type}_seg"
json_path = f"/media/data/alex_johannes/octa_data/Cairo/{data_type}_json"
void_graph_save_path = f"/media/data/alex_johannes/octa_data/Cairo/{data_type}_void_graph"
hetero_edges_save_path = f"/media/data/alex_johannes/octa_data/Cairo/{data_type}_heter_edges"
image_path = f"/media/data/alex_johannes/octa_data/Cairo/{data_type}_images"


Gernerator_from_VVG = heterograph_from_vvg_generator.HeterographFromVVGGenerator(
                                                            seg_path= segmentation_path,
                                                            json_path= json_path,
                                                            void_graph_save_path= void_graph_save_path,
                                                            hetero_edges_save_path= hetero_edges_save_path,
                                                            image_path= image_path,
                                                            debug=False)
Gernerator_from_VVG.save_region_graphs()