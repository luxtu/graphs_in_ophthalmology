from generator import heterograph_from_vvg_generator


Gernerator_from_VVG = heterograph_from_vvg_generator.HeterographFromVVGGenerator("/media/data/alex_johannes/octa_data/Cairo/SCP_seg",
                                                              "/media/data/alex_johannes/octa_data/Cairo/SCP_json", 
                                                              "/media/data/alex_johannes/octa_data/Cairo/SCP_void_graph", 
                                                              "/media/data/alex_johannes/octa_data/Cairo/SCP_heter_edges",
                                                              image_path= "/media/data/alex_johannes/octa_data/Cairo/SCP_images",
                                                              debug=False)
Gernerator_from_VVG.save_region_graphs()