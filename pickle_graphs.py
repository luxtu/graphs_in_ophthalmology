from loader import hetero_graph_loader, hetero_graph_loader_faz
import pickle
from utils import prep

data_type = "DCP"
faz_node_bool = True


vessel_graph_path = f"../data/{data_type}_vessel_graph"
label_file = "../data/splits"

mode_cv = "cv"
mode_final_test = "final_test"
octa_dr_dict = {"Healthy": 0, "DM": 0, "PDR": 2, "Early NPDR": 1, "Late NPDR": 1}
label_names = ["Healthy/DM", "NPDR", "PDR"]

if not faz_node_bool:
    void_graph_path = f"../data/{data_type}_void_graph"
    hetero_edges_path = f"../data/{data_type}_heter_edges"

    cv_pickle = f"../data/{data_type}_{mode_cv}_dataset.pkl"
    cv_dataset = hetero_graph_loader.HeteroGraphLoaderTorch(graph_path_1=vessel_graph_path,
                                                        graph_path_2=void_graph_path,
                                                        hetero_edges_path_12=hetero_edges_path,
                                                        mode = mode_cv,
                                                        label_file = label_file, 
                                                        line_graph_1 =True, 
                                                        class_dict = octa_dr_dict,
                                                        pickle_file = cv_pickle #f"../{data_type}_{mode_train}_dataset_faz.pkl" # f"../{data_type}_{mode_train}_dataset_faz.pkl"
                                                        )

    final_test_pickle = f"../data/{data_type}_{mode_final_test}_dataset.pkl"
    final_test_dataset = hetero_graph_loader.HeteroGraphLoaderTorch(graph_path_1=vessel_graph_path,
                                                            graph_path_2=void_graph_path,
                                                            hetero_edges_path_12=hetero_edges_path,
                                                            mode = mode_final_test,
                                                            label_file = label_file, 
                                                            line_graph_1 =True, 
                                                            class_dict = octa_dr_dict,
                                                            pickle_file = final_test_pickle #f"../{data_type}_{mode_train}_dataset_faz.pkl" # f"../{data_type}_{mode_train}_dataset_faz.pkl"
                                                            )

else:
    void_graph_path = f"../data/{data_type}_void_graph_faz_node_more_features" # _more_features
    hetero_edges_path = f"../data/{data_type}_hetero_edges_faz_node"
    faz_node_path = f"../data/{data_type}_faz_nodes"
    faz_region_edge_path = f"../data/{data_type}_faz_region_edges"
    faz_vessel_edge_path = f"../data/{data_type}_faz_vessel_edges"

    cv_pickle = f"../data/{data_type}_{mode_cv}_dataset_faz_isol_not_removed_more_features.pkl"  # _isol_not_removed    _more_features
    cv_dataset = hetero_graph_loader_faz.HeteroGraphLoaderTorch(graph_path_1=vessel_graph_path,
                                                            graph_path_2=void_graph_path,
                                                            graph_path_3=faz_node_path,
                                                            hetero_edges_path_12=hetero_edges_path,
                                                            hetero_edges_path_13=faz_vessel_edge_path,
                                                            hetero_edges_path_23=faz_region_edge_path,
                                                            mode = mode_cv,
                                                            label_file = label_file, 
                                                            line_graph_1 =True, 
                                                            class_dict = octa_dr_dict,
                                                            pickle_file = cv_pickle, #f"../{data_type}_{mode_train}_dataset_faz.pkl" # f"../{data_type}_{mode_train}_dataset_faz.pkl"
                                                            remove_isolated_nodes=False
                                                            )

    final_test_pickle = f"../data/{data_type}_{mode_final_test}_dataset_faz_isol_not_removed_more_features.pkl"   # _isol_not_removed     _more_features
    final_test_dataset = hetero_graph_loader_faz.HeteroGraphLoaderTorch(graph_path_1=vessel_graph_path,
                                                            graph_path_2=void_graph_path,
                                                            graph_path_3=faz_node_path,
                                                            hetero_edges_path_12=hetero_edges_path,
                                                            hetero_edges_path_13=faz_vessel_edge_path,
                                                            hetero_edges_path_23=faz_region_edge_path,
                                                            mode = mode_final_test,
                                                            label_file = label_file, 
                                                            line_graph_1 =True, 
                                                            class_dict = octa_dr_dict,
                                                            pickle_file = final_test_pickle, #f"../{data_type}_{mode_train}_dataset_faz.pkl" # f"../{data_type}_{mode_train}_dataset_faz.pkl"
                                                            remove_isolated_nodes=False
                                                            )

print(f"CV dataset length: {len(cv_dataset)}")
print(f"Final test dataset length: {len(final_test_dataset)}")


image_path = f"../data/{data_type}_images"
json_path = f"../data/{data_type}_json"
seg_folder = f"../data/{data_type}_seg"
seg_size = 1216

#cv_dataset_cl = prep.add_centerline_statistics_multi(cv_dataset, image_path, json_path, seg_size)
#final_test_dataset_cl = prep.add_centerline_statistics_multi(final_test_dataset, image_path, json_path, seg_size)
cv_dataset_cl = prep.add_vessel_region_statistics_multi(cv_dataset, image_path, json_path, seg_folder)
final_test_dataset_cl = prep.add_vessel_region_statistics_multi(final_test_dataset, image_path, json_path, seg_folder)

# clean the data
cv_dataset.hetero_graph_list = prep.hetero_graph_cleanup_multi(cv_dataset_cl)
final_test_dataset.hetero_graph_list = prep.hetero_graph_cleanup_multi(final_test_dataset_cl)

cv_dataset_dict = dict(zip([graph.graph_id for graph in cv_dataset.hetero_graph_list], cv_dataset.hetero_graph_list))
final_test_dataset_dict = dict(zip([graph.graph_id for graph in final_test_dataset.hetero_graph_list], final_test_dataset.hetero_graph_list))

# set the dictionaries
cv_dataset.hetero_graphs = cv_dataset_dict
final_test_dataset.hetero_graphs = final_test_dataset_dict

cv_pickle_processed = f"../data/{data_type}_{mode_cv}_region_vessels_no_del_smaller0.pkl"
final_test_pickle_processed = f"../data/{data_type}_{mode_final_test}_region_vessels_no_del_smaller0.pkl" 



# save the pickled datasets
with open(cv_pickle_processed, "wb") as file:
    pickle.dump(cv_dataset, file)

with open(final_test_pickle_processed, "wb") as file:
    pickle.dump(final_test_dataset, file)
