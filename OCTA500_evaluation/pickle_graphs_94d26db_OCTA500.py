import os
import pickle

from loader import hetero_graph_loader_faz
from utils import prep

faz_node_bool = True

label_file_octa500 = "../../OCTA_500_RELEVANT_fix/labels_only_DR_Healthy.csv"

octa_500_dict = {"NORMAL": 0, "DR": 1}

mode_cv = "cv"
mode_final_test = "final_test"
mode = "all"
label_names = ["Healthy/DM", "NPDR", "PDR"]


vessel_graph_path_octa500 = "../../OCTA_500_RELEVANT_fix/graph_new"
void_graph_path_octa500 = (
    "../../OCTA_500_RELEVANT_fix/void_graph_faz_more_features"  # _more_features
)
hetero_edges_path_octa500 = "../../OCTA_500_RELEVANT_fix/hetero_edges_faz"
faz_node_path_octa500 = "../../OCTA_500_RELEVANT_fix/faz_node"
faz_region_edge_path_octa500 = "../../OCTA_500_RELEVANT_fix/faz_region_edges"
faz_vessel_edge_path_octa500 = "../../OCTA_500_RELEVANT_fix/faz_vessel_edges"

# check that the folders are not empty
assert len(os.listdir(void_graph_path_octa500)) > 0
assert len(os.listdir(hetero_edges_path_octa500)) > 0
assert len(os.listdir(faz_node_path_octa500)) > 0
assert len(os.listdir(faz_region_edge_path_octa500)) > 0
assert len(os.listdir(faz_vessel_edge_path_octa500)) > 0


octa500_pickle = "../../OCTA500_cv_more_features.pkl"
octa500_dataset = hetero_graph_loader_faz.HeteroGraphLoaderTorch(
    graph_path_1=vessel_graph_path_octa500,
    graph_path_2=void_graph_path_octa500,
    graph_path_3=faz_node_path_octa500,
    hetero_edges_path_12=hetero_edges_path_octa500,
    hetero_edges_path_13=faz_vessel_edge_path_octa500,
    hetero_edges_path_23=faz_region_edge_path_octa500,
    mode=mode,
    label_file=label_file_octa500,
    line_graph_1=True,
    class_dict=octa_500_dict,
    pickle_file=octa500_pickle,  # f"../{data_type}_{mode_train}_dataset_faz.pkl" # f"../{data_type}_{mode_train}_dataset_faz.pkl"
    remove_isolated_nodes=False,
)

print(f"OCTA500 length: {len(octa500_dataset)}")


print(octa500_dataset[0])
# exit()

image_path = "../../OCTA_500_RELEVANT_fix/images"
json_path = "../../OCTA_500_RELEVANT_fix/json"
seg_folder = "../../OCTA_500_RELEVANT_fix/segs"
seg_size = 1216

octa500_dataset_cl = prep.add_centerline_statistics_multi(
    octa500_dataset, image_path, json_path, seg_size
)

# clean the data
octa500_dataset.hetero_graph_list = prep.hetero_graph_cleanup_multi(octa500_dataset_cl)

octa500_dataset_dict = dict(
    zip(
        [graph.graph_id for graph in octa500_dataset.hetero_graph_list],
        octa500_dataset.hetero_graph_list,
    )
)

# set the dictionaries
octa500_dataset.hetero_graphs = octa500_dataset_dict

octa500_processed = (
    "../../OCTA_500_RELEVANT_fix/all_OCTA500_selected_sweep_repeat_v2.pkl"
)


# save the pickled datasets
with open(octa500_processed, "wb") as file:
    pickle.dump(octa500_dataset, file)
