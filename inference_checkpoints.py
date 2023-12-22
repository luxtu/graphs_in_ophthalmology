import os
from loader import hetero_graph_loader, hetero_graph_loader_faz
from utils import prep
import json
import copy
import torch
from models import graph_classifier, global_node_gnn
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn import GATConv, SAGEConv, GraphConv, GCNConv
from torch_geometric.loader import DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, balanced_accuracy_score, cohen_kappa_score


check_point_folder = label_file = "../data/best_checkpoints"

# extract the files with .pt extension
check_point_files = [f for f in os.listdir(check_point_folder) if f.endswith(".pt")]

# extract the run id from the file name
# run id are the characters after the last underscore
run_ids = [f.split("_")[-1].split(".")[0] for f in check_point_files]

# find the json file with the same run id
# json file name is the same as the run id
json_files = [f for f in os.listdir(check_point_folder) if f.endswith(".json")]


data_type = "DCP"
mode_final_test = "final_test"
# load the data

octa_dr_dict = {"Healthy": 0, "DM": 0, "PDR": 2, "Early NPDR": 1, "Late NPDR": 1}
label_names = ["Healthy/DM", "NPDR", "PDR"]

vessel_graph_path = f"../data/{data_type}_vessel_graph"
label_file = "../data/splits"

mode_cv = "cv"
mode_final_test = "final_test"

faz_node_bool = True

if not faz_node_bool:
    void_graph_path = f"..data/{data_type}_void_graph"
    hetero_edges_path = f"..data/{data_type}_heter_edges"

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
    void_graph_path = f"..data/{data_type}_void_graph_faz_node"
    hetero_edges_path = f"..data/{data_type}_heter_edges_faz_node"
    faz_node_path = f"../data/{data_type}_faz_nodes"
    faz_region_edge_path = f"../data/{data_type}_faz_region_edges"
    faz_vessel_edge_path = f"../data/{data_type}_faz_vessel_edges"

    cv_pickle = f"../data/{data_type}_{mode_cv}_dataset_faz.pkl"
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
                                                            pickle_file = cv_pickle #f"../{data_type}_{mode_train}_dataset_faz.pkl" # f"../{data_type}_{mode_train}_dataset_faz.pkl"
                                                            )

    final_test_pickle = f"../data/{data_type}_{mode_final_test}_dataset_faz.pkl"
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
                                                            pickle_file = final_test_pickle #f"../{data_type}_{mode_train}_dataset_faz.pkl" # f"../{data_type}_{mode_train}_dataset_faz.pkl"
                                                            )


agg_mode_dict = {"mean": global_mean_pool, "max": global_max_pool, "add": global_add_pool, "add_max": [global_add_pool, global_max_pool], "max_mean": [global_max_pool, global_mean_pool], "add_mean": [global_add_pool, global_mean_pool]}
homogeneous_conv_dict = {"gat": GATConv, "sage":SAGEConv, "graph" : GraphConv, "gcn" : GCNConv}
heterogeneous_conv_dict = {"gat": GATConv, "sage":SAGEConv, "graph" : GraphConv}
activation_dict = {"relu":torch.nn.functional.relu, "leaky" : torch.nn.functional.leaky_relu, "elu":torch.nn.functional.elu}


df_report = pd.DataFrame()
df_metrics = pd.DataFrame()


for chekpoint_file, run_id in zip(check_point_files, run_ids):
    print(chekpoint_file, run_id)
    for json_file in json_files:
        if run_id in json_file:
            # load the json file
            print(json_file)
            sweep_configuration = json.load(open(os.path.join(check_point_folder, json_file), "r"))
            break

    # load the model
    state_dict = torch.load(os.path.join(check_point_folder, chekpoint_file))
    node_types = ["graph_1", "graph_2"]
    # load the model
    out_channels = 3
    model = global_node_gnn.GNN_global_node(hidden_channels= sweep_configuration["parameters"]["hidden_channels"],
                                                        out_channels= out_channels,
                                                        num_layers= sweep_configuration["parameters"]["num_layers"],
                                                        dropout = 0, 
                                                        aggregation_mode= agg_mode_dict[sweep_configuration["parameters"]["aggregation_mode"]], 
                                                        node_types = node_types,
                                                        num_pre_processing_layers = sweep_configuration["parameters"]["pre_layers"],
                                                        num_post_processing_layers = sweep_configuration["parameters"]["post_layers"],
                                                        batch_norm = sweep_configuration["parameters"]["batch_norm"],
                                                        conv_aggr = sweep_configuration["parameters"]["conv_aggr"],
                                                        hetero_conns = sweep_configuration["parameters"]["hetero_conns"],
                                                        #meta_data = train_dataset[0].metadata())
                                                        homogeneous_conv= homogeneous_conv_dict[sweep_configuration["parameters"]["homogeneous_conv"]],
                                                        heterogeneous_conv= heterogeneous_conv_dict[sweep_configuration["parameters"]["heterogeneous_conv"]],
                                                        activation= activation_dict[sweep_configuration["parameters"]["activation"]],
                                                        faz_node = sweep_configuration["parameters"]["faz_node"],
                                                        start_rep = sweep_configuration["parameters"]["start_rep"],
                                                        aggr_faz = sweep_configuration["parameters"]["aggr_faz"],
                                                        )



    split = sweep_configuration["split"]
    print(split)
    _, _, test_dataset = prep.adjust_data_for_split(cv_dataset, final_test_dataset, split, faz = True)

    with open("label_dict.json", "r") as file:
        label_dict_full = json.load(file)
        #features_label_dict = json.load(file)
    features_label_dict = copy.deepcopy(label_dict_full)

    eliminate_features = {"graph_1":["num_voxels", "maxRadiusAvg", "hasNodeAtSampleBorder", "maxRadiusStd"], 
                          "graph_2":["centroid_weighted-0", "centroid_weighted-1", "feret_diameter_max", "orientation"]}

    if faz_node_bool:
        eliminate_features["faz"] = ["centroid_weighted-0", "centroid_weighted-1","feret_diameter_max", "orientation"]


    # get positions of features to eliminate and remove them from the feature label dict and the graphs
    for key in eliminate_features.keys():
        for feat in eliminate_features[key]:
            idx = features_label_dict[key].index(feat)
            features_label_dict[key].remove(feat)
            for data in test_dataset:
                data[key].x = torch.cat([data[key].x[:, :idx], data[key].x[:, idx+1:]], dim = 1)






    irrelevant_loss = torch.nn.CrossEntropyLoss()
    clf = graph_classifier.graphClassifierHetero(model, irrelevant_loss) 

    test_loader_set = DataLoader(test_dataset[:4], batch_size = 4, shuffle=False)
    test_loader_full = DataLoader(test_dataset, batch_size = 64, shuffle=False)

    # must run on an input to set the parameters
    # load the state dict
    clf.predict(test_loader_set)
    clf.model.load_state_dict(state_dict)

    y_prob_test, y_test = clf.predict(test_loader_full)

    y_pred = np.argmax(y_prob_test, axis = 1)

    report = classification_report(y_test, y_pred, target_names=label_names, output_dict=True)

    acc= accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred, weights="quadratic")

    metrics = {}

    metrics["accuracy"] = acc
    metrics["balanced_accuracy"] = bal_acc
    metrics["kappa"] = kappa

    df_metrics = pd.concat([df_metrics, pd.DataFrame(metrics, index = [split])])
    df_report = pd.concat([df_report, pd.DataFrame(report).transpose()])

df_metrics.to_csv('GNN_CV_metrics_3class.csv')
df_report.to_csv('GNN_CV_report_3class.csv')