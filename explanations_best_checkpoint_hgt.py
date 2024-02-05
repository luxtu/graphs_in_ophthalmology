import os
from loader import hetero_graph_loader, hetero_graph_loader_faz
from utils import prep
import json
import copy
import torch
from models import graph_classifier, hgt_gnn
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn import GATConv, SAGEConv, GraphConv, GCNConv, summary
from torch_geometric.loader import DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, balanced_accuracy_score, cohen_kappa_score, roc_auc_score

from torch_geometric.explain import Explainer,  CaptumExplainer, AttentionExplainer
from explainability import grad_cam, torch_geom_explanation, explanation_in_raw_data, baseline_lookup
from types import SimpleNamespace
from graph_plotting import graph_2D
from matplotlib import pyplot as plt


check_point_folder = label_file = "../data/best_checkpoints_hgt"

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

    cv_pickle = f"../data/{data_type}_{mode_cv}_dataset_faz_isol_not_removed.pkl"
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
                                                            remove_isolated_nodes= False
                                                            )

    final_test_pickle = f"../data/{data_type}_{mode_final_test}_dataset_faz_isol_not_removed.pkl"
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
                                                            remove_isolated_nodes= False
                                                            )


agg_mode_dict = {"mean": global_mean_pool, "max": global_max_pool, "add": global_add_pool, "add_max": [global_add_pool, global_max_pool], "max_mean": [global_max_pool, global_mean_pool], "add_mean": [global_add_pool, global_mean_pool]}
homogeneous_conv_dict = {"gat": GATConv, "sage":SAGEConv, "graph" : GraphConv, "gcn" : GCNConv}
heterogeneous_conv_dict = {"gat": GATConv, "sage":SAGEConv, "graph" : GraphConv}
activation_dict = {"relu":torch.nn.functional.relu, "leaky" : torch.nn.functional.leaky_relu, "elu":torch.nn.functional.elu}


df_report = pd.DataFrame()
df_metrics = pd.DataFrame()


# select the checkpoint file with run id 1dfnjuto
#1dfnjuto
run_id = "0llh849s"
chekpoint_file = [f for f in check_point_files if run_id in f][0]
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



split = sweep_configuration["split"]
print(split)

image_path = f"../data/{data_type}_images"
json_path = f"../data/{data_type}_json"
seg_size = 1216
#
cv_dataset_cl = prep.add_centerline_statistics_multi(cv_dataset, image_path, json_path, seg_size)
final_test_dataset_cl = prep.add_centerline_statistics_multi(final_test_dataset, image_path, json_path, seg_size)
#
## clean the data
#cv_dataset.hetero_graph_list = prep.hetero_graph_cleanup_multi(cv_dataset_cl)
#final_test_dataset.hetero_graph_list = prep.hetero_graph_cleanup_multi(final_test_dataset_cl)
#
cv_dataset.hetero_graph_list = cv_dataset_cl
final_test_dataset.hetero_graph_list = final_test_dataset_cl

cv_dataset_dict = dict(zip([graph.graph_id for graph in cv_dataset.hetero_graph_list], cv_dataset.hetero_graph_list))
final_test_dataset_dict = dict(zip([graph.graph_id for graph in final_test_dataset.hetero_graph_list], final_test_dataset.hetero_graph_list))
#
## set the dictionaries
cv_dataset.hetero_graphs = cv_dataset_dict
final_test_dataset.hetero_graphs = final_test_dataset_dict


train_dataset, val_dataset, test_dataset = prep.adjust_data_for_split(cv_dataset, final_test_dataset, split, faz = True)

out_channels = 3

model = hgt_gnn.HGT_GNN(hidden_channels= sweep_configuration["parameters"]["hidden_channels"],
                                                    out_channels= out_channels,
                                                    num_layers= sweep_configuration["parameters"]["num_layers"],
                                                    dropout = 0, 
                                                    aggregation_mode= agg_mode_dict[sweep_configuration["parameters"]["aggregation_mode"]], 
                                                    node_types = node_types,
                                                    num_pre_processing_layers = sweep_configuration["parameters"]["pre_layers"],
                                                    num_post_processing_layers = sweep_configuration["parameters"]["post_layers"],
                                                    batch_norm = sweep_configuration["parameters"]["batch_norm"],
                                                    meta_data= train_dataset[0].metadata(),
                                                    num_heads= sweep_configuration["parameters"]["num_heads"],
                                                    activation= activation_dict[sweep_configuration["parameters"]["activation"]],
                                                    faz_node = sweep_configuration["parameters"]["faz_node"],
                                                    start_rep = sweep_configuration["parameters"]["start_rep"],
                                                    aggr_faz = sweep_configuration["parameters"]["aggr_faz"],
                                                    )




with open("feature_name_dict_new.json", "r") as file:
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
        for data in train_dataset:
            data[key].x = torch.cat([data[key].x[:, :idx], data[key].x[:, idx+1:]], dim = 1)
        for data in val_dataset:
            data[key].x = torch.cat([data[key].x[:, :idx], data[key].x[:, idx+1:]], dim = 1)
    

features_label_dict["graph_1"] = features_label_dict["graph_1"] + ["cl_mean", "cl_std", "cl_025", "cl_075"]

print(features_label_dict["graph_1"])

#print(label_dict_full)
#print(features_label_dict)

irrelevant_loss = torch.nn.CrossEntropyLoss()
clf = graph_classifier.graphClassifierHetero(model, irrelevant_loss) 
test_loader_set = DataLoader(test_dataset[:4], batch_size = 4, shuffle=False)
test_loader_full = DataLoader(test_dataset, batch_size = 64, shuffle=False)


# must run on an input to set the parameters
# load the state dict
clf.predict(test_loader_set)
clf.model.load_state_dict(state_dict)
y_prob_test, y_test = clf.predict(test_loader_full)
#print(y_prob_test.argmax(axis = 1))
#print(y_test)

# create the lookup tree for the integrated gradients baseline


edges = False
if edges:
    edge_mask_type = "object"
else:
    edge_mask_type = None
explain_type = "IntegratedGradients"  #IntegratedGradients" #Saliency"

gnnexp_loader = DataLoader(test_dataset, batch_size = 1, shuffle=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


lookup_tree = baseline_lookup.Baseline_Lookup([train_dataset], node_types, device = device)

print(test_dataset[0].metadata())
print(test_dataset[0])

for idx, data in enumerate(gnnexp_loader):
    data = data.to(device)
    data_label = data.graph_id[0]
    print(f"Generating explanations for {data_label}")
    #print(summary(clf.model,data.x_dict, data.edge_index_dict, batch_dict = data.batch_dict))


    graph_1_shape = data.x_dict["graph_1"].shape
    graph_2_shape = data.x_dict["graph_2"].shape
    faz_shape = data.x_dict["faz"].shape

    print(f"Graph 1 shape: {graph_1_shape}")
    print(f"Graph 2 shape: {graph_2_shape}")
    print(f"FAZ shape: {faz_shape}")

    baseline_type = "lookup"
    if baseline_type == "zero":
        baseline_graph_1 = torch.zeros_like(data.x_dict["graph_1"]).unsqueeze(0)
        baseline_graph_2 = torch.zeros_like(data.x_dict["graph_2"]).unsqueeze(0)
        baseline_faz = torch.zeros_like(data.x_dict["faz"], device= device).unsqueeze(0)

    elif baseline_type == "lookup": #get_k_nearst_neighbors_avg_features
        baseline_graph_1 = lookup_tree.get_k_nearst_neighbors_avg_features_quantile_correct("graph_1", data["graph_1"].pos.cpu().numpy(), 100).unsqueeze(0)
        baseline_graph_2 = lookup_tree.get_k_nearst_neighbors_avg_features_quantile_correct("graph_2", data["graph_2"].pos.cpu().numpy(), 100).unsqueeze(0)
        baseline_faz = torch.zeros_like(data.x_dict["faz"], device= device).unsqueeze(0)

    #print(f"Baseline Graph 1 shape: {baseline_graph_1.shape}")
    #print(f"Baseline Graph 2 shape: {baseline_graph_2.shape}")
    #print(f"Baseline FAZ shape: {baseline_faz.shape}")

    if baseline_type is not None:
        baselines = (baseline_graph_1, baseline_graph_2, baseline_faz)

    #del data[("graph_1", "to", "graph_1")] #["edge_index_dict"]
    #del data[("graph_2", "to", "graph_2")] #["edge_index_dict"]
    #del data[("graph_1", "to", "graph_2")] #["edge_index_dict"]
    #del data[("graph_2", "rev_to", "graph_1")] #["edge_index_dict"]
    #del data[("graph_1", "to", "faz")] #["edge_index_dict"]
    #del data[("faz", "to", "graph_2")] #["edge_index_dict"]
    #del data[("faz", "rev_to", "graph_1")] #["edge_index_dict"]
    #del data[("graph_2", "rev_to", "faz")] #["edge_index_dict"]
    #del data[("faz", "to", "faz")] #["edge_index_dict"]

    #data.edge_index_dict[("faz", "to", "graph_2")] = torch.tensor([[0],[1]], dtype = torch.long, device = device).unsqueeze(0)
#
    #print(data)


    # add baseline for the edges
    if edge_mask_type is not None and baseline_type is not None:
        # iterate throuh all edge types and add zero baseline
        for edge_type in data.edge_index_dict.keys():
            #print(edge_type)
            edge_index = data.edge_index_dict[edge_type]
            #edge_attr = data.edge_attr_dict[edge_type]
            #print(edge_index.shape)
            #print(edge_attr.shape)
            baseline_edge_attr = torch.zeros(edge_index.shape[1], device= device).unsqueeze(0)
            baselines += (baseline_edge_attr,)

    if baseline_type is not None:
        algorithm = CaptumExplainer(explain_type, baselines = baselines)
    else:
        algorithm = CaptumExplainer(explain_type)
    explainer = Explainer(
                        model= clf.model,
                        algorithm= algorithm, #CaptumExplainer(explain_type),#AttentionExplainer #hetero_gnn_explainer.HeteroGNNExplainer(epochs = 200)
                        explanation_type='phenomenon',
                        node_mask_type='attributes', # common attributes works well
                        edge_mask_type=edge_mask_type,
                        model_config=dict(
                            mode='multiclass_classification',
                            task_level='graph',
                            return_type='raw',
                        )
                    )

    # turn of grads for the batch dict
    for key in data.batch_dict.keys():
        data.batch_dict[key].requires_grad = False

    prediction = explainer.get_prediction(data.x_dict, data.edge_index_dict, batch_dict = data.batch_dict)
    target = explainer.get_target(prediction)
    print(f"Target: {target}, (Predicted Class)")
    print(f"True Label: {data.y[0]}")


    explanation = explainer(data.x_dict, data.edge_index_dict, target =target,  batch_dict = data.batch_dict,  grads = False)

    explain_out_path = "../data/explain_out_hgt"

    #print(explanation)
    # for GNN Explainer forces sparsity, thres
    threshold = "adaptive"
    importance_threshold_path = f'{explain_out_path}/feature_importance/feature_importance_{data_label}_{explain_type}_{baseline_type}_{run_id}_{threshold}_quantile_attributes_new.png'
    torch_geom_explanation.visualize_feature_importance(explanation,data,  importance_threshold_path, features_label_dict, threshold= threshold)
    #without threshold
    importance_path = f'{explain_out_path}/feature_importance/feature_importance_{data_label}_{explain_type}_{baseline_type}_{run_id}_no_threshold_quantile_attributes_new.png'
    torch_geom_explanation.visualize_feature_importance(explanation,data, importance_path, features_label_dict)
    torch_geom_explanation.visualize_relevant_subgraph(explanation, data, f"{explain_out_path}/subgraph/subgraph_{data_label}_{explain_type}_{run_id}_attributes.png", threshold = "adaptive", edge_threshold = "node_threshold", edges = edges, faz_node= faz_node_bool) #"adaptive"
    #graph_2D.HeteroGraphPlotter2D().plot_graph_2D_faz(data, edges= True, path = f"{explain_out_path}/fullgraph/fullgraph_{data_label}_{explain_type}.png")
    torch_geom_explanation.visualize_node_importance_histogram(explanation, f"{explain_out_path}/histogram/node_hist_{data_label}_{explain_type}_{baseline_type}_{run_id}_quantile_attributes_new.png", faz_node= faz_node_bool)

    #torch_geom_explanation.store_relevant_nodes_csv(explanation, data,  f"{explain_out_path}/csv_out/", features_label_dict , faz_node=faz_node_bool)

    # generate overlay image
    segmentation_path = f"../data/{data_type}_seg"
    json_path = f"../data/{data_type}_json"
    image_path = f"../data/{data_type}_images"


    raw_data_explainer = explanation_in_raw_data.RawDataExplainer(raw_image_path= image_path, segmentation_path= segmentation_path, vvg_path= json_path)
    overlay_path = f"{explain_out_path}/overlay/overlay_subgraph_{data_label}_{baseline_type}_{run_id}_quantile_new.png"
    raw_data_explainer.create_explanation_image(explanation, data, data_label, path=overlay_path, faz_node= faz_node_bool, threshold = "adaptive", edge_threshold = "adaptive", label_names=label_names, target = target)


    heatmap_path = f"{explain_out_path}/heatmap/integrate_gradients_{data_label}_{baseline_type}_{run_id}_quantile_new.png"
    torch_geom_explanation.integrate_gradients_heatmap(explanation, data, heatmap_path, faz_node= faz_node_bool)

