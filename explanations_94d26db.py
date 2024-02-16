import os
from loader import hetero_graph_loader, hetero_graph_loader_faz
from utils import dataprep_utils
import pickle
import json
import copy
import torch
from models import graph_classifier, heterogeneous_gnn, global_node_gnn
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn import GATConv, SAGEConv, GraphConv, GCNConv, summary
from torch_geometric.loader import DataLoader
import pandas as pd

from torch_geometric.explain import Explainer,  CaptumExplainer
from explainability import torch_geom_explanation, explanation_in_raw_data, baseline_lookup



check_point_folder = "../data/test_checkpoints" # test_checkpoints

# extract the files with .pt extension
check_point_files = [f for f in os.listdir(check_point_folder) if f.endswith(".pt")]

# extract the run id from the file name
# run id are the characters after the last underscore
run_ids = [f.split("_")[-1].split(".")[0] for f in check_point_files]

# find the json file with the same run id
# json file name is the same as the run id
json_files = [f for f in os.listdir(check_point_folder) if f.endswith(".json")]


data_type = "DCP"
label_names = ["Healthy/DM", "NPDR", "PDR"]
mode_cv = "cv"
mode_final_test = "final_test"
faz_node_bool = True


agg_mode_dict = {"mean": global_mean_pool, "max": global_max_pool, "add": global_add_pool, "add_max": [global_add_pool, global_max_pool], "max_mean": [global_max_pool, global_mean_pool], "add_mean": [global_add_pool, global_mean_pool]}
homogeneous_conv_dict = {"gat": GATConv, "sage":SAGEConv, "graph" : GraphConv, "gcn" : GCNConv}
heterogeneous_conv_dict = {"gat": GATConv, "sage":SAGEConv, "graph" : GraphConv}
activation_dict = {"relu":torch.nn.functional.relu, "leaky" : torch.nn.functional.leaky_relu, "elu":torch.nn.functional.elu}


df_report = pd.DataFrame()
df_metrics = pd.DataFrame()


# select the checkpoint file with run id 5d3llji3 # nwjxejqg
#p94pkq2c
run_id = "14d0c0l2" # ""kxmjgi6k" # "5ughh17h" # "14d0c0l2" #"ig9zvma3" # "14d0c0l2"
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
out_channels = 3
model = global_node_gnn.GNN_global_node(hidden_channels= sweep_configuration["parameters"]["hidden_channels"], # global_node_gnn.GNN_global_node
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
                                                    homogeneous_conv= homogeneous_conv_dict[sweep_configuration["parameters"]["homogeneous_conv"]],
                                                    heterogeneous_conv= heterogeneous_conv_dict[sweep_configuration["parameters"]["heterogeneous_conv"]],
                                                    activation= activation_dict[sweep_configuration["parameters"]["activation"]],
                                                    faz_node = sweep_configuration["parameters"]["faz_node"],
                                                    start_rep = sweep_configuration["parameters"]["start_rep"],
                                                    aggr_faz = sweep_configuration["parameters"]["aggr_faz"],
                                                    )
split = sweep_configuration["split"]
print(split)

# print the state dict
#print(state_dict.keys())
## delte all key and values that contain "faz"
if not sweep_configuration["parameters"]["faz_node"] or (not sweep_configuration["parameters"]["aggr_faz"] and not sweep_configuration["parameters"]["hetero_conns"] and not sweep_configuration["parameters"]["start_rep"]):
    state_dict = {k: v for k, v in state_dict.items() if "faz" not in k}

#state_dict = {k: v for k, v in state_dict.items() if "faz" not in k}
#print(state_dict.keys())


# pickle the datasets
cv_pickle_processed = f"../data/{data_type}_{mode_cv}_selected_sweep_repeat_v2.pkl"
final_test_pickle_processed = f"../data/{data_type}_{mode_final_test}_selected_sweep_repeat_v2.pkl" 
split = 3

import pickle
#load the pickled datasets
with open(cv_pickle_processed, "rb") as file:
    cv_dataset = pickle.load(file)

with open(final_test_pickle_processed, "rb") as file:
    final_test_dataset = pickle.load(file)


train_dataset, val_dataset, test_dataset = dataprep_utils.adjust_data_for_split(cv_dataset, final_test_dataset, split, faz = True)



with open("training_configs/feature_name_dict.json", "r") as file:
    label_dict_full = json.load(file)
    #features_label_dict = json.load(file)
features_label_dict = copy.deepcopy(label_dict_full)


eliminate_features = {"graph_1":["num_voxels","hasNodeAtSampleBorder", "maxRadiusAvg", "maxRadiusStd"], #"graph_1":["num_voxels", "maxRadiusAvg", "hasNodeAtSampleBorder", "maxRadiusStd"], 
                      "graph_2":["centroid_weighted-0", "centroid_weighted-1", "feret_diameter_max", "orientation"]}


if faz_node_bool:
    eliminate_features["faz"] = ["centroid_weighted-0", "centroid_weighted-1", "feret_diameter_max", "orientation"]


# get positions of features to eliminate and remove them from the feature label dict and the graphs
for key in eliminate_features.keys():
    for feat in eliminate_features[key]:
        idx = features_label_dict[key].index(feat)
        features_label_dict[key].remove(feat)
        for data in train_dataset:
            data[key].x = torch.cat([data[key].x[:, :idx], data[key].x[:, idx+1:]], dim = 1)
        for data in val_dataset:
            data[key].x = torch.cat([data[key].x[:, :idx], data[key].x[:, idx+1:]], dim = 1)
        for data in test_dataset:
            data[key].x = torch.cat([data[key].x[:, :idx], data[key].x[:, idx+1:]], dim = 1)

    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

irrelevant_loss = torch.nn.CrossEntropyLoss()
clf = graph_classifier.graphClassifierHetero(model, torch.tensor([0.,0.,0.], device= device)) 
test_loader_set = DataLoader(test_dataset[:4], batch_size = 4, shuffle=False)
test_loader_full = DataLoader(test_dataset, batch_size = 64, shuffle=False)
# must run on an input to set the parameters

# print the state dict of the model
print(clf.model)
print(state_dict.keys())


print(train_dataset[0])

# load the state dict
clf.predict(test_loader_set)
clf.model.load_state_dict(state_dict)
y_prob_test, y_test = clf.predict(test_loader_full)
# create the lookup tree for the integrated gradients baseline


edges = False
if edges:
    edge_mask_type = "object"
else:
    edge_mask_type = None
explain_type = "IntegratedGradients"  #IntegratedGradients" #Saliency"

gnnexp_loader = DataLoader(val_dataset, batch_size = 1, shuffle=False)
lookup_tree = baseline_lookup.Baseline_Lookup([train_dataset], node_types, device = device)


save_path = "../data/explain_94d26db"

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

    baseline_type = "lookup" #"zero" # "lookup" #None
    if baseline_type == "zero":
        baseline_graph_1 = torch.zeros_like(data.x_dict["graph_1"]).unsqueeze(0)
        baseline_graph_2 = torch.zeros_like(data.x_dict["graph_2"]).unsqueeze(0)
        baseline_faz = torch.zeros_like(data.x_dict["faz"], device= device).unsqueeze(0)

    elif baseline_type == "lookup": #get_k_nearst_neighbors_avg_features
        baseline_graph_1 = lookup_tree.get_k_nearst_neighbors_avg_features_quantile_correct("graph_1", data["graph_1"].pos.cpu().numpy(), 100).unsqueeze(0)
        baseline_graph_2 = lookup_tree.get_k_nearst_neighbors_avg_features_quantile_correct("graph_2", data["graph_2"].pos.cpu().numpy(), 100).unsqueeze(0)
        baseline_faz = torch.zeros_like(data.x_dict["faz"], device= device).unsqueeze(0)

    # if aggr.faz, hetero_conns and start_rep are all false, then the faz node is not used
    # delete the faz from the data in this case
    if not sweep_configuration["parameters"]["aggr_faz"] and not sweep_configuration["parameters"]["hetero_conns"] and not sweep_configuration["parameters"]["start_rep"]:
        print("Deleting FAZ")
        del data["faz"]
        del data[("graph_1", "to", "faz")]
        del data[("faz", "to", "graph_2")]
        del data[("faz", "rev_to", "graph_1")]
        del data[("graph_2", "rev_to", "faz")]
        del data[("faz", "to", "faz")]

    # add baseline for the edges
    if edge_mask_type is not None:
        # iterate throuh all edge types and add zero baseline
        for edge_type in data.edge_index_dict.keys():
            #print(edge_type)
            edge_index = data.edge_index_dict[edge_type]
            #edge_attr = data.edge_attr_dict[edge_type]
            #print(edge_index.shape)
            #print(edge_attr.shape)
            baseline_edge_attr = torch.zeros(edge_index.shape[1], device= device).unsqueeze(0)
            #baselines += (baseline_edge_attr,)


    explainer = Explainer(
                        model= clf.model,
                        algorithm= CaptumExplainer(explain_type), # baselines = baselines #CaptumExplainer(explain_type),#AttentionExplainer #hetero_gnn_explainer.HeteroGNNExplainer(epochs = 200)
                        explanation_type='phenomenon',
                        node_mask_type='attributes', # common attributes works well
                        edge_mask_type=edge_mask_type,
                        model_config=dict(
                            mode='multiclass_classification',
                            task_level='graph',
                            return_type='raw',
                        )
                    )

    prediction = explainer.get_prediction(data.x_dict, data.edge_index_dict, batch_dict = data.batch_dict)
    target = explainer.get_target(prediction)
    print(f"Target: {target}, (Predicted Class)")
    print(f"True Label: {data.y[0]}")

    explanation = explainer(data.x_dict, data.edge_index_dict, target =target,  batch_dict = data.batch_dict,  grads = False)

    #print(explanation)
    # for GNN Explainer forces sparsity, thres
    threshold = "adaptive"
    importance_threshold_path = f'{save_path}/feature_importance/feature_importance_{data_label}_{explain_type}_{baseline_type}_{run_id}_{threshold}_quantile_attributes_new.png'
    #torch_geom_explanation.visualize_feature_importance(explanation,data,  importance_threshold_path, features_label_dict, explained_gradient= 0.8) # means all of the gradient is explained
    #without threshold
    importance_path = f'{save_path}/feature_importance/feature_importance_{data_label}_{explain_type}_{baseline_type}_{run_id}_no_threshold_quantile_attributes_new.png'
    #torch_geom_explanation.visualize_feature_importance(explanation,data, importance_path, features_label_dict, explained_gradient= None)
    #torch_geom_explanation.visualize_relevant_subgraph(explanation, data, f"{save_path}/subgraph_{data_label}_{explain_type}_{run_id}_attributes.png", threshold = "adaptive", edge_threshold = "node_threshold", edges = edges, faz_node= faz_node_bool) #"adaptive"
    #graph_2D.HeteroGraphPlotter2D().plot_graph_2D_faz(data, edges= True, path = f"{save_path}/fullgraph/fullgraph_{data_label}_{explain_type}.png")
    #torch_geom_explanation.visualize_node_importance_histogram(explanation, f"{save_path}/histogram/node_hist_{data_label}_{explain_type}_{baseline_type}_{run_id}_quantile_attributes_new.png", faz_node= faz_node_bool)

    #torch_geom_explanation.store_relevant_nodes_csv(explanation, data,  f"{save_path}/csv_out/", features_label_dict , faz_node=faz_node_bool)

    # generate overlay image
    segmentation_path = f"../data/{data_type}_seg"
    json_path = f"../data/{data_type}_json"
    image_path = f"../data/{data_type}_images"


    raw_data_explainer = explanation_in_raw_data.RawDataExplainer(raw_image_path= image_path, segmentation_path= segmentation_path, vvg_path= json_path)
    overlay_path = f"{save_path}/overlay/overlay_subgraph_{data_label}_{baseline_type}_{run_id}_quantile_new.png"
    raw_data_explainer.create_explanation_image(explanation, data, data_label, path=overlay_path, label_names=label_names, target = target, heatmap= True, explained_gradient= 0.8)

    #heatmap_path = f"{save_path}/heatmap/integrate_gradients_{data_label}_{baseline_type}_{run_id}_quantile_new.png"
    #torch_geom_explanation.integrate_gradients_heatmap(explanation, data, heatmap_path, faz_node= faz_node_bool)

