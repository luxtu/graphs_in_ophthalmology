import os
import matplotlib.pyplot as plt

from utils import dataprep_utils, train_utils
import pickle
import json
import copy
import torch
from models import graph_classifier,  global_node_gnn
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn import GATConv, SAGEConv, GraphConv, GCNConv
from torch_geometric.loader import DataLoader


from torch_geometric.explain import Explainer,  CaptumExplainer
from explainability import baseline_lookup



check_point_folder = "../data/checkpoints_94d26db_final_27022024"  #"../data/checkpoints_94d26db_split2_retrain" #test_checkpoints" # test_checkpoints

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



# select the checkpoint file with run id 5d3llji3 # nwjxejqg
#p94pkq2c
run_id = "mtm0lor9"  #"z6af0tyz" #"mtm0lor9" # "90ym9iu4" #"pxeo8tv2" #"3zhbi3es" #"zng1r2l7" # "j69dzrn5" # "zng1r2l7" # "14d0c0l2" # ""kxmjgi6k" # "5ughh17h" # "14d0c0l2" #"ig9zvma3" # "14d0c0l2"
chekpoint_file = [f for f in check_point_files if run_id in f][0]
for json_file in json_files:
    if run_id in json_file:
        # load the json file
        print(json_file)
        sweep_configuration = json.load(open(os.path.join(check_point_folder, json_file), "r"))
        break
# load the model
state_dict = torch.load(os.path.join(check_point_folder, chekpoint_file))

# count the number of parameters, only count the initialized parameters
param_sum = 0
ct = 0
for key in state_dict.keys():
    try:
        param_sum += state_dict[key].numel()
    except ValueError:
        ct +=1
        
print(ct)
print(param_sum)


print(sweep_configuration)

node_types = ["graph_1", "graph_2"]
# load the model
out_channels = 3
model = global_node_gnn.GNN_global_node(hidden_channels= sweep_configuration["parameters"]["hidden_channels"], # global_node_gnn.GNN_global_node
                                                    out_channels= out_channels,
                                                    num_layers= sweep_configuration["parameters"]["num_layers"],
                                                    dropout = 0.1, 
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


## delte all key and values that contain "faz"
if not sweep_configuration["parameters"]["faz_node"] or (not sweep_configuration["parameters"]["aggr_faz"] and not sweep_configuration["parameters"]["hetero_conns"] and not sweep_configuration["parameters"]["start_rep"]):
    print("Deleting FAZ")
    state_dict = {k: v for k, v in state_dict.items() if "faz" not in k}

#state_dict = {k: v for k, v in state_dict.items() if "faz" not in k}
#print(state_dict.keys())


# pickle the datasets
cv_pickle_processed = f"../data/{data_type}_{mode_cv}_selected_sweep_repeat_v2.pkl"
final_test_pickle_processed = f"../data/{data_type}_{mode_final_test}_selected_sweep_repeat_v2.pkl" 


#load the pickled datasets
with open(cv_pickle_processed, "rb") as file:
    cv_dataset = pickle.load(file)

with open(final_test_pickle_processed, "rb") as file:
    final_test_dataset = pickle.load(file)


train_dataset, val_dataset, test_dataset = dataprep_utils.adjust_data_for_split(cv_dataset, final_test_dataset, split, faz = True)



with open("training_configs/feature_name_dict_new.json", "r") as file:
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

# load the features to keep
#with open("training_configs/included_features_rf_importance_corr_98.json", "r") as file:
#    included_features = json.load(file)

# eliminate the features
#dataprep_utils.eliminate_features(included_features, features_label_dict, [train_dataset, val_dataset, test_dataset])

print(train_dataset[0])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# class weight extraction
train_labels = [int(data.y[0]) for data in train_dataset]
class_weights = dataprep_utils.get_class_weights(train_labels, verbose=False)
class_weights = torch.tensor(class_weights, device=device)
class_weights_weak = class_weights ** 0.5 

num_classes = class_weights.shape[0]
print(f"num classes: {num_classes}")
node_types = ["graph_1", "graph_2"]

# add the new features to the feature label dict
features_label_dict["graph_1"] = features_label_dict["graph_1"][:-1] + ["cl_mean", "cl_std", "q25", "q75"] + ["degree"] #["int_mean", "int_max", "int_min", "std"]
features_label_dict["graph_2"] = features_label_dict["graph_2"][:-1] + ["q25", "q75", "std"] + ["degree"] 
features_label_dict["faz"] = features_label_dict["faz"] # + ["q25", "q75", "std"]


train_dataset.to(device)
val_dataset.to(device)
test_dataset.to(device)

# create data loaders for training and test set
train_loader = DataLoader(train_dataset, batch_size = 8, shuffle=True, drop_last = True)
val_loader = DataLoader(val_dataset, batch_size = 64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size = 1, shuffle=False)

balanced_loss = torch.nn.CrossEntropyLoss(class_weights)

clf = graph_classifier.graphClassifierHetero_94d26db(model, balanced_loss, lr = 0.01, weight_decay = 0.001, smooth_label_loss = True) 


# load the state dict
#clf.predict(test_loader_set)
loss, y_prob_train, y_true_train, _  = clf.train(train_loader)
res_dict, y_pred_softmax_val, y_true_val = train_utils.evaluate_model(clf, val_loader, test_loader)
#print(state_dict.keys())
#print(clf.model.state_dict().keys())
clf.model.load_state_dict(state_dict)


edges = False
if edges:
    edge_mask_type = "object"
else:
    edge_mask_type = None
explain_type = "IntegratedGradients"  #IntegratedGradients" #Saliency"

gnnexp_loader = DataLoader(test_dataset, batch_size = 1, shuffle=False)
lookup_tree = baseline_lookup.Baseline_Lookup([train_dataset], node_types, device = device)


save_path = "../data/explain_94d26db/" + run_id


most_important_features_npdr = {}
most_important_features_npdr["graph_1"] = None
most_important_features_npdr["graph_2"] = None
most_important_features_npdr["faz"] = None

most_important_features_pdr = {}
most_important_features_pdr["graph_1"] = None
most_important_features_pdr["graph_2"] = None
most_important_features_pdr["faz"] = None

for idx, data in enumerate(gnnexp_loader):
    data = data.to(device)
    data_label = data.graph_id[0]
    print(f"Generating explanations for {data_label}")
    #print(summary(clf.model,data.x_dict, data.edge_index_dict, batch_dict = data.batch_dict))
    if data.y[0] == 0:
        continue

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

    # delete all the edges to see if the prediction changes
    for edge_type in data.edge_index_dict.keys():
        # assign each edge type to an empty tensor with shape (2,0)
        data.edge_index_dict[edge_type] = torch.zeros((2,0), device= device)


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
    print(f"Target: {target.item()}, (Predicted Class)")
    print(f"True Label: {data.y[0]}")

    if target.item() != data.y[0]:
        print("Prediction does not match true label")
        continue

    explanation = explainer(data.x_dict, data.edge_index_dict, target =target, batch_dict = data.batch_dict, grads = False)

    work_dict = {}
    for key in explanation.node_mask_dict.keys():
        work_dict[key] = copy.deepcopy(explanation.node_mask_dict[key])
    print(work_dict.keys())
    # only consider the 100 most important nodes for each node type
    score = {}
    sum_scores = None
    num = 0
    for key in explanation.node_mask_dict.keys():
        
        if data.y[0] == target.item() == 1:
            if most_important_features_npdr[key] is None:
                most_important_features_npdr[key] = work_dict[key].sum(dim = 0).cpu().numpy()
            else:
                most_important_features_npdr[key] += work_dict[key].sum(dim = 0).cpu().numpy()
        elif data.y[0] == target.item() == 2:
            if most_important_features_pdr[key] is None:
                most_important_features_pdr[key] = work_dict[key].sum(dim = 0).cpu().numpy()
            else:
                most_important_features_pdr[key] += work_dict[key].sum(dim = 0).cpu().numpy()
        
        

# print the most important features for each node type with the highest score
# if the score is negative, the feature is not important
# if the score is positive, the feature is important

# sort the scores
combined_features_npdr = []
combined_features_pdr = []
for key in most_important_features_npdr.keys():
    # transfer the scores to a list
    most_important_features_npdr[key] = most_important_features_npdr[key].tolist()
    most_important_features_pdr[key] = most_important_features_pdr[key].tolist()
    # order the list descending, also with the feature names
    res_npdr = sorted(zip(most_important_features_npdr[key], features_label_dict[key]), key=lambda pair: pair[0], reverse = True)
    res_pdr = sorted(zip(most_important_features_pdr[key], features_label_dict[key]), key=lambda pair: pair[0], reverse = True)
    # add the name of the node type to the feature name
    res_npdr = [(f"{key}:{feat[1]}", feat[0]) for feat in res_npdr]
    res_pdr = [(f"{key}:{feat[1]}", feat[0]) for feat in res_pdr]
    # add the features to a combined list
    combined_features_npdr.extend(res_npdr)
    combined_features_pdr.extend(res_pdr)


# order the combined features by importance
combined_features_npdr = sorted(combined_features_npdr, key = lambda x: x[1], reverse = True)
combined_features_pdr = sorted(combined_features_pdr, key = lambda x: x[1], reverse = True)


top_k = 10
# only keep the features with a positive score
combined_features_npdr = [f for f in combined_features_npdr if f[1] > 0][:top_k]
combined_features_pdr = [f for f in combined_features_pdr if f[1] > 0][:top_k]

# divide the scores by the max score to get a relative importance
max_score_npdr = max([f[1] for f in combined_features_npdr])
max_score_pdr = max([f[1] for f in combined_features_pdr])

combined_features_npdr = [(f[0], f[1]/max_score_npdr) for f in combined_features_npdr]
combined_features_pdr = [(f[0], f[1]/max_score_pdr) for f in combined_features_pdr]

# create a bar plot of the most important features
node_typeto_color_char = {"graph_1": "#CC0000", "graph_2": "#0075E8", "faz": "#007E01"} 

color_list_npdr = []
# get the top features, assign them a color according to the node type
for idx, feat in enumerate(combined_features_npdr):
    color_list_npdr.append(node_typeto_color_char[feat[0].split(":")[0]])

color_list_pdr = []
# get the top features, assign them a color according to the node type
for idx, feat in enumerate(combined_features_pdr):
    color_list_pdr.append(node_typeto_color_char[feat[0].split(":")[0]])



fig, ax_out = plt.subplots(figsize=(10, 7))
ax_out.bar([f[0] for f in combined_features_npdr], [f[1] for f in combined_features_npdr], color = color_list_npdr)
# rotate the x labels
plt.xticks(rotation=45)
# remove the x labels
plt.xticks([])
plt.ylabel("Relative Feature Importance")
plt.tight_layout() 
#save the plot as vector graphic and png
print(save_path)
plt.savefig(f"{save_path}_most_important_features_npdr_no_labels.png")
plt.savefig(f"{save_path}_most_important_features_npdr_no_labels.svg")
plt.close("all")


fig, ax_out = plt.subplots(figsize=(10, 5))
ax_out.bar([f[0] for f in combined_features_pdr], [f[1] for f in combined_features_pdr], color = color_list_pdr)
# rotate the x labels 
plt.xticks(rotation=45)
# remove the x labels
plt.xticks([])
plt.ylabel("Relative Feature Importance")
plt.tight_layout() 
#save the plot as vector graphic and png
plt.savefig(f"{save_path}_most_important_features_pdr_no_labels.png")
plt.savefig(f"{save_path}_most_important_features_pdr_no_labels.svg")
plt.close("all")


            

    
