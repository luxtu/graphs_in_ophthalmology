
# %%
from loader import hetero_graph_loader
import torch
import copy
import json
import pandas as pd

from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from models import graph_classifier, homogeneous_gnn
from torch_geometric.loader import DataLoader
from utils import dataprep_utils, to_homogeneous_graph
import wandb
from torch_geometric.nn import GATConv, SAGEConv, GraphConv, GCNConv

# empty cuda cache

# load the sweep configuration
with open("sweep_configs/sweep_config_workshop_BVD.json", "rb") as file:
    sweep_configuration = json.load(file)

#sweep_id = wandb.sweep(sweep=sweep_configuration, project= "workshop_first_trials")
# loading data

data_type = sweep_configuration["parameters"]["dataset"]["values"][0]

octa_dr_dict = {"Healthy": 0, "DM": 0, "PDR": 2, "Early NPDR": 1, "Late NPDR": 1}
label_names = ["Healthy/DM", "NPDR", "PDR"]

vessel_graph_path = f"../data/{data_type}_vessel_graph"
label_file = "../data/splits"

mode_cv = "cv"
mode_final_test = "final_test"


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

#
split = sweep_configuration["parameters"]["split"]["values"][0]
train_dataset, val_dataset, test_dataset = dataprep_utils.adjust_data_for_split(cv_dataset, final_test_dataset, split, faz = False)



with open("feature_configs/feature_name_dict.json", "rb") as file:
    label_dict_full = json.load(file)
    #features_label_dict = json.load(file)
features_label_dict = copy.deepcopy(label_dict_full)

eliminate_features = {"graph_1":["num_voxels", "maxRadiusAvg", "hasNodeAtSampleBorder", "maxRadiusStd"], 
                      "graph_2":["centroid_weighted-0", "centroid_weighted-1", "feret_diameter_max", "orientation"]}


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

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# class weight extraction
train_labels = [int(data.y[0]) for data in train_dataset]
class_weights = dataprep_utils.get_class_weights(train_labels, verbose=False)
class_weights = torch.tensor(class_weights, device=device)
class_weights_weak = class_weights ** 0.5 

num_classes = class_weights.shape[0]
print(f"num classes: {num_classes}")
node_types = ["graph_1", "graph_2"]

agg_mode_dict = {"mean": global_mean_pool, "max": global_max_pool, "add": global_add_pool, "add_max": [global_add_pool, global_max_pool], "max_mean": [global_max_pool, global_mean_pool], "add_mean": [global_add_pool, global_mean_pool]}
homogeneous_conv_dict = {"gat": GATConv, "sage":SAGEConv, "graph" : GraphConv, "gcn" : GCNConv}
activation_dict = {"relu":torch.nn.functional.relu, "leaky" : torch.nn.functional.leaky_relu, "elu":torch.nn.functional.elu}


# create homogeneous graphs out of the heterogenous graphs
#train_dataset = to_homogeneous_graph.to_vessel_graph(train_dataset)
#val_dataset = to_homogeneous_graph.to_vessel_graph(val_dataset)
#test_dataset = to_homogeneous_graph.to_vessel_graph(test_dataset)

train_dataset = to_homogeneous_graph.to_void_graph(train_dataset)
val_dataset = to_homogeneous_graph.to_void_graph(val_dataset)
test_dataset = to_homogeneous_graph.to_void_graph(test_dataset)

for data in train_dataset:
    data.to(device)
for data in val_dataset:
    data.to(device)
for data in test_dataset:
    data.to(device)

# %%
# load the biomarker data
fractal_bvd = pd.read_csv("../data/fractal_BVD.csv", index_col="ID")

target_biomarker = "Fractal"
# set Fractal value of the biomarkers to the y value of the graph
for data in train_dataset:
    data.y = torch.tensor([fractal_bvd.loc[data.graph_id, target_biomarker]], dtype = torch.float32, device=device)
for data in val_dataset:
    data.y = torch.tensor([fractal_bvd.loc[data.graph_id, target_biomarker]], dtype = torch.float32, device=device)
for data in test_dataset:
    data.y = torch.tensor([fractal_bvd.loc[data.graph_id, target_biomarker]], dtype = torch.float32, device=device)



# %%
sweep_id = wandb.sweep(sweep=sweep_configuration, project= f"workshop_biomarker_pred_{target_biomarker}_void_graph")
def main():
    #torch.autograd.set_detect_anomaly(True)
    run = wandb.init()

    model = homogeneous_gnn.Homogeneous_GNN(hidden_channels= wandb.config.hidden_channels,
                                                        out_channels= 1,
                                                        num_layers= wandb.config.num_layers, 
                                                        dropout = wandb.config.dropout, 
                                                        aggregation_mode= agg_mode_dict[wandb.config.aggregation_mode], 
                                                        num_pre_processing_layers = wandb.config.pre_layers,
                                                        num_post_processing_layers = wandb.config.post_layers,
                                                        batch_norm = wandb.config.batch_norm,
                                                        homogeneous_conv= homogeneous_conv_dict[wandb.config.homogeneous_conv],
                                                        activation= activation_dict[wandb.config.activation],
                                                        )

    # create data loaders for training and test set
    train_loader = DataLoader(train_dataset, batch_size = wandb.config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size = 64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size = 64, shuffle=False)



    # weigthings for imbalanced classes 
    l1_loss = torch.nn.L1Loss()

    classifier = graph_classifier.graphRegressorSimple(model, l1_loss, lr = wandb.config.lr, weight_decay =wandb.config.weight_decay) 

    best_mae = 1000
    best_rmse = 1000

    for epoch in range(1, wandb.config.epochs + 1):
        loss, _, _  = classifier.train(train_loader)

        y_pred_train, y_true_train = classifier.predict(train_loader)
        y_pred_val, y_true_val = classifier.predict(val_loader)
        #y_pred_test, y_true_test = classifier.predict(test_loader)

        # squeeze the last dimension
        y_pred_train = y_pred_train.squeeze()
        y_pred_val = y_pred_val.squeeze()

        # calculate the metrics
        train_mae = l1_loss(torch.tensor(y_pred_train), torch.tensor(y_true_train)).item()
        val_mae = l1_loss(torch.tensor(y_pred_val), torch.tensor(y_true_val)).item()

        # calculate teh root mean squared error
        train_rmse = torch.sqrt(torch.nn.functional.mse_loss(torch.tensor(y_pred_train), torch.tensor(y_true_train))).item()
        val_rmse = torch.sqrt(torch.nn.functional.mse_loss(torch.tensor(y_pred_val), torch.tensor(y_true_val))).item()
        res_dict =  {"loss": loss, "val_mae": val_mae, "val_rmse": val_rmse, "train_mae": train_mae, "train_rmse": train_rmse}

        # save the best model
        if val_mae < best_mae:
            best_mae = val_mae
            best_rmse = val_rmse
            torch.save(classifier.model.state_dict(), f"checkpoints_homogeneous/{wandb.config.dataset}_model_{wandb.config.split}_{target_biomarker}_{run.id}.pt")
            res_dict["best_mae"] = best_mae

            # evaluate the model on the test set
            y_pred_test, y_true_test = classifier.predict(test_loader)
            y_pred_test = y_pred_test.squeeze()
            test_mae = l1_loss(torch.tensor(y_pred_test), torch.tensor(y_true_test)).item()
            test_rmse = torch.sqrt(torch.nn.functional.mse_loss(torch.tensor(y_pred_test), torch.tensor(y_true_test))).item()
            res_dict["test_mae"] = test_mae
            res_dict["test_rmse"] = test_rmse


        wandb.log(res_dict)



wandb.agent(sweep_id, function=main, count=60)

# %%
