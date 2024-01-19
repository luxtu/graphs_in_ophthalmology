from loader import hetero_graph_loader_faz
import numpy as np 
import torch
import copy
import json

from sklearn.metrics import accuracy_score, balanced_accuracy_score
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from models import graph_classifier, hetero_gnn, global_node_gnn, hierarchical_gnn, spatial_pooling_gnn
from torch_geometric.loader import DataLoader

from sklearn.metrics import roc_auc_score, roc_curve, auc
from utils import prep
from evaluation import evaluation
import wandb
from sklearn.preprocessing import LabelBinarizer
from torch_geometric.nn import GATConv, SAGEConv, GraphConv, GCNConv


# Define sweep config
sweep_configuration = {
    "method": "random",
    "name": "CIS vs HC",
    "metric": {"goal": "maximize", "name": "best_val_bal_acc"},
    "parameters": {
        "batch_size": {"values": [8, 16, 32, 64]},
        "epochs": {"values": [100]},
        "lr": {"max": 0.05, "min": 0.005}, # learning rate to high does not work
        "weight_decay": {"max": 0.01, "min": 0.00001},
        "hidden_channels": {"values": [16, 32, 64]}, #[64, 128]
        "dropout": {"values": [0.1, 0.3, 0.4, 0.6]}, # 0.2,  more droput looks better
        "num_layers": {"values": [1,2,5]},
        "aggregation_mode": {"values": ["mean", "add", "max", "add_max", "max_mean", "add_mean"]},#, "global_mean_pool",  "global_add_pool" # add pool does not work
        "pre_layers": {"values": [1,2,4]},
        "post_layers": {"values": [1,2,4]},
        "batch_norm": {"values": [True, False]},
        "hetero_conns": {"values": [True, False]},
        "conv_aggr": {"values": ["cat", "sum", "mean"]}, # cat, sum, mean
        "class_weights": {"values": ["unbalanced", "balanced", "weak_balanced"]}, # "balanced", 
        "dataset": {"values": ["DVC"]}, #, "DCP"
        "regression": {"values": [False]},
        "homogeneous_conv": {"values": ["gat", "sage", "graph", "gcn"]},
        "heterogeneous_conv": {"values": ["gat", "sage", "graph"]},
        "activation": {"values": ["relu", "leaky", "elu"]},
    },
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="tum_neuro_gnn")
# loading data

data_type = sweep_configuration["parameters"]["dataset"]["values"][0]

octa_dr_dict = {"HC": 0, "MS": 2, "NMOSD": 2, "NNO": 2, "CIS": 1, "Fabry": 2, "Sjogren": 2}
label_names = ["HC", "MS","Rest"]

vessel_graph_path = f"../../TumNeuro/checked_seg_{data_type}_graph_new_idx/graph"
label_file = f"../../TumNeuro/checked_seg_{data_type}_graph_new_idx/{data_type}_graph_new_idx.csv"
#void_graph_path = f"/../../TumNeuro/checked_seg_{data_type}_void_graph"
#hetero_edges_path = f"/../../TumNeuro/checked_seg_{data_type}_heter_edges"

void_graph_path = f"../../TumNeuro/checked_seg_{data_type}_graph_new_idx/void_graph_faz"
hetero_edges_path = f"../../TumNeuro/checked_seg_{data_type}_graph_new_idx/hetero_edges_faz"

faz_node_path = f"../../TumNeuro/checked_seg_{data_type}_graph_new_idx/faz_node"
faz_region_edge_path = f"../../TumNeuro/checked_seg_{data_type}_graph_new_idx/faz_region_edges"
faz_vessel_edge_path = f"../../TumNeuro/checked_seg_{data_type}_graph_new_idx/faz_vessel_edges"

mode_train = "train"
train_pickle = f"../../TumNeuro/checked_seg_{data_type}_graph_new_idx/{data_type}_{mode_train}_dataset_cis_faz.pkl"
train_dataset = hetero_graph_loader_faz.HeteroGraphLoaderTorch(graph_path_1=vessel_graph_path,
                                                        graph_path_2=void_graph_path,
                                                        graph_path_3=faz_node_path,
                                                        hetero_edges_path_12=hetero_edges_path,
                                                        hetero_edges_path_13=faz_vessel_edge_path,
                                                        hetero_edges_path_23=faz_region_edge_path,
                                                        mode = mode_train,
                                                        label_file = label_file, 
                                                        line_graph_1 =True, 
                                                        class_dict = octa_dr_dict,
                                                        pickle_file = train_pickle #f"../{data_type}_{mode_train}_dataset_faz.pkl" # f"../{data_type}_{mode_train}_dataset_faz.pkl"
                                                        )
mode_val = "val"
val_pickle = f"../../TumNeuro/checked_seg_{data_type}_graph_new_idx/{data_type}_{mode_val}_dataset_cis_faz.pkl"
val_dataset = hetero_graph_loader_faz.HeteroGraphLoaderTorch(graph_path_1=vessel_graph_path,
                                                        graph_path_2=void_graph_path,
                                                        graph_path_3=faz_node_path,
                                                        hetero_edges_path_12=hetero_edges_path,
                                                        hetero_edges_path_13=faz_vessel_edge_path,
                                                        hetero_edges_path_23=faz_region_edge_path,
                                                        mode = mode_val,
                                                        label_file = label_file, 
                                                        line_graph_1 =True, 
                                                        class_dict = octa_dr_dict,
                                                        pickle_file = val_pickle # f"../{data_type}_{mode_test}_dataset_faz.pkl" #f"../{data_type}_{mode_test}_dataset_faz.pkl")
                                                        )

print(len(train_dataset))
print(len(val_dataset))

# delete all samples from the list that have label 2
train_dataset.hetero_graph_list = [data for data in train_dataset.hetero_graph_list if data.y[0] != 2]
train_dataset.hetero_graphs = {k: v for k, v in train_dataset.hetero_graphs.items() if v.y[0] != 2}

val_dataset.hetero_graph_list = [data for data in val_dataset.hetero_graph_list if data.y[0] != 2]
val_dataset.hetero_graphs = {k: v for k, v in val_dataset.hetero_graphs.items() if v.y[0] != 2}


# len after the elimination of class 2
print(len(train_dataset))
print(len(val_dataset))

# print the number of samples per class
print("train")
print(np.unique([data.y[0] for data in train_dataset], return_counts=True))
print("val")
print(np.unique([data.y[0] for data in val_dataset], return_counts=True))


# imputation and normalization
prep.hetero_graph_imputation(train_dataset)
prep.hetero_graph_imputation(val_dataset)

prep.add_node_features(train_dataset, ["graph_1", "graph_2"])
prep.add_node_features(val_dataset, ["graph_1", "graph_2"])

#prep.add_global_node(train_dataset)
#prep.add_global_node(val_dataset)

node_mean_tensors, node_std_tensors = prep.hetero_graph_normalization_params(train_dataset)

# save the normalization parameters

prep.hetero_graph_normalization(train_dataset, node_mean_tensors, node_std_tensors)
prep.hetero_graph_normalization(val_dataset, node_mean_tensors, node_std_tensors)

with open("feature_name_dict.json", "r") as file:
    label_dict_full = json.load(file)
    #features_label_dict = json.load(file)
features_label_dict = copy.deepcopy(label_dict_full)

#eliminate_features = {"graph_1":["num_voxels", "maxRadiusAvg", "hasNodeAtSampleBorder", "maxRadiusStd"], 
#                      "graph_2":["centroid_weighted-0", "centroid_weighted-1", "feret_diameter_max", "equivalent_diameter"]}
#eliminate_features = {"graph_1":["num_voxels", "maxRadiusAvg", "hasNodeAtSampleBorder", "maxRadiusStd"], 
#                      "graph_2":["centroid_weighted-0", "centroid_weighted-1", "centroid-0", "centroid-1", "feret_diameter_max", "equivalent_diameter", "orientation"]}
## get positions of features to eliminate and remove them from the feature label dict and the graphs
#for key in eliminate_features.keys():
#    for feat in eliminate_features[key]:
#        idx = features_label_dict[key].index(feat)
#        features_label_dict[key].remove(feat)
#        for data in train_dataset:
#            data[key].x = torch.cat([data[key].x[:, :idx], data[key].x[:, idx+1:]], dim = 1)
#        for data in val_dataset:
#            data[key].x = torch.cat([data[key].x[:, :idx], data[key].x[:, idx+1:]], dim = 1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# class weight extraction
train_labels = [int(data.y[0]) for data in train_dataset]
class_weights = prep.get_class_weights(train_labels, verbose=False)
class_weights = torch.tensor(class_weights, device=device)
class_weights_weak = class_weights ** 0.5 

num_classes = class_weights.shape[0]
node_types = ["graph_1", "graph_2"]

agg_mode_dict = {"mean": global_mean_pool, "max": global_max_pool, "add": global_add_pool, "add_max": [global_add_pool, global_max_pool], "max_mean": [global_max_pool, global_mean_pool], "add_mean": [global_add_pool, global_mean_pool]}
homogeneous_conv_dict = {"gat": GATConv, "sage":SAGEConv, "graph" : GraphConv, "gcn" : GCNConv}
heterogeneous_conv_dict = {"gat": GATConv, "sage":SAGEConv, "graph" : GraphConv}
activation_dict = {"relu":torch.nn.functional.relu, "leaky" : torch.nn.functional.leaky_relu, "elu":torch.nn.functional.elu}


train_dataset.to(device)
val_dataset.to(device)


def main():
    #torch.autograd.set_detect_anomaly(True)
    # initialize wandb
    wandb.init()


    model = global_node_gnn.GNN_global_node(hidden_channels= wandb.config.hidden_channels,
                                                        out_channels= num_classes,
                                                        num_layers= wandb.config.num_layers, 
                                                        dropout = wandb.config.dropout, 
                                                        aggregation_mode= agg_mode_dict[wandb.config.aggregation_mode], 
                                                        node_types = node_types,
                                                        num_pre_processing_layers = wandb.config.pre_layers,
                                                        num_post_processing_layers = wandb.config.post_layers,
                                                        batch_norm = wandb.config.batch_norm,
                                                        conv_aggr = wandb.config.conv_aggr,
                                                        hetero_conns = wandb.config.hetero_conns,
                                                        #meta_data = train_dataset[0].metadata())
                                                        homogeneous_conv= homogeneous_conv_dict[wandb.config.homogeneous_conv],
                                                        heterogeneous_conv= heterogeneous_conv_dict[wandb.config.heterogeneous_conv],
                                                        activation= activation_dict[wandb.config.activation],
                                                        )

    # create data loaders for training and val set
    train_loader = DataLoader(train_dataset, batch_size = wandb.config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size = 1, shuffle=False)

    # weigthings for imbalanced classes 
    balanced_loss = torch.nn.CrossEntropyLoss(class_weights)
    weak_balanced_loss = torch.nn.CrossEntropyLoss(class_weights_weak)
    unbalanced_loss = torch.nn.CrossEntropyLoss()

    loss_dict = {"balanced": balanced_loss, "unbalanced": unbalanced_loss, "weak_balanced": weak_balanced_loss}

    if wandb.config.regression:
        reg_loss = torch.nn.SmoothL1Loss()
        classifier = graph_classifier.graphClassifierHetero(model,reg_loss , lr = wandb.config.lr, weight_decay =wandb.config.weight_decay, regression=wandb.config.regression)
    else:
        classifier = graph_classifier.graphClassifierHetero(model, loss_dict[wandb.config.class_weights], lr = wandb.config.lr, weight_decay =wandb.config.weight_decay, regression=wandb.config.regression) 

    best_val_bal_acc = 0
    best_mean_auc = 0
    best_pred = None
    print("Start training")
    for epoch in range(1, wandb.config.epochs + 1):
        loss, y_prob_train, y_true_train, _  = classifier.train(train_loader)
        y_prob_val, y_true_val = classifier.predict(val_loader)

        if wandb.config.regression:
            y_pred_val = y_prob_val.squeeze() 
        else:
            y_pred_val = y_prob_val.argmax(axis=1)
        
        train_acc = accuracy_score(y_true_train, y_prob_train.argmax(axis=1))
        train_bal_acc = balanced_accuracy_score(y_true_train, y_prob_train.argmax(axis=1))

        val_acc = accuracy_score(y_true_val, y_pred_val)
        val_bal_acc = balanced_accuracy_score(y_true_val, y_pred_val)
        res_dict =  {"loss": loss, "val_acc": val_acc, "val_bal_acc": val_bal_acc, "train_acc": train_acc, "train_bal_acc": train_bal_acc}

        if not wandb.config.regression:
            y_p_softmax = torch.nn.functional.softmax(torch.tensor(y_prob_val), dim = 1).detach().numpy()
            if num_classes == 2:
                mean_auc = roc_auc_score(
                    y_true=y_true_val,
                    y_score = y_p_softmax[:,1],
                    multi_class="ovr",
                    average="macro",
                )
            else:
                mean_auc = roc_auc_score(
                    y_true=y_true_val,
                    y_score = y_p_softmax,
                    multi_class="ovr",
                    average="macro",
                )
                label_binarizer = LabelBinarizer().fit(y_true_val)
                y_onehot_val = label_binarizer.transform(y_true_val)
                # get auc for each class
                fpr = dict()
                tpr = dict()
                roc_auc = dict()
                for i in range(num_classes):
                    fpr[i], tpr[i], _ = roc_curve(y_onehot_val[:, i], y_p_softmax[:, i])
                    roc_auc[i] = auc(fpr[i], tpr[i])
                    res_dict[f"roc_auc_{label_names[i]}"] = roc_auc[i]

            if mean_auc > best_mean_auc or (mean_auc == best_mean_auc and val_bal_acc > best_val_bal_acc):
                best_mean_auc = mean_auc
                best_pred = y_p_softmax

            #if best_mean_auc > th_swp:
            #    torch.save(model.state_dict(), f'./checkpoints/model_{wandb.config.aggregation_mode}_best_auc.pt')

            res_dict["mean_auc"] = mean_auc
            res_dict["best_mean_auc"] = best_mean_auc

        if val_bal_acc > best_val_bal_acc:
            best_val_bal_acc = val_bal_acc


        res_dict["best_val_bal_acc"]= best_val_bal_acc

        wandb.log(res_dict)

    if not wandb.config.regression:
        wandb.log({"roc": wandb.plot.roc_curve(y_true_val, best_pred, labels = label_names)})


wandb.agent(sweep_id, function=main, count=100)