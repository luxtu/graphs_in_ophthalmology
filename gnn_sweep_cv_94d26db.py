import torch
import copy
import json

from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from models import graph_classifier, global_node_gnn
from torch_geometric.loader import DataLoader

from utils import prep, dataprep_utils, train_utils
import wandb
from torch_geometric.nn import GATConv, SAGEConv, GraphConv, GCNConv

# empty cuda cache

# Define sweep config
with open('training_configs/sweep_config_94d26b_split3.json', 'r') as file:
    sweep_configuration = json.load(file)

sweep_configuration["method"] = "random"
sweep_configuration["name"] = "Selected sweep, repeat heterogeneous edges, 3 splits, 94d26db"
sweep_id = wandb.sweep(sweep=sweep_configuration, project= "selected_sweep_repeat")
# change the name of the sweep

# loading data

data_type = sweep_configuration["parameters"]["dataset"]["values"][0]
faz_node_bool = sweep_configuration["parameters"]["faz_node"]["values"][0]


label_names = ["Healthy/DM", "NPDR", "PDR"]


mode_cv = "cv"
mode_final_test = "final_test"

# adjust the dataset for the split
cv_pickle_processed = f"../data/{data_type}_{mode_cv}_selected_sweep_repeat_v2.pkl"
final_test_pickle_processed = f"../data/{data_type}_{mode_final_test}_selected_sweep_repeat_v2.pkl" 


import pickle
with open(cv_pickle_processed, "rb") as file:
    cv_dataset = pickle.load(file)
with open(final_test_pickle_processed, "rb") as file:
    final_test_dataset = pickle.load(file)


split = sweep_configuration["parameters"]["split"]["values"][0]
train_dataset, val_dataset, test_dataset = dataprep_utils.adjust_data_for_split(cv_dataset, final_test_dataset, split, faz = sweep_configuration["parameters"]["faz_node"]["values"][0])


with open("training_configs/feature_name_dict.json", "r") as file:
    label_dict_full = json.load(file)
    #features_label_dict = json.load(file)
features_label_dict = copy.deepcopy(label_dict_full)

eliminate_features = {"graph_1":["num_voxels","hasNodeAtSampleBorder", "maxRadiusAvg", "maxRadiusStd"], 
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
heterogeneous_conv_dict = {"gat": GATConv, "sage":SAGEConv, "graph" : GraphConv}
activation_dict = {"relu":torch.nn.functional.relu, "leaky" : torch.nn.functional.leaky_relu, "elu":torch.nn.functional.elu}


train_dataset.to(device)
val_dataset.to(device)
test_dataset.to(device)


def main():
    #torch.autograd.set_detect_anomaly(True)
    run = wandb.init()

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
                                                        homogeneous_conv= homogeneous_conv_dict[wandb.config.homogeneous_conv],
                                                        heterogeneous_conv= heterogeneous_conv_dict[wandb.config.heterogeneous_conv],
                                                        activation= activation_dict[wandb.config.activation],
                                                        faz_node = wandb.config.faz_node,
                                                        start_rep = wandb.config.start_rep,
                                                        aggr_faz = wandb.config.aggr_faz,
                                                        faz_conns= wandb.config.faz_conns
                                                        )

    # create data loaders for training and test set
    train_loader = DataLoader(train_dataset, batch_size = wandb.config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size = 64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size = 64, shuffle=False)



    # weigthings for imbalanced classes 
    balanced_loss = torch.nn.CrossEntropyLoss(class_weights)
    weak_balanced_loss = torch.nn.CrossEntropyLoss(class_weights_weak)
    unbalanced_loss = torch.nn.CrossEntropyLoss()

    loss_dict = {"balanced": balanced_loss, "unbalanced": unbalanced_loss, "weak_balanced": weak_balanced_loss}


    classifier = graph_classifier.graphClassifierHetero_94d26db(model, loss_dict[wandb.config.class_weights], lr = wandb.config.lr, weight_decay =wandb.config.weight_decay, smooth_label_loss = wandb.config.smooth_label_loss) 

    best_val_bal_acc = 0
    test_bal_acc_best_val = 0
    best_mean_auc = 0
    best_pred = None
    for epoch in range(1, wandb.config.epochs + 1):
        loss, y_prob_train, y_true_train, _  = classifier.train(train_loader)
        res_dict, y_pred_softmax_val, y_true_val = train_utils.evaluate_model(classifier, val_loader, test_loader)
        train_acc = accuracy_score(y_true_train, y_prob_train.argmax(axis=1))
        train_bal_acc = balanced_accuracy_score(y_true_train, y_prob_train.argmax(axis=1))

        res_dict["train_acc"] = train_acc
        res_dict["train_bal_acc"] = train_bal_acc
        res_dict["loss"] = loss

        if res_dict["val_bal_acc"] > best_val_bal_acc and res_dict["train_bal_acc"] > 0.65: # only update if the training accuracy is high enough, is now not really best val bac anymore, but best val bac with high enough train acc
            best_val_bal_acc = res_dict["val_bal_acc"]
            test_bal_acc_best_val = res_dict["test_bal_acc"]
            if best_val_bal_acc > 0.65:
                torch.save(classifier.model.state_dict(), f"checkpoints/{wandb.config.dataset}_model_{wandb.config.split}_faz_node_{wandb.config.faz_node}_{run.id}.pt")
                print("#"*20)
                test_acc = res_dict["test_acc"]
                test_bal_acc = res_dict["test_bal_acc"]
                print(f"Test accuracy: {test_acc}")
                print(f"Test balanced accuracy: {test_bal_acc}")
                print("#"*20)
        if res_dict["val_mean_auc"] > best_mean_auc or (res_dict["val_mean_auc"] == best_mean_auc and res_dict["val_bal_acc"]  > best_val_bal_acc):
            best_mean_auc = res_dict["val_mean_auc"]
            best_pred = y_pred_softmax_val
        res_dict["best_val_bal_acc"]= best_val_bal_acc
        res_dict["test_bal_acc_best_val"] = test_bal_acc_best_val
        wandb.log(res_dict)

    wandb.log({"roc": wandb.plot.roc_curve(y_true_val, best_pred, labels = label_names)})

wandb.agent(sweep_id, function=main, count=1000)
