from loader import hetero_graph_loader, hetero_graph_loader_faz
import numpy as np 
import torch
import copy
import json

from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from models import graph_classifier, hetero_gnn, hgt_gnn
from torch_geometric.loader import DataLoader

from sklearn.metrics import roc_auc_score, roc_curve, auc
from utils import prep
from evaluation import evaluation
import wandb
from sklearn.preprocessing import LabelBinarizer
from torch_geometric.nn import GATConv, SAGEConv, GraphConv, GCNConv

# empty cuda cache

# Define sweep config
sweep_configuration = {
    "method": "random",
    "name": "HGT, Less LR, Less Heads, Less Layers Grid Search 3 Class, Split 3, Eliminated Features, Smooth Label Loss(Really), No Graph Cleaning, Faz Node, Isolated Nodes Removed",
    "metric": {"goal": "maximize", "name": "best_val_bal_acc"},
    "parameters": {
        "batch_size": {"values": [8, 16, 32, 64]},
        "epochs": {"values": [100]},
        "lr": {"max": 0.01, "min": 0.001}, # learning rate to high does not work # "max": 0.05, "min": 0.005} prev, put lower
        "weight_decay": {"max": 0.01, "min": 0.00001},
        "hidden_channels": {"values": [16, 32, 64]}, #[64, 128]
        "dropout": {"values": [0.1, 0.3, 0.4]}, # 0.2,  more droput looks better
        "num_layers": {"values": [1, 2]}, # ,2,5
        "aggregation_mode": {"values": ["add_max", "max_mean", "add", "mean"]},# removed, "add", "max", "add_mean", "mean", "add_max",
        "pre_layers": {"values": [1,2,4]},
        "post_layers": {"values": [1,2,4]},
        "final_layers": {"values": [1,2,4]},
        "num_heads": {"values": [1, 2, 4]}, # , 8,
        "batch_norm": {"values": [True]},  #, False
        "class_weights": {"values": ["balanced"]}, # "balanced", #  # "unbalanced", # , "weak_balanced" 
        "dataset": {"values": ["DCP"]}, #, "DCP"
        "activation": {"values": ["leaky"]}, # "relu", "elu",
        "faz_node": {"values": [True]},
        "split": {"values": [3]},
        "start_rep": {"values": [True, False]},
        "aggr_faz": {"values": [True, False]},
        "smooth_label_loss": {"values": [True, False]},
    },
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project= "gnn_cv_3_class_hgt")
# loading data

data_type = sweep_configuration["parameters"]["dataset"]["values"][0]
faz_node_bool = sweep_configuration["parameters"]["faz_node"]["values"][0]

octa_dr_dict = {"Healthy": 0, "DM": 0, "PDR": 2, "Early NPDR": 1, "Late NPDR": 1}
label_names = ["Healthy/DM", "NPDR", "PDR"]

#vessel_graph_path = f"..data/{data_type}_vessel_graph"
#void_graph_path = f"/media/data/alex_johannes/octa_data/Cairo/{data_type}_void_graph"
#hetero_edges_path = f"/media/data/alex_johannes/octa_data/Cairo/{data_type}_heter_edges"
#label_file = "/media/data/alex_johannes/octa_data/Cairo/labels.csv"
    #label_file = "..data/labels.csv"
vessel_graph_path = f"../data/{data_type}_vessel_graph"
label_file = "../data/splits"

mode_cv = "cv"
mode_final_test = "final_test"

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
    void_graph_path = f"../data/{data_type}_void_graph_faz_node"
    hetero_edges_path = f"../data/{data_type}_hetero_edges_faz_node"
    faz_node_path = f"../data/{data_type}_faz_nodes"
    faz_region_edge_path = f"../data/{data_type}_faz_region_edges"
    faz_vessel_edge_path = f"../data/{data_type}_faz_vessel_edges"

    cv_pickle = f"../data/{data_type}_{mode_cv}_dataset_faz_isol_not_removed_more_features.pkl"  # _isol_not_removed
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

    final_test_pickle = f"../data/{data_type}_{mode_final_test}_dataset_faz_isol_not_removed_more_features.pkl"   # _isol_not_removed
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


#octa_dr_dict = {"Healthy": 0, "DM": 0, "PDR": 3, "Early NPDR": 1, "Late NPDR": 2}
#label_names = ["Healthy/DM", "Early NPDR", "Late NPDR", "PDR"]
# update the label dict
#final_test_dataset.update_class(octa_dr_dict)
#cv_dataset.update_class(octa_dr_dict)
image_path = f"../data/{data_type}_images"
json_path = f"../data/{data_type}_json"
seg_size = 1216
#
cv_dataset_cl = prep.add_centerline_statistics_multi(cv_dataset, image_path, json_path, seg_size)
final_test_dataset_cl = prep.add_centerline_statistics_multi(final_test_dataset, image_path, json_path, seg_size)
#
## clean the data
cv_dataset.hetero_graph_list = prep.hetero_graph_cleanup_multi(cv_dataset_cl)
final_test_dataset.hetero_graph_list = prep.hetero_graph_cleanup_multi(final_test_dataset_cl)

cv_dataset.hetero_graph_list = cv_dataset_cl
final_test_dataset.hetero_graph_list = final_test_dataset_cl

cv_dataset_dict = dict(zip([graph.graph_id for graph in cv_dataset.hetero_graph_list], cv_dataset.hetero_graph_list))
final_test_dataset_dict = dict(zip([graph.graph_id for graph in final_test_dataset.hetero_graph_list], final_test_dataset.hetero_graph_list))
#
## set the dictionaries
cv_dataset.hetero_graphs = cv_dataset_dict
final_test_dataset.hetero_graphs = final_test_dataset_dict


split = sweep_configuration["parameters"]["split"]["values"][0]
train_dataset, val_dataset, test_dataset = prep.adjust_data_for_split(cv_dataset, final_test_dataset, split, faz = sweep_configuration["parameters"]["faz_node"]["values"][0])


# remove label noise, samples to exclude are stored in label_noise.json
#with open("label_noise.json", "r") as file:
#    label_noise_dict = json.load(file)
#prep.remove_label_noise(train_dataset, label_noise_dict)


with open("feature_name_dict.json", "r") as file:
    label_dict_full = json.load(file)
    #features_label_dict = json.load(file)
features_label_dict = copy.deepcopy(label_dict_full)

eliminate_features = {"graph_1":["num_voxels", "maxRadiusAvg", "hasNodeAtSampleBorder", "maxRadiusStd"], 
                      "graph_2":["centroid_weighted-0", "centroid_weighted-1", "feret_diameter_max", "orientation"]}

#eliminate_features = {"graph_1":["maxRadiusAvg", "hasNodeAtSampleBorder", "maxRadiusStd", "minRadiusAvg",  "avgRadiusStd", "roundnessStd", "roundnessAvg", "curveness"], 
#                      "graph_2":["centroid_weighted-0", "centroid_weighted-1", "feret_diameter_max", "orientation", "intensity_max", "centroid-0", "centroid-1", "solidity", "extent", "degree"]}

if faz_node_bool:
    eliminate_features["faz"] = ["centroid_weighted-0", "centroid_weighted-1","feret_diameter_max", "orientation"] # , "centroid-0", "centroid-1", "solidity", "extent"


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
class_weights = prep.get_class_weights(train_labels, verbose=False)
class_weights = torch.tensor(class_weights, device=device)
class_weights_weak = class_weights ** 0.5 

num_classes = class_weights.shape[0]
print(f"num classes: {num_classes}")
node_types = ["graph_1", "graph_2"]

agg_mode_dict = {"mean": global_mean_pool, "max": global_max_pool, "add": global_add_pool, "add_max": [global_add_pool, global_max_pool], "max_mean": [global_max_pool, global_mean_pool], "add_mean": [global_add_pool, global_mean_pool]}
activation_dict = {"relu":torch.nn.functional.relu, "leaky" : torch.nn.functional.leaky_relu, "elu":torch.nn.functional.elu}


train_dataset.to(device)
val_dataset.to(device)
test_dataset.to(device)

# print example graph
print(train_dataset[0])

def main():
    #torch.autograd.set_detect_anomaly(True)
    run = wandb.init()
    #api = wandb.Api()
    # ptint the run id
    #print(f"Run ID: {run.id}")
    #sweep = api.sweep("luxtu/gnn_cv_3_class/" + sweep_id)
    #try: 
    #    th_swp = sweep.best_run(order='best_mean_auc').summary_metrics['best_mean_auc']
    #except KeyError:
    #    th_swp = 0
    #except AttributeError:
    #    th_swp = 0


    model = hgt_gnn.HGT_GNN(hidden_channels= wandb.config.hidden_channels,
                                                        out_channels= num_classes,
                                                        num_layers= wandb.config.num_layers, 
                                                        dropout = wandb.config.dropout, 
                                                        aggregation_mode= agg_mode_dict[wandb.config.aggregation_mode], 
                                                        node_types = node_types,
                                                        num_pre_processing_layers = wandb.config.pre_layers,
                                                        num_post_processing_layers = wandb.config.post_layers,
                                                        batch_norm = wandb.config.batch_norm,
                                                        meta_data= train_dataset[0].metadata(),
                                                        num_heads= wandb.config.num_heads,
                                                        activation= activation_dict[wandb.config.activation],
                                                        faz_node = wandb.config.faz_node,
                                                        start_rep = wandb.config.start_rep,
                                                        aggr_faz = wandb.config.aggr_faz,
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


    classifier = graph_classifier.graphClassifierHetero(model, loss_dict[wandb.config.class_weights], lr = wandb.config.lr, weight_decay =wandb.config.weight_decay, regression=False, smooth_label_loss = wandb.config.smooth_label_loss) 

    best_val_bal_acc = 0
    best_val_bal_acc_post = 0

    best_mean_auc = 0
    best_pred = None
    for epoch in range(1, wandb.config.epochs + 1):
        loss, y_prob_train, y_true_train, _  = classifier.train(train_loader)
        y_prob_val, y_true_val = classifier.predict(val_loader)
        y_prob_test, y_true_test = classifier.predict(test_loader)


        y_pred_val = y_prob_val.argmax(axis=1)
        y_pred_test = y_prob_test.argmax(axis=1)

        
        train_acc = accuracy_score(y_true_train, y_prob_train.argmax(axis=1))
        train_bal_acc = balanced_accuracy_score(y_true_train, y_prob_train.argmax(axis=1))

        val_acc = accuracy_score(y_true_val, y_pred_val)
        val_bal_acc = balanced_accuracy_score(y_true_val, y_pred_val)

        test_acc = accuracy_score(y_true_test, y_pred_test)
        test_bal_acc = balanced_accuracy_score(y_true_test, y_pred_test)

        res_dict =  {"loss": loss, "val_acc": val_acc, "val_bal_acc": val_bal_acc, "train_acc": train_acc, "train_bal_acc": train_bal_acc}
        res_dict["test_acc"] = test_acc
        res_dict["test_bal_acc"] = test_bal_acc



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


        if val_bal_acc > 0.60 and val_bal_acc > best_val_bal_acc_post and epoch > 50:
            best_val_bal_acc_post = val_bal_acc
            torch.save(classifier.model.state_dict(), f"checkpoints/HGT_{wandb.config.dataset}_model_{wandb.config.split}_faz_node_{wandb.config.faz_node}_{run.id}.pt")
            print("#"*20)
            print(f"Test accuracy: {test_acc}")
            print(f"Test balanced accuracy: {test_bal_acc}")
            print("#"*20)

            res_dict["best_val_bal_acc_post"] = best_val_bal_acc_post



        kappa = cohen_kappa_score(y_true_val, y_pred_val, weights= "quadratic")

        res_dict["kappa"] = kappa
        res_dict["best_val_bal_acc"]= best_val_bal_acc

        wandb.log(res_dict)


    wandb.log({"roc": wandb.plot.roc_curve(y_true_val, best_pred, labels = label_names)})

wandb.agent(sweep_id, function=main, count=10)
