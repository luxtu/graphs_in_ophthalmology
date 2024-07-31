from loader import hetero_graph_loader
import torch
import copy
import json

from sklearn.metrics import accuracy_score, balanced_accuracy_score
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from models import graph_classifier, homogeneous_gnn
from torch_geometric.loader import DataLoader

from sklearn.metrics import roc_auc_score, roc_curve, auc
from utils import dataprep_utils, to_homogeneous_graph
import wandb
from sklearn.preprocessing import LabelBinarizer
from torch_geometric.nn import GATConv, SAGEConv, GraphConv, GCNConv

# empty cuda cache

# load the sweep configuration
with open("sweep_configs/sweep_config_workshop_trial.json", "rb") as file:
    sweep_configuration = json.load(file)

sweep_id = wandb.sweep(sweep=sweep_configuration, project= "workshop_first_trials")
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
train_dataset = to_homogeneous_graph.to_vessel_graph(train_dataset)
val_dataset = to_homogeneous_graph.to_vessel_graph(val_dataset)
test_dataset = to_homogeneous_graph.to_vessel_graph(test_dataset)

for data in train_dataset:
    data.to(device)
for data in val_dataset:
    data.to(device)
for data in test_dataset:
    data.to(device)


#train_dataset.to(device)
#val_dataset.to(device)
#test_dataset.to(device)


def main():
    #torch.autograd.set_detect_anomaly(True)
    run = wandb.init()

    model = homogeneous_gnn.Homogeneous_GNN(hidden_channels= wandb.config.hidden_channels,
                                                        out_channels= num_classes,
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
    balanced_loss = torch.nn.CrossEntropyLoss(class_weights)
    weak_balanced_loss = torch.nn.CrossEntropyLoss(class_weights_weak)
    unbalanced_loss = torch.nn.CrossEntropyLoss()

    loss_dict = {"balanced": balanced_loss, "unbalanced": unbalanced_loss, "weak_balanced": weak_balanced_loss}


    classifier = graph_classifier.graphClassifierSimple(model, loss_dict[wandb.config.class_weights], lr = wandb.config.lr, weight_decay =wandb.config.weight_decay, smooth_label_loss = wandb.config.smooth_label_loss) 

    best_val_bal_acc = 0
    best_mean_auc = 0
    best_pred = None
    for epoch in range(1, wandb.config.epochs + 1):
        loss, y_prob_train, y_true_train  = classifier.train(train_loader)
        y_prob_val, y_true_val = classifier.predict(val_loader)
        y_prob_test, y_true_test = classifier.predict(test_loader)
        #print("reached")

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
            if best_val_bal_acc > 0.65:

                torch.save(classifier.model.state_dict(), f"checkpoints_homogeneous/{wandb.config.dataset}_model_{wandb.config.split}_{run.id}.pt")

                # run inference on test set
                y_prob_test, y_true_test = classifier.predict(test_loader)

                y_pred_test = y_prob_test.argmax(axis=1)
                test_acc = accuracy_score(y_true_test, y_pred_test)
                test_bal_acc = balanced_accuracy_score(y_true_test, y_pred_test)
                res_dict["test_acc"] = test_acc
                res_dict["test_bal_acc"] = test_bal_acc

                print("#"*20)
                print(f"Test accuracy: {test_acc}")
                print(f"Test balanced accuracy: {test_bal_acc}")
                print("#"*20)

        res_dict["best_val_bal_acc"]= best_val_bal_acc

        wandb.log(res_dict)

    wandb.log({"roc": wandb.plot.roc_curve(y_true_val, best_pred, labels = label_names)})


wandb.agent(sweep_id, function=main, count=200)
