from loader import hetero_graph_loader
import numpy as np 
import torch

from sklearn.metrics import accuracy_score, balanced_accuracy_score
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from models import graph_classifier, hetero_gnn, global_node_gnn, hierarchical_gnn
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose, RemoveIsolatedNodes

from sklearn.metrics import roc_auc_score
from utils import prep
from evaluation import evaluation
import wandb

torch.cuda.empty_cache()

# Define sweep config
sweep_configuration = {
    "method": "bayes",
    "name": "sweep",
    "metric": {"goal": "maximize", "name": "best_mean_auc"},
    "parameters": {
        "batch_size": {"values": [2]},
        "epochs": {"values": [50]},
        "lr": {"max": 0.05, "min": 0.005}, # learning rate to high does not work
        "weight_decay": {"max": 0.01, "min": 0.00001},
        "hidden_channels": {"values": [16, 32]}, #[64, 128]
        "dropout": {"values": [0.3, 0.4]}, # 0.2,  more droput looks better
        "num_layers": {"values": [1,2,3]},
        "aggregation_mode": {"values": ["global_mean_pool", "global_max_pool"]},#, "global_mean_pool",  "global_add_pool" # add pool does not work
        "class_weights": {"values": ["unbalanced"]}, # "balanced", 
        "dataset": {"values": ["DCP"]} #, "DCP"
    },
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="graph_pathology")
# loading data

data_type = sweep_configuration["parameters"]["dataset"]["values"][0]

octa_dr_dict = {"Healthy": 0, "DM": 0, "PDR": 1, "Early NPDR": 2, "Late NPDR": 2}
label_names = ["Healthy/DM", "PDR", "NPDR"]

vessel_graph_path = f"/media/data/alex_johannes/octa_data/Cairo/{data_type}_vessel_graph"
void_graph_path = f"/media/data/alex_johannes/octa_data/Cairo/{data_type}_void_graph"
hetero_edges_path = f"/media/data/alex_johannes/octa_data/Cairo/{data_type}_heter_edges"
label_file = "/media/data/alex_johannes/octa_data/Cairo/labels.csv"


train_dataset = hetero_graph_loader.HeteroGraphLoaderTorch(vessel_graph_path,
                                                        void_graph_path,
                                                        hetero_edges_path,
                                                        mode = "train",
                                                        label_file = label_file, 
                                                        line_graph_1 =True, 
                                                        class_dict = octa_dr_dict)

test_dataset = hetero_graph_loader.HeteroGraphLoaderTorch(vessel_graph_path,
                                                        void_graph_path,
                                                        hetero_edges_path,
                                                        mode = "test",
                                                        label_file = label_file, 
                                                        line_graph_1 =True, 
                                                        class_dict = octa_dr_dict)

# remove isolated nodes
for data in train_dataset:
    RemoveIsolatedNodes()(data) 

for data in test_dataset:
    RemoveIsolatedNodes()(data) 

# imputation and normalization
prep.hetero_graph_imputation(train_dataset)
prep.hetero_graph_imputation(test_dataset)

prep.add_global_node(train_dataset)
prep.add_global_node(test_dataset)


#node_mean_tensors, node_std_tensors = prep.hetero_graph_normalization_params(train_dataset)
node_mean_tensors = torch.load(f"checkpoints/{data_type}_node_mean_tensors_global_node.pt")
node_std_tensors = torch.load(f"checkpoints/{data_type}_node_std_tensors_global_node.pt")

prep.hetero_graph_normalization(train_dataset, node_mean_tensors, node_std_tensors)
prep.hetero_graph_normalization(test_dataset, node_mean_tensors, node_std_tensors)

node_nums = []
for data in train_dataset:
    node_nums.append(data.x_dict["graph_1"].shape[0] + data.x_dict["graph_2"].shape[0])

for data in test_dataset:
    node_nums.append(data.x_dict["graph_1"].shape[0] + data.x_dict["graph_2"].shape[0])

max_nodes = np.max(node_nums) # probably need to take a closer look at the adj matrix creation
print(max_nodes)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# class weight extraction
train_labels = [int(data.y[0]) for data in train_dataset]
class_weights = prep.get_class_weights(train_labels, verbose=False)
class_weights = torch.tensor(class_weights, device= device)


num_classes = 3
node_types = ["graph_1", "graph_2"]


agg_mode_dict = {"global_mean_pool": global_mean_pool, "global_max_pool": global_max_pool, "global_add_pool": global_add_pool}



def main():
    torch.autograd.set_detect_anomaly(True)
    run = wandb.init()
    api = wandb.Api()
    sweep = api.sweep("luxtu/graph_pathology/" + sweep_id)
    try: 
        th_swp = sweep.best_run(order='best_mean_auc').summary_metrics['best_mean_auc']
    except KeyError:
        th_swp = 0
    except AttributeError:
        th_swp = 0

    # create the model
    # gnn_models.HeteroGNN
    #global_node_gnn.GNN_global_node
    model = hierarchical_gnn.DiffPool_GNN(hidden_channels= wandb.config.hidden_channels,
                                        out_channels= num_classes,
                                        num_layers= wandb.config.num_layers,
                                        dropout = wandb.config.dropout,
                                        node_types = node_types,
                                        max_nodes = max_nodes
                                        )



    #model = global_node_gnn.GNN_global_node(hidden_channels = wandb.config.hidden_channels, 
    #                          out_channels= num_classes, 
    #                          num_layers= wandb.config.num_layers, 
    #                          dropout = wandb.config.dropout, 
    #                          aggregation_mode= agg_mode_dict[wandb.config.aggregation_mode],
    #                          node_types = node_types,
    #                          )

    # create data loaders for training and test set
    train_loader = DataLoader(train_dataset, batch_size = wandb.config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size = 1, shuffle=False)

    # weigthings for imbalanced classes 
    balanced_loss = torch.nn.CrossEntropyLoss(class_weights)
    unbalanced_loss = torch.nn.CrossEntropyLoss()

    loss_dict = {"balanced": balanced_loss, "unbalanced": unbalanced_loss}

    classifier = graph_classifier.graphClassifierHetero(model, loss_dict[wandb.config.class_weights], lr = wandb.config.lr, weight_decay =wandb.config.weight_decay)

    best_val_bal_acc = 0
    best_mean_auc = 0
    best_pred = None
    for epoch in range(1, wandb.config.epochs + 1):
        loss = classifier.train(train_loader)
        train_acc = classifier.test(train_loader)
        outList, yList = classifier.predict(test_loader)
        y_p = np.array([item.argmax().cpu().detach().numpy() for sublist in outList for item in sublist])
        y_t = np.array([item.detach().cpu().numpy() for sublist in yList for item in sublist])
        #print(f'Epoch: {epoch:03d}, Loss: {loss:.6f}, Train Acc: {train_acc:.4f}, Test Acc: {accuracy_score(y_t, y_p):.4f}, Test Bal Acc: {balanced_accuracy_score(y_t, y_p):.4f}')

        test_acc = accuracy_score(y_t, y_p)
        test_bal_acc = balanced_accuracy_score(y_t, y_p)

        if test_bal_acc > best_val_bal_acc:
            best_val_bal_acc = test_bal_acc

        _, _, roc_auc = evaluation.roc_auc_multiclass(y_t, torch.nn.functional.softmax(torch.tensor(np.array(outList).squeeze()), dim = 1))
        mean_auc = np.mean(list(roc_auc.values()))
                           
        if mean_auc > best_mean_auc or (mean_auc == best_mean_auc and test_bal_acc > best_val_bal_acc):
            best_mean_auc = mean_auc
            best_pred = outList

            if best_mean_auc > th_swp:
                torch.save(model.state_dict(), f'./checkpoints/model_{wandb.config.aggregation_mode}_best_auc.pt')
             

        res_dict =  {"loss": loss, "train_acc": train_acc, "val_acc": test_acc, "val_bal_acc": test_bal_acc, "best_val_bal_acc": best_val_bal_acc,"mean_auc" : mean_auc,"best_mean_auc": best_mean_auc}


        for i in range(num_classes):
            res_dict[f"roc_auc_{label_names[i]}"] = roc_auc[i]

        wandb.log(res_dict)

    wandb.log({"roc": wandb.plot.roc_curve(y_t, torch.nn.functional.softmax(torch.tensor(np.array(best_pred).squeeze()), dim = 1),labels = label_names)})

    #roc_auc = roc_auc_score(y_t, torch.nn.functional.softmax(torch.tensor(np.array(best_pred).squeeze()), dim = 1), multi_class='ovr')

    print(roc_auc)
    for i in range(num_classes):
        wandb.log({f"roc_auc_{label_names[i]}": roc_auc[i]})


wandb.agent(sweep_id, function=main, count=100)
