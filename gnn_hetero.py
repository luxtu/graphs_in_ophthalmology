from loader import hetero_graph_loader
import numpy as np 
import torch

from sklearn.metrics import accuracy_score, balanced_accuracy_score
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from models import graph_classifier, gnn_models
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose, RemoveIsolatedNodes


from utils import prep
from evaluation import evaluation
import wandb



# Define sweep config
sweep_configuration = {
    "method": "bayes",
    "name": "sweep",
    "metric": {"goal": "maximize", "name": "val_bal_acc"},
    "parameters": {
        "batch_size": {"values": [16, 32, 64]},
        "epochs": {"values": [100]},
        "lr": {"max": 0.05, "min": 0.0001}, # learning rate to high does not work
        "weight_decay": {"max": 0.01, "min": 0.00001},
        "hidden_channels": {"values": [32, 64, 128]},
        "dropout": {"values": [0.2, 0.3, 0.4]},
        "num_layers": {"values": [1,2,3]},
        "aggregation_mode": {"values": ["global_mean_pool", "global_max_pool"]},#, "global_add_pool" # add pool does not work
        "class_weights": {"values": ["balanced", "unbalanced"]},
        "dataset": {"values": ["DCP"]} #, "DCP"
    },
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="graph_pathology")


# loading data

data_type = sweep_configuration["parameters"]["dataset"]["values"][0]

octa_dr_dict = {"Healthy": 0, "DM": 0, "PDR": 1, "Early NPDR": 2, "Late NPDR": 2}

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

# imputation and normalization
prep.hetero_graph_imputation(train_dataset)
prep.hetero_graph_imputation(test_dataset)

node_mean_tensors, node_std_tensors = prep.hetero_graph_normalization_params(train_dataset)
prep.hetero_graph_normalization(train_dataset, node_mean_tensors, node_std_tensors)
prep.hetero_graph_normalization(test_dataset, node_mean_tensors, node_std_tensors)

# remove isolated nodes
for data in train_dataset:
    RemoveIsolatedNodes()(data) 

for data in test_dataset:
    RemoveIsolatedNodes()(data) 

# class weight extraction
train_labels = [int(data.y[0]) for data in train_dataset]
class_weights = prep.get_class_weights(train_labels, verbose=False)



num_classes = 3
epochs = 200
node_types = ["graph_1", "graph_2"]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

agg_mode_dict = {"global_mean_pool": global_mean_pool, "global_max_pool": global_max_pool, "global_add_pool": global_add_pool}








def main():
    run = wandb.init()
    # create the model
    model = gnn_models.HeteroGNN(hidden_channels = wandb.config.hidden_channels, 
                              out_channels= num_classes, 
                              num_layers= wandb.config.num_layers, 
                              dropout = wandb.config.dropout, 
                              aggregation_mode= agg_mode_dict[wandb.config.aggregation_mode],
                              node_types = node_types,
                              )

    # create data loaders for training and test set
    train_loader = DataLoader(train_dataset.to(device), batch_size = wandb.config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset.to(device), batch_size = 1, shuffle=False)

    # weigthings for imbalanced classes 
    balanced_loss = torch.nn.CrossEntropyLoss(torch.tensor(class_weights).to(device).float())
    unbalanced_loss = torch.nn.CrossEntropyLoss()

    loss_dict = {"balanced": balanced_loss, "unbalanced": unbalanced_loss}

    classifier = graph_classifier.graphClassifierHetero(model, train_loader, test_loader, loss_dict[wandb.config.class_weights], lr = wandb.config.lr, weight_decay =wandb.config.weight_decay)

    best_val_bal_acc = 0
    for epoch in range(1, wandb.config.epochs + 1):
        loss = classifier.train()
        train_acc = classifier.test(train_loader)
        outList, yList = classifier.predict(test_loader)
        #print(outList)
        y_p = np.array([item.argmax().cpu().detach().numpy() for sublist in outList for item in sublist])
        y_t = np.array([item.detach().cpu().numpy() for sublist in yList for item in sublist])
        #print(f'Epoch: {epoch:03d}, Loss: {loss:.6f}, Train Acc: {train_acc:.4f}, Test Acc: {accuracy_score(y_t, y_p):.4f}, Test Bal Acc: {balanced_accuracy_score(y_t, y_p):.4f}')

        test_acc = accuracy_score(y_t, y_p)
        test_bal_acc = balanced_accuracy_score(y_t, y_p)

        if test_bal_acc > best_val_bal_acc:
            best_val_bal_acc = test_bal_acc
            torch.save(model.state_dict(), f'./model_{wandb.config.aggregation_mode}.pt')
             

        wandb.log({"loss": loss, "train_acc": train_acc, "val_acc": test_acc, "val_bal_acc": test_bal_acc, "best_val_bal_acc": best_val_bal_acc})

        #if balanced_accuracy_score(y_t, y_p) > 0.70:
        #    fig, ax = plt.subplots()
        #    evaluation.plot_confusion_matrix(y_t, y_p, ["Healthy/DM", "PDR", "NPDR"], ax) # , 3, 4
        #    plt.show()
        #    evaluation.plot_roc_curve(y_t, torch.nn.functional.softmax(torch.tensor(np.array(outList).squeeze()), dim = 1), class_labels = ["Healthy/DM", "PDR", "NPDR"]) # , 3, 4
        #    break

wandb.agent(sweep_id, function=main, count=20)