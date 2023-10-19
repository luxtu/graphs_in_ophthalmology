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
from types import SimpleNamespace

torch.cuda.empty_cache()

# Create a SimpleNamespace object from a dictionary
my_dict = {'key1': 42, 'key2': 'value', 'key3': [1, 2, 3]}
my_obj = SimpleNamespace(**my_dict)


# Define sweep config
sweep_config_dict = {
        "batch_size": 4,
        "epochs": 10,
        "lr": 0.005,
        "weight_decay": 0.001,
        "hidden_channels": 32,
        "dropout":  0.1, 
        "num_layers":  2,
        "aggregation_mode":  global_mean_pool,
        "class_weights": "unbalanced", 
        "dataset": "DCP"} 

sweep_config = SimpleNamespace(**sweep_config_dict)


data_type = sweep_config.dataset

octa_dr_dict = {"Healthy": 0, "DM": 0, "PDR": 1, "Early NPDR": 2, "Late NPDR": 2}
label_names = ["Healthy/DM", "PDR", "NPDR"]

vessel_graph_path = f"/media/data/alex_johannes/octa_data/Cairo/{data_type}_vessel_graph"
void_graph_path = f"/media/data/alex_johannes/octa_data/Cairo/{data_type}_void_graph"
hetero_edges_path = f"/media/data/alex_johannes/octa_data/Cairo/{data_type}_heter_edges"
label_file = "/media/data/alex_johannes/octa_data/Cairo/labels.csv"


train_dataset = hetero_graph_loader.HeteroGraphLoaderTorch(vessel_graph_path,
                                                        void_graph_path,
                                                        hetero_edges_path,
                                                        mode = "debug",
                                                        label_file = label_file, 
                                                        line_graph_1 =True, 
                                                        class_dict = octa_dr_dict)

test_dataset = hetero_graph_loader.HeteroGraphLoaderTorch(vessel_graph_path,
                                                        void_graph_path,
                                                        hetero_edges_path,
                                                        mode = "debug",
                                                        label_file = label_file, 
                                                        line_graph_1 =True, 
                                                        class_dict = octa_dr_dict)
print(len(train_dataset))

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

max_nodes = np.max(node_nums)
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

    model = hierarchical_gnn.DiffPool_GNN(hidden_channels= sweep_config.hidden_channels,
                                        out_channels= num_classes,
                                        num_layers= sweep_config.num_layers,
                                        dropout = sweep_config.dropout,
                                        node_types = node_types,
                                        max_nodes = max_nodes
                                        )

    # create data loaders for training and test set
    train_loader = DataLoader(train_dataset, batch_size = sweep_config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size = 1, shuffle=False)

    # weigthings for imbalanced classes 
    balanced_loss = torch.nn.CrossEntropyLoss(class_weights)
    unbalanced_loss = torch.nn.CrossEntropyLoss()

    loss_dict = {"balanced": balanced_loss, "unbalanced": unbalanced_loss}

    classifier = graph_classifier.graphClassifierHetero(model, loss_dict[sweep_config.class_weights], lr = sweep_config.lr, weight_decay =sweep_config.weight_decay)

    best_val_bal_acc = 0
    best_mean_auc = 0

    for epoch in range(1, sweep_config.epochs + 1):
        loss = classifier.train(train_loader)
        print(f"Epoch: {epoch:03d}")
        print(torch.cuda.memory_summary())
        train_acc = classifier.test(train_loader)
        outList, yList = classifier.predict(test_loader)
        #print(outList)
        y_p = np.array([item.argmax().cpu().detach().numpy() for sublist in outList for item in sublist])
        y_t = np.array([item.detach().cpu().numpy() for sublist in yList for item in sublist])
        #print(f'Epoch: {epoch:03d}, Loss: {loss:.6f}, Train Acc: {train_acc:.4f}, Test Acc: {accuracy_score(y_t, y_p):.4f}, Test Bal Acc: {balanced_accuracy_score(y_t, y_p):.4f}')

        test_acc = accuracy_score(y_t, y_p)
        test_bal_acc = balanced_accuracy_score(y_t, y_p)

        print(test_acc, test_bal_acc, loss)

        if test_bal_acc > best_val_bal_acc:
            best_val_bal_acc = test_bal_acc

        _, _, roc_auc = evaluation.roc_auc_multiclass(y_t, torch.nn.functional.softmax(torch.tensor(np.array(outList).squeeze()), dim = 1))
        mean_auc = np.mean(list(roc_auc.values()))
                           
        if mean_auc > best_mean_auc or (mean_auc == best_mean_auc and test_bal_acc > best_val_bal_acc):
            best_mean_auc = mean_auc


        res_dict =  {"loss": loss, "train_acc": train_acc, "val_acc": test_acc, "val_bal_acc": test_bal_acc, "best_val_bal_acc": best_val_bal_acc,"mean_auc" : mean_auc,"best_mean_auc": best_mean_auc}
        for i in range(num_classes):
            res_dict[f"roc_auc_{label_names[i]}"] = roc_auc[i]


    print(roc_auc)

main()