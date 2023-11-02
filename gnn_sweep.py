from loader import hetero_graph_loader
import numpy as np 
import torch

from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from models import graph_classifier, hetero_gnn, global_node_gnn, hierarchical_gnn, spatial_pooling_gnn
from torch_geometric.loader import DataLoader

from sklearn.metrics import roc_auc_score, roc_curve, auc
from utils import prep
from evaluation import evaluation
import wandb
from sklearn.preprocessing import LabelBinarizer




# Define sweep config
sweep_configuration = {
    "method": "random",
    "name": "HGN",
    "metric": {"goal": "maximize", "name": "best_mean_auc"},
    "parameters": {
        "batch_size": {"values": [8, 16, 32, 64]},
        "epochs": {"values": [100]},
        "lr": {"max": 0.05, "min": 0.005}, # learning rate to high does not work
        "weight_decay": {"max": 0.01, "min": 0.00001},
        "hidden_channels": {"values": [32, 64]}, #[64, 128]
        "dropout": {"values": [0.1, 0.3, 0.4]}, # 0.2,  more droput looks better
        "num_layers": {"values": [1,2,5]},
        "aggregation_mode": {"values": ["global_mean_pool", "global_add_pool", "combined"]},#, "global_mean_pool",  "global_add_pool" # add pool does not work
        "pre_layers": {"values": [1,2,4]},
        "post_layers": {"values": [1,2,4]},
        "batch_norm": {"values": [True, False]},
        "hetero_conns": {"values": [True, False]},
        "conv_aggr": {"values": ["cat", "sum", "mean"]}, # cat, sum, mean
        "class_weights": {"values": ["unbalanced", "balanced"]}, # "balanced", 
        "dataset": {"values": ["DCP"]}, #, "DCP"
        "regression": {"values": [False]},
    },
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="graph_pathology")
# loading data

data_type = sweep_configuration["parameters"]["dataset"]["values"][0]

octa_dr_dict = {"Healthy": 0, "DM": 0, "PDR": 2, "Early NPDR": 1, "Late NPDR": 1}
label_names = ["Healthy/DM", "NPDR", "PDR"]

#vessel_graph_path = f"/media/data/alex_johannes/octa_data/Cairo/{data_type}_vessel_graph"
#void_graph_path = f"/media/data/alex_johannes/octa_data/Cairo/{data_type}_void_graph"
#hetero_edges_path = f"/media/data/alex_johannes/octa_data/Cairo/{data_type}_heter_edges"
#label_file = "/media/data/alex_johannes/octa_data/Cairo/labels.csv"

vessel_graph_path = f"../{data_type}_vessel_graph"
void_graph_path = f"../{data_type}_void_graph"
hetero_edges_path = f"../{data_type}_heter_edges"
label_file = "../labels.csv"



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

prep.add_node_features(train_dataset, ["graph_1", "graph_2"])
prep.add_node_features(test_dataset, ["graph_1", "graph_2"])

prep.add_global_node(train_dataset)
prep.add_global_node(test_dataset)

node_mean_tensors, node_std_tensors = prep.hetero_graph_normalization_params(train_dataset)

# save the normalization parameters
#torch.save(node_mean_tensors, f"checkpoints/{data_type}_node_mean_tensors_global_node_node_degs.pt")
#torch.save(node_std_tensors, f"checkpoints/{data_type}_node_std_tensors_global_node_node_degs.pt")

#node_mean_tensors, node_std_tensors = prep.hetero_graph_normalization_params(train_dataset)
node_mean_tensors = torch.load(f"../{data_type}_node_mean_tensors_global_node_node_degs.pt")
node_std_tensors = torch.load(f"../{data_type}_node_std_tensors_global_node_node_degs.pt")

#node_mean_tensors = torch.load(f"checkpoints/{data_type}_node_mean_tensors_global_node_node_degs.pt")
#node_std_tensors = torch.load(f"checkpoints/{data_type}_node_std_tensors_global_node_node_degs.pt")

prep.hetero_graph_normalization(train_dataset, node_mean_tensors, node_std_tensors)
prep.hetero_graph_normalization(test_dataset, node_mean_tensors, node_std_tensors)

max_nodes = np.max([graph.num_nodes for graph in train_dataset] + [graph.num_nodes for graph in test_dataset])
print(max_nodes)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# class weight extraction
train_labels = [int(data.y[0]) for data in train_dataset]
class_weights = prep.get_class_weights(train_labels, verbose=False)
class_weights = torch.tensor(class_weights, device=device)
class_weights = class_weights ** 0.5 

num_classes = class_weights.shape[0]
node_types = ["graph_1", "graph_2"]

agg_mode_dict = {"global_mean_pool": global_mean_pool, "global_max_pool": global_max_pool, "global_add_pool": global_add_pool, "combined": [global_mean_pool, global_add_pool]}



def main():
    #torch.autograd.set_detect_anomaly(True)
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
    #model = hierarchical_gnn.DiffPool_GNN(hidden_channels= wandb.config.hidden_channels,
    #                                    out_channels= num_classes,
    #                                    num_layers= wandb.config.num_layers,
    #                                    dropout = wandb.config.dropout,
    #                                    node_types = node_types,
    #                                    max_nodes = max_nodes
    #                                    )

    model = global_node_gnn.GNN_global_node(hidden_channels= wandb.config.hidden_channels,
                                                        out_channels= num_classes,
                                                        num_layers= wandb.config.num_layers, 
                                                        dropout = wandb.config.dropout, 
                                                        aggregation_mode= agg_mode_dict[sweep_config.aggregation_mode], 
                                                        node_types = node_types,
                                                        num_pre_processing_layers = wandb.config.pre_layers,
                                                        num_post_processing_layers = wandb.config.post_layers,
                                                        batch_norm = wandb.config.batch_norm,
                                                        conv_aggr = wandb.config.conv_aggr,
                                                        hetero_conns = wandb.config.hetero_conns,
                                                        meta_data = train_dataset[0].metadata())


    #model = spatial_pooling_gnn.GeomPool_GNN(hidden_channels = wandb.config.hidden_channels, 
    #                          out_channels= num_classes, 
    #                          num_layers= wandb.config.num_layers, 
    #                          dropout = wandb.config.dropout, 
    #                          aggregation_mode= agg_mode_dict[wandb.config.aggregation_mode],
    #                          node_types = node_types,
    #                          meta_data= train_dataset[0].metadata(),
    #                          )

    # create data loaders for training and test set
    train_loader = DataLoader(train_dataset, batch_size = wandb.config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size = 1, shuffle=False)

    # weigthings for imbalanced classes 
    balanced_loss = torch.nn.CrossEntropyLoss(class_weights)
    unbalanced_loss = torch.nn.CrossEntropyLoss()

    loss_dict = {"balanced": balanced_loss, "unbalanced": unbalanced_loss}

    if wandb.config.regression:
        reg_loss = torch.nn.SmoothL1Loss()
        classifier = graph_classifier.graphClassifierHetero(model,reg_loss , lr = wandb.config.lr, weight_decay =wandb.config.weight_decay, regression=wandb.config.regression)
    else:
        classifier = graph_classifier.graphClassifierHetero(model, loss_dict[wandb.config.class_weights], lr = wandb.config.lr, weight_decay =wandb.config.weight_decay, regression=wandb.config.regression) 

    best_val_bal_acc = 0
    best_mean_auc = 0
    best_pred = None
    for epoch in range(1, wandb.config.epochs + 1):
        loss = classifier.train(train_loader)
        train_acc = classifier.test(train_loader)
        outList, yList = classifier.predict(test_loader)

        if wandb.config.regression:
            y_p = np.array(outList).squeeze() #y_p = np.array([item.cpu().detach().numpy() for sublist in outList for item in sublist])
        else:
            y_p = np.array(outList).argmax(axis=1)# y_p = np.array([item.argmax().cpu().detach().numpy() for sublist in outList for item in sublist])
        
        y_t = np.array([item.detach().cpu().numpy() for sublist in yList for item in sublist])
        #print(f'Epoch: {epoch:03d}, Loss: {loss:.6f}, Train Acc: {train_acc:.4f}, Test Acc: {accuracy_score(y_t, y_p):.4f}, Test Bal Acc: {balanced_accuracy_score(y_t, y_p):.4f}')

        test_acc = accuracy_score(y_t, y_p)
        test_bal_acc = balanced_accuracy_score(y_t, y_p)

        #print(test_acc, test_bal_acc, loss)

        res_dict =  {"loss": loss, "train_acc": train_acc, "val_acc": test_acc, "val_bal_acc": test_bal_acc}

        if not wandb.config.regression:
            y_p_softmax = torch.nn.functional.softmax(torch.tensor(np.array(outList).squeeze()), dim = 1).detach().numpy()
            if num_classes == 2:
                mean_auc = roc_auc_score(
                    y_true=y_t,
                    y_score = y_p_softmax[:,1],
                    multi_class="ovr",
                    average="macro",
                )
            else:
                mean_auc = roc_auc_score(
                    y_true=y_t,
                    y_score = y_p_softmax,
                    multi_class="ovr",
                    average="macro",
                )
                label_binarizer = LabelBinarizer().fit(y_t)
                y_onehot_test = label_binarizer.transform(y_t)
                # get auc for each class
                fpr = dict()
                tpr = dict()
                roc_auc = dict()
                for i in range(num_classes):
                    fpr[i], tpr[i], _ = roc_curve(y_onehot_test[:, i], y_p_softmax[:, i])
                    roc_auc[i] = auc(fpr[i], tpr[i])
                    res_dict[f"roc_auc_{label_names[i]}"] = roc_auc[i]


            if mean_auc > best_mean_auc or (mean_auc == best_mean_auc and test_bal_acc > best_val_bal_acc):
                best_mean_auc = mean_auc
                best_pred = y_p_softmax

            #if best_mean_auc > th_swp:
            #    torch.save(model.state_dict(), f'./checkpoints/model_{wandb.config.aggregation_mode}_best_auc.pt')

            res_dict["mean_auc"] = mean_auc
            res_dict["best_mean_auc"] = best_mean_auc

        if test_bal_acc > best_val_bal_acc:
            best_val_bal_acc = test_bal_acc

        kappa = cohen_kappa_score(y_t, y_p, weights= "quadratic")



        res_dict["kappa"] = kappa
        res_dict["best_val_bal_acc"]= best_val_bal_acc

        wandb.log(res_dict)

    if not wandb.config.regression:
        wandb.log({"roc": wandb.plot.roc_curve(y_t, best_pred,labels = label_names)})




wandb.agent(sweep_id, function=main, count=100)
