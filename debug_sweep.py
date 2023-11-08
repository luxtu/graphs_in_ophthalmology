from loader import hetero_graph_loader, hetero_graph_loader_faz
import numpy as np 
import torch

from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from models import graph_classifier, hetero_gnn, global_node_gnn, hierarchical_gnn, spatial_pooling_gnn
from torch_geometric.loader import DataLoader
from torch_geometric.explain import Explainer, GNNExplainer, CaptumExplainer, AttentionExplainer
import json
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D



from sklearn.metrics import roc_auc_score
from utils import prep
from explainability import grad_cam, torch_geom_explanation
from types import SimpleNamespace
from graph_plotting import graph_2D
from matplotlib import pyplot as plt
import copy
import pandas as pd

torch.cuda.empty_cache()


# Define sweep config
sweep_config_dict = {
        "batch_size": 48,
        "epochs": 100,
        "lr": 0.00232,
        "weight_decay": 0.008261,
        "hidden_channels": 32,
        "dropout":  0.1, 
        "num_layers":  5,
        "aggregation_mode":  "max", # combined
        "pre_layers": 3,
        "post_layers": 3,
        "batch_norm": False,
        "hetero_conns": True, # False
        "conv_aggr": "cat", # cat, sum, mean
        "faz_node": True,
        "class_weights": "balanced", 
        "dataset": "DCP"} 

sweep_config = SimpleNamespace(**sweep_config_dict)


data_type = sweep_config.dataset

octa_dr_dict = {"Healthy": 0, "DM": 0, "PDR": 2, "Early NPDR": 1, "Late NPDR": 1}
label_names = ["Healthy/DM", "NPDR", "PDR"]

vessel_graph_path = f"/media/data/alex_johannes/octa_data/Cairo/{data_type}_vessel_graph"
label_file = "/media/data/alex_johannes/octa_data/Cairo/labels.csv"
#void_graph_path = f"/media/data/alex_johannes/octa_data/Cairo/{data_type}_void_graph"
#hetero_edges_path = f"/media/data/alex_johannes/octa_data/Cairo/{data_type}_heter_edges"

void_graph_path = f"/media/data/alex_johannes/octa_data/Cairo/{data_type}_void_graph_faz_node"
hetero_edges_path = f"/media/data/alex_johannes/octa_data/Cairo/{data_type}_heter_edges_faz_node"

faz_node_path = f"/media/data/alex_johannes/octa_data/Cairo/{data_type}_faz_nodes"
faz_region_edge_path = f"/media/data/alex_johannes/octa_data/Cairo/{data_type}_faz_region_edges"
faz_vessel_edge_path = f"/media/data/alex_johannes/octa_data/Cairo/{data_type}_faz_vessel_edges"

#vessel_graph_path = f"../{data_type}_vessel_graph"
#void_graph_path = f"../{data_type}_void_graph"
#hetero_edges_path = f"../{data_type}_heter_edges"
#label_file = "../labels.csv"


mode_train = "train"
train_pickle = f"/media/data/alex_johannes/octa_data/Cairo/{data_type}_{mode_train}_dataset_faz.pkl"
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
mode_test = "test"
test_pickle = f"/media/data/alex_johannes/octa_data/Cairo/{data_type}_{mode_test}_dataset_faz.pkl"
test_dataset = hetero_graph_loader_faz.HeteroGraphLoaderTorch(graph_path_1=vessel_graph_path,
                                                        graph_path_2=void_graph_path,
                                                        graph_path_3=faz_node_path,
                                                        hetero_edges_path_12=hetero_edges_path,
                                                        hetero_edges_path_13=faz_vessel_edge_path,
                                                        hetero_edges_path_23=faz_region_edge_path,
                                                        mode = mode_test,
                                                        label_file = label_file, 
                                                        line_graph_1 =True, 
                                                        class_dict = octa_dr_dict,
                                                        pickle_file = test_pickle # f"../{data_type}_{mode_test}_dataset_faz.pkl" #f"../{data_type}_{mode_test}_dataset_faz.pkl")
                                                        )



#mode_train = "train"
#train_pickle = f"/media/data/alex_johannes/octa_data/Cairo/{data_type}_{mode_train}_dataset.pkl"
#train_dataset = hetero_graph_loader.HeteroGraphLoaderTorch(vessel_graph_path,
#                                                        void_graph_path,
#                                                        hetero_edges_path,
#                                                        mode = mode_train,
#                                                        label_file = label_file, 
#                                                        line_graph_1 =True, 
#                                                        class_dict = octa_dr_dict,
#                                                        pickle_file = train_pickle
#                                                        )
#
#mode_test = "test"
#test_pickle = f"/media/data/alex_johannes/octa_data/Cairo/{data_type}_{mode_test}_dataset.pkl"
#test_dataset = hetero_graph_loader.HeteroGraphLoaderTorch(vessel_graph_path,
#                                                        void_graph_path,
#                                                        hetero_edges_path,
#                                                        mode = mode_test,
#                                                        label_file = label_file, 
#                                                        line_graph_1 =True, 
#                                                        class_dict = octa_dr_dict,
#                                                        pickle_file = test_pickle
#                                                        )


# imputation and normalization
prep.hetero_graph_imputation(train_dataset)
prep.hetero_graph_imputation(test_dataset)

prep.add_node_features(train_dataset, ["graph_1", "graph_2"])
prep.add_node_features(test_dataset, ["graph_1", "graph_2"])

#prep.add_global_node(train_dataset)
#prep.add_global_node(test_dataset)


node_mean_tensors, node_std_tensors = prep.hetero_graph_normalization_params(train_dataset)
#node_mean_tensors = torch.load(f"../{data_type}_node_mean_tensors_global_node_node_degs.pt")
#node_std_tensors = torch.load(f"../{data_type}_node_std_tensors_global_node_node_degs.pt")

#print(node_mean_tensors)
#print(node_std_tensors)

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
class_weights = torch.tensor(class_weights, device= device)

class_weights = class_weights ** 0.5




num_classes = class_weights.shape[0]
node_types = ["graph_1", "graph_2"]


agg_mode_dict = {"mean": global_mean_pool, "max": global_max_pool, "add": global_add_pool, "combined_mean_add": [global_mean_pool, global_add_pool], "combined_mean_max":[global_mean_pool, global_max_pool], "combined_max_add":[global_max_pool, global_add_pool]}



def main():
    torch.autograd.set_detect_anomaly(True)

    model = global_node_gnn.GNN_global_node(hidden_channels= sweep_config.hidden_channels,
                                                         out_channels= num_classes, # irrelevant
                                                         num_layers= sweep_config.num_layers, 
                                                         dropout = sweep_config.dropout, 
                                                         aggregation_mode= agg_mode_dict[sweep_config.aggregation_mode], 
                                                         node_types = node_types,
                                                         num_pre_processing_layers = sweep_config.pre_layers,
                                                         num_post_processing_layers = sweep_config.post_layers,
                                                         batch_norm = sweep_config.batch_norm,
                                                         conv_aggr = sweep_config.conv_aggr,
                                                         hetero_conns = sweep_config.hetero_conns,
                                                         meta_data = train_dataset[0].metadata(),
                                                         faz_node= sweep_config.faz_node,)
    
    # create data loaders for training and test set
    train_loader = DataLoader(train_dataset, batch_size = sweep_config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size = 64, shuffle=False)

    # weigthings for imbalanced classes 
    balanced_loss = torch.nn.CrossEntropyLoss(class_weights)
    unbalanced_loss = torch.nn.CrossEntropyLoss()
    #reg_loss = torch.nn.SmoothL1Loss()

    loss_dict = {"balanced": balanced_loss, "unbalanced": unbalanced_loss}

    classifier = graph_classifier.graphClassifierHetero(model, loss_dict[sweep_config.class_weights] , lr = sweep_config.lr, weight_decay =sweep_config.weight_decay, regression=False) #loss_dict[sweep_config.class_weights]

    best_val_bal_acc = 0
    best_mean_auc = 0

    for epoch in range(1, sweep_config.epochs + 1):
        loss = classifier.train(train_loader)
        print(f"Epoch: {epoch:03d}")

        outList, y_t = classifier.predict(test_loader)

        reg = False
        if not reg:
            y_p = np.array(outList).argmax(axis=1)
        else:
            y_p = np.array(outList).squeeze()

        test_acc = accuracy_score(y_t, y_p)
        test_bal_acc = balanced_accuracy_score(y_t, y_p)
        kappa = cohen_kappa_score(y_t, y_p, weights="quadratic")

        print(f"Test Acc: {test_acc:.4f}, Test Bal Acc: {test_bal_acc:.4f}, Kappa: {kappa:.4f}, Loss: {loss:.12f}")

        if test_bal_acc > best_val_bal_acc:
            best_val_bal_acc = test_bal_acc


        with open("label_dict.json", "r") as file:
            features_label_dict = json.load(file)
        

        if kappa > 0.80:
            print("start_explain")

            indices = [0, 1, 13, 15]
            selected_samples = [test_dataset[i] for i in indices]

            gnnexp_loader = DataLoader(selected_samples, batch_size = 1, shuffle=False) # 
            print(y_t)
            print(y_p)
            for idx, data in enumerate(gnnexp_loader):
                cam_cls = [0,1,2]
                # get next data from loader
                #data = next(iter(cam_loader))
                data = data.to(device)
                print(data.y)

                # predict with no grad
                with torch.no_grad():
                    out = model(data.x_dict, data.edge_index_dict, data.batch_dict, pos_dict = data.pos_dict, regression =False)
                    #print(out)
                    #out = torch.nn.functional.softmax(out, dim = 1)
                    print(out)
                    out = out.argmax(dim=1)

                cam_model = copy.deepcopy(model)
                cam_cls = [0,1,2] # god knows why
                cam = grad_cam.grad_cam_data(cam_model, data, cam_cls, scale = False, relu = True, start = False, abs=True)
                fig, ax = plt.subplots(1,3, figsize=(15,5))
                # add titles for each subplot in the figure
                titles = ["Healthy/DM", "NPDR", "PDR"]
                for i, ax_iter in enumerate(ax):
                    ax_iter.set_title(titles[i])

                # add legend to the figure that assigns the square marker ("s") to the intercapillary region and the circle marker to the vessel region    
                legend_items = [
                    plt.Line2D([0], [0], marker='s', color='red', alpha = 0.8, label='ICP Region', markerfacecolor='red', markersize=6, linestyle='None'),
                    plt.Line2D([0], [0], marker='o', color='red', alpha = 0.8, label='Vessel', markerfacecolor='red', markersize=6, linestyle='None'),
                ]

                # Add legend to the plot
                ax[2].legend(handles=legend_items, loc='upper right')


                for ax_idx, cam_res in enumerate(cam):
                    plotter2d = graph_2D.HeteroGraphPlotter2D()
                    #plotter2d.set_val_range(1, 0)
                    plotter2d.plot_graph_2D(data ,edges= False, pred_val_dict = cam_res, ax = ax[ax_idx])

                fig.legend()
                plt.tight_layout()
                plt.savefig(f'explain_out/gradCAM_{idx}_plot_{epoch}_final_fordel_faz.png' ) 
                plt.close()


                #data = next(iter(gnnexp_loader))
                #data = data.to(device)
                gnnexp_model = copy.deepcopy(model)
                explain_type = "IntegratedGradients" #Saliency"

                edges = True
                if edges:
                    edge_mask_type = "object"
                else:
                    edge_mask_type = None

                explainer = Explainer(
                    model=gnnexp_model,
                    algorithm= CaptumExplainer(explain_type), #CaptumExplainer("IntegratedGradients"),#AttentionExplainer
                    explanation_type='phenomenon',
                    node_mask_type='attributes',
                    edge_mask_type=edge_mask_type,
                    model_config=dict(
                        mode='multiclass_classification',
                        task_level='graph',
                        return_type='raw',
                    ),
                )
                prediction = explainer.get_prediction(data.x_dict, data.edge_index_dict, batch_dict = data.batch_dict)
                target = explainer.get_target(prediction)
                print(target)
                #pos_dict = {}
                #for key in ["graph_1", "graph_2"]:
                #    pos_dict[key] = data[key].pos.requires_grad_(False)

                explanation = explainer(data.x_dict, data.edge_index_dict,target =target,  batch_dict = data.batch_dict,  grads = False)#, index= 0 # # pos_dict = pos_dict,
                #print(explanation)

                torch_geom_explanation.visualize_feature_importance(explanation, f'explain_out/feature_importance_{idx}_{explain_type}_{epoch}_fordel_faz.png', features_label_dict, top_k = 50)
                torch_geom_explanation.visualize_relevant_subgraph(explanation, data, f"explain_out/subgraph_{idx}_{explain_type}_{epoch}_fordel_faz.png", threshold = "adaptive", edge_threshold = "adaptive", edges = edges)

                #if epoch == 1:
                #    graph_2D.HeteroGraphPlotter2D().plot_graph_2D(data, edges= True, path = f"fullbgraph_{idx}_{explain_type}.png")

            break
main()

