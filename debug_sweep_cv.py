from loader import hetero_graph_loader, hetero_graph_loader_faz
import numpy as np 
import torch

from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, classification_report
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from models import graph_classifier, hetero_gnn, global_node_gnn, hierarchical_gnn, spatial_pooling_gnn
from torch_geometric.loader import DataLoader
from torch_geometric.explain import Explainer,  CaptumExplainer, AttentionExplainer #GNNExplainer
import json
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import time


from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GraphConv
from gnn_explainer import hetero_gnn_explainer

from sklearn.metrics import roc_auc_score
from utils import prep
from explainability import grad_cam, torch_geom_explanation, explanation_in_raw_data
from types import SimpleNamespace
from graph_plotting import graph_2D
from matplotlib import pyplot as plt
import copy
import pandas as pd

torch.cuda.empty_cache()


# Define sweep config
sweep_config_dict = {
        "batch_size": 32, # 48 works well
        "epochs": 50,
        "lr": 0.00832,
        "weight_decay": 0.005261,
        "hidden_channels": 32,
        "dropout":  0.3, # 0.1 works well 
        "num_layers":  1, # many layer are not good
        "aggregation_mode":  "max", # combined
        "pre_layers": 1,
        "post_layers": 2,
        "final_layers": 2, # 2
        "batch_norm": True,
        "hetero_conns": True, # False
        "conv_aggr": "sum", # cat, sum, mean # cat works well
        "faz_node": False,
        "global_node": False,
        "homogeneous_conv": GCNConv, #SAGEConv, # GCNConv 
        "heterogeneous_conv": GraphConv, #GATConv, #SAGEConv, # GATConv # GraphConv
        "class_weights": "balanced",  #weak_balanced
        "dataset": "DCP"} 

sweep_config = SimpleNamespace(**sweep_config_dict)


data_type = sweep_config.dataset

octa_dr_dict = {"Healthy": 0, "DM": 0, "PDR": 2, "Early NPDR": 1, "Late NPDR": 1}
label_names = ["Healthy/DM", "NPDR", "PDR"]


vessel_graph_path = f"../data/{data_type}_vessel_graph"
#label_file = "../data/labels.csv"
label_file = "../data/splits"
#void_graph_path = f"../data/{data_type}_void_graph"
#hetero_edges_path = f"../data/{data_type}_heter_edges"

void_graph_path = f"../data/{data_type}_void_graph_faz_node"
hetero_edges_path = f"../data/{data_type}_heter_edges_faz_node"

faz_node_path = f"../data/{data_type}_faz_nodes"
faz_region_edge_path = f"../data/{data_type}_faz_region_edges"
faz_vessel_edge_path = f"../data/{data_type}_faz_vessel_edges"

#vessel_graph_path = f"../{data_type}_vessel_graph"
#void_graph_path = f"../{data_type}_void_graph"
#hetero_edges_path = f"../{data_type}_heter_edges"
#label_file = "../labels.csv"


mode_cv = "cv"
cv_pickle = f"../data/{data_type}_{mode_cv}_dataset_faz.pkl"
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
                                                        pickle_file = cv_pickle #f"../{data_type}_{mode_train}_dataset_faz.pkl" # f"../{data_type}_{mode_train}_dataset_faz.pkl"
                                                        )


mode_final_test = "final_test"
final_test_pickle = f"../data/{data_type}_{mode_final_test}_dataset_faz.pkl"
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
                                                        pickle_file = final_test_pickle #f"../{data_type}_{mode_train}_dataset_faz.pkl" # f"../{data_type}_{mode_train}_dataset_faz.pkl"
                                                        )


#void_graph_path = f"../data/{data_type}_void_graph"
#hetero_edges_path = f"../data/{data_type}_heter_edges"
#
#
#mode_cv = "cv"
#cv_pickle = f"../data/{data_type}_{mode_cv}_dataset.pkl"
#cv_dataset = hetero_graph_loader.HeteroGraphLoaderTorch(graph_path_1=vessel_graph_path,
#                                                        graph_path_2=void_graph_path,
#                                                        hetero_edges_path_12=hetero_edges_path,
#                                                        mode = mode_cv,
#                                                        label_file = label_file, 
#                                                        line_graph_1 =True, 
#                                                        class_dict = octa_dr_dict,
#                                                        pickle_file = cv_pickle #f"../{data_type}_{mode_train}_dataset_faz.pkl" # f"../{data_type}_{mode_train}_dataset_faz.pkl"
#                                                        )
#
#
#mode_final_test = "final_test"
#final_test_pickle = f"../data/{data_type}_{mode_final_test}_dataset.pkl"
#final_test_dataset = hetero_graph_loader.HeteroGraphLoaderTorch(graph_path_1=vessel_graph_path,
#                                                        graph_path_2=void_graph_path,
#                                                        hetero_edges_path_12=hetero_edges_path,
#                                                        mode = mode_final_test,
#                                                        label_file = label_file, 
#                                                        line_graph_1 =True, 
#                                                        class_dict = octa_dr_dict,
#                                                        pickle_file = final_test_pickle #f"../{data_type}_{mode_train}_dataset_faz.pkl" # f"../{data_type}_{mode_train}_dataset_faz.pkl"
#                                                        )

octa_dr_dict = {"Healthy": 0, "DM": 0, "PDR": 3, "Early NPDR": 1, "Late NPDR": 2}
label_names = ["Healthy/DM", "Early NPDR", "Late NPDR", "PDR"]
# update the label dict

final_test_dataset.update_class(octa_dr_dict)
cv_dataset.update_class(octa_dr_dict)


split = 1
train_dataset, val_dataset, test_dataset = prep.adjust_data_for_split(cv_dataset, final_test_dataset, split, faz = sweep_config.faz_node)


# remove label noise, samples to exclude are stored in label_noise.json
#with open("label_noise.json", "r") as file:
#    label_noise_dict = json.load(file)
#prep.remove_label_noise(train_dataset, label_noise_dict)



with open("label_dict.json", "r") as file:
    label_dict_full = json.load(file)
    #features_label_dict = json.load(file)
features_label_dict = copy.deepcopy(label_dict_full)

eliminate_features = {"graph_1":["num_voxels", "hasNodeAtSampleBorder"], 
                      "graph_2":["centroid_weighted-0", "centroid_weighted-1", "feret_diameter_max", "equivalent_diameter", "orientation"],}
                      #"faz":["centroid_weighted-0", "centroid_weighted-1", "feret_diameter_max", "equivalent_diameter", "orientation"]}
#eliminate_features = {"graph_1":["num_voxels", "maxRadiusAvg", "hasNodeAtSampleBorder", "maxRadiusStd"], 
#                      "graph_2":["centroid_weighted-0", "centroid_weighted-1", "centroid-0", "centroid-1", "feret_diameter_max", "equivalent_diameter", "orientation","solidity"]}

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

print(train_dataset[0])


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# class weight extraction
train_labels = [int(data.y[0]) for data in train_dataset]
class_weights = prep.get_class_weights(train_labels, verbose=False)
class_weights = torch.tensor(class_weights, device= device)

weak_class_weights = class_weights ** 0.5


train_dataset.to(device)
val_dataset.to(device)
test_dataset.to(device)


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
                                                        num_final_lin_layers = sweep_config.final_layers,
                                                        batch_norm = sweep_config.batch_norm,
                                                        conv_aggr = sweep_config.conv_aggr,
                                                        hetero_conns = sweep_config.hetero_conns,
                                                        homogeneous_conv = sweep_config.homogeneous_conv,
                                                        heterogeneous_conv = sweep_config.heterogeneous_conv,
                                                        meta_data = train_dataset[0].metadata(),
                                                        faz_node= sweep_config.faz_node,
                                                        global_node = sweep_config.global_node,
                                                        )

    # xavier initialization
    for m in model.modules():
        if isinstance(m, (torch.nn.Linear)):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
    
    # create data loaders for training and test set
    train_loader = DataLoader(train_dataset, batch_size = sweep_config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size = 64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size = 64, shuffle=False)

    # weigthings for imbalanced classes 
    # torch.nn.CrossEntropyLoss(weight=class_weights) BCEWithLogitsLoss
    balanced_loss = torch.nn.CrossEntropyLoss(class_weights)
    unbalanced_loss = torch.nn.CrossEntropyLoss()
    weak_balanced_loss = torch.nn.CrossEntropyLoss(weak_class_weights)
    #reg_loss = torch.nn.SmoothL1Loss()

    loss_dict = {"balanced": balanced_loss, "unbalanced": unbalanced_loss, "weak_balanced": weak_balanced_loss}

    classifier = graph_classifier.graphClassifierHetero(model, loss_dict[sweep_config.class_weights] , lr = sweep_config.lr, weight_decay =sweep_config.weight_decay, regression=False) #loss_dict[sweep_config.class_weights]

    best_val_bal_acc = 0
    best_mean_auc = 0
    data_loss_dict = {}

    for epoch in range(1, sweep_config.epochs + 1):

        loss, y_prob_train, y_true_train, data_loss_dict = classifier.train(train_loader, data_loss_dict)
        y_prob_val, y_true_val = classifier.predict(val_loader)


        y_pred_val = y_prob_val.argmax(axis=1)


        val_acc = accuracy_score(y_true_val, y_pred_val)
        val_bal_acc = balanced_accuracy_score(y_true_val, y_pred_val)
        kappa = cohen_kappa_score(y_true_val, y_pred_val, weights="quadratic")

        print(f"Epoch: {epoch}, Val Acc: {val_acc:.4f}, Val Bal Acc: {val_bal_acc:.4f}, Kappa: {kappa:.4f}, Loss: {loss:.12f}")

        if val_bal_acc > best_val_bal_acc:
            best_val_bal_acc = val_bal_acc

            #checkpoint the model
            data_type = sweep_config.dataset
            split = 1
            torch.save(classifier.model.state_dict(), f"checkpoints/{data_type}_model_{split}_faz_node.pt")



        if  val_bal_acc > 1.50 or epoch == 199: # kappa > 0.70 or epoch == 15:
            print("start_explain")

            indices = [0, 1, 13, 15]
            selected_samples = [test_dataset[i] for i in indices]

            gnnexp_loader = DataLoader(selected_samples, batch_size = 1, shuffle=False) # 
            for idx, data in enumerate(gnnexp_loader):

                data = data.to(device)
                print(data.y)
                data_label = data.graph_id[0]

                # predict with no grad
                with torch.no_grad():
                    out = model(data.x_dict, data.edge_index_dict, data.batch_dict, pos_dict = data.pos_dict, regression =False)
                    #print(out)
                    #out = torch.nn.functional.softmax(out, dim = 1)
                    print(out)
                    out = out.argmax(dim=1)


                gnnexp_model = copy.deepcopy(model)
                explain_type = "IntegratedGradients"  #IntegratedGradients" #Saliency"

                edges = True
                if edges:
                    edge_mask_type = "object"
                else:
                    edge_mask_type = None

                explainer = Explainer(
                    model=gnnexp_model,
                    algorithm= CaptumExplainer(explain_type), #CaptumExplainer(explain_type),#AttentionExplainer #hetero_gnn_explainer.HeteroGNNExplainer(epochs = 200)
                    explanation_type='phenomenon',
                    node_mask_type='attributes', # common attributes works well
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

                explanation1 = explainer(data.x_dict, data.edge_index_dict,target =target,  batch_dict = data.batch_dict,  grads = False)#, index= 0 # # pos_dict = pos_dict,
                # for GNN Explainer forces sparsity, thres
                torch_geom_explanation.visualize_feature_importance(explanation1, f'explain_out/feature_importance_{data_label}_{explain_type}_{epoch}_attributes_expnum_1.png', features_label_dict, top_k = 50)
                torch_geom_explanation.visualize_relevant_subgraph(explanation1, data, f"explain_out/subgraph_{data_label}_{explain_type}_{epoch}_attributes_expnum_1.png", threshold = "adaptive", edge_threshold = "adaptive", edges = edges, faz_node= sweep_config.faz_node) #"adaptive"
                graph_2D.HeteroGraphPlotter2D().plot_graph_2D_faz(data, edges= True, path = f"explain_out/fullgraph_{data_label}_{explain_type}.png")
                torch_geom_explanation.visualize_node_importance_histogram(explanation1, f"explain_out/node_hist_{data_label}_{explain_type}_{epoch}_attributes_expnum_1.png")


                data_type = "DCP"

                segmentation_path = f"/media/data/alex_johannes/octa_data/Cairo/{data_type}_seg"
                json_path = f"/media/data/alex_johannes/octa_data/Cairo/{data_type}_json"
                image_path = f"/media/data/alex_johannes/octa_data/Cairo/{data_type}_images"

                raw_data_explainer = explanation_in_raw_data.RawDataExplainer(raw_image_path= image_path, segmentation_path= segmentation_path, vvg_path= json_path)
                raw_data_explainer.create_explanation_image(explanation1,data, data_label, f"explain_out/overlay_{data_label}_{epoch}.png", faz_node= sweep_config.faz_node, threshold = "adaptive", edge_threshold = "adaptive")


                graph_ax = raw_data_explainer.create_explanation_image(explanation1,data,data_label, path=None, faz_node= sweep_config.faz_node, threshold = "adaptive", edge_threshold = "adaptive")
                torch_geom_explanation.visualize_relevant_subgraph(explanation1, data, f"explain_out/overlay_subgraph_{data_label}_{epoch}.png", 
                                                                        threshold = "adaptive",
                                                                        edge_threshold = "adaptive",
                                                                        edges = edges, 
                                                                        faz_node= sweep_config.faz_node,
                                                                        ax=graph_ax) #"adaptive"

                fig, ax = plt.subplots()
                # add titles for each subplot in the figure

                # add legend to the figure that assigns the square marker ("s") to the intercapillary region and the circle marker to the vessel region    
                if sweep_config.faz_node:
                    legend_items = [
                        plt.Line2D([0], [0], marker='s', color='red', alpha = 0.8, label='ICP Region', markerfacecolor='red', markersize=6, linestyle='None'),
                        plt.Line2D([0], [0], marker='o', color='red', alpha = 0.8, label='Vessel', markerfacecolor='red', markersize=6, linestyle='None'),
                        plt.Line2D([0], [0], marker="D", color='red', alpha = 0.8, label='Fovea', markerfacecolor='blue', markersize=6, linestyle='None'),
                    ]
                else:
                    legend_items = [
                        plt.Line2D([0], [0], marker='s', color='red', alpha = 0.8, label='ICP Region', markerfacecolor='red', markersize=6, linestyle='None'),
                        plt.Line2D([0], [0], marker='o', color='red', alpha = 0.8, label='Vessel', markerfacecolor='red', markersize=6, linestyle='None'),
                    ]

                plotter2d = graph_2D.HeteroGraphPlotter2D()
                #plotter2d.set_val_range(1, 0)

                importance_dict = {}
                for key in explanation1.x_dict.keys():
                    importance_dict[key] = explanation1.x_dict[key].abs().sum(dim=-1) 

                if sweep_config.faz_node:
                    plotter2d.plot_graph_2D_faz(data ,edges= False, pred_val_dict = importance_dict, ax = ax)
                else:
                    plotter2d.plot_graph_2D(data ,edges= False, pred_val_dict = importance_dict, ax = ax)

                fig.legend()
                plt.tight_layout()
                plt.savefig(f'explain_out/integrate_gradients_{data_label}_plot_{epoch}_final.png' ) 
                plt.close()


            break

    # load best model
    data_type = "DCP"
    split = 1
    classifier.model.load_state_dict(torch.load(f"checkpoints/{data_type}_model_{split}_faz_node.pt"))
    #model.load_state_dict(torch.load(f"model_checkpoints/{data_type}_model_{split}_faz_node.pt"))
    #model.eval()

    # evaluate on test set
    y_prob_test, y_true_test = classifier.predict(test_loader)
    print(classification_report(y_true_test, y_prob_test.argmax(axis=1), target_names=label_names))

main()

