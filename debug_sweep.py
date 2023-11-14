from loader import hetero_graph_loader, hetero_graph_loader_faz
import numpy as np 
import torch

from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from models import graph_classifier, hetero_gnn, global_node_gnn, hierarchical_gnn, spatial_pooling_gnn
from torch_geometric.loader import DataLoader
from torch_geometric.explain import Explainer,  CaptumExplainer, AttentionExplainer #GNNExplainer
import json
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import time


from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from gnn_explainer import hetero_gnn_explainer

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
        "batch_size": 16, # 48 works well
        "epochs": 100,
        "lr": 0.0232,
        "weight_decay": 0.008261,
        "hidden_channels": 32,
        "dropout":  0.2, # 0.1 works well 
        "num_layers":  3,
        "aggregation_mode":  "max", # combined
        "pre_layers": 2,
        "post_layers": 2,
        "batch_norm": False,
        "hetero_conns": True, # False
        "conv_aggr": "mean", # cat, sum, mean # cat works well
        "faz_node": False,
        "global_node": False,
        "homogeneous_conv": SAGEConv,
        "heterogeneous_conv": SAGEConv,
        "class_weights": "balanced",  
        "dataset": "DCP"} 

sweep_config = SimpleNamespace(**sweep_config_dict)


data_type = sweep_config.dataset

#octa_dr_dict = {"Healthy": 0, "DM": 0, "PDR": 2, "Early NPDR": 1, "Late NPDR": 1}
#label_names = ["Healthy/DM", "NPDR", "PDR"]

octa_dr_dict = {"Healthy": 0, "DM": 0, "PDR": 3, "Early NPDR": 1, "Late NPDR": 2}
label_names = ["Healthy/DM", "Early NPDR","Late NPDR", "PDR"]

vessel_graph_path = f"/media/data/alex_johannes/octa_data/Cairo/{data_type}_vessel_graph"
label_file = "/media/data/alex_johannes/octa_data/Cairo/labels.csv"
void_graph_path = f"/media/data/alex_johannes/octa_data/Cairo/{data_type}_void_graph"
hetero_edges_path = f"/media/data/alex_johannes/octa_data/Cairo/{data_type}_heter_edges"

#void_graph_path = f"/media/data/alex_johannes/octa_data/Cairo/{data_type}_void_graph_faz_node"
#hetero_edges_path = f"/media/data/alex_johannes/octa_data/Cairo/{data_type}_heter_edges_faz_node"

#faz_node_path = f"/media/data/alex_johannes/octa_data/Cairo/{data_type}_faz_nodes"
#faz_region_edge_path = f"/media/data/alex_johannes/octa_data/Cairo/{data_type}_faz_region_edges"
#faz_vessel_edge_path = f"/media/data/alex_johannes/octa_data/Cairo/{data_type}_faz_vessel_edges"

#vessel_graph_path = f"../{data_type}_vessel_graph"
#void_graph_path = f"../{data_type}_void_graph"
#hetero_edges_path = f"../{data_type}_heter_edges"
#label_file = "../labels.csv"


#mode_train = "train"
#train_pickle = f"/media/data/alex_johannes/octa_data/Cairo/{data_type}_{mode_train}_dataset_faz.pkl"
#train_dataset = hetero_graph_loader_faz.HeteroGraphLoaderTorch(graph_path_1=vessel_graph_path,
#                                                        graph_path_2=void_graph_path,
#                                                        graph_path_3=faz_node_path,
#                                                        hetero_edges_path_12=hetero_edges_path,
#                                                        hetero_edges_path_13=faz_vessel_edge_path,
#                                                        hetero_edges_path_23=faz_region_edge_path,
#                                                        mode = mode_train,
#                                                        label_file = label_file, 
#                                                        line_graph_1 =True, 
#                                                        class_dict = octa_dr_dict,
#                                                        pickle_file = train_pickle #f"../{data_type}_{mode_train}_dataset_faz.pkl" # f"../{data_type}_{mode_train}_dataset_faz.pkl"
#                                                        )
#mode_test = "test"
#test_pickle = f"/media/data/alex_johannes/octa_data/Cairo/{data_type}_{mode_test}_dataset_faz.pkl"
#test_dataset = hetero_graph_loader_faz.HeteroGraphLoaderTorch(graph_path_1=vessel_graph_path,
#                                                        graph_path_2=void_graph_path,
#                                                        graph_path_3=faz_node_path,
#                                                        hetero_edges_path_12=hetero_edges_path,
#                                                        hetero_edges_path_13=faz_vessel_edge_path,
#                                                        hetero_edges_path_23=faz_region_edge_path,
#                                                        mode = mode_test,
#                                                        label_file = label_file, 
#                                                        line_graph_1 =True, 
#                                                        class_dict = octa_dr_dict,
#                                                        pickle_file = test_pickle # f"../{data_type}_{mode_test}_dataset_faz.pkl" #f"../{data_type}_{mode_test}_dataset_faz.pkl")
#                                                        )



mode_train = "train"
train_pickle = f"/media/data/alex_johannes/octa_data/Cairo/{data_type}_{mode_train}_dataset.pkl"
train_dataset = hetero_graph_loader.HeteroGraphLoaderTorch(vessel_graph_path,
                                                        void_graph_path,
                                                        hetero_edges_path,
                                                        mode = mode_train,
                                                        label_file = label_file, 
                                                        line_graph_1 =True, 
                                                        class_dict = octa_dr_dict,
                                                        pickle_file = train_pickle
                                                        )

mode_test = "test"
test_pickle = f"/media/data/alex_johannes/octa_data/Cairo/{data_type}_{mode_test}_dataset.pkl"
test_dataset = hetero_graph_loader.HeteroGraphLoaderTorch(vessel_graph_path,
                                                        void_graph_path,
                                                        hetero_edges_path,
                                                        mode = mode_test,
                                                        label_file = label_file, 
                                                        line_graph_1 =True, 
                                                        class_dict = octa_dr_dict,
                                                        pickle_file = test_pickle
                                                        )

# make sure that if data is read from file the classes are still the correct
train_dataset.update_class(octa_dr_dict)
test_dataset.update_class(octa_dr_dict)

# imputation and normalization
prep.hetero_graph_imputation(train_dataset)
prep.hetero_graph_imputation(test_dataset)

prep.add_node_features(train_dataset, ["graph_1", "graph_2"])
prep.add_node_features(test_dataset, ["graph_1", "graph_2"])

prep.add_global_node(train_dataset)
prep.add_global_node(test_dataset)


node_mean_tensors, node_std_tensors = prep.hetero_graph_normalization_params(train_dataset)
#node_mean_tensors = torch.load(f"../{data_type}_node_mean_tensors_global_node_node_degs.pt")
#node_std_tensors = torch.load(f"../{data_type}_node_std_tensors_global_node_node_degs.pt")

#print(node_mean_tensors)
#print(node_std_tensors)

#node_mean_tensors = torch.load(f"checkpoints/{data_type}_node_mean_tensors_global_node_node_degs.pt")
#node_std_tensors = torch.load(f"checkpoints/{data_type}_node_std_tensors_global_node_node_degs.pt")

prep.hetero_graph_normalization(train_dataset, node_mean_tensors, node_std_tensors)
prep.hetero_graph_normalization(test_dataset, node_mean_tensors, node_std_tensors)

# remove label noise, samples to exclude are stored in label_noise.json
with open("label_noise.json", "r") as file:
    label_noise_dict = json.load(file)
prep.remove_label_noise(train_dataset, label_noise_dict)



with open("label_dict.json", "r") as file:
    label_dict_full = json.load(file)
    #features_label_dict = json.load(file)
features_label_dict = copy.deepcopy(label_dict_full)

#eliminate_features = {"graph_1":["num_voxels", "maxRadiusAvg", "hasNodeAtSampleBorder", "maxRadiusStd"], 
#                      "graph_2":["centroid_weighted-0", "centroid_weighted-1", "feret_diameter_max", "equivalent_diameter"]}
#eliminate_features = {"graph_1":["num_voxels", "maxRadiusAvg", "hasNodeAtSampleBorder", "maxRadiusStd"], 
#                      "graph_2":["centroid_weighted-0", "centroid_weighted-1", "centroid-0", "centroid-1", "feret_diameter_max", "equivalent_diameter", "orientation","solidity"]}
## get positions of features to eliminate and remove them from the feature label dict and the graphs
#for key in eliminate_features.keys():
#    for feat in eliminate_features[key]:
#        idx = features_label_dict[key].index(feat)
#        features_label_dict[key].remove(feat)
#        for data in train_dataset:
#            data[key].x = torch.cat([data[key].x[:, :idx], data[key].x[:, idx+1:]], dim = 1)
#        for data in test_dataset:
#            data[key].x = torch.cat([data[key].x[:, :idx], data[key].x[:, idx+1:]], dim = 1)


max_nodes = np.max([graph.num_nodes for graph in train_dataset] + [graph.num_nodes for graph in test_dataset])
print(max_nodes)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# class weight extraction
train_labels = [int(data.y[0]) for data in train_dataset]
class_weights = prep.get_class_weights(train_labels, verbose=False)
class_weights = torch.tensor(class_weights, device= device)

weak_class_weights = class_weights ** 0.5


train_dataset.to(device)
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
                                                        batch_norm = sweep_config.batch_norm,
                                                        conv_aggr = sweep_config.conv_aggr,
                                                        hetero_conns = sweep_config.hetero_conns,
                                                        homogeneous_conv = sweep_config.homogeneous_conv,
                                                        heterogeneous_conv = sweep_config.heterogeneous_conv,
                                                        meta_data = train_dataset[0].metadata(),
                                                        faz_node= sweep_config.faz_node,
                                                        global_node = sweep_config.global_node,
                                                        )
    
    # create data loaders for training and test set
    train_loader = DataLoader(train_dataset, batch_size = sweep_config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size = 64, shuffle=False)

    # weigthings for imbalanced classes 
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

        t0 = time.time()
        
        loss, y_prob_train, y_true_train, data_loss_dict = classifier.train(train_loader, data_loss_dict)
        print(f"Epoch: {epoch:03d}")
        t1 = time.time()
        print(f"Training time: {t1-t0:.4f}")

        t0 = time.time()
        y_prob_test, y_true_test = classifier.predict(test_loader)
        t1 = time.time()
        print(f"Prediction time: {t1-t0:.4f}")

        #y_p_softmax = torch.nn.functional.softmax(torch.tensor(y_prob_test), dim = 1).detach().numpy()
        #print(y_p_softmax)

        reg = False
        if not reg:
            y_pred_test = y_prob_test.argmax(axis=1)
        else:
            y_pred_test = y_prob_test.squeeze()

        test_acc = accuracy_score(y_true_test, y_pred_test)
        test_bal_acc = balanced_accuracy_score(y_true_test, y_pred_test)
        kappa = cohen_kappa_score(y_true_test, y_pred_test, weights="quadratic")

        print(f"Test Acc: {test_acc:.4f}, Test Bal Acc: {test_bal_acc:.4f}, Kappa: {kappa:.4f}, Loss: {loss:.12f}")

        if test_bal_acc > best_val_bal_acc:
            best_val_bal_acc = test_bal_acc

        if epoch % 10 == 0:
            # print the sample with top 20 highest average loss, with the loss and the label
            print("Highest loss samples")
            # sort the loss dict by loss, descending, loss is the value of the dict
            sorted_loss_dict = {k: v for k, v in sorted(data_loss_dict.items(), key=lambda item: item[1], reverse=True)}
            # print the top 20 samples
            for key in list(sorted_loss_dict.keys())[:20]:
                print(f"Sample: {key}, Loss: {sorted_loss_dict[key]:.4f}")
        

        if epoch == 100:  # kappa > 0.70 or epoch == 20
            print("start_explain")

            indices = [0, 1, 13, 15]
            selected_samples = [test_dataset[i] for i in indices]

            gnnexp_loader = DataLoader(selected_samples, batch_size = 1, shuffle=False) # 
            print(y_true_test)
            print(y_pred_test)
            for idx, data in enumerate(gnnexp_loader):
                cam_cls = np.arange(num_classes)
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
                cam_cls = np.arange(num_classes) # god knows why
                cam = grad_cam.grad_cam_data(cam_model, data, cam_cls, scale = False, relu = True, start = False, abs=True)
                fig, ax = plt.subplots(1,num_classes, figsize=(5*num_classes,5))
                # add titles for each subplot in the figure
                for i, ax_iter in enumerate(ax):
                    ax_iter.set_title(label_names[i])

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
                plt.savefig(f'explain_out/gradCAM_{idx}_plot_{epoch}_final.png' ) 
                plt.close()


                #data = next(iter(gnnexp_loader))
                #data = data.to(device)
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
                #explanation2 = explainer(data.x_dict, data.edge_index_dict,target =target,  batch_dict = data.batch_dict,  grads = True)
                #explanation3 = explainer(data.x_dict, data.edge_index_dict,target =target,  batch_dict = data.batch_dict,  grads = False, index= 0)

                #print(explanation)

                # for GNN Explainer forces sparsity, thres
                torch_geom_explanation.visualize_feature_importance(explanation1, f'explain_out/feature_importance_{idx}_{explain_type}_{epoch}_attributes_expnum_1.png', features_label_dict, top_k = 50)
                #torch_geom_explanation.visualize_feature_importance(explanation2, f'explain_out/feature_importance_{idx}_{explain_type}_{epoch}_attributes_expnum_2.png', features_label_dict, top_k = 50)
                #torch_geom_explanation.visualize_feature_importance(explanation3, f'explain_out/feature_importance_{idx}_{explain_type}_{epoch}_attributes_expnum_3.png', features_label_dict, top_k = 50)
                torch_geom_explanation.visualize_relevant_subgraph(explanation1, data, f"explain_out/subgraph_{idx}_{explain_type}_{epoch}_attributes_expnum_1.png", threshold = "adaptive", edge_threshold = "adaptive", edges = edges) #"adaptive"
                #torch_geom_explanation.visualize_relevant_subgraph(explanation2, data, f"explain_out/subgraph_{idx}_{explain_type}_{epoch}_attributes_expnum_2.png", threshold = "adaptive", edge_threshold = "adaptive", edges = edges) #"adaptive"
                #torch_geom_explanation.visualize_relevant_subgraph(explanation3, data, f"explain_out/subgraph_{idx}_{explain_type}_{epoch}_attributes_expnum_3.png", threshold = "adaptive", edge_threshold = "adaptive", edges = edges) #"adaptive"
                #if epoch == 1:
                #    graph_2D.HeteroGraphPlotter2D().plot_graph_2D(data, edges= True, path = f"fullbgraph_{idx}_{explain_type}.png")

            break
main()

