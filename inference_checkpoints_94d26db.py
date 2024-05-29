import os
from loader import hetero_graph_loader, hetero_graph_loader_faz
from utils import prep, dataprep_utils, train_utils
import json
import copy
import torch
from models import graph_classifier, heterogeneous_gnn
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn import GATConv, SAGEConv, GraphConv, GCNConv
from torch_geometric.loader import DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, balanced_accuracy_score, cohen_kappa_score
import time


check_point_folder = "../data/checkpoints_94d26db_final_27022024" 

# extract the files with .pt extension
check_point_files = [f for f in os.listdir(check_point_folder) if f.endswith(".pt")]

# extract the run id from the file name
# run id are the characters after the last underscore
run_ids = [f.split("_")[-1].split(".")[0] for f in check_point_files]

# find the json file with the same run id
# json file name is the same as the run id
json_files = [f for f in os.listdir(check_point_folder) if f.endswith(".json")]


data_type = "DCP"
mode_final_test = "final_test"
# load the data

octa_dr_dict = {"Healthy": 0, "DM": 0, "PDR": 2, "Early NPDR": 1, "Late NPDR": 1}
label_names = ["Healthy/DM", "NPDR", "PDR"]



mode_cv = "cv"
mode_final_test = "final_test"

faz_node_bool = True

# pickle the datasets
cv_pickle_processed = f"../data/{data_type}_{mode_cv}_selected_sweep_repeat_v2.pkl"
final_test_pickle_processed = f"../data/{data_type}_{mode_final_test}_selected_sweep_repeat_v2.pkl" 

import pickle
#load the pickled datasets
with open(cv_pickle_processed, "rb") as file:
    cv_dataset = pickle.load(file)

with open(final_test_pickle_processed, "rb") as file:
    final_test_dataset = pickle.load(file)



agg_mode_dict = {"mean": global_mean_pool, "max": global_max_pool, "add": global_add_pool, "add_max": [global_add_pool, global_max_pool], "max_mean": [global_max_pool, global_mean_pool], "add_mean": [global_add_pool, global_mean_pool]}
homogeneous_conv_dict = {"gat": GATConv, "sage":SAGEConv, "graph" : GraphConv, "gcn" : GCNConv}
heterogeneous_conv_dict = {"gat": GATConv, "sage":SAGEConv, "graph" : GraphConv}
activation_dict = {"relu":torch.nn.functional.relu, "leaky" : torch.nn.functional.leaky_relu, "elu":torch.nn.functional.elu}


df_report = pd.DataFrame()
df_metrics = pd.DataFrame()

# shufffle check_point_files and run_ids the same way
check_point_files, run_ids = zip(*sorted(zip(check_point_files, run_ids)))
for chekpoint_file, run_id in zip(check_point_files, run_ids):

    print(chekpoint_file, run_id)
    for json_file in json_files:
        if run_id in json_file:
            # load the json file
            print(json_file)
            sweep_configuration = json.load(open(os.path.join(check_point_folder, json_file), "r"))
            break

    # load the model
    state_dict = torch.load(os.path.join(check_point_folder, chekpoint_file))
    node_types = ["graph_1", "graph_2"]
    # load the model
    out_channels = 3
    model = heterogeneous_gnn.Heterogeneous_GNN(hidden_channels= sweep_configuration["parameters"]["hidden_channels"],
                                                        out_channels= out_channels,
                                                        num_layers= sweep_configuration["parameters"]["num_layers"],
                                                        dropout = 0, 
                                                        aggregation_mode= agg_mode_dict[sweep_configuration["parameters"]["aggregation_mode"]], 
                                                        node_types = node_types,
                                                        num_pre_processing_layers = sweep_configuration["parameters"]["pre_layers"],
                                                        num_post_processing_layers = sweep_configuration["parameters"]["post_layers"],
                                                        batch_norm = sweep_configuration["parameters"]["batch_norm"],
                                                        conv_aggr = sweep_configuration["parameters"]["conv_aggr"],
                                                        hetero_conns = sweep_configuration["parameters"]["hetero_conns"],
                                                        homogeneous_conv= homogeneous_conv_dict[sweep_configuration["parameters"]["homogeneous_conv"]],
                                                        heterogeneous_conv= heterogeneous_conv_dict[sweep_configuration["parameters"]["heterogeneous_conv"]],
                                                        activation= activation_dict[sweep_configuration["parameters"]["activation"]],
                                                        faz_node = sweep_configuration["parameters"]["faz_node"],
                                                        start_rep = sweep_configuration["parameters"]["start_rep"],
                                                        aggr_faz = sweep_configuration["parameters"]["aggr_faz"],
                                                        )



    split = sweep_configuration["split"]
    print(split)
    train_dataset, val_dataset, test_dataset = dataprep_utils.adjust_data_for_split(cv_dataset, final_test_dataset, split, faz = True, use_full_cv = False, min_max=False)

    # put the datasets on the gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset.to(device)
    #val_dataset.to(device)
    #test_dataset.to(device)

    # delete all the edges to check what effect they have on the prediction
    #for data in test_dataset:
    #    for edge_type in data.edge_index_dict.keys():
    #        # assign each edge type to an empty tensor with shape (2,0)
    #        data[edge_type].edge_index = torch.zeros((2,0), dtype = torch.long)

    print(train_dataset[0])
    

    with open("training_configs/feature_name_dict_new.json", "r") as file:
        label_dict_full = json.load(file)
        #features_label_dict = json.load(file)
    features_label_dict = copy.deepcopy(label_dict_full)

    eliminate_features = {"graph_1":["num_voxels", "maxRadiusAvg", "hasNodeAtSampleBorder", "maxRadiusStd"], 
                          "graph_2":["centroid_weighted-0", "centroid_weighted-1", "feret_diameter_max", "orientation"]}

    if faz_node_bool:
        eliminate_features["faz"] = ["centroid_weighted-0", "centroid_weighted-1", "feret_diameter_max", "orientation"]

    for key in features_label_dict.keys():
        print(key, len(features_label_dict[key]))
    


    # get positions of features to eliminate and remove them from the feature label dict and the graphs
    for key in eliminate_features.keys():
        for feat in eliminate_features[key]:
            idx = features_label_dict[key].index(feat)
            features_label_dict[key].remove(feat)
            for data in test_dataset:
                data[key].x = torch.cat([data[key].x[:, :idx], data[key].x[:, idx+1:]], dim = 1)
            for data in val_dataset:
                data[key].x = torch.cat([data[key].x[:, :idx], data[key].x[:, idx+1:]], dim = 1)



    irrelevant_loss = torch.nn.CrossEntropyLoss()
    clf = graph_classifier.graphClassifierHetero_94d26db(model, irrelevant_loss) 

    train_loader = DataLoader(train_dataset, batch_size = 16, shuffle=True)

    # run 100 epochs of training and check the time it takes

    start = time.time()
    print("Start training")
    for epoch in range(1, 100 + 1):
        loss, y_prob_train, y_true_train, _  = clf.train(train_loader)
    end = time.time()
    print(f"Training time: {end - start}")
    exit()

    test_loader_set = DataLoader(test_dataset[:4], batch_size = 4, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size = 64, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size = 64, shuffle=False)

    print(test_dataset[0])

    # must run on an input to set the parameters
    # load the state dict
    clf.predict(test_loader_set)
    clf.model.load_state_dict(state_dict)

    y_prob_test, y_test = clf.predict(test_loader)
    y_prob_val, y_val = clf.predict(val_loader)

    y_pred = np.argmax(y_prob_test, axis = 1)
    y_val_pred = np.argmax(y_prob_val, axis = 1)

    report = classification_report(y_test, y_pred, target_names=label_names, output_dict=True)

    acc= accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred, weights="quadratic")

    val_acc= accuracy_score(y_val, y_val_pred)
    val_bal_acc = balanced_accuracy_score(y_val, y_val_pred)
    val_kappa = cohen_kappa_score(y_val, y_val_pred, weights="quadratic")

    metrics = {}
    val_metrics = {}

    metrics["accuracy"] = acc
    metrics["balanced_accuracy"] = bal_acc
    metrics["kappa"] = kappa

    val_metrics["accuracy"] = val_acc
    val_metrics["balanced_accuracy"] = val_bal_acc
    val_metrics["kappa"] = val_kappa

    df_metrics = pd.concat([df_metrics, pd.DataFrame(metrics, index = [split])])
    df_report = pd.concat([df_report, pd.DataFrame(report).transpose()])

    #res_dict, y_pred_softmax_val, y_true_val = train_utils.evaluate_model(clf, val_loader, test_loader)
    #print(res_dict)

    print(metrics)
    print(val_metrics)

#df_metrics.to_csv('GNN_CV_metrics_94d26db_v1.csv')
#df_report.to_csv('GNN_CV_report_94d26db_v1.csv')

print(df_metrics)
print(df_report)