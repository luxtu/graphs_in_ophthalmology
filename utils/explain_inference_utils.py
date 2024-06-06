import copy
import json
import os
import pickle
import time

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
)
from torch_geometric.nn import (
    GATConv,
    GCNConv,
    GraphConv,
    SAGEConv,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)

from models import heterogeneous_gnn
from utils import dataprep_utils


def evaluate_training_time(clf, loader, epochs=100):
    start = time.time()
    print("Start training")
    for epoch in range(1, epochs + 1):
        _, _, _, _ = clf.train(loader)
    end = time.time()
    print(f"Training time: {end - start}")
    return end - start


def delete_edges(dataset):
    # for data in test_dataset:
    for data in dataset:
        for edge_type in data.edge_index_dict.keys():
            # assign each edge type to an empty tensor with shape (2,0)
            data[edge_type].edge_index = torch.zeros((2, 0), dtype=torch.long)
    return dataset


def evaluate_performance(
    clf, loader, OCTA500=False, label_names=["Healthy/DM", "NPDR", "PDR"]
):
    y_prob, y_true = clf.predict(loader)
    y_pred = np.argmax(y_prob, axis=1)
    if OCTA500:
        y_pred[y_pred > 0] = 1
        label_names = ["Healthy", "DR"]
    report = classification_report(
        y_true, y_pred, target_names=label_names, output_dict=True
    )
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    metrics = {"accuracy": acc, "balanced_accuracy": bal_acc}

    return report, metrics


def get_param_count(state_dict):
    # count the number of parameters, only count the initialized parameters
    param_sum = 0
    ct = 0
    for key in state_dict.keys():
        try:
            param_sum += state_dict[key].numel()
        except ValueError:
            ct += 1

    return param_sum, ct


def load_state_dict_and_model_config(checkpoint_folder, run_id):
    """
    Load the state dict into the model
    """
    # extract the checkpoint files
    check_point_files = [f for f in os.listdir(checkpoint_folder) if f.endswith(".pt")]

    # find the json file that contains the model configuration
    json_files = [f for f in os.listdir(checkpoint_folder) if f.endswith(".json")]

    for json_file in json_files:
        if run_id in json_file:
            # load the json file
            model_config = json.load(
                open(os.path.join(checkpoint_folder, json_file), "r")
            )
            break
    # load the model
    state_dict = torch.load(
        os.path.join(
            checkpoint_folder, [f for f in check_point_files if run_id in f][0]
        )
    )
    return state_dict, model_config


def load_private_pickled_data():
    """
    Load the pickled data
    """
    data_type = "DCP"

    mode_cv = "cv"
    mode_final_test = "final_test"

    # load the datasets
    cv_pickle_processed = f"../data/{data_type}_{mode_cv}_selected_sweep_repeat_v2.pkl"
    final_test_pickle_processed = (
        f"../data/{data_type}_{mode_final_test}_selected_sweep_repeat_v2.pkl"
    )
    # load the pickled datasets
    with open(cv_pickle_processed, "rb") as file:
        cv_dataset = pickle.load(file)

    with open(final_test_pickle_processed, "rb") as file:
        final_test_dataset = pickle.load(file)

    return cv_dataset, final_test_dataset


def load_private_datasets(split, faz_node_bool):
    """
    Load the datasets
    """
    cv_dataset, final_test_dataset = load_private_pickled_data()

    train_dataset, val_dataset, test_dataset = dataprep_utils.adjust_data_for_split(
        cv_dataset, final_test_dataset, split, faz=True
    )

    dataset_list = [train_dataset, val_dataset, test_dataset]
    dataset_list, features_label_dict = delete_highly_correlated_features(
        dataset_list, faz_node_bool
    )

    return dataset_list, features_label_dict


def delete_highly_correlated_features(dataset_list, faz_node_bool):
    with open("feature_configs/feature_name_dict_new.json", "r") as file:
        label_dict_full = json.load(file)
    features_label_dict = copy.deepcopy(label_dict_full)

    # eliminate features with extremely high correlation to other features
    eliminate_features = {
        "graph_1": [
            "num_voxels",
            "hasNodeAtSampleBorder",
            "maxRadiusAvg",
            "maxRadiusStd",
        ],
        "graph_2": [
            "centroid_weighted-0",
            "centroid_weighted-1",
            "feret_diameter_max",
            "orientation",
        ],
    }

    if faz_node_bool:
        eliminate_features["faz"] = [
            "centroid_weighted-0",
            "centroid_weighted-1",
            "feret_diameter_max",
            "orientation",
        ]

    # get positions of features to eliminate and remove them from the feature label dict and the graphs
    for key in eliminate_features.keys():
        for feat in eliminate_features[key]:
            idx = features_label_dict[key].index(feat)
            features_label_dict[key].remove(feat)
            for dataset in dataset_list:
                for data in dataset:
                    data[key].x = torch.cat(
                        [data[key].x[:, :idx], data[key].x[:, idx + 1 :]], dim=1
                    )

    # add the new features to the feature label dict
    features_label_dict["graph_1"] = (
        features_label_dict["graph_1"][:-1]
        + ["cl_mean", "cl_std", "q25", "q75"]
        + ["degree"]
    )
    features_label_dict["graph_2"] = (
        features_label_dict["graph_2"][:-1] + ["q25", "q75", "std"] + ["degree"]
    )
    features_label_dict["faz"] = features_label_dict["faz"]

    return dataset_list, features_label_dict


def create_model(model_config, node_types, out_channels):
    """
    Create the model
    """
    agg_mode_dict = {
        "mean": global_mean_pool,
        "max": global_max_pool,
        "add": global_add_pool,
        "add_max": [global_add_pool, global_max_pool],
        "max_mean": [global_max_pool, global_mean_pool],
        "add_mean": [global_add_pool, global_mean_pool],
    }
    homogeneous_conv_dict = {
        "gat": GATConv,
        "sage": SAGEConv,
        "graph": GraphConv,
        "gcn": GCNConv,
    }
    heterogeneous_conv_dict = {"gat": GATConv, "sage": SAGEConv, "graph": GraphConv}
    activation_dict = {
        "relu": torch.nn.functional.relu,
        "leaky": torch.nn.functional.leaky_relu,
        "elu": torch.nn.functional.elu,
    }

    model = heterogeneous_gnn.Heterogeneous_GNN(
        hidden_channels=model_config["parameters"]["hidden_channels"],
        out_channels=out_channels,
        num_layers=model_config["parameters"]["num_layers"],
        dropout=0,
        aggregation_mode=agg_mode_dict[model_config["parameters"]["aggregation_mode"]],
        node_types=node_types,
        num_pre_processing_layers=model_config["parameters"]["pre_layers"],
        num_post_processing_layers=model_config["parameters"]["post_layers"],
        batch_norm=model_config["parameters"]["batch_norm"],
        conv_aggr=model_config["parameters"]["conv_aggr"],
        hetero_conns=model_config["parameters"]["hetero_conns"],
        homogeneous_conv=homogeneous_conv_dict[
            model_config["parameters"]["homogeneous_conv"]
        ],
        heterogeneous_conv=heterogeneous_conv_dict[
            model_config["parameters"]["heterogeneous_conv"]
        ],
        activation=activation_dict[model_config["parameters"]["activation"]],
        faz_node=model_config["parameters"]["faz_node"],
        start_rep=model_config["parameters"]["start_rep"],
        aggr_faz=model_config["parameters"]["aggr_faz"],
    )
    return model
