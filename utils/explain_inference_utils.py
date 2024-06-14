import json
import os
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

from models import heterogeneous_gnn, homogeneous_gnn


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


def evaluate_performance_baselines(
    clf, data_x, data_y, OCTA500=False, label_names=["Healthy/DM", "NPDR", "PDR"]
):
    y_pred = clf.predict(data_x)
    if OCTA500:
        y_pred[y_pred > 0] = 1
        label_names = ["Healthy", "DR"]
    report = classification_report(
        data_y, y_pred, target_names=label_names, output_dict=True
    )
    acc = accuracy_score(data_y, y_pred)
    bal_acc = balanced_accuracy_score(data_y, y_pred)
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


def create_homogeneous_model(model_config, out_channels):
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
    activation_dict = {
        "relu": torch.nn.functional.relu,
        "leaky": torch.nn.functional.leaky_relu,
        "elu": torch.nn.functional.elu,
    }

    model = homogeneous_gnn.Homogeneous_GNN(
        hidden_channels=model_config["parameters"]["hidden_channels"],
        out_channels=out_channels,
        num_layers=model_config["parameters"]["num_layers"],
        dropout=0,
        aggregation_mode=agg_mode_dict[model_config["parameters"]["aggregation_mode"]],
        num_pre_processing_layers=model_config["parameters"]["pre_layers"],
        num_post_processing_layers=model_config["parameters"]["post_layers"],
        batch_norm=model_config["parameters"]["batch_norm"],
        homogeneous_conv=homogeneous_conv_dict[model_config["parameters"]["homogeneous_conv"]],
        activation=activation_dict[model_config["parameters"]["activation"]],
    )
    return model