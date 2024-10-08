import copy
import json
import pickle
from collections import Counter

import numpy as np
import torch


def get_class_weights(train_labels, verbose=False):
    """Calculates the class weights for the dataset

    Paramters
    ---------
    train_labels: A 1D tensor containing the labels of the training set
    verbose: A boolean indicating if the label distribution and class weights should be printed

    Returns
    -------
    class_weights: A 1D tensor containing the class weights
    """

    # Calculate the label distribution
    label_distribution = Counter(train_labels)

    # Calculate the class weights
    total_samples = len(train_labels)
    class_weights = [
        total_samples / (len(label_distribution) * count)
        for label, count in label_distribution.items()
    ]

    if verbose:
        # Print or use the label distribution as needed
        print("Label distribution in the training set:")
        for label, count in label_distribution.items():
            print(f"Class {label}: {count} samples")

        print("Class weights:")
        for label, weight in zip(label_distribution.keys(), class_weights):
            print(f"Class {label}: Weight {weight}")

    return class_weights


def hetero_graph_imputation(dataset):
    """Performs imputation on the dataset

    Paramters
    ---------
    dataset: An iterable of torch_geometric.data.Data objects

    Returns
    -------
    """
    for data in dataset:
        # check if the .x and .edge_index contains nan values
        for key, val in data.x_dict.items():
            if torch.any(torch.isnan(val)):
                # print("nan values in x")
                nan_pos = torch.where(torch.isnan(val))
                # imputation with mean value of feature
                val[nan_pos] = 0
            if torch.any(torch.isinf(val)):
                # print("inf values in x")
                inf_pos = torch.where(torch.isinf(val))
                val[inf_pos] = 0


def mad(data, dim=None):
    median = torch.median(data, dim=dim).values
    mad = torch.median(torch.abs(data - median), dim=dim).values
    scale_factor = 1.4826  # Scaling factor for normal distribution
    mad *= scale_factor
    return mad


def hetero_graph_standardization_params(train_dataset, robust=False):
    """Extracts the mean and std of the node features from the dataset

    Paramters
    ---------
    train_dataset: An iterable of torch_geometric.data.Data objects

    Returns
    -------
    node_mean_tensors: A dictionary of the mean of the node features
    node_std_tensors: A dictionary of the std of the node features
    """
    try:
        iterable = train_dataset.hetero_graph_list
    except AttributeError:
        iterable = train_dataset

    node_tensors = {}
    node_mean_tensors = {}
    node_std_tensors = {}
    for key, val in iterable[0].x_dict.items():
        node_tensors[key] = None

    for data in iterable:
        for key, val in data.x_dict.items():
            node_tensors[key] = (
                torch.cat([node_tensors[key], val])
                if node_tensors[key] is not None
                else val
            )

    for key, val in node_tensors.items():
        if robust:
            node_mean_tensors[key] = torch.median(val, dim=0).values
            node_std_tensors[key] = mad(val, dim=0)
        else:
            node_mean_tensors[key] = torch.mean(val, dim=0)
            node_std_tensors[key] = torch.std(val, dim=0)

    return node_mean_tensors, node_std_tensors


def hetero_graph_min_max_params(train_dataset, robust=False):
    """Extracts the min and max of the node features from the dataset

    Paramters
    ---------
    train_dataset: An iterable of torch_geometric.data.Data objects

    Returns
    -------
    node_min_tensors: A dictionary of the min of the node features
    node_max_tensors: A dictionary of the max of the node features
    """
    try:
        iterable = train_dataset.hetero_graph_list
    except AttributeError:
        iterable = train_dataset

    node_tensors = {}
    node_min_tensors = {}
    node_max_tensors = {}
    for key, val in iterable[0].x_dict.items():
        node_tensors[key] = None

    for data in iterable:
        for key, val in data.x_dict.items():
            node_tensors[key] = (
                torch.cat([node_tensors[key], val])
                if node_tensors[key] is not None
                else val
            )

    for key, val in node_tensors.items():
        if robust:
            node_min_tensors[key] = torch.quantile(val, 0.25, dim=0)
            node_max_tensors[key] = torch.quantile(val, 0.75, dim=0)
        else:
            node_min_tensors[key] = torch.min(val, dim=0).values
            node_max_tensors[key] = torch.max(val, dim=0).values

    return node_min_tensors, node_max_tensors


def hetero_graph_min_max_scaling(dataset, node_min_tensors, node_max_tensors):
    """Scales the node features in the dataset

    Paramters
    ---------
    dataset: An iterable of torch_geometric.data.Data objects
    node_min_tensors: A dictionary of the min of the node features
    node_max_tensors: A dictionary of the max of the node features

    Returns
    -------
    """
    for data in dataset:
        for key in data.x_dict.keys():
            data.x_dict[key] -= node_min_tensors[key]
            data.x_dict[key] /= node_max_tensors[key] - node_min_tensors[key]


def hetero_graph_standardization(dataset, node_mean_tensors, node_std_tensors):
    """Normalizes the node features in the dataset

    Paramters
    ---------
    dataset: An iterable of torch_geometric.data.Data objects
    node_mean_tensors: A dictionary of the mean of the node features
    node_std_tensors: A dictionary of the std of the node features

    Returns
    -------
    """

    for key in node_std_tensors.keys():
        node_std_tensors[key] = torch.where(
            node_std_tensors[key] == 0,
            torch.ones_like(node_std_tensors[key]),
            node_std_tensors[key],
        )

    for data in dataset:
        for key in data.x_dict.keys():
            data.x_dict[key] -= node_mean_tensors[key]
            # avoid division by zero
            # data.x_dict[key] /= node_std_tensors[key] + 1e-8
            data.x_dict[key] /= node_std_tensors[key]
            # set values larger than 10 to 10
            # data.x_dict[key] = torch.where(data.x_dict[key] > 10, torch.ones_like(data.x_dict[key]) * 10, data.x_dict[key])
            ## set values smaller than -10 to -10
            # data.x_dict[key] = torch.where(data.x_dict[key] < -10, torch.ones_like(data.x_dict[key]) * -10, data.x_dict[key])


def add_virtual_node(dataset):
    """Adds a virtual node to the dataset, that is connected to all other nodes and has 0 embeddings

    Paramters
    ---------
    dataset: An iterable of torch_geometric.data.Data objects

    Returns
    -------
    """

    for data in dataset:
        # add a virtual node that connects to all other nodes, with 0 embeddings
        data["global"].x = torch.zeros((1, 64)).float()
        for key in data.x_dict.keys():
            # connect the virtual node to all other nodes

            data[("global", "to", key)].edge_index = torch.zeros(
                (2, data.x_dict[key].shape[0])
            ).long()
            data[("global", "to", key)].edge_index[0, :] = 0
            data[("global", "to", key)].edge_index[1, :] = torch.arange(
                data.x_dict[key].shape[0]
            )

            data[(key, "rev_to", "global")].edge_index = torch.zeros(
                (2, data.x_dict[key].shape[0])
            ).long()
            data[(key, "rev_to", "global")].edge_index[0, :] = torch.arange(
                data.x_dict[key].shape[0]
            )
            data[(key, "rev_to", "global")].edge_index[1, :] = 0


def add_node_features(dataset, node_types):
    """Adds the node degrees to the node features in the dataset

    Paramters
    ---------
    dataset: An iterable of torch_geometric.data.Data objects
    node_types: A list of the node types in the dataset

    Returns
    -------
    """
    for data in dataset:
        if len(node_types) == 2:
            heter_node_degrees_1, heter_node_degrees_2 = calculate_node_degrees(
                data.edge_index_dict[(node_types[0], "to", node_types[1])],
                data.x_dict[node_types[0]].shape[0],
                data.x_dict[node_types[1]].shape[0],
            )
            heter_node_degrees_1 = heter_node_degrees_1.unsqueeze(1).float()
            heter_node_degrees_2 = heter_node_degrees_2.unsqueeze(1).float()

        for i, key in enumerate(node_types):
            node_degrees = calculate_node_degrees(
                data.edge_index_dict[(key, "to", key)], data.x_dict[key].shape[0]
            )
            node_degrees = node_degrees.unsqueeze(1).float()
            if len(node_types) == 2:
                if i == 0:
                    data[key].x = torch.cat(
                        (data.x_dict[key], node_degrees, heter_node_degrees_1), dim=1
                    )
                else:
                    data[key].x = torch.cat(
                        (data.x_dict[key], node_degrees, heter_node_degrees_2), dim=1
                    )
            else:
                data[key].x = torch.cat((data.x_dict[key], node_degrees), dim=1)


def calculate_node_degrees(edge_index, num_nodes, num_nodes_2=None):
    """Calculates the degrees of the nodes in the graph

    Paramters
    ---------
    edge_index: A 2D tensor of shape (2, num_edges)
    num_nodes: The number of nodes in the graph
    num_nodes_2: The number of nodes in the second graph

    Returns
    -------
    degrees: A 1D tensor of shape (num_nodes,) containing the degrees of the nodes
    """

    # Ensure that edge_index is a 2D tensor with two rows
    if edge_index.shape[0] != 2:
        raise ValueError("edge_index should have two rows.")

    if num_nodes_2 is not None:
        degrees = torch.bincount(edge_index[0], minlength=num_nodes)
        degrees_2 = torch.bincount(edge_index[1], minlength=num_nodes_2)

        return degrees, degrees_2
    else:
        degrees = torch.bincount(edge_index[0], minlength=num_nodes)
        degrees += torch.bincount(edge_index[1], minlength=num_nodes)

        return degrees


def remove_label_noise(dataset, label_noise_dict):
    """Removes the nodes with label noise from the dataset

    Paramters
    ---------
    dataset: An iterable of torch_geometric.data.Data objects
    label_noise_dict: A dictionary of the graphs with label noise

    Returns
    -------
    """
    for key in label_noise_dict.keys():
        for idx in label_noise_dict[key]:
            dataset.hetero_graphs.pop(idx)

    dataset.hetero_graph_list = list(dataset.hetero_graphs.values())


def adjust_data_for_split(
    cv_dataset,
    final_test_dataset,
    split,
    faz=False,
    use_full_cv=False,
    robust=False,
    min_max=False,
):
    """Adjusts the datasets for the split, set the split, impute, normalize and add node features

    Paramters
    ---------
    cv_dataset: An iterable of torch_geometric.data.Data objects
    final_test_dataset: An iterable of torch_geometric.data.Data objects
    split: The split to adjust the datasets for
    faz: A boolean indicating if the datasets contain a faz node type

    Returns
    -------
    train_dataset: The training dataset
    val_dataset: The validation dataset
    test_dataset: The test dataset
    """
    if use_full_cv:
        cv_dataset_cp = copy.deepcopy(cv_dataset)
    train_dataset = copy.deepcopy(cv_dataset)
    val_dataset = copy.deepcopy(cv_dataset)
    val_dataset.set_split("val", split)
    train_dataset.set_split("train", split)
    test_dataset = copy.deepcopy(final_test_dataset)
    print(f"train dataset: {len(train_dataset)}")
    print(f"val dataset: {len(val_dataset)}")
    print(f"test dataset: {len(test_dataset)}")

    # data imputation
    hetero_graph_imputation(train_dataset)
    hetero_graph_imputation(val_dataset)
    hetero_graph_imputation(test_dataset)

    # check if the datasets have an faz node type
    node_types = ["graph_1", "graph_2"]
    if faz:
        node_types.append("faz")

    # adding degree features
    add_node_features(train_dataset, node_types)
    add_node_features(val_dataset, node_types)
    add_node_features(test_dataset, node_types)
    if use_full_cv:
        add_node_features(cv_dataset_cp, node_types)

    # extract normalization parameters
    if use_full_cv:
        #
        if min_max:
            node_min_tensors, node_max_tensors = hetero_graph_min_max_params(
                cv_dataset_cp, robust=robust
            )
        else:
            node_mean_tensors, node_std_tensors = hetero_graph_standardization_params(
                cv_dataset_cp, robust=robust
            )

    else:
        if min_max:
            node_min_tensors, node_max_tensors = hetero_graph_min_max_params(
                train_dataset, robust=robust
            )
        else:
            node_mean_tensors, node_std_tensors = hetero_graph_standardization_params(
                train_dataset, robust=robust
            )
        # node_mean_tensors, node_std_tensors = hetero_graph_standardization_params(train_dataset, robust= robust)

    if min_max:
        hetero_graph_min_max_scaling(train_dataset, node_min_tensors, node_max_tensors)
        hetero_graph_min_max_scaling(val_dataset, node_min_tensors, node_max_tensors)
        hetero_graph_min_max_scaling(test_dataset, node_min_tensors, node_max_tensors)
    else:
        hetero_graph_standardization(train_dataset, node_mean_tensors, node_std_tensors)
        hetero_graph_standardization(val_dataset, node_mean_tensors, node_std_tensors)
        hetero_graph_standardization(test_dataset, node_mean_tensors, node_std_tensors)

    return train_dataset, val_dataset, test_dataset


def eliminate_features(included_features, features_label_dict, datasets):
    """Eliminates the features that are not included in the included_features dictionary

    Paramters
    ---------
    included_features: A dictionary of the included features
    features_label_dict: A dictionary of all the available features
    datasets: An iterable of iterables of torch_geometric.data.Data objects

    Returns
    -------
    """
    eliminate_features = {}
    for key in features_label_dict.keys():
        eliminate_features[key] = []
        for feat in features_label_dict[key]:
            if feat not in included_features[key]:
                eliminate_features[key].append(feat)

    for key in eliminate_features.keys():
        for feat in eliminate_features[key]:
            idx = features_label_dict[key].index(feat)
            features_label_dict[key].remove(feat)
            for dataset in datasets:
                for data in dataset:
                    data[key].x = torch.cat(
                        [data[key].x[:, :idx], data[key].x[:, idx + 1 :]], dim=1
                    )


def log_scaling(log_scale_dict, feature_label_dict, datasets):
    """Applies log scaling to the features in the datasets

    Paramters
    ---------
    log_scale_dict: A dictionary of the features that should be log scaled
    feature_label_dict: A dictionary of the available features
    datasets: An iterable of iterables of torch_geometric.data.Data objects

    Returns
    -------
    """
    for key in log_scale_dict.keys():
        for feat in log_scale_dict[key]:
            idx = feature_label_dict[key].index(feat)
            for dataset in datasets:
                for data in dataset:
                    data[key].x[:, idx] = torch.log(data[key].x[:, idx] + 1)

    # also need to adjust the dictionary of the graphs

    for key in log_scale_dict.keys():
        for feat in log_scale_dict[key]:
            idx = feature_label_dict[key].index(feat)
            for dataset in datasets:
                for data in dataset.hetero_graphs.values():
                    data[key].x[:, idx] = torch.log(data[key].x[:, idx] + 1)


def aggreate_graph(dataset, features_label_dict, faz=False, agg_type="sum", merge_faz=False):
    # check if the datasets have an faz node type
    node_types = ["graph_1", "graph_2"]

    if faz:
        node_types.append("faz")
        x_1_shape = (
            len(features_label_dict["graph_2"])
            + len(features_label_dict["graph_1"])
            + len(features_label_dict["faz"])
        )
    else:
        x_1_shape = len(features_label_dict["graph_2"]) + len(
            features_label_dict["graph_1"]
        )

    x_0_shape = len(dataset)
    x = np.zeros((x_0_shape, x_1_shape))
    y = np.zeros((x_0_shape,))

    for i in range(len(dataset)):
        graph_1 = dataset[i]["graph_1"]
        graph_2 = dataset[i]["graph_2"]
        y[i] = dataset[i].y

        # average the features of all nodes in the graph

        if agg_type == "sum":
            graph_1_aggr = graph_1.x.numpy().sum(axis=0)
            graph_2_aggr = graph_2.x.numpy().sum(axis=0)
        elif agg_type == "mean":
            graph_1_aggr = graph_1.x.numpy().mean(axis=0)
            graph_2_aggr = graph_2.x.numpy().mean(axis=0)
        elif agg_type == "max":
            graph_1_aggr = graph_1.x.numpy().max(axis=0)
            graph_2_aggr = graph_2.x.numpy().max(axis=0)

        x[i, : len(features_label_dict["graph_1"])] = graph_1_aggr

        if faz and not merge_faz:
            x[
                i,
                len(features_label_dict["graph_1"]) : len(
                    features_label_dict["graph_2"] + features_label_dict["graph_1"]
                ),
            ] = graph_2_aggr
            faz_node = dataset[i]["faz"]
            faz_sum = faz_node.x.numpy().sum(axis=0)
            x[
                i,
                len(features_label_dict["graph_1"])
                + len(features_label_dict["graph_2"]) :,
            ] = faz_sum

        # merge all but the last values of the faz with the graph_2 values
        elif faz and merge_faz and agg_type == "sum":
            faz_node = dataset[i]["faz"]
            faz_sum = faz_node.x.numpy().sum(axis=0)
            graph_2_aggr[: len(faz_sum)-1] += faz_sum[: len(faz_sum)-1]
            graph_2_aggr[-1] += faz_sum[-1]

            x[
                i,
                len(features_label_dict["graph_1"]) : len(
                    features_label_dict["graph_2"] + features_label_dict["graph_1"]
                ),
            ] = graph_2_aggr

    return x, y


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

    train_dataset, val_dataset, test_dataset = adjust_data_for_split(
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
