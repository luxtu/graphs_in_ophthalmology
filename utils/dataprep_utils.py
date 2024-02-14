import torch
import copy
from collections import Counter

def get_class_weights(train_labels, verbose = False):
    """ Calculates the class weights for the dataset

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
    class_weights = [total_samples / (len(label_distribution) * count) for label, count in label_distribution.items()]

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
    """ Performs imputation on the dataset

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
                #print("nan values in x")
                nan_pos = torch.where(torch.isnan(val))
                # imputation with mean value of feature
                val[nan_pos] = 0
            if torch.any(torch.isinf(val)):
                #print("inf values in x")
                inf_pos = torch.where(torch.isinf(val))
                val[inf_pos] = 0


def hetero_graph_normalization_params(train_dataset, clean = False):
    """ Extracts the mean and std of the node features from the dataset

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
            node_tensors[key] = torch.cat([node_tensors[key], val]) if node_tensors[key] is not None else val

    for key, val in node_tensors.items():
        if clean and key == "graph_1":
            # find the indices of the values that are >0
            idx = torch.where(val > 0)[0]
            # select only the values that are >0
            val = val[idx]
            # calculate the mean and std
            node_mean_tensors[key] = torch.mean(val, dim=0)
            node_std_tensors[key] = torch.std(val, dim=0)

        else:
            node_mean_tensors[key] = torch.mean(val, dim=0)
            node_std_tensors[key] = torch.std(val, dim=0)

    return node_mean_tensors, node_std_tensors

def hetero_graph_normalization(dataset, node_mean_tensors, node_std_tensors):

    """ Normalizes the node features in the dataset

    Paramters
    ---------
    dataset: An iterable of torch_geometric.data.Data objects
    node_mean_tensors: A dictionary of the mean of the node features
    node_std_tensors: A dictionary of the std of the node features

    Returns
    -------
    """

    for key in node_std_tensors.keys():
        node_std_tensors[key] = torch.where(node_std_tensors[key] == 0, torch.ones_like(node_std_tensors[key]), node_std_tensors[key])

    for data in dataset:
        for key in data.x_dict.keys():
            data.x_dict[key] -=  node_mean_tensors[key]
            # avoid division by zero
            #data.x_dict[key] /= node_std_tensors[key] + 1e-8
            data.x_dict[key] /= node_std_tensors[key]

def add_virtual_node(dataset):
    """ Adds a virtual node to the dataset, that is connected to all other nodes and has 0 embeddings

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

        
            data[("global", "to", key)].edge_index = torch.zeros((2, data.x_dict[key].shape[0])).long()
            data[("global", "to", key)].edge_index[0,:] = 0
            data[("global", "to", key)].edge_index[1,:] = torch.arange(data.x_dict[key].shape[0])

            data[(key, "rev_to", "global")].edge_index = torch.zeros((2, data.x_dict[key].shape[0])).long()
            data[(key, "rev_to", "global")].edge_index[0,:] = torch.arange(data.x_dict[key].shape[0])
            data[(key, "rev_to", "global")].edge_index[1,:] = 0

def add_node_features(dataset, node_types):
    """ Adds the node degrees to the node features in the dataset

    Paramters
    ---------
    dataset: An iterable of torch_geometric.data.Data objects
    node_types: A list of the node types in the dataset

    Returns
    -------
    """
    for data in dataset:
        if len(node_types) == 2:
            heter_node_degrees_1, heter_node_degrees_2 = calculate_node_degrees(data.edge_index_dict[(node_types[0], "to", node_types[1])], data.x_dict[node_types[0]].shape[0], data.x_dict[node_types[1]].shape[0])
            heter_node_degrees_1 = heter_node_degrees_1.unsqueeze(1).float()
            heter_node_degrees_2 = heter_node_degrees_2.unsqueeze(1).float()

        for i, key in enumerate(node_types):
            node_degrees = calculate_node_degrees(data.edge_index_dict[(key, "to", key)], data.x_dict[key].shape[0])
            node_degrees = node_degrees.unsqueeze(1).float()
            if len(node_types) == 2:
                if i == 0:
                    data[key].x = torch.cat((data.x_dict[key], node_degrees, heter_node_degrees_1), dim=1)
                else:
                    data[key].x = torch.cat((data.x_dict[key], node_degrees, heter_node_degrees_2), dim=1)
            else:
                data[key].x = torch.cat((data.x_dict[key], node_degrees), dim=1)


def calculate_node_degrees(edge_index, num_nodes, num_nodes_2=None):
    """ Calculates the degrees of the nodes in the graph

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
        degrees = torch.bincount(edge_index[0], minlength= num_nodes)
        degrees_2 = torch.bincount(edge_index[1], minlength= num_nodes_2)

        return degrees, degrees_2
    else:
        degrees = torch.bincount(edge_index[0], minlength= num_nodes)
        degrees += torch.bincount(edge_index[1], minlength= num_nodes)

        return degrees
    

def remove_label_noise(dataset, label_noise_dict):
    """ Removes the nodes with label noise from the dataset

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


def adjust_data_for_split(cv_dataset, final_test_dataset, split, faz = False):
    """ Adjusts the datasets for the split, set the split, impute, normalize and add node features

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

    train_dataset = copy.deepcopy(cv_dataset)
    val_dataset = copy.deepcopy(cv_dataset)
    val_dataset.set_split("val", split)
    train_dataset.set_split("train",split)
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

    # extract normalization parameters
    node_mean_tensors, node_std_tensors = hetero_graph_normalization_params(train_dataset, clean= False)

    # data normalization
    hetero_graph_normalization(train_dataset, node_mean_tensors, node_std_tensors)
    hetero_graph_normalization(val_dataset, node_mean_tensors, node_std_tensors)
    hetero_graph_normalization(test_dataset, node_mean_tensors, node_std_tensors)

    return train_dataset, val_dataset, test_dataset


def eliminate_features(included_features, features_label_dict, datasets):
    """ Eliminates the features that are not included in the included_features dictionary

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
                    data[key].x = torch.cat([data[key].x[:, :idx], data[key].x[:, idx+1:]], dim = 1)
