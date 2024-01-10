


def get_class_weights(train_labels, verbose = False):

    from collections import Counter

    # Calculate the label distribution
    label_distribution = Counter(train_labels)



    # If you want to use the label distribution for balanced weights in CrossEntropyLoss
    total_samples = len(train_labels)
    class_weights = [total_samples / (len(label_distribution) * count) for label, count in label_distribution.items()]

    if verbose:
        # Print or use the label distribution as needed
        print("Label distribution in the training set:")
        for label, count in label_distribution.items():
            print(f"Class {label}: {count} samples")

        print("Class weights for balanced CrossEntropyLoss:")
        for label, weight in zip(label_distribution.keys(), class_weights):
            print(f"Class {label}: Weight {weight}")

    return class_weights


def evaluate_cnn(model, dataloader):
    import torch
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    all_labels = []
    all_predictions = []

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predictions = torch.max(outputs, 1)
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predictions.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    balanced_accuracy = balanced_accuracy_score(all_labels, all_predictions)
    kappa = cohen_kappa_score(all_labels, all_predictions)

    res_dict = {"accuracy": accuracy, "balanced_accuracy": balanced_accuracy, "kappa": kappa}


    del inputs, labels, outputs, predictions
    torch.cuda.empty_cache()

    return accuracy, balanced_accuracy


def hetero_graph_imputation(dataset):
    import torch
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

#def heter_graph_global_node_norm_param(dataset):
#    import torch
#
#    for data in dataset:
#        res = torch.cat([data["global"].x, res]) if res is not None else data["global"].x
#
#    return torch.mean(res, dim=0), torch.std(res, dim=0)
                



def hetero_graph_normalization_params(train_dataset, clean = False):

    # iterable dataset
    try:
        iterable = train_dataset.hetero_graph_list
    except AttributeError:
        iterable = train_dataset

    import torch
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
            # if in the matrix there are values that are <=0 then we want to ignore them
            # and only use the values that are >0

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
    import torch

    for key in node_std_tensors.keys():
        node_std_tensors[key] = torch.where(node_std_tensors[key] == 0, torch.ones_like(node_std_tensors[key]), node_std_tensors[key])

    for data in dataset:
        for key in data.x_dict.keys():
            data.x_dict[key] -=  node_mean_tensors[key]
            # avoid division by zero
            #data.x_dict[key] /= node_std_tensors[key] + 1e-8
            data.x_dict[key] /= node_std_tensors[key]


# combined feature dict

def create_combined_feature_dict(g_feature_dict, faz_feature_dict, seg_feature_dict, dataset):
    import numpy as np
    comb_feature_dict = {}
    for key, val in g_feature_dict.items():
        comb_feature_dict[key] = (np.concatenate([val["graph_1"],val["graph_2"], np.array(faz_feature_dict[key]), seg_feature_dict[key]], axis = 0), int(dataset.hetero_graphs[key].y[0]))
    return comb_feature_dict



def add_global_node(dataset):
    import torch
    for data in dataset:
        global_features = []
        for node_type in data.x_dict.keys():
            node_num = data.x_dict[node_type].shape[0]
            edge_num = data.edge_index_dict[(node_type, "to", node_type)].shape[1]
            avg_deg = 2*edge_num/node_num
            global_features += [node_num, edge_num, avg_deg]

            edges = torch.zeros((2, node_num))
            edges[0,:] = 0
            edges[1,:] = torch.arange(node_num)
            edges = edges.long()
            data[("global", "to", node_type)].edge_index = edges


        heter_edge_num = data.edge_index_dict[("graph_1", "to", "graph_2")].shape[1]
        global_features += [heter_edge_num, data.eye]
        global_features = torch.tensor(global_features).unsqueeze(0)
        data["global"].x = global_features
        data[("global", "to", "global")].edge_index = torch.zeros((2,1)).long()


def add_node_features(dataset, node_types):
    import torch
    
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
    import torch
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
    

def remove_label_noise_list(dataset, label_noise_dict):

    # remove label noise from hetero_graph_list

    for key in label_noise_dict.keys():
        for idx in label_noise_dict[key]:
            dataset.hetero_graphs.pop(idx)

    dataset.hetero_graph_list = list(dataset.hetero_graphs.values())

    

def remove_label_noise(dataset, label_noise_dict):

    for key in label_noise_dict.keys():
        for idx in label_noise_dict[key]:
            dataset.hetero_graphs.pop(idx)

    dataset.hetero_graph_list = list(dataset.hetero_graphs.values())


def adjust_data_for_split(cv_dataset, final_test_dataset, split, faz = False):
    import copy
    import torch

    train_dataset = copy.deepcopy(cv_dataset)
    val_dataset = copy.deepcopy(cv_dataset)
    val_dataset.set_split("val", split)
    train_dataset.set_split("train",split)
    test_dataset = copy.deepcopy(final_test_dataset)
    print(f"train dataset: {len(train_dataset)}")
    print(f"val dataset: {len(val_dataset)}")
    print(f"test dataset: {len(test_dataset)}")

    # make sure that if data is read from file the classes are still the correct
    #train_dataset.update_class(octa_dr_dict)
    #val_dataset.update_class(octa_dr_dict)

    # data imputation and normalization
    hetero_graph_imputation(train_dataset)
    hetero_graph_imputation(val_dataset)
    hetero_graph_imputation(test_dataset)

    # check if the datasets have an faz node type
    node_types = ["graph_1", "graph_2"]
    #try:
    #    train_dataset[0]["faz"]
    #    node_types.append("faz")
    #except KeyError:
    #    pass
    #print(node_types)
    if faz:
        node_types.append("faz")

    add_node_features(train_dataset, node_types)
    add_node_features(val_dataset, node_types)
    add_node_features(test_dataset, node_types)


    node_mean_tensors, node_std_tensors = hetero_graph_normalization_params(train_dataset, clean= False)
    #print(node_mean_tensors)
    #print(node_std_tensors)
    #node_mean_tensors, node_std_tensors = hetero_graph_normalization_params(train_dataset, clean = True)
    #print(node_mean_tensors)
    #print(node_std_tensors)

    # why are they not the same?
    




    # save the normalization parameters
    #torch.save(node_mean_tensors, f"../../OCTA_500_RELEVANT/DCP_node_mean_tensors_faz_{faz}_from_CV.pt")
    #torch.save(node_std_tensors, f"../../OCTA_500_RELEVANT/DCP_node_std_tensors__faz_{faz}_from_CV.pt")
    #node_mean_tensors = torch.load(f"../{data_type}_node_mean_tensors_global_node_node_degs.pt")
    #node_std_tensors = torch.load(f"../{data_type}_node_std_tensors_global_node_node_degs.pt")


    hetero_graph_normalization(train_dataset, node_mean_tensors, node_std_tensors)
    hetero_graph_normalization(val_dataset, node_mean_tensors, node_std_tensors)
    hetero_graph_normalization(test_dataset, node_mean_tensors, node_std_tensors)

    return train_dataset, val_dataset, test_dataset
