import torch
import numpy as np
from tqdm import tqdm


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

    comb_feature_dict = {}
    for key, val in g_feature_dict.items():
        comb_feature_dict[key] = (np.concatenate([val["graph_1"],val["graph_2"], np.array(faz_feature_dict[key]), seg_feature_dict[key]], axis = 0), int(dataset.hetero_graphs[key].y[0]))
    return comb_feature_dict



def add_global_node(dataset):

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



def hetero_graph_cleanup(dataset):

    relevant_nodes = ["graph_1", "graph_2"]
    for data in tqdm(dataset):
        # get the indices of the nodes that have nan or inf values
        del_nodes_dict = {}
        new_idx_dict = {"graph_1": {}, "graph_2": {}}
        for key in relevant_nodes: #data.x_dict.items():
            del_nodes = torch.where(torch.isnan(data.x_dict[key]) | torch.isinf(data.x_dict[key]))[0]

            if key == "graph_1":
                # if a node has feature values <0, then remove it
                idx = torch.where(data.x_dict[key] < 0)[0]
                del_nodes = torch.cat([del_nodes, idx], dim=0)
            elif key == "graph_2":
                # if the 3rd feature is <10, then remove it
                idx = torch.where(data.x_dict[key][:,2] <= 10)[0]
                del_nodes = torch.cat([del_nodes, idx], dim=0)

            # remove duplicates
            del_nodes = torch.unique(del_nodes)
            del_nodes_dict[key] = del_nodes


            # print the number of nodes that are removed
            #print(f"Number of nodes removed from {key}: {len(del_nodes)}")
            old_node_num = data.x_dict[key].shape[0]

            # remove the nodes from the x_dict, select all indices that are not in del_nodes
            keep_node_mask = ~torch.isin(torch.arange(old_node_num), del_nodes)
            data[key].x = data.x_dict[key][keep_node_mask, :]
            # remove the nodes from the pos_dict
            data[key].pos = data.pos_dict[key][keep_node_mask, :]

            # create a dict that maps the old node indices to the new node indices
            # the new node indices are shifted by the number of nodes that are removed wtih a lower index
            # e.g. if node 0 and 1 are removed, then the new node 0 is the old node 2
            # and the new node 1 is the old node 3
            # iterate over the old nodes
            #for i in range(old_node_num):
            #    # if the node is removed the new idx is None
            #    if i in del_nodes:
            #        new_idx_dict[key][i] = None
            #    else:
            #        # count the number of nodes that are removed with a lower index
            #        # and shift the new index by this number
            #        new_idx_dict[key][i] = torch.tensor(i - torch.sum(del_nodes < i))   
            # no visible speed up
            new_idx_dict[key] = {i.item(): torch.tensor(new_i) for new_i, i in enumerate(torch.where(keep_node_mask)[0])}


        # remove the nodes from the edge_index
        # start by removing edges between same type nodes
        
        for key in relevant_nodes:
            # remove edges between same type nodes
            # get the indices of the edges that have to be removed
            del_mask = torch.any(torch.isin(data.edge_index_dict[(key, "to", key)], del_nodes_dict[key]), dim=0)
            # print number of edges that are removed
            #print(f"Number of edges removed from {key} to {key}: {torch.sum(del_mask)}")
            # remove the edges
            data[key, "to", key].edge_index = data.edge_index_dict[(key, "to", key)][:, ~del_mask]

            # update the indices of the edges
            # iterate over the edges
            for i in range(data.edge_index_dict[(key, "to", key)].shape[1]):
                # update the indices of the edges
                data.edge_index_dict[(key, "to", key)][0, i] = new_idx_dict[key][data.edge_index_dict[(key, "to", key)][0, i].item()]
                data.edge_index_dict[(key, "to", key)][1, i] = new_idx_dict[key][data.edge_index_dict[(key, "to", key)][1, i].item()]


        # remove edges between graph_1 and graph_2
        # get the indices of the edges that have to be removed
        
        del_mask_12_p1 = torch.isin(data.edge_index_dict[(relevant_nodes[0], "to", relevant_nodes[1])][0], del_nodes_dict[relevant_nodes[0]])
        del_mask_12_p2 = torch.isin(data.edge_index_dict[(relevant_nodes[0], "to", relevant_nodes[1])][1], del_nodes_dict[relevant_nodes[1]])

        del_mask_12 = del_mask_12_p1 | del_mask_12_p2
        # print number of edges that are removed
        #print(f"Number of edges removed from {relevant_nodes[0]} to {relevant_nodes[1]}: {torch.sum(del_mask_12)}")

        # remove the edges
        data[relevant_nodes[0], "to", relevant_nodes[1]].edge_index = data.edge_index_dict[(relevant_nodes[0], "to", relevant_nodes[1])][:, ~del_mask_12]
        # remove the same edges from the other direction
        data[relevant_nodes[1], "rev_to", relevant_nodes[0]].edge_index = data.edge_index_dict[(relevant_nodes[1], "rev_to", relevant_nodes[0])][:, ~del_mask_12]

        # update the indices of the edges
        # iterate over the edges
        for i in range(data.edge_index_dict[(relevant_nodes[0], "to", relevant_nodes[1])].shape[1]):
            # update the indices of the edges
            data.edge_index_dict[(relevant_nodes[0], "to", relevant_nodes[1])][0, i] = new_idx_dict[relevant_nodes[0]][data.edge_index_dict[(relevant_nodes[0], "to", relevant_nodes[1])][0, i].item()]
            data.edge_index_dict[(relevant_nodes[0], "to", relevant_nodes[1])][1, i] = new_idx_dict[relevant_nodes[1]][data.edge_index_dict[(relevant_nodes[0], "to", relevant_nodes[1])][1, i].item()]
            data.edge_index_dict[(relevant_nodes[1], "rev_to", relevant_nodes[0])][0, i] = new_idx_dict[relevant_nodes[1]][data.edge_index_dict[(relevant_nodes[1], "rev_to", relevant_nodes[0])][0, i].item()]
            data.edge_index_dict[(relevant_nodes[1], "rev_to", relevant_nodes[0])][1, i] = new_idx_dict[relevant_nodes[0]][data.edge_index_dict[(relevant_nodes[1], "rev_to", relevant_nodes[0])][1, i].item()]


        # remove edges between graph_1 and faz
        # get the indices of the edges that have to be removed
        del_mask_1f = torch.isin(data.edge_index_dict[(relevant_nodes[0], "to", "faz")][0], del_nodes_dict[relevant_nodes[0]])

        # print number of edges that are removed
        #print(f"Number of edges removed from {relevant_nodes[0]} to faz: {torch.sum(del_mask_1f)}")

        # remove the edges
        data[relevant_nodes[0], "to", "faz"].edge_index = data.edge_index_dict[(relevant_nodes[0], "to", "faz")][:, ~del_mask_1f]
        # remove the same edges from the other direction
        data["faz", "rev_to", relevant_nodes[0]].edge_index = data.edge_index_dict[("faz", "rev_to", relevant_nodes[0])][:, ~del_mask_1f]

        # update the indices of the edges
        # iterate over the edges
        for i in range(data.edge_index_dict[(relevant_nodes[0], "to", "faz")].shape[1]):
            # update the indices of the edges, the index of the faz node is not changed
            data.edge_index_dict[(relevant_nodes[0], "to", "faz")][0, i] = new_idx_dict[relevant_nodes[0]][data.edge_index_dict[(relevant_nodes[0], "to", "faz")][0, i].item()]
            data.edge_index_dict[("faz", "rev_to", relevant_nodes[0])][1, i] = new_idx_dict[relevant_nodes[0]][data.edge_index_dict[("faz", "rev_to",  relevant_nodes[0])][1, i].item()]




        # remove edges between graph_2 and faz
        # get the indices of the edges that have to be removed
        del_mask_2f = torch.isin(data.edge_index_dict[("faz", "to", relevant_nodes[1])][1], del_nodes_dict[relevant_nodes[1]])

        # print number of edges that are removed
        #print(f"Number of edges removed from faz to {relevant_nodes[1]}: {torch.sum(del_mask_2f)}")

        # remove the edges
        data["faz", "to", relevant_nodes[1]].edge_index = data.edge_index_dict[("faz", "to", relevant_nodes[1])][:, ~del_mask_2f]
        # remove the same edges from the other direction
        data[relevant_nodes[1], "rev_to", "faz"].edge_index = data.edge_index_dict[(relevant_nodes[1], "rev_to", "faz")][:, ~del_mask_2f]

        # update the indices of the edges
        # iterate over the edges
        for i in range(data.edge_index_dict[("faz", "to", relevant_nodes[1])].shape[1]):
            # update the indices of the edges, the index of the faz node is not changed
            data.edge_index_dict[("faz", "to", relevant_nodes[1])][1, i] = new_idx_dict[relevant_nodes[1]][data.edge_index_dict[("faz", "to", relevant_nodes[1])][1, i].item()]
            data.edge_index_dict[(relevant_nodes[1], "rev_to", "faz")][0, i] = new_idx_dict[relevant_nodes[1]][data.edge_index_dict[(relevant_nodes[1], "rev_to", "faz")][0, i].item()]


    return dataset



def hetero_graph_cleanup_multi(dataset):

    # replace multiprocessing with torch.multiprocessing
    import torch.multiprocessing as mp
    torch.multiprocessing.set_sharing_strategy('file_system')
    num_processes = 16  # Use all available CPU cores, except one
    with mp.Pool(num_processes) as pool:
        updated_dataset = list(tqdm(pool.imap(process_data_wrapper, dataset), total=len(dataset)))
    
    return updated_dataset

def process_data_wrapper(data):
    return heter_graph_cleanup_singel_data(data)


def heter_graph_cleanup_singel_data(data):
    relevant_nodes = ["graph_1", "graph_2"]
    # get the indices of the nodes that have nan or inf values
    del_nodes_dict = {}
    new_idx_dict = {"graph_1": {}, "graph_2": {}}
    for key in relevant_nodes: #data.x_dict.items():
        del_nodes = torch.where(torch.isnan(data.x_dict[key]) | torch.isinf(data.x_dict[key]))[0]

        if key == "graph_1":
            # if a node has feature values <0, then remove it
            idx = torch.where(data.x_dict[key] < 0)[0]
            del_nodes = torch.cat([del_nodes, idx], dim=0)
        elif key == "graph_2":
            # if the 3rd feature is <10, then remove it
            idx = torch.where(data.x_dict[key][:,2] <= 10)[0]
            del_nodes = torch.cat([del_nodes, idx], dim=0)

        # remove duplicates
        del_nodes = torch.unique(del_nodes)
        del_nodes_dict[key] = del_nodes


        # print the number of nodes that are removed
        #print(f"Number of nodes removed from {key}: {len(del_nodes)}")
        old_node_num = data.x_dict[key].shape[0]

        # remove the nodes from the x_dict, select all indices that are not in del_nodes
        keep_node_mask = ~torch.isin(torch.arange(old_node_num), del_nodes)
        data[key].x = data.x_dict[key][keep_node_mask, :]
        # remove the nodes from the pos_dict
        data[key].pos = data.pos_dict[key][keep_node_mask, :]

        # create a dict that maps the old node indices to the new node indices
        # the new node indices are shifted by the number of nodes that are removed wtih a lower index
        # e.g. if node 0 and 1 are removed, then the new node 0 is the old node 2
        # and the new node 1 is the old node 3
        # iterate over the old nodes
        #for i in range(old_node_num):
        #    # if the node is removed the new idx is None
        #    if i in del_nodes:
        #        new_idx_dict[key][i] = None
        #    else:
        #        # count the number of nodes that are removed with a lower index
        #        # and shift the new index by this number
        #        new_idx_dict[key][i] = torch.tensor(i - torch.sum(del_nodes < i))   
        # no visible speed up
        new_idx_dict[key] = {i.item(): torch.tensor(new_i) for new_i, i in enumerate(torch.where(keep_node_mask)[0])}


    # remove the nodes from the edge_index
    # start by removing edges between same type nodes

    for key in relevant_nodes:
        # remove edges between same type nodes
        # get the indices of the edges that have to be removed
        del_mask = torch.any(torch.isin(data.edge_index_dict[(key, "to", key)], del_nodes_dict[key]), dim=0)
        # print number of edges that are removed
        #print(f"Number of edges removed from {key} to {key}: {torch.sum(del_mask)}")
        # remove the edges
        data[key, "to", key].edge_index = data.edge_index_dict[(key, "to", key)][:, ~del_mask]

        # update the indices of the edges
        # iterate over the edges
        for i in range(data.edge_index_dict[(key, "to", key)].shape[1]):
            # update the indices of the edges
            data.edge_index_dict[(key, "to", key)][0, i] = new_idx_dict[key][data.edge_index_dict[(key, "to", key)][0, i].item()]
            data.edge_index_dict[(key, "to", key)][1, i] = new_idx_dict[key][data.edge_index_dict[(key, "to", key)][1, i].item()]


    # remove edges between graph_1 and graph_2
    # get the indices of the edges that have to be removed

    del_mask_12_p1 = torch.isin(data.edge_index_dict[(relevant_nodes[0], "to", relevant_nodes[1])][0], del_nodes_dict[relevant_nodes[0]])
    del_mask_12_p2 = torch.isin(data.edge_index_dict[(relevant_nodes[0], "to", relevant_nodes[1])][1], del_nodes_dict[relevant_nodes[1]])

    del_mask_12 = del_mask_12_p1 | del_mask_12_p2
    # print number of edges that are removed
    #print(f"Number of edges removed from {relevant_nodes[0]} to {relevant_nodes[1]}: {torch.sum(del_mask_12)}")

    # remove the edges
    data[relevant_nodes[0], "to", relevant_nodes[1]].edge_index = data.edge_index_dict[(relevant_nodes[0], "to", relevant_nodes[1])][:, ~del_mask_12]
    # remove the same edges from the other direction
    data[relevant_nodes[1], "rev_to", relevant_nodes[0]].edge_index = data.edge_index_dict[(relevant_nodes[1], "rev_to", relevant_nodes[0])][:, ~del_mask_12]

    # update the indices of the edges
    # iterate over the edges
    for i in range(data.edge_index_dict[(relevant_nodes[0], "to", relevant_nodes[1])].shape[1]):
        # update the indices of the edges
        data.edge_index_dict[(relevant_nodes[0], "to", relevant_nodes[1])][0, i] = new_idx_dict[relevant_nodes[0]][data.edge_index_dict[(relevant_nodes[0], "to", relevant_nodes[1])][0, i].item()]
        data.edge_index_dict[(relevant_nodes[0], "to", relevant_nodes[1])][1, i] = new_idx_dict[relevant_nodes[1]][data.edge_index_dict[(relevant_nodes[0], "to", relevant_nodes[1])][1, i].item()]
        data.edge_index_dict[(relevant_nodes[1], "rev_to", relevant_nodes[0])][0, i] = new_idx_dict[relevant_nodes[1]][data.edge_index_dict[(relevant_nodes[1], "rev_to", relevant_nodes[0])][0, i].item()]
        data.edge_index_dict[(relevant_nodes[1], "rev_to", relevant_nodes[0])][1, i] = new_idx_dict[relevant_nodes[0]][data.edge_index_dict[(relevant_nodes[1], "rev_to", relevant_nodes[0])][1, i].item()]


    # remove edges between graph_1 and faz
    # get the indices of the edges that have to be removed
    del_mask_1f = torch.isin(data.edge_index_dict[(relevant_nodes[0], "to", "faz")][0], del_nodes_dict[relevant_nodes[0]])

    # print number of edges that are removed
    #print(f"Number of edges removed from {relevant_nodes[0]} to faz: {torch.sum(del_mask_1f)}")

    # remove the edges
    data[relevant_nodes[0], "to", "faz"].edge_index = data.edge_index_dict[(relevant_nodes[0], "to", "faz")][:, ~del_mask_1f]
    # remove the same edges from the other direction
    data["faz", "rev_to", relevant_nodes[0]].edge_index = data.edge_index_dict[("faz", "rev_to", relevant_nodes[0])][:, ~del_mask_1f]

    # update the indices of the edges
    # iterate over the edges
    for i in range(data.edge_index_dict[(relevant_nodes[0], "to", "faz")].shape[1]):
        # update the indices of the edges, the index of the faz node is not changed
        data.edge_index_dict[(relevant_nodes[0], "to", "faz")][0, i] = new_idx_dict[relevant_nodes[0]][data.edge_index_dict[(relevant_nodes[0], "to", "faz")][0, i].item()]
        data.edge_index_dict[("faz", "rev_to", relevant_nodes[0])][1, i] = new_idx_dict[relevant_nodes[0]][data.edge_index_dict[("faz", "rev_to",  relevant_nodes[0])][1, i].item()]




    # remove edges between graph_2 and faz
    # get the indices of the edges that have to be removed
    del_mask_2f = torch.isin(data.edge_index_dict[("faz", "to", relevant_nodes[1])][1], del_nodes_dict[relevant_nodes[1]])

    # print number of edges that are removed
    #print(f"Number of edges removed from faz to {relevant_nodes[1]}: {torch.sum(del_mask_2f)}")

    # remove the edges
    data["faz", "to", relevant_nodes[1]].edge_index = data.edge_index_dict[("faz", "to", relevant_nodes[1])][:, ~del_mask_2f]
    # remove the same edges from the other direction
    data[relevant_nodes[1], "rev_to", "faz"].edge_index = data.edge_index_dict[(relevant_nodes[1], "rev_to", "faz")][:, ~del_mask_2f]

    # update the indices of the edges
    # iterate over the edges
    for i in range(data.edge_index_dict[("faz", "to", relevant_nodes[1])].shape[1]):
        # update the indices of the edges, the index of the faz node is not changed
        data.edge_index_dict[("faz", "to", relevant_nodes[1])][1, i] = new_idx_dict[relevant_nodes[1]][data.edge_index_dict[("faz", "to", relevant_nodes[1])][1, i].item()]
        data.edge_index_dict[(relevant_nodes[1], "rev_to", "faz")][0, i] = new_idx_dict[relevant_nodes[1]][data.edge_index_dict[(relevant_nodes[1], "rev_to", "faz")][0, i].item()]


    return data



def add_centerline_statistics(dataset, image_folder, vvg_folder, seg_size):
    import os
    import json
    import pandas as pd
    from PIL import Image

    # read all the json/json.gz files in the vvg folder
    vvg_files = os.listdir(vvg_folder)
    vvg_files = [file for file in vvg_files if file.endswith(".json") or file.endswith(".json.gz")]

    # read all the images in the image folder
    image_files = os.listdir(image_folder)
    image_files = [file for file in image_files if file.endswith(".png")]

    # match 


    # iterate over the dataset
    for data in dataset:
        # get the id of the graph
        graph_id = data.graph_id
        # find the corresponding json file
        json_file = [file for file in vvg_files if graph_id in file][0]
        # find the corresponding image file
        image_file = [file for file in image_files if graph_id in file][0]
        # load the json file into a df
        df = vvg_to_df(os.path.join(vvg_folder, json_file))
        # load the image
        image = Image.open(os.path.join(image_folder, image_file))
        # turn the iamge into a numpy array
        image = np.array(image)
        # get the size of the image
        image_size = image.shape[0]
        # get the ratio between the image size and the seg size
        ratio = image_size / seg_size
        avg_intensities = cl_pos_to_intensities(df, image, ratio)

        print(avg_intensities.shape)
        print(data["graph_1"].x.shape)

        # add the avg intensities to the data
        try:
            data["graph_1"].x = torch.cat([data["graph_1"].x, torch.tensor(avg_intensities)], dim=1)
        except RuntimeError:
            print("RuntimeError")




        

def vvg_to_df(vvg_path):
    # Opening JSON file
    import json
    import pandas as pd
    import gzip
    if vvg_path[-3:] == ".gz":
        with gzip.open(vvg_path, "rt") as gzipped_file:
            # Read the decompressed JSON data
            json_data = gzipped_file.read()
            data = json.loads(json_data)

    else:
        f = open(vvg_path)
        data = json.load(f)
        f.close()

    id_col = []
    pos_col = []
    node1_col = []
    node2_col = []

    for i in data["graph"]["edges"]:
        positions = []
        id_col.append(i["id"])
        node1_col.append(i["node1"])
        node2_col.append(i["node2"])

        try:
            i["skeletonVoxels"]
        except KeyError:
            pos_col.append(None)
            #print("skeletonVoxels KeyError")
            continue
        for j in i["skeletonVoxels"]:
            positions.append(np.array(j["pos"]))
        pos_col.append(positions)


    d = {'id_col': id_col,'pos_col' : pos_col, "node1_col" : node1_col, "node2_col" : node2_col}
    df = pd.DataFrame(d)
    df.set_index('id_col')
    return df


def cl_pos_to_intensities(cl_pos_df, image, ratio):
    import numpy as np

    # create a df that contains the intensities of the centerline
    # iterate over the rows of the df
    avg_int = np.zeros((len(cl_pos_df), 4))
    for i in range(len(cl_pos_df)):
        # get the positions of the centerline
        positions = cl_pos_df.iloc[i]["pos_col"]
        # get the positions in the image
        try:
            positions = np.array([np.array([int(pos[0] * ratio), int(pos[1] * ratio), int(pos[2] * ratio)]) for pos in positions])
        except TypeError:
            avg_int[i] = None
            continue

        # get the intensities of the centerline
        intensities = image[positions[:,0], positions[:,1], positions[:,2]]

        # calculate the average intensity
        avg_int[i,0] = np.mean(intensities)
        avg_int[i,1] = np.std(intensities)
        # quantiles
        avg_int[i,2] = np.quantile(intensities, 0.25)
        avg_int[i,3] = np.quantile(intensities, 0.75)
        

    return avg_int









def add_centerline_statistics_multi(dataset, image_folder, vvg_folder, seg_size):

    import os
    # replace multiprocessing with torch.multiprocessing
    import torch.multiprocessing as mp
    torch.multiprocessing.set_sharing_strategy('file_system')


    vvg_files = os.listdir(vvg_folder)
    vvg_files = [file for file in vvg_files if file.endswith(".json") or file.endswith(".json.gz")]

    image_files = os.listdir(image_folder)
    image_files = [file for file in image_files if file.endswith(".png")]

    # match the graphs with corresponding json and image files
    # iterate over the dataset

    matches = []

    for data in dataset:
        # get the id of the graph
        graph_id = data.graph_id
        # find the corresponding json file
        json_file = [file for file in vvg_files if graph_id in file][0]
        # find the corresponding image file
        image_file = [file for file in image_files if graph_id in file][0]

        # add folder to the file names
        json_file = os.path.join(vvg_folder, json_file)
        image_file = os.path.join(image_folder, image_file)

        matches.append((data, json_file, image_file, seg_size))

    with mp.Pool(16) as pool:
        updated_dataset = list(tqdm(pool.imap(process_data_cl, matches), total=len(dataset)))

    return updated_dataset






def process_data_cl(matched_list):
    # unpack the tuple
    from PIL import Image

    data, json_file, image_file, seg_size = matched_list
    df = vvg_to_df(json_file)
    # load the image
    image = Image.open(image_file)
    # turn the iamge into a numpy array
    image = np.array(image)
    # get the size of the image
    image_size = image.shape[0]
    # get the ratio between the image size and the seg size
    ratio = image_size / seg_size
    avg_intensities = cl_pos_to_intensities(df, image, ratio)

    # add the avg intensities to the data
    try:
        data["graph_1"].x = torch.cat([data["graph_1"].x, torch.tensor(avg_intensities, dtype= data["graph_1"].x.dtype)], dim=1)
    except RuntimeError:
        print("RuntimeError")

    return data



def check_centerline_on_image(dataset, image_folder, vvg_folder, seg_size, save_image_path):
    import os
    from PIL import Image

    vvg_files = os.listdir(vvg_folder)
    vvg_files = [file for file in vvg_files if file.endswith(".json") or file.endswith(".json.gz")]

    image_files = os.listdir(image_folder)
    image_files = [file for file in image_files if file.endswith(".png")]

    # match the graphs with corresponding json and image files
    # iterate over the dataset

    matches = []

    for data in dataset:
        # get the id of the graph
        graph_id = data.graph_id
        # find the corresponding json file
        json_file = [file for file in vvg_files if graph_id in file][0]
        # find the corresponding image file
        image_file = [file for file in image_files if graph_id in file][0]

        # add folder to the file names
        json_file = os.path.join(vvg_folder, json_file)
        image_file = os.path.join(image_folder, image_file)

        matches.append((data, json_file, image_file, seg_size))

    data, json_file, image_file, seg_size = matches[0]
    df = vvg_to_df(json_file)
    # load the image
    image = Image.open(image_file)
    # turn the iamge into a numpy array
    image = np.array(image)
    # get the size of the image
    image_size = image.shape[0]
    # get the ratio between the image size and the seg size
    ratio = image_size / seg_size


    for i in range(len(df)):
        # get the positions of the centerline
        positions = df.iloc[i]["pos_col"]
        # get the positions in the image
        try:
            positions = np.array([np.array([int(pos[0] * ratio), int(pos[1] * ratio), int(pos[2] * ratio)]) for pos in positions])
        except TypeError:
            continue

        # draw the centerline on the image
        try:
            image[positions[:,0], positions[:,1], positions[:,2]] = 255
        except IndexError:
            image[positions[:,0], positions[:,1]] = 0


    # save the image
    image = Image.fromarray(image)
    image.save(save_image_path)
