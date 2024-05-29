from scipy.spatial import cKDTree
import torch
import numpy as np


class Baseline_Lookup():
    """
    This class takes a list of datasets, where every single datapoint is a heterogeneous graph.
    Each node of each node type is then stored in a search tree based on the nodes position in the graph.

    Then based on a positional input it is poosible to query the k-nearst neighbors for a node of a specific type.
    """

    def __init__(self, dataset_list, node_types, device = "cpu"):
        self.dataset_list = dataset_list

        # get the number of nodes for each type
        self.node_types = node_types
        self.num_nodes = {}
        for node_type in node_types:
            self.num_nodes[node_type] = 0

        # get the feature length for each node type
        self.nodes_feature_length = {}
        for node_type in node_types:
            self.nodes_feature_length[node_type] = dataset_list[0][0][node_type].x.shape[1]

    
        for dataset in dataset_list:
            for data in dataset:
                for key in node_types:
                    self.num_nodes[key] += data[key].pos.shape[0]

        # create matrix with all nodes
        self.nodes = {}
        self.node_pos = {}
        for node_type in node_types:
            self.nodes[node_type] = torch.zeros((self.num_nodes[node_type], self.nodes_feature_length[node_type]), device = device)
            self.node_pos[node_type] = torch.zeros((self.num_nodes[node_type], 2))

        # iterate over all datasets and add the nodes to the matrix
        current_node = {}
        for node_type in node_types:
            current_node[node_type] = 0
        for dataset in dataset_list:
            for data in dataset:
                for key in node_types:
                    self.nodes[key][current_node[key]:current_node[key] + data[key].pos.shape[0], :] = data[key].x
                    self.node_pos[key][current_node[key]:current_node[key] + data[key].pos.shape[0], :] = data[key].pos
                    current_node[key] += data[key].pos.shape[0]


        # create search trees for each node type
        self.trees = {}
        for node_type in node_types:
            self.trees[node_type] = cKDTree(self.node_pos[node_type].numpy())
        

    def get_k_nearest_neighbors(self, node_type, pos, k):
        """
        This function returns the k nearest neighbors of a node of a specific type.
        """
        return self.trees[node_type].query(pos, k=k)
    
    
    def get_k_nearest_neighbors_feaures(self, node_type, pos, k):
        """
        This function returns the k nearest neighbors of a node of a specific type.
        """
        dist, ind = self.trees[node_type].query(pos, k=k)
        return self.nodes[node_type][ind, :]
    

    def get_k_nearst_neighbors_avg_features(self, node_type, pos, k):

        dist, ind = self.trees[node_type].query(pos, k=k)
        avg = self.nodes[node_type][ind, :].mean(axis=1) # axis = 1 for featurewise mean over all neighbors
        # axis = 0 are all the baseline nodes
        # axis = 1 are the neighbors of the baseline nodes
        # axis = 2 are the features of the nodes
        #median, idcs = self.nodes[node_type][ind, :].median(axis=1)

        # set all values >1 to 1
        avg[avg > 1] = 1
        # set all values <-1 to -1
        avg[avg < -1] = -1

        return avg

    
    def get_k_nearst_neighbors_median_features(self, node_type, pos, k):
            
        dist, ind = self.trees[node_type].query(pos, k=k)
        median = torch.median(self.nodes[node_type][ind, :], dim=1).values
        # axis = 0 are all the baseline nodes
        # axis = 1 are the neighbors of the baseline nodes
        # axis = 2 are the features of the nodes

        # set all values >1 to 1
        #median[median > 1] = 1
        # set all values <-1 to -1
        #median[median < -1] = -1
    
        return median
    

    def get_k_nearst_neighbors_avg_features_quantile(self, node_type, pos, k, lower_quantile = 0.1, upper_quantile = 0.9):

        # query the spatially k nearest neighbors for each node
        dist, ind = self.trees[node_type].query(pos, k=k)
        neighbors = self.nodes[node_type][ind, :]
        # axis = 0 are all the baseline nodes
        # axis = 1 are the neighbors of the baseline nodes
        # axis = 2 are the features of the nodes

        # calculate the quantiles for each node
        lower_quantiles = torch.quantile(neighbors, lower_quantile, dim=1)
        upper_quantiles = torch.quantile(neighbors, upper_quantile, dim=1)

        # use the whole data range if the quantiles are the same
        upper_quantiles[lower_quantiles == upper_quantiles] = 1000
        lower_quantiles[lower_quantiles == upper_quantiles] = -1000

        # calculate the average of the neighbors that are in the quantile range
        in_quantile_range = (neighbors > lower_quantiles[:, None, :]) & (neighbors < upper_quantiles[:, None, :])
        avg = torch.where(in_quantile_range, neighbors, torch.tensor(0.0, device=neighbors.device))
        avg = avg.mean(dim=1)

        # set all values >1 to 1 and all values <-1 to -1
        avg[avg > 1] = 1
        avg[avg < -1] = -1

        return avg


    def get_baselines(self, baseline_type, data, k=100):
        device = data.x_dict["graph_1"].device

        if baseline_type == "zero":
            baseline_graph_1 = torch.zeros_like(data.x_dict["graph_1"]).unsqueeze(0)
            baseline_graph_2 = torch.zeros_like(data.x_dict["graph_2"]).unsqueeze(0)
            baseline_faz = torch.zeros_like(data.x_dict["faz"], device= device).unsqueeze(0)

        elif baseline_type == "lookup_quantile": #get_k_nearst_neighbors_avg_features
            baseline_graph_1 = self.get_k_nearst_neighbors_avg_features_quantile("graph_1", data["graph_1"].pos.cpu().numpy(), k).unsqueeze(0)
            baseline_graph_2 = self.get_k_nearst_neighbors_avg_features_quantile("graph_2", data["graph_2"].pos.cpu().numpy(), k).unsqueeze(0)
            baseline_faz = torch.zeros_like(data.x_dict["faz"], device= device).unsqueeze(0)

        elif baseline_type == "lookup_median":
            baseline_graph_1 = self.get_k_nearst_neighbors_median_features("graph_1", data["graph_1"].pos.cpu().numpy(), k).unsqueeze(0)
            baseline_graph_2 = self.get_k_nearst_neighbors_median_features("graph_2", data["graph_2"].pos.cpu().numpy(), k).unsqueeze(0)
            baseline_faz = torch.zeros_like(data.x_dict["faz"], device= device).unsqueeze(0)
        
        elif baseline_type == "lookup_mean":
            baseline_graph_1 = self.get_k_nearst_neighbors_avg_features("graph_1", data["graph_1"].pos.cpu().numpy(), k).unsqueeze(0)
            baseline_graph_2 = self.get_k_nearst_neighbors_avg_features("graph_2", data["graph_2"].pos.cpu().numpy(), k).unsqueeze(0)
            baseline_faz = torch.zeros_like(data.x_dict["faz"], device= device).unsqueeze(0)


        else:
            raise ValueError("Baseline type not recognized")
        
        baselines = (baseline_graph_1, baseline_graph_2, baseline_faz)

        return baselines
