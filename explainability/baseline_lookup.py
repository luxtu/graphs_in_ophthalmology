from scipy.spatial import cKDTree
import torch


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
        """
        This function returns the k nearest neighbors of a node of a specific type.
        """
        dist, ind = self.trees[node_type].query(pos, k=k)

        avg = self.nodes[node_type][ind, :].mean(axis=1)
        median, idcs = self.nodes[node_type][ind, :].median(axis=1)

        # set all values >1 to 1
        avg[avg > 1] = 1
        # set all values <-1 to -1
        avg[avg < -1] = -1

        # print the average for each feature
        print(avg.mean(axis=0))

        return avg

        
        

        



