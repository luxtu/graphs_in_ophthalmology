import numpy as np
import pandas as pd
import os 
import networkx as nx
import re
import torch
from torch_geometric.utils.convert import from_networkx



class_dict = {"NORMAL": 0, "DR": 1, "CNV" : 2, "AMD" : 3}
class_dict_2cls = {"NORMAL": 0, "DR": 1, "CNV" : 0, "AMD" : 0}


def create_graph(nodesFile, edgesFile):
    """ Creates an networkX undirected multigraph from provided csv files.
    Paramters
    ---------
    nodesFile: A csv file containing information about the nodes
    edgesFile: A csv file containing information about the edges
    id: an identifier to avoid ambiguous combinations of graphs
    Returns
    -------
    G: A networkX multigraph object
    """

    nodes = nodesFile
    edges = edgesFile
    # create undirected graph 
    G = nx.MultiGraph()

    for idxN, node in nodes.iterrows():
        G.add_node(idxN, x = (), pos = (float(node["pos_x"]),float(node["pos_y"]), float(node["pos_z"])))
    # add the edges
    for idxE, edge in edges.iterrows():
        G.add_edge(edge["node1id"],edge["node2id"], x = (edge[2:]))

    return G



def make_dual(G, include_orientation = True):
    """ Returns the dual graph of a given networkX graph. Encodes the position of the new nodes as the centers of the old nodes.
    Paramters
    ---------
    G: A networkX graph
    Returns
    D: The dual graph
    -------
    """
    D = nx.line_graph(G)
    dual_node_features ={}
    dual_node_centers = {}
    for edge in G.edges:

        
        posA = np.array(G.nodes[edge[0]]["pos"])
        posB = np.array(G.nodes[edge[1]]["pos"])
        edge_cent = (posA + posB) /2
        dual_node_centers[edge] = edge_cent


        if include_orientation:
            vec = posA-posB
            direction = (vec)/ np.linalg.norm(vec)

            if direction[0] < 0 or (direction[0] == 0 and direction[1] <0) or (direction[2] == -1):
                direction = direction*-1
                
            feat = np.concatenate((G.edges[edge]["x"], direction))
            dual_node_features[edge] = feat

        else:
            dual_node_features[edge] = G.edges[edge]["x"]

    
    nx.set_node_attributes(D, dual_node_features, name = "x")
    nx.set_node_attributes(D, dual_node_centers, name = "pos")

    return D


def add_structural_features(G, geom_data):

    sparse_adj_mat =nx.adjacency_matrix(G)
    
    # creates 4 more features
    degs_np = np.array(list(G.degree()))[:,1]
    adj2 = sparse_adj_mat**2
    adj3 = sparse_adj_mat**3
    adj4 = sparse_adj_mat**4

    # adding the features 
    new_x = np.zeros((geom_data.x.shape[0], geom_data.x.shape[1]+4))
    new_x[:,:geom_data.x.shape[1]] = geom_data.x
    new_x[:,geom_data.x.shape[1]] = degs_np
    new_x[:,geom_data.x.shape[1]+1] = adj2.diagonal()
    new_x[:,geom_data.x.shape[1]+2] = adj3.diagonal()
    new_x[:,geom_data.x.shape[1]+3] = adj4.diagonal()

    geom_data.x = torch.tensor(new_x)

    return geom_data


class RetinaLoader:


    def __init__(self, graph_path, label_file):
        self.graph_path = graph_path
        self.label_file = label_file
        self.label_data = self.read_labels(label_file)
        self.full_data = self.read_graphs(graph_path)
        self.dual_data_list = None
        self.data_list = None
    

    def read_labels(self, label_file):
        label_data = pd.read_excel(label_file, index_col= "ID")
        return label_data


    def read_graphs(self, graph_path):
        files = os.listdir(graph_path)

        node_dict = {}
        edge_dict = {}
        for file in sorted(files):
            idx = re.findall(r'\d+', str(file))[0]
            idx = int(idx)
            if "edges" in str(file):
                edge_dict[idx] = pd.read_csv(os.path.join(graph_path,file), sep = ";", index_col= "id")
            elif "nodes" in str(file):
                node_dict[idx] = pd.read_csv(os.path.join(graph_path,file), sep = ";", index_col= "id")

        graph_dict = {}
        for key in node_dict.keys():
            nodes = node_dict[key]
            edges = edge_dict[key]
            g = create_graph(nodes, edges)
            graph_dict[key] = g


        return graph_dict


    def get_dual_data_list(self, add_structural = False, two_cls = False, force = False):
        if self.dual_data_list is not None and not force:
            return self.dual_data_list

        else:
            data_list = []
            for key, val in self.full_data.items():
                disease = self.label_data.loc[key]["Disease"]
                if two_cls:
                    cls = class_dict_2cls[disease]
                else:
                    cls = class_dict[disease]
                d = make_dual(val, include_orientation= False)
                geom_data = from_networkx(d)
                if add_structural:
                    d = add_structural_features(d, geom_data)
                

                geom_data.y = cls
                geom_data.name = str(key)
                data_list.append(geom_data)
            self.dual_data_list = data_list
            return data_list




    def get_data_list(self):
        if self.data_list is not None:
            return self.data_list

        else:
            data_list = []
            for key, val in self.full_data.items():
                disease = self.label_data.loc[key]
                if disease == "NORMAL":
                    cls = 0
                else:
                    cls = 1
                geom_data = from_networkx(val)

                geom_data.y = cls
                geom_data.name = str(key)
                data_list.append(geom_data)
            self.data_list = data_list
            return data_list



