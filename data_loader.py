import numpy as np
import pandas as pd
import os
import networkx as nx
import re
import torch
from torch_geometric.utils.convert import from_networkx
from torch_geometric.transforms import line_graph
from torch_geometric.data import Data

import time

class_dict = {"NORMAL": 0, "DR": 1, "CNV": 2, "AMD": 3}
class_dict_2cls = {"NORMAL": 0, "DR": 1, "CNV": 0, "AMD": 0}


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
        G.add_node(idxN, x=(), pos=(float(node["pos_x"]), float(
            node["pos_y"]), float(node["pos_z"])))
    # add the edges
    for idxE, edge in edges.iterrows():
        G.add_edge(edge["node1id"], edge["node2id"], x=(edge[2:]))

    return G


def make_dual(G, include_orientation=True):
    """ Returns the dual graph of a given networkX graph. Encodes the position of the new nodes as the centers of the old nodes.
    Paramters
    ---------
    G: A networkX graph
    Returns
    D: The dual graph
    -------
    """
    D = nx.line_graph(G)
    dual_node_features = {}
    dual_node_centers = {}
    for edge in G.edges:

        posA = np.array(G.nodes[edge[0]]["pos"])
        posB = np.array(G.nodes[edge[1]]["pos"])
        edge_cent = (posA + posB) / 2
        dual_node_centers[edge] = edge_cent

        if include_orientation:
            vec = posA-posB
            direction = (vec) / np.linalg.norm(vec)

            if direction[0] < 0 or (direction[0] == 0 and direction[1] < 0) or (direction[2] == -1):
                direction = direction*-1

            feat = np.concatenate((G.edges[edge]["x"], direction))
            dual_node_features[edge] = feat

        else:
            dual_node_features[edge] = G.edges[edge]["x"]

    nx.set_node_attributes(D, dual_node_features, name="x")
    nx.set_node_attributes(D, dual_node_centers, name="pos")

    return D


def add_structural_features(G, geom_data):

    sparse_adj_mat = nx.adjacency_matrix(G)

    # creates 4 more features
    degs_np = np.array(list(G.degree()))[:, 1]
    adj2 = sparse_adj_mat**2
    adj3 = sparse_adj_mat**3
    adj4 = sparse_adj_mat**4

    # adding the features
    new_x = np.zeros((geom_data.x.shape[0], geom_data.x.shape[1]+4))
    new_x[:, :geom_data.x.shape[1]] = geom_data.x
    new_x[:, geom_data.x.shape[1]] = degs_np
    new_x[:, geom_data.x.shape[1]+1] = adj2.diagonal()
    new_x[:, geom_data.x.shape[1]+2] = adj3.diagonal()
    new_x[:, geom_data.x.shape[1]+3] = adj4.diagonal()

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
        label_data = pd.read_excel(label_file, index_col="ID")
        return label_data

    def read_graphs(self, graph_path):
        files = os.listdir(graph_path)

        node_dict = {}
        edge_dict = {}
        for file in sorted(files):
            idx = re.findall(r'\d+', str(file))[0]
            idx = int(idx)
            if "edges" in str(file):
                edge_dict[idx] = pd.read_csv(os.path.join(
                    graph_path, file), sep=";", index_col="id")
            elif "nodes" in str(file):
                node_dict[idx] = pd.read_csv(os.path.join(
                    graph_path, file), sep=";", index_col="id")

        graph_dict = {}
        for key in node_dict.keys():
            nodes = node_dict[key]
            edges = edge_dict[key]
            g = create_graph(nodes, edges)
            graph_dict[key] = g

        return graph_dict

    def get_dual_data_list(self, add_structural=False, two_cls=False, force=False):
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
                d = make_dual(val, include_orientation=False)
                geom_data = from_networkx(d)
                if add_structural:
                    d = add_structural_features(d, geom_data)

                geom_data.y = cls
                geom_data.name = str(key)
                data_list.append(geom_data)
            self.dual_data_list = data_list
            return data_list

    def get_data_list(self, two_cls=False, force=False):
        if self.data_list is not None and not force:
            return self.data_list

        else:
            data_list = []
            for key, val in self.full_data.items():
                disease = self.label_data.loc[key]["Disease"]
                if two_cls:
                    cls = class_dict_2cls[disease]
                else:
                    cls = class_dict[disease]

                # is this slow?
                t0 = time.time()
                geom_data = from_networkx(val)
                print(time.time()-t0)

                geom_data.y = cls
                geom_data.name = str(key)
                data_list.append(geom_data)
            self.data_list = data_list
            return data_list


class RetinaLoaderTorch:

    def __init__(self, graph_path, label_file, two_cls=False):

        self.two_cls = two_cls
        self.graph_path = graph_path
        self.label_file = label_file
        self.label_data = self.read_labels(label_file)
        self.full_data = self.read_graphs(graph_path)
        self.line_data = self.line_graphs_alt()

    def line_graphs(self):
        line_graph_dict = {}

        for key, value in self.full_data.items():
            lg_transform = line_graph.LineGraph()
            lg = lg_transform(value.clone())
            lg.num_nodes = lg.x.shape[0]
            lg.y = value.y
            line_graph_dict[key] = lg

        return line_graph_dict

    def line_graphs_alt(self):
        line_graph_dict = {}

        for key, value in self.full_data.items():

            # Determine the number of nodes in the original graph
            num_nodes = value.edge_index.max().item() + 1

            # Create the line graph nodes (edges in the original graph)
            line_graph_nodes = torch.arange(
                value.edge_index.size(1), dtype=torch.long)

            # Create the line graph edge indices (connecting edges that share a common node)
            line_graph_edge_indices = []
            for i in range(num_nodes):
                # Find the edges connected to node i
                connected_edges = (value.edge_index[0] == i) | (
                    value.edge_index[1] == i)
                connected_edges = connected_edges.nonzero().squeeze()

                if connected_edges.numel() == 1:
                    continue

                # Create pairs of connected edges (edge indices)
                for j in range(connected_edges.size(0)):
                    for k in range(j + 1, connected_edges.size(0)):
                        line_graph_edge_indices.append(
                            [connected_edges[j].item(), connected_edges[k].item()])

            line_graph_edge_indices = torch.tensor(
                line_graph_edge_indices, dtype=torch.long).t()

            # Create the line graph attributes (edge attributes from the original graph)
            line_graph_edge_attrs = value.edge_attr

            # Create the Data object for the line graph
            line_graph = Data(
                x=line_graph_edge_attrs,
                y=value.y,
                pos=value.pos,
                edge_index=line_graph_edge_indices.contiguous(),
                num_nodes=line_graph_nodes.size(0),
                graph_id = key,
                edge_pos = value.edge_pos
            )

            line_graph_dict[key] = line_graph

        return line_graph_dict


    def create_torch_geom_data(self, node_df, edge_df, label=None):
        node_features = torch.tensor(
            node_df.loc[:, "pos_x":"isAtSampleBorder"].values, dtype=torch.float32)
        node_pos = torch.tensor(
            node_df.loc[:, "pos_x":"pos_z"].values, dtype=torch.float32)
        edge_index = torch.tensor(
            edge_df[['node1id', 'node2id']].values, dtype=torch.long).t().contiguous()

        edge_attributes = torch.tensor(
            edge_df.iloc[:, 2:].values, dtype=torch.float32)
        label = torch.tensor(label)

        # extracts for every edge its position which is the average of the two nodes

        edge_pos = (node_pos[edge_index[0]] + node_pos[edge_index[1]])/2

        
        #edge_attributes_comb = torch.cat((edge_attributes, edge_pos), 1)

        if label is None:
            data = Data(x=node_features, edge_index=edge_index,
                        edge_attr=edge_attributes, pos=node_pos, edge_pos =edge_pos )
        else:
            data = Data(x=node_features, edge_index=edge_index,
                        edge_attr=edge_attributes, pos=node_pos, edge_pos =edge_pos,  y=torch.tensor([label]))
        return data

    def read_labels(self, label_file):
        label_data = pd.read_excel(label_file, index_col="ID")
        return label_data

    def read_graphs(self, graph_path):
        files = os.listdir(graph_path)
        node_dict = {}
        edge_dict = {}
        for file in sorted(files):
            idx = re.findall(r'\d+', str(file))[0]
            idx = int(idx)
            if "edges" in str(file):
                edge_dict[idx] = pd.read_csv(os.path.join(
                    graph_path, file), sep=";", index_col="id")
            elif "nodes" in str(file):
                node_dict[idx] = pd.read_csv(os.path.join(
                    graph_path, file), sep=";", index_col="id")

        graph_dict = {}
        for key in node_dict.keys():
            nodes = node_dict[key]
            edges = edge_dict[key]
            disease = self.label_data.loc[key]["Disease"]
            if self.two_cls:
                cls = class_dict_2cls[disease]
            else:
                cls = class_dict[disease]
            g = self.create_torch_geom_data(nodes, edges, label=cls)
            g.graph_id = key
            graph_dict[key] = g

        return graph_dict
