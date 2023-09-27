import numpy as np
import pandas as pd
import os
import networkx as nx
import re
import torch
import time
import re

from torch_geometric.utils.convert import from_networkx
from torch_geometric.transforms import line_graph, ToUndirected
from torch_geometric.data import Data, HeteroData



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


class VesselLoaderNetworkX:

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


class GraphLoaderTorch:

    def __init__(self, graph_path, label_file = None, two_cls=False, create_line = True):

        self.two_cls = two_cls
        self.graph_path = graph_path
        self.label_file = label_file
        if self.label_file is not None:
            self.label_data = self.read_labels(label_file)
        self.full_data = self.read_graphs(graph_path)

        if create_line:
            self.line_data = self.line_graphs_alt()
        else:
            self.line_data = None

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
            node_df.values, dtype=torch.float32) # .loc[:, "pos_x":"isAtSampleBorder"]
        node_pos = torch.tensor(
            node_df.iloc[:, 0:2].values, dtype=torch.float32) # "pos_x":"pos_z"
        edge_index = torch.tensor(
            edge_df[['node1id', 'node2id']].values, dtype=torch.long).t().contiguous()

        edge_attributes = torch.tensor(
            edge_df.iloc[:, 2:].values, dtype=torch.float32)


        # extracts for every edge its position which is the average of the two nodes

        edge_pos = (node_pos[edge_index[0]] + node_pos[edge_index[1]])/2

        if label is None:
            data = Data(x=node_features, edge_index=edge_index,
                        edge_attr=edge_attributes, pos=node_pos, edge_pos =edge_pos )
        else:
            label = torch.tensor(label)
            data = Data(x=node_features, edge_index=edge_index,
                        edge_attr=edge_attributes, pos=node_pos, edge_pos =edge_pos,  y=torch.tensor([label]))
        return data

    def read_labels(self, label_file):
        label_data = pd.read_excel(label_file, index_col="ID")
        return label_data

    def read_graphs(self, graph_path):
        files = os.listdir(graph_path)
        # use only the files with ending .csv
        files = [file for file in files if ".csv" in file]
        node_dict = {}
        edge_dict = {}
        for file in sorted(files):
            idx = re.findall(r'\d+', str(file))[0]
            idx = int(idx)
            if "edges" in str(file):
                edge_dict[idx] = pd.read_csv(os.path.join(
                    graph_path, file), sep=";", index_col=0)#"id"
            elif "nodes" in str(file):
                node_dict[idx] = pd.read_csv(os.path.join(
                    graph_path, file), sep=";", index_col=0)#"id"

        graph_dict = {}
        for key in node_dict.keys():
            #print(key)
            nodes = node_dict[key]
            edges = edge_dict[key]
            if self.label_file is not None:
                disease = self.label_data.loc[key]["Disease"]
                if self.two_cls:
                    cls = class_dict_2cls[disease]
                else:
                    cls = class_dict[disease]
            else:
                cls = None
            g = self.create_torch_geom_data(nodes, edges, label=cls)
            g.graph_id = key
            graph_dict[key] = g


        return graph_dict



class HeteroGraphLoaderTorch:

    octa_dr_dict = {"Healthy": 0, "DM": 1, "PDR": 2, "Early NPDR": 3, "Late NPDR": 4}

    def __init__(self, 
                 graph_path_1, 
                 graph_path_2, 
                 hetero_edges_path_12, 
                 label_file = None, 
                 line_graph_1 = False, 
                 line_graph_2 = False, 
                 class_dict = None, 
                 ):

        self.graph_path_1 = graph_path_1
        self.graph_path_2 = graph_path_2
        self.hetero_edges_path_12 = hetero_edges_path_12
        self.label_file = label_file

        if class_dict is not None:
            self.octa_dr_dict = class_dict


        if self.label_file is not None:
            self.label_data = self.read_labels(label_file)
        
        self.full_graphs_1 = self.read_graphs(graph_path_1)
        self.full_graphs_2 = self.read_graphs(graph_path_2)
        self.hetero_edges_12 = self.read_hetero_edges(hetero_edges_path_12)

        if line_graph_1:
            self.line_graphs_1 = self.line_graphs_alt(self.full_graphs_1)
        else:
            self.line_graphs_1 = None

        if line_graph_2:
            self.line_graphs_2 = self.line_graphs_alt(self.full_graphs_2)
        else:
            self.line_graphs_2 = None

        # assign line graph if exists else full graph
        graphs_1 = self.line_graphs_1 if self.line_graphs_1 is not None else self.full_graphs_1 
        graphs_2 = self.line_graphs_2 if self.line_graphs_2 is not None else self.full_graphs_2 

        self.hetero_graphs = self.create_hetero_graphs(graphs_1, graphs_2)



    def create_hetero_graphs(self, g1 , g2):

        het_graph_dict = {}
        # iterate through the hetero edges

        for key, het_edge_index in self.hetero_edges_12.items():
            # get the two graphs
            try:
                graph_1 = g1[key]
                graph_2 = g2[key]
            except KeyError:
                continue
            # create the hetero graph
            het_graph = self.create_hetero_graph(graph_1, graph_2, het_edge_index)
            het_graph_dict[key] = het_graph


        return het_graph_dict

    def create_hetero_graph(self, graph_1, graph_2, het_edge_index):
            
            # create the hetero graph
            het_graph = HeteroData()
    
            het_graph['graph_1'].x = graph_1.x
            het_graph['graph_2'].x = graph_2.x
    
            het_graph['graph_1'].pos = graph_1.pos
            het_graph['graph_2'].pos = graph_2.pos

            het_graph['graph_1', 'to', 'graph_1'].edge_index = graph_1.edge_index
            het_graph['graph_2', 'to', 'graph_2'].edge_index = graph_2.edge_index
            het_graph['graph_1', 'to', 'graph_2'].edge_index = het_edge_index

            try:
                het_graph.y = graph_1.y
            except KeyError:
                pass

            het_graph = ToUndirected()(het_graph)
    
            return het_graph

    def read_hetero_edges(self, hetero_edges_path_12):

        # iterate through all files in the folder
        files = os.listdir(hetero_edges_path_12)
        # use only the files with ending .csv
        files = [file for file in files if ".csv" in file]
        edge_dict = {}
        for file in sorted(files):
            idx = re.findall(r'\d+', str(file))[0]
            if "_OD" in file:
                eye = "OD"
            else:
                eye = "OS"
            idx_dict = idx + "_" + eye
            edge_df_ves_reg = pd.read_csv(os.path.join(
                hetero_edges_path_12, file), sep=";", index_col=0)
            
            edge_dict[idx_dict] = torch.tensor(
                                    edge_df_ves_reg[['node1id', 'node2id']].values, dtype=torch.long).t().contiguous()

        return edge_dict


    def create_torch_geom_data(self, node_df, edge_df, label=None):



        node_features = torch.tensor(
            node_df.values, dtype=torch.float32) # .loc[:, "pos_x":"isAtSampleBorder"]
        node_pos = torch.tensor(
            node_df.iloc[:, 0:2].values, dtype=torch.float32) # "pos_x":"pos_z"
        edge_index = torch.tensor(
            edge_df[['node1id', 'node2id']].values, dtype=torch.long).t().contiguous()

        edge_attributes = torch.tensor(
            edge_df.iloc[:, 2:].values, dtype=torch.float32)
        
        #if not self.load_center_nodes:
#
        #    # find the nodes with largest area (feature 2)
        #    node_areas = node_features[:,2]
        #    max_area_idx = node_areas.argmax()
#
        #    # remove the node from the node features
        #    node_features = torch.cat((node_features[:max_area_idx], node_features[max_area_idx+1:]))
        #    node_pos = torch.cat((node_pos[:max_area_idx], node_pos[max_area_idx+1:]))
#
        #    # remove the edges that contain the node
        #    edge_index = edge_index[:, (edge_index[0] != max_area_idx) & (edge_index[1] != max_area_idx)]
        #    edge_attributes = edge_attributes[(edge_index[0] != max_area_idx) & (edge_index[1] != max_area_idx)]
#
        #    # adjust the edge indices
        #    edge_index[0, edge_index[0] > max_area_idx] -= 1
        #    edge_index[1, edge_index[1] > max_area_idx] -= 1



        # extracts for every edge its position which is the average of the two nodes

        edge_pos = (node_pos[edge_index[0]] + node_pos[edge_index[1]])/2

        if label is None:
            data = Data(x=node_features, edge_index=edge_index,
                        edge_attr=edge_attributes, pos=node_pos, edge_pos =edge_pos )
        else:
            label = torch.tensor(label)
            data = Data(x=node_features, edge_index=edge_index,
                        edge_attr=edge_attributes, pos=node_pos, edge_pos =edge_pos,  y=torch.tensor([label]))
        return data

    def read_labels(self, label_file):
        label_data = pd.read_csv(label_file)

        label_dict = {}

        for i, row in label_data.iterrows():
            index = row["Subject"]
            # convert to string with leading zeros, 4 digits
            index = str(index).zfill(4)
            eye = row["Eye"]
            label_dict[index + "_" + eye] = row["Group"]
        
        return label_dict

    def read_graphs(self, graph_path):
        files = os.listdir(graph_path)
        # use only the files with ending .csv
        files = [file for file in files if ".csv" in file]
        node_dict = {}
        edge_dict = {}
        for file in sorted(files):
            idx = re.findall(r'\d+', str(file))[0]

            if "_OD" in file:
                eye = "OD"
            else:
                eye = "OS"
            idx_dict = idx + "_" + eye
            if "edges" in str(file):
                try:
                    edge_dict[idx_dict] = pd.read_csv(os.path.join(
                        graph_path, file), sep=";", index_col=0)#"id"
                except pd.errors.EmptyDataError:
                    pass
            elif "nodes" in str(file):
                try:
                    node_dict[idx_dict] = pd.read_csv(os.path.join(
                        graph_path, file), sep=";", index_col=0)#"id"
                except pd.errors.EmptyDataError:
                    pass

        graph_dict = {}
        for key in node_dict.keys():
            #print(key)
            try:
                nodes = node_dict[key]
                edges = edge_dict[key]
            except KeyError:
                continue
            if self.label_file is not None:
                try:
                    disease = self.label_data[key]
                    cls = self.octa_dr_dict[disease]
                except KeyError:
                    print("no label for ", key)
                    continue
            else:
                cls = None
            g = self.create_torch_geom_data(nodes, edges, label=cls)
            g.graph_id = key
            graph_dict[key] = g


        return graph_dict


    def line_graphs_alt(self, full_data):
        line_graph_dict = {}

        for key, value in full_data.items():

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
                pos=value.edge_pos,
                edge_index=line_graph_edge_indices.contiguous(),
                num_nodes=line_graph_nodes.size(0),
                graph_id = key,
                edge_pos = value.pos
            )

            line_graph_dict[key] = line_graph

        return line_graph_dict