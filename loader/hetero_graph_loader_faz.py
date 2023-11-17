import os
import re
import pandas as pd
import torch
import pickle
from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import ToUndirected, RemoveIsolatedNodes
from sklearn.model_selection import train_test_split

class HeteroGraphLoaderTorch:

    octa_dr_dict = {"Healthy": 0, "DM": 1, "PDR": 2, "Early NPDR": 3, "Late NPDR": 4}

    def __init__(self, 
                 graph_path_1, 
                 graph_path_2,
                 graph_path_3,
                 hetero_edges_path_12, 
                 hetero_edges_path_13,
                 hetero_edges_path_23,
                 mode,
                 label_file = None, 
                 line_graph_1 = False, 
                 line_graph_2 = False, 
                 class_dict = None, 
                 pickle_file = None
                 ):

        self.graph_path_1 = graph_path_1
        self.graph_path_2 = graph_path_2
        self.graph_path_3 = graph_path_3
        self.hetero_edges_path_12 = hetero_edges_path_12
        self.hetero_edges_path_13 = hetero_edges_path_13
        self.hetero_edges_path_23 = hetero_edges_path_23

        self.label_file = label_file
        self.mode = mode

        if class_dict is not None:
            self.octa_dr_dict = class_dict


        if self.label_file is not None:
            self.label_data = self.read_labels(label_file)

        if pickle_file is None:
        
            self.full_graphs_1 = self.read_graphs(graph_path_1)
            self.full_graphs_2 = self.read_graphs(graph_path_2)
            self.full_graphs_3 = self.read_node_graphs(graph_path_3)

            #print(self.full_graphs_1)
            #print(self.full_graphs_3)

            #for graph in self.full_graphs_3.values():
            #    print(graph)

            self.hetero_edges_12 = self.read_hetero_edges(hetero_edges_path_12)
            self.hetero_edges_13 = self.read_hetero_edges(hetero_edges_path_13)
            self.hetero_edges_23 = self.read_hetero_edges(hetero_edges_path_23)

            if line_graph_1:
                self.line_graphs_1 = self.line_graphs(self.full_graphs_1)
            else:
                self.line_graphs_1 = None

            if line_graph_2:
                self.line_graphs_2 = self.line_graphs(self.full_graphs_2)
            else:
                self.line_graphs_2 = None

            # assign line graph if exists else full graph
            graphs_1 = self.line_graphs_1 if self.line_graphs_1 is not None else self.full_graphs_1 
            graphs_2 = self.line_graphs_2 if self.line_graphs_2 is not None else self.full_graphs_2 
            graphs_3 = self.full_graphs_3

            self.hetero_graphs = self.create_hetero_graphs(graphs_1, graphs_2, graphs_3)
            for key, value in self.hetero_graphs.items():
                value.graph_id = key
            self.hetero_graph_list = list(self.hetero_graphs.values())


        else:
            try: 
                with open(pickle_file, "rb") as f:
                    self.hetero_graphs = pickle.load(f)
                for key, value in self.hetero_graphs.items():
                    value.graph_id = key
                self.hetero_graph_list = list(self.hetero_graphs.values())
            except FileNotFoundError:
                print("Pickle file not found, creating new one")
                self.full_graphs_1 = self.read_graphs(graph_path_1)
                self.full_graphs_2 = self.read_graphs(graph_path_2)
                self.full_graphs_3 = self.read_node_graphs(graph_path_3)
                
                self.hetero_edges_12 = self.read_hetero_edges(hetero_edges_path_12)
                self.hetero_edges_13 = self.read_hetero_edges(hetero_edges_path_13)
                self.hetero_edges_23 = self.read_hetero_edges(hetero_edges_path_23)

                if line_graph_1:
                    self.line_graphs_1 = self.line_graphs(self.full_graphs_1)
                else:
                    self.line_graphs_1 = None

                if line_graph_2:
                    self.line_graphs_2 = self.line_graphs(self.full_graphs_2)
                else:
                    self.line_graphs_2 = None

                # assign line graph if exists else full graph
                graphs_1 = self.line_graphs_1 if self.line_graphs_1 is not None else self.full_graphs_1 
                graphs_2 = self.line_graphs_2 if self.line_graphs_2 is not None else self.full_graphs_2 
                graphs_3 = self.full_graphs_3

                self.hetero_graphs = self.create_hetero_graphs(graphs_1, graphs_2, graphs_3)
                for key, value in self.hetero_graphs.items():
                    value.graph_id = key
                self.hetero_graph_list = list(self.hetero_graphs.values())

                with open(pickle_file, "wb") as f:
                    pickle.dump(self.hetero_graphs, f)



    def update_class(self, class_dict):

        UserWarning("Only hetero_graphs and hetero_graph_list are updated, not the full_graphs")

        self.octa_dr_dict = class_dict

        for i, (key, value) in enumerate(self.hetero_graphs.items()):
            try:
                disease = self.label_data[key]
                cls = self.octa_dr_dict[disease]
                value.y = torch.tensor([cls])
                # update in list
                self.hetero_graph_list[i].y = torch.tensor([cls])
            except KeyError:
                pass


    def read_labels(self, label_file):
        label_data = pd.read_csv(label_file)


        train, temp = train_test_split(label_data, test_size=0.3, random_state=42, stratify=label_data["Group"])
        test, val = train_test_split(temp, test_size=0.5, random_state=42, stratify=temp["Group"])
        del temp

        _, debug = train_test_split(test, test_size=0.10, random_state=42, stratify=test["Group"])

        if self.mode == "train":
            label_data = train            
        elif self.mode == "test":
            label_data = test
        elif self.mode == "val":
            label_data = val
        elif self.mode == "debug":
            label_data = debug


        label_dict = {}

        for i, row in label_data.iterrows():
            index = row["Subject"]
            # convert to string with leading zeros, 4 digits
            index = str(index).zfill(4)
            eye = row["Eye"]
            label_dict[index + "_" + eye] = row["Group"]
        
        return label_dict
    
    def read_node_graphs(self, graph_path):
        files = os.listdir(graph_path)
        # use only the files with ending .csv
        files = [file for file in files if ".csv" in file]
        node_dict = {}
        for file in sorted(files):
            idx = re.findall(r'\d+', str(file))[0]

            if "_OD" in file:
                eye = "OD"
            else:
                eye = "OS"
            idx_dict = idx + "_" + eye

            if self.label_data is not None:
                try:
                    disease = self.label_data[idx_dict]
                except KeyError:
                    continue

            try:
                node_dict[idx_dict] = pd.read_csv(os.path.join(
                    graph_path, file), sep=";", index_col=0)#"id"
            except pd.errors.EmptyDataError:
                pass

        graph_dict = {}
        for key in node_dict.keys():
            if self.label_file is not None:
                nodes = node_dict[key]
                try:
                    disease = self.label_data[key]
                    cls = self.octa_dr_dict[disease]
                except KeyError:
                    print("no label for ", key)
                    continue
            else:
                cls = None

            # nodes contains the features of a single node
            node_features = torch.tensor(
                nodes.values, dtype=torch.float32)
            node_features = node_features.squeeze().unsqueeze(0)

            node_pos = node_features[:, 0:2]
            #node_features = torch.tensor(
            #    nodes.values, dtype=torch.float32) # .loc[:, "pos_x":"isAtSampleBorder"]
            #node_pos = torch.tensor(
            #    nodes.iloc[:, 0:2].values, dtype=torch.float32) # "pos_x":"pos_z"
            
            if cls is None:
                data = Data(x=node_features, edge_index=None,
                            edge_attr=None, pos=node_pos)
            else:
                label = torch.tensor(cls)
                data = Data(x=node_features, edge_index=None,
                            edge_attr=None, pos=node_pos, y=torch.tensor([label]))
            g = data
            g.graph_id = key
            graph_dict[key] = g

        return graph_dict


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

            if self.label_data is not None:
                try:
                    disease = self.label_data[idx_dict]
                except KeyError:
                    continue

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

    def read_hetero_edges(self, hetero_edges_path_12):

        # iterate through all files in the folder
        files = os.listdir(hetero_edges_path_12)
        # use only the files with ending .csv
        files = [file for file in files if ".csv" in file]
        edge_dict = {}
        for file in sorted(files):
            #print(file)
            idx = re.findall(r'\d+', str(file))[0]
            if "_OD" in file:
                eye = "OD"
            else:
                eye = "OS"
            idx_dict = idx + "_" + eye
            edge_df_ves_reg = pd.read_csv(os.path.join(
                hetero_edges_path_12, file), sep=";", index_col=0)
            
            #print(edge_df_ves_reg)
            #print(os.path.join(
            #    hetero_edges_path_12, file))
            
            try:
                edge_dict[idx_dict] = torch.tensor(
                                        edge_df_ves_reg[['node1id', 'node2id']].values, dtype=torch.long).t().contiguous()
            except TypeError:
                pass
            #raise TypeError


        return edge_dict

    def line_graphs(self, full_data):
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
    

    def create_hetero_graphs(self, g1 , g2, g3):

        het_graph_dict = {}
        # iterate through the hetero edges

        for key, het_edge_index in self.hetero_edges_12.items():
            # get the two graphs
            try:
                graph_1 = g1[key]
                graph_2 = g2[key]
                graph_3 = g3[key]
            except KeyError:
                continue

            # find the other matching edge indices, hetero_edges_13 and hetero_edges_23
            
            try:
                het_edge_index_13 = self.hetero_edges_13[key]
                het_edge_index_23 = self.hetero_edges_23[key]
            except KeyError:
                continue


            eye = True if "_OD" in key else False

            # create the hetero graph
            het_graph = self.create_hetero_graph(graph_1, graph_2, graph_3, het_edge_index, het_edge_index_13, het_edge_index_23, eye)
            het_graph_dict[key] = het_graph


        return het_graph_dict

    def create_hetero_graph(self, graph_1, graph_2, graph_3, het_edge_index_12, het_edge_index_13, het_edge_index_23, eye):
            
            # create the hetero graph
            het_graph = HeteroData()
    
            het_graph['graph_1'].x = graph_1.x
            het_graph['graph_2'].x = graph_2.x
            het_graph['faz'].x = graph_3.x
    
            het_graph['graph_1'].pos = graph_1.pos
            het_graph['graph_2'].pos = graph_2.pos
            het_graph['faz'].pos = graph_3.pos

            het_graph['graph_1', 'to', 'graph_1'].edge_index = graph_1.edge_index
            het_graph['graph_2', 'to', 'graph_2'].edge_index = graph_2.edge_index
            het_graph['graph_1', 'to', 'graph_2'].edge_index = het_edge_index_12

            het_edge_index_13[1] = 0
            het_edge_index_23[0] = 0

            #print(het_edge_index_13)
            #print("################################")
            #print(het_edge_index_23)


            het_graph['graph_1', 'to', 'faz'].edge_index = het_edge_index_13
            het_graph['faz', 'to', 'graph_2'].edge_index = het_edge_index_23
            faz_edge_index = torch.tensor([[0, 0]], dtype=torch.long)
            faz_edge_index = faz_edge_index.transpose(0, 1)
            het_graph['faz', 'to', 'faz'].edge_index = faz_edge_index

            het_graph.eye = eye

            try:
                het_graph.y = graph_1.y
            except KeyError:
                pass

            het_graph = ToUndirected()(het_graph)
            het_graph = RemoveIsolatedNodes()(het_graph)
    
            return het_graph


    def __len__(self):
        return len(self.hetero_graph_list)
    
    def __getitem__(self, idx):
        return self.hetero_graph_list[idx]
    

    def to(self, device):
        for graph in self.hetero_graph_list:
            graph.to(device)
        return self
    
