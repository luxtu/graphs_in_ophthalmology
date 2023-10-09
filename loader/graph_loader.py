import os
import re
import pandas as pd
import torch
from torch_geometric.transforms import line_graph
from torch_geometric.data import Data


class_dict = {"NORMAL": 0, "DR": 1, "CNV": 2, "AMD": 3}
class_dict_2cls = {"NORMAL": 0, "DR": 1, "CNV": 0, "AMD": 0}



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



