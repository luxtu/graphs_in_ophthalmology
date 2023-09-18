import numpy as np
import pandas as pd
import os
import networkx as nx
import re
import torch
import time

from skimage import measure, morphology
from PIL import Image

from skimage.morphology import skeletonize
from torch_geometric.utils.convert import from_networkx
from torch_geometric.transforms import line_graph
from torch_geometric.data import Data





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


class VesselLoaderTorch:

    def __init__(self, graph_path, label_file, two_cls=False, create_line = True):

        self.two_cls = two_cls
        self.graph_path = graph_path
        self.label_file = label_file
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
        label = torch.tensor(label)

        # extracts for every edge its position which is the average of the two nodes

        edge_pos = (node_pos[edge_index[0]] + node_pos[edge_index[1]])/2


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
                    graph_path, file), sep=";", index_col=0)#"id"
            elif "nodes" in str(file):
                node_dict[idx] = pd.read_csv(os.path.join(
                    graph_path, file), sep=";", index_col=0)#"id"

        graph_dict = {}
        for key in node_dict.keys():
            #print(key)
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






class VoidGraphGenerator:

    def __init__(self, seg_path, save_path):

        self.seg_path = seg_path
        self.save_path = save_path


    def save_region_graphs(self):
        region_graphs = {}
        files = os.listdir(self.seg_path)
        for file in sorted(files):
            idx = int(file[-9:-4])
            region_graphs[idx] = self.generate_region_graph(file)

            print(file)
    

    def generate_region_graph(self, file):


        idx = int(file[-9:-4])
        #read image
        seg = Image.open(os.path.join(self.seg_path, file))
        seg = np.array(seg)
        seg = seg.astype(np.uint8)

        # background is 1
        ## region properties don't make sense with extremely small regions

        region_labels = measure.label(morphology.remove_small_holes(seg, area_threshold=5, connectivity=1).astype("uint8"), background=1)

        # remove all the labels that contain less than 9 pixels
        #region_labels = morphology.remove_small_objects(region_labels, min_size=9, connectivity=1)


        props = measure.regionprops_table(region_labels, properties=('centroid',
                                                 'area',
                                                 'perimeter',
                                                 'eccentricity',
                                                 'equivalent_diameter',
                                                 'orientation',
                                                 'solidity',
                                                 'feret_diameter_max',
                                                 'extent',
                                                 'axis_major_length',
                                                 'axis_minor_length'))
        
        df = pd.DataFrame(props)


        df.to_csv(os.path.join(self.save_path, str(idx)+ "_nodes"+".csv"), sep = ";")

        edge_index_df = self.generate_edge_index(region_labels, seg)
        edge_index_df.to_csv(os.path.join(self.save_path, str(idx) + "_edges"+".csv"), sep = ";")



    def generate_edge_index(self, region_labels, seg):

        # skeletonization may remove the connections to the image border.
        # this results in a connection between regions that was not there before


        # skeletonize the image
        skel_labels = self.generate_skel_labels(seg, skel_type = "border")
        skel_neighbor_list = self.generate_skel_neighbor_list(skel_labels)


        matching_regions = self.find_matching_regions(skel_labels, region_labels)

        # fail means that there is an edge in the skeletonized image that is not in the original image
        # could be because there are regions in the skeletonized image that do not exist in the real image

        neighbor_regions_list_seg = []
        success = 0
        fail = 0
        for tup in skel_neighbor_list:
            try:
                neighbor_regions_list_seg.append([matching_regions[tup[0]]-1, matching_regions[tup[1]]-1])
                success += 1
            except KeyError:
                fail += 1
        print(success, fail)

        # create a dataframe for the edge list where the first column is an ID column and the second and third column are the indices of the nodes connected by the edge

        return pd.DataFrame(neighbor_regions_list_seg, columns=["node1id", "node2id"])


    def generate_skel_labels(self, seg, skel_type = "border"):

        if skel_type == "border":
            seg_work = seg.copy()
            # make all border pixels 1
            seg_work = seg_work.astype(np.uint8)
            seg_work[0,:] = 255
            seg_work[-1,:] = 255
            seg_work[:,0] = 255
            seg_work[:,-1] = 255

            skeleton = skeletonize(seg_work)
            skeleton = skeleton.astype(np.uint8)
            skel_labels_border = measure.label(skeleton, background=1, connectivity=1)

            del seg_work

            return skel_labels_border

        elif skel_type == "crop":
            skeleton = skeletonize(seg)
            skeleton = skeleton.astype(np.uint8)

            crop_val = 4
            # cropping is essential here, ske
            skel_crop = skeleton[crop_val:-crop_val,crop_val:-crop_val]
            skel_labels = measure.label(skel_crop, background=1, connectivity=1)
            skel_labels_crop = np.zeros_like(skeleton, dtype= skel_labels.dtype)
            skel_labels_crop[crop_val:-crop_val,crop_val:-crop_val] = skel_labels

            return skel_labels_crop

        else: 
            raise ValueError("skel_type must be either 'border' or 'crop'")





    def generate_skel_neighbor_list(self, skel_labels):

        # this might not generate all the edges
        # might be some degenerate cases

        #Initialize a set to store neighboring regions
        neighbor_regions = set()

        # Iterate through rows from top to bottom (sweep line)
        for row in range(skel_labels.shape[0]):
            # Identify the regions intersected by the sweep line at this row

            intersected_regions = skel_labels[row, 1:][np.diff(skel_labels[row, :]) != 0]
            intersected_regions = intersected_regions[intersected_regions != 0]

            intersected_regions = intersected_regions.tolist()
            intersected_regions.insert(0, skel_labels[row, 0])


            # Add newly intersected regions to the priority queue
            for i in range(len(intersected_regions)-1):
            
                r1 = min(intersected_regions[i], intersected_regions[i+1])
                r2 = max(intersected_regions[i], intersected_regions[i+1])
                neighbor_regions.add((r1, r2))


        # Iterate through columns from top to bottom (sweep line)
        for col in range(skel_labels.shape[1]):

            # Identify the regions intersected by the sweep line at this col
            intersected_regions = np.array(skel_labels[1:,col])[np.diff(skel_labels[:,col]) != 0]
            intersected_regions = intersected_regions[intersected_regions != 0]

            intersected_regions = intersected_regions.tolist()
            intersected_regions.insert(0, skel_labels[0,col])

            # Add newly intersected regions to the priority queue
            for i in range(len(intersected_regions)-1):

                r1 = min(intersected_regions[i], intersected_regions[i+1])
                r2 = max(intersected_regions[i], intersected_regions[i+1])
                neighbor_regions.add((r1, r2))

        return list(neighbor_regions)


    def find_matching_regions(self, skel_labeled, seg_labeled):

        # finds for every region in the original image the corresponding region in the skeletonized image
        
        # if no regions was found in the skeletonized image, the region is not added to the dictionary
        # actually this should not happen 

        matching_regions = {}

        for lab_val in np.unique(seg_labeled):
            skel_vals, counts = np.unique(skel_labeled[np.where(seg_labeled == lab_val)], return_counts=True) 

            count_sort_ind = np.argsort(-counts)
            skel_vals = skel_vals[count_sort_ind]

            if skel_vals[0] ==0:
                try:
                    matching_regions[skel_vals[1]] = lab_val
                except IndexError:
                    pass
            else:
                matching_regions[skel_vals[0]] = lab_val

        return matching_regions







"""

class VoidGraphLoader:


    class_dict = {"NORMAL": 0, "DR": 1, "CNV": 2, "AMD": 3}
    class_dict_2cls = {"NORMAL": 0, "DR": 1, "CNV": 0, "AMD": 0}

    def __init__(self, seg_path, label_file, two_cls=False):


        self.two_cls = two_cls
        self.seg_path = seg_path
        self.label_file = label_file
        self.label_data = self.read_labels(label_file)
        self.region_graphs = self.generate_region_graphs()


    def read_labels(self, label_file):
        label_data = pd.read_excel(label_file, index_col="ID")
        return label_data
    
    def generate_region_graphs(self):
        region_graphs = {}
        files = os.listdir(self.seg_path)
        for file in sorted(files)[:100]:
            idx = int(file[-9:-4])
            region_graphs[idx] = self.generate_region_graph(file)

            print(file)
        return region_graphs
    

    def generate_region_graph(self, file):
        #read image
        seg = Image.open(os.path.join(self.seg_path, file))
        seg = np.array(seg)
        seg = seg.astype(np.uint8)
        region_labels = measure.label(seg, background=255)

        props = measure.regionprops_table(region_labels, properties=(
                                                 'area',
                                                 'perimeter',
                                                 'eccentricity',
                                                 'equivalent_diameter',
                                                 'orientation',
                                                 'solidity',
                                                 'feret_diameter_max',
                                                 'extent',
                                                 'axis_major_length',
                                                 'axis_minor_length'))
        
        centroid = measure.regionprops_table(region_labels, properties=(
                                                 'centroid'))
        
        df_centroid = pd.DataFrame(centroid)

        edge_index = self.generate_edge_index(region_labels, seg)

        

        # edge index is a list of tuples (i,j) where i and j are the indices of the nodes connected by the edge

        idx = int(file[-9:-4])
        if self.two_cls:
            y_label = VoidGraphLoader.class_dict_2cls[self.label_data.loc[idx]["Disease"]]
        else:
            y_label = VoidGraphLoader.class_dict[self.label_data.loc[idx]["Disease"]]
        

        #print(props[["centroid-0" ,"centroid-1"]])

        df = pd.DataFrame(props)
        # Create a list to store property values for each region
        region_property_values = []

        # Iterate through the DataFrame and extract property values
        for idx, row in df.iterrows():
            # Extract the properties and convert them to a list of floats
            property_values = [float(val) for val in row[1:]]
            region_property_values.append(property_values)

        # Convert the list of lists into a PyTorch tensor
        properties_tensor = torch.tensor(region_property_values, dtype=torch.float32)
        graph = Data(x=properties_tensor, edge_index=edge_index, y=torch.tensor([y_label]), pos = torch.tensor(df_centroid[["centroid-0" ,"centroid-1"]].to_numpy()))


        return graph






    def generate_edge_index(self, region_labels, seg):

        # skeletonization may remove the connections to the image border.
        # this results in a connection between regions that was not there before


        # skeletonize the image
        skel_labels = self.generate_skel_labels(seg, skel_type = "border")
        skel_neighbor_list = self.generate_skel_neighbor_list(skel_labels)


        matching_regions = self.find_matching_regions(skel_labels, region_labels)

        # fail means that there is an edge in the skeletonized image that is not in the original image
        # could be because there are regions in the skeletonized image that do not exist in the real image

        neighbor_regions_list_seg = []
        success = 0
        fail = 0
        for tup in skel_neighbor_list:
            try:
                neighbor_regions_list_seg.append([matching_regions[tup[0]]-1, matching_regions[tup[1]]-1])
                success += 1
            except KeyError:
                fail += 1

        print(success, fail)
        print(torch.tensor(neighbor_regions_list_seg, dtype= torch.long).T.shape)

        return torch.tensor(neighbor_regions_list_seg, dtype= torch.long).T


   

"""