import networkx as nx
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
from scipy.spatial import KDTree
from scipy.sparse import csr_matrix
from scipy.sparse import dok_array



def create_graph(nodesFile, edgesFile, index_addon = None):
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
    if type(nodesFile) ==str and type(nodesFile) ==str :
        nodes = pd.read_csv(nodesFile, sep = ";", index_col= "id")
        edges = pd.read_csv(edgesFile, sep = ";", index_col= "id")
    else:
        nodes = nodesFile
        edges = edgesFile
    # create undirected graph 
    G = nx.MultiGraph()


    if index_addon is not None:
        for idxN, node in nodes.iterrows():
            G.add_node(str(int(idxN)) + index_addon, x = (), pos = (float(node["pos_x"]),float(node["pos_y"]), float(node["pos_z"])))

        # add the edges
        for idxE, edge in edges.iterrows():
            G.add_edge(str(int(edge["node1id"]))  + index_addon , str(int(edge["node2id"])) + index_addon, x = (edge[2:])) # edge["distance"], edge["length"], edge["minRadiusAvg"], edge["curveness"], edge["avgRadiusStd"], edge["num_voxels"]

    else:
        for idxN, node in nodes.iterrows():
            G.add_node(idxN, x = (), pos = (float(node["pos_x"]),float(node["pos_y"]), float(node["pos_z"])))

        # add the edges
        for idxE, edge in edges.iterrows():
            G.add_edge(edge["node1id"],edge["node2id"], x = (edge[2:]))

    return G



def to_einfach(G_multi, self_loops = False, isolates = False):
    """ Creates an networkX simple graph from a networkX multigraph. Also removes all isolated nodes and self loops

    Paramters
    ---------
    G_multi: A networkX graph object 

    -------
    Returns

    G: A simple networkX graph without selfloops, parallel edges and isolated nodes.
    """
    G_einfach = nx.Graph(G_multi)
    if not self_loops:
        G_einfach.remove_edges_from(list(nx.selfloop_edges(G_einfach)))
    if not isolates:
        G_einfach.remove_nodes_from(list(nx.isolates(G_einfach)))

    return G_einfach



def enrich_node_attrs(G):
    """ Enriches the x. attribute of the nodes of a networkX graph with features extracted from the incident edges.

    Paramters
    ---------    
    G: A networkX graph object 
    
    """
    feature_dict = {}
    for node in G.nodes:
        nodes_edges = G.edges(node)

        tot_vol = 0
        avg_vol_l = []
        tot_len = 0
        avg_len_l = []
        curveness = []
        avg_rad_sd = []
        tot_voxel = 0
        tot_dis = 0
        avg_dis = []
        for sedge in nodes_edges:
            tot_vol += G.get_edge_data(*sedge)["x"][2]
            tot_len += G.get_edge_data(*sedge)["x"][1]
            avg_vol_l.append(G.get_edge_data(*sedge)["x"][2])
            avg_len_l.append(G.get_edge_data(*sedge)["x"][1])
            curveness.append(G.get_edge_data(*sedge)["x"][3])
            avg_rad_sd.append(G.get_edge_data(*sedge)["x"][4])
            tot_voxel += G.get_edge_data(*sedge)["x"][5]
            tot_dis += G.get_edge_data(*sedge)["x"][0]
            avg_dis.append(G.get_edge_data(*sedge)["x"][0])

        feature_dict[node] = (float(tot_vol), float(tot_len), float(np.mean(avg_vol_l)), float(np.mean(avg_len_l)), float(np.mean(curveness)), float(np.mean(avg_rad_sd)), float(tot_voxel), float(tot_dis), float(np.mean(avg_dis)), float(np.std(avg_vol_l)), float(np.std(avg_len_l)), float(np.std(curveness)), float(np.std(avg_rad_sd)),float(np.std(avg_dis)))
    nx.set_node_attributes(G, feature_dict,'x')
    #['node1id', 'node2id', 'length', 'distance', 'curveness', 'volume', 'avgCrossSection', 'minRadiusAvg', 'minRadiusStd', 'avgRadiusAvg',
    #   'avgRadiusStd', 'maxRadiusAvg', 'maxRadiusStd', 'roundnessAvg','roundnessStd', 'node1_degree', 'node2_degree', 'num_voxels', 'hasNodeAtSampleBorder']


def graph_summary(G):
    """ Prints out a quick summary of the characteristics of a networkX graph.

    Paramters
    ---------
    G: A networkX graph object 

    """
    nodeNum = G.order()
    edgeNum = G.size()
    con_comp = nx.algorithms.components.number_connected_components(G)
    self_loops = nx.number_of_selfloops(G)
    avg_deg_multi = edgeNum*2/nodeNum
    isolatedNodes = len(list(nx.isolates(G)))

    print("***************")
    print("Number of Nodes: " +  str(nodeNum))
    print("Number of Edges: " + str(edgeNum))
    print("Number of Connected Components: " + str(con_comp))
    print("Number of Self Loops: " + str(self_loops))
    print("Number of Isolated Nodes: " + str(isolatedNodes))
    print("Average Node Degree: " + str(avg_deg_multi))    
    print("***************")
    return


def scale_position(df, scaleVector):
    """ Scales the position encoding attributes in a node dataframe.

    Paramters
    ---------
    df: A pandas dataframe with node infromation
    scaleVector: A 3D vector with the scaling information for x, y, z

    Returns
    -------
    df: A scaled dataframe
    """
    df["pos_x"] = df["pos_x"]*scaleVector[0]
    df["pos_y"] = df["pos_y"]*scaleVector[1]
    df["pos_z"] = df["pos_z"]*scaleVector[2]

    return df


def connected_components_dict(labels):
    """ Takes connected components as list and return them as dict.

    Paramters
    ---------
    df: Labels of connected components.

    Returns
    -------
    con_comp: A dictionary with the values lists corresponing to connected components.
    """
    con_comp = {}
    for i in range(len(labels)):
        try:
            con_comp[labels[i]].add(i)
        except KeyError:
            con_comp[labels[i]] = {i}

    return con_comp



def relevant_connected_components(con_comp, node1_size, labels, rel_th = 1):
    """ Takes connected components as dict and returns only the components that are bigger than the threshold rel_th.
    Additionally the node1_size encodes the start of the second group in the connected components list. Labels is then used to change the names of the nodes in the CCs according to their group.

    Paramters
    ---------
    con_comp: A dicitonary for connected components
    node1_size: Indicator where two node groups are split
    labels: Name Addons for the two groups
    rel_th: A threshold indicating which connected components should be regarded

    Returns
    rel_comp: A dictionary containing the relevant (by threshold) connected components with changed name.
    -------

    """
    rel_comp = {}
    for k, v in con_comp.items():
        if len(v)>rel_th:
            resList = []
            for val in v:
                if val >=node1_size:
                    resList.append(str(val- node1_size) + labels[1])
                else:
                    resList.append(str(val) + labels[0])
            rel_comp[str(k) +"c"] = resList
    return rel_comp




def relable_edges_nodes(edges, nodes, index_addon):
    """ Changes the name columns in edges and node files.

    Paramters
    ---------
    edges: Dataframe containing the edge information of a graph
    nodes: Dataframe containing the node information of a graph
    index_addon: A string that is attached to all columns relevant for naming entities. 


    Returns
    edges: Dataframe for edge information with changed name
    nodes: Dataframe for node information with changed name
    -------

    """
    idx = list(nodes.index)
    idx_new = [str(elem) + index_addon for elem in idx]
    nodes.index = idx_new

    edges['node1id'] = edges['node1id'].apply(lambda x: str(x) + index_addon)
    edges['node2id'] = edges['node2id'].apply(lambda x: str(x) + index_addon) 

    return edges, nodes


def distance_based_adjacency(nodes_1, nodes_2, th):
    """ Returns a sparse adjacency matrix containing pairs of nodes from two sets that have a distance below a certain threshold. 

    Paramters
    ---------
    nodes_1: A dataframe containing node information
    nodes_2: A dataframe containing node information
    th: A distance threshold


    Returns
    adjMcsr: The resulting adjacencey values. Weights indicate the distance between the node pairs.
    -------

    """

    dist_mat_sparse = network_sparse_distance_matrix(nodes_1, nodes_2, th)
    adjM = dok_array((nodes_1.shape[0]+ nodes_2.shape[0], nodes_1.shape[0]+ nodes_2.shape[0]), dtype=np.uint8)

    for key in dist_mat_sparse.keys():
        adjM[key[0], key[1] + nodes_1.shape[0]] = 1
        adjM[key[1]+ nodes_1.shape[0], key[0]] = 1

    adjMcsr = adjM.tocsr()

    return adjMcsr


def network_sparse_distance_matrix(nodes_1, nodes_2, th):
    """ Returns a sparse distance matrix for two node sets with a given threshold for relevant conncetions.

    Paramters
    ---------
    nodes_1: A dataframe containing node information
    nodes_2: A dataframe containing node information
    th: A distance threshold


    Returns
    dist_mat_sparse: The resulting sparse distance matrix.
    -------

    """

    points_1 = np.array(nodes_1[["pos_x","pos_y","pos_z"]])
    points_2 = np.array(nodes_2[["pos_x","pos_y","pos_z"]])

    kd_1 = KDTree(points_1)
    kd_2 = KDTree(points_2)

    dist_mat_sparse = kd_1.sparse_distance_matrix(kd_2, th)

    return dist_mat_sparse


def label_dual(D, node_lab = None, overrule = None):
    # combines hashval to class
    if node_lab is None:
         node_lab =  {}
         class_label = len(list(node_lab.keys()))-1
    else:
        class_label = -1

    # combines class label to type incident node 
    node_class_comb = {}

    class_label_list = []
    for k, node in D.nodes.items():


        if k[0][-1] == overrule or k[1][-1] == overrule:
            hashval = hash(overrule) + hash(overrule)

        else:
            hashval = hash(k[0][-1]) + hash(k[1][-1])

        try:
            class_label_list.append(node_lab[hashval])

        except KeyError:
            class_label = class_label +1
            node_class_comb[class_label] = (k[0][-1], k[1][-1])
            node_lab[hashval] = class_label
            class_label_list.append(node_lab[hashval])

    return class_label_list, node_lab, node_class_comb



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


def vvg_to_df(vvg_path):
    # Opening JSON file
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

        for j in i["skeletonVoxels"]:
            positions.append(np.array(j["pos"]))
        pos_col.append(positions)


    d = {'id_col': id_col,'pos_col' : pos_col, "node1_col" : node1_col, "node2_col" : node2_col}
    df = pd.DataFrame(d)
    df.set_index('id_col')
    return df



def find_pixel_pos(voxel_size, position, mask_center):

    offset = np.rint(np.array(position)/ voxel_size)
    node_pos_idx = offset + mask_center
    node_pos_idx = node_pos_idx.astype("int")

    return node_pos_idx



def label_edges_centerline(mask_list, voxel_size, df_centerline, mask_center = None):
    no_classes = len(mask_list)
    label_dict = {}
    label_dict_node_ids = {}
    sha = np.array(mask_list[0].shape)
    if mask_center is None:
         mask_center = sha/2


    for idx, row in df_centerline.iterrows():
        positions = row["pos_col"]
        mask_counts = np.zeros((no_classes,))

        for i in range(len(positions)):
            pos = find_pixel_pos(voxel_size, positions[i], mask_center)
            for i, mask in enumerate(mask_list):
                val = mask[pos[0], pos[1], pos[2]] > 0
                mask_counts [i] += val
        
        label_dict[idx] = np.argmax(mask_counts)
        label_dict_node_ids[(row["node1_col"],row["node2_col"])] = np.argmax(mask_counts)

    return label_dict, label_dict_node_ids