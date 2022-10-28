import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial import KDTree
from scipy.sparse import csr_matrix
from scipy.sparse import dok_array


def createGraph(nodesFile, edgesFile, index_addon = None):
    """ Creates an networkX undirected multigraph from provided csv files.
    :nodesFile: A csv file containing information about the nodes
    :edgesFile: A csv file containing information about the edges
    :id: an identifier to avoid ambiguous combinations of graphs

    :return G: A networkX multigraph object
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
            G.add_edge(str(int(edge["node1id"]))  + index_addon , str(int(edge["node2id"])) + index_addon, x = (edge["distance"], edge["length"], edge["minRadiusAvg"], edge["curveness"], edge["avgRadiusStd"], edge["num_voxels"]))

    else:
        for idxN, node in nodes.iterrows():
            G.add_node(idxN, x = (), pos = (float(node["pos_x"]),float(node["pos_y"]), float(node["pos_z"])))

        # add the edges
        for idxE, edge in edges.iterrows():
            G.add_edge(edge["node1id"],edge["node2id"], x = (edge["distance"], edge["length"], edge["minRadiusAvg"], edge["curveness"], edge["avgRadiusStd"], edge["num_voxels"]))

    return G



def convertToEinfach(G_multi, self_loops = False, isolates = False):
    """ Creates an networkX simple graph from a networkX multigraph. Also removes all isolated nodes and self loops
    :G_multi: A networkX graph object 

    :return G: A simple networkX graph without selfloops, parallel edges and isolated nodes.
    """
    G_einfach = nx.Graph(G_multi)
    if not self_loops:
        G_einfach.remove_edges_from(list(nx.selfloop_edges(G_einfach)))
    if not isolates:
        G_einfach.remove_nodes_from(list(nx.isolates(G_einfach)))

    return G_einfach



def enrichNodeAttributes(G):
    """ Enriches the x. attribute of the nodes of a networkX graph with features extracted from the incident edges.
    :G: A networkX graph object 
    
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


def graphSummary(G):
    """ A method that prints out a quick summary of the characteristics of a networkX graph.
    :G: A networkX graph object 

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


def scalePosition(df, scaleVector):
    df["pos_x"] = df["pos_x"]*scaleVector[0]
    df["pos_y"] = df["pos_y"]*scaleVector[1]
    df["pos_z"] = df["pos_z"]*scaleVector[2]

    return df


def connected_components_dict(labels):
    con_comp = {}
    for i in range(len(labels)):
        try:
            con_comp[labels[i]].add(i)
        except KeyError:
            con_comp[labels[i]] = {i}

    return con_comp



def relevant_connected_components(con_comp, node1_size, labels, rel_th = 1):
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
    idx = list(nodes.index)
    idx_new = [str(elem) + index_addon for elem in idx]
    nodes.index = idx_new

    edges['node1id'] = edges['node1id'].apply(lambda x: str(x) + index_addon)
    edges['node2id'] = edges['node2id'].apply(lambda x: str(x) + index_addon) 

    return edges, nodes


def distance_based_adjacency(nodes_1, nodes_2, th):

    dist_mat_sparse = network_sparse_distance_matrix(nodes_1, nodes_2, th)
    adjM = dok_array((nodes_1.shape[0]+ nodes_2.shape[0], nodes_1.shape[0]+ nodes_2.shape[0]), dtype=np.uint8)

    for key in dist_mat_sparse.keys():
        adjM[key[0], key[1] + nodes_1.shape[0]] = 1
        adjM[key[1]+ nodes_1.shape[0], key[0]] = 1

    adjMcsr = adjM.tocsr()

    return adjMcsr


def network_sparse_distance_matrix(nodes_1, nodes_2, th):

    points_1 = np.array(nodes_1[["pos_x","pos_y","pos_z"]])
    points_2 = np.array(nodes_2[["pos_x","pos_y","pos_z"]])

    kd_1 = KDTree(points_1)
    kd_2 = KDTree(points_2)

    dist_mat_sparse = kd_1.sparse_distance_matrix(kd_2, th)

    return dist_mat_sparse



def connection_edges(nodes_1, nodes_2, th):

    dist_mat_sparse = network_sparse_distance_matrix(nodes_1, nodes_2, th = th)
    indices = dist_mat_sparse.keys()

    df = pd.DataFrame(data = list(indices))

    df.columns = ["node1id", "node2id"]
    new_edges = pd.merge(pd.merge(nodes_1, df, left_on='id', right_on='node1id'),nodes_2, left_on = "node2id", right_on = "id")
    new_edges = new_edges.drop(columns = ['degree_x', 'isAtSampleBorder_x', 'degree_y', 'isAtSampleBorder_y'])

    return new_edges





def contractEdges(rel_edges, G):
    node_transfer = {}
    pbar = tqdm(total=rel_edges.shape[0])
    for idxR, edge in rel_edges.iterrows():
        pbar.update(1)
        try:
            G = nx.contracted_nodes(G, edge["node1id"], edge["node2id"])

            #node = G.nodes[edge["node1id"]]
            #posA = np.array(node["pos"])
            #posB = np.array(node["contraction"][edge["node2id"]]["pos"])
            #node["pos"] = tuple((posA+posB)/2)
            #del node["contraction"]

            node_transfer[edge["node2id"]] = edge["node1id"]
        except KeyError:
            G = nx.contracted_nodes(G, edge["node1id"], node_transfer[edge["node2id"]])

            #node = G.nodes[edge["node1id"]]
            #posA = np.array(node["pos"])
            #posB = np.array(node["contraction"][node_transfer[edge["node2id"]]]["pos"])
            #node["pos"] = tuple((posA+posB)/2)
            #del node["contraction"]

            node_transfer[edge["node2id"]] = edge["node1id"]
    pbar.close()

    # give the connecting edges a new name
    merge_nodes = set(node_transfer.values())
    merge_node_names = np.arange(0, len(merge_nodes))
    merge_node_names = [str(elem)+ "c" for elem in merge_node_names]
    merge_node_dict = dict(zip(merge_nodes, merge_node_names))
    G = nx.relabel_nodes(G, merge_node_dict)


    transfer_dict = {k:set() for k in merge_node_names}
    for k,v in node_transfer.items():
        try:
            key =  merge_node_dict[k]
        except KeyError:
            key = merge_node_dict[node_transfer[k]]
        transfer_dict[key].add(k)
        transfer_dict[key].add(v)

    return G, transfer_dict



def contractedCombinedGraph(G1, G2, nodes1csv, nodes2csv, th = 0.01):
    # creating a new graph by combining the two networks
    G_whole = nx.compose(G1, G2)
    print("Before Contraction")
    graphSummary(G_whole)

    # loading edge informations
    nodes1_df = pd.read_csv(nodes1csv, sep = ";", index_col= "id")
    nodes2_df = pd.read_csv(nodes2csv, sep = ";", index_col= "id")

    # creating a df for the connecting edges
    con_edges = connection_edges(nodes1_df, nodes2_df, th)
    rel_edges = con_edges[["node1id", "node2id"]]

    # make the names unique for each network type
    rel_edges['node1id'] = rel_edges['node1id'].apply(lambda x: str(x) + "n")
    rel_edges['node2id'] = rel_edges['node2id'].apply(lambda x: str(x) + "l") 

    # contract the connecting edges in the new graph
    G_whole, transfer_dict = contractEdges(rel_edges, G_whole)


    print("After Contraction")
    graphSummary(G_whole)

    return G_whole, transfer_dict
