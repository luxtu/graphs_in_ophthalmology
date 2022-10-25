import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm


def createGraph(nodesFile, edgesFile):
    """ Creates an networkX undirected multigraph from provided csv files.
    :nodesFile: A csv file containing information about the nodes
    :edgesFile: A csv file containing information about the edges

    :return G: A networkX multigraph object
    """
    nodes = pd.read_csv(nodesFile, sep = ";", index_col= "id")
    edges = pd.read_csv(edgesFile, sep = ";", index_col= "id")
    # create undirected graph 
    G = nx.MultiGraph()

    # add the vertices
    for idxN, node in nodes.iterrows():
        G.add_node(idxN, x = (), pos = (float(node["pos_x"]),float(node["pos_y"]), float(node["pos_z"])))

    # add the edges
    for idxE, edge in edges.iterrows():
        G.add_edge(edge["node1id"], edge["node2id"], x = (edge["distance"], edge["length"], edge["minRadiusAvg"], edge["curveness"], edge["avgRadiusStd"], edge["num_voxels"]))

    return G



def convertToEinfach(G_multi):
    """ Creates an networkX simple graph from a networkX multigraph. Also removes all isolated nodes and self loops
    :G_multi: A networkX graph object 

    :return G: A simple networkX graph without selfloops, parallel edges and isolated nodes.
    """
    G_einfach = nx.Graph(G_multi)
    G_einfach.remove_edges_from(list(nx.selfloop_edges(G_einfach)))
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

        feature_dict[node] = (float(tot_vol), float(tot_len), float(np.mean(avg_vol_l)), float(np.mean(avg_len_l)), float(np.mean(curveness)), float(np.mean(avg_rad_sd)), float(tot_voxel), float(tot_dis), float(np.mean(avg_dis)))
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