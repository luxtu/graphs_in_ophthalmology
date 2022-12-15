import pandas as pd 
import preprocessing.preprocessing as pp
import numpy as np




def graphContraction(nodes_1, nodes_2, edges_1, edges_2, connection_dict):


    reverse_dict= {}
    for k, v in connection_dict.items():
        for val in v:
            reverse_dict[val] = k


    merged_nodes = pd.concat([nodes_1.loc[:,["pos_x", "pos_y", "pos_z"]], nodes_2.loc[:,["pos_x", "pos_y", "pos_z"]]])
    new_nodes = pd.DataFrame(columns=merged_nodes.columns )
    
    # replace the contracted node with new nodes
    # the position of the new node is an average of all the previous nodes
    for k, valList in connection_dict.items():
        new_nodes.loc[k] = merged_nodes.loc[valList].mean()
        merged_nodes.drop(valList, inplace = True)
    
    
    # concat the all nodes and the new nodes
    merged_nodes = pd.concat([merged_nodes, new_nodes])
    
    # createa a combined edge file
    merged_edges = pd.concat([edges_1, edges_2], ignore_index = True)
    
    drop_count = 0
    # change the names of the edges to the new names
    for idxE, edge in merged_edges.iterrows():
        if edge["node1id"] in reverse_dict.keys():
            merged_edges.loc[idxE,"node1id"] = reverse_dict[edge["node1id"]]
        if edge["node2id"] in reverse_dict.keys():
            merged_edges.loc[idxE,"node2id"] = reverse_dict[edge["node2id"]]
        if edge["hasNodeAtSampleBorder"]:
            merged_edges.drop([idxE], inplace = True)
            drop_count +=1
    
    
    # create a new graph based on the old information
    G_contract = pp.create_graph(merged_nodes, merged_edges)

    return G_contract




def graphLinking(nodes_1, nodes_2, edges_1, edges_2, sparse_dist_mat):


    merged_edges_link = pd.concat([edges_1, edges_2], ignore_index = True)
    new_links = pd.DataFrame(data = np.zeros((sparse_dist_mat.nnz, len(merged_edges_link.columns))), columns=merged_edges_link.columns)

    median_edge = merged_edges_link.median()
    ld_ratio = median_edge[0] / median_edge[1]

    i = -1
    for key, val in sparse_dist_mat.items():
        i = i+ 1
        new_links.loc[i][2:] = median_edge


    new_links["node1id"] = [nodes_1.iloc[elem[0]].name for elem in sparse_dist_mat.keys()]
    new_links["node2id"] = [nodes_2.iloc[elem[1]].name for elem in sparse_dist_mat.keys()]
    new_links["node1_degree"] = [nodes_1.iloc[elem[0]]["degree"] + 1  for elem in sparse_dist_mat.keys()]
    new_links["node2_degree"] = [nodes_2.iloc[elem[1]]["degree"] + 1  for elem in sparse_dist_mat.keys()]
    new_links["distance"] = [float(val) for val in sparse_dist_mat.values()]
    new_links["length"] = [float(val*ld_ratio) for val in sparse_dist_mat.values()]


    merged_edges_link_f = pd.concat([merged_edges_link, new_links], ignore_index = True)
    merged_nodes_link_f = pd.concat([nodes_1.loc[:,["pos_x", "pos_y", "pos_z"]], nodes_2.loc[:,["pos_x", "pos_y", "pos_z"]]])


    # create a new graph based on the old information
    G_link = pp.create_graph(merged_nodes_link_f, merged_edges_link_f)
    return G_link