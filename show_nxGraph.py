import pandas as pd
from scipy.sparse.csgraph import connected_components
import preprocessing.preprocessing as pp
from visualization import graph_to_mesh, mesh_viewer
import graph_matching.graph_matching as gm

nodesFileNerve =  "~/Documents/Intestine/nerve-mask/nodes_nerve_bs2_fh.csv"
edgesFileNerve = "~/Documents/Intestine/nerve-mask/edges_nerve_bs2_fh.csv"

nodesFileLymph =  "~/Documents/Intestine/lymph-mask/nodes_lymph_bs2_fh.csv"
edgesFileLymph = "~/Documents/Intestine/lymph-mask/edges_lymph_bs2_fh.csv"

nodes_n = pd.read_csv(nodesFileNerve, sep = ";", index_col= "id")
edges_n = pd.read_csv(edgesFileNerve, sep = ";", index_col= "id")
nodes_l = pd.read_csv(nodesFileLymph, sep = ";", index_col= "id")
edges_l = pd.read_csv(edgesFileLymph, sep = ";", index_col= "id")


# scaling with the factors provided by luciano
nodes_l = pp.scalePosition(nodes_l, (1.65,1.65,6))
nodes_n = pp.scalePosition(nodes_n, (1.65,1.65,6))


# giving nodes from different files unique names
edges_n, nodes_n = pp.relable_edges_nodes(edges_n, nodes_n, "n")
edges_l, nodes_l = pp.relable_edges_nodes(edges_l, nodes_l, "l")

# contracting very close nodes
adjMcsr = pp.distance_based_adjacency(nodes_n, nodes_l, th = 0.01)
num, labels = connected_components(csgraph=adjMcsr, directed = False)
con_comp = pp.connected_components_dict(labels)
rel_comp = pp.relevant_connected_components(con_comp, nodes_n.shape[0], ("n","l"))

reverse_dict= {}
for k, v in rel_comp.items():
    for val in v:
        reverse_dict[val] = k

merged_nodes = pd.concat([nodes_l.loc[:,["pos_x", "pos_y", "pos_z"]], nodes_n.loc[:,["pos_x", "pos_y", "pos_z"]]])
new_nodes = pd.DataFrame(columns=merged_nodes.columns )


# replace the contracted node with new nodes
# the position of the new node is an average of all the previous nodes
for k, valList in rel_comp.items():
    new_nodes.loc[k] = merged_nodes.loc[valList].mean()
    merged_nodes.drop(valList, inplace = True)


# concat the all nodes and the new nodes
merged_nodes = pd.concat([merged_nodes, new_nodes])

# create a combined edge file
merged_edges = pd.concat([edges_l, edges_n], ignore_index = True)


# change the names of the edges to the new names
for idxE, edge in merged_edges.iterrows():
    if edge["node1id"] in reverse_dict.keys():
        merged_edges.loc[idxE,"node1id"] = reverse_dict[edge["node1id"]]
    if edge["node2id"] in reverse_dict.keys():
        merged_edges.loc[idxE,"node2id"] = reverse_dict[edge["node2id"]]


# create a new graph with contracted nodes
G_contract = pp.createGraph(merged_nodes, merged_edges)
G_contract_einf = pp.convertToEinfach(G_contract, self_loops = False, isolates = False)

###########################################


nodesFileComb =  "~/Documents/Intestine/combined-mask/nodes_bs2_fh.csv"
edgesFileComb = "~/Documents/Intestine/combined-mask/edges_bs2_fh.csv"

nodes_c = pd.read_csv(nodesFileComb, sep = ";", index_col= "id")
edges_c = pd.read_csv(edgesFileComb, sep = ";", index_col= "id")

# scaling with the factors provided by luciano
nodes_c = pp.scalePosition(nodes_c, (1.65,1.65,6))


G_contract_comb = pp.createGraph(nodes_c, edges_c)
G_contract_einf_comb = pp.convertToEinfach(G_contract_comb, self_loops = False, isolates = False)

G_contract_einf_comb_relab = gm.nearestNeighborLabeling(G_contract_einf, G_contract_einf_comb)

#mesh_viewer.renderNXGraph(G_contract_einf_comb_relab, vtk = 0)
###########################################

mesh_viewer.renderNXGraph(G_contract_einf, vtk = 0)


#
#print("now")
#nodeMeshes, unique_nodes = graph_to_mesh.nodeMeshesNx(G_contract_einf)
#edgeMeshes, unique_edges = graph_to_mesh.edgeMeshesNx(G_contract_einf)
#
#print(unique_nodes)
#print(unique_edges)
#
##showList_trimesh
#mesh_viewer.showList_trimesh(nodeMeshes + edgeMeshes)
##mesh_viewer.gifList_trimesh(nodeMeshes + edgeMeshes, "test_animate")







