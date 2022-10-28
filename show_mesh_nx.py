import pandas as pd
from scipy.spatial import KDTree
from scipy.sparse import csr_matrix
from scipy.sparse import dok_array
from scipy.sparse.csgraph import connected_components
import preprocessing as pp
from visualization import graph_to_mesh, mesh_viewer

nodesFileNerve =  "~/Documents/Intestine/nerve-mask/nodes_nerve_bs2.csv"
edgesFileNerve = "~/Documents/Intestine/nerve-mask/edges_nerve_bs2.csv"

nodesFileLymph =  "~/Documents/Intestine/lymph-mask/nodes_lymph_bs2.csv"
edgesFileLymph = "~/Documents/Intestine/lymph-mask/edges_lymph_bs2.csv"

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


adjMcsr = pp.distance_based_adjacency(nodes_n, nodes_l, th = 0.03)
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

# createa a combined edge file
merged_edges = pd.concat([edges_l, edges_n], ignore_index = True)

# change the names of the edges to the new names
for idxE, edge in merged_edges.iterrows():
    if edge["node1id"] in reverse_dict.keys():
        merged_edges.loc[idxE,"node1id"] = reverse_dict[edge["node1id"]]
    if edge["node2id"] in reverse_dict.keys():
        merged_edges.loc[idxE,"node2id"] = reverse_dict[edge["node2id"]]


# create a new graph based on the old information

G_contract = pp.createGraph(merged_nodes, merged_edges)
G_contract_einf = pp.convertToEinfach(G_contract, self_loops = False, isolates = True)
#pp.enrichNodeAttributes(G_contract_einf)


nodeMeshes, unique_nodes = graph_to_mesh.nodeMeshesNx(G_contract_einf)
edgeMeshes, unique_edges = graph_to_mesh.edgeMeshesNx(G_contract_einf)

print(unique_nodes)
print(unique_edges)

mesh_viewer.gifList_trimesh(nodeMeshes + edgeMeshes, "test_animate")



