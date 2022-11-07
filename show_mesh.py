import pandas as pd
import preprocessing as pp
from visualization import graph_to_mesh, mesh_viewer



#nodesFile =  "~/Documents/skull_compare/Mcao3_new/nodes_mcao3_fill_ones.csv"
#edgesFile = "~/Documents/skull_compare/Mcao3_new/edges_mcao3_fill_ones.csv"

nodesFile =  "~/Documents/Intestine/combined-mask/nodes_bs2_fh.csv"
edgesFile = "~/Documents/Intestine/combined-mask/edges_bs2_fh.csv"
#
#nodesFile =  "~/Documents/Intestine/nerve-mask/nodes_nerve_bs2_fh.csv"
#edgesFile = "~/Documents/Intestine/nerve-mask/edges_nerve_bs2_fh.csv"

#nodesFile =  "~/Documents/Intestine/lymph-mask/nodes_lymph_bs2_fh.csv"
#edgesFile = "~/Documents/Intestine/lymph-mask/edges_lymph_bs2_fh.csv"



nodes = pd.read_csv(nodesFile, sep = ";", index_col= "id")
edges = pd.read_csv(edgesFile, sep = ";", index_col= "id")


print(nodes.shape[0])
print(edges.shape[0])

nodes = pp.scalePosition(nodes, (1.65,1.65,6))

mesh_viewer.renderGraph(nodes, edges, vtk = 0)



#meshFinal = graph_to_mesh.createMesh(nodes, edges)
#mesh_viewer.showList_trimesh([meshFinal])