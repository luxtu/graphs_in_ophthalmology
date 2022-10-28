import pandas as pd
import trimesh
import preprocessing as pp
from visualization import graph_to_mesh, mesh_viewer

nodesFileNerve =  "~/Documents/Intestine/nerve-mask/nodes_nerve_bs2.csv"
edgesFileNerve = "~/Documents/Intestine/nerve-mask/edges_nerve_bs2.csv"

nodesFileLymph =  "~/Documents/Intestine/lymph-mask/nodes_lymph_bs2.csv"
edgesFileLymph = "~/Documents/Intestine/lymph-mask/edges_lymph_bs2.csv"



nodesN = pd.read_csv(nodesFileNerve, sep = ";", index_col= "id")
edgesN = pd.read_csv(edgesFileNerve, sep = ";", index_col= "id")
nodesN = pp.scalePosition(nodesN, (1.65,1.65,6))


nodesL = pd.read_csv(nodesFileLymph, sep = ";", index_col= "id")
edgesL = pd.read_csv(edgesFileLymph, sep = ";", index_col= "id")
nodesL = pp.scalePosition(nodesL, (1.65,1.65,6))


new_edges = pp.connection_edges(nodesN, nodesL, th = 0.03)

print(len(new_edges))

edge_list =  graph_to_mesh.createEdgesFromList(new_edges)
meshConnection = trimesh.util.concatenate(edge_list)


meshNerve = graph_to_mesh.createMesh(nodesN, edgesN)
meshLymph = graph_to_mesh.createMesh(nodesL, edgesL)

mesh_viewer.show3_trimeshs(meshNerve, meshLymph, meshConnection)
