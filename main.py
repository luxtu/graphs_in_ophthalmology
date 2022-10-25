import pandas as pd
from visualization import graph_to_mesh, mesh_viewer

nodesFileNerve =  "~/Documents/Intestine/nerve-mask/nodes_nerve_bs2.csv"
edgesFileNerve = "~/Documents/Intestine/nerve-mask/edges_nerve_bs2.csv"

nodesFileLymph =  "~/Documents/Intestine/lymph-mask/nodes_lymph_bs2.csv"
edgesFileLymph = "~/Documents/Intestine/lymph-mask/edges_lymph_bs2.csv"



nodesN = pd.read_csv(nodesFileNerve, sep = ";", index_col= "id")
edgesN = pd.read_csv(edgesFileNerve, sep = ";", index_col= "id")

nodesN["pos_x"] = nodesN["pos_x"]*1.65
nodesN["pos_y"] = nodesN["pos_y"]*1.65
nodesN["pos_z"] = nodesN["pos_z"]*6


nodesL = pd.read_csv(nodesFileLymph, sep = ";", index_col= "id")
edgesL = pd.read_csv(edgesFileLymph, sep = ";", index_col= "id")

nodesL["pos_x"] = nodesL["pos_x"]*1.65
nodesL["pos_y"] = nodesL["pos_y"]*1.65
nodesL["pos_z"] = nodesL["pos_z"]*6

meshNerve = graph_to_mesh.createMesh(nodesN, edgesN)
meshLymph = graph_to_mesh.createMesh(nodesL, edgesL)

mesh_viewer.show2_trimeshs(meshNerve, meshLymph)