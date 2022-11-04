import pandas as pd
import preprocessing as pp
from visualization import graph_to_mesh, mesh_viewer



#nodesFile =  "~/Documents/skull_compare/Mcao3_new/nodes_mcao3_fill_ones.csv"
#edgesFile = "~/Documents/skull_compare/Mcao3_new/edges_mcao3_fill_ones.csv"

nodesFile =  "~/Documents/skull_compare/Sham2new/nodes_sham2_new_fh.csv"
edgesFile = "~/Documents/skull_compare/Sham2new/edges_sham2_new_fh.csv"



nodes = pd.read_csv(nodesFile, sep = ";", index_col= "id")
edges = pd.read_csv(edgesFile, sep = ";", index_col= "id")

nodes = pp.scalePosition(nodes, (1.65,1.65,6))


mesh_viewer.renderGraph(nodes, edges, vtk = 0, gif = "sham2new_gif")