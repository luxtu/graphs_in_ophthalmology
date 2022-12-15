from visualization import mesh_viewer, volume_to_mesh
import tifffile as tiff
import nibabel as nib
import graph_matching.graph_matching as gm
import preprocessing.preprocessing as pp
import pandas as pd
import numpy as np 
import pyvista
import networkx as nx
import trimesh


nodesFileComb =  "../Intestine/segmentation_full/node_seg_full.csv"
edgesFileComb = "../Intestine/segmentation_full/edge_seg_full.csv"

nodes_c = pd.read_csv(nodesFileComb, sep = ";", index_col= "id")
edges_c = pd.read_csv(edgesFileComb, sep = ";", index_col= "id")

# scaling with the factors provided by luciano
nodes_c = pp.scale_position(nodes_c, (1.625,1.625,6))


nodes_c["pos_x"] = nodes_c["pos_x"]*-1
nodes_c["pos_y"] = nodes_c["pos_y"]*-1
# somehow x and y got only negative positions
print(np.max(nodes_c["pos_x"]))
print(np.min(nodes_c["pos_x"]))

print(np.max(nodes_c["pos_y"]))
print(np.min(nodes_c["pos_y"]))

print(np.max(nodes_c["pos_z"]))
print(np.min(nodes_c["pos_z"]))


G_comb = pp.create_graph(nodes_c, edges_c)
G_einf_comb = pp.to_einfach(G_comb, self_loops = False, isolates = False)

# load the raw data for the masks
nerve_mask =  "../Intestine/segmentation_full/large_C01.nii.nii"
lymph_mask =  "../Intestine/segmentation_full/large_C02.nii.nii"

# read tif and convert to numpy 
nerve_mask_img = nib.load(nerve_mask)
lymph_mask_img = nib.load(lymph_mask)


nerve_mask_np =  np.asarray(nerve_mask_img.dataobj, dtype=bool)
lymph_mask_np =  np.asarray(lymph_mask_img.dataobj, dtype=bool)

print(nerve_mask_np.shape)

#takes forever if the data is not in the RAM
# create labeled graph
mask_labels_num = gm.assignNodeLabelsByMask([nerve_mask_np, lymph_mask_np], G = G_einf_comb, voxel_size= (1,1,1), scaling_vector= (1.625, 1.625, 6), kernel_size=9, mask_center= (0,0,0),assign_type = "mix") # mask_center= (5852,3457,0)

# create dictionary to convert int labels to char labels
char_class_reverse = {}
char_class_reverse[-1] = "f"
char_class_reverse[0] = "n"
char_class_reverse[1] = "l"
char_class_reverse[2] = "r"

# create a copy that will be used for relabeling
mask_labels = mask_labels_num.copy()

# adjust the node names
for key in mask_labels:
    mask_labels[key] = str(key) + char_class_reverse[mask_labels[key]]

# relabel the graph with the mask based labeling
G_comb_einf_lab = nx.relabel_nodes(G_einf_comb, mask_labels)

print(np.unique(np.array(list(mask_labels_num.values())), return_counts = True))

# graph
mesh_viewer.renderNXGraph(G_comb_einf_lab)
