from visualization import mesh_viewer, volume_to_mesh
import preprocessing.preprocessing as pp
import tifffile as tiff
import numpy as np 
import pandas as pd
import pyvista
import networkx as nx
import trimesh
import nibabel as nib



# load the raw data for the masks
#nerve_mask =  "../Intestine/nerve-mask/nerve_mask_stack_255_fh.tif"
#lymph_mask =  "../Intestine/lymph-mask/lymph_mask_stack_255_fh.tif"
#
## read tif and convert to numpy 
#nerve_mask_np = np.array(tiff.imread(nerve_mask)).T
#lymph_mask_np = np.array(tiff.imread(lymph_mask)).T


#mask_100 =  "../skull_compare/left_right/Mcao2new_probs_bin0p48_rotRotFlip_VR-skull-masked_LR-masked.nis100_fill_hole_ones255.nii.gz"
#mask_200 =  "../skull_compare/left_right/lymph_mask_stack_255_fh.tif"
#img = nib.load(mask_100)
#a = np.array(img.dataobj, dtype = "uint8")


#mask_100_nodes = "../skull_compare/mcao2results/mcao2_100_nodes.csv"
#mask_100_edges = "../skull_compare/mcao2results/mcao2_100_edges.csv"

mask_100_nodes = "../skull_compare/sham2results/sham2downsamp/sham2_100_nodes_downsamp.csv"
mask_100_edges = "../skull_compare/sham2results/sham2downsamp/sham2_100_edges_downsamp.csv"

#mask_100_nodes = "../skull_compare/sham2results/sham2_100_nodes.csv"
#mask_100_edges = "../skull_compare/sham2results/sham2_100_edges.csv"


mask_100_nodes_pd = pd.read_csv(mask_100_nodes, sep = ";", index_col= "id")
mask_100_edges_pd = pd.read_csv(mask_100_edges, sep = ";", index_col= "id")

#mask_100_nodes_pd_scaled = pp.scale_position(mask_100_nodes_pd, (1.625,1.625,6))
#mask_100_nodes_pd_scaled = pp.scale_position(mask_100_nodes_pd, (1,1,1))
G_mcao2_100_scaled = pp.create_graph(mask_100_nodes_pd, mask_100_edges_pd, "n")
#mask_100_nodes_pd_scaled = pp.scale_position(mask_100_nodes_pd, (1.625,1625,6))


G_mcao2_100 = pp.create_graph(mask_100_nodes, mask_100_edges, "n")

#mask_200_nodes = "../skull_compare/sham2results/sham2_200_nodes.csv"
#mask_200_edges = "../skull_compare/sham2results/sham2_200_edges.csv"

mask_200_nodes = "../skull_compare/sham2results/sham2downsamp/sham2_200_nodes_downsamp.csv"
mask_200_edges = "../skull_compare/sham2results/sham2downsamp/sham2_200_edges_downsamp.csv"


mask_200_nodes_pd = pd.read_csv(mask_200_nodes, sep = ";", index_col= "id")
mask_200_edges_pd = pd.read_csv(mask_200_edges, sep = ";", index_col= "id")

#mask_200_nodes_pd_scaled = pp.scale_position(mask_200_nodes_pd, (1.625,1.625,6))
#mask_200_nodes_pd_scaled = pp.scale_position(mask_200_nodes_pd, (1,1,1))
G_mcao2_200_scaled = pp.create_graph(mask_200_nodes_pd, mask_200_edges_pd, "l")


G_mcao2_200 = pp.create_graph(mask_200_nodes, mask_200_edges, "l")


#cont_edges = "../skull_compare/CD1-E-no1_iso3um_stitched_segmentation_bulge_size_3.0_edges.csv"
#cont_nodes = "../skull_compare/CD1-E-no1_iso3um_stitched_segmentation_bulge_size_3.0_nodes.csv"
#
#cont_nodes_pd = pd.read_csv(cont_nodes, sep = ";", index_col= "id")
#cont_edges_pd = pd.read_csv(cont_edges, sep = ";", index_col= "id")
#
#
#G_cont = pp.create_graph(cont_nodes_pd.sample(1000, axis = 0), cont_edges_pd.sample(0, axis = 0), "n")


#nodes_all =np.array(list(G_cont.nodes()))

#nodes_all_sample = np.random.choice(nodes_all, size=10000, replace=False, p=None)

#G_cont_subG = nx.subgraph(G_cont, nodes_all_sample)



#mask_100_nodes = "../skull_compare/left_right/naive2_100_nodes.csv"
#mask_100_edges = "../skull_compare/left_right/naive2_100_edges.csv"
#
#G_naive2_100 = pp.create_graph(mask_100_nodes, mask_100_edges, "n")
#
#mask_200_nodes = "../skull_compare/left_right/naive2_200_nodes.csv"
#mask_200_edges = "../skull_compare/left_right/naive2_200_edges.csv"
#
#G_naive2_200 = pp.create_graph(mask_200_nodes, mask_200_edges, "l")

#mcao2_100_mesh = volume_to_mesh.meshFromVolume(a.astype(bool), (1,1,1), 10, 10)


# load the pickled graph
#G_load = nx.read_gpickle("saved_data/graph_gt_pickle")

# create/ load the renderings for the mask 
#spacing = (1.625*0.00217391, 1.625*0.00217391, 6*0.00217391)  
#
#nerve_mesh = volume_to_mesh.meshFromVolume(nerve_mask_np, spacing, 10, 10, save_stl= True, stl_name = "Nerve_mask_mesh")
#lymph_mesh = volume_to_mesh.meshFromVolume(lymph_mask_np, spacing, 10, 10, save_stl= True, stl_name = "Lymph_mask_mesh")

#nerve_mesh = volume_to_mesh.meshFromVolume(nerve_mask_np, spacing, 10, 10, save_stl= True, stl_name = "Nerve_mask_mesh")
#lymph_mesh = volume_to_mesh.meshFromVolume(lymph_mask_np, spacing, 10, 10, save_stl= True, stl_name = "Lymph_mask_mesh")

#nerve_mesh = trimesh.load_mesh('saved_data/Nerve_mask_mesh.stl')
#lymph_mesh = trimesh.load_mesh('saved_data/Lymph_mask_mesh.stl')


# actors for the graph
actor_list = mesh_viewer.renderNXGraph(G_mcao2_100_scaled, get_actors=True)
actor_list2 = mesh_viewer.renderNXGraph(G_mcao2_200_scaled, get_actors=True)

#actor_list = mesh_viewer.renderNXGraph(G_naive2_100, get_actors=True)
#actor_list2 = mesh_viewer.renderNXGraph(G_naive2_200, get_actors=True)

# set a global theme for plotting
pyvista.set_plot_theme('paraview')
p = pyvista.Plotter()


# add the mesh for lymph and nerve mask
#p.add_mesh(nerve_mesh, color="Red", opacity=0.5)
#p.add_mesh(lymph_mesh, color="Yellow", opacity=0.5)

#p.add_mesh(mcao2_100_mesh, color="Yellow", opacity=0.5)


# add the actors for the graph
for actor in actor_list:
    p.add_actor(actor)
for actor in actor_list2:
    p.add_actor(actor)
p.show()



