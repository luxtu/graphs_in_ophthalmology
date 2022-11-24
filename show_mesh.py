from visualization import mesh_viewer, volume_to_mesh
import tifffile as tiff
import numpy as np 
import pyvista
import networkx as nx
import trimesh



# load the raw data for the masks
nerve_mask =  "../Intestine/nerve-mask/nerve_mask_stack_255_fh.tif"
lymph_mask =  "../Intestine/lymph-mask/lymph_mask_stack_255_fh.tif"

# read tif and convert to numpy 
nerve_mask_np = np.array(tiff.imread(nerve_mask)).T
lymph_mask_np = np.array(tiff.imread(lymph_mask)).T

# load the pickled graph
G_load = nx.read_gpickle("saved_data/graph_gt_pickle")

# create/ load the renderings for the mask 
#spacing = (1.625*0.00217391, 1.625*0.00217391, 6*0.00217391)  
#
#nerve_mesh = volume_to_mesh.meshFromVolume(nerve_mask_np, spacing, 10, 10, save_stl= True, stl_name = "Nerve_mask_mesh")
#lymph_mesh = volume_to_mesh.meshFromVolume(lymph_mask_np, spacing, 10, 10, save_stl= True, stl_name = "Lymph_mask_mesh")

nerve_mesh = trimesh.load_mesh('saved_data/Nerve_mask_mesh.stl')
lymph_mesh = trimesh.load_mesh('saved_data/Lymph_mask_mesh.stl')


# actors for the graph
actor_list = mesh_viewer.renderNXGraph(G_load, get_actors=True)

# set a global theme for plotting
pyvista.set_plot_theme('paraview')
p = pyvista.Plotter()


# add the mesh for lymph and nerve mask
p.add_mesh(nerve_mesh, color="Red", opacity=0.5)
p.add_mesh(lymph_mesh, color="Yellow", opacity=0.5)


# add the actors for the graph
for actor in actor_list:
    p.add_actor(actor)
p.show()



