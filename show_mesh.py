from visualization import mesh_viewer
import tifffile as tiff
import numpy as np 
import pyvista
import networkx as nx
import trimesh
from skimage import measure

# load teh raw data for the masks
nerve_mask =  "../Intestine/nerve-mask/nerve_mask_stack_255_fh.tif"
lymph_mask =  "../Intestine/lymph-mask/lymph_mask_stack_255_fh.tif"

# read tif and convert to numpy 
nerve_mask_np = np.array(tiff.imread(nerve_mask)).T
lymph_mask_np = np.array(tiff.imread(lymph_mask)).T

# load the pickled graph
G_load = nx.read_gpickle("graph_gt_pickle")

actor_list = mesh_viewer.renderNXGraph(G_load, get_actors=True)


# create the mesh for the nerve and lymph mask -> alternative could be surface nets
spacing = (1.625*0.00217391, 1.625*0.00217391, 6*0.00217391)  
verts, faces, normals, values = measure.marching_cubes(nerve_mask_np, spacing = (1.625*0.00217391, 1.625*0.00217391, 6*0.00217391))
nerve_mesh = trimesh.Trimesh(vertices=verts, faces=faces, noramls = normals)

spacing = (1.625*0.00217391, 1.625*0.00217391, 6*0.00217391)  
verts, faces, normals, values = measure.marching_cubes(lymph_mask_np, spacing = (1.625*0.00217391, 1.625*0.00217391, 6*0.00217391))
lymph_mesh = trimesh.Trimesh(vertices=verts, faces=faces, noramls = normals)

# find the translation to align with the graph mesh
tranlsation = nerve_mask_np.shape * np.array(spacing) *-1/2

# apply the transformation
nerve_mesh.apply_translation(tranlsation)
lymph_mesh.apply_translation(tranlsation)


# set a global theme for plotting
pyvista.set_plot_theme('paraview')
p = pyvista.Plotter()

# add the mesh for lymph and nerve mask
p.add_mesh(nerve_mesh, color="Red", opacity=0.1)
p.add_mesh(lymph_mesh, color="Yellow", opacity=0.1)

# add the actors for the graph
for actor in actor_list:
    p.add_actor(actor)
p.show()