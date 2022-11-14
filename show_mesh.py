import pandas as pd
import preprocessing as pp
from visualization import graph_to_mesh, mesh_viewer
import tifffile as tiff
import numpy as np 
import pyvista
import networkx as nx


nerve_mask =  "../Intestine/nerve-mask/nerve_mask_stack_255_fh.tif"
lymph_mask =  "../Intestine/lymph-mask/lymph_mask_stack_255_fh.tif"

# read tif and convert to numpy 
nerve_mask_np = np.array(tiff.imread(nerve_mask)).T
lymph_mask_np = np.array(tiff.imread(lymph_mask)).T

# load the pickled graph
G_load = nx.read_gpickle("graph_pickle")


pyvista.global_theme.edge_color = 'blue'
pyvista.set_plot_theme('paraview')

# Create the spatial reference
grid_N = pyvista.UniformGrid()
grid_L = pyvista.UniformGrid()

# Set the grid dimensions: shape + 1 because we want to inject our values on
#   the CELL data
grid_N.dimensions = np.array(nerve_mask_np.shape) + 1
grid_L.dimensions = np.array(lymph_mask_np.shape) + 1

# Edit the spatial reference
grid_N.origin = (0, 0, 0)  # The bottom left corner of the data set
grid_N.spacing = (1.625*0.00217391, 1.625*0.00217391, 6*0.00217391)  # These are the cell sizes along each axis
grid_L.origin = (0, 0, 0)  # The bottom left corner of the data set
grid_L.spacing = (1.625*0.00217391, 1.625*0.00217391, 6*0.00217391)  # These are the cell sizes along each axis

# Add the data values to the cell data
grid_N.cell_data["values"] = nerve_mask_np.flatten(order="F")  # Flatten the array!
grid_L.cell_data["values"] = lymph_mask_np.flatten(order="F")  # Flatten the array!

# Now plot the grid!
#grid.plot(show_edges=True)


opacity = [0, 0.1]
nerve_mask
p = pyvista.Plotter()
p.add_volume(grid_N, cmap="Reds", opacity=opacity)
p.add_volume(grid_L, cmap="Greens", opacity=opacity)
p.show()