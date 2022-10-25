import numpy as np 
import pyvista



def show2_trimeshs(trimesh1, trimesh2):
    """" Creates an interactive visualization of two trimesh objects. 
    :trimesh1/2: Two trimesh objects
    
    """

    mesh1 = pyvista.wrap(trimesh1)
    mesh2 = pyvista.wrap(trimesh2)

    plotter = pyvista.Plotter()

    actor = plotter.add_mesh(mesh1, color='red')
    actor = plotter.add_mesh(mesh2, color='yellow')

    print("File 1 is red")
    print("File 2 is yellow")

    plotter.show()