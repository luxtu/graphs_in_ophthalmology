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


def show3_trimeshs(trimesh1, trimesh2, trimesh3):
    """" Creates an interactive visualization of two trimesh objects. 
    :trimesh1/2: Two trimesh objects
    
    """

    mesh1 = pyvista.wrap(trimesh1)
    mesh2 = pyvista.wrap(trimesh2)
    mesh3 = pyvista.wrap(trimesh3)

    plotter = pyvista.Plotter()

    actor = plotter.add_mesh(mesh1, color='red')
    actor = plotter.add_mesh(mesh2, color='yellow')
    actor = plotter.add_mesh(mesh3, color='green')

    print("File 1 is red")
    print("File 2 is yellow")
    print("File 3 is green")

    plotter.show()