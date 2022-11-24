from skimage import measure
import trimesh
import numpy as np







def meshFromVolume(volume, spacing : tuple, iterations: int, simp_fact: int, save_stl: bool = False, stl_name: str = None):
    """ Constructs a trimesh based on a binary volume. Used the marching cubes algorithm, an alternative could be surfac nets.

    Paramters
    ---------
    volume: A numpy array representing a volume
    spacing: Spacing for the coordinate space
    iterations: Number of iterations for the smoothing
    simp_factor: A factor for division that results in the desired size of the mesh after simplification
    save_json: Bool indicating if a json file should be saved or not.

    Returns
    mesh: The created mesh
    -------

    """
    # create a mesh with the marching cubes algortihm
    verts, faces, normals, values = measure.marching_cubes(volume, spacing = spacing)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, normals = normals)

    #find the translation to align with the graph mesh
    tranlsation = volume.shape * np.array(spacing) *-1/2
       
    # apply the transformation
    mesh.apply_translation(tranlsation)

    #smooth the created mesh with laplacian smoothing
    trimesh.smoothing.filter_laplacian(mesh, lamb=0.5, iterations=iterations)

    mesh = mesh.simplify_quadratic_decimation(mesh.vertices.shape[0]/simp_fact)


    if save_stl:
        mesh.export(stl_name + '.stl')

    return mesh
