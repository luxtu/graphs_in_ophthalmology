import numpy as np
import trimesh
import multiprocessing
from tqdm import tqdm
from visualization import util




def createEdge(edge_extended):
    """ Creates a mesh for an edges that is connecting two nodes. The mesh represents a cylinder with fixed radius.
    :edge_extended: An object that contains the information about start and end coordinates of the edge.

    :return meshE: The mesh for the corresponding edge.
    """
    edge_rad = 0.0025

    x_cent, y_cent, z_cent = (edge_extended["pos_x_x"] + edge_extended["pos_x_y"])/2, (edge_extended["pos_y_x"] + edge_extended["pos_y_y"])/2, (edge_extended["pos_z_x"] + edge_extended["pos_z_y"])/2
    x_dir, y_dir, z_dir = edge_extended["pos_x_x"] - edge_extended["pos_x_y"], edge_extended["pos_y_x"] - edge_extended["pos_y_y"], edge_extended["pos_z_x"] - edge_extended["pos_z_y"]

    v_hat = (x_dir, y_dir, z_dir) / np.linalg.norm((x_dir, y_dir, z_dir))
    v_hat = np.array(([v_hat[0]],[v_hat[1]],[v_hat[2]])).T

    rot = util.rotation_matrix_from_vectors((0,0,1),v_hat)
    mat = util.transform_from_rot_trans(rot, (x_cent, y_cent, z_cent))
    meshE = trimesh.primitives.Cylinder(radius = edge_rad, height = np.linalg.norm((x_dir, y_dir, z_dir)) , transform = mat)

    return meshE



def createNodesFromList(nodes):
    """ A function that returns a list of meshes for each node in a pandas dataframe.
    nodes: A pandas dataframe with the node information

    return meshList: A list of meshes for all the provided nodes.
    """
    meshList = []
    node_rad = 0.005
    for idxN, node in nodes.iterrows():
        mesh = trimesh.primitives.Sphere(center = (node["pos_x"], node["pos_y"], node["pos_z"]), radius = node_rad)
        meshList.append(mesh)
    return meshList




def createEdgesFromList(edges):
    """ Returns a list of meshes for each edge in a pandas dataframe.
    edges: A pandas dataframe with the edge information. Edge information must contain start and end coordinates. 

    return meshList: A list of meshes for all the provided edges.
    """
    meshList = []
    for idxN, edge in edges.iterrows():
        if edge["node1id"]!= edge["node2id"]:
            mesh = createEdge(edge)
            meshList.append(mesh)
    return meshList
    
    
    
    

def connectedComponentMeshMulti(args):
    """ Creates a trimesh for a single connected component.
    Parameters
    ----------
    args: A iterable with 2 objects, node information and egde information

    Returns
    -------
    meshGlob: A single trimesh object that represents the graph of a single connected component
    """

    nodes_comp = args[0]
    edges_comp = args[1]
    
    outputs_nodes = createNodesFromList(nodes_comp)
    outputs_edges = createEdgesFromList(edges_comp)
    
    work_nodes = []
    work_edges = []
    workSplit_nodes = np.array_split(outputs_nodes,4)
    workSplit_edges = np.array_split(outputs_edges,4)
    
    for i in workSplit_nodes:
        if len(i) >0:
            work_nodes.append(i)
            
    for i in workSplit_edges:
        if len(i) >0:
            work_edges.append(i)
            
    final_list = []    
                       
    pool = multiprocessing.Pool(4)
    outputs_mesh_list_n = pool.map(trimesh.util.concatenate, work_nodes)
    pool.close()
    pool.join()
    if len(outputs_mesh_list_n)>0:
        meshGlobNodes = trimesh.util.concatenate(outputs_mesh_list_n)
        final_list.append(meshGlobNodes)
    
    
    pool = multiprocessing.Pool(4)
    outputs_mesh_list_e = pool.map(trimesh.util.concatenate, work_edges)
    pool.close()
    pool.join()
    if len(outputs_mesh_list_e) >0:
        meshGlobEdges = trimesh.util.concatenate(outputs_mesh_list_e)
        final_list.append(meshGlobEdges)
        
    meshGlob = trimesh.util.concatenate(final_list)    
    
    return meshGlob




def createMesh(nodes, edges, cc = True):
    """ Creates a trimesh object corresponding to the information provided in node and edge files.
    Parameters
    ----------
    nodes: A pandas dataframe with the node information
    edges: A pandas dataframe with the edge information.

    Returns
    -------
    meshFinal: A single trimesh object that represents the full graph
    """

    # finds connected components with corresponding edge/node sets
    if cc:
        numComp, setsNodes, setsEdges = util.findConnectedComponents(nodes, edges)
    else:
        numComp, setsNodes, setsEdges = 1, [np.arange(nodes.shape[0])], [np.arange(edges.shape[0])]

    # creates a task for each connected component
    work = []
    for i in range(numComp):
        edges_extended = util.edges_with_node_info(nodes.loc[setsNodes[i]], edges.loc[setsEdges[i]])
        work.append((nodes.loc[setsNodes[i]],edges_extended))
    
    # puts the result for every single connected compoennt in a list
    meshList = []    
    pbar = tqdm(total=numComp)
    pbar.set_description("Processing " + str(numComp)+ " connected components")
    for i in range(numComp):
        resMesh = connectedComponentMeshMulti(work[i])
        pbar.update(1)
        meshList.append(resMesh)
    pbar.close()

    # combines the meshes from all connected components
    meshFinal = trimesh.util.concatenate(meshList)
    
    return meshFinal



def createNodesNx(G, nodes, node_rad):
    """ A function that returns a list of meshes for each node in a pandas dataframe.
    Parameters
    ----------
    nodes: A pandas dataframe with the node information

    Returns
    ----------
    meshList: A list of meshes for all the provided nodes.
    """
    meshList = []

    pbar = tqdm(total=len(nodes))
    pbar.set_description("Creating " + str(len(nodes))+ " nodes")
    for node in nodes:
        pos = G.nodes[node]["pos"]
        mesh = trimesh.primitives.Sphere(center = pos, radius = node_rad)
        meshList.append(mesh)
        pbar.update(1)
    pbar.close()
    return meshList



def nodeMeshesNx(G, concat = True, mask = None, node_rad = 0.005):
    """
    Parameters
    ----------
    Returns
    -------
    """
    if mask is not None:
        node_lis = np.array(G.nodes)[mask]
    else:
        node_lis = np.array(G.nodes)

    node_types = [elem[-1] for elem in node_lis]
    unique_nodes = np.unique(node_types)

    nodes_meshes_identifiers = []
    nodes_meshes = []
    for label in unique_nodes:
        mask = [elem == label for elem in node_types]
        meshList = createNodesNx(G, node_lis[mask],node_rad)
        if concat:
            mesh = util.parallelConcat(meshList)
        else:
            mesh = meshList
            meshIdentList = node_lis[mask]
            nodes_meshes_identifiers.append(meshIdentList)
        nodes_meshes.append(mesh)


    return nodes_meshes, unique_nodes, nodes_meshes_identifiers



def createEdgeNx(G, edge, edge_rad):
    """
    Parameters
    ----------
    Returns
    -------
    """

    node1_pos = np.array(G.nodes[edge[0]]["pos"])
    node2_pos = np.array(G.nodes[edge[1]]["pos"])

    cent = (node1_pos+ node2_pos)/2
    dir = node1_pos - node2_pos

    v_hat = dir / np.linalg.norm(dir)
    v_hat = np.array(([v_hat[0]],[v_hat[1]],[v_hat[2]])).T

    rot = util.rotation_matrix_from_vectors((0,0,1),v_hat)
    mat = util.transform_from_rot_trans(rot, cent)
    meshE = trimesh.primitives.Cylinder(radius = edge_rad, height = np.linalg.norm(dir) , transform = mat)

    return meshE




def createEdgesNx(G, edges, edge_rad):
    """
    Parameters
    ----------
    Returns
    -------
    """
    meshList = []
    pbar = tqdm(total=len(edges))
    pbar.set_description("Creating " + str(len(edges))+ " edges")
    for edge in edges:
        if edge[0]!= edge[1]:
            mesh = createEdgeNx(G, edge, edge_rad)
            meshList.append(mesh)
        pbar.update(1)
    pbar.close()
    return meshList



def edgeMeshesNx(G, concat = True, edge_rad = 0.0025):
    """
    Parameters
    ----------
    Returns
    -------
    """
    edgesList = list(G.edges)
    hash_dict = {}
    edge_dict = {}
    for edge in edgesList:
        key = hash(edge[0][-1]) + hash(edge[1][-1])
        try:
            edge_dict[key].add(edge)
        except KeyError:
            edge_dict[key] = set()
            hash_dict[key] = [edge[0][-1], edge[1][-1]]
    
    unqiue_edges = []
    edge_meshes = [] 
    for k, edgeType in edge_dict.items():
        meshList = createEdgesNx(G, edgeType, edge_rad)
        if concat:
            mesh = util.parallelConcat(meshList)
        else:
            mesh = meshList
        edge_meshes.append(mesh)
        unqiue_edges.append(hash_dict[k])

    return edge_meshes,unqiue_edges


def createMeshNX(G, combine = False):
    """ A function that creates a trimesh object corresponding to a provided networkX graph.

    Parameters
    ----------
    G: A networkX graph
    combine: boolean indicating if node and edge mesh should be combined or not.

    Returns
    -------

    """
    nodeMeshes, unqiue_nodes = nodeMeshesNx(G)
    edgeMeshes, unqiue_nodes = edgeMeshesNx(G)

    if not combine:
        return nodeMeshes, edgeMeshes

    else:
        return trimesh.util.concat(nodeMeshes+edgeMeshes)




