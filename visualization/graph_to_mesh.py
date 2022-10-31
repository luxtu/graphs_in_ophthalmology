import numpy as np
import pandas as pd
import trimesh
from scipy import sparse
import multiprocessing
from tqdm import tqdm
import multiprocessing as mp



def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    vec1: A 3d "source" vector
    vec2: A 3d "destination" vector
    return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    
    return rotation_matrix
    


def transform_from_rot_trans(rot_mat, trans_vec):
    """ Creates a 4x4 transformation matrix
    rot_mat: A 3x3 rotation matrix 
    trans_vex: A x,y,z translation vector
    return mat: A 4x4 transformation matrix
    """
    mat = np.eye(4)
    mat[0,3] = trans_vec[0]
    mat[1,3] = trans_vec[1]
    mat[2,3] = trans_vec[2]
    mat[:3,:3] = rot_mat
    
    return mat



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

    rot = rotation_matrix_from_vectors((0,0,1),v_hat)
    mat = transform_from_rot_trans(rot, (x_cent, y_cent, z_cent))
    meshE = trimesh.primitives.Cylinder(radius = edge_rad, height = np.linalg.norm((x_dir, y_dir, z_dir)) , transform = mat)

    return meshE



def findConnectedComponents(nodes, edges):
    """ A function to find all the connected components of a graph and the corresponding sets of nodes and vertices.
    :nodes: A pandas dataframe with the node information
    :edges: A pandas dataframe with the edge information

    :return num: The mesh for the corresponding edge.
    :return sets_nodes: A list of sets that contain the nodes specific to each connected component.
    :return sets_edges: A list of sets that contain the edges specific to each connected component.
    """
    adjNodes = np.zeros((nodes.shape[0], nodes.shape[0]))
    for idx, edge in edges.iterrows():
        n1 = int(edge["node1id"])
        n2 = int(edge["node2id"])
        adjNodes[n1,n2] = 1
        adjNodes[n2,n1] = 1
    num, comp = sparse.csgraph.connected_components(adjNodes, directed=False)
    sets_nodes = []
    sets_edges = []
    for i in range(num):
        sets_nodes.append(set(np.where(comp == i)[0]))
        sets_edges.append(set())
    
    for i in range(len(edges)):
        n1 = int(edges.loc[i]["node1id"])
        n2 = int(edges.loc[i]["node2id"])
        for j in range(num):
            if n1 in sets_nodes[j] or n2 in sets_nodes[j]:
                sets_edges[j].add(i)
               
    return num, sets_nodes, sets_edges
    


def createNodesFromList(nodes):
    """ A function that returns a list of meshes for each node in a pandas dataframe.
    :nodes: A pandas dataframe with the node information

    :return meshList: A list of meshes for all the provided nodes.
    """
    meshList = []
    node_rad = 0.005
    for idxN, node in nodes.iterrows():
        mesh = trimesh.primitives.Sphere(center = (node["pos_x"], node["pos_y"], node["pos_z"]), radius = node_rad)
        meshList.append(mesh)
    return meshList



def createEdgesFromList(edges):
    """ A function that returns a list of meshes for each edge in a pandas dataframe.
    :edges: A pandas dataframe with the edge information. Edge information must contain start and end coordinates. 

    :return meshList: A list of meshes for all the provided edges.
    """
    meshList = []
    for idxN, edge in edges.iterrows():
        if edge["node1id"]!= edge["node2id"]:
            mesh = createEdge(edge)
            meshList.append(mesh)
    return meshList
    
    
    
def edges_with_node_info(nodes, edges):
    """ A function that joins node and edge information.
    :nodes: A pandas dataframe with the node information
    :edges: A pandas dataframe with the edge information.

    :return res: A pandas datframe that contains the start and end coordinates for each edge.
    """

    edges_ext =pd.merge(pd.merge(nodes, edges, left_on='id', right_on='node1id'),nodes, left_on = "node2id", right_on = "id")
    edges_ext = edges_ext.drop(columns = ['degree_x', 'isAtSampleBorder_x',
       'length', 'distance', 'curveness', 'volume',
       'avgCrossSection', 'minRadiusAvg', 'minRadiusStd', 'avgRadiusAvg',
       'avgRadiusStd', 'maxRadiusAvg', 'maxRadiusStd', 'roundnessAvg',
       'roundnessStd', 'node1_degree', 'node2_degree', 'num_voxels',
       'hasNodeAtSampleBorder', 'degree_y',
       'isAtSampleBorder_y'])
    return edges_ext
    
    

def connectedComponentMeshMulti(args):
    """ A function that creates a trimesh for a single connected component.
    :args: A iterable with 2 objects, node information and egde information

    :return meshGlob: A single trimesh object that represents the graph of a single connected component
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




def createMesh(nodes, edges):
    """ A function that creates a trimesh object corresponding to the information provided in node and edge files.
    :nodes: A pandas dataframe with the node information
    :edges: A pandas dataframe with the edge information.

    :return meshFinal: A single trimesh object that represents the full graph
    """

    # finds connected components with corresponding edge/node sets
    numComp, setsNodes, setsEdges = findConnectedComponents(nodes, edges)

    # creates a task for each connected component
    work = []
    for i in range(numComp):
        edges_extended = edges_with_node_info(nodes.loc[setsNodes[i]], edges.loc[setsEdges[i]])
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


    

def createNodesNx(G, nodes):
    """ A function that returns a list of meshes for each node in a pandas dataframe.
    :nodes: A pandas dataframe with the node information

    :return meshList: A list of meshes for all the provided nodes.
    """
    meshList = []
    node_rad = 0.005

    pbar = tqdm(total=len(nodes))
    pbar.set_description("Creating " + str(len(nodes))+ " nodes")
    for node in nodes:
        pos = G.nodes[node]["pos"]
        mesh = trimesh.primitives.Sphere(center = pos, radius = node_rad)
        meshList.append(mesh)
        pbar.update(1)
    return meshList


def nodeMeshesNx(G, concat = True, mask = None):

    if mask is not None:
        node_lis = np.array(G.nodes)[mask]
    else:
        node_lis = np.array(G.nodes)

    node_types = [elem[-1] for elem in node_lis]
    unique_nodes = np.unique(node_types)

    nodes_meshes = []
    for label in unique_nodes:
        mask = [elem == label for elem in node_types]
        meshList = createNodesNx(G, node_lis[mask])
        if concat:
            mesh = parallelConcat(meshList)
        else:
            mesh = meshList
        nodes_meshes.append(mesh)

    return nodes_meshes, unique_nodes



def createEdgeNx(G, edge):

    edge_rad = 0.0025

    node1_pos = np.array(G.nodes[edge[0]]["pos"])
    node2_pos = np.array(G.nodes[edge[1]]["pos"])

    cent = (node1_pos+ node2_pos)/2
    dir = node1_pos - node2_pos

    v_hat = dir / np.linalg.norm(dir)
    v_hat = np.array(([v_hat[0]],[v_hat[1]],[v_hat[2]])).T

    rot = rotation_matrix_from_vectors((0,0,1),v_hat)
    mat = transform_from_rot_trans(rot, cent)
    meshE = trimesh.primitives.Cylinder(radius = edge_rad, height = np.linalg.norm(dir) , transform = mat)

    return meshE

def parallelConcat(meshList):
    if len(meshList) > 40:
        pool = mp.Pool(4)
        work = np.array_split(meshList, 4)
        meshListMap = pool.map(trimesh.util.concatenate,work)
        pool.close()
        pool.join()
        mesh = trimesh.util.concatenate(meshListMap)
    else:
        mesh = trimesh.util.concatenate(meshList)
    return mesh


def createEdgesNx(G, edges):
    meshList = []
    pbar = tqdm(total=len(edges))
    pbar.set_description("Creating " + str(len(edges))+ " edges")
    for edge in edges:
        if edge[0]!= edge[1]:
            mesh = createEdgeNx(G, edge)
            meshList.append(mesh)
        pbar.update(1)
    return meshList


def edgeMeshesNx(G, concat = True):

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
        meshList = createEdgesNx(G, edgeType)
        if concat:
            mesh = parallelConcat(meshList)
        else:
            mesh = meshList
        edge_meshes.append(mesh)
        unqiue_edges.append(hash_dict[k])

    return edge_meshes,unqiue_edges


def createMeshNX(G, combine = False):
    """ A function that creates a trimseh object corresponding to a provided networkX graph.

    Parameters
    ----------
    G: A networkX graph
    combine: boolean indicating if node and edge mesh should be combined or not.


    """
    nodeMeshes, unqiue_nodes = nodeMeshesNx(G)
    edgeMeshes, unqiue_nodes = edgeMeshesNx(G)

    if not combine:
        return nodeMeshes, edgeMeshes

    else:
        return trimesh.util.concat(nodeMeshes+edgeMeshes)

