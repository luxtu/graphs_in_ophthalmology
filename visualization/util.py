import numpy as np 
from scipy import sparse
import pandas as pd
import multiprocessing as mp
import trimesh



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



def findConnectedComponents(nodes, edges):
    """ A function to find all the connected components of a graph and the corresponding sets of nodes and vertices.
    nodes: A pandas dataframe with the node information
    edges: A pandas dataframe with the edge information

    return num: The mesh for the corresponding edge.
    return sets_nodes: A list of sets that contain the nodes specific to each connected component.
    return sets_edges: A list of sets that contain the edges specific to each connected component.
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
    


def edges_with_node_info(nodes, edges):
    """ Joins node and edge information into a single dataframe.
    nodes: A pandas dataframe with the node information
    edges: A pandas dataframe with the edge information.

    return res: A pandas datframe that contains the start and end coordinates for each edge.
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


def parallelConcat(meshList, workers = 4):
    """
    Parameters
    ----------
    meshList: A list of trimesh objects. 
    workers: Number of workers for parallelization

    Returns
    -------
    A concatenated trimesh
    """
    if len(meshList) > 40:
        pool = mp.Pool(workers)
        work = np.array_split(meshList, 4)
        meshListMap = pool.map(trimesh.util.concatenate,work)
        pool.close()
        pool.join()
        mesh = trimesh.util.concatenate(meshListMap)
    else:
        mesh = trimesh.util.concatenate(meshList)
    return mesh

