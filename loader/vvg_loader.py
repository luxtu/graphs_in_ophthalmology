import json
import pandas as pd
import numpy as np
import gzip

def vvg_to_df(vvg_path):
    """ Returns a pandas dataframe that contains all the information of the centerlines of the edges

    Paramters
    ---------
    vvg_path: The path to the vvg file (voreen vessel graph) that contains the centerline info


    Returns
    df: A dataframe containing information for every edge (e.g. centerline)
    -------

    """
    # Opening JSON file


    if vvg_path[-3:] == ".gz":
        with gzip.open(vvg_path, "rt") as gzipped_file:
            # Read the decompressed JSON data
            json_data = gzipped_file.read()
            data = json.loads(json_data)

    else:
        f = open(vvg_path)
        data = json.load(f)
        f.close()

    id_col = []
    pos_col = []
    node1_col = []
    node2_col = []
    minDistToSurface_col = []
    maxDistToSurface_col = []
    avgDistToSurface_col = []
    numSurfaceVoxels_col = []
    volume_col = []
    nearOtherEdge_col = []


    node_id_col = []
    node_voxel_col = []
    node_radius_col = []

    for i in data["graph"]["nodes"]:
        node_id_col.append(i["id"])
        node_voxel_col.append(i["voxels_"])
        node_radius_col.append(i["radius"])


    d_nodes = {'id': node_id_col,'voxel_pos' : node_voxel_col, "radius": node_radius_col}
    df_nodes = pd.DataFrame(d_nodes)
    df_nodes.set_index('id')

    for i in data["graph"]["edges"]:
        positions = []
        minDistToSurface = []
        maxDistToSurface = []
        avgDistToSurface = []
        numSurfaceVoxels = []
        volume = []
        nearOtherEdge = []


        try:
            i["skeletonVoxels"]
        except KeyError:
            #print("fail vessel")
            continue

        id_col.append(i["id"])
        node1_col.append(i["node1"])
        node2_col.append(i["node2"])


        for j in i["skeletonVoxels"]:
            positions.append(np.array(j["pos"]))
            minDistToSurface.append(j["minDistToSurface"])
            maxDistToSurface.append(j["maxDistToSurface"])
            avgDistToSurface.append(j["avgDistToSurface"])
            numSurfaceVoxels.append(j["numSurfaceVoxels"])
            volume.append(j["volume"])
            nearOtherEdge.append(j["nearOtherEdge"] )

        pos_col.append(positions)
        minDistToSurface_col.append(minDistToSurface)
        maxDistToSurface_col.append(maxDistToSurface)
        avgDistToSurface_col.append(avgDistToSurface)
        numSurfaceVoxels_col.append(numSurfaceVoxels)
        volume_col.append(volume)
        nearOtherEdge_col.append(nearOtherEdge)

    


    d = {'id': id_col,'pos' : pos_col, "node1" : node1_col, "node2" : node2_col, "minDistToSurface": minDistToSurface_col,"maxDistToSurface":maxDistToSurface_col, "avgDistToSurface":avgDistToSurface_col, "numSurfaceVoxels":numSurfaceVoxels_col, "volume":volume_col,"nearOtherEdge":nearOtherEdge_col }
    df_edge = pd.DataFrame(d)
    df_edge.set_index('id')
    return df_edge, df_nodes