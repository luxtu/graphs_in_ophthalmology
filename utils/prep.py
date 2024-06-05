import numpy as np
import torch
from tqdm import tqdm

from utils import vvg_tools


def quartiles(regionmask, intensity):
    return np.percentile(intensity[regionmask], q=(5, 10, 25, 75, 90, 95))


def std_img(regionmask, intensity):
    return np.std(intensity[regionmask])


def create_combined_feature_dict(
    g_feature_dict, faz_feature_dict, seg_feature_dict, dataset
):
    comb_feature_dict = {}
    for key, val in g_feature_dict.items():
        comb_feature_dict[key] = (
            np.concatenate(
                [
                    val["graph_1"],
                    val["graph_2"],
                    np.array(faz_feature_dict[key]),
                    seg_feature_dict[key],
                ],
                axis=0,
            ),
            int(dataset.hetero_graphs[key].y[0]),
        )
    return comb_feature_dict


def add_global_node(dataset):
    for data in dataset:
        global_features = []
        for node_type in data.x_dict.keys():
            node_num = data.x_dict[node_type].shape[0]
            edge_num = data.edge_index_dict[(node_type, "to", node_type)].shape[1]
            avg_deg = 2 * edge_num / node_num
            global_features += [node_num, edge_num, avg_deg]

            edges = torch.zeros((2, node_num))
            edges[0, :] = 0
            edges[1, :] = torch.arange(node_num)
            edges = edges.long()
            data[("global", "to", node_type)].edge_index = edges

        heter_edge_num = data.edge_index_dict[("graph_1", "to", "graph_2")].shape[1]
        global_features += [heter_edge_num, data.eye]
        global_features = torch.tensor(global_features).unsqueeze(0)
        data["global"].x = global_features
        data[("global", "to", "global")].edge_index = torch.zeros((2, 1)).long()


def hetero_graph_cleanup(dataset, min_area_size=10):
    relevant_nodes = ["graph_1", "graph_2"]
    for data in tqdm(dataset):
        # get the indices of the nodes that have nan or inf values
        del_nodes_dict = {}
        new_idx_dict = {"graph_1": {}, "graph_2": {}}
        for key in relevant_nodes:  # data.x_dict.items():
            del_nodes = torch.where(
                torch.isnan(data.x_dict[key]) | torch.isinf(data.x_dict[key])
            )[0]

            if key == "graph_1":
                # if a node has feature values <0, then remove it
                idx = torch.where(data.x_dict[key] < 0)[0]
                del_nodes = torch.cat([del_nodes, idx], dim=0)
            elif key == "graph_2":
                # if the 3rd feature is <10, then remove it
                idx = torch.where(data.x_dict[key][:, 2] <= min_area_size)[0]
                del_nodes = torch.cat([del_nodes, idx], dim=0)

            # check if the node has a position in the region of the inlay
            # if the position is in the inlay (>1100 and <100), delete the nodes
            additioonal_del_nodes = torch.where(
                data.pos_dict[key][:, 0] > 1100 and data.pos_dict[key][:, 1] < 100
            )[0]

            del_nodes = torch.cat([del_nodes, additioonal_del_nodes], dim=0)

            # remove duplicates
            del_nodes = torch.unique(del_nodes)
            del_nodes_dict[key] = del_nodes

            # print the number of nodes that are removed
            # print(f"Number of nodes removed from {key}: {len(del_nodes)}")
            old_node_num = data.x_dict[key].shape[0]

            # remove the nodes from the x_dict, select all indices that are not in del_nodes
            keep_node_mask = ~torch.isin(torch.arange(old_node_num), del_nodes)
            data[key].x = data.x_dict[key][keep_node_mask, :]
            # remove the nodes from the pos_dict
            data[key].pos = data.pos_dict[key][keep_node_mask, :]

            # create a dict that maps the old node indices to the new node indices
            # the new node indices are shifted by the number of nodes that are removed wtih a lower index
            # e.g. if node 0 and 1 are removed, then the new node 0 is the old node 2
            # and the new node 1 is the old node 3
            # iterate over the old nodes
            # for i in range(old_node_num):
            #    # if the node is removed the new idx is None
            #    if i in del_nodes:
            #        new_idx_dict[key][i] = None
            #    else:
            #        # count the number of nodes that are removed with a lower index
            #        # and shift the new index by this number
            #        new_idx_dict[key][i] = torch.tensor(i - torch.sum(del_nodes < i))
            # no visible speed up
            new_idx_dict[key] = {
                i.item(): torch.tensor(new_i)
                for new_i, i in enumerate(torch.where(keep_node_mask)[0])
            }

        # remove the nodes from the edge_index
        # start by removing edges between same type nodes

        for key in relevant_nodes:
            # remove edges between same type nodes
            # get the indices of the edges that have to be removed
            del_mask = torch.any(
                torch.isin(data.edge_index_dict[(key, "to", key)], del_nodes_dict[key]),
                dim=0,
            )
            # print number of edges that are removed
            # print(f"Number of edges removed from {key} to {key}: {torch.sum(del_mask)}")
            # remove the edges
            data[key, "to", key].edge_index = data.edge_index_dict[(key, "to", key)][
                :, ~del_mask
            ]

            # update the indices of the edges
            # iterate over the edges
            for i in range(data.edge_index_dict[(key, "to", key)].shape[1]):
                # update the indices of the edges
                data.edge_index_dict[(key, "to", key)][0, i] = new_idx_dict[key][
                    data.edge_index_dict[(key, "to", key)][0, i].item()
                ]
                data.edge_index_dict[(key, "to", key)][1, i] = new_idx_dict[key][
                    data.edge_index_dict[(key, "to", key)][1, i].item()
                ]

        # remove edges between graph_1 and graph_2
        # get the indices of the edges that have to be removed

        del_mask_12_p1 = torch.isin(
            data.edge_index_dict[(relevant_nodes[0], "to", relevant_nodes[1])][0],
            del_nodes_dict[relevant_nodes[0]],
        )
        del_mask_12_p2 = torch.isin(
            data.edge_index_dict[(relevant_nodes[0], "to", relevant_nodes[1])][1],
            del_nodes_dict[relevant_nodes[1]],
        )

        del_mask_12 = del_mask_12_p1 | del_mask_12_p2
        # print number of edges that are removed
        # print(f"Number of edges removed from {relevant_nodes[0]} to {relevant_nodes[1]}: {torch.sum(del_mask_12)}")

        # remove the edges
        data[
            relevant_nodes[0], "to", relevant_nodes[1]
        ].edge_index = data.edge_index_dict[
            (relevant_nodes[0], "to", relevant_nodes[1])
        ][:, ~del_mask_12]
        # remove the same edges from the other direction
        data[
            relevant_nodes[1], "rev_to", relevant_nodes[0]
        ].edge_index = data.edge_index_dict[
            (relevant_nodes[1], "rev_to", relevant_nodes[0])
        ][:, ~del_mask_12]

        # update the indices of the edges
        # iterate over the edges
        for i in range(
            data.edge_index_dict[(relevant_nodes[0], "to", relevant_nodes[1])].shape[1]
        ):
            # update the indices of the edges
            data.edge_index_dict[(relevant_nodes[0], "to", relevant_nodes[1])][0, i] = (
                new_idx_dict[
                    relevant_nodes[0]
                ][
                    data.edge_index_dict[(relevant_nodes[0], "to", relevant_nodes[1])][
                        0, i
                    ].item()
                ]
            )
            data.edge_index_dict[(relevant_nodes[0], "to", relevant_nodes[1])][1, i] = (
                new_idx_dict[
                    relevant_nodes[1]
                ][
                    data.edge_index_dict[(relevant_nodes[0], "to", relevant_nodes[1])][
                        1, i
                    ].item()
                ]
            )
            data.edge_index_dict[(relevant_nodes[1], "rev_to", relevant_nodes[0])][
                0, i
            ] = new_idx_dict[relevant_nodes[1]][
                data.edge_index_dict[(relevant_nodes[1], "rev_to", relevant_nodes[0])][
                    0, i
                ].item()
            ]
            data.edge_index_dict[(relevant_nodes[1], "rev_to", relevant_nodes[0])][
                1, i
            ] = new_idx_dict[relevant_nodes[0]][
                data.edge_index_dict[(relevant_nodes[1], "rev_to", relevant_nodes[0])][
                    1, i
                ].item()
            ]

        # remove edges between graph_1 and faz
        # get the indices of the edges that have to be removed
        del_mask_1f = torch.isin(
            data.edge_index_dict[(relevant_nodes[0], "to", "faz")][0],
            del_nodes_dict[relevant_nodes[0]],
        )

        # print number of edges that are removed
        # print(f"Number of edges removed from {relevant_nodes[0]} to faz: {torch.sum(del_mask_1f)}")

        # remove the edges
        data[relevant_nodes[0], "to", "faz"].edge_index = data.edge_index_dict[
            (relevant_nodes[0], "to", "faz")
        ][:, ~del_mask_1f]
        # remove the same edges from the other direction
        data["faz", "rev_to", relevant_nodes[0]].edge_index = data.edge_index_dict[
            ("faz", "rev_to", relevant_nodes[0])
        ][:, ~del_mask_1f]

        # update the indices of the edges
        # iterate over the edges
        for i in range(data.edge_index_dict[(relevant_nodes[0], "to", "faz")].shape[1]):
            # update the indices of the edges, the index of the faz node is not changed
            data.edge_index_dict[(relevant_nodes[0], "to", "faz")][0, i] = new_idx_dict[
                relevant_nodes[0]
            ][data.edge_index_dict[(relevant_nodes[0], "to", "faz")][0, i].item()]
            data.edge_index_dict[("faz", "rev_to", relevant_nodes[0])][1, i] = (
                new_idx_dict[
                    relevant_nodes[0]
                ][
                    data.edge_index_dict[("faz", "rev_to", relevant_nodes[0])][
                        1, i
                    ].item()
                ]
            )

        # remove edges between graph_2 and faz
        # get the indices of the edges that have to be removed
        del_mask_2f = torch.isin(
            data.edge_index_dict[("faz", "to", relevant_nodes[1])][1],
            del_nodes_dict[relevant_nodes[1]],
        )

        # print number of edges that are removed
        # print(f"Number of edges removed from faz to {relevant_nodes[1]}: {torch.sum(del_mask_2f)}")

        # remove the edges
        data["faz", "to", relevant_nodes[1]].edge_index = data.edge_index_dict[
            ("faz", "to", relevant_nodes[1])
        ][:, ~del_mask_2f]
        # remove the same edges from the other direction
        data[relevant_nodes[1], "rev_to", "faz"].edge_index = data.edge_index_dict[
            (relevant_nodes[1], "rev_to", "faz")
        ][:, ~del_mask_2f]

        # update the indices of the edges
        # iterate over the edges
        for i in range(data.edge_index_dict[("faz", "to", relevant_nodes[1])].shape[1]):
            # update the indices of the edges, the index of the faz node is not changed
            data.edge_index_dict[("faz", "to", relevant_nodes[1])][1, i] = new_idx_dict[
                relevant_nodes[1]
            ][data.edge_index_dict[("faz", "to", relevant_nodes[1])][1, i].item()]
            data.edge_index_dict[(relevant_nodes[1], "rev_to", "faz")][0, i] = (
                new_idx_dict[
                    relevant_nodes[1]
                ][
                    data.edge_index_dict[(relevant_nodes[1], "rev_to", "faz")][
                        0, i
                    ].item()
                ]
            )

    return dataset


def hetero_graph_cleanup_multi(dataset):
    # replace multiprocessing with torch.multiprocessing
    import torch.multiprocessing as mp

    torch.multiprocessing.set_sharing_strategy("file_system")
    num_processes = 16  # Use all available CPU cores, except one
    with mp.Pool(num_processes) as pool:
        updated_dataset = list(
            tqdm(pool.imap(process_data_wrapper, dataset), total=len(dataset))
        )

    return updated_dataset


def process_data_wrapper(data):
    return heter_graph_cleanup_singel_data(data)


def heter_graph_cleanup_singel_data(data):
    relevant_nodes = ["graph_1", "graph_2"]
    # get the indices of the nodes that have nan or inf values
    del_nodes_dict = {}
    new_idx_dict = {"graph_1": {}, "graph_2": {}}
    for key in relevant_nodes:  # data.x_dict.items():
        del_nodes = torch.where(
            torch.isnan(data.x_dict[key]) | torch.isinf(data.x_dict[key])
        )[0]

        if key == "graph_1":
            # if a node has feature values <0, then remove it
            # idx = torch.where(data.x_dict[key] < 0)[0]
            pass
            # del_nodes = torch.cat([del_nodes, idx], dim=0)
        elif key == "graph_2":
            # if the 3rd feature is <10, then remove it
            idx = torch.where(data.x_dict[key][:, 2] <= 10)[0]
            del_nodes = torch.cat([del_nodes, idx], dim=0)

        # remove duplicates
        del_nodes = torch.unique(del_nodes)
        del_nodes_dict[key] = del_nodes

        # print the number of nodes that are removed
        # print(f"Number of nodes removed from {key}: {len(del_nodes)}")
        old_node_num = data.x_dict[key].shape[0]

        # remove the nodes from the x_dict, select all indices that are not in del_nodes
        keep_node_mask = ~torch.isin(torch.arange(old_node_num), del_nodes)
        data[key].x = data.x_dict[key][keep_node_mask, :]
        # remove the nodes from the pos_dict
        data[key].pos = data.pos_dict[key][keep_node_mask, :]

        # create a dict that maps the old node indices to the new node indices
        # the new node indices are shifted by the number of nodes that are removed wtih a lower index
        # e.g. if node 0 and 1 are removed, then the new node 0 is the old node 2
        # and the new node 1 is the old node 3
        # iterate over the old nodes
        # for i in range(old_node_num):
        #    # if the node is removed the new idx is None
        #    if i in del_nodes:
        #        new_idx_dict[key][i] = None
        #    else:
        #        # count the number of nodes that are removed with a lower index
        #        # and shift the new index by this number
        #        new_idx_dict[key][i] = torch.tensor(i - torch.sum(del_nodes < i))
        # no visible speed up
        new_idx_dict[key] = {
            i.item(): torch.tensor(new_i)
            for new_i, i in enumerate(torch.where(keep_node_mask)[0])
        }

    # remove the nodes from the edge_index
    # start by removing edges between same type nodes

    for key in relevant_nodes:
        # remove edges between same type nodes
        # get the indices of the edges that have to be removed
        del_mask = torch.any(
            torch.isin(data.edge_index_dict[(key, "to", key)], del_nodes_dict[key]),
            dim=0,
        )
        # print number of edges that are removed
        # print(f"Number of edges removed from {key} to {key}: {torch.sum(del_mask)}")
        # remove the edges
        data[key, "to", key].edge_index = data.edge_index_dict[(key, "to", key)][
            :, ~del_mask
        ]

        # update the indices of the edges
        # iterate over the edges
        for i in range(data.edge_index_dict[(key, "to", key)].shape[1]):
            # update the indices of the edges
            data.edge_index_dict[(key, "to", key)][0, i] = new_idx_dict[key][
                data.edge_index_dict[(key, "to", key)][0, i].item()
            ]
            data.edge_index_dict[(key, "to", key)][1, i] = new_idx_dict[key][
                data.edge_index_dict[(key, "to", key)][1, i].item()
            ]

    # remove edges between graph_1 and graph_2
    # get the indices of the edges that have to be removed

    del_mask_12_p1 = torch.isin(
        data.edge_index_dict[(relevant_nodes[0], "to", relevant_nodes[1])][0],
        del_nodes_dict[relevant_nodes[0]],
    )
    del_mask_12_p2 = torch.isin(
        data.edge_index_dict[(relevant_nodes[0], "to", relevant_nodes[1])][1],
        del_nodes_dict[relevant_nodes[1]],
    )

    del_mask_12 = del_mask_12_p1 | del_mask_12_p2
    # print number of edges that are removed
    # print(f"Number of edges removed from {relevant_nodes[0]} to {relevant_nodes[1]}: {torch.sum(del_mask_12)}")

    # remove the edges
    data[relevant_nodes[0], "to", relevant_nodes[1]].edge_index = data.edge_index_dict[
        (relevant_nodes[0], "to", relevant_nodes[1])
    ][:, ~del_mask_12]
    # remove the same edges from the other direction
    data[
        relevant_nodes[1], "rev_to", relevant_nodes[0]
    ].edge_index = data.edge_index_dict[
        (relevant_nodes[1], "rev_to", relevant_nodes[0])
    ][:, ~del_mask_12]

    # update the indices of the edges
    # iterate over the edges
    for i in range(
        data.edge_index_dict[(relevant_nodes[0], "to", relevant_nodes[1])].shape[1]
    ):
        # update the indices of the edges
        data.edge_index_dict[(relevant_nodes[0], "to", relevant_nodes[1])][0, i] = (
            new_idx_dict[
                relevant_nodes[0]
            ][
                data.edge_index_dict[(relevant_nodes[0], "to", relevant_nodes[1])][
                    0, i
                ].item()
            ]
        )
        data.edge_index_dict[(relevant_nodes[0], "to", relevant_nodes[1])][1, i] = (
            new_idx_dict[
                relevant_nodes[1]
            ][
                data.edge_index_dict[(relevant_nodes[0], "to", relevant_nodes[1])][
                    1, i
                ].item()
            ]
        )
        data.edge_index_dict[(relevant_nodes[1], "rev_to", relevant_nodes[0])][0, i] = (
            new_idx_dict[
                relevant_nodes[1]
            ][
                data.edge_index_dict[(relevant_nodes[1], "rev_to", relevant_nodes[0])][
                    0, i
                ].item()
            ]
        )
        data.edge_index_dict[(relevant_nodes[1], "rev_to", relevant_nodes[0])][1, i] = (
            new_idx_dict[
                relevant_nodes[0]
            ][
                data.edge_index_dict[(relevant_nodes[1], "rev_to", relevant_nodes[0])][
                    1, i
                ].item()
            ]
        )

    # remove edges between graph_1 and faz
    # get the indices of the edges that have to be removed
    del_mask_1f = torch.isin(
        data.edge_index_dict[(relevant_nodes[0], "to", "faz")][0],
        del_nodes_dict[relevant_nodes[0]],
    )

    # print number of edges that are removed
    # print(f"Number of edges removed from {relevant_nodes[0]} to faz: {torch.sum(del_mask_1f)}")

    # remove the edges
    data[relevant_nodes[0], "to", "faz"].edge_index = data.edge_index_dict[
        (relevant_nodes[0], "to", "faz")
    ][:, ~del_mask_1f]
    # remove the same edges from the other direction
    data["faz", "rev_to", relevant_nodes[0]].edge_index = data.edge_index_dict[
        ("faz", "rev_to", relevant_nodes[0])
    ][:, ~del_mask_1f]

    # update the indices of the edges
    # iterate over the edges
    for i in range(data.edge_index_dict[(relevant_nodes[0], "to", "faz")].shape[1]):
        # update the indices of the edges, the index of the faz node is not changed
        data.edge_index_dict[(relevant_nodes[0], "to", "faz")][0, i] = new_idx_dict[
            relevant_nodes[0]
        ][data.edge_index_dict[(relevant_nodes[0], "to", "faz")][0, i].item()]
        data.edge_index_dict[("faz", "rev_to", relevant_nodes[0])][1, i] = new_idx_dict[
            relevant_nodes[0]
        ][data.edge_index_dict[("faz", "rev_to", relevant_nodes[0])][1, i].item()]

    # remove edges between graph_2 and faz
    # get the indices of the edges that have to be removed
    del_mask_2f = torch.isin(
        data.edge_index_dict[("faz", "to", relevant_nodes[1])][1],
        del_nodes_dict[relevant_nodes[1]],
    )

    # print number of edges that are removed
    # print(f"Number of edges removed from faz to {relevant_nodes[1]}: {torch.sum(del_mask_2f)}")

    # remove the edges
    data["faz", "to", relevant_nodes[1]].edge_index = data.edge_index_dict[
        ("faz", "to", relevant_nodes[1])
    ][:, ~del_mask_2f]
    # remove the same edges from the other direction
    data[relevant_nodes[1], "rev_to", "faz"].edge_index = data.edge_index_dict[
        (relevant_nodes[1], "rev_to", "faz")
    ][:, ~del_mask_2f]

    # update the indices of the edges
    # iterate over the edges
    for i in range(data.edge_index_dict[("faz", "to", relevant_nodes[1])].shape[1]):
        # update the indices of the edges, the index of the faz node is not changed
        data.edge_index_dict[("faz", "to", relevant_nodes[1])][1, i] = new_idx_dict[
            relevant_nodes[1]
        ][data.edge_index_dict[("faz", "to", relevant_nodes[1])][1, i].item()]
        data.edge_index_dict[(relevant_nodes[1], "rev_to", "faz")][0, i] = new_idx_dict[
            relevant_nodes[1]
        ][data.edge_index_dict[(relevant_nodes[1], "rev_to", "faz")][0, i].item()]

    return data


def add_centerline_statistics(dataset, image_folder, vvg_folder, seg_size):
    import os

    from PIL import Image

    # read all the json/json.gz files in the vvg folder
    vvg_files = os.listdir(vvg_folder)
    vvg_files = [
        file
        for file in vvg_files
        if file.endswith(".json") or file.endswith(".json.gz")
    ]

    # read all the images in the image folder
    image_files = os.listdir(image_folder)
    image_files = [file for file in image_files if file.endswith(".png")]

    # match

    # iterate over the dataset
    for data in dataset:
        # get the id of the graph
        graph_id = data.graph_id
        # find the corresponding json file
        json_file = [file for file in vvg_files if graph_id in file][0]
        # find the corresponding image file
        image_file = [file for file in image_files if graph_id in file][0]
        # load the json file into a df
        df = vvg_to_df(os.path.join(vvg_folder, json_file))
        # load the image
        image = Image.open(os.path.join(image_folder, image_file))
        # turn the iamge into a numpy array
        image = np.array(image)
        # get the size of the image
        image_size = image.shape[0]
        # get the ratio between the image size and the seg size
        ratio = image_size / seg_size
        avg_intensities = cl_pos_to_intensities(df, image, ratio)

        print(avg_intensities.shape)
        print(data["graph_1"].x.shape)

        # add the avg intensities to the data
        try:
            data["graph_1"].x = torch.cat(
                [data["graph_1"].x, torch.tensor(avg_intensities)], dim=1
            )
        except RuntimeError:
            print("RuntimeError")


def vvg_to_df(vvg_path):
    # Opening JSON file
    import gzip
    import json

    import pandas as pd

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

    for i in data["graph"]["edges"]:
        positions = []
        id_col.append(i["id"])
        node1_col.append(i["node1"])
        node2_col.append(i["node2"])

        try:
            i["skeletonVoxels"]
        except KeyError:
            pos_col.append(None)
            # print("skeletonVoxels KeyError")
            continue
        for j in i["skeletonVoxels"]:
            positions.append(np.array(j["pos"]))
        pos_col.append(positions)

    d = {
        "id_col": id_col,
        "pos_col": pos_col,
        "node1_col": node1_col,
        "node2_col": node2_col,
    }
    df = pd.DataFrame(d)
    df.set_index("id_col")
    return df


def cl_pos_to_intensities(cl_pos_df, image, ratio):
    import numpy as np

    # create a df that contains the intensities of the centerline
    # iterate over the rows of the df
    avg_int = np.zeros((len(cl_pos_df), 4))
    for i in range(len(cl_pos_df)):
        # get the positions of the centerline
        positions = cl_pos_df.iloc[i]["pos_col"]
        # get the positions in the image
        try:
            positions = np.array(
                [
                    np.array(
                        [int(pos[0] * ratio), int(pos[1] * ratio), int(pos[2] * ratio)]
                    )
                    for pos in positions
                ]
            )
        except TypeError:
            avg_int[i] = None
            continue

        # get the intensities of the centerline
        try:
            intensities = image[positions[:, 0], positions[:, 1], positions[:, 2]]
        except IndexError:
            intensities = image[positions[:, 0], positions[:, 1]]

        # calculate the average intensity
        avg_int[i, 0] = np.mean(intensities)
        avg_int[i, 1] = np.std(intensities)
        # quantiles
        avg_int[i, 2] = np.quantile(intensities, 0.25)
        avg_int[i, 3] = np.quantile(intensities, 0.75)

    return avg_int


def add_centerline_statistics_multi(dataset, image_folder, vvg_folder, seg_size):
    import os

    # replace multiprocessing with torch.multiprocessing
    import torch.multiprocessing as mp

    torch.multiprocessing.set_sharing_strategy("file_system")

    vvg_files = os.listdir(vvg_folder)
    vvg_files = [
        file
        for file in vvg_files
        if file.endswith(".json") or file.endswith(".json.gz")
    ]

    image_files = os.listdir(image_folder)
    image_files = [file for file in image_files if file.endswith(".png")]

    # match the graphs with corresponding json and image files
    # iterate over the dataset

    matches = []

    for data in dataset:
        # get the id of the graph
        graph_id = data.graph_id
        # find the corresponding json file
        json_file = [file for file in vvg_files if graph_id in file][0]
        # find the corresponding image file
        image_file = [file for file in image_files if graph_id in file][0]

        # add folder to the file names
        json_file = os.path.join(vvg_folder, json_file)
        image_file = os.path.join(image_folder, image_file)

        matches.append((data, json_file, image_file, seg_size))

    with mp.Pool(16) as pool:
        updated_dataset = list(
            tqdm(pool.imap(process_data_cl, matches), total=len(dataset))
        )

    return updated_dataset


def add_vessel_region_statistics_multi(dataset, image_folder, vvg_folder, seg_folder):
    import os

    # replace multiprocessing with torch.multiprocessing
    import torch.multiprocessing as mp

    torch.multiprocessing.set_sharing_strategy("file_system")

    vvg_files = os.listdir(vvg_folder)
    vvg_files = [
        file
        for file in vvg_files
        if file.endswith(".json") or file.endswith(".json.gz")
    ]

    image_files = os.listdir(image_folder)
    image_files = [file for file in image_files if file.endswith(".png")]

    seg_files = os.listdir(seg_folder)
    seg_files = [file for file in seg_files if file.endswith(".png")]

    # match the graphs with corresponding json and image files
    # iterate over the dataset

    matches = []

    for data in dataset:
        # get the id of the graph
        graph_id = data.graph_id
        # find the corresponding json file
        json_file = [file for file in vvg_files if graph_id in file][0]
        # find the corresponding image file
        image_file = [file for file in image_files if graph_id in file][0]
        # find the corresponding seg file
        seg_file = [file for file in seg_files if graph_id in file][0]

        # add folder to the file names
        json_file = os.path.join(vvg_folder, json_file)
        image_file = os.path.join(image_folder, image_file)
        seg_file = os.path.join(seg_folder, seg_file)

        matches.append((data, json_file, image_file, seg_file))

    with mp.Pool(16) as pool:
        updated_dataset = list(
            tqdm(pool.imap(process_data_cl_region_props, matches), total=len(dataset))
        )

    return updated_dataset


def process_data_cl_region_props(matched_list):
    import numpy as np
    import pandas as pd
    from PIL import Image
    from skimage import measure, morphology, transform

    from loader import vvg_loader

    data, json_file, image_file, seg_file = matched_list
    vvg_df_edges, vvg_df_nodes = vvg_loader.vvg_to_df(json_file)
    # load the image
    image = Image.open(image_file)
    # turn the iamge into a numpy array
    try:
        image = np.array(image)[:, :, 0]
    except IndexError:
        image = np.array(image)
    # load the seg
    seg = Image.open(seg_file)
    # turn the seg into a numpy array
    seg = np.array(seg)
    # make seg binary
    seg = seg > 0
    # get the ratio between the image size and the seg size
    image = transform.resize(image, seg.shape, order=0, preserve_range=True)

    final_seg_label = np.zeros_like(seg, dtype=np.uint16)
    final_seg_label[seg != 0] = 1
    cl_vessel = vvg_tools.vvg_df_to_centerline_array_unique_label(
        vvg_df_edges, vvg_df_nodes, (1216, 1216), vessel_only=True
    )
    label_cl = cl_vessel  # measure.label(cl_vessel)
    label_cl[label_cl != 0] = label_cl[label_cl != 0] + 1
    final_seg_label[label_cl != 0] = label_cl[label_cl != 0]

    last_sum = 0
    new_sum = None
    ct = 0
    while last_sum != new_sum:
        ct += 1
        last_sum = new_sum

        label_cl = morphology.dilation(label_cl, morphology.square(3))
        label_cl = label_cl * seg
        # get the values of final_seg_label where no semantic segmentation is present
        final_seg_label[final_seg_label == 1] = label_cl[final_seg_label == 1]
        # get indices where label_cl==0 and seg !=0
        mask = (final_seg_label == 0) & (seg != 0)
        new_sum = np.sum(mask)
        final_seg_label[mask] = 1

    # pixels that are still 1 are turned into 0
    final_seg_label[final_seg_label == 1] = 0
    # labels for the rest are corrected by -1
    final_seg_label[final_seg_label != 0] = final_seg_label[final_seg_label != 0] - 1

    # extract the intensity statistics for the different regions
    props = measure.regionprops_table(
        final_seg_label,
        intensity_image=image,
        properties=(
            "label",
            "centroid",
            "area",
            "perimeter",
            "eccentricity",
            "equivalent_diameter",
            "orientation",
            "solidity",
            "feret_diameter_max",
            "extent",
            "axis_major_length",
            "axis_minor_length",
            "intensity_max",
            "intensity_mean",
            "intensity_min",
            "centroid_weighted",
        ),
        extra_properties=(std_img, quartiles),
    )  # remove quartiles, shouldnt be too many properties , "intensity_max", "intensity_min"
    props_df = pd.DataFrame(props)
    # match the regions with the nodes
    # iterate over the edges
    # create array with as many nans as props_df.columns
    # number of colums
    df_col_num = len(props_df.columns)
    nan_props = np.full((df_col_num), np.nan)

    # print the number of rows in the df
    # print(len(props_df))
    # print the max labels in final_seg_label
    # print(np.max(final_seg_label))
    # this should be the same as the number of rows in the df, the number of vessels can differ
    # print(len(vvg_df_edges))

    added_props = []
    for i in range(len(vvg_df_edges)):
        # get the positions of the centerline
        positions = vvg_df_edges.iloc[i]["pos"]
        # check if positions is an empty list
        if len(positions) == 0:
            # as many nans as props_df.columns
            added_props.append(
                nan_props[1:]
            )  # two less np.nan without quartiles # , np.nan, np.nan
            continue
        # get the most frequent label in the region
        frequent_label = {}
        for pos in positions:
            label = final_seg_label[int(pos[0]), int(pos[1])]
            if label in frequent_label:
                frequent_label[label] += 1
            else:
                frequent_label[label] = 1
        # get the most frequent label, by the number of occurences
        max_label = max(frequent_label, key=frequent_label.get)

        # if the max label is 0 take the second most frequent label
        if max_label == 0:
            try:
                frequent_label.pop(max_label)
                max_label = max(frequent_label, key=frequent_label.get)
            except ValueError:
                max_label = 0

        # get the properties of the region
        if max_label != 0:
            # get the column where the label is the max label
            # get the row where 'label' is the max label
            prop = props_df[props_df["label"] == max_label]
            # don't take the label column
            if len(np.array(prop)[0]) != df_col_num:
                print("number of properties is not correct")
                print(len(np.array(prop)[0]))
                print(df_col_num)
                print(np.array(prop)[0])
            added_props.append(
                np.array(prop)[0][1:]
            )  # not need : [1:] if the label is not included
        else:
            # give the properties nan values
            added_props.append(
                nan_props[1:]
            )  # two less np.nan without quartiles # , np.nan, np.nan
            # this should not happen
            print("max label is 0")

    data["graph_1"].x = torch.cat(
        [
            data["graph_1"].x,
            torch.tensor(np.array(added_props), dtype=data["graph_1"].x.dtype),
        ],
        dim=1,
    )

    return data


def process_data_cl(matched_list):
    # unpack the tuple
    from PIL import Image

    data, json_file, image_file, seg_size = matched_list
    df = vvg_to_df(json_file)
    # load the image
    image = Image.open(image_file)
    # turn the iamge into a numpy array
    image = np.array(image)
    # get the size of the image
    image_size = image.shape[0]
    # get the ratio between the image size and the seg size
    ratio = image_size / seg_size
    avg_intensities = cl_pos_to_intensities(df, image, ratio)

    # add the avg intensities to the data
    try:
        data["graph_1"].x = torch.cat(
            [
                data["graph_1"].x,
                torch.tensor(avg_intensities, dtype=data["graph_1"].x.dtype),
            ],
            dim=1,
        )
    except RuntimeError:
        print("RuntimeError")

    return data


def check_centerline_on_image(
    dataset, image_folder, vvg_folder, seg_size, save_image_path
):
    import os

    from PIL import Image

    vvg_files = os.listdir(vvg_folder)
    vvg_files = [
        file
        for file in vvg_files
        if file.endswith(".json") or file.endswith(".json.gz")
    ]

    image_files = os.listdir(image_folder)
    image_files = [file for file in image_files if file.endswith(".png")]

    # match the graphs with corresponding json and image files
    # iterate over the dataset

    matches = []

    for data in dataset:
        # get the id of the graph
        graph_id = data.graph_id
        # find the corresponding json file
        json_file = [file for file in vvg_files if graph_id in file][0]
        # find the corresponding image file
        image_file = [file for file in image_files if graph_id in file][0]

        # add folder to the file names
        json_file = os.path.join(vvg_folder, json_file)
        image_file = os.path.join(image_folder, image_file)

        matches.append((data, json_file, image_file, seg_size))

    data, json_file, image_file, seg_size = matches[0]
    df = vvg_to_df(json_file)
    # load the image
    image = Image.open(image_file)
    # turn the iamge into a numpy array
    image = np.array(image)
    # get the size of the image
    image_size = image.shape[0]
    # get the ratio between the image size and the seg size
    ratio = image_size / seg_size

    for i in range(len(df)):
        # get the positions of the centerline
        positions = df.iloc[i]["pos_col"]
        # get the positions in the image
        try:
            positions = np.array(
                [
                    np.array(
                        [int(pos[0] * ratio), int(pos[1] * ratio), int(pos[2] * ratio)]
                    )
                    for pos in positions
                ]
            )
        except TypeError:
            continue

        # draw the centerline on the image
        try:
            image[positions[:, 0], positions[:, 1], positions[:, 2]] = 255
        except IndexError:
            image[positions[:, 0], positions[:, 1]] = 0

    # save the image
    image = Image.fromarray(image)
    image.save(save_image_path)
