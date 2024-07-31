# trying to create overlays of the explanations on the raw data


import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from skimage import measure, morphology, transform

from explainability import utils_homogeneous as utils
from loader import vvg_loader
from utils import vvg_tools


class RawDataExplainer:
    def __init__(self, raw_image_path, segmentation_path, vvg_path, graph_type):
        self.raw_image_path = raw_image_path
        self.segmentation_path = segmentation_path
        self.vvg_path = vvg_path

        self.raw_images = os.listdir(self.raw_image_path)
        self.segmentations = os.listdir(self.segmentation_path)
        self.vvgs = os.listdir(self.vvg_path)

        self.graph_type = graph_type
        # assert that the graph type is either "vessel" or "region"
        assert (
            self.graph_type == "vessel" or self.graph_type == "region"
        ), "The graph type should be either 'vessel' or 'region'"

        # extract all files from vvg files that end with .json or .json.gz
        self.vvgs = [
            file
            for file in self.vvgs
            if file.endswith(".json") or file.endswith(".json.gz")
        ]

        # extract all files from seg files that end with .png
        self.segmentations = [
            file for file in self.segmentations if file.endswith(".png")
        ]

        # extract all files from raw image files that end with .png
        self.raw_images = [file for file in self.raw_images if file.endswith(".png")]


    def _process_files(self, graph_id):
        seg_file = [file for file in self.segmentations if graph_id in file]
        assert (
            len(seg_file) == 1
        ), "There should be only one segmentation file for this graph_id"
        seg_file = seg_file[0]

        raw_file = [file for file in self.raw_images if graph_id in file]
        assert (
            len(raw_file) == 1
        ), "There should be only one raw image file for this graph_id"
        raw_file = raw_file[0]

        vvg_file = [file for file in self.vvgs if graph_id in file]
        assert len(vvg_file) == 1, "There should be only one vvg file for this graph_id"
        vvg_file = vvg_file[0]

        return seg_file, raw_file, vvg_file

    def _identify_faz_region_label(self, region_labels):
        # faz_region is the region with label at 600,600
        faz_region_label = region_labels[600, 600]
        idx_y = 600
        while faz_region_label == 0:
            idx_y += 1
            faz_region_label = region_labels[idx_y, 600]
        return faz_region_label


    def _heatmap_relevant_vessels(
        self,
        raw,
        relevant_vessels_pre,
        vvg_df_edges,
        explanations,
        graph,
        vvg_df_nodes,
        matched_list,
        only_positive=False,
        intensity_value=None,
    ):
        # for the vessels create a mask that containts the centerline of the relevant vessels
        # also store the positions of the center points on the relevant vessels

        #  match the vessels in the graph to the vessels in the vvg
        cl_arr = np.zeros_like(raw, dtype=np.float32)
        vessel_alphas = np.zeros_like(raw, dtype=np.float32)
        # extract the node positions from the graph
        node_positions = graph.pos.cpu().detach().numpy()
        # get the importance of the vessels
        if only_positive:
            importance = (
                explanations.node_mask
                .sum(dim=-1)
                .cpu()
                .detach()
                .numpy()
            )
        else:
            importance = (
                explanations.node_mask
                .abs()
                .sum(dim=-1)
                .cpu()
                .detach()
                .numpy()
            )
        # get the positions of the relevant vessels
        relevant_vessels = np.where(relevant_vessels_pre)[0]
        # get the center points of the relevant vessels
        center_points = node_positions[relevant_vessels]
        # get the importance of the relevant vessels
        relevant_importance = importance[relevant_vessels]
        # get the positions of the nodes in vvg_df_nodes
        bifurcation_nodes = vvg_df_nodes["pos"].values
        # calculate the center point for all vessels in the vvg
        center_points_vvg = []
        for i, node1 in enumerate(vvg_df_edges["node1"]):
            node2 = vvg_df_edges["node2"].iloc[i]
            # get the positions of the nodes
            pos1 = bifurcation_nodes[node1]
            pos2 = bifurcation_nodes[node2]
            # calculate the center point
            center_point = (np.array(pos1) + np.array(pos2)) / 2
            center_points_vvg.append(center_point)
        center_points_vvg = np.array(center_points_vvg)[:, :2]

        # create a regions for every vessel in the vvgs
        image, seg = matched_list
        # make seg binary
        seg = seg > 0

        final_seg_label = np.zeros_like(seg, dtype=np.uint16)
        final_seg_label[seg != 0] = 1

        cl_vessel = vvg_tools.vvg_df_to_centerline_array_unique_label(
            vvg_df_edges, vvg_df_nodes, (1216, 1216), vessel_only=True
        )
        label_cl = cl_vessel  # measure.label(cl_vessel)
        label_cl[label_cl != 0] = label_cl[label_cl != 0] + 1
        final_seg_label[label_cl != 0] = label_cl[label_cl != 0]

        for i in range(100):
            label_cl = morphology.dilation(label_cl, morphology.square(3))
            label_cl = label_cl * seg
            # get the values of final_seg_label where no semantic segmentation is present
            final_seg_label[final_seg_label == 1] = label_cl[final_seg_label == 1]
            # get indices where label_cl==0 and seg !=0
            mask = (final_seg_label == 0) & (seg != 0)
            final_seg_label[mask] = 1

        # pixels that are still 1 are turned into 0
        final_seg_label[final_seg_label == 1] = 0
        # labels for the rest are corrected by -1
        final_seg_label[final_seg_label != 0] = (
            final_seg_label[final_seg_label != 0] - 1
        )
        cl_center_points = []
        for i, cp in enumerate(center_points):
            # get the closest center point in the vvg
            dist = np.linalg.norm(center_points_vvg - cp, axis=1)
            #
            closest_vessel = np.argmin(dist)
            # if there are multiple vessels with the same distance, then create a list of the closest vessels
            closest_vessels = np.where(dist == dist[closest_vessel])[0]


            # get the closest vessel in the vessel_vvg
            positions = vvg_df_edges["pos"].iloc[closest_vessel]
            frequent_label = {}

            # take the longer vessels, others are artifacts
            if len(closest_vessels) > 1:
                for vessel in closest_vessels:
                    positions = vvg_df_edges["pos"].iloc[vessel]
                    if len(positions) > len(vvg_df_edges["pos"].iloc[closest_vessel]):
                        closest_vessel = vessel

            #for closest_vessel in closest_vessels:
            try:
                cl_center_points.append(positions[int(len(positions) / 2)])
            except IndexError:
                # append some point at 0,0 if the vessel is empty
                cl_center_points.append([0, 0, 0])
                continue
            for pos in positions:
                label = final_seg_label[int(pos[0]), int(pos[1])]
                if label in frequent_label:
                    frequent_label[label] += 1
                else:
                    frequent_label[label] = 1
            # get the most frequent label, by the number of occurences
            max_label = max(frequent_label, key=frequent_label.get)
            # set the cl_arr_value to the importance of the vessel
            if intensity_value is not None:
                cl_arr[final_seg_label == max_label] = intensity_value
            else:
                cl_arr[final_seg_label == max_label] = relevant_importance[i]
            vessel_alphas[final_seg_label == max_label] = 0.7

        cl_center_points = np.array(cl_center_points)

        return cl_arr, cl_center_points, vessel_alphas



    def _color_relevant_vessels(self, raw, relevant_vessels_pre, vvg_df_edges):
        # for the vessels create a mask that containts the centerline of the relevant vessels
        # also store the positions of the center points on the relevant vessels
        relevant_vessels = np.where(relevant_vessels_pre)[0]
        cl_arr = np.zeros_like(raw, dtype=np.float32)
        cl_center_points = []
        for i, cl in enumerate(vvg_df_edges["pos"]):
            # only add the centerline of the relevant vessels
            if i in relevant_vessels:
                # extract the centerpoints of the relevant vessels
                cl_center_points.append(cl[int(len(cl) / 2)])
                for pos in cl:
                    cl_arr[int(pos[0]), int(pos[1])] = 1
                    # also color the neighboring pixels if they exist
                    # create indices for the neighboring pixels
                    neigh_pix = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
                    # include the 2nd order neighbors
                    neigh_pix = np.concatenate(
                        (neigh_pix, np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]]))
                    )
                    # include the 3rd order neighbors
                    neigh_pix = np.concatenate(
                        (neigh_pix, np.array([[-2, 0], [2, 0], [0, -2], [0, 2]]))
                    )
                    # convert pos to int
                    pos = np.array([int(pos[0]), int(pos[1])]).astype(int)
                    for neigb_pos in neigh_pix:
                        neigb_pos += pos
                    neigh_pix = neigh_pix.astype(int)
                    # check if the neighboring pixels are in the image
                    neigh_pix = neigh_pix[
                        (neigh_pix[:, 0] >= 0)
                        & (neigh_pix[:, 0] < cl_arr.shape[0])
                        & (neigh_pix[:, 1] >= 0)
                        & (neigh_pix[:, 1] < cl_arr.shape[1])
                    ]
                    # set the neighboring pixels to 1
                    cl_arr[neigh_pix[:, 0], neigh_pix[:, 1]] = 1

        cl_center_points = np.array(cl_center_points)
        # assue that 2 dimensions are always returned
        if len(cl_center_points.shape) == 1:
            cl_center_points = np.expand_dims(cl_center_points, axis=0)

        return cl_arr, cl_center_points

    def _color_relevant_regions(
        self, raw, seg, region_labels, df, pos
    ):
        relevant_region_labels = []
        for position in pos:
            lab_1 = df.loc[np.isclose(df["centroid-0"], position[0])]["label"].values
            lab_2 = df.loc[np.isclose(df["centroid-1"], position[1])]["label"].values

            for val_1 in lab_1:
                for val_2 in lab_2:
                    if val_1 == val_2:
                        relevant_region_labels.append(val_1)

        # extract the relevant regions
        regions = np.zeros_like(raw, dtype="uint8")
        alphas = np.zeros_like(raw, dtype=np.float32)
        alphas += 0.2
        # this highlights all vessels
        alphas[seg != 0] = 1

        # set the alpha of the relevant regions to 1
        dyn_region_label = 1
        for label in relevant_region_labels:
            alphas[region_labels == label] = 0.85
            regions[region_labels == label] = dyn_region_label
            dyn_region_label += 1


        return regions, alphas

    def _heatmap_relevant_regions(
        self,
        raw,
        seg,
        region_labels,
        df,
        pos,
        explanations,
        only_positive=False,
        intensity_value=None,
    ):
        relevant_region_labels = []
        relevant_region_indices = []
        for i, position in enumerate(pos):
            lab_1 = df.loc[np.isclose(df["centroid-0"], position[0])]["label"].values
            lab_2 = df.loc[np.isclose(df["centroid-1"], position[1])]["label"].values

            for val_1 in lab_1:
                for val_2 in lab_2:
                    if val_1 == val_2:
                        relevant_region_labels.append(val_1)
                        relevant_region_indices.append(i)
                        # can be break here, since the labels are unique
                        break

        # extract the relevant regions
        regions = np.zeros_like(raw, dtype=np.float32)
        alphas = np.zeros_like(raw, dtype=np.float32)

        # get the importance of the relevant regions
        if only_positive:
            importance_array = (
                explanations.node_mask
                .sum(dim=-1)
                .cpu()
                .detach()
                .numpy()
            )
        else:
            importance_array = (
                explanations.node_mask
                .abs()
                .sum(dim=-1)
                .cpu()
                .detach()
                .numpy()
            )
        for i, label in enumerate(relevant_region_labels):
            importance = importance_array[relevant_region_indices[i]].item()
            # alphas[region_labels == label] = importance
            if intensity_value is not None:
                regions[region_labels == label] = intensity_value
            else:
                regions[region_labels == label] = importance

        alphas[alphas == 0] = 0
        alphas[seg != 0] = 0
        alphas[regions != 0] = 0.5

        return regions, alphas

    def create_explanation_image(
        self,
        explanation,
        graph,
        graph_id,
        path,
        label_names=None,
        target=None,
        heatmap=False,
        explained_gradient=0.95,
        only_positive=False,
        points=False,
        intensity_value=None,
    ):
        # extract the relevant segmentation, raw image and vvg
        # search the file strings for the graph_id
        seg_file, raw_file, vvg_file = self._process_files(graph_id)

        # load the segmentation, raw image and vvg
        seg = Image.open(os.path.join(self.segmentation_path, seg_file))
        seg = np.array(seg)
        seg = seg.astype(np.uint8)

        raw = Image.open(os.path.join(self.raw_image_path, raw_file))
        raw = np.array(raw)[:, :, 0]
        raw = transform.resize(raw, seg.shape, order=0, preserve_range=True)

        # relevant nodes dict, getting the nodes above a certain threshold, includes all types of nodes
        if isinstance(explained_gradient, float):
            graph_rel_pos_idcs = utils.identifiy_relevant_nodes(
                explanation,
                graph,
                explained_gradient=explained_gradient,
                only_positive=only_positive,
            )
        elif isinstance(explained_gradient, int):
            graph_rel_pos_idcs = utils.top_k_important_nodes(
                explanation,
                graph,
                top_k=explained_gradient,
                only_positive=only_positive,
            )
        elif explained_gradient is None:
            graph_rel_pos_idcs = np.ones(
                graph.x.shape[0], dtype=bool
            )
        else:
            raise ValueError(
                "explained_gradient must be either a float, an int or None"
            )
        
        if self.graph_type == "vessel":

            # extract the relevant vessels
            vvg_df_edges, vvg_df_nodes = vvg_loader.vvg_to_df(
                os.path.join(self.vvg_path, vvg_file)
            )


            # extract the relevant vessels, with a segmentation mask and the center points of the relevant vessels
            if heatmap:
                cl_arr, cl_center_points, vessel_alphas = self._heatmap_relevant_vessels(
                    raw,
                    graph_rel_pos_idcs,
                    vvg_df_edges,
                    explanation,
                    graph,
                    vvg_df_nodes,
                    [raw, seg],
                    only_positive=only_positive,
                    intensity_value=intensity_value,
                )
            else:
                cl_arr, cl_center_points = self._color_relevant_vessels(
                    raw, graph_rel_pos_idcs, vvg_df_edges
                )

            # create the intensity image
            # remove .png from the path
            path = path[:-4]
            intensity_path = path + "_intensity.png"

            self.create_vessel_intensity_image(
                raw,
                cl_arr,
                cl_center_points,
                intensity_path,
                label_names,
                target,
                points,
                graph,
                vessel_alphas,
                intensity_value=intensity_value,
            )

        elif self.graph_type == "region":


            # extract the relevant regions
            region_labels = measure.label(
                morphology.remove_small_holes(seg, area_threshold=5, connectivity=1).astype(
                    "uint8"
                ),
                background=1,
            )
            props = measure.regionprops_table(
                region_labels,
                properties=(
                    "label",
                    "centroid",
                    "area",
                ),
            )
            df = pd.DataFrame(props)


            # extract relevant positions
            relevant_regions = np.where(graph_rel_pos_idcs)[0]
            # pos are the positions of relevant regions
            pos = graph.pos.cpu().detach().numpy()
            pos = pos[relevant_regions]
            if heatmap:
                regions, alphas = self._heatmap_relevant_regions(
                    raw,
                    seg,
                    region_labels,
                    df,
                    pos,
                    explanation,
                    only_positive,
                    intensity_value=intensity_value,
                )
            else:
                regions, alphas = self._color_relevant_regions(
                    raw, seg, region_labels, df, pos
                )

            path = path[:-4]
            intensity_path = path + "_intensity.png"

            self.create_region_intensity_image(
                raw,
                regions,
                alphas,
                intensity_path,
                label_names,
                target,
                points,
                pos,
                graph,
                intensity_value=intensity_value,
            )


    def create_vessel_intensity_image(
        self,
        raw,
        cl_arr,
        cl_center_points,
        path,
        label_names,
        target,
        points,
        graph,
        vessel_alphas,
        intensity_value=None,
    ):
        #cl_arr = cl_arr / cl_arr.max()
        # alphas = alphas/alphas.max()
        # normalize the raw image
        raw = raw - raw.min()
        raw = raw / raw.max()

        h, w = raw.shape
        figsize = (w / 100, h / 100)

        #fig = plt.figure(figsize=figsize)
        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")

        # use the cl_arr and regions as independent grad cam masks on the raw image
        ax.imshow(raw, alpha=1, cmap="Greys_r")
        if intensity_value is not None:
            ax.imshow(cl_arr, alpha=vessel_alphas, cmap="OrRd", vmin=0, vmax=1.1)
        else:
            vabs = max(np.abs(cl_arr.min()), cl_arr.max())
            im = ax.imshow(cl_arr, alpha=vessel_alphas, cmap="jet", vmin=-vabs, vmax=vabs)
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.5)
            fig.colorbar(im, cax=cax, orientation='vertical')

        if points:
            # check if cl_center_points is empty
            if cl_center_points.size != 0 and points:
                # print the number of relevant vessels
                print(f"Number of relevant vessels: {len(cl_center_points)}")
                ax.scatter(
                    cl_center_points[:, 1],
                    cl_center_points[:, 0],
                    c="blue",
                    s=15,
                    alpha=1,
                )

            # if label_names and target is not None:
            #    textstr = '\n'.join((
            #        "True Label: %s" % (label_names[hetero_graph.y[0].item()], ),
            #        "Predicted Label: %s" % (label_names[target], )))
            #    plt.text(0.6, 0.98, textstr, transform=plt.transAxes, fontsize=16,
            #        verticalalignment='top', bbox=dict(boxstyle='square', facecolor='grey', alpha=1))


        if path is not None:
            # check if the path exists otherwise create it
            directory = os.path.dirname(path)
            if not os.path.exists(directory):
                os.makedirs(directory)
            fig.savefig(path, bbox_inches='tight')
            plt.close("all")


    def create_region_intensity_image(
        self,
        raw,
        regions,
        alphas,
        path,
        label_names,
        target,
        points,
        pos,
        graph,
        intensity_value=None,
    ):
        #regions = regions / regions.max()
        # alphas = alphas/alphas.max()
        # normalize the raw image
        raw = raw - raw.min()
        raw = raw / raw.max()

        h, w = raw.shape
        figsize = (w / 100, h / 100)

        #fig = plt.figure(figsize=figsize)
        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")

        # use the cl_arr and regions as independent grad cam masks on the raw image
        ax.imshow(raw, alpha=1, cmap="Greys_r")
        if intensity_value is not None:
            ax.imshow(regions, alpha=alphas, cmap="ocean", vmin=0, vmax=2)
        else:
            vabs = max(np.abs(regions.min()), regions.max())
            im = ax.imshow(regions, alpha=alphas, cmap="jet", vmin=-vabs, vmax=vabs)
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.5)
            fig.colorbar(im, cax=cax, orientation='vertical')

        if points:
            # print the number of relevant regions
            print(f"Number of relevant regions: {len(pos)}")
            ax.scatter(pos[:, 1], pos[:, 0], c="orange", s=15, alpha=1, marker="s")


            # if label_names and target is not None:
            #    textstr = '\n'.join((
            #        "True Label: %s" % (label_names[hetero_graph.y[0].item()], ),
            #        "Predicted Label: %s" % (label_names[target], )))
            #    plt.text(0.6, 0.98, textstr, transform=plt.transAxes, fontsize=16,
            #        verticalalignment='top', bbox=dict(boxstyle='square', facecolor='grey', alpha=1))

        if path is not None:
            # check if the path exists otherwise create it
            directory = os.path.dirname(path)
            if not os.path.exists(directory):
                os.makedirs(directory)
            fig.savefig(path, bbox_inches='tight')
            plt.close("all")

    def create_segmentation_image(
        self,
        raw,
        seg,
        cl_arr,
        regions,
        alphas,
        cl_center_points,
        path,
        label_names,
        target,
        points,
        pos,
        hetero_graph,
        intensity_value=None,
    ):
        cl_arr = cl_arr / cl_arr.max()
        regions = regions / regions.max()
        # alphas = alphas/alphas.max()
        # normalize the raw image
        raw = raw - raw.min()
        raw = raw / raw.max()

        # creatae alpha mask with 1 where regions and 0 anywhere else
        alphas = np.zeros_like(raw, dtype=np.float32)
        alphas[regions != 0] = 1
        # do the same for the vessel alphas
        vessel_alphas = np.zeros_like(raw, dtype=np.float32)
        vessel_alphas[cl_arr != 0] = 1

        h, w = raw.shape
        figsize = (w / 100, h / 100)

        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")

        # plotting the segmentation
        ax.imshow(seg, cmap="Greys_r")
        if intensity_value is not None:
            ax.imshow(cl_arr, alpha=vessel_alphas, cmap="OrRd", vmin=0, vmax=1.1)
            ax.imshow(regions, alpha=alphas, cmap="ocean", vmin=0, vmax=2)
        else:
            ax.imshow(cl_arr, alpha=vessel_alphas, cmap="jet", vmin=0, vmax=1)
            ax.imshow(regions, alpha=alphas, cmap="jet", vmin=0, vmax=1)

        if points:
            # print the number of relevant regions
            print(f"Number of relevant regions: {len(pos)}")
            ax.scatter(pos[:, 1], pos[:, 0], c="orange", s=15, alpha=1, marker="s")

            # check if cl_center_points is empty
            if cl_center_points.size != 0 and points:
                # print the number of relevant vessels
                print(f"Number of relevant vessels: {len(cl_center_points)}")
                ax.scatter(
                    cl_center_points[:, 1],
                    cl_center_points[:, 0],
                    c="blue",
                    s=15,
                    alpha=1,
                )

            # if label_names and target is not None:
            #    textstr = '\n'.join((
            #        "True Label: %s" % (label_names[hetero_graph.y[0].item()], ),
            #        "Predicted Label: %s" % (label_names[target], )))
            #    plt.text(0.6, 0.98, textstr, transform=plt.transAxes, fontsize=16,
            #        verticalalignment='top', bbox=dict(boxstyle='square', facecolor='grey', alpha=1))

        if path is not None:
            # check if the path exists otherwise create it
            directory = os.path.dirname(path)
            if not os.path.exists(directory):
                os.makedirs(directory)
            fig.savefig(path)
            plt.close("all")
