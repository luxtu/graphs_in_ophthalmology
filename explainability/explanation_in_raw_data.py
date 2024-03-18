# trying to create overlays of the explanations on the raw data


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
from PIL import Image
from loader import vvg_loader
from skimage import measure, morphology
import matplotlib.colors as mcolors
from skimage import transform
import torch
from explainability import torch_geom_explanation
from utils import vvg_tools



class RawDataExplainer():


    def __init__(self, raw_image_path, segmentation_path, vvg_path):
        self.raw_image_path = raw_image_path
        self.segmentation_path = segmentation_path
        self.vvg_path = vvg_path

        self.raw_images = os.listdir(self.raw_image_path)
        self.segmentations = os.listdir(self.segmentation_path)
        self.vvgs = os.listdir(self.vvg_path)


        #print(f"Found {len(self.vvgs)} vvgs")
        #print(f"Found {len(self.segmentations)} segmentations")
        #print(f"Found {len(self.raw_images)} raw images")
#
        #print('#'*50)

        # extract all files from vvg files that end with .json or .json.gz
        self.vvgs = [file for file in self.vvgs if file.endswith(".json") or file.endswith(".json.gz")]
        #print(f"Found {len(self.raw_images)} vvgs with correct file ending")
        # extract all files from seg files that end with .png
        self.segmentations = [file for file in self.segmentations if file.endswith(".png")]
        #print(f"Found {len(self.segmentations)} segmentations with correct file ending")
        # extract all files from raw image files that end with .png
        self.raw_images = [file for file in self.raw_images if file.endswith(".png")]
        #print(f"Found {len(self.raw_images)} raw_images with correct file ending")



    def _process_files(self, graph_id):
        seg_file = [file for file in self.segmentations if graph_id in file]
        assert len(seg_file) == 1, "There should be only one segmentation file for this graph_id"
        seg_file = seg_file[0]
        

        raw_file = [file for file in self.raw_images if graph_id in file]
        assert len(raw_file) == 1, "There should be only one raw image file for this graph_id"
        raw_file = raw_file[0]

        vvg_file = [file for file in self.vvgs if graph_id in file]
        assert len(vvg_file) == 1, "There should be only one vvg file for this graph_id"
        vvg_file = vvg_file[0]

        return seg_file, raw_file, vvg_file


    def _identify_faz_region_label(self, region_labels):
        # faz_region is the region with label at 600,600
        faz_region_label = region_labels[600,600]
        idx_y = 600
        while faz_region_label == 0:
            idx_y += 1
            faz_region_label = region_labels[idx_y,600]
        return faz_region_label
    
    def _get_one_hop_neighbors(self, edge_index, relevant_nodes):
        """
        Takes the edge indices and the relevant nodes and returns the one hop neighbors of the relevant nodes
        """

        # get the one hop neighbors of the relevant nodes
        one_hop_neighbors = []
        for node in relevant_nodes:
            # get the indices of the edges where the node is the source
            source_edges = np.where(edge_index[0] == node)[0]
            # get the target nodes of the source edges
            target_nodes = edge_index[1][source_edges]
            # append the target nodes to the one_hop_neighbors
            one_hop_neighbors.append(target_nodes)
        # flatten the list
        one_hop_neighbors = np.concatenate(one_hop_neighbors)
        # remove duplicates
        one_hop_neighbors = np.unique(one_hop_neighbors)
        return one_hop_neighbors
        


    def _heatmap_relevant_vessels(self, raw, relevant_vessels_pre, vvg_df_edges, explanations, graph, vvg_df_nodes, matched_list, one_hop_neighbors = None, only_positive = False):
        # for the vessels create a mask that containts the centerline of the relevant vessels
        # also store the positions of the center points on the relevant vessels

        #  match the vessels in the graph to the vessels in the vvg
        cl_arr = np.zeros_like(raw, dtype=np.float32)
        vessel_alphas = np.zeros_like(raw, dtype=np.float32)
        #extract the node positions from the graph
        node_positions = graph["graph_1"].pos.cpu().detach().numpy()
        # get the importance of the vessels
        if only_positive:
            importance = explanations.node_mask_dict["graph_1"].sum(dim=-1).cpu().detach().numpy()
        else:
            importance = explanations.node_mask_dict["graph_1"].abs().sum(dim=-1).cpu().detach().numpy()
        # get the positions of the relevant vessels
        relevant_vessels = np.where(relevant_vessels_pre)[0]
        if one_hop_neighbors is not None:
            relevant_vessels = np.concatenate((relevant_vessels, one_hop_neighbors))
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
            center_point = (np.array(pos1) + np.array(pos2))/2
            center_points_vvg.append(center_point)
        center_points_vvg = np.array(center_points_vvg)[:,:2]


        # create a regions for every vessel in the vvgs
        image, seg = matched_list
        # make seg binary
        seg = seg > 0

        final_seg_label = np.zeros_like(seg, dtype = np.uint16)
        final_seg_label[seg!=0] = 1
        
        cl_vessel = vvg_tools.vvg_df_to_centerline_array_unique_label(vvg_df_edges, vvg_df_nodes, (1216, 1216), vessel_only=True)
        label_cl = cl_vessel# measure.label(cl_vessel)
        label_cl[label_cl!=0] = label_cl[label_cl!=0] + 1
        final_seg_label[label_cl!=0] = label_cl[label_cl!=0]

        for i in range (100):
            label_cl = morphology.dilation(label_cl, morphology.square(3))
            label_cl = label_cl * seg
            # get the values of final_seg_label where no semantic segmentation is present
            final_seg_label[final_seg_label==1] = label_cl[final_seg_label==1]
            # get indices where label_cl==0 and seg !=0
            mask = (final_seg_label == 0) & (seg != 0)
            final_seg_label[mask] = 1

        # pixels that are still 1 are turned into 0
        final_seg_label[final_seg_label==1] = 0
        # labels for the rest are corrected by -1
        final_seg_label[final_seg_label!=0] = final_seg_label[final_seg_label!=0] - 1
        cl_center_points = []
        for i, cp in enumerate(center_points):
            # get the closest center point in the vvg
            dist = np.linalg.norm(center_points_vvg - cp, axis=1)
            closest_vessel = np.argmin(dist)
            # get the closest vessel in the vessel_vvg
            positions = vvg_df_edges["pos"].iloc[closest_vessel]
            frequent_label = {}
            if one_hop_neighbors is not None:
                # only add the centerline points if the vessel is not in the one_hop_neighbors
                if i not in one_hop_neighbors:
                    cl_center_points.append(positions[int(len(positions)/2)])

            else:
                cl_center_points.append(positions[int(len(positions)/2)])
            for pos in positions:
                label = final_seg_label[int(pos[0]), int(pos[1])]
                if label in frequent_label:
                    frequent_label[label] += 1
                else:
                    frequent_label[label] = 1
            # get the most frequent label, by the number of occurences
            max_label = max(frequent_label, key=frequent_label.get)
            # set the cl_arr_value to the importance of the vessel
            cl_arr[final_seg_label == max_label] = relevant_importance[i]
            vessel_alphas[final_seg_label == max_label] = 0.7
        
        cl_center_points = np.array(cl_center_points)


        return cl_arr, cl_center_points, vessel_alphas




        """
        importance_array = explanations.node_mask_dict["graph_1"].abs().sum(dim=-1).cpu().detach().numpy()
        cl_arr = np.zeros_like(raw, dtype=np.float32)
        vessel_alphas = np.zeros_like(raw, dtype=np.float32)
        cl_center_points = []
        print("Vessel importance")
        for i, cl in enumerate(vvg_df_edges["pos"]):
            # only add the centerline of the relevant vessels
            if i in relevant_vessels:
                # extract the centerpoints of the relevant vessels
                # extract the importance of the relevant vessels from the explanation
                importance = importance_array[i].item()
                
                print(importance)
                cl_center_points.append(cl[int(len(cl)/2)])
                for pos in cl:
                    cl_arr[int(pos[0]), int(pos[1])] = importance
                    vessel_alphas[int(pos[0]), int(pos[1])] = 0.7
                    # also color the neighboring pixels if they exist
                    # create indices for the neighboring pixels
                    neigh_pix = np.array([[-1,0], [1,0], [0,-1], [0,1]])
                    # include the 2nd order neighbors
                    neigh_pix = np.concatenate((neigh_pix, np.array([[-1,-1], [-1,1], [1,-1], [1,1]])))
                    # include the 3rd order neighbors
                    neigh_pix = np.concatenate((neigh_pix, np.array([[-2,0], [2,0], [0,-2], [0,2]])))
                    # convert pos to int
                    pos = np.array([int(pos[0]), int(pos[1])]).astype(int)
                    for neigb_pos in neigh_pix:
                        neigb_pos += pos
                    neigh_pix = neigh_pix.astype(int)
                    # check if the neighboring pixels are in the image
                    neigh_pix = neigh_pix[(neigh_pix[:,0] >= 0) & (neigh_pix[:,0] < cl_arr.shape[0]) & (neigh_pix[:,1] >= 0) & (neigh_pix[:,1] < cl_arr.shape[1])]
                    # set the neighboring pixels to 
                    cl_arr[neigh_pix[:,0], neigh_pix[:,1]] = importance
                    vessel_alphas[neigh_pix[:,0], neigh_pix[:,1]] = 0.7

        cl_center_points = np.array(cl_center_points)
        # assue that 2 dimensions are always returned
        if len(cl_center_points.shape) == 1:
            cl_center_points = np.expand_dims(cl_center_points, axis=0)

        return cl_arr, cl_center_points, vessel_alphas 
        """




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
                cl_center_points.append(cl[int(len(cl)/2)])
                for pos in cl:
                    cl_arr[int(pos[0]), int(pos[1])] = 1
                    # also color the neighboring pixels if they exist
                    # create indices for the neighboring pixels
                    neigh_pix = np.array([[-1,0], [1,0], [0,-1], [0,1]])
                    # include the 2nd order neighbors
                    neigh_pix = np.concatenate((neigh_pix, np.array([[-1,-1], [-1,1], [1,-1], [1,1]])))
                    # include the 3rd order neighbors
                    neigh_pix = np.concatenate((neigh_pix, np.array([[-2,0], [2,0], [0,-2], [0,2]])))
                    # convert pos to int
                    pos = np.array([int(pos[0]), int(pos[1])]).astype(int)
                    for neigb_pos in neigh_pix:
                        neigb_pos += pos
                    neigh_pix = neigh_pix.astype(int)
                    # check if the neighboring pixels are in the image
                    neigh_pix = neigh_pix[(neigh_pix[:,0] >= 0) & (neigh_pix[:,0] < cl_arr.shape[0]) & (neigh_pix[:,1] >= 0) & (neigh_pix[:,1] < cl_arr.shape[1])]
                    # set the neighboring pixels to 1
                    cl_arr[neigh_pix[:,0], neigh_pix[:,1]] = 1

        cl_center_points = np.array(cl_center_points)
        # assue that 2 dimensions are always returned
        if len(cl_center_points.shape) == 1:
            cl_center_points = np.expand_dims(cl_center_points, axis=0)

        return cl_arr, cl_center_points

    def _color_relevant_regions(self, raw, seg, region_labels, faz_region_label, relevant_faz, df, pos):

        relevant_region_labels = []
        for position in pos:
            lab_1 = df.loc[np.isclose(df['centroid-0'], position[0])]["label"].values
            lab_2 = df.loc[np.isclose(df['centroid-1'], position[1])]["label"].values            

            for val_1 in lab_1:
                for val_2 in lab_2:
                    if val_1 == val_2:
                        relevant_region_labels.append(val_1)


        # extract the relevant regions
        regions = np.zeros_like(raw, dtype= "uint8")
        alphas = np.zeros_like(raw, dtype=np.float32)
        alphas += 0.2
        # this highlights all vessels
        alphas[seg!=0] = 1

        # set the alpha of the relevant regions to 1
        dyn_region_label = 1
        for label in relevant_region_labels:
            alphas[region_labels == label] = 0.85
            regions[region_labels == label] = dyn_region_label
            dyn_region_label += 1

        if self.faz_node and relevant_faz is not None:
            alphas[region_labels == faz_region_label] = 0.85
            regions[region_labels == faz_region_label] = dyn_region_label
            dyn_region_label += 1

        return regions, alphas


    def _heatmap_relevant_regions(self, raw, seg, region_labels, faz_region_label, relevant_faz, df, pos, explanations, only_positive = False):
        relevant_region_labels = []
        relevant_region_indices = []
        for i, position in enumerate(pos):
            lab_1 = df.loc[np.isclose(df['centroid-0'], position[0])]["label"].values
            lab_2 = df.loc[np.isclose(df['centroid-1'], position[1])]["label"].values            

            for val_1 in lab_1:
                for val_2 in lab_2:
                    if val_1 == val_2:
                        relevant_region_labels.append(val_1)
                        relevant_region_indices.append(i)
                        # can be break here, since the labels are unique
                        break


        # extract the relevant regions
        regions = np.zeros_like(raw, dtype= np.float32)
        alphas = np.zeros_like(raw, dtype=np.float32)
        

        # get the importance of the relevant regions
        if only_positive:
            importance_array = explanations.node_mask_dict["graph_2"].sum(dim=-1).cpu().detach().numpy()
        else:
            importance_array = explanations.node_mask_dict["graph_2"].abs().sum(dim=-1).cpu().detach().numpy()
        for i, label in enumerate(relevant_region_labels):
            importance = importance_array[relevant_region_indices[i]].item()
            #alphas[region_labels == label] = importance
            regions[region_labels == label] = importance


        if self.faz_node and relevant_faz is not None:
            if only_positive:
                importance = explanations.node_mask_dict["faz"].sum(dim=-1).cpu().detach().numpy()[0].item()
            else:
                importance = explanations.node_mask_dict["faz"].abs().sum(dim=-1).cpu().detach().numpy()[0].item()
            #alphas[region_labels == faz_region_label] = importance
            regions[region_labels == faz_region_label] = importance

        alphas[alphas == 0] = 0
        alphas[seg!=0] = 0

        alphas[regions!=0] = 0.3
        
        return regions, alphas


    def create_explanation_image(self, explanation, hetero_graph,  graph_id, path, label_names = None, target = None, heatmap = False, explained_gradient = 0.95, only_positive = False, points = False):
        
        
        # extract the relevant segmentation, raw image and vvg
        # search the file strings for the graph_id
        seg_file, raw_file, vvg_file = self._process_files(graph_id)

        # load the segmentation, raw image and vvg
        seg = Image.open(os.path.join(self.segmentation_path, seg_file))
        seg = np.array(seg)
        seg = seg.astype(np.uint8)        

        raw = Image.open(os.path.join(self.raw_image_path, raw_file))
        raw = np.array(raw)[:,:,0]
        raw = transform.resize(raw, seg.shape, order = 0, preserve_range = True)


        self.faz_node = False
        if "faz" in explanation.node_mask_dict.keys():
            self.faz_node = True
        # relevant nodes dict, getting the nodes above a certain threshold, includes all types of nodes
        if isinstance(explained_gradient, float):
            het_graph_rel_pos_dict  = torch_geom_explanation.identifiy_relevant_nodes(explanation,hetero_graph, faz_node=self.faz_node, explained_gradient= explained_gradient, only_positive = only_positive)
        elif isinstance(explained_gradient, int):
            het_graph_rel_pos_dict  = torch_geom_explanation.top_k_important_nodes(explanation,hetero_graph, faz_node=self.faz_node, top_k= explained_gradient, only_positive = only_positive)
        elif explained_gradient == None:
            het_graph_rel_pos_dict = {}
            for key in hetero_graph.x_dict.keys():
                het_graph_rel_pos_dict[key] = np.ones(hetero_graph[key].x.shape[0], dtype=bool)
        else:
            raise ValueError("explained_gradient must be either a float, an int or None")

        # extract the relevant vessels
        vvg_df_edges, vvg_df_nodes = vvg_loader.vvg_to_df(os.path.join(self.vvg_path, vvg_file))

        # find the one-hop neighbors of the relevant vessels
        one_hop_neighbors_bool = False
        if one_hop_neighbors_bool:
            one_hop_neighbors = self._get_one_hop_neighbors(hetero_graph[("graph_1", "to", "graph_1")].edge_index.cpu().detach().numpy(), np.where(het_graph_rel_pos_dict["graph_1"])[0])
            # remove the relevant vessels from the one_hop_neighbors
            one_hop_neighbors = np.setdiff1d(one_hop_neighbors, np.where(het_graph_rel_pos_dict["graph_1"])[0])



        # rem

        
        # extract the relevant vessels, with a segmentation mask and the center points of the relevant vessels
        if heatmap:
            cl_arr, cl_center_points, vessel_alphas = self._heatmap_relevant_vessels(raw, het_graph_rel_pos_dict["graph_1"], vvg_df_edges, explanation, hetero_graph, vvg_df_nodes, [raw, seg], one_hop_neighbors = None, only_positive = only_positive)
        else:
            cl_arr, cl_center_points = self._color_relevant_vessels(raw, het_graph_rel_pos_dict["graph_1"], vvg_df_edges, )


        # extract the relevant regions
        region_labels = measure.label(morphology.remove_small_holes(seg, area_threshold=5, connectivity=1).astype("uint8"), background=1)
        props = measure.regionprops_table(region_labels, properties=('label', 'centroid', 'area',))
        df = pd.DataFrame(props)  

        # process the faz region if it is relevant
        if self.faz_node:
            faz_region_label = self._identify_faz_region_label(region_labels)
            # create a dataframe with only the faz region
            df_faz_node = df.iloc[faz_region_label-1]
            # create a dataframe with all the other regions
            df = df.drop(faz_region_label-1)
            ## adjust labels of the regions
            df.index = np.arange(len(df))
            # check if the faz region is relevant
            relevant_faz = np.where(het_graph_rel_pos_dict["faz"])[0]
            # check if the faz region is an empty array
            if relevant_faz.size == 0:
                faz_region_label = None
                relevant_faz = None
        else:
            faz_region_label = None
            relevant_faz = None
        print(relevant_faz)

        # extract relevant positions
        relevant_regions = np.where(het_graph_rel_pos_dict["graph_2"])[0]
        # pos are the positions of relevant regions
        pos = hetero_graph["graph_2"].pos.cpu().detach().numpy()
        pos = pos[relevant_regions]
        if heatmap:
            regions, alphas = self._heatmap_relevant_regions(raw, seg, region_labels, faz_region_label, relevant_faz, df, pos, explanation, only_positive)
        else:
            regions, alphas = self._color_relevant_regions(raw, seg, region_labels, faz_region_label, relevant_faz, df, pos)





        fig, ax  = plt.subplots(1,2, figsize=(20,10))
        # plotting the raw image 
        # scal the alpha values between 0 and 1

        if heatmap:
            print(cl_arr.max(), regions.max())
            print(cl_arr.min(), regions.min())
            print(raw.max(), raw.min())

            cl_arr = cl_arr/cl_arr.max()
            regions = regions/regions.max()
            #alphas = alphas/alphas.max()
            # normalize the raw image
            raw = raw - raw.min()
            raw = raw/raw.max()

            # use the cl_arr and regions as independent grad cam masks on the raw image
            ax[0].imshow(raw, alpha = 0.8, cmap = "Greys_r")
            ax[0].imshow(regions, alpha = alphas,  cmap="jet", vmin=0, vmax=1) #, alpha = alphas, , cmap="jet"
            ax[0].imshow(cl_arr,  alpha = vessel_alphas,  cmap="jet", vmin=0, vmax=1) 

            # creatae alpha mask with 1 where regions and 0 anywhere else
            alphas = np.zeros_like(raw, dtype=np.float32)
            alphas[regions!=0] = 1
            # do the same for the vessel alphas
            vessel_alphas = np.zeros_like(raw, dtype=np.float32)
            vessel_alphas[cl_arr!=0] = 1
            
            # plotting the segmentation
            ax[1].imshow(seg, cmap="Greys_r")
            ax[1].imshow(regions, alpha = alphas, cmap="jet", vmin=0, vmax=1)
            ax[1].imshow(cl_arr, alpha = vessel_alphas, cmap="jet", vmin=0, vmax=1)

        else:
            ax[0].imshow(raw, alpha = alphas, cmap="gray")
            ax[0].imshow(regions, alpha=alphas, cmap="Greys_r", vmin=0, vmax=1) 
            ax[0].imshow(cl_arr,  alpha=cl_arr,  cmap="Greys_r", vmin=0, vmax=1)

            # plotting the segmentation
            ax[1].imshow(regions, alpha=alphas) # cmap="Greys_r", vmin=0, vmax=1
            ax[1].imshow(cl_arr, alpha=cl_arr,  cmap="Greys_r", vmin=0, vmax=1)


        # plot the center of the relevant regions
        if points:
            ax[0].scatter(pos[:,1], pos[:,0], c="orange", s=15, alpha=1, marker = "s")
            ax[1].scatter(pos[:,1], pos[:,0], c="orange", s=15, alpha=1, marker = "s")

        # check if cl_center_points is empty
        if cl_center_points.size != 0 and points:
            ax[0].scatter(cl_center_points[:,1], cl_center_points[:,0], c="blue", s=15, alpha=1)
            ax[1].scatter(cl_center_points[:,1], cl_center_points[:,0], c="blue", s=15, alpha=1)

        # plot center of faz region if it is relevant
        if self.faz_node and relevant_faz is not None and points:
            # get the center of the faz region
            faz_pos = df_faz_node[["centroid-1", "centroid-0"]].values
            ax[0].scatter(faz_pos[0], faz_pos[1], c="red", s=15, alpha=1, marker="D")
            ax[1].scatter(faz_pos[0], faz_pos[1], c="red", s=15, alpha=1, marker="D")
            #ax[0].scatter(600, 600, c="red", s=15, alpha=0.5)
            #ax[1].scatter(600, 600, c="red", s=15, alpha=0.5)

        if label_names and target is not None:
            textstr = '\n'.join((
                "True Label: %s" % (label_names[hetero_graph.y[0].item()], ),
                "Predicted Label: %s" % (label_names[target], )))

            ax[1].text(0.6, 0.98, textstr, transform=ax[1].transAxes, fontsize=16,
                verticalalignment='top', bbox=dict(boxstyle='square', facecolor='grey', alpha=1))

        if path is not None:
            plt.tight_layout()
            plt.savefig(path)
            plt.close()
        else:
            return ax
