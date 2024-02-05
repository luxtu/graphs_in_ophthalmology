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

    def _heatmap_relevant_vessels(self, raw, relevant_vessels_pre, vvg_df_edges, explanations):
        # for the vessels create a mask that containts the centerline of the relevant vessels
        # also store the positions of the center points on the relevant vessels


        importance_array = explanations.node_mask_dict["graph_1"].abs().sum(dim=-1).cpu().detach().numpy()
        print(len(importance_array))
        print(len(vvg_df_edges["pos"]))

        relevant_vessels = np.where(relevant_vessels_pre)[0]
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


    def _heatmap_relevant_regions(self, raw, seg, region_labels, faz_region_label, relevant_faz, df, pos, explanations):
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
        importance_array = explanations.node_mask_dict["graph_2"].abs().sum(dim=-1).cpu().detach().numpy()
        print("Region importance")
        for i, label in enumerate(relevant_region_labels):
            importance = importance_array[relevant_region_indices[i]].item()
            print(importance)
            #alphas[region_labels == label] = importance
            regions[region_labels == label] = importance


        if self.faz_node and relevant_faz is not None:
            print("Faz importance")
            importance = explanations.node_mask_dict["faz"].abs().sum(dim=-1).cpu().detach().numpy()[0].item()
            print(importance)
            #alphas[region_labels == faz_region_label] = importance
            regions[region_labels == faz_region_label] = importance

        alphas[alphas == 0] = 0
        alphas[seg!=0] = 0

        alphas[regions!=0] = 0.3
        
        return regions, alphas


    def create_explanation_image(self, explanation, hetero_graph,  graph_id, path, faz_node=False, label_names = None, target = None, heatmap = False, explained_gradient = 0.95, **kwargs):
        
        
        self.faz_node = faz_node
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


        # relevant nodes dict, getting the nodes above a certain threshold, includes all types of nodes
        het_graph_rel_pos_dict  = torch_geom_explanation.identifiy_relevant_nodes(explanation,hetero_graph, faz_node=self.faz_node, explained_gradient= explained_gradient)

        # extract the relevant vessels
        vvg_df_edges, vvg_df_nodes = vvg_loader.vvg_to_df(os.path.join(self.vvg_path, vvg_file))
        
        # extract the relevant vessels, with a segmentation mask and the center points of the relevant vessels
        if heatmap:
            cl_arr, cl_center_points, vessel_alphas = self._heatmap_relevant_vessels(raw, het_graph_rel_pos_dict["graph_1"], vvg_df_edges, explanation)
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

        # extract relevant positions
        relevant_regions = np.where(het_graph_rel_pos_dict["graph_2"])[0]
        # pos are the positions of relevant regions
        pos = hetero_graph["graph_2"].pos.cpu().detach().numpy()
        pos = pos[relevant_regions]
        if heatmap:
            regions, alphas = self._heatmap_relevant_regions(raw, seg, region_labels, faz_region_label, relevant_faz, df, pos, explanation)
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
        ax[0].scatter(pos[:,1], pos[:,0], c="orange", s=15, alpha=1, marker = "s")
        ax[1].scatter(pos[:,1], pos[:,0], c="orange", s=15, alpha=1, marker = "s")

        # check if cl_center_points is empty
        if cl_center_points.size != 0:
            ax[0].scatter(cl_center_points[:,1], cl_center_points[:,0], c="blue", s=15, alpha=1)
            ax[1].scatter(cl_center_points[:,1], cl_center_points[:,0], c="blue", s=15, alpha=1)

        # plot center of faz region if it is relevant
        if faz_node and relevant_faz is not None:
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
