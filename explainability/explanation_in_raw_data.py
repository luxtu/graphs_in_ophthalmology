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





    def create_explanation_image(self, explanation, hetero_graph,  graph_id, path, faz_node=False, threshold = "adaptive", edge_threshold = "adaptive", label_names = None, target = None):
        # extract the relevant segmentation, raw image and vvg
        # search the file strings for the graph_id
        self.faz_node = faz_node

        seg_file = [file for file in self.segmentations if graph_id in file]
        assert len(seg_file) == 1, "There should be only one segmentation file for this graph_id"
        seg_file = seg_file[0]
        

        raw_file = [file for file in self.raw_images if graph_id in file]
        assert len(raw_file) == 1, "There should be only one raw image file for this graph_id"
        raw_file = raw_file[0]

        vvg_file = [file for file in self.vvgs if graph_id in file]
        assert len(vvg_file) == 1, "There should be only one vvg file for this graph_id"
        vvg_file = vvg_file[0]

        # load the segmentation, raw image and vvg
        seg = Image.open(os.path.join(self.segmentation_path, seg_file))
        seg = np.array(seg)
        seg = seg.astype(np.uint8)        

        raw = Image.open(os.path.join(self.raw_image_path, raw_file))
        raw = np.array(raw)[:,:,0]
        raw = transform.resize(raw, seg.shape, order = 0, preserve_range = True)


        vvg_df_edges, vvg_df_nodes = vvg_loader.vvg_to_df(os.path.join(self.vvg_path, vvg_file))

        region_labels = measure.label(morphology.remove_small_holes(seg, area_threshold=5, connectivity=1).astype("uint8"), background=1)

        props = measure.regionprops_table(region_labels, 
                                          properties=('label', 'centroid', 'area',))

        faz_region_label = region_labels[600,600]
        df = pd.DataFrame(props)  
        # copy the dataframe

        if self.faz_node:
            # faz_region is the region with label at 600,600
            faz_region_label = region_labels[600,600]
            idx_y = 600
            while faz_region_label == 0:
                idx_y += 1
                faz_region_label = region_labels[idx_y,600]

#
            #print(faz_region_label)
            # create a dataframe with only the faz region
            df_faz_node = df.iloc[faz_region_label-1]

            # create a dataframe with all the other regions
            df = df.drop(faz_region_label-1)

            ## adjust labels of the regions
            df.index = np.arange(len(df))


        # relevant nodes dict
        het_graph_rel_pos_dict  = torch_geom_explanation.identifiy_relevant_nodes(explanation,hetero_graph, faz_node=self.faz_node)

        relevant_vessels = np.where(het_graph_rel_pos_dict["graph_1"])[0]
        relevant_regions = np.where(het_graph_rel_pos_dict["graph_2"])[0]

        if self.faz_node:
            relevant_faz = np.where(het_graph_rel_pos_dict["faz"])[0]


        # for the vessels create a mask that containts the centerline of the relevant vessels
        # also store the positions of the center points on the relevant vessels
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


            

        pos = hetero_graph["graph_2"].pos.cpu().detach().numpy()
        pos = pos[relevant_regions]
        relevant_region_labels = []
        # extract relevant positions
       
        for position in pos:
            lab_1 = df.loc[np.isclose(df['centroid-0'], position[0])]["label"].values
            lab_2 = df.loc[np.isclose(df['centroid-1'], position[1])]["label"].values            

            for val_1 in lab_1:
                for val_2 in lab_2:
                    if val_1 == val_2:
                        relevant_region_labels.append(val_1)
        
        
        #print(relevant_region_labels)
         #in props find the regions that match the centroid_0 and centroid_1 of the relevant regions

        
        # finde the labels that correspond to the relevant regions
        #print(relevant_regions)
        #relevant_region_labels = df["label"].iloc[relevant_regions].values

        # also find the center of the relevant regions
        #cent_0 = df["centroid-0"].iloc[relevant_regions].values,
        #cent_1 = df["centroid-1"].iloc[relevant_regions].values



        #print(relevant_regions)
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

        if faz_node and relevant_faz is not None:
            alphas[region_labels == faz_region_label] = 0.85
            regions[region_labels == faz_region_label] = dyn_region_label
            dyn_region_label += 1




        fig, ax  = plt.subplots(1,2, figsize=(20,10))
        # plotting the raw image 
        ax[0].imshow(raw, alpha = alphas, cmap="gray")
        ax[0].imshow(cl_arr,  alpha=cl_arr, cmap="Greys_r", vmin=0, vmax=1)

        # plotting the segmentation
        ax[1].imshow(regions,  alpha=alphas)
        ax[1].imshow(cl_arr,  alpha=cl_arr, cmap="Greys_r", vmin=0, vmax=1)


        # plot the center of the relevant regions
        ax[0].scatter(pos[:,1], pos[:,0], c="orange", s=15, alpha=1, marker = "s")
        ax[1].scatter(pos[:,1], pos[:,0], c="orange", s=15, alpha=1, marker = "s")


        # plot the center of the relevant vessels
        ax[0].scatter(np.array(cl_center_points)[:,1], np.array(cl_center_points)[:,0], c="blue", s=15, alpha=1)
        ax[1].scatter(np.array(cl_center_points)[:,1], np.array(cl_center_points)[:,0], c="blue", s=15, alpha=1)


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

