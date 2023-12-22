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





    def create_explanation_image(self, explanation, hetero_graph,  graph_id, path, faz_node=False, threshold = "adaptive", edge_threshold = "adaptive"):
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



        relevant_vessels, relevant_regions, relevant_faz = self.identifiy_relevant_nodes(explanation,hetero_graph, threshold = threshold, edge_threshold = edge_threshold, faz_node=faz_node)

        #print("raw_data_regions")
        #print(relevant_regions)
        # for the vessels create a mask that containts the centerline of the relevant vessels

        cl_arr = np.zeros_like(raw, dtype=np.float32)
        for i, cl in enumerate(vvg_df_edges["pos"]):
            # only add the centerline of the relevant vessels
            if i in relevant_vessels:
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
        #ax[0].imshow(raw)
        #ax[1].imshow(seg)
        #ax[2].imshow(region_labels)
        ax[0].imshow(raw, alpha = alphas, cmap="gray")
        ax[0].imshow(cl_arr,  alpha=cl_arr, cmap="Greys_r", vmin=0, vmax=1)

        ax[1].imshow(regions,  alpha=alphas)
        ax[1].imshow(cl_arr,  alpha=cl_arr, cmap="Greys_r", vmin=0, vmax=1)

        #ax.imshow(alphas, cmap="gray")

        # plot the center of the relevant regions
        #ax[0].scatter(pos[:,1], pos[:,0], c="red", s=15, alpha=0.5)
        #ax[1].scatter(pos[:,1], pos[:,0], c="red", s=15, alpha=0.5)

        # plot center of faz region if it is relevant
        #if faz_node and relevant_faz is not None:
        #    ax[0].scatter(600, 600, c="red", s=15, alpha=0.5)
        #    ax[1].scatter(600, 600, c="red", s=15, alpha=0.5)

        if path is not None:
            plt.tight_layout()
            plt.savefig(path)
            plt.close()
        else:
            return ax



    def identifiy_relevant_nodes(self, explanation_graph, hetero_graph, threshold="adaptive", edge_threshold="adaptive", faz_node=False):

        graph_1_name = "graph_1"
        graph_2_name = "graph_2"
        graph_3_name = "faz"


        if threshold == "adaptive":

            total_grad = explanation_graph.node_mask_dict[graph_1_name].abs().sum() + explanation_graph.node_mask_dict[graph_2_name].abs().sum()
            #avg_grad = total_grad / (explanation_graph.node_mask_dict[graph_1_name].shape[0] + explanation_graph.node_mask_dict[graph_2_name].shape[0])

            if faz_node:
                total_grad += explanation_graph.node_mask_dict[graph_3_name].abs().sum()

            # find a threshold such that 90% of the total gradient is explained

            node_value = torch.cat((explanation_graph.node_mask_dict[graph_1_name].abs().sum(dim=-1), explanation_graph.node_mask_dict[graph_2_name].abs().sum(dim=-1)))
            if faz_node:
                node_value = torch.cat((explanation_graph.node_mask_dict[graph_3_name].abs().sum(dim=-1), node_value))


            sorted_node_value = torch.sort(node_value, descending=True)[0]
            # get rid of the nodes that contribute less than 0.1% to the total gradient
            sorted_node_value = sorted_node_value[sorted_node_value > 0.0005 * total_grad]


            cum_sum = torch.cumsum(sorted_node_value, dim=0)
            cropped_grad = cum_sum[-1]
            threshold = sorted_node_value[cum_sum < 0.95 * cropped_grad][-1]

            #threshold = avg_grad #max_val * 0.01
            print("threshold", threshold)



        het_graph1_rel_pos = explanation_graph.node_mask_dict[graph_1_name].abs().sum(dim=-1) > threshold
        het_graph2_rel_pos = explanation_graph.node_mask_dict[graph_2_name].abs().sum(dim=-1) > threshold

        het_graph1_rel_pos = het_graph1_rel_pos.cpu().detach().numpy()
        het_graph2_rel_pos = het_graph2_rel_pos.cpu().detach().numpy()

        # remove the nodes from the corner

        het_graph1_pos = hetero_graph[graph_1_name].pos.cpu().detach().numpy()
        het_graph2_pos = hetero_graph[graph_2_name].pos.cpu().detach().numpy()


        het_graph1_rel_pos = het_graph1_rel_pos & (het_graph1_pos[:,0] < 1100) & (het_graph1_pos[:,1] > 100) 
        het_graph2_rel_pos = het_graph2_rel_pos & (het_graph2_pos[:,0] < 1100) & (het_graph2_pos[:,1] > 100)


        if faz_node:
            het_graph3_rel_pos = explanation_graph.node_mask_dict[graph_3_name].abs().sum(dim=-1) > threshold
            het_graph3_rel_pos = het_graph3_rel_pos.cpu().detach().numpy()
        else:
            het_graph3_rel_pos = None

        return np.where(het_graph1_rel_pos)[0], np.where(het_graph2_rel_pos)[0], np.where(het_graph3_rel_pos)[0]

