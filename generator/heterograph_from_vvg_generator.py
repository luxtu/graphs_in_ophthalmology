import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from skimage import measure, morphology, transform
import json
import re

from utils import vvg_tools
from loader import vvg_loader



class HeterographFromVVGGenerator:

    def __init__(self, seg_path, vvg_path, void_graph_save_path, hetero_edges_save_path, image_path = None ,debug = False):

        self.seg_path = seg_path
        self.vvg_path = vvg_path
        self.void_graph_save_path = void_graph_save_path
        self.hetero_edges_save_path = hetero_edges_save_path

        # setting an image path will only influence node features
        self.image_path = image_path

        self.debug = debug

    def find_matching_files(self, seg_files, vvg_files, image_files = None):
        pattern = r'\d+'
        for seg_file in seg_files:
            for vvg_file in vvg_files:
                # Use re.findall() to extract all sequences of digits from the string
                idx_seg = re.findall(pattern, seg_file)[0]
                idx_vvg = re.findall(pattern, vvg_file)[0]

                if image_files is not None:
                    for image_file in image_files:
                        idx_image = re.findall(pattern, image_file)[0]
                        # check if the numbers are the same
                        idx_check =  idx_seg == idx_vvg == idx_image
                        # check if both filenames contain either "_OD" or "_OS"
                        eye_check = ("_OD" in seg_file and "_OD" in vvg_file and "_OD" in image_file) or ("_OS" in seg_file and "_OS" in vvg_file and "_OS" in image_file)
                        if idx_check and eye_check:
                            if "_OD" in seg_file:
                                eye = "OD"
                            else:
                                eye = "OS"
                            idx_dict = idx_seg + "_" + eye
                            print(idx_dict)
                            self.generate_region_graph(seg_file, vvg_file, idx_dict, image_file)
                    
                else:
                    # check if the numbers are the same
                    idx_check =  idx_seg == idx_vvg
                    # check if both filenames contain either "_OD" or "_OS"
                    eye_check = ("_OD" in seg_file and "_OD" in vvg_file) or ("_OS" in seg_file and "_OS" in vvg_file)
                    if idx_check and eye_check:

                        if "_OD" in seg_file:
                            eye = "OD"
                        else:
                            eye = "OS"
                        idx_dict = idx_seg + "_" + eye
                        print(idx_dict)
                        self.generate_region_graph(seg_file, vvg_file, idx_dict)


    def save_region_graphs(self):
        
        seg_files = os.listdir(self.seg_path)
        vvg_files = os.listdir(self.vvg_path)

        # extract all files from vvg files that end with .json or .json.gz
        vvg_files = [file for file in vvg_files if file.endswith(".json") or file.endswith(".json.gz")]
        # extract all files from seg files that end with .png
        seg_files = [file for file in seg_files if file.endswith(".png")]

        if self.image_path is not None:
            image_files = os.listdir(self.image_path)
            image_files = [file for file in image_files if file.endswith(".png")]
            self.find_matching_files(seg_files, vvg_files, image_files)

        else:
            self.find_matching_files(seg_files, vvg_files)

        # find matching files in both folders, not all the files will be matched


    def generate_region_graph(self, seg_file, vvg_file, idx_dict, image_file = None):

        seg = Image.open(os.path.join(self.seg_path, seg_file))
        seg = np.array(seg)
        seg = seg.astype(np.uint8)

        if image_file is not None:
            image = Image.open(os.path.join(self.image_path, image_file))
            image = np.array(image)[:,:,0]
            image = transform.resize(image, seg.shape, order = 0, preserve_range = True)

            #image = image.astype(np.uint8)
        # background is 1
        ## region properties don't make sense with extremely small regions, also they probably come from failed segmentation
        region_labels = measure.label(morphology.remove_small_holes(seg, area_threshold=5, connectivity=1).astype("uint8"), background=1)

        if self.debug:
            plt.imshow(region_labels)
            #plt.show()

        if self.image_path is not None:
            props = measure.regionprops_table(region_labels,intensity_image=image, 
                                              properties=('centroid',
                                                 'area',
                                                 'perimeter',
                                                 'eccentricity',
                                                 'equivalent_diameter',
                                                 'orientation',
                                                 'solidity',
                                                 'feret_diameter_max',
                                                 'extent',
                                                 'axis_major_length',
                                                 'axis_minor_length',
                                                 "intensity_max",
                                                 "intensity_mean",
                                                 "intensity_min",
                                                 "centroid_weighted"))
            
        else:

            props = measure.regionprops_table(region_labels, 
                                              properties=('centroid',
                                                 'area',
                                                 'perimeter',
                                                 'eccentricity',
                                                 'equivalent_diameter',
                                                 'orientation',
                                                 'solidity',
                                                 'feret_diameter_max',
                                                 'extent',
                                                 'axis_major_length',
                                                 'axis_minor_length'))
        
        df = pd.DataFrame(props)

        try:
            edge_index_df, edge_index_region_vessel_df = self.generate_edge_index(region_labels, vvg_file)
        except json.decoder.JSONDecodeError:
            print("JSONDecodeError")
            return None

        if not self.debug:
            df.to_csv(os.path.join(self.void_graph_save_path, str(idx_dict)+ "_region_nodes" + ".csv"), sep = ";")
            edge_index_df.to_csv(os.path.join(self.void_graph_save_path, str(idx_dict) + "_region_edges"+".csv"), sep = ";")
            edge_index_region_vessel_df.to_csv(os.path.join(self.hetero_edges_save_path, str(idx_dict) + "_region_vessel_edges"+".csv"), sep = ";")
        else:
            # plot the nodes
            plt.scatter(df["centroid-1"], df["centroid-0"], s = 8, c= "orange")
            # plot the connections between the nodes
            for id, edge in edge_index_df.iterrows():
                plt.plot([df["centroid-1"][edge["node1id"]], df["centroid-1"][edge["node2id"]]], [df["centroid-0"][edge["node1id"]], df["centroid-0"][edge["node2id"]]], c = "black", linewidth = 0.5)

            plt.show()


    def generate_edge_index(self, region_labels, vvg_file):


        # generate the centerline image from the vvg file
        skel_labels = self.generate_skel_labels(vvg_file, region_labels.shape)
        matching_regions = self.find_matching_regions(skel_labels, region_labels)


        region_neighbor_set = set()
        vessel_to_region_neighbor_set = set()
        success = 0
        fail = 0

        vvg_df_edges, _ = vvg_loader.vvg_to_df(os.path.join(self.vvg_path, vvg_file))

        for id, edge in vvg_df_edges.iterrows():
            for pix in edge["pos"]:

                pix_0 = int(pix[0])
                pix_1 = int(pix[1])
                # get the 4 neighborhood of the pixel
                neighbor_pixel_idcs = [(pix_0-1, pix_1), (pix_0+1, pix_1), (pix_0, pix_1-1), (pix_0, pix_1+1)]
                neighbor_pixel = []
                for neighb in neighbor_pixel_idcs:
                    try:
                        neighbor_pixel.append(skel_labels[neighb])
                    except IndexError:
                        pass

                for neighb in neighbor_pixel:
                    if neighb  == skel_labels[pix_0, pix_1]:
                        pass
                    else:
                        try:     
                            vessel_to_region_neighbor_set.add((id, matching_regions[neighb]-1))
                            success += 1
                        except KeyError:
                            fail += 1

                # create all combinations of the 4 neighbors
                for neighb1 in neighbor_pixel:
                    for neighb2 in neighbor_pixel:
                        if neighb1 == neighb2:
                            pass
                        elif neighb1 == skel_labels[pix_0, pix_1] or neighb2 == skel_labels[pix_0, pix_1]:
                            pass
                        else:
                            try:
                                r1 = min(matching_regions[neighb1], matching_regions[neighb2])
                                r2 = max(matching_regions[neighb1], matching_regions[neighb2])
                                region_neighbor_set.add((r1-1, r2-1))
                                success += 1
                            except KeyError:
                                fail += 1

        print(len(vessel_to_region_neighbor_set))
        print(success, fail)

        # create a dataframe for the edge list where the first column is an ID column and the second and third column are the indices of the nodes connected by the edge

        return pd.DataFrame(region_neighbor_set, columns=["node1id", "node2id"]), pd.DataFrame(vessel_to_region_neighbor_set, columns=["node1id", "node2id"])


    def generate_skel_labels(self, vvg_file, seg_shape):

        vvg_df_edges, vvg_df_nodes = vvg_loader.vvg_to_df(os.path.join(self.vvg_path, vvg_file))
        cl_arr = vvg_tools.vvg_df_to_centerline_array(vvg_df_edges,vvg_df_nodes, seg_shape)

        # make all border pixels 1
        cl_arr[:2,:] = 1
        cl_arr[-2:,:] = 1
        cl_arr[:,:2] = 1
        cl_arr[:,-2:] = 1
        skel_labels_border = measure.label(cl_arr, background=1, connectivity=1)

        return skel_labels_border

    def find_matching_regions(self, skel_labeled, seg_labeled):

        # finds for every region in the original image the corresponding region in the skeletonized image
        # if no regions was found in the skeletonized image, the region is not added to the dictionary
        # actually this should not happen 

        matching_regions = {}

        matched_regions = 0
        failed_regions = 0


        for skel_val in np.unique(skel_labeled)[np.unique(skel_labeled) != 0]:
            lab_vals, counts = np.unique(seg_labeled[np.where(skel_labeled == skel_val)], return_counts=True) 

            count_sort_ind = np.argsort(-counts)
            lab_vals = lab_vals[count_sort_ind]

            if lab_vals[0] ==0:
                try:
                    matching_regions[skel_val] = lab_vals[1]
                    matched_regions += 1
                except IndexError:
                    failed_regions += 1
                    pass
            else:
                matching_regions[skel_val] = lab_vals[0]
                matched_regions += 1

        print(matched_regions, failed_regions)
        return matching_regions