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

    def __init__(self, seg_path, vvg_path, void_graph_save_path, hetero_edges_save_path, faz_node = False, image_path = None ,debug = False,force = False, faz_node_save_path = None, faz_region_edges_save_path = None, faz_vessel_edges_save_path = None):

        self.seg_path = seg_path
        self.vvg_path = vvg_path
        self.void_graph_save_path = void_graph_save_path
        self.hetero_edges_save_path = hetero_edges_save_path

        # setting an image path will only influence node features
        self.image_path = image_path
        self.faz_node = faz_node
        self.debug = debug
        self.force = False

        # with a faz node
        if self.faz_node:
            self.faz_node_save_path = faz_node_save_path
            self.faz_region_edges_save_path = faz_region_edges_save_path
            self.faz_vessel_edges_save_path = faz_vessel_edges_save_path

    def check_existing_heterograph(self, idx_dict):
        # check if there is already a heterograph file stored with this index in the save path
        # if there is, return True, else return False

        if self.faz_node:
            file_name = os.path.join(self.void_graph_save_path, str(idx_dict)+ "_region_nodes_faz_node" + ".csv")
        else:
            file_name = os.path.join(self.void_graph_save_path, str(idx_dict)+ "_region_nodes" + ".csv")  

        if os.path.exists(file_name): #os.path.join(self.void_graph_save_path, str(idx_dict)+ "_region_nodes" + ".csv")):
            return True
        else:
            return False

    def find_matching_files(self, seg_files, vvg_files, image_files = None):
        pattern = r'\d+'

        #extract the indices once from all the files
        seg_file_indx = [re.findall(pattern, file)[0] for file in seg_files]
        vvg_file_indx = [re.findall(pattern, file)[0] for file in vvg_files]

        # OD = True
        # OS = False
        seg_file_eye = [True if "_OD" in file else False for file in seg_files]
        vvg_file_eye = [True if "_OD" in file else False for file in vvg_files]

        if image_files is not None:
            image_file_indx = [re.findall(pattern, file)[0] for file in image_files]
            image_file_eye = [True if "_OD" in file else False for file in image_files]


        for i, seg_file in enumerate(seg_files):
            found_match = False
            
            for j, vvg_file in enumerate(vvg_files):
                # Use re.findall() to extract all sequences of digits from the string
                idx_seg = seg_file_indx[i]
                idx_vvg = vvg_file_indx[j]

                # check if the segmentation file and the vvg file have the same index and eye
                if idx_seg != idx_vvg or seg_file_eye[i] != vvg_file_eye[j]:
                    continue


                if image_files is not None:
                    for k, image_file in enumerate(image_files):
                        idx_image = image_file_indx[k]
                        # check if the numbers are the same
                        idx_check =  idx_seg == idx_vvg == idx_image
                        # check if all filenames contain either "_OD" or "_OS"
                        eye_check = seg_file_eye[i] == vvg_file_eye[j] == image_file_eye[k]
                        if idx_check and eye_check:
                            eye = "OD" if seg_file_eye[i] else "OS"
                            idx_dict = idx_seg + "_" + eye
                            print(idx_dict)
                            # only regenerate the graph if it does not exist yet or if the force flag is set
                            if not self.check_existing_heterograph(idx_dict) or self.force:
                                self.generate_region_graph(seg_file, vvg_file, idx_dict, image_file)

                            # break of inner loop
                            found_match = True
                            break

                else:
                    # if the idx and eye do not match the loop will continue 
                    eye = "OD" if seg_file_eye[i] else "OS"
                    idx_dict = idx_seg + "_" + eye
                    print(idx_dict)
                    # only regenerate the graph if it does not exist yet or if the force flag is set
                    if not self.check_existing_heterograph(idx_dict) or self.force:
                        self.generate_region_graph(seg_file, vvg_file, idx_dict)

                # break of outer loop
                if found_match:
                    break


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
            try:
                image = np.array(image)[:,:,0]
            except IndexError:
                image = np.array(image)
            image = transform.resize(image, seg.shape, order = 0, preserve_range = True)

            #image = image.astype(np.uint8)
        # background is 1
        ## region properties don't make sense with extremely small regions, also they probably come from failed segmentation
        region_labels = measure.label(morphology.remove_small_holes(seg, area_threshold=5, connectivity=1).astype("uint8"), background=1)


        if self.debug:
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(region_labels)
            
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


        if self.faz_node:
            # faz_region is the region with label at 600,600
            faz_region_label = region_labels[600,600]
            idx_y = 600
            while faz_region_label == 0:
                idx_y += 1
                faz_region_label = region_labels[idx_y,600]

            # find region in props with largest area
            #pos = df["area"].idxmax()
            #print(pos)
#
            #print(faz_region_label)
            # create a dataframe with only the faz region
            df_faz_node = df.iloc[faz_region_label-1]

            # create a dataframe with all the other regions
            df = df.drop(faz_region_label-1)

            ## adjust labels of the regions
            df.index = np.arange(len(df))


        try:
            edge_index_df, edge_index_region_vessel_df, faz_region_df, faz_vessel_df  = self.generate_edge_index(region_labels, vvg_file)
        except json.decoder.JSONDecodeError:
            print("JSONDecodeError")
            return None

        if not self.debug:

            if self.faz_node:
                region_node_file = os.path.join(self.void_graph_save_path, str(idx_dict)+ "_region_nodes_faz_node" + ".csv")
                region_edges_file = os.path.join(self.void_graph_save_path, str(idx_dict) + "_region_edges_faz_node"+".csv")
                region_vessel_edges_file = os.path.join(self.hetero_edges_save_path, str(idx_dict) + "_region_vessel_edges_faz_node"+".csv")

                faz_node_file = os.path.join(self.faz_node_save_path, str(idx_dict) + "_faz_node_faz_node"+".csv")
                faz_region_edges_file = os.path.join(self.faz_region_edges_save_path, str(idx_dict) + "_faz_region_edges_faz_node"+".csv")
                faz_region_vessel_edges_file = os.path.join(self.faz_vessel_edges_save_path, str(idx_dict) + "_faz_region_vessel_edges_faz_node"+".csv")

                df_faz_node.to_csv(faz_node_file, sep = ";")
                faz_region_df.to_csv(faz_region_edges_file, sep = ";")
                faz_vessel_df.to_csv(faz_region_vessel_edges_file, sep = ";")
            
            else:
                region_node_file = os.path.join(self.void_graph_save_path, str(idx_dict)+ "_region_nodes" + ".csv")
                region_edges_file = os.path.join(self.void_graph_save_path, str(idx_dict) + "_region_edges"+".csv")
                region_vessel_edges_file = os.path.join(self.hetero_edges_save_path, str(idx_dict) + "_region_vessel_edges"+".csv")


            df.to_csv(region_node_file, sep = ";")
            edge_index_df.to_csv(region_edges_file, sep = ";")
            edge_index_region_vessel_df.to_csv(region_vessel_edges_file, sep = ";")
        else:
            # plot the nodes
            ax.scatter(df["centroid-1"], df["centroid-0"], s = 8, c= "orange")
            # plot the connections between the nodes
            for id, edge in edge_index_df.iterrows():
                #print(edge["node1id"], edge["node2id"])
                ax.plot([df["centroid-1"][edge["node1id"]], df["centroid-1"][edge["node2id"]]], [df["centroid-0"][edge["node1id"]], df["centroid-0"][edge["node2id"]]], c = "black", linewidth = 0.5)

            if self.faz_node:
                ax.scatter(df_faz_node["centroid-1"], df_faz_node["centroid-0"], s = 12, c= "red")

                faz_centroid_1 = df_faz_node["centroid-1"]
                faz_centroid_0 = df_faz_node["centroid-0"]

                # print edges to faz region, node1id is redundant because it is always the faz region
                for id, edge in faz_region_df.iterrows():
                    ax.plot([faz_centroid_1, df["centroid-1"][edge["node2id"]]], [faz_centroid_0, df["centroid-0"][edge["node2id"]]], c = "red", linewidth = 0.5)

            plt.show()
            plt.close()


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

        if self.faz_node:
            # faz_region is the region with label at 600,600
            faz_region = region_labels[600,600]
            idx_y = 600
            while faz_region == 0:
                idx_y += 1
                faz_region = region_labels[idx_y,600]

            faz_region_neighbor_set = set()
            faz_vessel_neighbor_set = set()

            # extract all the edges that are connected to the faz region from the region_neighbor_set and the vessel_to_region_neighbor_set
            # also remove the edges from the other sets
            new_region_neighbor_set = set()
            for edge in region_neighbor_set:
                if edge[0] == faz_region-1 or edge[1] == faz_region-1:

                    new_edge = (edge[0], edge[1]) if edge[0] == faz_region-1 else (edge[1], edge[0])
                    faz_region_neighbor_set.add(new_edge)
                else:
                    new_region_neighbor_set.add(edge)
            region_neighbor_set = new_region_neighbor_set

            new_vessel_to_region_neighbor_set = set()
            for edge in vessel_to_region_neighbor_set:
                if edge[1] == faz_region-1:
                    faz_vessel_neighbor_set.add(edge)
                else:
                    new_vessel_to_region_neighbor_set.add(edge)
            vessel_to_region_neighbor_set = new_vessel_to_region_neighbor_set

            # create a dataframe for the edge list where the first column is an ID column and the second and third column are the indices of the nodes connected by the edge

            faz_region_df = pd.DataFrame(faz_region_neighbor_set, columns=["node1id", "node2id"])
            faz_vessel_df = pd.DataFrame(faz_vessel_neighbor_set, columns=["node1id", "node2id"])

            # adjust the indices of the edges in the faz_region_df and the faz_vessel_df
            # ids larger than the faz region are shifted by -1

            faz_region_df["node1id"] = faz_region_df["node1id"].apply(lambda x: x-1 if x > faz_region-1 else x)
            faz_region_df["node2id"] = faz_region_df["node2id"].apply(lambda x: x-1 if x > faz_region-1 else x)

            # vessel to region only regions and not vessels are shifted by -1
            faz_vessel_df["node2id"] = faz_vessel_df["node2id"].apply(lambda x: x-1 if x > faz_region-1 else x)


            region_region_df = pd.DataFrame(region_neighbor_set, columns=["node1id", "node2id"])
            vessel_region_df = pd.DataFrame(vessel_to_region_neighbor_set, columns=["node1id", "node2id"])

            # adjust the indices of the edges in the region_region_df and the vessel_region_df
            # ids larger than the faz region are shifted by -1

            region_region_df["node1id"] = region_region_df["node1id"].apply(lambda x: x-1 if x > faz_region-1 else x)
            region_region_df["node2id"] = region_region_df["node2id"].apply(lambda x: x-1 if x > faz_region-1 else x)

            # vessel to region only regions and not vessels are shifted by -1
            vessel_region_df["node2id"] = vessel_region_df["node2id"].apply(lambda x: x-1 if x > faz_region-1 else x)
        

        else:
            faz_region_df = None
            faz_vessel_df = None

            region_region_df = pd.DataFrame(region_neighbor_set, columns=["node1id", "node2id"])
            vessel_region_df = pd.DataFrame(vessel_to_region_neighbor_set, columns=["node1id", "node2id"])
            

        print(len(vessel_to_region_neighbor_set))
        print(success, fail)


        # create a dataframe for the edge list where the first column is an ID column and the second and third column are the indices of the nodes connected by the edge
        return region_region_df, vessel_region_df, faz_region_df, faz_vessel_df


    def generate_skel_labels(self, vvg_file, seg_shape):



        vvg_df_edges, vvg_df_nodes = vvg_loader.vvg_to_df(os.path.join(self.vvg_path, vvg_file))
        cl_arr = vvg_tools.vvg_df_to_centerline_array(vvg_df_edges,vvg_df_nodes, seg_shape) # debug = self.debug

        #if self.debug:
        #    fig, ax = plt.subplots(figsize=(10, 10))
        #    ax.imshow(cl_arr)
        #    plt.show()
        #    plt.close()

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
