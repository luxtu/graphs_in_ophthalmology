# import all the necessary packages
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from skimage import measure, morphology
from skimage.morphology import skeletonize

from utils import vvg_tools
from loader import vvg_loader

class VoidGraphGenerator:

    def __init__(self, seg_path, save_path):

        self.seg_path = seg_path
        self.save_path = save_path

    def save_region_graphs(self):
        region_graphs = {}
        files = os.listdir(self.seg_path)
        for file in sorted(files):
            idx = int(file[-9:-4])
            region_graphs[idx] = self.generate_region_graph(file)

            print(file)
    
    def generate_region_graph(self, file):


        idx = int(file[-9:-4])
        #read image
        seg = Image.open(os.path.join(self.seg_path, file))
        seg = np.array(seg)
        seg = seg.astype(np.uint8)

        # background is 1
        ## region properties don't make sense with extremely small regions

        region_labels = measure.label(morphology.remove_small_holes(seg, area_threshold=5, connectivity=1).astype("uint8"), background=1)

        # remove all the labels that contain less than 9 pixels
        #region_labels = morphology.remove_small_objects(region_labels, min_size=9, connectivity=1)


        props = measure.regionprops_table(region_labels, properties=('centroid',
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


        df.to_csv(os.path.join(self.save_path, str(idx)+ "_nodes"+".csv"), sep = ";")

        edge_index_df = self.generate_edge_index(region_labels, seg)
        edge_index_df.to_csv(os.path.join(self.save_path, str(idx) + "_edges"+".csv"), sep = ";")



    def generate_edge_index(self, region_labels, seg):

        # skeletonization may remove the connections to the image border.
        # this results in a connection between regions that was not there before


        # skeletonize the image
        skel_labels = self.generate_skel_labels(seg, skel_type = "border")
        skel_neighbor_list = self.generate_skel_neighbor_list(skel_labels)


        matching_regions = self.find_matching_regions(skel_labels, region_labels)

        # fail means that there is an edge in the skeletonized image that is not in the original image
        # could be because there are regions in the skeletonized image that do not exist in the real image

        neighbor_regions_list_seg = []
        success = 0
        fail = 0
        for tup in skel_neighbor_list:
            try:
                neighbor_regions_list_seg.append([matching_regions[tup[0]]-1, matching_regions[tup[1]]-1])
                success += 1
            except KeyError:
                fail += 1
        print(success, fail)

        # create a dataframe for the edge list where the first column is an ID column and the second and third column are the indices of the nodes connected by the edge

        return pd.DataFrame(neighbor_regions_list_seg, columns=["node1id", "node2id"])

    def generate_skel_labels(self, seg, skel_type = "border"):

        if skel_type == "border":
            seg_work = seg.copy()
            # make all border pixels 1
            seg_work = seg_work.astype(np.uint8)
            seg_work[0,:] = 255
            seg_work[-1,:] = 255
            seg_work[:,0] = 255
            seg_work[:,-1] = 255

            skeleton = skeletonize(seg_work)
            skeleton = skeleton.astype(np.uint8)
            skel_labels_border = measure.label(skeleton, background=1, connectivity=1)

            del seg_work

            return skel_labels_border

        elif skel_type == "crop":
            skeleton = skeletonize(seg)
            skeleton = skeleton.astype(np.uint8)

            crop_val = 4
            # cropping is essential here, ske
            skel_crop = skeleton[crop_val:-crop_val,crop_val:-crop_val]
            skel_labels = measure.label(skel_crop, background=1, connectivity=1)
            skel_labels_crop = np.zeros_like(skeleton, dtype= skel_labels.dtype)
            skel_labels_crop[crop_val:-crop_val,crop_val:-crop_val] = skel_labels

            return skel_labels_crop

        else: 
            raise ValueError("skel_type must be either 'border' or 'crop'")

    def generate_skel_neighbor_list(self, skel_labels):

        # this might not generate all the edges
        # might be some degenerate cases

        #Initialize a set to store neighboring regions
        neighbor_regions = set()

        # Iterate through rows from top to bottom (sweep line)
        for row in range(skel_labels.shape[0]):
            # Identify the regions intersected by the sweep line at this row

            intersected_regions = skel_labels[row, 1:][np.diff(skel_labels[row, :]) != 0]
            intersected_regions = intersected_regions[intersected_regions != 0]

            intersected_regions = intersected_regions.tolist()
            intersected_regions.insert(0, skel_labels[row, 0])


            # Add newly intersected regions to the priority queue
            for i in range(len(intersected_regions)-1):
            
                r1 = min(intersected_regions[i], intersected_regions[i+1])
                r2 = max(intersected_regions[i], intersected_regions[i+1])
                neighbor_regions.add((r1, r2))


        # Iterate through columns from top to bottom (sweep line)
        for col in range(skel_labels.shape[1]):

            # Identify the regions intersected by the sweep line at this col
            intersected_regions = np.array(skel_labels[1:,col])[np.diff(skel_labels[:,col]) != 0]
            intersected_regions = intersected_regions[intersected_regions != 0]

            intersected_regions = intersected_regions.tolist()
            intersected_regions.insert(0, skel_labels[0,col])

            # Add newly intersected regions to the priority queue
            for i in range(len(intersected_regions)-1):

                r1 = min(intersected_regions[i], intersected_regions[i+1])
                r2 = max(intersected_regions[i], intersected_regions[i+1])
                neighbor_regions.add((r1, r2))

        return list(neighbor_regions)


    def find_matching_regions(self, skel_labeled, seg_labeled):

        # finds for every region in the original image the corresponding region in the skeletonized image
        
        # if no regions was found in the skeletonized image, the region is not added to the dictionary
        # actually this should not happen 

        matching_regions = {}

        for lab_val in np.unique(seg_labeled):
            skel_vals, counts = np.unique(skel_labeled[np.where(seg_labeled == lab_val)], return_counts=True) 

            count_sort_ind = np.argsort(-counts)
            skel_vals = skel_vals[count_sort_ind]

            if skel_vals[0] ==0:
                try:
                    matching_regions[skel_vals[1]] = lab_val
                except IndexError:
                    pass
            else:
                matching_regions[skel_vals[0]] = lab_val

        return matching_regions





class VoidGraphFromVVG:

    def __init__(self, seg_path, vvg_path, save_path):

        self.seg_path = seg_path
        self.vvg_path = vvg_path
        self.save_path = save_path


    def save_region_graphs(self):
        region_graphs = {}
        seg_files = os.listdir(self.seg_path)
        vvg_files = os.listdir(self.vvg_path)

        # extract all files from vvg files that end with .json or .json.gz
        vvg_files = [file for file in vvg_files if file.endswith(".json") or file.endswith(".json.gz")]
        # extract all files from seg files that end with .png
        seg_files = [file for file in seg_files if file.endswith(".png")]

        vvg_files = sorted(vvg_files)
        seg_files = sorted(seg_files)

        #iterate through matching pairs of files
        for i, seg_file in enumerate(sorted(seg_files)):

            # extcract all the numbers from the file name
            pattern = r'\d+'
            # Use re.findall() to extract all sequences of digits from the string
            idx_seg = re.findall(pattern, seg_file)[0]
            idx_vvg = re.findall(pattern, vvg_files[i])[0]

            # check if the numbers are the same
            assert idx_seg == idx_vvg

            # check if both filenames contain either "_OD" or "_OS"
            assert ("_OD" in seg_file and "_OD" in vvg_files[i]) or ("_OS" in seg_file and "_OS" in vvg_files[i])

            if "_OD" in seg_file:
                eye = "OD"
            else:
                eye = "OS"

            idx_dict = idx_seg + "_" + eye

            region_graphs[idx_dict] = self.generate_region_graph(seg_file, vvg_files[i], idx_dict)

            print(seg_file)



    def generate_region_graph(self, seg_file, vvg_file, idx_dict):

        seg = Image.open(os.path.join(self.seg_path, seg_file))
        seg = np.array(seg)
        seg = seg.astype(np.uint8)

        # background is 1
        ## region properties don't make sense with extremely small regions

        region_labels = measure.label(morphology.remove_small_holes(seg, area_threshold=5, connectivity=1).astype("uint8"), background=1)

        # remove all the labels that contain less than 9 pixels
        #region_labels = morphology.remove_small_objects(region_labels, min_size=9, connectivity=1)


        props = measure.regionprops_table(region_labels, properties=('centroid',
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


        df.to_csv(os.path.join(self.save_path, str(idx_dict)+ "_region_nodes" + ".csv"), sep = ";")

        edge_index_df, edge_index_region_vessel_df = self.generate_edge_index(region_labels, vvg_file)
        edge_index_df.to_csv(os.path.join(self.save_path, str(idx_dict) + "_region_edges"+".csv"), sep = ";")
        edge_index_region_vessel_df.to_csv(os.path.join(self.save_path, str(idx_dict) + "_region_vessel_edges"+".csv"), sep = ";")


    def generate_edge_index(self, region_labels, vvg_file):


        # generate the centerline image from the vvg file
        skel_labels = self.generate_skel_labels(vvg_file, region_labels.shape)

        plt.imshow(skel_labels)
        plt.show()

        matching_regions = self.find_matching_regions(skel_labels, region_labels)

        # fail means that there is an edge in the skeletonized image that is not in the original image
        # could be because there are regions in the skeletonized image that do not exist in the real image

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
                try:
                    n1 = skel_labels[pix_0-1, pix_1]
                except IndexError:
                    n1 = None
                try:
                    n2 = skel_labels[pix_0+1, pix_1]
                except IndexError:
                    n2 = None
                try:
                    n3 = skel_labels[pix_0, pix_1-1]
                except IndexError:
                    n3 = None
                try:
                    n4 = skel_labels[pix_0, pix_1+1]
                except IndexError:
                    n4 = None

                for neighb in [n1,n2,n3,n4]:
                    if neighb == None:
                        pass
                    elif neighb  == skel_labels[pix_0, pix_1]:
                        pass
                    else:
                        try:     
                            vessel_to_region_neighbor_set.add((id, matching_regions[neighb]-1))
                            success += 1
                        except KeyError:
                            fail += 1

                # create all combinations of the 4 neighbors

                for neighb1 in [n1,n2,n3,n4]:
                    for neighb2 in [n1,n2,n3,n4]:
                        if neighb1 == None or neighb2 == None:
                            pass
                        elif neighb1 == neighb2:
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

        for lab_val in np.unique(seg_labeled):
            skel_vals, counts = np.unique(skel_labeled[np.where(seg_labeled == lab_val)], return_counts=True) 

            count_sort_ind = np.argsort(-counts)
            skel_vals = skel_vals[count_sort_ind]

            if skel_vals[0] ==0:
                try:
                    matching_regions[skel_vals[1]] = lab_val
                    matched_regions += 1
                except IndexError:
                    failed_regions += 1
                    pass
            else:
                matching_regions[skel_vals[0]] = lab_val
                matched_regions += 1

        return matching_regions


