
import numpy as np
import os
import re
from PIL import Image
from skimage import measure, morphology

class SegFeatureExtractor():


    def __init__(self, seg_path):
        self.seg_path = seg_path
        
        self.seg_dict = self.read_seg()

        self.feature_dict = self.extract_features()


    def read_seg(self):
        seg_dict = {}
        for filename in os.listdir(self.seg_path):
            # load all png files, and adjust their size
            if filename.endswith(".png"):
                
                idx = re.findall(r'\d+', str(filename))[0]
                # convert to string with leading zeros, 4 digits
                idx = str(idx).zfill(4)
                if "_OD" in filename:
                    eye = "OD"
                else:
                    eye = "OS"
                idx_name = idx + "_" + eye
                #try:
                #    self.labels[idx_name]
                #except KeyError:
                #    #print("No label for image: ", idx_name)
                #    continue

                
                img = Image.open(os.path.join(self.seg_path, filename))
                img = np.array(img)

                seg_dict[idx_name] = img

        return seg_dict
    

    def extract_features(self):
        feature_dict = {}
        for key, img in self.seg_dict.items():
            feature_dict[key] = self.extract_features_from_seg(img)
        return feature_dict
    

    def extract_features_from_seg(self, img):
        features = []
        img = img.astype(bool)

        vessel_density_features = self.vessel_density_features(img)
        features += vessel_density_features

        return features


    def vessel_density_features(self, img):
        global_mask_ratio = np.sum(img) / img.size

        # get the ratio of the mask in the 4 quadrants
        # top left
        top_left = img[:img.shape[0]//2, :img.shape[1]//2]
        top_left_ratio = np.sum(top_left) / top_left.size
        # top right
        top_right = img[:img.shape[0]//2, img.shape[1]//2:]
        top_right_ratio = np.sum(top_right) / top_right.size
        # bottom left
        bottom_left = img[img.shape[0]//2:, :img.shape[1]//2]
        bottom_left_ratio = np.sum(bottom_left) / bottom_left.size
        # bottom right
        bottom_right = img[img.shape[0]//2:, img.shape[1]//2:]
        bottom_right_ratio = np.sum(bottom_right) / bottom_right.size

        return [global_mask_ratio, top_left_ratio, top_right_ratio, bottom_left_ratio, bottom_right_ratio]

            


    def get_feature_dict(self):
        return self.feature_dict