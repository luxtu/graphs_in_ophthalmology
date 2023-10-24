import os
import numpy as np
from PIL import Image
import pandas as pd
import re
from torch import from_numpy
from sklearn.model_selection import train_test_split

class CNNImageLoader():

    octa_dr_dict = {"Healthy": 0, "DM": 0, "PDR": 2, "Early NPDR": 1, "Late NPDR": 1}

    def __init__(self, path, label_file, mode, transform=None):
        self.path = path
        self.label_path = label_file
        self.mode = mode
        self.labels = self.read_labels()
        self.images, self.image_names = self.read_images()
        self.transform = transform


    def update_class(self, new_class_dict):

        self.octa_dr_dict = new_class_dict



    def read_images(self):
        images = []
        image_names = []
        for filename in os.listdir(self.path):
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
                try:
                    self.labels[idx_name]
                except KeyError:
                    #print("No label for image: ", idx_name)
                    continue

                
                img = Image.open(os.path.join(self.path, filename)).convert('RGB')
                image_names.append(idx_name)
                images.append(img)

        return images, image_names


    def read_labels(self):
        label_data = pd.read_csv(self.label_path)

        train, temp = train_test_split(label_data, test_size=0.3, random_state=42, stratify=label_data["Group"])
        test, val = train_test_split(temp, test_size=0.5, random_state=42, stratify=temp["Group"])
        del temp

        _, debug = train_test_split(test, test_size=0.10, random_state=42, stratify=test["Group"])

        if self.mode == "train":
            label_data = train            
        elif self.mode == "test":
            label_data = test
        elif self.mode == "val":
            label_data = val
        elif self.mode == "debug":
            label_data = debug

        label_dict = {}

        for i, row in label_data.iterrows():
            index = row["Subject"]
            # convert to string with leading zeros, 4 digits
            index = str(index).zfill(4)
            eye = row["Eye"]
            label_dict[index + "_" + eye] = row["Group"]
        
        return label_dict


    def __len__(self):
        return len(self.images)


    def __getitem__(self, idx):
        image = self.images[idx]
        image_name = self.image_names[idx]
        label = self.octa_dr_dict[self.labels[image_name]]

        if self.transform:
            image = self.transform(image)

        #if self.target_transform:
        #    label = self.target_transform(label)
        return image, label

