import os
import numpy as np
from PIL import Image
import pandas as pd
import re
from torch import from_numpy


class CNNImageLoader():

    octa_dr_dict = {"Healthy": 0, "DM": 0, "PDR": 1, "Early NPDR": 2, "Late NPDR": 2}

    def __init__(self, path, label_file, image_size, transform=None):
        self.path = path
        self.label_path = label_file
        self.image_size = image_size
        self.labels = self.read_labels()
        self.images, self.image_names = self.read_images()
        self.transform = transform


    def read_images(self):
        images = []
        image_names = []
        for filename in os.listdir(self.path):
            # load all png files, and adjust their size
            if filename.endswith(".png"):
                img = Image.open(os.path.join(self.path, filename)).convert('RGB')
                
                idx = re.findall(r'\d+', str(filename))[0]
                idx = str(idx).zfill(4)
                if "_OD" in filename:
                    eye = "OD"
                else:
                    eye = "OS"
                idx_name = idx + "_" + eye
                try:
                    self.labels[idx_name]
                except KeyError:
                    print("No label for image: ", idx_name)
                    continue

                # convert to string with leading zeros, 4 digits
                image_names.append(idx_name)
                images.append(img)

        return images, image_names


    def read_labels(self):
        label_data = pd.read_csv(self.label_path)

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

