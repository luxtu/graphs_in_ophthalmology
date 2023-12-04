# data split generation

import os
import pandas as pd
import numpy as np



label_file = "/media/data/alex_johannes/octa_data/Cairo/labels.csv"
label_data = pd.read_csv(label_file)

# get the number of samples per class
class_names = label_data["Group"].unique()
class_names = label_data["Group"].value_counts()

# split the data frame by class


# get the total number of samples
total_samples = class_names.sum()
print(total_samples)

print(class_names)
# create 6 splits, so divide by 6
class_names = class_names / 6
print(class_names)

# turn class names into dict
class_names_dict = class_names.to_dict()
print(class_names_dict)

# sample 6 splits, such that the number of samples per class is the same
splits = []

# 3 sets with 214 and 3 sets with 215

splits_sizes = [214, 214, 214, 215, 215, 215]
for i in range(6):
    splits.append(pd.DataFrame())
    group_count = {"Healthy": 0, "DM": 0, "Early NPDR": 0, "Late NPDR": 0, "PDR": 0}
    for group in group_count.keys():
        while group_count[group] <= class_names_dict[group] -1.5:
            # sample and item with the proper group from the dataframe
            try:
                sample = label_data[label_data["Group"] == group].sample(n=1)
            except ValueError:
                break
            same_id_samples = label_data[label_data["Subject"] == sample["Subject"].values[0]]
            # get the group of every sample with the same id
            same_id_samples_groups = same_id_samples["Group"].values

            for id_group in same_id_samples_groups:
                group_count[id_group] += 1
            # add the samples to the dataframe
            splits[i] = pd.concat([splits[i], same_id_samples])
            # remove the samples from the original dataframe
            label_data = label_data.drop(same_id_samples.index)
    print(group_count)
    # number of samples in the split
    print(splits[i].shape[0])

    print("Split done")

print(label_data)

# distribute the remaining samples to the splits that have the least samples
for index, row in label_data.iterrows():
    # get the split with the least samples
    min_split = min(splits, key=lambda x: x.shape[0])
    # add the sample to the dataframe
    min_split = pd.concat([min_split, row.to_frame().transpose()])
    # remove the sample from the original dataframe
    label_data = label_data.drop(index)



# check if there are any samples with the same id in different splits, raise an error if so
for i in range(len(splits)):
    for j in range(i+1, len(splits)):
        if splits[i]["Subject"].isin(splits[j]["Subject"]).any():
            raise ValueError("Same id in different splits")




# for each split, save print the number of samples per class
for i in range(len(splits)):
    print("Split " + str(i))
    print(splits[i]["Group"].value_counts())

    # save the splits to csv files
    pd.DataFrame.to_csv(splits[i], "/media/data/alex_johannes/octa_data/Cairo/splits/split_" + str(i) + ".csv", index=False)