# %%
# import the necessary modules
import copy
import os
import pickle
import sys

# change the path to the root folder
sys.path.append("..")
os.chdir("..")

import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from models import graph_classifier
from utils import dataprep_utils, explain_inference_utils

# %% load the data
check_point_folder = "../data/checkpoints_94d26db_final_27022024"

# extract the files with .pt extension
check_point_files = [f for f in os.listdir(check_point_folder) if f.endswith(".pt")]

# extract the run id from the file name
# run id are the characters after the last underscore
run_ids = [f.split("_")[-1].split(".")[0] for f in check_point_files]

# find the json file with the same run id
# json file name is the same as the run id
json_files = [f for f in os.listdir(check_point_folder) if f.endswith(".json")]

faz_node_bool = True

# load the data
# cv_dataset, final_test_dataset = dataprep_utils.load_private_pickled_data()

# OCTA datasets
octa500_pickle = "../../OCTA_500_RELEVANT_fix/all_OCTA500_selected_sweep_repeat_v2.pkl"

with open(octa500_pickle, "rb") as file:
    octa500_dataset = pickle.load(file)


test_dataset = copy.deepcopy(octa500_dataset)
dataprep_utils.hetero_graph_imputation(test_dataset)

node_types = ["graph_1", "graph_2", "faz"]
dataprep_utils.add_node_features(test_dataset, node_types)
node_mean_tensors, node_std_tensors = (
    dataprep_utils.hetero_graph_standardization_params(test_dataset, robust=True)
)
dataprep_utils.hetero_graph_standardization(
    test_dataset, node_mean_tensors, node_std_tensors
)

dataset_list = [test_dataset]
dataset_list, features_label_dict = dataprep_utils.delete_highly_correlated_features(
    dataset_list, faz_node_bool
)

test_dataset = dataset_list[0]


df_report = pd.DataFrame()
df_metrics = pd.DataFrame()


# %% load the model and evaluate the performance
for run_id in sorted(run_ids):
    state_dict, model_config = explain_inference_utils.load_state_dict_and_model_config(
        check_point_folder, run_id
    )
    node_types = ["graph_1", "graph_2"]
    # load the model
    out_channels = 3
    model = explain_inference_utils.create_model(model_config, node_types, out_channels)

    split = model_config["split"]
    print(split)
    # _, _, test_dataset = dataprep_utils.adjust_data_for_split(cv_dataset, octa500_dataset, split, faz = True, use_full_cv = False, min_max=False)

    # put the datasets on the gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dataset.to(device)

    init_loss = torch.nn.CrossEntropyLoss()
    clf = graph_classifier.graphClassifierHetero_94d26db(model, init_loss)

    test_loader_set = DataLoader(test_dataset[:4], batch_size=4, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # must run on an input to set the parameters
    # load the state dict
    clf.predict(test_loader_set)
    clf.model.load_state_dict(state_dict)

    # evaluate the performance
    report, metrics = explain_inference_utils.evaluate_performance(
        clf, test_loader, OCTA500=True, label_names=["Healthy", "DR"]
    )

    df_metrics = pd.concat([df_metrics, pd.DataFrame(metrics, index=[split])])
    df_report = pd.concat([df_report, pd.DataFrame(report).transpose()])

    print(metrics)

# df_metrics.to_csv("OCTA500_GNN_CV_metrics_94d26db_v1.csv")
# df_report.to_csv("OCTA500_GNN_CV_report_94d26db_v1.csv")

print(df_metrics)
print(df_report)

# %%
