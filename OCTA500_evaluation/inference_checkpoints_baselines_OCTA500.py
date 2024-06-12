# %%
# import the necessary modules
import os
import sys
import numpy as np
import json
import copy
import pickle
import pandas as pd

# change the path to the root folder
sys.path.append("..")
os.chdir("..")

from loader import hetero_graph_loader
from utils import dataprep_utils, explain_inference_utils


# %%
# load the data

octa_dr_dict = {"Healthy": 0, "DM": 0, "PDR": 2, "Early NPDR": 1, "Late NPDR": 1}
label_names = ["Healthy/DM", "NPDR", "PDR"]

data_type = "DCP"

vessel_graph_path = f"../data/{data_type}_vessel_graph"
label_file = "../data/labels.csv"
void_graph_path = f"../data/{data_type}_void_graph"
hetero_edges_path = f"../data/{data_type}_heter_edges"


label_file = "../data/splits"

mode_cv = "cv"
cv_pickle = f"../data/{data_type}_{mode_cv}_dataset.pkl"
cv_dataset = hetero_graph_loader.HeteroGraphLoaderTorch(
    vessel_graph_path,
    void_graph_path,
    hetero_edges_path,
    mode=mode_cv,
    label_file=label_file,
    line_graph_1=True,
    class_dict=octa_dr_dict,
    pickle_file=cv_pickle,
)


mode_final_test = "final_test"
final_test_pickle = f"../data/{data_type}_{mode_final_test}_dataset.pkl"
final_test_dataset = hetero_graph_loader.HeteroGraphLoaderTorch(
    vessel_graph_path,
    void_graph_path,
    hetero_edges_path,
    mode=mode_final_test,
    label_file=label_file,
    line_graph_1=True,
    class_dict=octa_dr_dict,
    pickle_file=final_test_pickle,
)

mode = "all"
vessel_graph_path_octa500 = "../../OCTA_500_RELEVANT_fix/vessel_graph"
void_graph_path_octa500 = "../../OCTA_500_RELEVANT_fix/void_graph"
hetero_edges_path_octa500 = "../../OCTA_500_RELEVANT_fix/hetero_edges"
vessel_graph_path_octa500 = "../../OCTA_500_RELEVANT_fix/graph_new"
label_file_octa500 = "../../OCTA_500_RELEVANT_fix/labels_only_DR_Healthy.csv"

octa_500_dict = {"NORMAL": 0, "DR": 1}
label_names_OCTA500 = ["NORMAL", "DR"]

octa500_pickle = (
    f"../../OCTA_500_RELEVANT_fix/OCTA500_{data_type}_{mode}_dataset_no_faz.pkl"
)
octa500_data = hetero_graph_loader.HeteroGraphLoaderTorch(
    graph_path_1=vessel_graph_path_octa500,
    graph_path_2=void_graph_path_octa500,
    hetero_edges_path_12=hetero_edges_path_octa500,
    mode=mode,
    label_file=label_file_octa500,
    line_graph_1=True,
    class_dict=octa_500_dict,
    pickle_file=octa500_pickle,  # f"../{data_type}_{mode_train}_dataset_faz.pkl" # f"../{data_type}_{mode_train}_dataset_faz.pkl"
)


num_classes = len(np.unique(list(octa_dr_dict.values())))

# %%

test_dataset = copy.deepcopy(octa500_data)
dataprep_utils.hetero_graph_imputation(test_dataset)

node_types = ["graph_1", "graph_2"]
dataprep_utils.add_node_features(test_dataset, node_types)
node_mean_tensors, node_std_tensors = (
    dataprep_utils.hetero_graph_standardization_params(test_dataset, robust=False)
)
dataprep_utils.hetero_graph_standardization(
    test_dataset, node_mean_tensors, node_std_tensors
)
with open("feature_configs/feature_name_dict.json", "r") as file:
    label_dict_full = json.load(file)
    # features_label_dict = json.load(file)
features_label_dict = copy.deepcopy(label_dict_full)


x_test_octa500, y_test_octa500 = dataprep_utils.aggreate_graph(
    test_dataset, features_label_dict
)


# %%

with open("feature_configs/feature_name_dict.json", "r") as file:
    label_dict_full = json.load(file)
    # features_label_dict = json.load(file)
features_label_dict = copy.deepcopy(label_dict_full)

_, _, octa500test_dataset_work = dataprep_utils.adjust_data_for_split(
    cv_dataset, octa500_data, 3
)
x_test_octa500, y_test_octa500 = dataprep_utils.aggreate_graph(
    octa500test_dataset_work, features_label_dict
)


# %%
svm_clf_dict = {1: None, 2: None, 3: None, 4: None, 5: None}
rf_clf_dict = {1: None, 2: None, 3: None, 4: None, 5: None}

for clf_name in svm_clf_dict.keys():
    with open(
        f"../data/best_checkpoints_svm/best_clf_svm_sumAGG_allFeatures_{clf_name}.pkl",
        "rb",
    ) as file:
        svm_clf_dict[clf_name] = pickle.load(file)

    with open(
        f"../data/best_checkpoints_rf/best_clf_rf_sumAGG_allFeatures_{clf_name}.pkl",
        "rb",
    ) as file:
        rf_clf_dict[clf_name] = pickle.load(file)


# %%

rf_df_report = pd.DataFrame()
rf_df_metrics = pd.DataFrame()
rf_df_best_params = pd.DataFrame()

svm_df_report = pd.DataFrame()
svm_df_metrics = pd.DataFrame()
svm_df_best_params = pd.DataFrame()


for key_idx in svm_clf_dict.keys():
    report, metrics = explain_inference_utils.evaluate_performance_baselines(
        svm_clf_dict[key_idx],
        x_test_octa500,
        y_test_octa500,
        OCTA500=True,
        label_names=label_names_OCTA500,
    )

    svm_df_metrics = pd.concat(
        [svm_df_metrics, pd.DataFrame(metrics, index=[clf_name])]
    )
    svm_df_report = pd.concat([svm_df_report, pd.DataFrame(report).transpose()])

    report, metrics = explain_inference_utils.evaluate_performance_baselines(
        rf_clf_dict[key_idx],
        x_test_octa500,
        y_test_octa500,
        OCTA500=True,
        label_names=label_names_OCTA500,
    )

    rf_df_metrics = pd.concat([rf_df_metrics, pd.DataFrame(metrics, index=[clf_name])])
    rf_df_report = pd.concat([rf_df_report, pd.DataFrame(report).transpose()])


print(svm_df_metrics)
print(svm_df_report)

print(rf_df_metrics)
print(rf_df_report)


# %%
