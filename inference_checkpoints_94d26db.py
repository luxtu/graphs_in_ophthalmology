# %%
# import the necessary modules
import os

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


label_names = ["Healthy/DM", "NPDR", "PDR"]
faz_node_bool = True

# load the data
cv_dataset, final_test_dataset = explain_inference_utils.load_private_pickled_data()


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

    # set the correct split
    split = model_config["split"]
    print(f"Best model {run_id} split: {split}")
    train_dataset, val_dataset, test_dataset = dataprep_utils.adjust_data_for_split(
        cv_dataset,
        final_test_dataset,
        split,
        faz=True,
        use_full_cv=False,
        min_max=False,
    )

    # eliminate highly correlated features
    dataset_list = [train_dataset, val_dataset, test_dataset]
    dataset_list, features_label_dict = (
        explain_inference_utils.delete_highly_correlated_features(
            dataset_list, faz_node_bool
        )
    )
    train_dataset, val_dataset, test_dataset = dataset_list

    # put the datasets on the gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset.to(device)
    val_dataset.to(device)
    test_dataset.to(device)

    init_loss = torch.nn.CrossEntropyLoss()
    clf = graph_classifier.graphClassifierHetero_94d26db(model, init_loss)

    # run 100 epochs of training and check the time it takes
    # train_loader = DataLoader(train_dataset, batch_size = 16, shuffle=True)
    # evaluate_training_time(clf, train_loader, epochs = 100)

    test_loader_set = DataLoader(test_dataset[:4], batch_size=4, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # must run on an input to set the parameters
    # load the state dict
    clf.predict(test_loader_set)
    clf.model.load_state_dict(state_dict)

    # evaluate the performance
    test_report, test_metrics = explain_inference_utils.evaluate_performance(clf, test_loader)
    val_report, val_metrics = explain_inference_utils.evaluate_performance(clf, val_loader)

    df_metrics = pd.concat([df_metrics, pd.DataFrame(test_metrics, index=[split])])
    df_report = pd.concat([df_report, pd.DataFrame(test_report).transpose()])

# df_metrics.to_csv('GNN_CV_metrics_94d26db_v1.csv')
# df_report.to_csv('GNN_CV_report_94d26db_v1.csv')

print(df_metrics)
print(df_report)
# %%
