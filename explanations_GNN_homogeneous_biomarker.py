# %%
# import the necessary modules
import copy
import os
import json
import torch
from torch_geometric.explain import CaptumExplainer, Explainer, PGExplainer, GNNExplainer
from torch_geometric.loader import DataLoader
import pandas as pd

from loader import hetero_graph_loader
from explainability import (
    explanation_in_raw_data_homogeneous,
    sample_feature_importance_homogeneous,
)
from models import graph_classifier
from utils import explain_inference_utils, dataprep_utils, to_homogeneous_graph

# %%
# loading the state dict and model config
check_point_folder = "../data/checkpoints_homogeneous_biomarker_vessel_graph"
#run_id = "20oc6okd" #"tvib49x7" # "r619bjft"  
target_biomarker = "Fractal"
graph_type = "vessel"
aggr_type = "best_mean"

if graph_type == "vessel":
    model_dict_Fractal = {"best_mean": "t9o7s4qg", "best_max": "rkojjd8n", "best_add": "cpdztid0"} # _vessel
    model_dict_BVD = {"best_mean": "n4psptec", "best_max": "okix2w9w", "best_add": "xoiuickc"}
elif graph_type == "region":
    model_dict_Fractal = {"best_mean": "sac7mjva", "best_max": "wg8yogjn", "best_add": "jko779d7"}
    model_dict_BVD = {}
else:
    raise ValueError

if target_biomarker == "Fractal":
    run_id = model_dict_Fractal[aggr_type]
elif target_biomarker == "BVD":
    run_id = model_dict_BVD[aggr_type]
else:
    raise ValueError

state_dict, model_config = explain_inference_utils.load_state_dict_and_model_config(
    check_point_folder, run_id
)
split = model_config["split"]


# %%
# get the number of parameters
param_sum, ct = explain_inference_utils.get_param_count(state_dict)
print(f"Number of parameters: {param_sum}")
print(f"Number of uninitialized parameters: {ct}")


# %%
# initialize the model

out_channels = 1
model = explain_inference_utils.create_homogeneous_model(model_config, out_channels)
split = model_config["split"]

# %%
# load the data
data_type = "DCP"

octa_dr_dict = {"Healthy": 0, "DM": 0, "PDR": 2, "Early NPDR": 1, "Late NPDR": 1}
label_names = ["Healthy/DM", "NPDR", "PDR"]

vessel_graph_path = f"../data/{data_type}_vessel_graph"
label_file = "../data/splits"

mode_cv = "cv"
mode_final_test = "final_test"


void_graph_path = f"../data/{data_type}_void_graph"
hetero_edges_path = f"../data/{data_type}_heter_edges"

cv_pickle = f"../data/{data_type}_{mode_cv}_dataset.pkl"
cv_dataset = hetero_graph_loader.HeteroGraphLoaderTorch(graph_path_1=vessel_graph_path,
                                                    graph_path_2=void_graph_path,
                                                    hetero_edges_path_12=hetero_edges_path,
                                                    mode = mode_cv,
                                                    label_file = label_file, 
                                                    line_graph_1 =True, 
                                                    class_dict = octa_dr_dict,
                                                    pickle_file = cv_pickle #f"../{data_type}_{mode_train}_dataset_faz.pkl" # f"../{data_type}_{mode_train}_dataset_faz.pkl"
                                                    )

final_test_pickle = f"../data/{data_type}_{mode_final_test}_dataset.pkl"
final_test_dataset = hetero_graph_loader.HeteroGraphLoaderTorch(graph_path_1=vessel_graph_path,
                                                        graph_path_2=void_graph_path,
                                                        hetero_edges_path_12=hetero_edges_path,
                                                        mode = mode_final_test,
                                                        label_file = label_file, 
                                                        line_graph_1 =True, 
                                                        class_dict = octa_dr_dict,
                                                        pickle_file = final_test_pickle #f"../{data_type}_{mode_train}_dataset_faz.pkl" # f"../{data_type}_{mode_train}_dataset_faz.pkl"
                                                        )

train_dataset, val_dataset, test_dataset = dataprep_utils.adjust_data_for_split(cv_dataset, final_test_dataset, split, faz = False)



with open("feature_configs/feature_name_dict.json", "rb") as file:
    label_dict_full = json.load(file)
    #features_label_dict = json.load(file)
features_label_dict = copy.deepcopy(label_dict_full)

eliminate_features = {"graph_1":["num_voxels", "maxRadiusAvg", "hasNodeAtSampleBorder", "maxRadiusStd"], 
                      "graph_2":["centroid_weighted-0", "centroid_weighted-1", "feret_diameter_max", "orientation"]}


# get positions of features to eliminate and remove them from the feature label dict and the graphs
for key in eliminate_features.keys():
    for feat in eliminate_features[key]:
        idx = features_label_dict[key].index(feat)
        features_label_dict[key].remove(feat)
        for data in train_dataset:
            data[key].x = torch.cat([data[key].x[:, :idx], data[key].x[:, idx+1:]], dim = 1)
        for data in val_dataset:
            data[key].x = torch.cat([data[key].x[:, :idx], data[key].x[:, idx+1:]], dim = 1)
        for data in test_dataset:
            data[key].x = torch.cat([data[key].x[:, :idx], data[key].x[:, idx+1:]], dim = 1)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# create homogeneous graphs out of the heterogenous graphs
if graph_type == "vessel":
    train_dataset = to_homogeneous_graph.to_vessel_graph(train_dataset)
    val_dataset = to_homogeneous_graph.to_vessel_graph(val_dataset)
    test_dataset = to_homogeneous_graph.to_vessel_graph(test_dataset)
elif graph_type == "region":
    train_dataset = to_homogeneous_graph.to_void_graph(train_dataset)
    val_dataset = to_homogeneous_graph.to_void_graph(val_dataset)
    test_dataset = to_homogeneous_graph.to_void_graph(test_dataset)
else:
    raise ValueError

for data in train_dataset:
    data.to(device)
for data in val_dataset:
    data.to(device)
for data in test_dataset:
    data.to(device)

# %%
# load the biomarker data
fractal_bvd = pd.read_csv("../data/fractal_BVD.csv", index_col="ID")

# set Fractal value of the biomarkers to the y value of the graph
for data in train_dataset:
    data.y = torch.tensor([fractal_bvd.loc[data.graph_id, target_biomarker]], dtype = torch.float32, device=device)
for data in val_dataset:
    data.y = torch.tensor([fractal_bvd.loc[data.graph_id, target_biomarker]], dtype = torch.float32, device=device)
for data in test_dataset:
    data.y = torch.tensor([fractal_bvd.loc[data.graph_id, target_biomarker]], dtype = torch.float32, device=device)


# create data loaders for training and test set
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


# %%
# load the state dict into the model and initialize the classifier
init_loss = torch.nn.L1Loss()
clf = graph_classifier.graphRegressorSimple(
    model, init_loss, lr=0.01, weight_decay=0.001
)
# load the state dict
loss, y_prob_train, y_true_train = clf.train(
    train_loader
)  # necessary to initialize the model
clf.model.load_state_dict(state_dict)


# %%
# set up explainer
# "GuidedBP"
explain_type = "PGExplainer"
gnnexp_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

save_path = "../data/explain_homogeneous_biomarker_void_graph/" + run_id


# %%
# iterate over the test set and save inference results to a csv
ids = []
# get the ids 
for data in test_dataset:
    ids.append(data.graph_id)

# create a test dataset with 0 values
test_dataset_zero = copy.deepcopy(test_dataset)
for data in test_dataset_zero:
    data.x = torch.zeros_like(data.x)

# create a zero graph loader
test_loader_zero = DataLoader(test_dataset_zero, batch_size=1, shuffle=False)

# get the predictions for the zero graph
y_pred_zero_graph, y_true = clf.predict(test_loader_zero)
y_pred_zero_graph = y_pred_zero_graph.squeeze()
y_pred, y_true = clf.predict(test_loader)
y_pred = y_pred.squeeze()

print(y_true.shape)
print(y_pred.shape)


# create a dataframe and save it to a csv
results = pd.DataFrame({"ID": ids, "y_pred_zero_graph": y_pred_zero_graph, "y_pred": y_pred, "y_true": y_true})
# check if the folder exists and create it if it does not
if not os.path.exists(save_path):
    os.makedirs(save_path)
results.to_csv(f"{save_path}/inference_results_test_set_{run_id}_{target_biomarker}.csv", index=False)


# %%
# iterate over the test set and generate explanations
for idx, data in enumerate(gnnexp_loader):
    data = data.to(device)
    data_label = data.graph_id[0]
    print(f"Generating explanations for {data_label}")

    #algorithm = CaptumExplainer(explain_type)
    algorithm = GNNExplainer()
    explainer = Explainer(
        model=clf.model,
        algorithm=algorithm, #explain_type
        explanation_type="phenomenon",
        node_mask_type="attributes",
        edge_mask_type=None,
        model_config=dict(
            mode="regression",
            task_level="graph",
            return_type="raw",
        ),
    )

    # print the number of nodes and edges
    print(f"Number of nodes: {data.num_nodes}")
    
    # get prediction for a 0 version of the data
    zero_prediction = explainer.get_prediction(
        data.x*.0, data.edge_index, batch=data.batch
    )
    print(f"Prediction: {zero_prediction}")

    prediction = explainer.get_prediction(
        data.x, data.edge_index, batch=data.batch
    )
    print(f"Prediction: {prediction}")
    target = explainer.get_target(prediction)
    print(f"Target: {target}, (Predicted Class)")
    print(f"True Value: {data.y[0]}")

    explanation = explainer(
        data.x, data.edge_index, target=target, batch=data.batch
    )
    only_positive = True


    # feature importance using the 20 most important nodes
    explained_gradient = None

    sample_explainer = sample_feature_importance_homogeneous.SampleFeatureImportance(
        explanation, data, features_label_dict, "region"
    )

    # creates a histogram of the node importance
    histogram_path = f"{save_path}/histogram/node_hist_noabs_{data_label}_{explain_type}_{run_id}_quantile_attributes_new.png"

    sample_explainer.visualize_node_importance_histogram(histogram_path, abs=False)

    importance_path = f"{save_path}/feature_importance/feature_importance_{data_label}_{explain_type}_{run_id}_no_threshold.png"
    sample_explainer.visualize_feature_importance(
        path=importance_path,
        explained_gradient=None,
        only_positive=False,
        with_boxplot=False,
        num_features=5,
    )
    #continue

    # generate overlay image with top 20 nodes
    data_type = "DCP"
    segmentation_path = f"../data/{data_type}_seg"
    json_path = f"../data/{data_type}_json"
    image_path = f"../data/{data_type}_images"
    label_names = ["healthy", "NPDR", "PDR"]

    raw_data_explainer = explanation_in_raw_data_homogeneous.RawDataExplainer(
        raw_image_path=image_path,
        segmentation_path=segmentation_path,
        vvg_path=json_path,
        graph_type=graph_type,
    )
    overlay_path = f"{save_path}/overlay/overlay_positive_subgraph_all_{data_label}_NoHeatmap_{run_id}_quantile_new.png"
    raw_data_explainer.create_explanation_image(
        explanation,
        data,
        data_label,
        path=overlay_path,
        label_names=label_names,
        target=target,
        heatmap=True,
        explained_gradient=explained_gradient,
        only_positive=only_positive,
        points=False,
        intensity_value=None,
    )
    if idx == 30:
        break


# %%
# force reload of the explainability module
import importlib
importlib.reload(explanation_in_raw_data_homogeneous)
importlib.reload(sample_feature_importance_homogeneous)
importlib.reload(explain_inference_utils)

# %%
