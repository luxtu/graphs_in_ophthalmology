# %%
# import the necessary modules
import copy
import json
import torch
from torch_geometric.explain import CaptumExplainer, Explainer, PGExplainer, GNNExplainer
from torch_geometric.loader import DataLoader
import pandas as pd

from explainability import (
    explanation_in_raw_data_homogeneous,
    sample_feature_importance_homogeneous,
)
from models import graph_classifier
from utils import explain_inference_utils

# %%
# loading the state dict and model config
check_point_folder = "checkpoints/checkpoints_homogeneous_biomarker_vessel_graph"

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
# load the demo data from the pickle file
with open("demo_data/demo_data_vessel.pkl", "rb") as file:
     test_dataset = torch.load(
        file
    )
     
# load the feature dictionary
with open("feature_configs/feature_name_dict_demo.json", "rb") as file:
    features_label_dict = json.load(file)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %%
# load the biomarker data
fractal_bvd = pd.read_csv("demo_data/demo_fractal_BVD.csv", index_col="ID")

# set Fractal value of the biomarkers to the y value of the graph
for data in test_dataset:
    data.y = torch.tensor([fractal_bvd.loc[data.graph_id, target_biomarker]], dtype = torch.float32, device=device)


for data in test_dataset:
    data.to(device)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


# %%
# load the state dict into the model and initialize the classifier
init_loss = torch.nn.L1Loss()
clf = graph_classifier.graphRegressorSimple(
    model, init_loss, lr=0.01, weight_decay=0.001
)
# load the state dict
loss, y_prob_train, y_true_train = clf.train(
    test_loader
)  # necessary to initialize the model
clf.model.load_state_dict(state_dict)


# %%
# set up explainer
# "GuidedBP"
explain_type = "IntegratedGradients"
gnnexp_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

save_path = f"../data/explain_homogeneous_biomarker_{graph_type}_graph/" + run_id


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


# %%
# iterate over the test set and generate explanations
for idx, data in enumerate(gnnexp_loader):
    data = data.to(device)
    data_label = data.graph_id[0]
    print(f"Generating explanations for {data_label}")

    algorithm = CaptumExplainer(explain_type)
    #algorithm = GNNExplainer()
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


    sample_explainer = sample_feature_importance_homogeneous.SampleFeatureImportance(
        explanation, data, features_label_dict, graph_type
    )

    # creates a histogram of the node importance
    histogram_path = f"{save_path}/histogram/node_hist_noabs_{data_label}_{explain_type}_{run_id}_quantile_attributes_new.png"

    sample_explainer.visualize_node_importance_histogram(histogram_path, abs=False)

    importance_path = f"{save_path}/feature_importance/feature_importance_{data_label}_{explain_type}_{run_id}_no_threshold.png"
    sample_explainer.visualize_feature_importance(
        path=importance_path,
    )

    # generate overlay image with top 20 nodes
    segmentation_path = "demo_data/seg"
    json_path = "demo_data/json"
    image_path = "demo_data/images"
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
    )
