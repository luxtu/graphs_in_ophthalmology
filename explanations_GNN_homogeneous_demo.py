# %%
# import the necessary modules
import json
import torch
from torch_geometric.explain import CaptumExplainer, Explainer, GNNExplainer
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score

from explainability import (
    explanation_in_raw_data_homogeneous,
    sample_feature_importance_homogeneous,
)
from models import graph_classifier
from utils import explain_inference_utils


# %%
# loading the state dict and model config
check_point_folder = "checkpoints/checkpoints_homogeneous_staging_vessel_graph"


model_dict_vessel = {"best_mean": "3dr9fd83", "best_max": "zx4a6vs1", "best_add": "aoxg6zaf"}
run_id = model_dict_vessel["best_mean"]


state_dict, model_config = explain_inference_utils.load_state_dict_and_model_config(
    check_point_folder, run_id
)
split = model_config["split"]


# %%
# initialize the model

out_channels = 3
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

for data in test_dataset:
    data.to(device)

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# %%
# load the state dict into the model and initialize the classifier
init_loss = torch.nn.CrossEntropyLoss()
clf = graph_classifier.graphClassifierSimple(
    model, init_loss, lr=0.01, weight_decay=0.001, smooth_label_loss=True
)
# load the state dict
loss, y_prob_train, y_true_train = clf.train(
    test_loader
)  # necessary to initialize the model
clf.model.load_state_dict(state_dict)


# %%
# evaluate the model on the test data set
y_prob_test, y_true_test = clf.predict(test_loader)
y_p_softmax_test = torch.nn.functional.softmax(torch.tensor(y_prob_test), dim = 1).detach().numpy()

# not for single demo sample
#mean_auc = roc_auc_score(
#    y_true=y_true_test,
#    y_score = y_p_softmax_test,
#    multi_class="ovr",
#    average="macro",
#)
#print(f"Mean AUC Test: {mean_auc}")


# %%
# set up explainer
explain_type = "IntegratedGradients"
gnnexp_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

save_path = "../data/explain_homogeneous_staging/" + run_id


# %%
# iterate over the test set and generate explanations
for idx, data in enumerate(gnnexp_loader):
    data = data.to(device)
    data_label = data.graph_id[0]

    # skip healthy samples
    #if data.y[0] == 0:
    #    print(f"Skipping healthy sample {data_label}")
    #    continue
    print(f"Generating explanations for {data_label}")

    explainer = Explainer(
        model=clf.model,
        algorithm=CaptumExplainer(explain_type), #explain_type
        explanation_type="phenomenon",
        node_mask_type="attributes",
        edge_mask_type=None,
        model_config=dict(
            mode="multiclass_classification",
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
    print(f"Prediction: {torch.nn.functional.softmax(zero_prediction, dim=1)}")

    prediction = explainer.get_prediction(
        data.x, data.edge_index, batch=data.batch
    )
    print(f"Prediction: {prediction}")
    print(f"Prediction: {torch.nn.functional.softmax(prediction, dim=1)}")
    target = explainer.get_target(prediction)
    print(f"Target: {target}, (Predicted Class)")
    print(f"True Label: {data.y[0]}")

    explanation = explainer(
        data.x, data.edge_index, target=target, batch=data.batch
    )
    only_positive = True

    # feature importance using the 20 most important nodes
    explained_gradient = None

    sample_explainer = sample_feature_importance_homogeneous.SampleFeatureImportance(
        explanation, data, features_label_dict, "vessel"
    )

    # creates a histogram of the node importance
    histogram_path = f"{save_path}/histogram/node_hist_noabs_{data_label}_{explain_type}_{run_id}_quantile_attributes_new.png"
    sample_explainer.visualize_node_importance_histogram(histogram_path, abs=False)

    importance_path = f"{save_path}/feature_importance/feature_importance_{data_label}_{explain_type}_{run_id}_no_threshold.png"
    sample_explainer.visualize_feature_importance(
        path=importance_path,
    )

    # generate overlay images with all nodes
    segmentation_path = "demo_data/seg"
    json_path = "demo_data/json"
    image_path = "demo_data/images"
    label_names = ["healthy", "NPDR", "PDR"]

    raw_data_explainer = explanation_in_raw_data_homogeneous.RawDataExplainer(
        raw_image_path=image_path,
        segmentation_path=segmentation_path,
        vvg_path=json_path,
        graph_type="vessel",
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


# %%
