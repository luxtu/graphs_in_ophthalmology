# %%
# import the necessary modules
from utils import explain_inference_utils
import torch
from models import graph_classifier
from torch_geometric.loader import DataLoader

from torch_geometric.explain import Explainer,  CaptumExplainer
from explainability import torch_geom_explanation, explanation_in_raw_data, baseline_lookup

# %%
# loading the state dict and model config
check_point_folder = "../data/checkpoints_94d26db_final_27022024"  #"../data/checkpoints_94d26db_split2_retrain" #test_checkpoints" # test_checkpoints
run_id = "mtm0lor9"  #"z6af0tyz" #"mtm0lor9" # "90ym9iu4" #"pxeo8tv2" #"3zhbi3es" #"zng1r2l7" # "j69dzrn5" # "zng1r2l7" # "14d0c0l2" # ""kxmjgi6k" # "5ughh17h" # "14d0c0l2" #"ig9zvma3" # "14d0c0l2"
state_dict, model_config = explain_inference_utils.load_state_dict_and_model_config(check_point_folder, run_id)
split = model_config["split"]


# %%
# get the number of parameters
param_sum, ct = explain_inference_utils.get_param_count(state_dict)
print(f"Number of parameters: {param_sum}")
print(f"Number of uninitialized parameters: {ct}")



# %%
# initialize the model
node_types = ["graph_1", "graph_2"]
out_channels = 3
model = explain_inference_utils.create_model(model_config, node_types, out_channels)
split = model_config["split"]
## delete all keys and values that contain "faz"
if not model_config["parameters"]["faz_node"] or (not model_config["parameters"]["aggr_faz"] and not model_config["parameters"]["hetero_conns"] and not model_config["parameters"]["start_rep"]):
    print("Deleting FAZ")
    state_dict = {k: v for k, v in state_dict.items() if "faz" not in k}
    faz_node_bool = False
else:
    faz_node_bool = True

# %%
# load the data
train_dataset, val_dataset, test_dataset, features_label_dict = explain_inference_utils.load_private_datasets(split, faz_node_bool)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_dataset.to(device)
val_dataset.to(device)
test_dataset.to(device)

# create data loaders for training and test set
train_loader = DataLoader(train_dataset, batch_size = 8, shuffle=True, drop_last = True)
val_loader = DataLoader(val_dataset, batch_size = 64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size = 1, shuffle=False)


# %% 
# load the state dict into the model and initialize the classifier
init_loss = torch.nn.CrossEntropyLoss()
clf = graph_classifier.graphClassifierHetero_94d26db(model, init_loss, lr = 0.01, weight_decay = 0.001, smooth_label_loss = True) 
# load the state dict
loss, y_prob_train, y_true_train, _  = clf.train(train_loader) # necessary to initialize the model
clf.model.load_state_dict(state_dict)


# %%
# set up explainer
explain_type = "IntegratedGradients"
gnnexp_loader = DataLoader(test_dataset, batch_size = 1, shuffle=False)
lookup_tree = baseline_lookup.Baseline_Lookup([train_dataset], node_types, device = device)
save_path = "../data/explain_94d26db_2905_zero/" + run_id
baseline_type = "zero"


# %%
# iterate over the test set and generate explanations
for idx, data in enumerate(gnnexp_loader):
    data = data.to(device)
    data_label = data.graph_id[0]

    # skip healthy samples
    if data.y[0] == 0:
        print(f"Skipping healthy sample {data_label}")
        continue
    print(f"Generating explanations for {data_label}")
    print(baseline_type)
    baselines = lookup_tree.get_baselines(baseline_type=baseline_type, data= data)

    # delete faz node if it is not used in the model
    if not faz_node_bool:
        print("Deleting FAZ information")
        # iterate through the keys and delete the ones that contain "faz" and delete them
        for key in list(data.keys):
            if "faz" in key:
                del data[key]



    explainer = Explainer(
                        model= clf.model,
                        algorithm= CaptumExplainer(explain_type, baselines= baselines),
                        explanation_type='phenomenon',
                        node_mask_type='attributes', 
                        edge_mask_type=None,
                        model_config=dict(
                            mode='multiclass_classification',
                            task_level='graph',
                            return_type='raw',
                        )
                    )

    prediction = explainer.get_prediction(data.x_dict, data.edge_index_dict, batch_dict = data.batch_dict)
    target = explainer.get_target(prediction)
    print(f"Target: {target}, (Predicted Class)")
    print(f"True Label: {data.y[0]}")

    explanation = explainer(data.x_dict, data.edge_index_dict, target =target, batch_dict = data.batch_dict, grads = False)
    

    only_positive = True

    # feature importance using the 20 most important nodes
    explained_gradient = 20
    importance_threshold_path = f'{save_path}/feature_importance/feature_importance_top_{explained_gradient}_{data_label}_{explain_type}_{baseline_type}_{run_id}_quantile_attributes_new_not_each_type.png'
    torch_geom_explanation.visualize_feature_importance(explanation,data,  importance_threshold_path, features_label_dict, explained_gradient= explained_gradient, only_positive = only_positive, with_boxplot = True, num_features=5, each_type= False)

    # feature importance using all nodes
    importance_path = f'{save_path}/feature_importance/feature_importance_{data_label}_{explain_type}_{baseline_type}_{run_id}_no_threshold_quantile_attributes_new_not_each_type.png'
    torch_geom_explanation.visualize_feature_importance(explanation,data, importance_path, features_label_dict, explained_gradient= None, only_positive = only_positive, with_boxplot = True , num_features= 8, each_type= False)

    # plot the full graph
    #graph_2D.HeteroGraphPlotter2D().plot_graph_2D_faz(data, edges= True, path = f"{save_path}/fullgraph/fullgraph_{data_label}_{explain_type}.png")

    # creates a histogram of the node importance
    torch_geom_explanation.visualize_node_importance_histogram(explanation, f"{save_path}/histogram/node_hist_noabs_{data_label}_{explain_type}_{baseline_type}_{run_id}_quantile_attributes_new.png", faz_node= faz_node_bool, abs = False)


    # generate overlay image
    data_type = "DCP"
    segmentation_path = f"../data/{data_type}_seg"
    json_path = f"../data/{data_type}_json"
    image_path = f"../data/{data_type}_images"
    label_names = ["healthy", "NPDR", "PDR"]


    raw_data_explainer = explanation_in_raw_data.RawDataExplainer(raw_image_path= image_path, segmentation_path= segmentation_path, vvg_path= json_path)
    overlay_path = f"{save_path}/overlay/overlay_positive_subgraph_all_{data_label}_NoHeatmap_{baseline_type}_{run_id}_quantile_new.png"
    raw_data_explainer.create_explanation_image(explanation, data, data_label, path=overlay_path, label_names=label_names, target = target, heatmap= True, explained_gradient= explained_gradient, only_positive = only_positive, points = False, intensity_value = 0.1)
