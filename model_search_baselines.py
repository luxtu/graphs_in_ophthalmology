# %%
# import the necessary modules
import copy
import json

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from utils import dataprep_utils, train_utils

# %%
label_names = ["Healthy/DM", "NPDR", "PDR"]
faz_node_bool = True

# load the data
cv_dataset, final_test_dataset = dataprep_utils.load_private_pickled_data()

# %%

# rf_grid_file = 'hyperparam_grids/rf_grid_original.json'
rf_grid_file = "hyperparam_grids/rf_grid.json"
with open(rf_grid_file, "r") as file:
    rf_param_grid = json.load(file)

svm_grid_file = "hyperparam_grids/svc_grid_original.json"
with open(svm_grid_file, "r") as file:
    svm_param_grid = json.load(file)

rf_clf = RandomForestClassifier
svm_clf = SVC

rf_best_param_dict = {}
rf_best_score_dict = {}
rf_clf_dict = {}

svm_best_param_dict = {}
svm_best_score_dict = {}
svm_clf_dict = {}


for split in [1, 2, 3, 4, 5]:  # ,2,3,4,5]
    print(f"Split {split}")

    train_dataset, val_dataset, test_dataset = dataprep_utils.adjust_data_for_split(
        cv_dataset,
        final_test_dataset,
        split,
        faz=True,
        use_full_cv=False,
        robust=True,
        min_max=False,
    )

    # eliminate highly correlated features
    dataset_list = [train_dataset, val_dataset, test_dataset]
    dataset_list, features_label_dict = (
        dataprep_utils.delete_highly_correlated_features(dataset_list, faz_node_bool)
    )
    train_dataset, val_dataset, test_dataset = dataset_list

    train_dataset_work = copy.deepcopy(train_dataset.hetero_graph_list)
    val_dataset_work = copy.deepcopy(val_dataset.hetero_graph_list)
    test_dataset_work = copy.deepcopy(test_dataset.hetero_graph_list)

    x_train, y_train = dataprep_utils.aggreate_graph(
        train_dataset_work, features_label_dict, faz=True, agg_type="sum"
    )
    x_val, y_val = dataprep_utils.aggreate_graph(
        val_dataset_work, features_label_dict, faz=True, agg_type="sum"
    )
    x_test, y_test = dataprep_utils.aggreate_graph(
        test_dataset_work, features_label_dict, faz=True, agg_type="sum"
    )

    best_params, best_score, clf = train_utils.hyper_param_search_baseline(
        rf_param_grid, rf_clf, x_train, y_train, x_val, y_val
    )
    rf_best_param_dict[split] = best_params
    rf_best_score_dict[split] = best_score
    rf_clf_dict[split] = clf

    print(f"Best RF params: {best_params}")

    train_utils.evaluate_baseline_model(
        clf, x_test, y_test, label_names, title=f"Split {split}"
    )

    best_params, best_score, clf = train_utils.hyper_param_search_baseline(
        svm_param_grid, svm_clf, x_train, y_train, x_val, y_val
    )
    svm_best_param_dict[split] = best_params
    svm_best_score_dict[split] = best_score
    svm_clf_dict[split] = clf

    print(f"Best SVM params: {best_params}")

    train_utils.evaluate_baseline_model(
        clf, x_test, y_test, label_names, title=f"Split {split}"
    )

# %%
