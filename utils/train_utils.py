from itertools import cycle, product

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    RocCurveDisplay,
    accuracy_score,
    auc,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import LabelBinarizer


def evaluate_model(classifier, val_loader, test_loader=None, label_names=None):
    """Evaluate model on validation and test set

    Parameters
    ----------
    classifier : Classifier that contains the model to evaluate
    val_loader : DataLoader Validation set
    test_loader : DataLoader Test set

    Returns
    -------
    results : dict
        Dictionary containing the results of the evaluation
    """

    # evaluation on validation set
    y_prob_val, y_true_val = classifier.predict(val_loader)
    y_pred_val = y_prob_val.argmax(axis=1)
    val_acc = accuracy_score(y_true_val, y_pred_val)
    val_bal_acc = balanced_accuracy_score(y_true_val, y_pred_val)
    val_kappa = cohen_kappa_score(y_true_val, y_pred_val, weights="quadratic")

    res_dict = {
        "val_acc": val_acc,
        "val_bal_acc": val_bal_acc,
        "val_kappa": val_kappa,
    }
    # evaluation on test set
    if test_loader is not None:
        y_prob_test, y_true_test = classifier.predict(test_loader)
        y_pred_test = y_prob_test.argmax(axis=1)
        test_acc = accuracy_score(y_true_test, y_pred_test)
        test_bal_acc = balanced_accuracy_score(y_true_test, y_pred_test)
        test_kappa = cohen_kappa_score(y_true_test, y_pred_test, weights="quadratic")
        test_res_dict = {
            "test_acc": test_acc,
            "test_bal_acc": test_bal_acc,
            "test_kappa": test_kappa,
        }
        res_dict.update(test_res_dict)

    # dimension of the output
    num_classes = y_prob_val.shape[1]
    if label_names is None:
        label_names = [str(i) for i in range(num_classes)]
    y_p_softmax = (
        torch.nn.functional.softmax(torch.tensor(y_prob_val), dim=1).detach().numpy()
    )
    if num_classes == 2:
        val_mean_auc = roc_auc_score(
            y_true=y_true_val,
            y_score=y_p_softmax[:, 1],
            multi_class="ovr",
            average="macro",
        )
    else:
        val_mean_auc = roc_auc_score(
            y_true=y_true_val,
            y_score=y_p_softmax,
            multi_class="ovr",
            average="macro",
        )
        label_binarizer = LabelBinarizer().fit(y_true_val)
        y_onehot_val = label_binarizer.transform(y_true_val)
        # get auc for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(num_classes):
            fpr[i], tpr[i], _ = roc_curve(y_onehot_val[:, i], y_p_softmax[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            res_dict[f"roc_auc_{label_names[i]}"] = roc_auc[i]

    res_dict["val_mean_auc"] = val_mean_auc

    return res_dict, y_p_softmax, y_true_val


def plot_confusion_matrix(groundTruth, predicted, label_list, ax):
    cf_mat = confusion_matrix(groundTruth, predicted)
    ax.matshow(cf_mat)
    for (i, j), z in np.ndenumerate(cf_mat):
        ax.text(j, i, z, ha="center", va="center")

    ax.set_xticks(range(len(label_list)))
    ax.set_yticks(range(len(label_list)))
    ax.set_xticklabels(label_list)
    ax.set_yticklabels(label_list)
    ax.xaxis.set_label_position("top")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True Label")


def evaluate_baseline_model(model, x, y, label_names, title=None, num_classes=3):
    y_pred = model.predict(x)
    y_prob = model.predict_proba(x)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    plot_confusion_matrix(
        y, y_pred, label_names, ax[0]
    )  # , ["Healthy/DM","PDR", "NPDR"]"Early NPDR", "Late NPDR" , 3, 4 # ["Healthy/DM", "PDR", "NPDR"]
    colors = cycle(["aqua", "darkorange", "cornflowerblue"])

    label_binarizer = LabelBinarizer().fit(y)
    y_onehot = label_binarizer.transform(y)

    if num_classes == 2:
        RocCurveDisplay.from_predictions(
            y,
            y_prob[:, 1],
            name="ROC curve",
            ax=ax[1],
            color="red",
        )
    else:
        for class_id, color in zip(range(num_classes), colors):
            RocCurveDisplay.from_predictions(
                y_onehot[:, class_id],
                y_prob[:, class_id],
                name=f"{class_id} vs the rest",
                color=color,
                ax=ax[1],
                plot_chance_level=(class_id == 2),
            )
    # add title
    if title is not None:
        fig.suptitle(title)
    plt.show()

    if num_classes == 2:
        mean_auc = roc_auc_score(
            y_true=y,
            y_score=y_prob[:, 1],
            multi_class="ovo",
            average="macro",
        )
    else:
        mean_auc = roc_auc_score(
            y_true=y,
            y_score=y_prob,
            multi_class="ovo",
            average="macro",
        )

    # print metrics
    print(classification_report(y, y_pred, target_names=label_names))

    print(f"Accuracy Score: {accuracy_score(y, y_pred)}")
    print(f"Balanced Accuracy Score: {balanced_accuracy_score(y, y_pred)}")
    print(f"Mean AUC: {mean_auc}")
    print(
        f"Cohen Kappa Score (quadratic weights): {cohen_kappa_score(y, y_pred, weights='quadratic')}"
    )


def hyper_param_search_baseline(grid, classifier, x_train, y_train, x_val, y_val):
    best_score = 0
    best_params = None
    for param_set in product(*grid.values()):
        params = dict(zip(grid.keys(), param_set))
        clf = classifier(**params, random_state=42)
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_val)
        score = balanced_accuracy_score(y_val, y_pred)
        # score = cohen_kappa_score(y_val, y_pred, weights="quadratic")
        if score > best_score:
            best_score = score
            best_params = params
            best_clf = clf

    y_pred = best_clf.predict(x_val)

    final_accuracy = accuracy_score(y_val, y_pred)
    final_balanced_accuracy = balanced_accuracy_score(y_val, y_pred)
    final_kappa = cohen_kappa_score(
        y_val, y_pred, weights="quadratic"
    )  # , weights="quadratic"

    print(f"Validation Accuracy: {final_accuracy}")
    print(f"Validation Balanced Accuracy: {final_balanced_accuracy}")
    print(f"Validation Kappa: {final_kappa}")
    # return the best classifier
    return best_params, best_score, best_clf
