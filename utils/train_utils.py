from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, roc_auc_score, roc_curve, auc
import torch
from sklearn.preprocessing import LabelBinarizer

def evaluate_model(classifier, val_loader, test_loader = None, label_names = None):
    """ Evaluate model on validation and test set

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
    val_kappa = cohen_kappa_score(y_true_val, y_pred_val, weights= "quadratic")

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
        test_kappa = cohen_kappa_score(y_true_test, y_pred_test, weights= "quadratic")
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
    y_p_softmax = torch.nn.functional.softmax(torch.tensor(y_prob_val), dim = 1).detach().numpy()
    if num_classes == 2:
        val_mean_auc = roc_auc_score(
            y_true=y_true_val,
            y_score = y_p_softmax[:,1],
            multi_class="ovr",
            average="macro",
        )
    else:
        val_mean_auc = roc_auc_score(
            y_true=y_true_val,
            y_score = y_p_softmax,
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
