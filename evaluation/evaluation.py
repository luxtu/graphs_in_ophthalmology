from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, balanced_accuracy_score
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
import torch
import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle


def plot_confusion_matrix(groundTruth, predicted, label_list, ax):
    cf_mat = confusion_matrix(groundTruth, predicted)  
    ax.matshow(cf_mat)
    for (i, j), z in np.ndenumerate(cf_mat):
        ax.text(j, i, z, ha='center', va='center')

    ax.set_xticks(range(len(label_list)))
    ax.set_yticks(range(len(label_list)))
    ax.set_xticklabels(label_list)
    ax.set_yticklabels(label_list)
    ax.xaxis.set_label_position('top') 
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True Label")



def eval_roc_auc(groundTruth, predicted):

    try:
        gt = groundTruth.detach().numpy()
        pred = predicted.detach().numpy()
    except AttributeError:
        gt = groundTruth
        pred = predicted

    try: 
        res = roc_auc_score(gt, pred, multi_class='ovr')
    except ValueError:
        predicted_softmax = torch.softmax(torch.tensor(pred).float(), dim = -1)
        res = roc_auc_score(gt, predicted_softmax.detach().numpy(), multi_class='ovr')

    return res



def plot_roc_curve(groundtruth, predicted, class_labels=None):
    """
    Plot ROC curves for each class in a multiclass classification problem.

    Parameters:
    - groundtruth: (n_samples,) shaped array containing true labels.
    - predicted: (n_samples, k) shaped array containing predicted probabilities for k classes.
    - class_labels: List of class labels. If not provided, class indices will be used.

    Returns:
    - None (displays the plot)
    """

    # Binarize the groundtruth labels
    cls_int = np.arange(predicted.shape[1])
    y = label_binarize(groundtruth, classes=cls_int)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    n_classes = y.shape[1]

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y[:, i], predicted[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curves
    plt.figure(figsize=(8, 6))
    colors = cycle(['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan'])

    for i, color in zip(range(len(class_labels)), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve (class {class_labels[i]}) (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

    




def plot_loss_acc(loss_l, acc_l):

    loss_l = np.array(loss_l)
    acc_l = np.array(acc_l)

    if loss_l.shape != acc_l.shape:
        raise ValueError("Arrays must have the same shape.")


    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    try:
        comps = loss_l.shape[1]
        for i in range(comps):
            ax1.plot(loss_l[:,i], color = "red")
            ax2.plot(acc_l[:,i], color = "blue")

    except IndexError:
        ax1.plot(loss_l, color = "red")
        ax2.plot(acc_l, color = "blue")

    ax1.set_xlabel('Iterations / No.')
    ax1.set_ylabel('Loss', color='r')
    ax2.set_ylabel('Accuracy / %', color='b')