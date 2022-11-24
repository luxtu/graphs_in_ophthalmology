from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, balanced_accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
import torch
import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(groundTruth, predicted, label_list):
    cf_mat = confusion_matrix(groundTruth, predicted)  

    plt.figure()
    ax = plt.subplot()
    ax.matshow(cf_mat)
    for (i, j), z in np.ndenumerate(cf_mat):
        ax.text(j, i, z, ha='center', va='center')

    ax.set_xticks(range(len(label_list)))
    ax.set_yticks(range(len(label_list)))
    ax.set_xticklabels(label_list)
    ax.set_yticklabels(label_list)
    ax.xaxis.set_label_position('top') 
    ax.set_xlabel("Classified")
    ax.set_ylabel("True Label")
    plt.show()



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
        predicte_softmax = torch.softmax(torch.tensor(pred).float(), dim = -1)
        res = roc_auc_score(gt, predicte_softmax.detach().numpy(), multi_class='ovr')

    return res


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