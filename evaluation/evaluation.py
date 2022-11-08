from sklearn.metrics import confusion_matrix
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
