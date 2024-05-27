from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from itertools import combinations


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


def pca_transform_plot(y_pred_train, y_pred_test, y_t_train, y_t_test, labels = None):


    pca = PCA(n_components=2)
    pca.fit(y_pred_train)
    pca_train = pca.transform(y_pred_train)
    pca_test = pca.transform(y_pred_test)


    fig, ax = plt.subplots(1,2, figsize = (12,5))

    sc = ax[0].scatter(pca_train[:,0],pca_train[:,1], c = y_t_train)


    sc = ax[1].scatter(pca_test[:,0],pca_test[:,1], c = y_t_test)

    # add a legend with custom entries to the plot
    if labels is not None:
        ax[0].legend(sc.legend_elements()[0],labels, title="Classes")
    # avoid colorbar from taking up space on the plot
    fig.colorbar(sc)
    plt.show()

def tsne_transform_plot(y_pred_train, y_pred_test, y_t_train, y_t_test, labels = None):

    tsne = TSNE(n_components=2)

    y_prob_train_tsne = tsne.fit_transform(y_pred_train)
    y_prob_tsne = tsne.fit_transform(y_pred_test)

    fig, ax = plt.subplots(1,2, figsize = (12,5))

    sc = ax[0].scatter(y_prob_train_tsne[:,0],y_prob_train_tsne[:,1], c = y_t_train)
    sc = ax[1].scatter(y_prob_tsne[:,0],y_prob_tsne[:,1], c = y_t_test)
    if labels is not None:
        ax[0].legend(sc.legend_elements()[0],labels, title="Classes")
    fig.colorbar(sc)
    plt.show()

def embedding_plot(y_pred_train, y_pred_test, y_t_train, y_t_test, labels = None):
        
        num = y_pred_train.shape[1]
    
        fig, ax = plt.subplots(num,2, figsize = (10,5*num))

        # create all possible combinations numbers in range(num)
        combs = list(combinations(range(0, num ), 2))

        for k, comb in enumerate(combs):
            i = comb[0]
            j = comb[1]
            ax[k, 0].scatter(y_pred_train[:,i],y_pred_train[:,j], c = y_t_train)
            ax[k, 0].set_xlabel(f"Class {labels[i]}")
            ax[k, 0].set_ylabel(f"Class {labels[j]}")
            sc = ax[k, 1].scatter(y_pred_test[:,i],y_pred_test[:,j], c = y_t_test)
            ax[k, 1].set_xlabel(f"Class {labels[i]}")
            ax[k, 1].set_ylabel(f"Class {labels[j]}")  

            if labels is not None:
                ax[k, 0].legend(sc.legend_elements()[0],labels, title="Classes")

        fig.colorbar(sc)
        plt.show()

