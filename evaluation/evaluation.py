from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, balanced_accuracy_score
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
import torch
import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from itertools import combinations



def evaluate_cnn(model, dataloader, num_classes):
    import torch
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score
    from sklearn.metrics import roc_auc_score

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    all_labels = []
    all_predictions = []
    all_outputs = []

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predictions = torch.max(outputs, 1)
        all_outputs.extend(outputs.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())
        all_predictions.extend(predictions.detach().cpu().numpy())


    accuracy = accuracy_score(all_labels, all_predictions)
    balanced_accuracy = balanced_accuracy_score(all_labels, all_predictions)
    kappa = cohen_kappa_score(all_labels, all_predictions, weights="quadratic")
    labels_array = np.array(all_labels)
    predictions_array = torch.nn.functional.softmax(torch.tensor(np.array(all_outputs).squeeze()), dim = 1).detach().cpu().numpy()


    if num_classes == 2:
        mean_auc = roc_auc_score(
            y_true=labels_array,
            y_score = predictions_array[:,1],
            multi_class="ovr",
            average="macro",
        )
    else:
        mean_auc = roc_auc_score(
            y_true=labels_array,
            y_score = predictions_array,
            multi_class="ovr",
            average="macro",
        )

    #del inputs, labels, outputs, predictions
    #torch.cuda.empty_cache()

    return accuracy, balanced_accuracy, kappa, mean_auc


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

def roc_auc_multiclass(groundtruth, predicted):
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

    return fpr, tpr, roc_auc


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

    plt.figure(figsize=(8, 6))
    if len(class_labels)==2:
        y = label_binarize(groundtruth, classes= np.arange(predicted.shape[1]))
        fpr, tpr, _ = roc_curve(y, predicted[:,1])

        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'ROC curve {class_labels[0]} vs {class_labels[1]}(AUC = {roc_auc:.2f})')

    else:
        fpr, tpr, roc_auc = roc_auc_multiclass(groundtruth, predicted)



        # Plot ROC curves
        
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

