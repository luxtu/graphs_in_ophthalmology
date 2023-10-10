


def get_class_weights(train_labels, verbose = False):

    from collections import Counter

    # Calculate the label distribution
    label_distribution = Counter(train_labels)



    # If you want to use the label distribution for balanced weights in CrossEntropyLoss
    total_samples = len(train_labels)
    class_weights = [total_samples / (len(label_distribution) * count) for label, count in label_distribution.items()]

    if verbose:
        # Print or use the label distribution as needed
        print("Label distribution in the training set:")
        for label, count in label_distribution.items():
            print(f"Class {label}: {count} samples")

        print("Class weights for balanced CrossEntropyLoss:")
        for label, weight in zip(label_distribution.keys(), class_weights):
            print(f"Class {label}: Weight {weight}")

    return class_weights


def evaluate_cnn(model, dataloader):
    import torch
    from sklearn.metrics import accuracy_score, balanced_accuracy_score

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    all_labels = []
    all_predictions = []

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predictions = torch.max(outputs, 1)
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predictions.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    balanced_accuracy = balanced_accuracy_score(all_labels, all_predictions)

    del inputs, labels, outputs, predictions
    torch.cuda.empty_cache()

    return accuracy, balanced_accuracy


def hetero_graph_imputation(dataset):
    import torch
    for data in dataset:
        # check if the .x and .edge_index contains nan values
        for key, val in data.x_dict.items():
            if torch.any(torch.isnan(val)):
                #print("nan values in x")
                nan_pos = torch.where(torch.isnan(val))
                # imputation with mean value of feature
                val[nan_pos] = 0
            if torch.any(torch.isinf(val)):
                #print("inf values in x")
                inf_pos = torch.where(torch.isinf(val))
                val[inf_pos] = 0


def hetero_graph_normalization_params(train_dataset):

    import torch
    node_tensors = {}
    node_mean_tensors = {}
    node_std_tensors = {}
    for key, val in train_dataset.hetero_graph_list[0].x_dict.items():
        node_tensors[key] = None

    for data in train_dataset:
        for key, val in data.x_dict.items():
            node_tensors[key] = torch.cat([node_tensors[key], val]) if node_tensors[key] is not None else val

    for key, val in node_tensors.items():
        node_mean_tensors[key] = torch.mean(val, dim=0)
        node_std_tensors[key] = torch.std(val, dim=0)

    return node_mean_tensors, node_std_tensors

def hetero_graph_normalization(dataset, node_mean_tensors, node_std_tensors):

    for data in dataset:
        for key, val in data.x_dict.items():
            data.x_dict[key] = (val - node_mean_tensors[key]) / node_std_tensors[key]