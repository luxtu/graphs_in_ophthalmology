from torchvision.models import resnet18, ResNet18_Weights, efficientnet_b3, EfficientNet_B3_Weights
from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim
from loader import image_loader
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from evaluation import evaluation
from utils import prep

import wandb

# Define sweep config
sweep_configuration = {
    "method": "bayes",
    "name": "sweep",
    "metric": {"goal": "maximize", "name": "best_mean_auc"},
    "parameters": {
        "model" : {"values": ["resnet", "effnet"]},
        "batch_size": {"values": [16, 32, 64]},
        "epochs": {"values": [200]},
        "lr": {"max": 0.05, "min": 0.005}, # learning rate to high does not work
        "weight_decay": {"max": 0.01, "min": 0.00001},
        "dropout": {"values": [0.1, 0.3, 0.4]}, # 0.2,  more droput looks better
        "class_weights": {"values": ["unbalanced", "balanced"]}, # "balanced", 
        "dataset": {"values": ["DCP"]}, #, "DCP"
        "epochs": {"values": [100,200]},
        "optimizer" : {"values": ["adam", "sgd"]},
        "pretrained" : {"values": [False]},
    },
}


sweep_id = wandb.sweep(sweep=sweep_configuration, project="graph_pathology")
#torch.cuda.empty_cache()


# Define your data transformations (modify as needed)
transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


transform_train = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.Lambda(lambda img: transforms.functional.rotate(img, angle=90 * torch.randint(4, (1,)).item())),
    transforms.ToTensor(),
    # Finally the values are first rescaled to [0.0, 1.0]
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

#images = "/media/data/alex_johannes/octa_data/Cairo/SCP_images"
images = "../dcp_images"
#labels = "/media/data/alex_johannes/octa_data/Cairo/labels.csv"
labels = "../labels.csv"


train_dataset = image_loader.CNNImageLoader(path = images,
                                         label_file = labels,
                                         transform=transform_train,
                                         mode = "train"
                                         )
test_dataset = image_loader.CNNImageLoader(path = images,
                                         label_file = labels,
                                         transform=transform_test,
                                         mode = "test"
                                         )
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_labels = [label for _, label in train_dataset]
class_weights = prep.get_class_weights(train_labels, verbose = True)
class_weights = torch.tensor(class_weights, device=device)
class_weights = class_weights ** 0.5 

num_classes = class_weights.shape[0]


num_classes = class_weights.shape[0]  # Change this to the number of classes in your dataset
print("num_classes: ", num_classes)

train_dataset.update_class({"Healthy": 0, "DM": 0, "PDR": 2, "Early NPDR": 1, "Late NPDR": 1})
test_dataset.update_class({"Healthy": 0, "DM": 0, "PDR": 2, "Early NPDR": 1, "Late NPDR": 1})


def xavier_init(model):
    if isinstance(model, nn.Module):
        if isinstance(model, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(model.weight)
            if model.bias is not None:
                nn.init.constant_(model.bias, 0)



def main():
    run = wandb.init()


    if wandb.config.model == "resnet":
        # Load the pre-trained ResNet18 model
        #weights = ResNet18_Weights.DEFAULT
        model = resnet18() #weights=weights)
        model.apply(xavier_init)

    elif wandb.config.model == "effnet":
        #weights = EfficientNet_B3_Weights.DEFAULT
        model = efficientnet_b3() #weights=weights)
        model.apply(xavier_init)

    ## Train all layers
    for param in model.parameters():
        param.requires_grad = True

    

    # Modify the last fully connected layer to match the number of classes in your task

    if wandb.config.model == "resnet":
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif wandb.config.model == "effnet":
        model.classifier._modules['1'] = nn.Linear(model.classifier._modules['1'].in_features, num_classes)
    balanced_loss = torch.nn.CrossEntropyLoss(class_weights)
    unbalanced_loss = torch.nn.CrossEntropyLoss()

    loss_dict = {"balanced": balanced_loss, "unbalanced": unbalanced_loss}

    # Define your loss function and optimizer
    criterion = loss_dict[wandb.config.class_weights]

    optimizer_dict = {"adam": optim.Adam(model.parameters(), lr=wandb.config.lr, weight_decay=wandb.config.weight_decay),
                        "sgd": optim.SGD(model.parameters(), lr=wandb.config.lr, momentum=0.9, weight_decay=wandb.config.weight_decay)}

    optimizer = optimizer_dict[wandb.config.optimizer]

    # Training loop
    model.to(device)


    batch_size = wandb.config.batch_size  
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=1)

    for epoch in range(1, wandb.config.epochs +1):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        # get the test loss and accuracy

        test_loss = 0.0
        model.eval()  # prep model for evaluation

        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item() * data.size(0)

        # calculate avg test loss
        test_loss = test_loss / len(test_loader.dataset)
        #print("reaches")
        train_accuracy, train_balanced_accuracy, train_kappa, train_mean_auc  = evaluation.evaluate_cnn(model, train_loader, num_classes)
        test_accuracy, test_balanced_accuracy, test_kappa, test_mean_auc = evaluation.evaluate_cnn(model, test_loader, num_classes)

        wandb.log({"loss": loss,
                    "train_acc": train_accuracy, 
                    "test_acc": test_accuracy, 
                    "test_bal_acc": test_balanced_accuracy, 
                    "train_kappa": train_kappa, 
                    "test_kappa": test_kappa, 
                    "train_mean_auc": train_mean_auc, 
                    "test_mean_auc": test_mean_auc})



# Save the fine-tuned model if needed
#torch.save(resnet.state_dict(), 'fine_tuned_resnet18.pth')

wandb.agent(sweep_id, function=main, count=100)