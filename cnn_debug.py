from torchvision.models import resnet18, ResNet18_Weights, efficientnet_b3, EfficientNet_B3_Weights, efficientnet_b2, EfficientNet_B2_Weights, efficientnet_b0, EfficientNet_B0_Weights
from torchvision import transforms
from torchvision.models._api import WeightsEnum
from torch.hub import load_state_dict_from_url
import torch
import torch.nn as nn
import torch.optim as optim
from loader import image_loader
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from evaluation import evaluation
from utils import prep
from types import SimpleNamespace
import wandb

print("start")

# Define sweep config
sweep_config_dict = {
        "model" : "effnet",
        "batch_size": 16,
        "epochs": 100,
        "lr": 0.0002, # learning rate to high does not work
        "weight_decay": 0.01,
        "class_weights":"unbalanced", # "balanced", 
        "dataset": "DCP", #, "DCP"
        "epochs": 100,
        "optimizer" : "adam",
}

sweep_config = SimpleNamespace(**sweep_config_dict)


# Define your data transformations (modify as needed)
transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


transform_train = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomAdjustSharpness(2),
    transforms.RandomAdjustSharpness(0),
    transforms.RandomAutocontrast(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.Lambda(lambda img: transforms.functional.rotate(img, angle=90 * torch.randint(4, (1,)).item())),
    transforms.ToTensor(),
    # Finally the values are first rescaled to [0.0, 1.0]
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

#images = "/media/data/alex_johannes/octa_data/Cairo/DCP_images"
images = "../dcp_images"
#labels = "/media/data/alex_johannes/octa_data/Cairo/labels.csv"
labels = "../labels.csv"


train_dataset = image_loader.CNNImageLoader(path = images,
                                         label_file = labels,
                                         transform=transform_train,
                                         mode = "debug"
                                         )
test_dataset = image_loader.CNNImageLoader(path = images,
                                         label_file = labels,
                                         transform=transform_test,
                                         mode = "debug"
                                         )
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_labels = [label for _, label in train_dataset]
class_weights = prep.get_class_weights(train_labels, verbose = False)
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



from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.models._api import WeightsEnum
from torch.hub import load_state_dict_from_url



def main():

    if sweep_config.model == "resnet":
        # Load the pre-trained ResNet18 model
        #weights = ResNet18_Weights.DEFAULT
        model = resnet18() #weights=weights)
        model.apply(xavier_init)

    elif sweep_config.model == "effnet":
        #weights = EfficientNet_B2_Weights.DEFAULT
        #model = efficientnet_b2(weights=weights)

        def get_state_dict(self, *args, **kwargs):
            print(kwargs)
            kwargs.pop("check_hash")
            return load_state_dict_from_url(self.url, *args, **kwargs)
        WeightsEnum.get_state_dict = get_state_dict

        #model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        model = efficientnet_b0(weights="DEFAULT")

    ## Train all layers
    for param in model.parameters():
        param.requires_grad = True

    

    # Modify the last fully connected layer to match the number of classes in your task

    if sweep_config.model == "resnet":
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif sweep_config.model == "effnet":
        model.classifier._modules['1'] = nn.Linear(model.classifier._modules['1'].in_features, num_classes)
    balanced_loss = torch.nn.CrossEntropyLoss(class_weights)
    unbalanced_loss = torch.nn.CrossEntropyLoss()

    loss_dict = {"balanced": balanced_loss, "unbalanced": unbalanced_loss}

    # Define your loss function and optimizer
    criterion = loss_dict[sweep_config.class_weights]

    optimizer_dict = {"adam": optim.Adam(model.parameters(), lr=sweep_config.lr, weight_decay=sweep_config.weight_decay),
                        "sgd": optim.SGD(model.parameters(), lr=sweep_config.lr, momentum=0.9, weight_decay=sweep_config.weight_decay)}

    optimizer = optimizer_dict[sweep_config.optimizer]

    # Training loop
    model.to(device)


    batch_size = sweep_config.batch_size  
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=1)

    for epoch in range(1, sweep_config.epochs +1):
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

        print("Epoch: {} \Loss: {:.6f}  \tTraining Accuracy: {:.3f} \tTraining Balanced Accuracy: {:.3f}".format(epoch, loss, train_accuracy, train_balanced_accuracy))


# Save the fine-tuned model if needed
#torch.save(resnet.state_dict(), 'fine_tuned_resnet18.pth')

main()