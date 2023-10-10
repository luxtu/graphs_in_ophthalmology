from torchvision.models import resnet18, ResNet18_Weights, efficientnet_b3, EfficientNet_B3_Weights
from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim
from loader import image_loader
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from utils import prep

import argparse
import wandb

argparser = argparse.ArgumentParser()
argparser.add_argument('--run_name', type=str)
argparser.add_argument('--model', type=str)

args = argparser.parse_args()



torch.cuda.empty_cache()


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


train_dataset = image_loader.CNNImageLoader("/media/data/alex_johannes/octa_data/Cairo/SCP_images",
                                         "/media/data/alex_johannes/octa_data/Cairo/labels.csv",
                                         transform=transform_train,
                                         mode = "train"
                                         )
test_dataset = image_loader.CNNImageLoader("/media/data/alex_johannes/octa_data/Cairo/SCP_images",
                                         "/media/data/alex_johannes/octa_data/Cairo/labels.csv",
                                         transform=transform_test,
                                         mode = "val"
                                         )
                                        
train_labels = [label for _, label in train_dataset]
class_weights = prep.get_class_weights(train_labels, verbose = True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 3  # Change this to the number of classes in your dataset

if args.model == "resnet":
    # Load the pre-trained ResNet18 model
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)

elif args.model == "effnet":
    weights = EfficientNet_B3_Weights.DEFAULT
    model = efficientnet_b3(weights=weights)



## Freeze all layers
for param in model.parameters()[-5:]:
    param.requires_grad = False

    


# Modify the last fully connected layer to match the number of classes in your task

if args.model == "resnet":
    model.fc = nn.Linear(model.fc.in_features, num_classes)

elif args.model == "effnet":
    model.classifier._modules['1'] = nn.Linear(model.classifier._modules['1'].in_features, num_classes)


# Define your loss function and optimizer
criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights).to(device))
criterion_unbalanced = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-5)
#optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


wandb.init(project="graph_pathology")
wandb.run.name = args.run_name


# Training loop
num_epochs = 100 
model.to(device)


batch_size = 32  
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=1)

for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion_unbalanced(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

    # get the test loss and accuracy

    test_loss = 0.0
    model.eval()  # prep model for evaluation

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion_unbalanced(output, target)
        test_loss += loss.item() * data.size(0)

    # calculate avg test loss
    test_loss = test_loss / len(test_loader.dataset)

    train_accuracy, train_balanced_accuracy = prep.evaluate_cnn(model, train_loader)
    test_accuracy, test_balanced_accuracy = prep.evaluate_cnn(model, test_loader)

    wandb.log({"loss": loss, "train_acc": train_accuracy, "test_acc": test_accuracy, "test_bal_acc": test_balanced_accuracy})


# Save the fine-tuned model if needed
#torch.save(resnet.state_dict(), 'fine_tuned_resnet18.pth')