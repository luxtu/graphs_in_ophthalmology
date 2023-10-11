import torch
import numpy as np



class graphClassifierClassic():
    def __init__(self, model, train_loader, test_loader, loss_func, lr = 0.005, weight_decay = 5e-5):

        self.model = model
        self.optimizer= torch.optim.AdamW(self.model.parameters(), lr= lr , weight_decay= weight_decay)
        self.lossFunc = loss_func
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.model.to(self.device)


    def train(self):
        self.model.train()
        cum_loss = 0
        size_data_set = len(self.train_loader.dataset) # must be done before iterating/regenerating the dataset
        for data in self.train_loader:
            
            out = self.model(data.x.float(), data.edge_index, data.batch, training = True)  # Perform a single forward pass.
            #print(out.shape)
            #print(data.y.shape)
            loss = self.lossFunc(out, data.y)  # Compute the loss solely based on the training nodes.
            loss.backward()  # Derive gradients.
            self.optimizer.step()  # Update parameters based on gradients.
            self.optimizer.zero_grad()  # Clear gradients.
            cum_loss += loss.item()
        return cum_loss/size_data_set

        


    @torch.no_grad()
    def test(self, loader):
        self.model.eval()
        correct = 0
        size_data_set = len(loader.dataset) # must be done before iterating/regenerating the dataset
        for data in loader:  # Iterate in batches over the training/test dataset.
            out = self.model(data.x.float(), data.edge_index, data.batch, training = False)  
            pred = out.argmax(dim=1)  # Use the class with highest probability.
            correct += int((pred == data.y).sum())  # Check against ground-truth labels.
        return correct / size_data_set  # Derive ratio of correct predictions.

    @torch.no_grad()
    def predict(self, loader):
        self.model.eval()
        outList = []
        yList = []
        for data in loader:  # Iterate in batches over the training/test dataset.
            out = self.model(data.x.float(), data.edge_index, data.batch, training = False)  
            outList.append(out.cpu())
            yList.append(data.y.cpu())
        return outList, yList 



class graphClassifierHetero():
    def __init__(self, model, train_loader, test_loader, loss_func, lr = 0.005, weight_decay = 5e-5):

        self.model = model
        self.optimizer= torch.optim.AdamW(self.model.parameters(), lr= lr, weight_decay= weight_decay)
        self.lossFunc = loss_func
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.model.to(self.device)


    def train(self):
        self.model.train()
        cum_loss = 0
        size_data_set = len(self.train_loader.dataset) # must be done before iterating/regenerating the dataset
        for data in self.train_loader:
            
            out = self.model(data.x_dict, data.edge_index_dict, data.batch_dict, data._slice_dict, training = True)  # Perform a single forward pass.
            #print(out.shape)
            #print(data.y.shape)
            loss = self.lossFunc(out, data.y)  # Compute the loss solely based on the training nodes.
            loss.backward()  # Derive gradients.
            self.optimizer.step()  # Update parameters based on gradients.
            self.optimizer.zero_grad()  # Clear gradients.
            cum_loss += loss.item()
        return cum_loss/size_data_set


    @torch.no_grad()
    def test(self, loader):
        self.model.eval()
        correct = 0
        size_data_set = len(loader.dataset) # must be done before iterating/regenerating the dataset
        for data in loader:  # Iterate in batches over the training/test dataset.
            out = self.model(data.x_dict, data.edge_index_dict, data.batch_dict, data._slice_dict, training = False)  
            pred = out.argmax(dim=1)  # Use the class with highest probability.
            correct += int((pred == data.y).sum())  # Check against ground-truth labels.
        return correct / size_data_set  # Derive ratio of correct predictions.

    @torch.no_grad()
    def predict(self, loader):
        self.model.eval()
        outList = []
        yList = []
        for data in loader:  # Iterate in batches over the training/test dataset.
            out = self.model(data.x_dict, data.edge_index_dict, data.batch_dict, data._slice_dict, training = False)  
            outList.append(out.cpu())
            yList.append(data.y.cpu())
        return outList, yList 
