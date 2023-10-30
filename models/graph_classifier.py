import torch
import numpy as np
from torch.optim.lr_scheduler import ExponentialLR

class graphClassifierHetero():
    def __init__(self, model, loss_func, lr = 0.005, weight_decay = 5e-5, regression = False):

        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model.to(self.device)
        self.optimizer= torch.optim.AdamW(self.model.parameters(), lr= lr, weight_decay= weight_decay)
        self.scheduler = ExponentialLR(self.optimizer, gamma=0.95)
        self.lossFunc = loss_func
        self.regression = regression

    def train(self, loader):
        self.model.train()
        cum_loss = 0
        size_data_set = len(loader.dataset) # must be done before iterating/regenerating the dataset
        for data in loader:
            data.to(self.device)
            #print(torch.cuda.memory_allocated()/1024/1024/1024)
            pos_dict = {}
            #print(data.y)
            for key in ["graph_1", "graph_2"]:
                pos_dict[key] = data[key].pos
            out = self.model(data.x_dict, data.edge_index_dict, data.batch_dict, training = True, pos_dict = pos_dict, regression =self.regression)  # Perform a single forward pass.
            #print(out.shape)
            #print(data.y.shape)
            loss = self.lossFunc(out, data.y)  # Compute the loss solely based on the training nodes.
            loss.backward()  # Derive gradients.
            self.optimizer.step()  # Update parameters based on gradients.
            self.optimizer.zero_grad()  # Clear gradients.
            cum_loss += loss.item()
        self.scheduler.step()
        return cum_loss/size_data_set


    @torch.no_grad()
    def test(self, loader):
        self.model.eval()
        correct = 0
        size_data_set = len(loader.dataset) # must be done before iterating/regenerating the dataset
        for data in loader:  # Iterate in batches over the training/test dataset.
            data.to(self.device)
            pos_dict = {}
            for key in ["graph_1", "graph_2"]:
                pos_dict[key] = data[key].pos
            out = self.model(data.x_dict, data.edge_index_dict, data.batch_dict, training = False, pos_dict = pos_dict, regression =self.regression) 

            if self.regression:
                # assign classes according to thresholds
                out = out.squeeze()
                out[out<0.5] = 0
                out[(out>=0.5) & (out<1.5)] = 1
                out[out>=1.5] = 2 # ) & (out<2.5)

                # convert to int
                pred = out.int()

            else: 
                pred = out.argmax(dim=1)  # Use the class with highest probability. 
            correct += int((pred == data.y).sum())  # Check against ground-truth labels.
        return correct / size_data_set  # Derive ratio of correct predictions.

    @torch.no_grad()
    def predict(self, loader):
        self.model.eval()
        outList = []
        yList = []
        for data in loader:  # Iterate in batches over the training/test dataset.
            data.to(self.device)
            pos_dict = {}
            for key in ["graph_1", "graph_2"]:
                pos_dict[key] = data[key].pos
            out = self.model(data.x_dict, data.edge_index_dict, data.batch_dict,  training = False, pos_dict = pos_dict, regression =self.regression)  
            if self.regression:
                # assign classes according to thresholds
                #out = out.squeeze()
                out[out<0.5] = 0
                out[(out>=0.5) & (out<1.5)] = 1
                out[out>=1.5] = 2
                out = out.int()

            outList.append(out.cpu().detach().numpy().squeeze())
            yList.append(data.y.cpu())
        return outList, yList 







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

