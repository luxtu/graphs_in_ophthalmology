import torch
import numpy as np
from scipy.ndimage import gaussian_filter
from torch.optim.lr_scheduler import ExponentialLR


def custom_loss(out_diseased, out_stage, y):

    out_diseased = out_diseased.squeeze()
    out_stage = out_stage.squeeze()
    y = y.float()


    # y_diseased is every value >1:
    y_diseased = torch.zeros_like(y)
    y_diseased[y>0] = 1

    diseased_loss = torch.nn.functional.binary_cross_entropy(out_diseased, y_diseased) # binary_cross_entropy_with_logits

    # y_stage only y values above 0 are relevant
    y_stage = torch.zeros_like(y[y>0])
    y_stage[y[y>0] == 2] = 1

    stage_loss = torch.nn.functional.binary_cross_entropy(out_stage[y>0], y_stage)

    #check if there are any diseased samples
    if y_diseased.sum() == 0:
        total_loss = diseased_loss
        #print("no diseased samples")
        #print(diseased_loss)
    else:
        total_loss = diseased_loss + stage_loss
        #print("combined loss")
        #print(total_loss)


    return total_loss



class graphClassifierHetero():
    def __init__(self, model, loss_func, lr = 0.005, weight_decay = 5e-5, regression = False):

        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model.to(self.device)
        self.optimizer= torch.optim.AdamW(self.model.parameters(), lr= lr, weight_decay= weight_decay)
        #self.optimizer = torch.optim.Adam(self.model.parameters(), lr= lr)
        #self.optimizer = torch.optim.SGD(self.model.parameters(), lr= lr, momentum=0.9)
        self.scheduler = ExponentialLR(self.optimizer, gamma=0.95)
        self.lossFunc = loss_func
        self.regression = regression

    def train(self, loader, data_loss_dict = None):
        self.model.train()
        cum_loss = 0
        raw_out = []
        y_out = []
        if data_loss_dict is None:
            data_loss_dict = {}

        size_data_set = len(loader.dataset) # must be done before iterating/regenerating the dataset
        for data in loader:
            #data.to(self.device)
            #pos_dict = {}
            #print(data.y)
            #for key in ["graph_1", "graph_2"]:
            #    pos_dict[key] = data[key].pos
            #out_dis, out_stage = self.model(data.x_dict, data.edge_index_dict, data.batch_dict, pos_dict = pos_dict, regression =self.regression)  # Perform a single forward pass.
            out =  self.model(data.x_dict, data.edge_index_dict, data.batch_dict, regression =self.regression)  # Perform a single forward pass. #  pos_dict = pos_dict, 
            #print(out.shape)
            #print(data.y.shape)

            smooth_y = torch.nn.functional.one_hot(data.y, num_classes=4).float()
            # gaussian filter the labels
            # smooth_y = torch.nn.functional.gaussian_filter(smooth_y, sigma=1) 
            # Convert PyTorch tensor to NumPy array
            smooth_y_np = smooth_y.cpu().numpy()  # Assuming you're on CPU
            # convert to float
            smooth_y_np = smooth_y_np.astype(float)
            # Apply Gaussian filter using scipy
            smooth_y_filtered = gaussian_filter(smooth_y_np, sigma=0.45)
            # Convert back to PyTorch tensor
            smooth_y_filtered = torch.from_numpy(smooth_y_filtered).to(self.device)
            # make sure its a float
            smooth_y_filtered = smooth_y_filtered.float()

            loss = self.lossFunc(out, smooth_y_filtered) # data.y)
            #loss = custom_loss(out_dis, out_stage, data.y)#.backward()  # Derive gradients.
            loss.backward()  # Derive gradients.
            self.optimizer.step()  # Update parameters based on gradients.
            self.optimizer.zero_grad()  # Clear gradients.
            cum_loss += loss.item()


            # get graph_ids 
            ids = data.graph_id
            for id in ids:
                if id not in data_loss_dict:
                    data_loss_dict[id] = 0
                data_loss_dict[id]+= loss.item()


            raw_out.append(out.cpu().detach().numpy())
            y_out.append(data.y.cpu().detach().numpy())
        
        #raw_out)
        pred = np.concatenate(raw_out, axis = 0)
        y = np.concatenate(y_out, axis = 0)
        self.scheduler.step()
        return cum_loss/size_data_set, pred, y, data_loss_dict


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
            out = self.model(data.x_dict, data.edge_index_dict, data.batch_dict, pos_dict = pos_dict, regression =self.regression) 

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

        raw_out = []
        y_out = []
        self.model.eval()
        for data in loader:  # Iterate in batches over the training/test dataset.
            data.to(self.device)
            pos_dict = {}
            for key in ["graph_1", "graph_2"]:
                pos_dict[key] = data[key].pos
            #out_1, out_2 = self.model(data.x_dict, data.edge_index_dict, data.batch_dict, pos_dict = pos_dict, regression =self.regression)
            out = self.model(data.x_dict, data.edge_index_dict, data.batch_dict, pos_dict = pos_dict, regression =self.regression)
            #out_1 = out_1.squeeze(1)
            #out_2 = out_2.squeeze(1)
            #print(out_1)
            #print(out_2)

            #out = torch.zeros_like(out_1)

            #out[out_1>0] = 1
            #out[out_2>0] = 2

            #out[out_1<=0] = 0
            #if self.regression:
            #    # assign classes according to thresholds
            #    out = out.squeeze()
            #    out[out<0.5] = 0
            #    out[(out>=0.5) & (out<1.5)] = 1
            #    out[out>=1.5] = 2
            #    out = out.int()
#
            ## concatenate the raw output
            #else:
            #    
#
            raw_out.append(out.cpu().detach().numpy())
            y_out.append(data.y.cpu().detach().numpy())
        
        #raw_out)
        pred = np.concatenate(raw_out, axis = 0)
        y = np.concatenate(y_out, axis = 0)

        return pred, y 







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

