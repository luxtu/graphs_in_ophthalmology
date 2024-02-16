import torch
import numpy as np
from scipy.ndimage import gaussian_filter
from torch.optim.lr_scheduler import ExponentialLR


def multi_label_loss(out, y, num_classes, loss_func, device):
    # create one hot encoding
    y_one_hot = torch.nn.functional.one_hot(y, num_classes = num_classes).float()
    for i, label in enumerate(y):
        y_one_hot[i,:label+1] = 1
        if label > 0:
            y_one_hot[i,0] = 0

    #print(y_one_hot)
    #print(out)

    pos_weight = torch.ones(num_classes, device = device)
    pos_weight -= 0.8
    for i, label in enumerate(y):
        y_one_hot[i,label] = 1

    loss_func.pos_weight = pos_weight

    loss = loss_func(out, y_one_hot) 
    return loss
    


def smoothed_label_loss(out, y, num_classes, loss_func, device):
    smooth_y = torch.nn.functional.one_hot(y, num_classes=num_classes).float()
    # gaussian filter the labels
    # smooth_y = torch.nn.functional.gaussian_filter(smooth_y, sigma=1) 
    # Convert PyTorch tensor to NumPy array
    smooth_y_np = smooth_y.cpu().numpy()  # Assuming you're on CPU
    # convert to float
    smooth_y_np = smooth_y_np.astype(float)
    # Apply Gaussian filter using scipy
    smooth_y_filtered = gaussian_filter(smooth_y_np, sigma=0.4) # 0.45 worked well # small sigma alsmost no smoothing
    # Convert back to PyTorch tensor
    smooth_y_filtered = torch.from_numpy(smooth_y_filtered).to(device)
    # make sure its a float
    smooth_y_filtered = smooth_y_filtered.float()
    loss = loss_func(out, smooth_y_filtered) # data.y)
    return loss





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
    def __init__(self, model, loss_weights = None, lr = 0.005, weight_decay = 5e-5, smooth_label_loss = False, SAM_opt = False):

        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.epoch = 0
        
        self.model.to(self.device)
        self.SAM_opt = SAM_opt
        
        #self.optimizer= torch.optim.Adam(self.model.parameters(), lr= lr, weight_decay= weight_decay) # usually adamW
        #self.optimizer = torch.optim.Adam(self.model.parameters(), lr= lr)
        #self.optimizer = torch.optim.SGD(self.model.parameters(), lr= lr, momentum=0.9)
        if self.SAM_opt:
            base_optimizer = torch.optim.AdamW  # define an optimizer for the "sharpness-aware" update
            self.optimizer = SAM(self.model.parameters(), base_optimizer, lr=lr, weight_decay = weight_decay, rho= 0.05) # , momentum=0.9  
            self.scheduler = ExponentialLR(self.optimizer.base_optimizer, gamma=0.95)
        else:
            self.optimizer= torch.optim.AdamW(self.model.parameters(), lr= lr, weight_decay= weight_decay)
            self.scheduler = ExponentialLR(self.optimizer, gamma=0.95)
        if smooth_label_loss:
            self.lossFunc = torch.nn.CrossEntropyLoss(weight=loss_weights, label_smoothing=0.1)
        else:
            self.lossFunc = torch.nn.CrossEntropyLoss(weight=loss_weights)
        self.smooth_label_loss = smooth_label_loss

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
            pos_dict = {}
            #print(data.y)
            for key in ["graph_1", "graph_2"]:
                pos_dict[key] = data[key].pos
            idx = torch.randint(0, len(self.model.aggr_keys), (1,)) # idx is only relevant for the random modality gnn
            if self.SAM_opt:
                enable_running_stats(self.model)
            else:
                self.optimizer.zero_grad()  # Clear gradients.
            out = self.model(data.x_dict, data.edge_index_dict, data.batch_dict, idx = idx) 
            loss = self.lossFunc(out, data.y)
            loss.backward()  # Derive gradients.

            if self.SAM_opt:
                # two steps for sam optimizer
                self.optimizer.first_step(zero_grad=True)
                disable_running_stats(self.model)
                self.lossFunc(self.model(data.x_dict, data.edge_index_dict, data.batch_dict, idx = idx), data.y).backward()
                self.optimizer.second_step(zero_grad=True)
            else:
                # single step for adam optimizer
                self.optimizer.step()
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
        self.epoch += 1
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
            out = self.model(data.x_dict, data.edge_index_dict, data.batch_dict, pos_dict = pos_dict) 

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
                
            out = self.model(data.x_dict, data.edge_index_dict, data.batch_dict, pos_dict = pos_dict)

            raw_out.append(out.cpu().detach().numpy())
            y_out.append(data.y.cpu().detach().numpy())
        
        #raw_out)
        pred = np.concatenate(raw_out, axis = 0)
        y = np.concatenate(y_out, axis = 0)

        return pred, y 




class graphClassifierSimple():
    def __init__(self, model, loss_func, lr = 0.005, weight_decay = 5e-5, smooth_label_loss = False):

        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model.to(self.device)
        self.optimizer= torch.optim.AdamW(self.model.parameters(), lr= lr, weight_decay= weight_decay)
        #self.optimizer = torch.optim.Adam(self.model.parameters(), lr= lr)
        #self.optimizer = torch.optim.SGD(self.model.parameters(), lr= lr, momentum=0.9)
        self.scheduler = ExponentialLR(self.optimizer, gamma=0.95)
        self.lossFunc = loss_func
        self.smooth_label_loss = smooth_label_loss

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

            out =  self.model(data.x, data.edge_index, data.batch)  # Perform a single forward pass. #  pos_dict = pos_dict, 
            #print(out.shape)
            #print(data.y.shape)

            # get the number of classes from the output
            num_classes = out.shape[1]
            if self.smooth_label_loss:
                loss = smoothed_label_loss(out, data.y, num_classes, self.lossFunc, self.device)
            else:
                #loss = multi_label_loss(out, data.y, num_classes, self.lossFunc, self.device)
                loss = self.lossFunc(out, data.y)
            #loss = custom_loss(out_dis, out_stage, data.y)#.backward()  # Derive gradients.
            loss.backward()  # Derive gradients.
            self.optimizer.step()  # Update parameters based on gradients.
            self.optimizer.zero_grad()  # Clear gradients.
            cum_loss += loss.item()


            raw_out.append(out.cpu().detach().numpy())
            y_out.append(data.y.cpu().detach().numpy())
        
        #raw_out)
        pred = np.concatenate(raw_out, axis = 0)
        y = np.concatenate(y_out, axis = 0)
        self.scheduler.step()
        return cum_loss/size_data_set, pred, y


    @torch.no_grad()
    def test(self, loader):
        self.model.eval()
        correct = 0
        size_data_set = len(loader.dataset) # must be done before iterating/regenerating the dataset
        for data in loader:  # Iterate in batches over the training/test dataset.
            data.to(self.device)
            out = self.model(data.x, data.edge_index, data.batch)

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
            out = self.model(data.x, data.edge_index, data.batch)
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



class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

import torch
from torch.nn.modules.batchnorm import _BatchNorm


def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)

def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)




class graphClassifierHetero_94d26db():
    def __init__(self, model, loss_func, lr = 0.005, weight_decay = 5e-5, regression = False, smooth_label_loss = False):

        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model.to(self.device)
        self.optimizer= torch.optim.AdamW(self.model.parameters(), lr= lr, weight_decay= weight_decay)
        self.scheduler = ExponentialLR(self.optimizer, gamma=0.95)
        self.lossFunc = loss_func
        self.regression = regression
        self.smooth_label_loss = smooth_label_loss

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
            pos_dict = {}
            #print(data.y)
            for key in ["graph_1", "graph_2"]:
                pos_dict[key] = data[key].pos
            out =  self.model(data.x_dict, data.edge_index_dict, data.batch_dict, regression =self.regression, pos_dict = pos_dict)  # Perform a single forward pass. #  pos_dict = pos_dict, 
            #print(out.shape)
            #print(data.y.shape)

            # get the number of classes from the output
            num_classes = out.shape[1]

            if self.smooth_label_loss:
                loss = smoothed_label_loss(out, data.y, num_classes, self.lossFunc, self.device)
            else:
                #loss = multi_label_loss(out, data.y, num_classes, self.lossFunc, self.device)
                loss = self.lossFunc(out, data.y)

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
            out = self.model(data.x_dict, data.edge_index_dict, data.batch_dict, pos_dict = pos_dict, regression =self.regression)
            raw_out.append(out.cpu().detach().numpy())
            y_out.append(data.y.cpu().detach().numpy())
        
        #raw_out)
        pred = np.concatenate(raw_out, axis = 0)
        y = np.concatenate(y_out, axis = 0)

        return pred, y 