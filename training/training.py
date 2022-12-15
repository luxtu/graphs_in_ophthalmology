import numpy as np
import torch
from tqdm import tqdm



def splitByValue(data, frac):
    num = data.shape[0]
    split = int(frac*num)

    idx_sort = np.argsort(data)
    data_sort = np.sort(data)

    train = idx_sort[:split]
    test = idx_sort[split:]

    split_val = data_sort[split]

    return np.array(train), np.array(test),split_val



def splitByValueTVT(data, frac):
    num = data.shape[0]

    splits = [round(f * num) for f in np.cumsum(frac)]
    splits = [0] + splits


    idx_sort = np.argsort(data)
    data_sort = np.sort(data)

    masks = []
    split_val = []

    for i in range(len(splits)-1):
        masks.append(idx_sort[splits[i]:splits[i+1]])

    for i in range(len(splits)-2):
        split_val.append(data_sort[splits[i+1]])

    return masks, split_val




class Splitter():
    def __init__(self, nxGraph, seed = None):
        self.nxGraph = nxGraph
        self.train_mask = None
        self.val_mask = None
        self.test_mask = None
        self.split_value = None
        self.split_plane_normal = None
        self.seed = seed

    def split_geometric_tvt(self, split_plane_normal, frac = (0.7,0.15, 0.15)):
        if np.sum(frac) != 1.0:
            raise ValueError("Fractions mut add up to 1!")

        positions = self.nxGraph["pos"].detach().numpy()

        res = positions* np.array(split_plane_normal)
        res = np.sum(res, axis = 1)
        
        masks, split_value = splitByValueTVT(res, frac)

        train_mask, val_mask, test_mask = masks[0], masks[1], masks[2]

        # convert to torch tensor objects
        train_mask= torch.tensor(train_mask)
        val_mask= torch.tensor(val_mask)
        test_mask= torch.tensor(test_mask)

        # update spliiter props
        self.train_mask = train_mask
        self.val_maks = val_mask
        self.test_mask = test_mask
        self.split_value = split_value
        self.split_plane_normal = split_plane_normal

        return train_mask,val_mask, test_mask, split_value
    
    
    def split_geometric(self, split_plane_normal, frac = 0.8):
        positions = self.nxGraph["pos"].detach().numpy()

        res = positions* np.array(split_plane_normal)
        res = np.sum(res, axis = 1)
        
        train_mask, test_mask, split_value = splitByValue(res, frac)

        # convert to torch tensor objects
        train_mask= torch.tensor(train_mask)
        test_mask= torch.tensor(test_mask)

        # update spliiter props
        self.train_mask = train_mask
        self.test_mask = test_mask
        self.split_value = split_value
        self.split_plane_normal = split_plane_normal

        return train_mask, test_mask, split_value


    def split_random(self, frac = 0.8, seed = None):
        # create the training and testing masks
        if seed is not None:
            self.seed = seed
        if self.seed is not None:
            np.random.seed(seed)

        node_num = self.nxGraph.y.shape[0]
        
        train_mask = np.random.choice(np.arange(0, node_num), size= int(node_num*frac), replace = False)
        test_mask = np.delete(np.arange(0, node_num), train_mask)

        # convert to torch tensor objects
        train_mask= torch.tensor(train_mask)
        test_mask= torch.tensor(test_mask)

        return train_mask, test_mask




class Trainer():
    def __init__(self, model, nxGraph, seed = 1234567, split = "random"):
        
        self.model = model
        self.seed = seed
        self.nxGraph = nxGraph
        self.nodeNum = nxGraph.y.shape[0]
        self.totalEpoch = 0 
        self.loss_l = []
        self.acc_l = []
        self.val_acc = None
        self.splitter = Splitter(nxGraph)


        if split == "random":
            self.train_mask, self.test_mask = self.splitter.split_random(frac = 0.8, seed = self.seed)

        else:
            self.planeNormal = split
            self.train_mask, self.test_mask, self.splitValue  = self.splitter.split_geometric(split_plane_normal = split, frac = 0.8)



    def trainXepochs(self, epochs = 100, verbose = False):

        if not verbose:
            pbar = tqdm(total = epochs)

        for epoch in range(1, epochs +1):
            self.totalEpoch += 1 

            loss = self.model.train(self.nxGraph, self.train_mask)
            self.loss_l.append(loss.detach().numpy().copy())
            test_acc = self.model.test(self.nxGraph, self.test_mask)
            if verbose:
                print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
                print(f'Epoch: {epoch:03d}, Val. Acc.: {test_acc:.4f}')
            else:
                pbar.update(1)
            self.acc_l.append(test_acc)

        if not verbose:
            pbar.close()

        return self.loss_l[-epochs:], self.acc_l[-epochs:]

    

    


#def trainEpochs(modelList, nxG, epochs = 100, verbose = False, seed = 1234567):
#
#    train_mask, test_mask = trainValSplit_random(nxG.y.shape[0], seed = seed)
#
#    loss_l = np.zeros((epochs, len(modelList)))
#    acc_l = np.zeros((epochs, len(modelList)))
#    for epoch in range(1, epochs +1):
#        for i, model in enumerate(modelList):
#            loss = model.train(nxG, train_mask)
#            loss_l[epoch-1, i] = loss.detach().numpy().copy()
#            if verbose:
#                print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
#            test_acc = model.test(nxG, test_mask)
#            acc_l[epoch-1, i] = test_acc
#
#    return loss_l, acc_l