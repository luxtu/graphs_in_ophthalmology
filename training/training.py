import numpy as np
import torch
from tqdm import tqdm


def trainValSplit_random(num, seed, frac = 0.8):
    # create the training and testing masks
    np.random.seed(seed)
    train_mask = np.random.choice(np.arange(0, num), size= int(num*frac), replace = False)
    test_mask = np.delete(np.arange(0, num), train_mask)

    # convert to torch tensor objects
    train_mask= torch.tensor(train_mask)
    test_mask= torch.tensor(test_mask)

    return train_mask, test_mask



def splitByValue(data, frac, reverse = False):
    num = data.shape[0]
    split = int(frac*num)

    print(split)

    split_rev = int(num-split)

    idx_sort = np.argsort(data)
    #data_sort = np.sort(data)

    if not reverse:
        train = idx_sort[:split]
        test = idx_sort[split:]
    else:
        test = idx_sort[:split_rev]
        train = idx_sort[split_rev:]

    return np.array(train), np.array(test)






def trainValSplit_geometric(nxGraph, frac = 0.8, slicing_plane = "xy"):
    # with the plane xy, the z value is found that first covers 80% of the data:

    positions = nxGraph["pos"].detach().numpy()

    reverse = True if slicing_plane[0] == "-" else False

    if "x" not in slicing_plane:
        pos_data = positions[:,0]

    elif "y" not in slicing_plane:
        pos_data = positions[:,1]
    
    elif "z" not in slicing_plane:
        pos_data = positions[:,2]
        
    train_mask, test_mask = splitByValue(pos_data, frac, reverse)

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

        if split == "random":
            self.train_mask, self.test_mask = trainValSplit_random(self.nodeNum, self.seed)

        else:
            self.train_mask, self.test_mask = trainValSplit_geometric(self.nxGraph, slicing_plane= "xy")



    def trainXepochs(self, epochs = 100, verbose = False):

        if not verbose:
            pbar = tqdm(total = epochs)

        for epoch in range(1, epochs +1):
            self.totalEpoch += 1 

            loss = self.model.train(self.nxGraph, self.train_mask)
            self.loss_l.append(loss.detach().numpy().copy())
            if verbose:
                print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
            else:
                pbar.update(1)
            test_acc = self.model.test(self.nxGraph, self.test_mask)
            self.acc_l.append(test_acc)

        pbar.close()

        return self.loss_l[-epochs:], self.acc_l[-epochs:]

    

    


def trainEpochs(modelList, nxG, epochs = 100, verbose = False, seed = 1234567):

    train_mask, test_mask = trainValSplit_random(nxG.y.shape[0], seed = seed)

    loss_l = np.zeros((epochs, len(modelList)))
    acc_l = np.zeros((epochs, len(modelList)))
    for epoch in range(1, epochs +1):
        for i, model in enumerate(modelList):
            loss = model.train(nxG, train_mask)
            loss_l[epoch-1, i] = loss.detach().numpy().copy()
            if verbose:
                print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
            test_acc = model.test(nxG, test_mask)
            acc_l[epoch-1, i] = test_acc

    return loss_l, acc_l