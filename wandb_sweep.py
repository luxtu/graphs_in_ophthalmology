import torch
import wandb
import numpy as np
import pandas as pd
import preprocessing.preprocessing as pp
from torch_geometric.utils.convert import from_networkx
import networkx as nx
import argparse
from models import nodeClassifier

##################
### Hyperparameter Optimization with WANDB for L node classification
##################

# use argparse to give the performed sweep a name
parser = argparse.ArgumentParser()
parser.add_argument('-s', type=str, required= True, help = "The name for the performed sweep.")
args = parser.parse_args()

sweep_name = args.s


# load the graph
G_comb_einf_lab = nx.read_gpickle("saved_data/graph_gt_pickle")

# convert to dual and convert to torch and assign labels
L_comb_einf_lab = pp.makeDual(G_comb_einf_lab, include_orientation= False)
LX_comb_einf_lab = from_networkx(L_comb_einf_lab)
class_label_list, node_lab, node_lab_explain =  pp.getLablesForDual(L_comb_einf_lab)

# assign the ground truth class information to the torch data obj
LX_comb_einf_lab.y = torch.tensor(class_label_list)

# extract the number of classes and features for model input
num_feat_dual_comb = LX_comb_einf_lab.x.shape[1]
num_class_dual_comb = len(np.unique(LX_comb_einf_lab.y))


# define parameters for the sweep
sweep_variable_layyer_num = {
    "name": sweep_name,
    "method": "bayes",
    "metric": {
        "name": "gcn/max_accuracy",
        "goal": "maximize",
    },
    "parameters": {
        
        "models": {
            "values": ["SAGE", "CLUST"]
        },
        "hidden_channels": {
            "values": [32, 64]
        },

        "weight_decay": {
            "distribution": "normal",
            "mu": 2e-4,
            "sigma": 2e-5,
        },
        "lr": {
            "min": 5e-5,
            "max": 1e-2
        },
        "dropout": {
            "values": [0.5]
        },
        "num_layers": {
            "values": [4,6,8]
        }
    }
}



sweep_id = wandb.sweep(sweep_variable_layyer_num, project= "node_classification_comb")

# define optimizer and loss
optimizerAdam = torch.optim.Adam
optimizerAdamW = torch.optim.AdamW
criterionCEL = torch.nn.CrossEntropyLoss

# set seed for reproducibility 
np.random.seed(1234567)

# create the training and testing masks
train_mask = np.random.choice(np.arange(0, LX_comb_einf_lab.y.shape[0]), size= int(LX_comb_einf_lab.y.shape[0]*0.8), replace = False)
test_mask = np.delete(np.arange(0, LX_comb_einf_lab.y.shape[0]), train_mask)

# convert to torch tensor objects
train_mask= torch.tensor(train_mask)
test_mask= torch.tensor(test_mask)


sweeper = nodeClassifier.nodeClassifierSweep(features = np.arange(num_feat_dual_comb), classes = num_class_dual_comb, optimizer = optimizerAdamW, lossFunc = criterionCEL, graph = LX_comb_einf_lab,  train_mask = train_mask, test_mask = test_mask, epochs = 2000)

#Run the Sweeps agent
wandb.agent(sweep_id, project="node_classification_comb", function=sweeper.agent_variable_size_model, count = 100)
