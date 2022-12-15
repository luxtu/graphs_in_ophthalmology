import torch
import wandb
import numpy as np
import pandas as pd
from scipy import stats
import preprocessing.preprocessing as pp
from torch_geometric.utils.convert import from_networkx
import networkx as nx
import argparse
from models import nodeClassifier
from visualization import mesh_viewer
import training.training as tt

##################
### Hyperparameter Optimization with WANDB for L node classification
##################


# use argparse to give the performed sweep a name
parser = argparse.ArgumentParser()
parser.add_argument('-s', type=str, required= True, help = "The name for the performed sweep.")
args = parser.parse_args()

sweep_name = args.s


# load the graph
G_comb_einf_lab = nx.read_gpickle("saved_data/graph_gt_pickle_upsamp")

# convert to dual and convert to torch and assign labels
L_comb_einf_lab = pp.makeDual(G_comb_einf_lab, include_orientation= False)
LX_comb_einf_lab = from_networkx(L_comb_einf_lab)
class_label_list, node_lab, node_lab_explain =  pp.getLablesForDual(L_comb_einf_lab)

# assign the ground truth class information to the torch data obj
LX_comb_einf_lab.y = torch.tensor(class_label_list)

# extract the number of classes and features for model input
num_feat_dual_comb = LX_comb_einf_lab.x.shape[1]
num_class_dual_comb = len(np.unique(LX_comb_einf_lab.y))

# split the data according to geometry not random 
splitter = tt.Splitter(LX_comb_einf_lab)

geom_train_mask, geom_test_mask, splitValue = splitter.split_geometric((0,1,0), frac = 0.6)

train_list = np.array(list(L_comb_einf_lab.nodes()))[geom_train_mask].tolist()
train_nodes = [(elem[0],elem[1]) for elem in train_list]

test_list = np.array(list(L_comb_einf_lab.nodes()))[geom_test_mask].tolist()
test_nodes = [(elem[0],elem[1]) for elem in test_list]

#creates the subgraphs (induces by the nodes selected node)
train_subG = L_comb_einf_lab.subgraph(train_nodes).copy()
test_subG = L_comb_einf_lab.subgraph(test_nodes).copy()

# combine the subgraphs again, all the combining edges between the geometric split are not present anymore
split_wholeG = nx.union(train_subG, test_subG)

#mesh_viewer.renderNXGraph(train_subG, dual = True, vtk = 0,backend = "static")
#mesh_viewer.renderNXGraph(test_subG, dual = True, vtk = 0,backend = "static")
#mesh_viewer.renderNXGraph(split_wholeG, dual = True, vtk = 0,backend = "static")

# make the graph a torch obj
split_wholeG_torch = from_networkx(split_wholeG)
split_wholeG_y, _, _ =  pp.getLablesForDual(split_wholeG, node_lab)

# normalizing the node features
xL_data = split_wholeG_torch.x.detach().numpy()
xL_data = xL_data[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,15]]
xL_data = (xL_data - np.median(xL_data, axis = 0)) / stats.median_abs_deviation(xL_data, axis = 0, scale = "normal")
#xL_data = (xL_data - xL_data.min(0)) / xL_data.ptp(0)
#np.nan_to_num(xL_data, copy = False)
split_wholeG_torch.x = torch.tensor(xL_data)

# assign the right classes 
split_wholeG_torch.y = torch.tensor(split_wholeG_y)
geom_train_mask_torch = np.arange(0, train_subG.order())
geom_test_mask_torch = np.arange(train_subG.order(),train_subG.order()+ test_subG.order())



sweep_variable_layyer_num = {
    "name": sweep_name,
    "method": "bayes",
    "metric": {
        "name": "gcn/max_accuracy",
        "goal": "maximize",
    },
    "parameters": {
        
        "models": {
            "values": ["GCN", "GAT", "CLUST", "SAGE"]
        },
        "hidden_channels": {
            "values": [32, 64, 128]
        },

        "weight_decay": {
            "distribution": "normal",
            "mu": 2e-4,
            "sigma": 8e-5,
        },
        "lr": {
            "min": 1e-5,
            "max": 1e-2
        },
        "dropout": {
            "values": [0.5]
        },
        "num_layers": {
            "values": [1,2,3,4,6]
        }
    }
}
# "norm": {"values": [True, False]}
# "skip": {"values": [True, False]}
# "MLP": {"values": [True]}
# "aggr": {"values": ["max", "std", "mean", "sum"]} #"add", "sum" ,"mean", "min", "max" 1st batch
        


sweep_id = wandb.sweep(sweep_variable_layyer_num, project= "node_class_upsamp")

# define optimizer and loss
optimizerAdam = torch.optim.Adam
optimizerAdamW = torch.optim.AdamW
criterionCEL = torch.nn.CrossEntropyLoss

# set seed for reproducibility 
np.random.seed(1234567)

# create the training and testing masks
train_mask_random = np.random.choice(np.arange(0, geom_train_mask_torch.shape[0]), size= int(geom_train_mask_torch.shape[0]*0.85), replace = False)
test_mask_random = np.delete(np.arange(0, geom_train_mask_torch.shape[0]), train_mask_random)


# convert to torch tensor objects
train_mask_random= torch.tensor(train_mask_random)
test_mask_random = torch.tensor(test_mask_random)

sweeper = nodeClassifier.nodeClassifierSweep(features = np.arange(num_feat_dual_comb), classes = num_class_dual_comb, optimizer = optimizerAdamW, lossFunc = criterionCEL, graph = split_wholeG_torch,  train_mask = train_mask_random, test_mask = test_mask_random, epochs = 1000, geom_test_mask= geom_test_mask_torch)

#Run the Sweeps agent
wandb.agent(sweep_id, project="node_classification_geom_split", function=sweeper.agent_variable_size_model, count = 20)
