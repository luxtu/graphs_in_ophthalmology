import torch
import wandb
import numpy as np
import pandas as pd
from scipy import stats
import preprocessing.preprocessing as pp
from torch_geometric.utils.convert import from_networkx
import networkx as nx
import argparse
import tifffile as tiff
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


# base the classifcation on the mask labels at the position of the nodes
nerve_mask =  "../Intestine/nerve-mask/nerve_mask_stack_255_fh_upsamp.tif"
lymph_mask =  "../Intestine/lymph-mask/lymph_mask_stack_255_fh_upsamp.tif"

# read tif and convert to numpy 
nerve_mask_np = np.array(tiff.imread(nerve_mask)).T
lymph_mask_np = np.array(tiff.imread(lymph_mask)).T


# load the graph
G_comb_einf_lab = nx.read_gpickle("saved_data/graph_gt_pickle_upsamp")
G_comb_einf = nx.read_gpickle("saved_data/graph_gt_pickle_nolab_upsamp")

# convert to dual and convert to torch and assign labels
L_comb_einf_lab = pp.make_dual(G_comb_einf, include_orientation= True)
path = "/home/laurin/Documents/Intestine/combined-mask/vg_bs2_fh_upsamp.vvg"
df_centerline = pp.vvg_to_df(path)
label_dict, label_dict_nodes = pp.label_edges_centerline([nerve_mask_np, lymph_mask_np], voxel_size = (0.00217391,0.00217391,0.00217391), df_centerline=df_centerline)


clenead_label_dict = {}
for key, val in label_dict_nodes.items():
    if key[0] == key[1]:
        continue
    try:
        L_comb_einf_lab[key]
        clenead_label_dict[key] = val
    except KeyError:
        L_comb_einf_lab[(key[1],key[0])]
        clenead_label_dict[(key[1],key[0])] = val


nx.set_node_attributes(L_comb_einf_lab,clenead_label_dict, "y")




LX_comb_einf_lab = from_networkx(L_comb_einf_lab)
#class_label_list, node_lab, node_lab_explain =  pp.label_dual(L_comb_einf_lab, overrule = "l")

# assign the ground truth class information to the torch data obj
#LX_comb_einf_lab.y = torch.tensor(class_label_list)

# split the data according to geometry not random 
splitter = tt.Splitter(LX_comb_einf_lab)
#train_mask, val_mask, test_mask, splitValue = splitter.split_geometric_tvt((0,1,0), frac = (0.8,0.19,0.01))
train_mask, val_mask, splitValue = splitter.split_geometric((0,1,0), frac = 0.8)

train_list = np.array(list(L_comb_einf_lab.nodes()))[train_mask].tolist()
train_nodes = [(elem[0],elem[1]) for elem in train_list]

val_list = np.array(list(L_comb_einf_lab.nodes()))[val_mask].tolist()
val_nodes = [(elem[0],elem[1]) for elem in val_list]

#test_list = np.array(list(L_comb_einf_lab.nodes()))[test_mask].tolist()
#test_nodes = [(elem[0],elem[1]) for elem in test_list]

#creates the subgraphs (induces by the nodes selected node)
train_subG = L_comb_einf_lab.subgraph(train_nodes).copy()
val_subG = L_comb_einf_lab.subgraph(val_nodes).copy()
#test_subG = L_comb_einf_lab.subgraph(test_nodes).copy()


#mesh_viewer.renderNXGraph(train_subG, dual = True, vtk = 0,backend = "static")
#mesh_viewer.renderNXGraph(val_subG, dual = True, vtk = 0,backend = "static")
#mesh_viewer.renderNXGraph(test_subG, dual = True, vtk = 0,backend = "static")

# combine the subgraphs again, all the combining edges between the geometric split are not present anymore
#split_wholeG = nx.union_all([train_subG, val_subG, test_subG])
split_wholeG = nx.union_all([train_subG, val_subG])


#sparse_adj_mat =nx.adjacency_matrix(split_wholeG)
#adj2 = sparse_adj_mat**2
#adj2_dok = adj2.todok()
#
#split_wholeG_nodes = np.array(list(split_wholeG.nodes()))
#
#for key, val in adj2_dok.items():
#    if not key[0] == key[1]:
#        split_wholeG.add_edge(tuple(split_wholeG_nodes[key[0]]), tuple(split_wholeG_nodes[key[1]]))



# make the graph a torch obj
split_wholeG_torch = from_networkx(split_wholeG)
#split_wholeG_y, _, _ =  pp.label_dual(split_wholeG, node_lab, overrule = "l")

# normalizing the node features
xL_data = split_wholeG_torch.x.detach().numpy()


degs_np = np.array(list(split_wholeG.degree()))[:,1]
#triangs_np = np.array(list(nx.triangles(split_wholeG).values()))

#introduce more features:

rel_idx = [0,1,2,3,4,5,6,7,8,9,10,11,12,15, 17,18,19]

new_data = np.zeros((xL_data.shape[0], len(rel_idx)+4))

new_data[:,:len(rel_idx)] = xL_data[:, rel_idx]

sparse_adj_mat =nx.adjacency_matrix(split_wholeG)

adj2 = sparse_adj_mat**2
adj3 = sparse_adj_mat**3
adj4 = sparse_adj_mat**4

new_data[:,len(rel_idx)] = degs_np
new_data[:,len(rel_idx)+1] = adj2.diagonal()
new_data[:,len(rel_idx)+2] = adj3.diagonal()
new_data[:,len(rel_idx)+3] = adj4.diagonal()


#new_data = (new_data - np.median(new_data, axis = 0)) / stats.median_abs_deviation(new_data, axis = 0, scale = "normal")
new_data = (new_data - np.mean(new_data, axis = 0)) / np.std(new_data, axis = 0)

#print(np.std(new_data, axis = 0))
#print(np.mean(new_data, axis = 0))
#xL_data = (xL_data - xL_data.min(0)) / xL_data.ptp(0)
#np.nan_to_num(xL_data, copy = False)
split_wholeG_torch.x = torch.tensor(new_data)

# extract the number of classes and features for model input
num_feat_dual_comb = len(rel_idx)+4
num_class_dual_comb = len(np.unique(LX_comb_einf_lab.y))

# assign the right classes 
#split_wholeG_torch.y = torch.tensor(split_wholeG_y)
train_mask_torch = np.arange(0, train_subG.order())
val_mask_torch = np.arange(train_subG.order(),train_subG.order()+ val_subG.order())
#test_mask_torch = np.arange(train_subG.order()+ val_subG.order(),train_subG.order()+ val_subG.order() + test_subG.order())



sweep_variable_layyer_num = {
    "name": sweep_name,
    "method": "bayes",
    "metric": {
        "name": "gcn/valacc",
        "goal": "maximize",
    },
    "parameters": {
        
        "models": {
            "values": ["GCN"]
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
            "values": [0.0,0.1,0.5]
        },
        "num_layers": {
            "values": [1,2,4,6]
        },
        "MLP": {
            "values": [True]
        },
        "skip": {
            "values": [True]
        }
    }
}


#sweep_variable_layyer_num = {
#    "name": sweep_name,
#    "method": "bayes",
#    "metric": {
#        "name": "gcn/max_accuracy",
#        "goal": "maximize",
#    },
#    "parameters": {
#        
#        "models": {
#            "values": ["SAGE"]
#        },
#        "hidden_channels": {
#            "values": [32, 64, 128]
#        },
#
#        "weight_decay": {
#            "distribution": "normal",
#            "mu": 2e-4,
#            "sigma": 8e-5,
#        },
#        "lr": {
#            "min": 1e-5,
#            "max": 1e-2
#        },
#        "dropout": {
#            "values": [0.5]
#        },
#        "num_layers": {
#            "values": [1,2,3,4,6]
#        },
#        "MLP": {
#            "values": [False]
#        },
#        "skip": {
#            "values": [False]
#        },
#         "norm": {
#            "values": [False]},
#        "aggr": {"values": ["max", "std", "mean", "sum", ["max", "std"], ["mean", "std"]]} 
#    }
#}
## "norm": {"values": [True, False]}
# "skip": {"values": [True, False]}
# "MLP": {"values": [True]}
# "aggr": {"values": ["max", "std", "mean", "sum"]} #"add", "sum" ,"mean", "min", "max" 1st batch
        


sweep_id = wandb.sweep(sweep_variable_layyer_num, project= "node_class_upsamp")

# define optimizer and loss
optimizerAdam = torch.optim.Adam
optimizerAdamW = torch.optim.AdamW
criterionCEL = torch.nn.CrossEntropyLoss

## set seed for reproducibility 
#np.random.seed(1234567)
#
## create the training and testing masks
#train_mask_random = np.random.choice(np.arange(0, geom_train_mask_torch.shape[0]), size= int(geom_train_mask_torch.shape[0]*0.85), replace = False)
#test_mask_random = np.delete(np.arange(0, geom_train_mask_torch.shape[0]), train_mask_random)


# convert to torch tensor objects
#train_mask_random= torch.tensor(train_mask_random)
#test_mask_random = torch.tensor(test_mask_random)

sweeper = nodeClassifier.nodeClassifierSweep(features = np.arange(num_feat_dual_comb), classes = num_class_dual_comb, optimizer = optimizerAdamW, lossFunc = criterionCEL, graph = split_wholeG_torch,  train_mask = train_mask_torch, val_mask = val_mask_torch, epochs = 500) #, test_mask= test_mask_torch

#Run the Sweeps agent
wandb.agent(sweep_id, project="node_class_upsamp", function=sweeper.agent_variable_size_model, count = 40)
