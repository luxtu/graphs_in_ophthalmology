import numpy as np 
import torch
from torch.nn import functional as F
from sklearn.preprocessing import MinMaxScaler


def max_cam_data(model, data, target_class, scale = False):
    """
    Assigns a 1 to all the nodes that participated in the prediction of the target class and a 0 to the rest.
    """
    model.eval()
    output = model(data.x_dict, data.edge_index_dict, data.batch_dict, data._slice_dict, training = True)

    if type(target_class) == int:
        target_class = [target_class] 

    heat_maps = []

    for c in target_class:
        output[:, c].backward(retain_graph=True)
        # Compute Grad-CAM
        node_heat_map = max_cam_heat_map(model)
        heat_maps.append(node_heat_map)

    if len(heat_maps) == 1:
        return heat_maps[0]
    else:  
        return heat_maps


def max_cam_heat_map(model):
    active_nodes_1 = torch.max(model.final_conv_grads_1, axis=0).indices.unique()[1:] # this is the mean across the features
    active_nodes_2 = torch.max(model.final_conv_grads_2, axis=0).indices.unique()[1:] # this is the mean across the features

    node_heat_map_1 = np.zeros(model.final_conv_acts_1.shape[0])
    node_heat_map_2 = np.zeros(model.final_conv_acts_2.shape[0])

    node_heat_map_1[active_nodes_1.cpu().detach().numpy()]=1
    node_heat_map_2[active_nodes_2.cpu().detach().numpy()]=1

    print(active_nodes_1)
    print(active_nodes_2)

    node_heat_map = {}
    #print(node_heat_map_1)
    node_heat_map["graph_1"] = torch.tensor(np.array(node_heat_map_1))
    node_heat_map["graph_2"] = torch.tensor(np.array(node_heat_map_2))

    return node_heat_map

def grad_cam_data(model, data, target_class, scale = False):
    # Perform forward pass for the target class
    model.eval()
    output = model(data.x_dict, data.edge_index_dict, data.batch_dict, data._slice_dict, training = True)

    if type(target_class) == int:
        target_class = [target_class] 

    heat_maps = []

    for c in target_class:
        output[:, c].backward(retain_graph=True)
        # Compute Grad-CAM
        node_heat_map = grad_cam_heat_map(model)
        heat_maps.append(node_heat_map)

    if scale:
        # Normalize heat maps across graphs and classes
        scaler = MinMaxScaler()
        min_scale = 1000000
        max_scale = -1
        for key in heat_maps[0].keys():
            for heat_map in heat_maps:
                scale = False
                if heat_map[key].max() > max_scale:
                    max_scale = heat_map[key].max()
                elif heat_map[key].min() < min_scale:
                    min_scale = heat_map[key].min()


        scaler.scale_ = np.array([1/(max_scale- min_scale)])
        scaler.min_ = np.array([-min_scale * scaler.scale_])


        for key in heat_maps[0].keys():
            for heat_map in heat_maps:

                heat_map[key] = torch.tensor(scaler.transform(heat_map[key].reshape(-1, 1)).reshape(-1))


    if len(heat_maps) == 1:
        return heat_maps[0]
    else:  
        return heat_maps


def grad_cam_heat_map(model):
    # Compute Grad-CAM
    alphas_1 = torch.mean(model.final_conv_grads_1, axis=0)
    alphas_2 = torch.mean(model.final_conv_grads_2, axis=0)
    node_heat_map_1 = []
    for n in range(model.final_conv_acts_1.shape[0]):
        node_heat = F.relu(alphas_1 @ model.final_conv_acts_1[n])#).item() F.relu(
        node_heat_map_1.append(node_heat.item())

    node_heat_map_2 = []
    for n in range(model.final_conv_acts_2.shape[0]):
        node_heat = F.relu(alphas_2 @ model.final_conv_acts_2[n])#).item() F.relu(
        node_heat_map_2.append(node_heat.item())

    node_heat_map = {}
    #print(node_heat_map_1)
    node_heat_map["graph_1"] = torch.tensor(np.array(node_heat_map_1))
    node_heat_map["graph_2"] = torch.tensor(np.array(node_heat_map_2))

    return node_heat_map