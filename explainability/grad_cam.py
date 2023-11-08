import numpy as np 
import torch
from torch.nn import functional as F
from sklearn.preprocessing import MinMaxScaler



def grad_cam_data(model, data, target_class, scale = False, relu = False, start = False, abs = False):
    # Perform forward pass for the target class
    model.eval()
    pos_dict = {}
    for key in ["graph_1", "graph_2"]:
        pos_dict[key] = data[key].pos
    output = model(data.x_dict, data.edge_index_dict, data.batch_dict,  grads = True, pos_dict = pos_dict) #data._slice_dict #training = True,

    if type(target_class) == int:
        target_class = [target_class] 

    heat_maps = []

    for c in target_class:
        output[:, c].backward(retain_graph=True)
        # Compute Grad-CAM
        node_heat_map = grad_cam_heat_map(model, relu = relu, start = start) # .gnn1_embed
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


def grad_cam_heat_map(model, start = False, relu = False, abs = False):
    # Compute Grad-CAM

    try:
        model = model.gnn1_embed
    except AttributeError:
        pass

    if start == True:
        conv_grads_1 = model.start_conv_grads_1
        conv_grads_2 = model.start_conv_grads_2
        conv_acts_1 = model.start_conv_acts_1
        conv_acts_2 = model.start_conv_acts_2
    else:
        conv_grads_1 = model.final_conv_grads_1
        conv_grads_2 = model.final_conv_grads_2
        conv_acts_1 = model.final_conv_acts_1
        conv_acts_2 = model.final_conv_acts_2

    # abs value of the gradients

    if abs:
        conv_grads_1 = torch.abs(conv_grads_1)
        conv_grads_2 = torch.abs(conv_grads_2)

    alphas_1 = torch.mean(conv_grads_1, axis=0)
    alphas_2 = torch.mean(conv_grads_2, axis=0)
    node_heat_map_1 = []
    for n in range(conv_acts_1.shape[0]):
        node_heat = alphas_1 @ conv_acts_1[n]#).item() F.relu(
        if relu:
            node_heat = F.relu(node_heat)
        node_heat_map_1.append(node_heat.item())

    node_heat_map_2 = []
    for n in range(conv_acts_2.shape[0]):
        node_heat = alphas_2 @ conv_acts_2[n]#).item() F.relu(
        if relu:
            node_heat = F.relu(node_heat)

        node_heat_map_2.append(node_heat.item())

    node_heat_map = {}
    #print(node_heat_map_1)
    node_heat_map["graph_1"] = torch.tensor(np.array(node_heat_map_1))
    node_heat_map["graph_2"] = torch.tensor(np.array(node_heat_map_2))

    return node_heat_map

