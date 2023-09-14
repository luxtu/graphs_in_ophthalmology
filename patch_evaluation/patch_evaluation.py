import numpy as np
import torch

def fix_outlist(outList):

    outlist_conv = []
    for i in outList:
        #print(i)
        try:
            outlist_conv.append([i[0]])
        except IndexError:
            outlist_conv.append(torch.tensor([[0.,0.]]))

    return outlist_conv


def get_rel_regions_for_id(id, last_id_list, last_region_list, outList):
    idx = np.where(np.array(last_id_list) == id)
    outList_rel = np.array(outList)[idx] 
    return np.array(last_region_list)[idx], outList_rel


def append_region_val_to_dict(rel_regions,outlist_rel, pos_dict, val_dict):
    
    ct = 0
    for region in rel_regions:
        x_min = region[0]
        x_max = region[1]
        y_min = region[2]
        y_max = region[3]
        for node, pos in pos_dict.items():

            if x_min < pos[0] < x_max and y_min < pos[1] < y_max:
                val_dict[node].append(outlist_rel[ct][0,1]-outlist_rel[ct][0,0]) # if the difference is large, the model is confident

        ct +=1


def extract_mean_from_dict(node_dict_val):
    mean_vals = []
    for node, val in node_dict_val.items():
        if len(val) == 0:
            mean_vals.append(0)
        else:
            mean_vals.append(np.mean(val))

    mean_vals = np.array(mean_vals)
    mean_vals = mean_vals.reshape(-1,1)

    return mean_vals


def nodewise_mean_val(dataloader, node_id, id_list, region_list, pred_list):

    pred_list = fix_outlist(pred_list)
    node_dict_pos = dict(zip(np.arange(0, dataloader.line_data[node_id].num_nodes), np.array(dataloader.line_data[node_id].edge_pos[:,:-1].cpu().detach().numpy())))
    node_dict_val = dict(zip(np.arange(0, dataloader.line_data[node_id].num_nodes), ([] for _ in np.arange(0, dataloader.line_data[node_id].num_nodes))))

    # extract all regions that were created for a graph
    rel_regions, rel_vals = get_rel_regions_for_id(node_id, id_list, region_list, pred_list)
    # extract the values for the regions and append to the fitting nodes
    append_region_val_to_dict(rel_regions,rel_vals, node_dict_pos, node_dict_val)

    # get the mean of the values for each node and append to the data object as a new feature
    mean_vals = extract_mean_from_dict(node_dict_val)

    return mean_vals