

def _calculate_adaptive_node_threshold(explanation_graph, explained_gradient = 0.95):
    """
    Calculate the threshold for the node importance such that 95% of the total gradient is explained (without the gradients from nodes that contribute less than 0.05% to the total gradient)
    """
    import torch
    import copy

    abs_dict = {}
    for key in explanation_graph.node_mask_dict.keys():
        abs_dict[key] = copy.deepcopy(torch.abs(explanation_graph.node_mask_dict[key]))

    total_grad = 0
    for key in explanation_graph.node_mask_dict.keys():
        total_grad += abs_dict[key].abs().sum()
    node_value = torch.cat([abs_dict[key].abs().sum(dim=-1) for key in explanation_graph.node_mask_dict.keys()], dim=0)
    sorted_node_value = torch.sort(node_value, descending=True)[0]
    # get rid of the nodes that contribute less than 0.05% to the total gradient
    sorted_node_value = sorted_node_value[sorted_node_value > 0.0005 * total_grad]
    cum_sum = torch.cumsum(sorted_node_value, dim=0)
    cropped_grad = cum_sum[-1]
    threshold = sorted_node_value[cum_sum < explained_gradient * cropped_grad][-1] # 30


    return threshold

def _calculate_adaptive_node_threshold_pos(explanation_graph, explained_gradient = 0.95):
    """
    Calculate the threshold for the node importance such that 95% of the total gradient is explained (without the gradients from nodes that contribute less than 0.05% to the total gradient)
    Only consider the positive gradients
    """
    import torch
    import copy

    abs_dict = {}
    for key in explanation_graph.node_mask_dict.keys():
        abs_dict[key] = copy.deepcopy(explanation_graph.node_mask_dict[key])
    
    total_pos_grad = 0
    for key in explanation_graph.node_mask_dict.keys():
        grad = abs_dict[key].sum(dim=-1)
        grad[grad < 0] = 0  
        total_pos_grad += grad.sum()
    node_value = torch.cat([abs_dict[key].sum(dim=-1) for key in explanation_graph.node_mask_dict.keys()], dim=0)
    node_value[node_value < 0] = 0
    sorted_node_value = torch.sort(node_value, descending=True)[0]
    # get rid of the nodes that contribute less than 0.05% to the total gradient
    sorted_node_value = sorted_node_value[sorted_node_value > 0.0005 * total_pos_grad]
    cum_sum = torch.cumsum(sorted_node_value, dim=0)
    cropped_grad = cum_sum[-1]
    threshold = sorted_node_value[cum_sum < explained_gradient * cropped_grad][-1] # 30
            
    return threshold
        
    

def _remove_inlay(hetero_graph, work_dict, node_key):
    """
    Remove the inlay from the node masks
    """

    # remove the inlay
    pos = hetero_graph[node_key].pos.cpu().detach().numpy()
    remove_pos = (pos[:,0] > 1100) & (pos[:,1] < 100)
    # again setting 0 does not affect the sum
    work_dict[node_key][remove_pos, :] = 0

    return work_dict

def _feature_importance_plot_each_type(explanation, path, features_label_dict, score, num_features = 5, only_positive = False):

    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    node_typeto_proper_name = {"graph_1": "Vessel", "graph_2": "ICP Region", "faz": "FAZ"}
    all_feat_labels = []
    for node_type in explanation.node_mask_dict.keys():
        all_feat_labels += [
            f'{node_typeto_proper_name[node_type]}: {label}' for label in features_label_dict[node_type]
        ]

    # three subplots below each other
    fig, axs = plt.subplots(3, 1, figsize=(10, 7))
    # get the 5 most important features for each graph and plot them below each other

    # get the max score for all graphs
    max_score = 0
    for node_type in explanation.node_mask_dict.keys():
        max_score = max(max_score, np.abs(score[node_type]).max())

#    max_score = np.abs(score).max()

    for i, node_type in enumerate(explanation.node_mask_dict.keys()):
        score_indv = score[node_type]
        df = pd.DataFrame({'score': score_indv}, index=features_label_dict[node_type])
        if only_positive:
            df_sorted = df.reindex(df.sort_values('score', ascending=False).index)
        else:
            df_sorted = df.reindex(df.abs().sort_values('score', ascending=False).index)
        df_sorted = df_sorted.head(num_features)
        # use the same xlim for all subplots
        # get rid of the legend
        ax_out = df_sorted.plot(
            kind='barh',
            title=node_typeto_proper_name[node_type],
            ylabel='Feature label',
            xlim= [-max_score - 0.1*max_score, max_score+ 0.1*max_score],
            legend=False,
            ax = axs[i]
        )
        ax_out.invert_yaxis()
    #plt.gca().invert_yaxis()
    #ax.bar_label(container=ax.containers[0], label_type='edge')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _feature_importance_plot(explanation, path, features_label_dict, score, num_features, only_positive):

    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    node_typeto_proper_name = {"graph_1": "Vessel", "graph_2": "ICP Region", "faz": "FAZ"}
    node_typeto_color_char = {"graph_1": "#CC0000", "graph_2": "#0075E8", "faz": "#007E01"}  #  #004080
    all_feat_labels = []
    for node_type in explanation.node_mask_dict.keys():
        all_feat_labels += [
            f'{node_typeto_proper_name[node_type]}: {label}' for label in features_label_dict[node_type]
        ]
    # also change the names in the features_label_dict
    features_label_dict = {key: [f'{node_typeto_proper_name[key]}: {label}' for label in features_label_dict[key]] for key in features_label_dict.keys()}
    

    # single plot
    fig, ax = plt.subplots(figsize=(10, 7))
    # get the 5 most important features for each graph and plot them below each other

    # get the max score for all graphs
    max_score = 0
    for node_type in explanation.node_mask_dict.keys():
        max_score = max(max_score, np.abs(score[node_type]).max())

    # plot the feature importance of all graphs in one plot
    # get the scores for each graph
    score_all = np.concatenate([score[node_type] for node_type in explanation.node_mask_dict.keys()])
    df = pd.DataFrame({'score': score_all}, index=all_feat_labels)

    if only_positive:
        df_sorted = df.reindex(df.sort_values('score', ascending=False).index)
        xlim = [0, max_score + 0.1*max_score]
    else:
        df_sorted = df.reindex(df.abs().sort_values('score', ascending=False).index)
        xlim = [-max_score - 0.1*max_score, max_score+ 0.1*max_score]
    df_sorted = df_sorted.head(num_features)

    color_list = []
    label_list = []
    # get the top features, assign them a color according to the node type
    for feature in df_sorted.index:
        for node_type in explanation.node_mask_dict.keys():
            if feature in features_label_dict[node_type]:
                color_list.append(node_typeto_color_char[node_type])
                label_list.append(node_typeto_proper_name[node_type])
                break

    #print(color_list)
    # use the same xlim for all subplots
    # get rid of the legend
    fig, ax_out = plt.subplots(figsize=(10, 7))
    ax_out.barh(df_sorted.index, df_sorted['score'], color = color_list)

    #ax_out = df_sorted.plot(
    #    kind='barh',
    #    title='Feature importance',
    #    ylabel='Feature label',
    #    xlim= xlim,
    #    legend=True,
    #    ax = ax,
    #    color = color_list,
    #    label = label_list
    #)
    ax_out.invert_yaxis()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _feature_importance_boxplot(hetero_graph, path, features_label_dict, het_graph_rel_pos_dict, score, num_features , only_positive):
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    # change the path by splitting it and adding _boxplot, split at the last dot
    path = path.rsplit(".", 1)[0] + "_boxplot." + path.rsplit(".", 1)[1]

    node_typeto_proper_name = {"graph_1": "Vessel", "graph_2": "ICP Area", "faz": "FAZ"}
    all_feat_labels = []
    for node_type in hetero_graph.x_dict.keys():
        all_feat_labels += [
            f'{node_typeto_proper_name[node_type]}: {label}' for label in features_label_dict[node_type]
        ]
    # change also the names in the features_label_dict
    features_label_dict = {key: [f'{node_typeto_proper_name[key]}: {label}' for label in features_label_dict[key]] for key in features_label_dict.keys()}

    # single plot
    fig, axs = plt.subplots(figsize=(10, 7))
    # get the most important features across all graphs
    max_score = 0
    for node_type in hetero_graph.x_dict.keys():
        max_score = max(max_score, np.abs(score[node_type]).max())

    # plot the feature importance of all graphs in one plot
    # get the scores for each graph
    score_all = np.concatenate([score[node_type] for node_type in hetero_graph.x_dict.keys()])
    df = pd.DataFrame({'score': score_all}, index=all_feat_labels)

    if only_positive:
        df_sorted = df.reindex(df.sort_values('score', ascending=False).index)
    else:
        df_sorted = df.reindex(df.abs().sort_values('score', ascending=False).index)


    # for each of the top features, get the values in the graph of the relevant nodes
    top_features = df_sorted.head(num_features).index
    feature_values = []
    for feature in top_features:
        for node_type in hetero_graph.x_dict.keys():
            try:
                feature_values.append(hetero_graph[node_type].x[het_graph_rel_pos_dict[node_type]][:, features_label_dict[node_type].index(feature)].cpu().detach().numpy())
            except ValueError:
                continue

    #df = pd.DataFrame(feature_values)
    # handle the case that there are different number of values for each feature, but there should still be a boxplot created
    fig, ax = plt.subplots()
    ax.boxplot(feature_values, vert = False)
    # invert the y axis
    ax.invert_yaxis()
    ax.set_title('Feature importance')
    ax.set_yticklabels(top_features)


    plt.tight_layout()
    plt.savefig(path)
    plt.close()


    

def _feature_importance_boxplot_each_type(hetero_graph, path, features_label_dict, het_graph_rel_pos_dict, score, num_features, only_positive):
        
    import matplotlib.pyplot as plt
    import pandas as pd

    # change the path by splitting it and adding _boxplot, split at the last dot
    path = path.rsplit(".", 1)[0] + "_boxplot." + path.rsplit(".", 1)[1]

    node_typeto_proper_name = {"graph_1": "Vessel", "graph_2": "ICP Area", "faz": "FAZ"}
    all_feat_labels = []
    for node_type in hetero_graph.x_dict.keys():
        all_feat_labels += [
            f'{node_typeto_proper_name[node_type]}: {label}' for label in features_label_dict[node_type]
        ]

    # three subplots below each other
    fig, axs = plt.subplots(3, 1, figsize=(10, 7))
    # get the 5 most important features for each graph and plot them below each other

    top_feature_dict = {}
    for i, node_type in enumerate(hetero_graph.x_dict.keys()):
        score_indv = score[node_type]
        df = pd.DataFrame({'score': score_indv}, index=features_label_dict[node_type])
        if only_positive:
            df_sorted = df.reindex(df.sort_values('score', ascending=False).index)
        else:
            df_sorted = df.reindex(df.abs().sort_values('score', ascending=False).index)
        df_sorted = df_sorted.head(num_features)
        top_feature_dict[node_type] = df_sorted.index
    
    # for each of the top features, get the values in the graph of the relevant nodes
    for i, node_type in enumerate(hetero_graph.x_dict.keys()):
        # get the top features
        top_features = top_feature_dict[node_type]
        feature_values = {}
        for feature in top_features:
            feature_values[feature] = hetero_graph[node_type].x[het_graph_rel_pos_dict[node_type]][:, features_label_dict[node_type].index(feature)].cpu().detach().numpy()
        df = pd.DataFrame(feature_values)
        ax_out = df.boxplot(ax = axs[i], vert = False)
        axs[i].set_title(node_typeto_proper_name[node_type])

        ax_out.invert_yaxis()

    plt.tight_layout()
    plt.savefig(path)
    plt.close()



    

def store_relevant_nodes_csv(explanation_graph, hetero_graph, path, features_label_dict, faz_node = False, explained_gradient = 0.95):

    import pandas as pd
    het_graph_rel_pos_dict = identifiy_relevant_nodes(explanation_graph, hetero_graph, explained_gradient = explained_gradient, faz_node = faz_node, )

    data_label = hetero_graph.graph_id[0]
    # get the features and gradients of the relevant nodes
    feature_dict = {}
    grad_dict = {}
    for key in explanation_graph.node_mask_dict.keys():
        feature_dict[key] = hetero_graph[key].x[het_graph_rel_pos_dict[key]]
        grad_dict[key] = explanation_graph.node_mask_dict[key][het_graph_rel_pos_dict[key]]
    # get the gradients of the relevant nodes


    # write the relevant nodes to a csv file
    for key in explanation_graph.node_mask_dict.keys():
        df = pd.DataFrame(feature_dict[key].cpu().detach().numpy(), columns = features_label_dict[key])
        df.to_csv(path + f"relevant_nodes_{key}_{data_label}_features.csv", index=False)
        df = pd.DataFrame(grad_dict[key].cpu().detach().numpy(), columns = features_label_dict[key])
        df.to_csv(path + f"relevant_nodes__{key}_{data_label}_gradients.csv", index=False)

    

    



def identifiy_relevant_nodes(explanation_graph, hetero_graph, explained_gradient, faz_node = False, only_positive = False):

    import numpy as np
        
    if only_positive:
        threshold = _calculate_adaptive_node_threshold_pos(explanation_graph, explained_gradient = explained_gradient)
        if threshold < 0:
            threshold = 0
    else:
        threshold = _calculate_adaptive_node_threshold(explanation_graph, explained_gradient = explained_gradient)

    graph_names = ["graph_1", "graph_2"]
    if faz_node:
        graph_names.append("faz")
    
    het_graph_rel_pos_dict = {}
    for graph_name in graph_names:
        #filter the nodes that are above the threshold
        if only_positive:
            het_graph_rel_pos = explanation_graph.node_mask_dict[graph_name].sum(dim=-1) > threshold
        else:
            het_graph_rel_pos = explanation_graph.node_mask_dict[graph_name].abs().sum(dim=-1) > threshold

        het_graph_rel_pos = het_graph_rel_pos.cpu().detach().numpy()
        # data should not contain this information anymore
        # remove the inlay
        # pos to cpu and detach
        het_graph_pos = hetero_graph[graph_name].pos.cpu().detach().numpy()
        het_graph_rel_pos = het_graph_rel_pos & (het_graph_pos[:,0] < 1100) & (het_graph_pos[:,1] > 100)
        het_graph_rel_pos_dict[graph_name] = het_graph_rel_pos
    return het_graph_rel_pos_dict



def top_k_important_nodes(explanation_graph, hetero_graph, faz_node = False ,top_k = 100, only_positive = False, each_type = False):
    """
    Returns the gradients of the top k most important nodes for each graph or across all graphs
    """
    if each_type:
        return _top_k_important_nodes_all_types(explanation_graph, hetero_graph, faz_node = faz_node ,top_k = top_k, only_positive = only_positive)
    else:
        return _top_k_important_nodes(explanation_graph, hetero_graph, faz_node = faz_node ,top_k = top_k, only_positive = only_positive)



def _top_k_important_nodes(explanation_graph, hetero_graph, faz_node = False ,top_k = 100, only_positive = False):
    """
    Returns the gradients of the top k most important nodes across all graphs
    """
    import numpy as np
    import torch
    import copy

    graph_names = ["graph_1", "graph_2"]
    if faz_node:
        graph_names.append("faz")

    het_graph_rel_pos_dict = {}
    # find the threshold that only includes the top k nodes
    
    for graph_name in graph_names:
        #filter the nodes that are above the threshold
        if only_positive:
            het_graph_weight = explanation_graph.node_mask_dict[graph_name].sum(dim=-1)
        else:
            het_graph_weight = explanation_graph.node_mask_dict[graph_name].abs().sum(dim=-1)
        het_graph_weight = het_graph_weight.cpu().detach().numpy()
        inlay_pos = (hetero_graph[graph_name].pos.cpu().detach().numpy()[:,0] > 1100) & (hetero_graph[graph_name].pos.cpu().detach().numpy()[:,1] < 100)
        het_graph_weight[inlay_pos] = -5
    
        #concat the weights
        if graph_name == "graph_1":
            all_weights = het_graph_weight
        else:
            all_weights = np.concatenate((all_weights, het_graph_weight))
        
    # set the threshold to the kth largest value
    threshold = np.sort(all_weights)[-top_k]
    if threshold < 0:
        threshold = 0 + 1e-6
    print("threshold", threshold)
    for graph_name in graph_names:
        #filter the nodes that are above the threshold
        if only_positive:
            het_graph_rel_pos = explanation_graph.node_mask_dict[graph_name].sum(dim=-1) >= threshold
        else:
            het_graph_rel_pos = explanation_graph.node_mask_dict[graph_name].abs().sum(dim=-1) >= threshold
        het_graph_rel_pos = het_graph_rel_pos.cpu().detach().numpy()
        # print the number of relevant nodes
        # remove the inlay
        # het_graph_rel_pos = het_graph_rel_pos & (hetero_graph[graph_name].pos.cpu().detach().numpy()[:,0] < 1100) & (hetero_graph[graph_name].pos.cpu().detach().numpy()[:,1] > 100)
        print(f"{graph_name}: {het_graph_rel_pos.sum()}")
        het_graph_rel_pos_dict[graph_name] = het_graph_rel_pos
    
    return het_graph_rel_pos_dict
                    

        
def _top_k_important_nodes_all_types(explanation_graph, hetero_graph, faz_node = False, top_k = 100, only_positive = False):
    """
    Returns the gradients of the top k most important nodes for each graph
    """
    import numpy as np
    import torch
    import copy
        
    graph_names = ["graph_1", "graph_2"]
    if faz_node:
        graph_names.append("faz")
    
    het_graph_rel_pos_dict = {}
    het_graph_rel_weight_dict = {}
    for graph_name in graph_names:
        #filter the nodes that are above the threshold
        if only_positive:
            het_graph_weight = explanation_graph.node_mask_dict[graph_name].sum(dim=-1)
        else:
            het_graph_weight = explanation_graph.node_mask_dict[graph_name].abs().sum(dim=-1)
        # set the weights of the inlay to 0
        het_graph_weight = het_graph_weight.cpu().detach().numpy()
        inlay_pos = (hetero_graph[graph_name].pos.cpu().detach().numpy()[:,0] > 1100) & (hetero_graph[graph_name].pos.cpu().detach().numpy()[:,1] < 100)
        het_graph_weight[inlay_pos] = 0
        het_graph_weight = torch.from_numpy(het_graph_weight)
        if graph_name == "faz":
            _, indices = torch.topk(het_graph_weight, 1)
            het_graph_rel_pos = torch.zeros(het_graph_weight.shape, dtype = torch.bool)
            het_graph_rel_pos[indices] = True
            if only_positive:
                het_graph_rel_pos = het_graph_rel_pos & (het_graph_weight > 0)
            het_graph_rel_pos = het_graph_rel_pos.cpu().detach().numpy()
        else:
            _, indices = torch.topk(het_graph_weight, top_k)
            het_graph_rel_pos = torch.zeros(het_graph_weight.shape, dtype = torch.bool)
            het_graph_rel_pos[indices] = True
            if only_positive:
                het_graph_rel_pos = het_graph_rel_pos & (het_graph_weight > 0)
            het_graph_rel_pos = het_graph_rel_pos.cpu().detach().numpy()
        het_graph_rel_pos_dict[graph_name] = het_graph_rel_pos
        het_graph_rel_weight_dict[graph_name] = het_graph_weight.cpu().detach().numpy()
        

    return het_graph_rel_pos_dict

    


def adaptive_important_nodes(explanation_graph, hetero_graph):
    """
    Returns the gradients of the nodes that are above the threshold for each graph, and features of these nodes
    """
    import torch
    import copy
    abs_dict = {}
    grad_dict = {}
    feature_dict = {}
    for key in explanation_graph.node_mask_dict.keys():
        abs_dict[key] = copy.deepcopy(torch.abs(explanation_graph.node_mask_dict[key]))

    threshold = _calculate_adaptive_node_threshold(explanation_graph)
    for node_key in ["graph_1", "graph_2"]:
        abs_dict = _remove_inlay(hetero_graph, abs_dict, node_key)

    for node_key in explanation_graph.node_mask_dict.keys():
        importances = abs_dict[node_key].sum(dim=-1)
        # get nodes that are above the threshold
        relevant_nodes = importances > threshold
        # get the gradients for the relevant nodes
        grad_dict[node_key] = explanation_graph.node_mask_dict[node_key][relevant_nodes]
        # get the features for the relevant nodes
        feature_dict[node_key] = hetero_graph[node_key].x[relevant_nodes]

    return grad_dict, feature_dict

def visualize_feature_importance(explanation, hetero_graph, path, features_label_dict, explained_gradient = None, only_positive = False, with_boxplot = False, num_features = 5, each_type = True):

    if each_type:
        _visualize_feature_importance_each_type(explanation, hetero_graph, path, features_label_dict, explained_gradient, only_positive, with_boxplot, num_features)
    else:
        _visualize_feature_importance(explanation, hetero_graph, path, features_label_dict, explained_gradient, only_positive, with_boxplot, num_features)



def _visualize_feature_importance(explanation, hetero_graph, path, features_label_dict, explained_gradient, only_positive, with_boxplot, num_features):
    """
    Wrapper for the pytorch geom function, since it does not include tight layout.
    If there is no threshold, all the nodes are considered for the importance score.
    If threshold is the string "adaptive", the threshold is calculated such that 95% of the total gradient is explained (without the gradients from nodes that contribute less than 0.05% to the total gradient)
    """
    import torch
    import copy

    # take abs for features in node mask to take into account negative values
    abs_dict = {}
    work_dict = {}
    for key in explanation.node_mask_dict.keys():
        if only_positive:
            abs_dict[key] = copy.deepcopy(explanation.node_mask_dict[key])
            abs_dict[key][abs_dict[key] < 0] = 0
        else:
            abs_dict[key] = copy.deepcopy(torch.abs(explanation.node_mask_dict[key]))
        work_dict[key] = copy.deepcopy(explanation.node_mask_dict[key])

    if explained_gradient is None:

        # make the score a dictionary with the keys being the graph names
        # score is the sum of the absolute values of the gradients for each feature for each nodetype
        score = {}
        for key in explanation.node_mask_dict.keys():
            score[key] = work_dict[key].sum(dim=0).cpu().numpy()

        het_graph_rel_pos_dict = {}
        
        for key in explanation.node_mask_dict.keys():
            het_graph_rel_pos_dict[key] = torch.ones(explanation.node_mask_dict[key].sum(dim=-1).shape, dtype = torch.bool).cpu().detach().numpy()
    elif isinstance(explained_gradient, float):
        faz_node = False
        if "faz" in explanation.node_mask_dict.keys():
            faz_node = True
        het_graph_rel_pos_dict = identifiy_relevant_nodes(explanation, hetero_graph, explained_gradient = explained_gradient, faz_node = faz_node, only_positive = only_positive)

        # print number of relevant nodes
        for key in explanation.node_mask_dict.keys():
            print(f"{key}: {het_graph_rel_pos_dict[key].sum()}")
        score = {}
        for key in explanation.node_mask_dict.keys():
            score[key] = work_dict[key][het_graph_rel_pos_dict[key]].sum(dim=0).cpu().numpy()
        
    elif isinstance(explained_gradient, int):
        faz_node = False
        if "faz" in explanation.node_mask_dict.keys():
            faz_node = True
        het_graph_rel_pos_dict = top_k_important_nodes(explanation, hetero_graph, faz_node = faz_node,top_k = explained_gradient, only_positive = only_positive)

        # print number of relevant nodes
        for key in explanation.node_mask_dict.keys():
            print(f"{key}: {het_graph_rel_pos_dict[key].sum()}")
        score = {}
        for key in explanation.node_mask_dict.keys():
            score[key] = work_dict[key][het_graph_rel_pos_dict[key]].sum(dim=0).cpu().numpy()


    _feature_importance_plot(explanation, path, features_label_dict, score, num_features=num_features, only_positive = only_positive)

    if with_boxplot:
        _feature_importance_boxplot(hetero_graph, path, features_label_dict, het_graph_rel_pos_dict, score, num_features=num_features, only_positive = only_positive)


    


def _visualize_feature_importance_each_type(explanation, hetero_graph, path, features_label_dict, explained_gradient, only_positive, with_boxplot, num_features):
    """
    Wrapper for the pytorch geom function, since it does not include tight layout.
    If there is no threshold, all the nodes are considered for the importance score.
    If threshold is the string "adaptive", the threshold is calculated such that 95% of the total gradient is explained (without the gradients from nodes that contribute less than 0.05% to the total gradient)
    """
    import torch
    import copy

    # take abs for features in node mask to take into account negative values
    abs_dict = {}
    work_dict = {}
    for key in explanation.node_mask_dict.keys():
        if only_positive:
            abs_dict[key] = copy.deepcopy(explanation.node_mask_dict[key])
            abs_dict[key][abs_dict[key] < 0] = 0
        else:
            abs_dict[key] = copy.deepcopy(torch.abs(explanation.node_mask_dict[key]))
        work_dict[key] = copy.deepcopy(explanation.node_mask_dict[key])

    if explained_gradient is None:

        # make the score a dictionary with the keys being the graph names
        # score is the sum of the absolute values of the gradients for each feature for each nodetype
        score = {}
        for key in explanation.node_mask_dict.keys():
            score[key] = work_dict[key].sum(dim=0).cpu().numpy()

        het_graph_rel_pos_dict = {}
        
        for key in explanation.node_mask_dict.keys():
            het_graph_rel_pos_dict[key] = torch.ones(explanation.node_mask_dict[key].sum(dim=-1).shape, dtype = torch.bool).cpu().detach().numpy()
    elif isinstance(explained_gradient, float):
        faz_node = False
        if "faz" in explanation.node_mask_dict.keys():
            faz_node = True
        het_graph_rel_pos_dict = identifiy_relevant_nodes(explanation, hetero_graph, explained_gradient = explained_gradient, faz_node = faz_node, only_positive = only_positive)

        # print number of relevant nodes
        for key in explanation.node_mask_dict.keys():
            print(f"{key}: {het_graph_rel_pos_dict[key].sum()}")
        score = {}
        for key in explanation.node_mask_dict.keys():
            score[key] = work_dict[key][het_graph_rel_pos_dict[key]].sum(dim=0).cpu().numpy()
        
    elif isinstance(explained_gradient, int):
        faz_node = False
        if "faz" in explanation.node_mask_dict.keys():
            faz_node = True
        het_graph_rel_pos_dict = top_k_important_nodes(explanation, hetero_graph, faz_node = faz_node,top_k = explained_gradient, only_positive = only_positive)

        # print number of relevant nodes
        for key in explanation.node_mask_dict.keys():
            print(f"{key}: {het_graph_rel_pos_dict[key].sum()}")
        score = {}
        for key in explanation.node_mask_dict.keys():
            score[key] = work_dict[key][het_graph_rel_pos_dict[key]].sum(dim=0).cpu().numpy()


    _feature_importance_plot_each_type(explanation, path, features_label_dict, score, num_features=num_features, only_positive = only_positive)

    if with_boxplot:
        _feature_importance_boxplot_each_type(hetero_graph, path, features_label_dict, het_graph_rel_pos_dict, score, num_features=num_features, only_positive = only_positive)





def visualize_relevant_subgraph(explanation_graph, hetero_graph, path, threshold = 0.0, edge_threshold = 0.0, edges = True, faz_node = False, ax = None):

    import matplotlib.pyplot as plt
    import pandas as pd
    import torch
    import numpy as np

    graph_1_name = "graph_1"
    graph_2_name = "graph_2"
    graph_3_name = "faz"

    graph_names = [graph_1_name, graph_2_name]
    if faz_node:
        graph_names.append(graph_3_name)

    if ax is None:
        fig, ax = plt.subplots()
        side_l = 5.5
        fig.set_figwidth(side_l)
        fig.set_figheight(side_l)
        plt.ylim(1216,0)
        plt.xlim(0,1216)
        ax = [ax]
    else:
        # check if the ax is a list or np array
        if isinstance(ax, list) or isinstance(ax, np.ndarray):
            ax = ax
        else:
            ax = [ax]

    het_graph_pos = {}
    for graph_name in graph_names:
        het_graph_pos[graph_name] = hetero_graph[graph_name].pos.cpu().detach().numpy()

    #het_graph1_pos = hetero_graph[graph_1_name].pos.cpu().detach().numpy()
    #het_graph2_pos = hetero_graph[graph_2_name].pos.cpu().detach().numpy()
    #if faz_node:
    #    het_graph3_pos = hetero_graph[graph_3_name].pos.cpu().detach().numpy()


    if threshold == "adaptive":

        threshold = _calculate_adaptive_node_threshold(explanation_graph)
    
    if edges:
        if edge_threshold == "adaptive":

            total_grad = explanation_graph.edge_mask_dict[graph_1_name, "to", graph_1_name].abs().sum() + explanation_graph.edge_mask_dict[graph_2_name, "to", graph_2_name].abs().sum() + explanation_graph.edge_mask_dict[graph_1_name, "to", graph_2_name].abs().sum() + explanation_graph.edge_mask_dict[graph_2_name, "rev_to", graph_1_name].abs().sum()
            #avg_grad = total_grad / (explanation_graph.edge_mask_dict[graph_1_name, "to", graph_1_name].shape[0] + explanation_graph.edge_mask_dict[graph_2_name, "to", graph_2_name].shape[0] + explanation_graph.edge_mask_dict[graph_1_name, "to", graph_2_name].shape[0] + explanation_graph.edge_mask_dict[graph_2_name, "rev_to", graph_1_name].shape[0])

            if faz_node:
                total_grad += explanation_graph.edge_mask_dict[graph_1_name, "to", graph_3_name].abs().sum() + explanation_graph.edge_mask_dict[graph_3_name, "rev_to", graph_1_name].abs().sum() + explanation_graph.edge_mask_dict[graph_2_name, "rev_to", graph_3_name].abs().sum() + explanation_graph.edge_mask_dict[graph_3_name, "to", graph_2_name].abs().sum()

            edge_value = torch.cat((explanation_graph.edge_mask_dict[graph_1_name, "to", graph_1_name].abs(), explanation_graph.edge_mask_dict[graph_2_name, "to", graph_2_name].abs(), explanation_graph.edge_mask_dict[graph_1_name, "to", graph_2_name].abs(), explanation_graph.edge_mask_dict[graph_2_name, "rev_to", graph_1_name].abs()))

            if faz_node:
                edge_value = torch.cat((explanation_graph.edge_mask_dict[graph_1_name, "to", graph_3_name].abs(), explanation_graph.edge_mask_dict[graph_3_name, "rev_to", graph_1_name].abs(), explanation_graph.edge_mask_dict[graph_2_name, "rev_to", graph_3_name].abs(), explanation_graph.edge_mask_dict[graph_3_name, "to", graph_2_name].abs(), edge_value))

            sorted_edge_value = torch.sort(edge_value, descending=True)[0]
            # get rid of the edges that contribute less than 0.033% to the total gradient
            sorted_edge_value = sorted_edge_value[sorted_edge_value > 0.00033 * total_grad]

            cum_sum = torch.cumsum(sorted_edge_value, dim=0)
            cropped_grad = cum_sum[-1]
            edge_threshold = sorted_edge_value[cum_sum < 0.95 * total_grad][-1] # 0.95, 

            #print("edge_threshold", avg_grad)
            #edge_threshold = max(avg_grad, 0.05*threshold) # max_val * 0.01
            #print("edge_threshold", edge_threshold)

        elif edge_threshold == "node_threshold":
            edge_threshold = threshold*0.01

    het_graph_rel_pos_dict = {}

    s_list = [8, 12, 12]
    marker_list = ["o", "s", "D"]
    colors = ["blue", "orange", "red"]

    for idx, graph_name in enumerate(graph_names):
        #filter the nodes that are above the threshold
        het_graph_rel_pos = explanation_graph.node_mask_dict[graph_name].abs().sum(dim=-1) > threshold
        het_graph_rel_pos = het_graph_rel_pos.cpu().detach().numpy()
        # remove the inlay
        het_graph_rel_pos = het_graph_rel_pos & (het_graph_pos[graph_name][:,0] < 1100) & (het_graph_pos[graph_name][:,1] > 100)
        het_graph_rel_pos_dict[graph_name] = het_graph_pos[graph_name][het_graph_rel_pos]

        for ax_ax in ax:
            ax_ax.scatter(het_graph_rel_pos_dict[graph_name][:,1], het_graph_rel_pos_dict[graph_name][:,0], zorder = 2, s = s_list[idx], alpha= 0.9, marker = marker_list[idx], c = colors[idx])

        

    if edges:

        het_graph1_rel_edges = explanation_graph.edge_mask_dict[graph_1_name, "to", graph_1_name] > edge_threshold
        het_graph2_rel_edges = explanation_graph.edge_mask_dict[graph_2_name, "to", graph_2_name] > edge_threshold

        het_graph1_rel_edges = het_graph1_rel_edges.cpu().detach().numpy()
        het_graph2_rel_edges = het_graph2_rel_edges.cpu().detach().numpy()

        print("het_graph1_rel_edges", het_graph1_rel_edges.sum())
        print("het_graph2_rel_edges", het_graph2_rel_edges.sum())

        het_graph12_rel_edges = explanation_graph.edge_mask_dict[graph_1_name, "to", graph_2_name] > edge_threshold
        het_graph21_rel_edges = explanation_graph.edge_mask_dict[graph_2_name, "rev_to", graph_1_name] > edge_threshold

        #if faz_node:
        #    het_graph13_rel_edges = explanation_graph.edge_mask_dict[graph_1_name, "to", graph_3_name] > edge_threshold
        #    het_graph31_rel_edges = explanation_graph.edge_mask_dict[graph_3_name, "rev_to", graph_1_name] > edge_threshold
        #    het_graph23_rel_edges = explanation_graph.edge_mask_dict[graph_2_name, "rev_to", graph_3_name] > edge_threshold
        #    het_graph32_rel_edges = explanation_graph.edge_mask_dict[graph_3_name, "to", graph_2_name] > edge_threshold


        for i, edge in enumerate(hetero_graph[graph_1_name, 'to', graph_1_name].edge_index.cpu().detach().numpy().T):
            if het_graph1_rel_edges[i]:
                for ax_ax in ax:
                    #ax_ax.plot(het_graph1_pos[edge,1], het_graph1_pos[edge,0], c="blue",linewidth=1, alpha=0.5, zorder = 1)
                    ax_ax.plot(het_graph_pos[graph_1_name][edge,1], het_graph_pos[graph_1_name][edge,0], c="blue",linewidth=1, alpha=0.5, zorder = 1)

        for i, edge in enumerate(hetero_graph[graph_2_name, 'to', graph_2_name].edge_index.cpu().detach().numpy().T):
            if het_graph2_rel_edges[i]:
                for ax_ax in ax:
                    #ax_ax.plot(het_graph2_pos[edge,1], het_graph2_pos[edge,0], c="blue",linewidth=1, alpha=0.5, zorder = 1)
                    ax_ax.plot(het_graph_pos[graph_2_name][edge,1], het_graph_pos[graph_2_name][edge,0], c="orange",linewidth=1, alpha=0.5, zorder = 1)
#
        for i, edge in enumerate(hetero_graph[graph_1_name, 'to', graph_2_name].edge_index.cpu().detach().numpy().T):
            if het_graph12_rel_edges[i] or het_graph21_rel_edges[i]:
                #x_pos = (het_graph1_pos[edge[0],1], het_graph2_pos[edge[1],1])
                #y_pos = (het_graph1_pos[edge[0],0], het_graph2_pos[edge[1],0])
                x_pos = (het_graph_pos[graph_1_name][edge[0],1], het_graph_pos[graph_2_name][edge[1],1])
                y_pos = (het_graph_pos[graph_1_name][edge[0],0], het_graph_pos[graph_2_name][edge[1],0])
                for ax_ax in ax:
                    ax_ax.plot(x_pos, y_pos, linewidth=1, alpha=0.5, c= "black", zorder = 1)
#
        #if faz_node:
        #    for i, edge in enumerate(hetero_graph[graph_1_name, 'to', graph_3_name].edge_index.cpu().detach().numpy().T):
        #        if het_graph13_rel_edges[i] or het_graph31_rel_edges[i]:
        #            #x_pos = (het_graph1_pos[edge[0],1], het_graph3_pos[edge[1],1])
        #            #y_pos = (het_graph1_pos[edge[0],0], het_graph3_pos[edge[1],0])
        #            x_pos = (het_graph_pos[graph_1_name][edge[0],1], het_graph_pos[graph_3_name][edge[1],1])
        #            y_pos = (het_graph_pos[graph_1_name][edge[0],0], het_graph_pos[graph_3_name][edge[1],0])
        #            for ax_ax in ax:
        #                ax_ax.plot(x_pos, y_pos, linewidth=1, alpha=0.5, c= "black", zorder = 1)
#
        #    for i, edge in enumerate(hetero_graph[graph_2_name, 'rev_to', graph_3_name].edge_index.cpu().detach().numpy().T):
        #        if het_graph23_rel_edges[i] or het_graph32_rel_edges[i]:
        #            #x_pos = (het_graph2_pos[edge[0],1], het_graph3_pos[edge[1],1])
        #            #y_pos = (het_graph2_pos[edge[0],0], het_graph3_pos[edge[1],0])
        #            x_pos = (het_graph_pos[graph_2_name][edge[0],1], het_graph_pos[graph_3_name][edge[1],1])
        #            y_pos = (het_graph_pos[graph_2_name][edge[0],0], het_graph_pos[graph_3_name][edge[1],0])
        #            for ax_ax in ax:
        #                ax_ax.plot(x_pos, y_pos, linewidth=1, alpha=0.5, c= "black", zorder = 1)


    if path is not None:
        plt.tight_layout()
        plt.savefig(path)
        plt.close()





def visualize_node_importance_histogram(explanation_graph, path, faz_node = False, abs = True):

    import matplotlib.pyplot as plt
    import numpy as np
    import copy

    graph_1_name = "graph_1"
    graph_2_name = "graph_2"
    graph_3_name = "faz"
        
    fig, ax = plt.subplots()

    # abs the node masks
    if abs:
        het_graph1_rel_pos = explanation_graph.node_mask_dict[graph_1_name].abs().sum(dim=-1) 
        het_graph2_rel_pos = explanation_graph.node_mask_dict[graph_2_name].abs().sum(dim=-1)

        if faz_node:
            het_graph3_rel_pos = explanation_graph.node_mask_dict[graph_3_name].abs().sum(dim=-1)
    else:
        het_graph1_rel_pos = explanation_graph.node_mask_dict[graph_1_name].sum(dim=-1) 
        het_graph2_rel_pos = explanation_graph.node_mask_dict[graph_2_name].sum(dim=-1)

        if faz_node:
            het_graph3_rel_pos = explanation_graph.node_mask_dict[graph_3_name].sum(dim=-1)


    # log transform data and add 0.1 to avoid log(0)

    #het_graph1_rel_pos = torch.log(het_graph1_rel_pos + 0.1)
    #het_graph2_rel_pos = torch.log(het_graph2_rel_pos + 0.1)

    offset = 0.0001
    offset = 0

    # set the range for the histogram
    max_val = max(het_graph1_rel_pos.cpu().detach().numpy().max(), het_graph2_rel_pos.cpu().detach().numpy().max())+offset
    #min
    min_val = min(het_graph1_rel_pos.cpu().detach().numpy().min(), het_graph2_rel_pos.cpu().detach().numpy().min()) + offset

    if faz_node:
        #het_graph3_rel_pos = torch.log(het_graph3_rel_pos)
        max_val = max(max_val, het_graph3_rel_pos.cpu().detach().numpy().max()+offset)
        min_val = min(min_val, het_graph3_rel_pos.cpu().detach().numpy().min()+offset)

    #min_val = offset
    # round the maxval to a power of 10
    #max_val = np.ceil(np.log10(max_val))
    #max_val = 10**max_val

    #max_val = offset*10**3

    #hist,bins = np.histogram(het_graph1_rel_pos.cpu().detach().numpy()+offset , bins = 100, range = (min_val, max_val) )
    #logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))

    # plot the histogram
    ax.hist(het_graph1_rel_pos.cpu().detach().numpy()+offset ,bins = 100, alpha =0.5,range = (min_val, max_val), label = "Vessel" ) # bins = logbins,
    ax.hist(het_graph2_rel_pos.cpu().detach().numpy()+offset ,bins = 100, alpha =0.5, range = (min_val, max_val), label = "ICP Region")
    if faz_node:
        ax.hist(het_graph3_rel_pos.cpu().detach().numpy() +offset,bins = 100, range = (min_val, max_val),  alpha =0.5, label = "FAZ")

    # create a legend for the histogram
    ax.legend(loc='upper right')

    plt.xlabel("Node importance (Abs sum of gradients)")
    plt.ylabel("Number of nodes")


    #plt.xscale('log')
    plt.xlim(min_val, max_val)
    # set xticks according to original values before offset
    # go from 0.01 to max_val in *10 increments

    #tick_max_val = np.floor(np.log10(max_val)) 
    #tick_max_val = 10**tick_max_val
#
    #x_ticks_old = []
    #while offset < tick_max_val or len(x_ticks_old) < 2:
    #    x_ticks_old.append(offset)
    #    offset *= 10
#
    #x_ticks_new = copy.deepcopy(x_ticks_old)
    #x_ticks_new[0] = 0



    #plt.xticks(x_ticks_old, x_ticks_new)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def integrate_gradients_heatmap(explanation_graph, hetero_graph, path, faz_node = False):

    import matplotlib.pyplot as plt
    from graph_plotting import graph_2D
    fig, ax = plt.subplots()
    # add titles for each subplot in the figu
    # add legend to the figure that assigns the square marker ("s") to the intercapillary region and the circle marker to the vessel region    
    if faz_node:
        legend_items = [
            plt.Line2D([0], [0], marker='s', color='red', alpha = 0.8, label='ICP Region', markerfacecolor='red', markersize=6, linestyle='None'),
            plt.Line2D([0], [0], marker='o', color='red', alpha = 0.8, label='Vessel', markerfacecolor='red', markersize=6, linestyle='None'),
            plt.Line2D([0], [0], marker="D", color='red', alpha = 0.8, label='Fovea', markerfacecolor='blue', markersize=6, linestyle='None'),
        ]
    else:
        legend_items = [
            plt.Line2D([0], [0], marker='s', color='red', alpha = 0.8, label='ICP Region', markerfacecolor='red', markersize=6, linestyle='None'),
            plt.Line2D([0], [0], marker='o', color='red', alpha = 0.8, label='Vessel', markerfacecolor='red', markersize=6, linestyle='None'),
        ]

    plotter2d = graph_2D.HeteroGraphPlotter2D()
    #plotter2d.set_val_range(1, 0)
    importance_dict = {}
    for key in explanation_graph.x_dict.keys():
        importance_dict[key] = explanation_graph.x_dict[key].abs().sum(dim=-1) 
    if faz_node:
        plotter2d.plot_graph_2D_faz(hetero_graph ,edges= False, pred_val_dict = importance_dict, ax = ax)
    else:
        plotter2d.plot_graph_2D(hetero_graph ,edges= False, pred_val_dict = importance_dict, ax = ax)
    fig.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()



