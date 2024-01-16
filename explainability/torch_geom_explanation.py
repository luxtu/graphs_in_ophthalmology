

def _calculate_adaptive_node_threshold(explanation_graph):
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
    # get rid of the nodes that contribute less than 0.1% to the total gradient
    sorted_node_value = sorted_node_value[sorted_node_value > 0.0005 * total_grad]
    cum_sum = torch.cumsum(sorted_node_value, dim=0)
    cropped_grad = cum_sum[-1]
    threshold = sorted_node_value[cum_sum < 0.95 * cropped_grad][-1]

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

def _feature_importance_plot(explanation, path, features_label_dict, score):

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
        df_sorted = df.reindex(df.abs().sort_values('score', ascending=False).index)
        df_sorted = df_sorted.head(5)
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


def store_relevant_nodes_csv(explanation_graph, hetero_graph, path, features_label_dict, faz_node = False):

    import pandas as pd
    het_graph_rel_pos_dict = identifiy_relevant_nodes(explanation_graph, hetero_graph, faz_node = faz_node)

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

    

    



def identifiy_relevant_nodes(explanation_graph, hetero_graph, faz_node = False):

    import numpy as np
    threshold = _calculate_adaptive_node_threshold(explanation_graph)

    graph_names = ["graph_1", "graph_2"]
    if faz_node:
        graph_names.append("faz")
    
    het_graph_rel_pos_dict = {}
    for graph_name in graph_names:
        #filter the nodes that are above the threshold
        het_graph_rel_pos = explanation_graph.node_mask_dict[graph_name].abs().sum(dim=-1) > threshold
        het_graph_rel_pos = het_graph_rel_pos.cpu().detach().numpy()
        # remove the inlay
        # pos to cpu and detach
        het_graph_pos = hetero_graph[graph_name].pos.cpu().detach().numpy()
        het_graph_rel_pos = het_graph_rel_pos & (het_graph_pos[:,0] < 1100) & (het_graph_pos[:,1] > 100)
        het_graph_rel_pos_dict[graph_name] = het_graph_rel_pos

    return het_graph_rel_pos_dict



def top_k_important_nodes(explanation_graph, hetero_graph, top_k = 100):
    """
    Returns the gradients of the top k most important nodes for each graph, and features of these nodes
    """
    import torch
    abs_dict = {}
    grad_dict = {}
    feature_dict = {}
    for key in explanation_graph.node_mask_dict.keys():
        abs_dict[key] = torch.abs(explanation_graph.node_mask_dict[key])
        # remove the inlay
        abs_dict = _remove_inlay(hetero_graph, abs_dict, key)
        # sort the gradients and get the top k
        _, indices = torch.sort(abs_dict[key].sum(dim=-1), descending=True)
        # get the top k indices
        top_k_indices = indices[:top_k]
        # get the gradients for the top k indices
        grad_dict[key] = explanation_graph.node_mask_dict[key][top_k_indices]
        # get the features for the top k indices
        feature_dict[key] = hetero_graph[key].x[top_k_indices]

    return grad_dict, feature_dict

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



def visualize_feature_importance(explanation, hetero_graph, path, features_label_dict, threshold = None):
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
        abs_dict[key] = copy.deepcopy(torch.abs(explanation.node_mask_dict[key]))
        work_dict[key] = copy.deepcopy(explanation.node_mask_dict[key])

    # how can the features still be negative?

    if threshold is None:


        #score = torch.cat([node_mask.sum(dim=0) for node_mask in work_dict.values()], dim=0) # dim 0 is the feature dimension
        #score = score.cpu().numpy() # .detach()

        # make the score a dictionary with the keys being the graph names
        score = {}
        for key in explanation.node_mask_dict.keys():
            score[key] = work_dict[key].sum(dim=0).cpu().numpy()


    elif threshold == "adaptive":

        faz_node = False
        if "faz" in explanation.node_mask_dict.keys():
            faz_node = True

        het_graph_rel_pos_dict = identifiy_relevant_nodes(explanation, hetero_graph, faz_node = faz_node)

        # print number of relevant nodes
        for key in explanation.node_mask_dict.keys():
            print(f"{key}: {het_graph_rel_pos_dict[key].sum()}")


        score = {}
        for key in explanation.node_mask_dict.keys():
            score[key] = work_dict[key][het_graph_rel_pos_dict[key]].sum(dim=0).cpu().numpy()


        # calculate the score for each node, index only the nodes that are relevant
        #score = torch.cat([node_mask[het_graph_rel_pos_dict[key]].sum(dim=0) for key, node_mask in work_dict.items()], dim=0) # dim 0 is the feature dimension
        #score = score.cpu().numpy() # .detach()

        

        #threshold = _calculate_adaptive_node_threshold(explanation)
#
        ## remove those nodes for the score that are below the threshold or in the inlay
        #for node_key in ["graph_1", "graph_2"]:
        #    # get the importances for each node and sum over the absulute values for each feature
        #    importances = abs_dict[node_key].sum(dim=-1)
        #    # set all rows to 0 that are below the threshold, this way the sum of the node mask is not affected
        #    work_dict[node_key][importances < threshold, :] = 0
        #    # remove the inlay
        #    work_dict = _remove_inlay(hetero_graph, work_dict, node_key)



        #score = torch.cat([node_mask.sum(dim=0) for node_mask in work_dict.values()], dim=0) # dim 0 is the feature dimension
        #score = score.cpu().numpy() # .detach()


    _feature_importance_plot(explanation, path, features_label_dict, score)




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
        plt.ylim(0,1200)
        plt.xlim(0,1200)
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
            edge_threshold = sorted_edge_value[cum_sum < 0.95 * total_grad][-1]

            #print("edge_threshold", avg_grad)
            #edge_threshold = max(avg_grad, 0.05*threshold) # max_val * 0.01
            #print("edge_threshold", edge_threshold)

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

        het_graph12_rel_edges = explanation_graph.edge_mask_dict[graph_1_name, "to", graph_2_name] > edge_threshold
        het_graph21_rel_edges = explanation_graph.edge_mask_dict[graph_2_name, "rev_to", graph_1_name] > edge_threshold

        if faz_node:
            het_graph13_rel_edges = explanation_graph.edge_mask_dict[graph_1_name, "to", graph_3_name] > edge_threshold
            het_graph31_rel_edges = explanation_graph.edge_mask_dict[graph_3_name, "rev_to", graph_1_name] > edge_threshold
            het_graph23_rel_edges = explanation_graph.edge_mask_dict[graph_2_name, "rev_to", graph_3_name] > edge_threshold
            het_graph32_rel_edges = explanation_graph.edge_mask_dict[graph_3_name, "to", graph_2_name] > edge_threshold


        for i, edge in enumerate(hetero_graph[graph_1_name, 'to', graph_1_name].edge_index.cpu().detach().numpy().T):
            if het_graph1_rel_edges[i]:
                for ax_ax in ax:
                    #ax_ax.plot(het_graph1_pos[edge,1], het_graph1_pos[edge,0], c="blue",linewidth=1, alpha=0.5, zorder = 1)
                    ax_ax.plot(het_graph_pos[graph_1_name][edge,1], het_graph_pos[graph_1_name][edge,0], c="blue",linewidth=1, alpha=0.5, zorder = 1)

        for i, edge in enumerate(hetero_graph[graph_2_name, 'to', graph_2_name].edge_index.cpu().detach().numpy().T):
            if het_graph2_rel_edges[i]:
                for ax_ax in ax:
                    #ax_ax.plot(het_graph2_pos[edge,1], het_graph2_pos[edge,0], c="blue",linewidth=1, alpha=0.5, zorder = 1)
                    ax_ax.plot(het_graph_pos[graph_2_name][edge,1], het_graph_pos[graph_2_name][edge,0], c="blue",linewidth=1, alpha=0.5, zorder = 1)

        for i, edge in enumerate(hetero_graph[graph_1_name, 'to', graph_2_name].edge_index.cpu().detach().numpy().T):
            if het_graph12_rel_edges[i] or het_graph21_rel_edges[i]:
                #x_pos = (het_graph1_pos[edge[0],1], het_graph2_pos[edge[1],1])
                #y_pos = (het_graph1_pos[edge[0],0], het_graph2_pos[edge[1],0])
                x_pos = (het_graph_pos[graph_1_name][edge[0],1], het_graph_pos[graph_2_name][edge[1],1])
                y_pos = (het_graph_pos[graph_1_name][edge[0],0], het_graph_pos[graph_2_name][edge[1],0])
                for ax_ax in ax:
                    ax_ax.plot(x_pos, y_pos, linewidth=1, alpha=0.5, c= "black", zorder = 1)

        if faz_node:
            for i, edge in enumerate(hetero_graph[graph_1_name, 'to', graph_3_name].edge_index.cpu().detach().numpy().T):
                if het_graph13_rel_edges[i] or het_graph31_rel_edges[i]:
                    #x_pos = (het_graph1_pos[edge[0],1], het_graph3_pos[edge[1],1])
                    #y_pos = (het_graph1_pos[edge[0],0], het_graph3_pos[edge[1],0])
                    x_pos = (het_graph_pos[graph_1_name][edge[0],1], het_graph_pos[graph_3_name][edge[1],1])
                    y_pos = (het_graph_pos[graph_1_name][edge[0],0], het_graph_pos[graph_3_name][edge[1],0])
                    for ax_ax in ax:
                        ax_ax.plot(x_pos, y_pos, linewidth=1, alpha=0.5, c= "black", zorder = 1)

            for i, edge in enumerate(hetero_graph[graph_2_name, 'rev_to', graph_3_name].edge_index.cpu().detach().numpy().T):
                if het_graph23_rel_edges[i] or het_graph32_rel_edges[i]:
                    #x_pos = (het_graph2_pos[edge[0],1], het_graph3_pos[edge[1],1])
                    #y_pos = (het_graph2_pos[edge[0],0], het_graph3_pos[edge[1],0])
                    x_pos = (het_graph_pos[graph_2_name][edge[0],1], het_graph_pos[graph_3_name][edge[1],1])
                    y_pos = (het_graph_pos[graph_2_name][edge[0],0], het_graph_pos[graph_3_name][edge[1],0])
                    for ax_ax in ax:
                        ax_ax.plot(x_pos, y_pos, linewidth=1, alpha=0.5, c= "black", zorder = 1)


    if path is not None:
        plt.tight_layout()
        plt.savefig(path)
        plt.close()





def visualize_node_importance_histogram(explanation_graph, path, faz_node = False):

    import matplotlib.pyplot as plt
    import numpy as np
    import copy

    graph_1_name = "graph_1"
    graph_2_name = "graph_2"
    graph_3_name = "faz"
        
    fig, ax = plt.subplots()

    # abs the node masks
    het_graph1_rel_pos = explanation_graph.node_mask_dict[graph_1_name].abs().sum(dim=-1) 
    het_graph2_rel_pos = explanation_graph.node_mask_dict[graph_2_name].abs().sum(dim=-1)

    if faz_node:
        het_graph3_rel_pos = explanation_graph.node_mask_dict[graph_3_name].abs().sum(dim=-1)


    # log transform data and add 0.1 to avoid log(0)

    #het_graph1_rel_pos = torch.log(het_graph1_rel_pos + 0.1)
    #het_graph2_rel_pos = torch.log(het_graph2_rel_pos + 0.1)

    offset = 0.01

    # set the range for the histogram
    max_val = max(het_graph1_rel_pos.cpu().detach().numpy().max(), het_graph2_rel_pos.cpu().detach().numpy().max())+offset
    min_val = min(het_graph1_rel_pos.cpu().detach().numpy().min(), het_graph2_rel_pos.cpu().detach().numpy().min())+offset

    if faz_node:
        #het_graph3_rel_pos = torch.log(het_graph3_rel_pos)
        max_val = max(max_val, het_graph3_rel_pos.cpu().detach().numpy().max()+offset)
        min_val = min(min_val, het_graph3_rel_pos.cpu().detach().numpy().min()+offset)

    min_val = offset
    # round the maxval to a power of 10
    #max_val = np.ceil(np.log10(max_val))
    #max_val = 10**max_val

    #max_val = offset*10**3

    hist,bins = np.histogram(het_graph1_rel_pos.cpu().detach().numpy()+offset , bins = 100, range = (min_val, max_val) )
    logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))

    # plot the histogram
    ax.hist(het_graph1_rel_pos.cpu().detach().numpy()+offset , bins = logbins, alpha =0.5, label = "Vessel" )
    ax.hist(het_graph2_rel_pos.cpu().detach().numpy()+offset , bins = logbins, alpha =0.5, label = "ICP Region")
    if faz_node:
        ax.hist(het_graph3_rel_pos.cpu().detach().numpy() +offset, bins = logbins, alpha =0.5, label = "FAZ")

    # create a legend for the histogram
    ax.legend(loc='upper right')

    plt.xlabel("Node importance")
    plt.ylabel("Number of nodes")


    plt.xscale('log')
    plt.xlim(offset, max_val)
    # set xticks according to original values before offset
    # go from 0.01 to max_val in *10 increments

    tick_max_val = np.floor(np.log10(max_val)) 
    tick_max_val = 10**tick_max_val

    x_ticks_old = []
    while offset < tick_max_val or len(x_ticks_old) < 2:
        x_ticks_old.append(offset)
        offset *= 10

    x_ticks_new = copy.deepcopy(x_ticks_old)
    x_ticks_new[0] = 0



    plt.xticks(x_ticks_old, x_ticks_new)
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



