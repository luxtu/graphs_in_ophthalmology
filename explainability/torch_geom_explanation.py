



def visualize_feature_importance(explanation, path, features_label_dict, top_k = 100, abs = False):
    """
    wrapper for the pytorch geom function, since it does not include tight layout
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import torch

    # take abs for features in node mask to take into account negative values
    abs_dict = {}
    for key in explanation.node_mask_dict.keys():
        if abs:
            abs_dict[key] = torch.abs(explanation.node_mask_dict[key])
        else:
            abs_dict[key] = explanation.node_mask_dict[key]

    # how can the features still be negative?


    score = torch.cat([node_mask.sum(dim=0) for node_mask in abs_dict.values()], dim=0) # explanation.node_mask_dict
    score = score.cpu().numpy() # .detach()
    #print(score)
    node_typeto_proper_name = {"graph_1": "Vessel", "graph_2": "ICP Region", "faz": "FAZ"}
    all_feat_labels = []
    for node_type in explanation.node_mask_dict.keys():
        all_feat_labels += [
            f'{node_typeto_proper_name[node_type]}: {label}' for label in features_label_dict[node_type]
        ]
    df = pd.DataFrame({'score': score}, index=all_feat_labels)
    # sort by absolute value
    #df = df.sort_values('score', ascending=False)
    df_sorted = df.reindex(df.abs().sort_values('score', ascending=False).index)
    #df = df.round(decimals=3)
    df_sorted = df_sorted.head(top_k)
    ax = df_sorted.plot(
           kind='barh',
           figsize=(10, 7),
#              title=title,
           ylabel='Feature label',
           #xlim=[0, float(df['score'].max()) + 0.3*float(df['score'].max())],
           legend=False,
       )
    plt.gca().invert_yaxis()
    #ax.bar_label(container=ax.containers[0], label_type='edge')
    plt.tight_layout()
    plt.savefig(path)


    # three subplots below each other
    fig, axs = plt.subplots(3, 1, figsize=(10, 7))
    # get the 5 most important features for each graph and plot them below each other

    import numpy as np
    max_score = np.abs(score).max()

    for i, node_type in enumerate(explanation.node_mask_dict.keys()):
        score_indv = explanation.node_mask_dict[node_type].sum(dim=0).cpu().numpy()
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



def visualize_relevant_subgraph(explanation_graph, hetero_graph, path, threshold = 0.0, edge_threshold = 0.0, edges = True, faz_node = False, ax = None):

    import matplotlib.pyplot as plt
    import pandas as pd
    import torch
    import numpy as np

    graph_1_name = "graph_1"
    graph_2_name = "graph_2"

    graph_3_name = "faz"


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
            ax = ax[0]

    het_graph1_pos = hetero_graph[graph_1_name].pos.cpu().detach().numpy()
    het_graph2_pos = hetero_graph[graph_2_name].pos.cpu().detach().numpy()
    if faz_node:
        het_graph3_pos = hetero_graph[graph_3_name].pos.cpu().detach().numpy()



    if threshold == "adaptive":


        total_grad = explanation_graph.node_mask_dict[graph_1_name].abs().sum() + explanation_graph.node_mask_dict[graph_2_name].abs().sum()
        #avg_grad = total_grad / (explanation_graph.node_mask_dict[graph_1_name].shape[0] + explanation_graph.node_mask_dict[graph_2_name].shape[0])

        if faz_node:
            total_grad += explanation_graph.node_mask_dict[graph_3_name].abs().sum()
        
        # find a threshold such that 90% of the total gradient is explained

        node_value = torch.cat((explanation_graph.node_mask_dict[graph_1_name].abs().sum(dim=-1), explanation_graph.node_mask_dict[graph_2_name].abs().sum(dim=-1)))
        if faz_node:
            node_value = torch.cat((explanation_graph.node_mask_dict[graph_3_name].abs().sum(dim=-1), node_value))


        sorted_node_value = torch.sort(node_value, descending=True)[0]
        # get rid of the nodes that contribute less than 0.1% to the total gradient
        sorted_node_value = sorted_node_value[sorted_node_value > 0.001 * total_grad]


        cum_sum = torch.cumsum(sorted_node_value, dim=0)
        cropped_grad = cum_sum[-1]
        threshold = sorted_node_value[cum_sum < 0.95 * cropped_grad][-1]

        #threshold = avg_grad #max_val * 0.01
        print("threshold", threshold)
    
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

        

    het_graph1_rel_pos = explanation_graph.node_mask_dict[graph_1_name].abs().sum(dim=-1) > threshold
    het_graph2_rel_pos = explanation_graph.node_mask_dict[graph_2_name].abs().sum(dim=-1) > threshold

    #import numpy as np
    #print("subgraph_regions")
    #print(np.where(het_graph2_rel_pos.cpu().detach().numpy()))

    het_graph1_rel_pos = het_graph1_rel_pos.cpu().detach().numpy()
    het_graph2_rel_pos = het_graph2_rel_pos.cpu().detach().numpy()

    het_graph1_rel_pos = het_graph1_pos[het_graph1_rel_pos]
    het_graph2_rel_pos = het_graph2_pos[het_graph2_rel_pos]

    if faz_node:
        het_graph3_rel_pos = explanation_graph.node_mask_dict[graph_3_name].abs().sum(dim=-1) > threshold
        het_graph3_rel_pos = het_graph3_rel_pos.cpu().detach().numpy()
        het_graph3_rel_pos = het_graph3_pos[het_graph3_rel_pos]


    ax.scatter(het_graph1_rel_pos[:,1], het_graph1_rel_pos[:,0], zorder = 2, s = 8, alpha= 0.9)
    ax.scatter(het_graph2_rel_pos[:,1], het_graph2_rel_pos[:,0], zorder = 2, s = 12, alpha= 0.9, marker = "s")

    if faz_node:
        ax.scatter(het_graph3_rel_pos[:,1], het_graph3_rel_pos[:,0], zorder = 2, s = 12, alpha= 0.9, marker = "D", c = "red")

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
                ax.plot(het_graph1_pos[edge,1], het_graph1_pos[edge,0], c="blue",linewidth=1, alpha=0.5, zorder = 1)

        for i, edge in enumerate(hetero_graph[graph_2_name, 'to', graph_2_name].edge_index.cpu().detach().numpy().T):
            if het_graph2_rel_edges[i]:
                ax.plot(het_graph2_pos[edge,1], het_graph2_pos[edge,0], c="red",linewidth=1, alpha=0.5, zorder = 1)

        for i, edge in enumerate(hetero_graph[graph_1_name, 'to', graph_2_name].edge_index.cpu().detach().numpy().T):
            if het_graph12_rel_edges[i] or het_graph21_rel_edges[i]:
                x_pos = (het_graph1_pos[edge[0],1], het_graph2_pos[edge[1],1])
                y_pos = (het_graph1_pos[edge[0],0], het_graph2_pos[edge[1],0])

                ax.plot(x_pos, y_pos, linewidth=1, alpha=0.5, c= "black", zorder = 1)

        if faz_node:
            for i, edge in enumerate(hetero_graph[graph_1_name, 'to', graph_3_name].edge_index.cpu().detach().numpy().T):
                if het_graph13_rel_edges[i] or het_graph31_rel_edges[i]:
                    x_pos = (het_graph1_pos[edge[0],1], het_graph3_pos[edge[1],1])
                    y_pos = (het_graph1_pos[edge[0],0], het_graph3_pos[edge[1],0])

                    ax.plot(x_pos, y_pos, linewidth=1, alpha=0.5, c= "black", zorder = 1)

            for i, edge in enumerate(hetero_graph[graph_2_name, 'rev_to', graph_3_name].edge_index.cpu().detach().numpy().T):
                if het_graph23_rel_edges[i] or het_graph32_rel_edges[i]:
                    x_pos = (het_graph2_pos[edge[0],1], het_graph3_pos[edge[1],1])
                    y_pos = (het_graph2_pos[edge[0],0], het_graph3_pos[edge[1],0])

                    ax.plot(x_pos, y_pos, linewidth=1, alpha=0.5, c= "black", zorder = 1)


    if path is not None:
        plt.tight_layout()
        plt.savefig(path)
        plt.close()





def visualize_node_importance_histogram(explanation_graph, path):

    import matplotlib.pyplot as plt
    import pandas as pd
    import torch

    graph_1_name = "graph_1"
    graph_2_name = "graph_2"
    graph_3_name = "faz"
        
    fig, ax = plt.subplots()

    # abs the node masks
    het_graph1_rel_pos = explanation_graph.node_mask_dict[graph_1_name].abs().sum(dim=-1) 
    het_graph2_rel_pos = explanation_graph.node_mask_dict[graph_2_name].abs().sum(dim=-1)

    print(het_graph1_rel_pos.max())
    print(het_graph2_rel_pos.max())


    ax.hist(het_graph1_rel_pos.cpu().detach().numpy(), bins = 100, alpha =0.5, label = "Vessel")
    ax.hist(het_graph2_rel_pos.cpu().detach().numpy(), bins = 100, alpha =0.5, label = "ICP Region")
    plt.xscale('log')

    plt.tight_layout()
    plt.savefig(path)
    plt.close()