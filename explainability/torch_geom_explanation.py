



def visualize_feature_importance(explanation,hetero_graph, path, features_label_dict, top_k = 100, threshold = None):
    """
    wrapper for the pytorch geom function, since it does not include tight layout
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import torch
    import numpy as np
    import copy


    # take abs for features in node mask to take into account negative values
    abs_dict = {}
    work_dict = {}
    for key in explanation.node_mask_dict.keys():
        abs_dict[key] = copy.deepcopy(torch.abs(explanation.node_mask_dict[key]))
        work_dict[key] = copy.deepcopy(explanation.node_mask_dict[key])

    # how can the features still be negative?

    if threshold is None:
        score = torch.cat([node_mask.sum(dim=0) for node_mask in work_dict.values()], dim=0) # dim 0 is the feature dimension
        score = score.cpu().numpy() # .detach()

    elif threshold == "adaptive":

        total_grad = 0
        for key in explanation.node_mask_dict.keys():
            total_grad += abs_dict[key].abs().sum()

        node_value = torch.cat([abs_dict[key].abs().sum(dim=-1) for key in explanation.node_mask_dict.keys()], dim=0)

        sorted_node_value = torch.sort(node_value, descending=True)[0]
        # get rid of the nodes that contribute less than 0.1% to the total gradient
        sorted_node_value = sorted_node_value[sorted_node_value > 0.0005 * total_grad]


        cum_sum = torch.cumsum(sorted_node_value, dim=0)
        cropped_grad = cum_sum[-1]
        threshold = sorted_node_value[cum_sum < 0.95 * cropped_grad][-1]

        from sklearn.cluster import DBSCAN

        # only take the nodes for the score of each type, that are above the threshold
        relevant_dict = {}
        for node_key in ["graph_1", "graph_2"]:
            # get the importances for each node
            #print(work_dict[node_key].shape)
            importances = abs_dict[node_key].sum(dim=-1)
            #print(importances.shape)
            # set all rows to 0 that are below the threshold
            work_dict[node_key][importances < threshold, :] = 0
            # create dict with all node with importance above threshold
            pos = hetero_graph[node_key].pos.cpu().detach().numpy()
            # select position where 
            remove_pos = (pos[:,0] > 1100) & (pos[:,1] < 100)
            work_dict[node_key][remove_pos, :] = 0

            relevant_dict[node_key] = work_dict[node_key][importances >= threshold, :]
            print(relevant_dict[node_key].shape)
            print(work_dict[node_key].shape)

            ## visualize the clusters in 2D with a PCA embedding
            #from sklearn.metrics.pairwise import cosine_similarity
            #from sklearn.cluster import KMeans
#
            ## get the cosine similarity between all nodes
            ##cos_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
#
            ## Step 2: Calculate cosine similarities
            #cosine_similarities = cosine_similarity(relevant_dict[node_key].cpu().detach().numpy())
#
            ## Step 3: Perform clustering (K-Means in this example)
            #num_clusters = 3
            #kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            #cluster_labels = kmeans.fit_predict(cosine_similarities)
#
            ## Step 4: Visualization (2D PCA for illustration purposes)
            #from sklearn.decomposition import PCA
#
            #pca = PCA(n_components=2)
            #embeddings_2d = pca.fit_transform(relevant_dict[node_key].cpu().detach().numpy())
#
            #plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=cluster_labels, cmap='viridis')
            #plt.title('Clustering based on Cosine Similarities')
            #plt.savefig(f"cluster_{node_key}.png")
            #plt.close()

            # get the cluster centers



            #from sklearn.decomposition import PCA
            #pca = PCA(n_components=2)
            #pca.fit(relevant_dict[node_key].cpu().detach().numpy())
            #pca_embedding = pca.transform(relevant_dict[node_key].cpu().detach().numpy())
#
            ## for each feature in the original embedding, plot the vector in the new embedding
            #for i in range(relevant_dict[node_key].shape[1]):
            #    plt.arrow(0, 0, pca.components_[0,i], pca.components_[1,i], head_width=0.05, head_length=0.1, fc='k', ec='k')
            #    plt.text(pca.components_[0,i], pca.components_[1,i], features_label_dict[node_key][i])
#
#
            ## cluster the nodes in the embedding
            #cluster = DBSCAN(eps=0.3, min_samples=1).fit(pca_embedding)
#
            ## print the explained variance
            #print("Explained variance:", pca.explained_variance_ratio_)
#
            #print("Number of clusters:", len(set(cluster.labels_)))
#
            ## plot the clusters
            #plt.scatter(pca_embedding[:,0], pca_embedding[:,1], c=cluster.labels_)
            #plt.title(f"Cluster for {node_key}")
            ## add the number of clusters to the plot
            #plt.text(0.5, 0.5, f"Number of clusters: {len(set(cluster.labels_))}", horizontalalignment='center', verticalalignment='center')
            #plt.savefig(f"cluster_{node_key}.png")
            #plt.close()



        





        score = torch.cat([node_mask.sum(dim=0) for node_mask in work_dict.values()], dim=0) # dim 0 is the feature dimension
        score = score.cpu().numpy() # .detach()



    #print(score)
    node_typeto_proper_name = {"graph_1": "Vessel", "graph_2": "ICP Region", "faz": "FAZ"}
    all_feat_labels = []
    for node_type in explanation.node_mask_dict.keys():
        all_feat_labels += [
            f'{node_typeto_proper_name[node_type]}: {label}' for label in features_label_dict[node_type]
        ]

    #print("all_feat_labels", len(all_feat_labels))
    #print("score", len(score))

    #print(all_feat_labels)
    #print(score)

    
    #df = pd.DataFrame({'score': score}, index=all_feat_labels)
    ## sort by absolute value
    ##df = df.sort_values('score', ascending=False)
    #df_sorted = df.reindex(df.abs().sort_values('score', ascending=False).index)
    ##df = df.round(decimals=3)
    #df_sorted = df_sorted.head(top_k)
    #ax = df_sorted.plot(
    #       kind='barh',
    #       figsize=(10, 7),
#   #           title=title,
    #       ylabel='Feature label',
    #       #xlim=[0, float(df['score'].max()) + 0.3*float(df['score'].max())],
    #       legend=False,
    #   )
    #plt.gca().invert_yaxis()
    ##ax.bar_label(container=ax.containers[0], label_type='edge')
    #plt.tight_layout()
    #plt.savefig(path)


    # three subplots below each other
    fig, axs = plt.subplots(3, 1, figsize=(10, 7))
    # get the 5 most important features for each graph and plot them below each other


    max_score = np.abs(score).max()

    for i, node_type in enumerate(explanation.node_mask_dict.keys()):
        score_indv = work_dict[node_type].sum(dim=0).cpu().numpy()
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
            ax = ax
        else:
            ax = [ax]

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
        sorted_node_value = sorted_node_value[sorted_node_value > 0.0005 * total_grad]


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

    het_graph1_rel_pos = het_graph1_rel_pos.cpu().detach().numpy()
    het_graph2_rel_pos = het_graph2_rel_pos.cpu().detach().numpy()

    # remove the inlay
    het_graph1_rel_pos = het_graph1_rel_pos & (het_graph1_pos[:,0] < 1100) & (het_graph1_pos[:,1] > 100) 
    het_graph2_rel_pos = het_graph2_rel_pos & (het_graph2_pos[:,0] < 1100) & (het_graph2_pos[:,1] > 100)
    # 
    #from sklearn.metrics.pairwise import cosine_similarity
    #from sklearn.cluster import KMeans
##
    ## Step 2: Calculate cosine similarities of the relevant nodes
    #cosine_similarities = cosine_similarity(explanation_graph.node_mask_dict[graph_1_name].cpu().detach().numpy()[het_graph1_rel_pos], explanation_graph.node_mask_dict[graph_1_name].cpu().detach().numpy()[het_graph1_rel_pos])
##
    ## Step 3: Perform clustering (K-Means in this example)
    #num_clusters = 2
    #kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    #cluster_labels = kmeans.fit_predict(cosine_similarities)
#
    het_graph1_rel_pos = het_graph1_pos[het_graph1_rel_pos]
    het_graph2_rel_pos = het_graph2_pos[het_graph2_rel_pos]
#
    ## separate the nodes in het_graph1_rel_pos into the clusters
    #cluster_1 = het_graph1_rel_pos[cluster_labels == 0]
    #cluster_2 = het_graph1_rel_pos[cluster_labels == 1]

    if faz_node:
        het_graph3_rel_pos = explanation_graph.node_mask_dict[graph_3_name].abs().sum(dim=-1) > threshold
        het_graph3_rel_pos = het_graph3_rel_pos.cpu().detach().numpy()
        het_graph3_rel_pos = het_graph3_pos[het_graph3_rel_pos]

    for ax_ax in ax:
        #ax_ax.scatter(cluster_1[:,1], cluster_1[:,0], zorder = 2, s = 8, alpha= 0.9)
        #ax_ax.scatter(cluster_2[:,1], cluster_2[:,0], zorder = 2, s = 12, alpha= 0.9, marker = "s")
        ax_ax.scatter(het_graph1_rel_pos[:,1], het_graph1_rel_pos[:,0], zorder = 2, s = 8, alpha= 0.9)
        ax_ax.scatter(het_graph2_rel_pos[:,1], het_graph2_rel_pos[:,0], zorder = 2, s = 12, alpha= 0.9, marker = "s")



    if faz_node:
        for ax_ax in ax:
            ax_ax.scatter(het_graph3_rel_pos[:,1], het_graph3_rel_pos[:,0], zorder = 2, s = 12, alpha= 0.9, marker = "D", c = "red")

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
                    ax_ax.plot(het_graph1_pos[edge,1], het_graph1_pos[edge,0], c="blue",linewidth=1, alpha=0.5, zorder = 1)

        for i, edge in enumerate(hetero_graph[graph_2_name, 'to', graph_2_name].edge_index.cpu().detach().numpy().T):
            if het_graph2_rel_edges[i]:
                for ax_ax in ax:
                    ax_ax.plot(het_graph2_pos[edge,1], het_graph2_pos[edge,0], c="blue",linewidth=1, alpha=0.5, zorder = 1)

        for i, edge in enumerate(hetero_graph[graph_1_name, 'to', graph_2_name].edge_index.cpu().detach().numpy().T):
            if het_graph12_rel_edges[i] or het_graph21_rel_edges[i]:
                x_pos = (het_graph1_pos[edge[0],1], het_graph2_pos[edge[1],1])
                y_pos = (het_graph1_pos[edge[0],0], het_graph2_pos[edge[1],0])
                for ax_ax in ax:
                    ax_ax.plot(x_pos, y_pos, linewidth=1, alpha=0.5, c= "black", zorder = 1)

        if faz_node:
            for i, edge in enumerate(hetero_graph[graph_1_name, 'to', graph_3_name].edge_index.cpu().detach().numpy().T):
                if het_graph13_rel_edges[i] or het_graph31_rel_edges[i]:
                    x_pos = (het_graph1_pos[edge[0],1], het_graph3_pos[edge[1],1])
                    y_pos = (het_graph1_pos[edge[0],0], het_graph3_pos[edge[1],0])

                    for ax_ax in ax:
                        ax_ax.plot(x_pos, y_pos, linewidth=1, alpha=0.5, c= "black", zorder = 1)

            for i, edge in enumerate(hetero_graph[graph_2_name, 'rev_to', graph_3_name].edge_index.cpu().detach().numpy().T):
                if het_graph23_rel_edges[i] or het_graph32_rel_edges[i]:
                    x_pos = (het_graph2_pos[edge[0],1], het_graph3_pos[edge[1],1])
                    y_pos = (het_graph2_pos[edge[0],0], het_graph3_pos[edge[1],0])
                    for ax_ax in ax:
                        ax_ax.plot(x_pos, y_pos, linewidth=1, alpha=0.5, c= "black", zorder = 1)


    if path is not None:
        plt.tight_layout()
        plt.savefig(path)
        plt.close()





def visualize_node_importance_histogram(explanation_graph, path, faz_node = False):

    import matplotlib.pyplot as plt
    import pandas as pd
    import torch
    import numpy as np

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
    import copy
    x_ticks_new = copy.deepcopy(x_ticks_old)
    x_ticks_new[0] = 0



    plt.xticks(x_ticks_old, x_ticks_new)


    


    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
