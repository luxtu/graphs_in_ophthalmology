



def visualize_feature_importance(explanation, path, features_label_dict, top_k = 100):
    """
    wrapper for the pytorch geom function, since it does not include tight layout
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import torch

    score = torch.cat([node_mask.sum(dim=0) for node_mask in explanation.node_mask_dict.values()], dim=0)
    score = score.cpu().numpy()
    all_feat_labels = []
    for node_type in explanation.node_mask_dict.keys():
        all_feat_labels += [
            f'{node_type}#{label}' for label in features_label_dict[node_type]
        ]
    df = pd.DataFrame({'score': score}, index=all_feat_labels)
    df = df.sort_values('score', ascending=False)
    df = df.round(decimals=3)
    df = df.head(top_k)
    ax = df.plot(
           kind='barh',
           figsize=(10, 7),
#              title=title,
           ylabel='Feature label',
           xlim=[0, float(df['score'].max()) + 0.3],
           legend=False,
       )
    plt.gca().invert_yaxis()
    ax.bar_label(container=ax.containers[0], label_type='edge')
    plt.tight_layout()
    plt.savefig(path)



def visualize_relevant_subgraph(explanation_graph, hetero_graph, path, threshold = 0.0, edge_threshold = 0.0):

    import matplotlib.pyplot as plt
    import pandas as pd
    import torch

    graph_1_name = "graph_1"
    graph_2_name = "graph_2"


    fig, ax = plt.subplots()
    side_l = 5.5
    fig.set_figwidth(side_l)
    fig.set_figheight(side_l)
    plt.ylim(0,1200)
    plt.xlim(0,1200)

    het_graph1_pos = hetero_graph[graph_1_name].pos.cpu().detach().numpy()
    het_graph2_pos = hetero_graph[graph_2_name].pos.cpu().detach().numpy()

    print(explanation_graph.node_mask_dict[graph_1_name].sum(dim=-1))
    print(explanation_graph.node_mask_dict[graph_2_name].sum(dim=-1).shape)
    print(explanation_graph.node_mask_dict[graph_2_name].sum(dim=-1))


    het_graph1_rel_pos = explanation_graph.node_mask_dict[graph_1_name].sum(dim=-1) > threshold
    het_graph2_rel_pos = explanation_graph.node_mask_dict[graph_2_name].sum(dim=-1) > threshold

    het_graph1_rel_pos = het_graph1_rel_pos.cpu().detach().numpy()
    het_graph2_rel_pos = het_graph2_rel_pos.cpu().detach().numpy()

    het_graph1_rel_pos = het_graph1_pos[het_graph1_rel_pos]
    het_graph2_rel_pos = het_graph2_pos[het_graph2_rel_pos]


    ax.scatter(het_graph1_rel_pos[:,1], het_graph1_rel_pos[:,0],zorder = 2, s = 8)
    ax.scatter(het_graph2_rel_pos[:,1], het_graph2_rel_pos[:,0],zorder = 2, s = 12, marker = "s")

    het_graph1_rel_edges = explanation_graph.edge_mask_dict[graph_1_name, "to", graph_1_name] > edge_threshold
    het_graph2_rel_edges = explanation_graph.edge_mask_dict[graph_2_name, "to", graph_2_name] > edge_threshold

    het_graph1_rel_edges = het_graph1_rel_edges.cpu().detach().numpy()
    het_graph2_rel_edges = het_graph2_rel_edges.cpu().detach().numpy()

    het_graph12_rel_edges = explanation_graph.edge_mask_dict[graph_1_name, "to", graph_2_name] > edge_threshold
    het_graph21_rel_edges = explanation_graph.edge_mask_dict[graph_2_name, "rev_to", graph_1_name] > edge_threshold

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



    plt.tight_layout()
    plt.savefig(path)