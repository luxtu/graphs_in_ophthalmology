def top_k_important_nodes(
    explanation_graph,
    hetero_graph,
    faz_node=False,
    top_k=100,
    only_positive=False,
    each_type=False,
):
    """
    Returns the gradients of the top k most important nodes for each graph or across all graphs
    """
    if each_type:
        return _top_k_important_nodes_all_types(
            explanation_graph,
            hetero_graph,
            faz_node=faz_node,
            top_k=top_k,
            only_positive=only_positive,
        )
    else:
        return _top_k_important_nodes(
            explanation_graph,
            hetero_graph,
            faz_node=faz_node,
            top_k=top_k,
            only_positive=only_positive,
        )


def _top_k_important_nodes(
    explanation_graph, hetero_graph, faz_node=False, top_k=100, only_positive=False
):
    """
    Returns the gradients of the top k most important nodes across all graphs
    """

    import numpy as np

    graph_names = ["graph_1", "graph_2"]
    if faz_node:
        graph_names.append("faz")

    het_graph_rel_pos_dict = {}
    # find the threshold that only includes the top k nodes

    for graph_name in graph_names:
        # filter the nodes that are above the threshold
        if only_positive:
            het_graph_weight = explanation_graph.node_mask_dict[graph_name].sum(dim=-1)
        else:
            het_graph_weight = (
                explanation_graph.node_mask_dict[graph_name].abs().sum(dim=-1)
            )
        het_graph_weight = het_graph_weight.cpu().detach().numpy()
        inlay_pos = (
            hetero_graph[graph_name].pos.cpu().detach().numpy()[:, 0] > 1100
        ) & (hetero_graph[graph_name].pos.cpu().detach().numpy()[:, 1] < 100)
        het_graph_weight[inlay_pos] = -5

        # concat the weights
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
        # filter the nodes that are above the threshold
        if only_positive:
            het_graph_rel_pos = (
                explanation_graph.node_mask_dict[graph_name].sum(dim=-1) >= threshold
            )
        else:
            het_graph_rel_pos = (
                explanation_graph.node_mask_dict[graph_name].abs().sum(dim=-1)
                >= threshold
            )
        het_graph_rel_pos = het_graph_rel_pos.cpu().detach().numpy()
        # print the number of relevant nodes
        # remove the inlay
        # het_graph_rel_pos = het_graph_rel_pos & (hetero_graph[graph_name].pos.cpu().detach().numpy()[:,0] < 1100) & (hetero_graph[graph_name].pos.cpu().detach().numpy()[:,1] > 100)
        print(f"{graph_name}: {het_graph_rel_pos.sum()}")
        het_graph_rel_pos_dict[graph_name] = het_graph_rel_pos

    return het_graph_rel_pos_dict


def _top_k_important_nodes_all_types(
    explanation_graph, hetero_graph, faz_node=False, top_k=100, only_positive=False
):
    """
    Returns the gradients of the top k most important nodes for each graph
    """

    import torch

    graph_names = ["graph_1", "graph_2"]
    if faz_node:
        graph_names.append("faz")

    het_graph_rel_pos_dict = {}
    het_graph_rel_weight_dict = {}
    for graph_name in graph_names:
        # filter the nodes that are above the threshold
        if only_positive:
            het_graph_weight = explanation_graph.node_mask_dict[graph_name].sum(dim=-1)
        else:
            het_graph_weight = (
                explanation_graph.node_mask_dict[graph_name].abs().sum(dim=-1)
            )
        # set the weights of the inlay to 0
        het_graph_weight = het_graph_weight.cpu().detach().numpy()
        inlay_pos = (
            hetero_graph[graph_name].pos.cpu().detach().numpy()[:, 0] > 1100
        ) & (hetero_graph[graph_name].pos.cpu().detach().numpy()[:, 1] < 100)
        het_graph_weight[inlay_pos] = 0
        het_graph_weight = torch.from_numpy(het_graph_weight)
        if graph_name == "faz":
            _, indices = torch.topk(het_graph_weight, 1)
            het_graph_rel_pos = torch.zeros(het_graph_weight.shape, dtype=torch.bool)
            het_graph_rel_pos[indices] = True
            if only_positive:
                het_graph_rel_pos = het_graph_rel_pos & (het_graph_weight > 0)
            het_graph_rel_pos = het_graph_rel_pos.cpu().detach().numpy()
        else:
            _, indices = torch.topk(het_graph_weight, top_k)
            het_graph_rel_pos = torch.zeros(het_graph_weight.shape, dtype=torch.bool)
            het_graph_rel_pos[indices] = True
            if only_positive:
                het_graph_rel_pos = het_graph_rel_pos & (het_graph_weight > 0)
            het_graph_rel_pos = het_graph_rel_pos.cpu().detach().numpy()
        het_graph_rel_pos_dict[graph_name] = het_graph_rel_pos
        het_graph_rel_weight_dict[graph_name] = het_graph_weight.cpu().detach().numpy()

    return het_graph_rel_pos_dict


def identifiy_relevant_nodes(
    explanation_graph,
    hetero_graph,
    explained_gradient,
    faz_node=False,
    only_positive=False,
):
    if only_positive:
        threshold = _calculate_adaptive_node_threshold_pos(
            explanation_graph, explained_gradient=explained_gradient
        )
        if threshold < 0:
            threshold = 0
    else:
        threshold = _calculate_adaptive_node_threshold(
            explanation_graph, explained_gradient=explained_gradient
        )

    graph_names = ["graph_1", "graph_2"]
    if faz_node:
        graph_names.append("faz")

    het_graph_rel_pos_dict = {}
    for graph_name in graph_names:
        # filter the nodes that are above the threshold
        if only_positive:
            het_graph_rel_pos = (
                explanation_graph.node_mask_dict[graph_name].sum(dim=-1) > threshold
            )
        else:
            het_graph_rel_pos = (
                explanation_graph.node_mask_dict[graph_name].abs().sum(dim=-1)
                > threshold
            )

        het_graph_rel_pos = het_graph_rel_pos.cpu().detach().numpy()
        # data should not contain this information anymore
        # remove the inlay
        # pos to cpu and detach
        het_graph_pos = hetero_graph[graph_name].pos.cpu().detach().numpy()
        het_graph_rel_pos = (
            het_graph_rel_pos
            & (het_graph_pos[:, 0] < 1100)
            & (het_graph_pos[:, 1] > 100)
        )
        het_graph_rel_pos_dict[graph_name] = het_graph_rel_pos
    return het_graph_rel_pos_dict


def _calculate_adaptive_node_threshold(explanation_graph, explained_gradient=0.95):
    """
    Calculate the threshold for the node importance such that 95% of the total gradient is explained (without the gradients from nodes that contribute less than 0.05% to the total gradient)
    """
    import copy

    import torch

    abs_dict = {}
    for key in explanation_graph.node_mask_dict.keys():
        abs_dict[key] = copy.deepcopy(torch.abs(explanation_graph.node_mask_dict[key]))

    total_grad = 0
    for key in explanation_graph.node_mask_dict.keys():
        total_grad += abs_dict[key].abs().sum()
    node_value = torch.cat(
        [
            abs_dict[key].abs().sum(dim=-1)
            for key in explanation_graph.node_mask_dict.keys()
        ],
        dim=0,
    )
    sorted_node_value = torch.sort(node_value, descending=True)[0]
    # get rid of the nodes that contribute less than 0.05% to the total gradient
    sorted_node_value = sorted_node_value[sorted_node_value > 0.0005 * total_grad]
    cum_sum = torch.cumsum(sorted_node_value, dim=0)
    cropped_grad = cum_sum[-1]
    threshold = sorted_node_value[cum_sum < explained_gradient * cropped_grad][-1]  # 30

    return threshold


def _calculate_adaptive_node_threshold_pos(explanation_graph, explained_gradient=0.95):
    """
    Calculate the threshold for the node importance such that 95% of the total gradient is explained (without the gradients from nodes that contribute less than 0.05% to the total gradient)
    Only consider the positive gradients
    """
    import copy

    import torch

    abs_dict = {}
    for key in explanation_graph.node_mask_dict.keys():
        abs_dict[key] = copy.deepcopy(explanation_graph.node_mask_dict[key])

    total_pos_grad = 0
    for key in explanation_graph.node_mask_dict.keys():
        grad = abs_dict[key].sum(dim=-1)
        grad[grad < 0] = 0
        total_pos_grad += grad.sum()
    node_value = torch.cat(
        [abs_dict[key].sum(dim=-1) for key in explanation_graph.node_mask_dict.keys()],
        dim=0,
    )
    node_value[node_value < 0] = 0
    sorted_node_value = torch.sort(node_value, descending=True)[0]
    # get rid of the nodes that contribute less than 0.05% to the total gradient
    sorted_node_value = sorted_node_value[sorted_node_value > 0.0005 * total_pos_grad]
    cum_sum = torch.cumsum(sorted_node_value, dim=0)
    cropped_grad = cum_sum[-1]
    threshold = sorted_node_value[cum_sum < explained_gradient * cropped_grad][-1]  # 30

    return threshold


def _remove_inlay(hetero_graph, work_dict, node_key):
    """
    Remove the inlay from the node masks
    """

    # remove the inlay
    pos = hetero_graph[node_key].pos.cpu().detach().numpy()
    remove_pos = (pos[:, 0] > 1100) & (pos[:, 1] < 100)
    # again setting 0 does not affect the sum
    work_dict[node_key][remove_pos, :] = 0

    return work_dict

def adaptive_important_nodes(explanation_graph, hetero_graph):
    """
    Returns the gradients of the nodes that are above the threshold for each graph, and features of these nodes
    """
    import copy

    import torch

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


def store_relevant_nodes_csv(
    explanation_graph,
    hetero_graph,
    path,
    features_label_dict,
    faz_node=False,
    explained_gradient=0.95,
):
    import pandas as pd

    het_graph_rel_pos_dict = identifiy_relevant_nodes(
        explanation_graph,
        hetero_graph,
        explained_gradient=explained_gradient,
        faz_node=faz_node,
    )

    data_label = hetero_graph.graph_id[0]
    # get the features and gradients of the relevant nodes
    feature_dict = {}
    grad_dict = {}
    for key in explanation_graph.node_mask_dict.keys():
        feature_dict[key] = hetero_graph[key].x[het_graph_rel_pos_dict[key]]
        grad_dict[key] = explanation_graph.node_mask_dict[key][
            het_graph_rel_pos_dict[key]
        ]
    # get the gradients of the relevant nodes

    # write the relevant nodes to a csv file
    for key in explanation_graph.node_mask_dict.keys():
        df = pd.DataFrame(
            feature_dict[key].cpu().detach().numpy(), columns=features_label_dict[key]
        )
        df.to_csv(path + f"relevant_nodes_{key}_{data_label}_features.csv", index=False)
        df = pd.DataFrame(
            grad_dict[key].cpu().detach().numpy(), columns=features_label_dict[key]
        )
        df.to_csv(
            path + f"relevant_nodes__{key}_{data_label}_gradients.csv", index=False
        )
