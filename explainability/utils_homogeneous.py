def top_k_important_nodes(
    explanation_graph,
    graph,
    top_k=100,
    only_positive=False,
):
    """
    Returns the gradients of the top k most important nodes for each graph or across all graphs
    """
    return _top_k_important_nodes(
        explanation_graph,
        graph,
        top_k=top_k,
        only_positive=only_positive,
    )



def _top_k_important_nodes(
    explanation_graph, hetero_graph, top_k=100, only_positive=False
):
    """
    Returns the gradients of the top k most important nodes for each graph
    """

    import torch

    # filter the nodes that are above the threshold
    if only_positive:
        het_graph_weight = explanation_graph.node_mask.sum(dim=-1)
    else:
        het_graph_weight = (
            explanation_graph.node_mask.abs().sum(dim=-1)
        )
    # set the weights of the inlay to 0
    het_graph_weight = het_graph_weight.cpu().detach().numpy()
    inlay_pos = (
        hetero_graph.pos.cpu().detach().numpy()[:, 0] > 1100
    ) & (hetero_graph.pos.cpu().detach().numpy()[:, 1] < 100)
    het_graph_weight[inlay_pos] = 0
    het_graph_weight = torch.from_numpy(het_graph_weight)

    _, indices = torch.topk(het_graph_weight, top_k)
    het_graph_rel_pos = torch.zeros(het_graph_weight.shape, dtype=torch.bool)
    het_graph_rel_pos[indices] = True
    if only_positive:
        het_graph_rel_pos = het_graph_rel_pos & (het_graph_weight > 0)
    het_graph_rel_pos = het_graph_rel_pos.cpu().detach().numpy()


    return het_graph_rel_pos


def identifiy_relevant_nodes(
    explanation_graph,
    graph,
    explained_gradient,
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


    if only_positive:
        het_graph_rel_pos = (
            explanation_graph.node_mask.sum(dim=-1) > threshold
        )
    else:
        het_graph_rel_pos = (
            explanation_graph.node_mask.abs().sum(dim=-1)
            > threshold
        )

    het_graph_rel_pos = het_graph_rel_pos.cpu().detach().numpy()
    # data should not contain this information anymore
    # remove the inlay
    # pos to cpu and detach
    het_graph_pos = graph.pos.cpu().detach().numpy()
    het_graph_rel_pos = (
        het_graph_rel_pos
        & (het_graph_pos[:, 0] < 1100)
        & (het_graph_pos[:, 1] > 100)
    )


    return het_graph_rel_pos


def _calculate_adaptive_node_threshold(explanation_graph, explained_gradient=0.95):
    """
    Calculate the threshold for the node importance such that 95% of the total gradient is explained (without the gradients from nodes that contribute less than 0.05% to the total gradient)
    """
    import copy

    import torch


    work_mask = copy.deepcopy(torch.abs(explanation_graph.node_mask))

    total_grad = work_mask.sum()

    node_value = work_mask.sum(dim=-1)
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

    work_mask = copy.deepcopy(torch.abs(explanation_graph.node_mask))

    work_mask[work_mask < 0] = 0

    total_pos_grad = work_mask.sum()

    node_value = work_mask.sum(dim=-1)
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
