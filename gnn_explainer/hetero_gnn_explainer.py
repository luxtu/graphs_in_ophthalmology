from math import sqrt
from typing import Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn.parameter import Parameter

from torch_geometric.explain import ExplainerConfig, Explanation, ModelConfig, HeteroExplanation
from torch_geometric.explain.algorithm import ExplainerAlgorithm
from torch_geometric.explain.algorithm.utils import clear_masks, set_masks, set_hetero_masks
from torch_geometric.explain.config import MaskType, ModelMode, ModelTaskLevel



class HeteroGNNExplainer(ExplainerAlgorithm):
    r"""The GNN-Explainer model from the `"GNNExplainer: Generating
    Explanations for Graph Neural Networks"
    <https://arxiv.org/abs/1903.03894>`_ paper for identifying compact subgraph
    structures and node features that play a crucial role in the predictions
    made by a GNN.

    .. note::

        For an example of using :class:`GNNExplainer`, see
        `examples/explain/gnn_explainer.py <https://github.com/pyg-team/
        pytorch_geometric/blob/master/examples/explain/gnn_explainer.py>`_,
        `examples/explain/gnn_explainer_ba_shapes.py <https://github.com/
        pyg-team/pytorch_geometric/blob/master/examples/
        explain/gnn_explainer_ba_shapes.py>`_, and `examples/explain/
        gnn_explainer_link_pred.py <https://github.com/pyg-team/
        pytorch_geometric/blob/master/examples/explain/gnn_explainer_link_pred.py>`_.

    .. note::

        The :obj:`edge_size` coefficient is multiplied by the number of nodes
        in the explanation at every iteration, and the resulting value is added
        to the loss as a regularization term, with the goal of producing
        compact explanations.
        A higher value will push the algorithm towards explanations with less
        elements.
        Consider adjusting the :obj:`edge_size` coefficient according to the
        average node degree in the dataset, especially if this value is bigger
        than in the datasets used in the original paper.

    Args:
        epochs (int, optional): The number of epochs to train.
            (default: :obj:`100`)
        lr (float, optional): The learning rate to apply.
            (default: :obj:`0.01`)
        **kwargs (optional): Additional hyper-parameters to override default
            settings in
            :attr:`~torch_geometric.explain.algorithm.GNNExplainer.coeffs`.
    """

    coeffs = {
        'edge_size': 0.005,
        'edge_reduction': 'sum',
        'node_feat_size': 1.0 , # 1 is standard
        'node_feat_reduction': 'mean',# sum doesnt make things better # mean is standard
        'edge_ent': 1.0,
        'node_feat_ent': 0.1,
        'EPS': 1e-15,
    }

    # should be fine, no changes for the hetero case
    def __init__(self, epochs: int = 100, lr: float = 0.01, **kwargs):
        super().__init__()
        self.epochs = epochs
        self.lr = lr
        self.coeffs.update(kwargs)

        self.node_mask_dict = self.hard_node_mask_dict = None # will be dicts
        self.edge_mask_dict = self.hard_edge_mask_dict = None # will be dicts

    # dicts expected for edge_index and x
    def forward(
        self,
        model: torch.nn.Module,
        x_dict: dict,
        edge_index_dict: dict,
        *,
        target: Tensor,
        index: Optional[Union[int, Tensor]] = None,
        **kwargs,
    ) -> HeteroExplanation:
        if isinstance(x_dict, Tensor) or isinstance(edge_index_dict, Tensor):
            raise ValueError(f"Only Explanations for heterogeneous graphs"
                             f"'{self.__class__.__name__}'")

        self._train(model, x_dict, edge_index_dict, target=target, index=index, **kwargs)

        # _post_process_mask() needs to be adapted to the dict case
        # adapted to the dict case
        node_mask_dict = self._post_process_mask_dict(
            self.node_mask_dict,
            self.hard_node_mask_dict,
            apply_sigmoid=True,
        )
        edge_mask_dict = self._post_process_mask_dict(
            self.edge_mask_dict,
            self.hard_edge_mask_dict,
            apply_sigmoid=True,
        )

        self._clean_model(model)

        # generate the explanation
        # previous: return Explanation(node_mask=node_mask, edge_mask=edge_mask)

        explanation = HeteroExplanation()
        explanation.set_value_dict('node_mask', node_mask_dict)
        explanation.set_value_dict('edge_mask', edge_mask_dict)

        return explanation


    def supports(self) -> bool:
        return True


    def _train(
        self,
        model: torch.nn.Module,
        x_dict: dict,
        edge_index_dict: dict,
        *,
        target: Tensor,
        index: Optional[Union[int, Tensor]] = None,
        **kwargs,
    ):
        # adapt to the dict case
        self._initialize_masks(x_dict, edge_index_dict)

        parameters = []
        if self.node_mask_dict is not None:
            # need to iterate through the dict
            for key in self.node_mask_dict.keys():
                parameters.append(self.node_mask_dict[key])
            #previous: parameters.append(self.node_mask)


        if self.edge_mask_dict is not None:
            #set_masks(model, self.edge_mask, edge_index, apply_sigmoid=True)
            # set hetero masks should be fine
            set_hetero_masks(model, self.edge_mask_dict, edge_index_dict, apply_sigmoid=True)
            for key in self.edge_mask_dict.keys():
                parameters.append(self.edge_mask_dict[key])

            #previous: parameters.append(self.edge_mask)

        optimizer = torch.optim.Adam(parameters, lr=self.lr)

        for i in range(self.epochs):
            optimizer.zero_grad()

            # need to iterate through the dict
            h = {}
            for key in x_dict.keys():
                x = x_dict[key]
                h[key] = x if self.node_mask_dict is None else x * self.node_mask_dict[key].sigmoid()


            #h = x if self.node_mask is None else x * self.node_mask.sigmoid()
            
            y_hat, y = model(h, edge_index_dict, **kwargs), target
            #y_hat, y = model(h, edge_index, **kwargs), target

            if index is not None:
                y_hat, y = y_hat[index], y[index]

            loss = self._loss(y_hat, y)

            loss.backward()
            optimizer.step()

            # In the first iteration, we collect the nodes and edges that are
            # involved into making the prediction. These are all the nodes and
            # edges with gradient != 0 (without regularization applied).
            if i == 0 and self.node_mask_dict is not None:

                for key in self.node_mask_dict.keys():
                    if self.node_mask_dict[key].grad is None:
                        raise ValueError("Could not compute gradients for node "
                                        "features. Please make sure that node "
                                        "features are used inside the model or "
                                        "disable it via `node_mask_type=None`.")
                    self.hard_node_mask_dict[key] = self.node_mask_dict[key].grad != 0.0
                

            if i == 0 and self.edge_mask_dict is not None:
                for key in edge_index_dict.keys():
                    if self.edge_mask_dict[key].grad is None:
                        raise ValueError("Could not compute gradients for edges. "
                                        "Please make sure that edges are used "
                                        "via message passing inside the model or "
                                        "disable it via `edge_mask_type=None`.")
                    self.hard_edge_mask_dict[key] = self.edge_mask_dict[key].grad != 0.0


            #previous: 
            #if i == 0 and self.node_mask is not None:
            #    if self.node_mask.grad is None:
            #        raise ValueError("Could not compute gradients for node "
            #                         "features. Please make sure that node "
            #                         "features are used inside the model or "
            #                         "disable it via `node_mask_type=None`.")
            #    self.hard_node_mask = self.node_mask.grad != 0.0
            #if i == 0 and self.edge_mask is not None:
            #    if self.edge_mask.grad is None:
            #        raise ValueError("Could not compute gradients for edges. "
            #                         "Please make sure that edges are used "
            #                         "via message passing inside the model or "
            #                         "disable it via `edge_mask_type=None`.")
            #    self.hard_edge_mask = self.edge_mask.grad != 0.0


    def _initialize_masks(self, x_dict: dict, edge_index_dict: dict):
        node_mask_type = self.explainer_config.node_mask_type
        edge_mask_type = self.explainer_config.edge_mask_type

        # check device for data in x_dict not for dict
        
        #previous: device = x_dict.device
        device = x_dict[list(x_dict.keys())[0]].device

        # create empty dicts for the node masks
        self.node_mask_dict = {}
        self.hard_node_mask_dict = {}

        # iterate through the dict
        for key in x_dict.keys():
            x = x_dict[key]

            (N, F) = x.size()
            std = 0.1
            if node_mask_type is None:
                self.node_mask_dict = None
            elif node_mask_type == MaskType.object:
                self.node_mask_dict[key] = Parameter(torch.randn(N, 1, device=device) * std)
            elif node_mask_type == MaskType.attributes:
                self.node_mask_dict[key] = Parameter(torch.randn(N, F, device=device) * std)
            elif node_mask_type == MaskType.common_attributes:
                self.node_mask_dict[key] = Parameter(torch.randn(1, F, device=device) * std)
            else:
                assert False


        
        #Previous:
        #(N, F), E = x.size(), edge_index.size(1)
#
        #std = 0.1
        #if node_mask_type is None:
        #    self.node_mask = None
        #elif node_mask_type == MaskType.object:
        #    self.node_mask = Parameter(torch.randn(N, 1, device=device) * std)
        #elif node_mask_type == MaskType.attributes:
        #    self.node_mask = Parameter(torch.randn(N, F, device=device) * std)
        #elif node_mask_type == MaskType.common_attributes:
        #    self.node_mask = Parameter(torch.randn(1, F, device=device) * std)
        #else:
        #    assert False
            

        
        # create empty dicts for the edge masks
        self.edge_mask_dict = {}
        self.hard_edge_mask_dict = {}

        # iterate through the dict
        for key in edge_index_dict.keys():
            edge_index = edge_index_dict[key]

            # get number of nodes for both node types
            N1 = x_dict[key[0]].size(0)
            N2 = x_dict[key[-1]].size(0)

            E = edge_index.size(1)

            if edge_mask_type is None:
                self.edge_mask_dict = None
            elif edge_mask_type == MaskType.object:
                std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (N1 + N2))
                self.edge_mask_dict[key] = Parameter(torch.randn(E, device=device) * std)
            else:
                assert False

    
        #Previous:
        #if edge_mask_type is None:
        #    self.edge_mask = None
        #elif edge_mask_type == MaskType.object:
        #    std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))
        #    self.edge_mask = Parameter(torch.randn(E, device=device) * std)
        #else:
        #    assert False



    def _loss(self, y_hat: Tensor, y: Tensor) -> Tensor:
        if self.model_config.mode == ModelMode.binary_classification:
            loss = self._loss_binary_classification(y_hat, y)
        elif self.model_config.mode == ModelMode.multiclass_classification:
            loss = self._loss_multiclass_classification(y_hat, y)
        elif self.model_config.mode == ModelMode.regression:
            loss = self._loss_regression(y_hat, y)
        else:
            assert False


        # check that it is not empty dict
        if self.hard_edge_mask_dict:
            assert bool(self.edge_mask_dict)
            
            for key in self.edge_mask_dict.keys():
                m = self.edge_mask_dict[key][self.hard_edge_mask_dict[key]].sigmoid()
                edge_reduce = getattr(torch, self.coeffs['edge_reduction'])
                loss = loss + self.coeffs['edge_size'] * edge_reduce(m)
                ent = -m * torch.log(m + self.coeffs['EPS']) - (
                    1 - m) * torch.log(1 - m + self.coeffs['EPS'])
                loss = loss + self.coeffs['edge_ent'] * ent.mean()

        if self.hard_node_mask_dict:
            assert bool(self.node_mask_dict)
            for key in self.node_mask_dict.keys():
                # m is the masked node feature tensor after sigmoid
                m = self.node_mask_dict[key][self.hard_node_mask_dict[key]].sigmoid()
                node_reduce = getattr(torch, self.coeffs['node_feat_reduction'])
                feat_loss = self.coeffs['node_feat_size'] * node_reduce(m)
                loss = loss + feat_loss #self.coeffs['node_feat_size'] * node_reduce(m)
                ent = -m * torch.log(m + self.coeffs['EPS']) - (
                    1 - m) * torch.log(1 - m + self.coeffs['EPS'])
                entropy_loss = self.coeffs['node_feat_ent'] * ent.mean()
                loss = loss + entropy_loss #self.coeffs['node_feat_ent'] * ent.mean()

        
        #if self.hard_edge_mask is not None:
        #    assert self.edge_mask is not None
        #    m = self.edge_mask[self.hard_edge_mask].sigmoid()
        #    edge_reduce = getattr(torch, self.coeffs['edge_reduction'])
        #    loss = loss + self.coeffs['edge_size'] * edge_reduce(m)
        #    ent = -m * torch.log(m + self.coeffs['EPS']) - (
        #        1 - m) * torch.log(1 - m + self.coeffs['EPS'])
        #    loss = loss + self.coeffs['edge_ent'] * ent.mean()
#
        #if self.hard_node_mask is not None:
        #    assert self.node_mask is not None
        #    m = self.node_mask[self.hard_node_mask].sigmoid()
        #    node_reduce = getattr(torch, self.coeffs['node_feat_reduction'])
        #    loss = loss + self.coeffs['node_feat_size'] * node_reduce(m)
        #    ent = -m * torch.log(m + self.coeffs['EPS']) - (
        #        1 - m) * torch.log(1 - m + self.coeffs['EPS'])
        #    loss = loss + self.coeffs['node_feat_ent'] * ent.mean()

        return loss
        
        


    def _clean_model(self, model):
        clear_masks(model)
        self.node_mask = self.hard_node_mask = None
        self.edge_mask = self.hard_edge_mask = None



    @staticmethod
    def _post_process_mask_dict(
        mask: Optional[dict],
        hard_mask: Optional[dict] = None,
        apply_sigmoid: bool = True,
    ) -> Optional[Tensor]:
        r""""Post processes any mask to not include any attributions of
        elements not involved during message passing.
        """
        if mask is None:
            return mask

        for key in mask.keys():
            mask[key] = mask[key].detach()
            if apply_sigmoid:
                mask[key] = mask[key].sigmoid()

        #mask = mask.detach()
        #if apply_sigmoid:
        #    mask = mask.sigmoid()

        if hard_mask is not None:
            for key in mask.keys():
                if mask[key].size(0) == hard_mask[key].size(0):
                    mask[key][~hard_mask[key]] = 0.
            #mask[~hard_mask] = 0.
        return mask