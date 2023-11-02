import torch
#from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, GraphConv, HeteroLinear, Linear, GATConv, EdgeConv, HeteroDictLinear, HANConv, HeteroBatchNorm
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool

class GNN_global_node(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers, dropout, aggregation_mode ,node_types, meta_data = None, num_pre_processing_layers = 3, num_post_processing_layers = 3, batch_norm = True, conv_aggr = "cat", hetero_conns = True):
        super().__init__()
        torch.manual_seed(1234567)

        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.aggregation_mode = aggregation_mode
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.node_types = node_types + ["global"]
        self.conv_aggr = conv_aggr


        self.num_pre_processing_layers = 3
        self.pre_processing_lin_layers = torch.nn.ModuleList()
        self.pre_processing_batch_norm = torch.nn.ModuleList()

        for _ in range(self.num_pre_processing_layers):
            pre_lin_dict = torch.nn.ModuleDict()
            pre_batch_norm_dict = torch.nn.ModuleDict()
            for node_type in self.node_types:
                pre_lin_dict[node_type] = Linear(-1, hidden_channels)
                if batch_norm:
                    pre_batch_norm_dict[node_type] = torch.nn.BatchNorm1d(hidden_channels)
                else:
                    pre_batch_norm_dict[node_type] = torch.nn.Identity()
                
            self.pre_processing_lin_layers.append(pre_lin_dict)
            self.pre_processing_batch_norm.append(pre_batch_norm_dict)

        self.num_post_processing_layers = 3
        self.post_processing_lin_layers = torch.nn.ModuleList()
        self.post_processing_batch_norm = torch.nn.ModuleList()

        for _ in range(self.num_post_processing_layers):
            post_lin_dict = torch.nn.ModuleDict()
            post_batch_norm_dict = torch.nn.ModuleDict()
            for node_type in self.node_types:
                post_lin_dict[node_type] = Linear(-1, hidden_channels)
                if batch_norm:
                    post_batch_norm_dict[node_type] = torch.nn.BatchNorm1d(hidden_channels)
                else:
                    post_batch_norm_dict[node_type] = torch.nn.Identity()
            self.post_processing_lin_layers.append(post_lin_dict)
            self.post_processing_batch_norm.append(post_batch_norm_dict)


        self.final_conv_acts = {"graph_1": None, "graph_2": None}
        self.final_conv_acts_1 = None
        self.final_conv_acts_2 = None
        self.final_conv_grads_1 = None
        self.final_conv_grads_2 = None

        self.start_conv_acts = {"graph_1": None, "graph_2": None}
        self.start_conv_acts_1 = None
        self.start_conv_acts_2 = None
        self.start_conv_grads_1 = None
        self.start_conv_grads_2 = None


        
        self.cat_comps = torch.nn.ModuleList()

        self.convs = torch.nn.ModuleList()
        #self.han_convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            if hetero_conns:
                conv = HeteroConv({
                    ('graph_1', 'to', 'graph_1'): GCNConv(-1, hidden_channels), # , aggr = ["mean", "std", "max"],
                    ('graph_2', 'to', 'graph_2'): GCNConv(-1, hidden_channels),
                    ('graph_1', 'to', 'graph_2'): GATConv((-1, -1), hidden_channels, add_self_loops=False),
                    ('graph_2', 'rev_to', 'graph_1'): GATConv((-1, -1), hidden_channels, add_self_loops=False),
                    ('global', 'to', 'graph_1'): GATConv((-1,-1), hidden_channels, add_self_loops=False), #
                    ('global', 'to', 'graph_2'): GATConv((-1, -1), hidden_channels, add_self_loops=False), #
                    ('global', 'to', 'global'): GCNConv(-1,hidden_channels), 
                }, aggr=self.conv_aggr)
            else:
                conv = HeteroConv({
                    ('graph_1', 'to', 'graph_1'): GCNConv(-1, hidden_channels), # , aggr = ["mean", "std", "max"],
                    ('graph_2', 'to', 'graph_2'): GCNConv(-1, hidden_channels),
                    ('global', 'to', 'global'): GCNConv(-1,hidden_channels), 
                }, aggr=self.conv_aggr)

            self.convs.append(conv)

            if self.conv_aggr == "cat":
                self.cat_comps.append(HeteroDictLinear(-1, hidden_channels, types= self.node_types))

            #han_conv = HANConv(-1, hidden_channels, metadata=meta_data, heads = 1)

            #self.han_convs.append(han_conv)

        self.lin1 = Linear(-1, hidden_channels*2)
        self.lin2 = Linear(hidden_channels*2, hidden_channels)
        self.lin3 = Linear(hidden_channels, out_channels)


    def forward(self, x_dict, edge_index_dict, batch_dict, grads = False, **kwargs):

        x_dict = self.forward_core(x_dict, edge_index_dict, grads = grads)

        # for each node type, aggregate over all nodes of that type
        type_specific_representations = []
        for key, x in x_dict.items():
            if isinstance(self.aggregation_mode, list):
                for j in range(len(self.aggregation_mode)):
                    # check if the aggregation mode is global_add_pool
                    if self.aggregation_mode[j] == global_add_pool:
                        type_specific_representations.append(self.aggregation_mode[j](x_dict[key], batch_dict[key])/1000) 
                    else:
                        type_specific_representations.append(self.aggregation_mode[j](x_dict[key], batch_dict[key]))
            else:
                type_specific_representations.append(self.aggregation_mode(x_dict[key], batch_dict[key]))

            #rep = self.aggregation_mode(x_dict[key], batch_dict[key])
            #type_specific_representations.append(rep)


        x = torch.cat(type_specific_representations, dim=1)  
        x = F.dropout(x, p=self.dropout, training = self.training)


        return self.lin3(self.lin2(self.lin1(x).relu()).relu())


        
    def forward_core(self, x_dict, edge_index_dict,  grads):
        # copy the input dict
        x_dict = x_dict.copy()
        # require gradients for the input
        if grads:
            for key in x_dict.keys():
                x_dict[key].requires_grad = True

        self.start_conv_acts_1 = x_dict["graph_1"]
        self.start_conv_acts_2 = x_dict["graph_2"]

        if grads:
            # register hooks for the gradients
            self.start_conv_acts_1.register_hook(self.activations_hook_start_1)
            self.start_conv_acts_2.register_hook(self.activations_hook_start_2)


        ########################################
        #pre processing
        for i in range(len(self.pre_processing_lin_layers)):
            for node_type in self.node_types:
                x_dict[node_type] = self.pre_processing_batch_norm[i][node_type](self.pre_processing_lin_layers[i][node_type](x_dict[node_type])).relu()
        #########################################
        #########################################
        # gnn convolutions
        for i, conv in enumerate(self.convs): # vs self.convs
            # apply the conv and then add the skip connections

            x_dict_old = x_dict.copy()
            x_dict = conv(x_dict, edge_index_dict)
            if self.conv_aggr == "cat":
                x_dict = self.cat_comps[i](x_dict)

            for node_type in self.node_types:
                x_dict[node_type] = x_dict[node_type] + x_dict_old[node_type] # skip connection
                #relu after skip connection
                x_dict[node_type] = x_dict[node_type].relu()

        #########################################
        ########################################
        #pre processing
        for i in range(len(self.post_processing_lin_layers)):
            for node_type in self.node_types:
                x_dict[node_type] = self.post_processing_batch_norm[i][node_type](self.post_processing_lin_layers[i][node_type](x_dict[node_type]))

            # relu if not last layer
            if i != len(self.post_processing_lin_layers) - 1:
                for node_type in self.node_types:
                    x_dict[node_type] = x_dict[node_type].relu()
        #########################################

        self.final_conv_acts_1 = x_dict["graph_1"]
        self.final_conv_acts_2 = x_dict["graph_2"]

        if grads:
            # register hooks for the gradients
            self.final_conv_acts_1.register_hook(self.activations_hook_1)
            self.final_conv_acts_2.register_hook(self.activations_hook_2)

        # no relu after the last layer / before the pooling

        return x_dict
    

    def activations_hook_1(self, grad):
        self.final_conv_grads_1 = grad

    def activations_hook_2(self, grad):
        self.final_conv_grads_2 = grad

    def activations_hook_start_1(self, grad):
        self.start_conv_grads_1 = grad

    def activations_hook_start_2(self, grad):
        self.start_conv_grads_2 = grad
