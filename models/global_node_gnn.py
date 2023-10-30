import torch
#from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, GraphConv, HeteroLinear,Linear, GATConv, EdgeConv, HeteroDictLinear, HANConv


class GNN_global_node(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers, dropout, aggregation_mode ,node_types, meta_data = None):
        super().__init__()
        torch.manual_seed(1234567)

        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.aggregation_mode = aggregation_mode
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.pre_lin_dict = torch.nn.ModuleDict()
        for node_type in node_types:
            self.pre_lin_dict[node_type] = Linear(-1, hidden_channels)

        self.post_lin_dict = torch.nn.ModuleDict()
        for node_type in node_types:
            self.post_lin_dict[node_type] = Linear(-1, hidden_channels)


        self.final_conv_acts = {"graph_1": None, "graph_2": None}
        self.final_conv_acts_1 = None
        self.final_conv_acts_2 = None
        self.final_conv_grads_1 = None
        self.final_conv_grads_2 = None

        self.node_types = node_types    
        
        self.cat_comps = torch.nn.ModuleList()

        self.convs = torch.nn.ModuleList()
        self.han_convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('graph_1', 'to', 'graph_1'): GATConv(-1, hidden_channels), # , aggr = ["mean", "std", "max"],
                ('graph_2', 'to', 'graph_2'): GATConv(-1, hidden_channels),
                ('graph_1', 'to', 'graph_2'): GATConv((-1, -1), hidden_channels, add_self_loops=False),
                ('graph_2', 'rev_to', 'graph_1'): GATConv((-1, -1), hidden_channels, add_self_loops=False),
                ('global', 'to', 'graph_1'): GATConv((-1,-1), hidden_channels, add_self_loops=False), #
                ('global', 'to', 'graph_2'): GATConv((-1, -1), hidden_channels, add_self_loops=False), #
                ('global', 'to', 'global'): GATConv(-1,hidden_channels), 
            }, aggr='cat')
            self.convs.append(conv)
            self.cat_comps.append(HeteroDictLinear(-1, hidden_channels, types= node_types + ["global"]))

            han_conv = HANConv(-1, hidden_channels, metadata=meta_data, heads = 8)

            self.han_convs.append(han_conv)



        # linear layers for skip connections
        #self.skip_lin = torch.nn.ModuleList()
        #for _ in range(num_layers):
        #    lin = torch.nn.ModuleDict()
        #    for node_type in self.node_types:
        #        lin[node_type] = Linear(-1, hidden_channels)
        #    self.skip_lin.append(lin)

        # createa batch norm layer for each node type
        self.batch_norm_dict_pre_conv = torch.nn.ModuleDict()
        for node_type in self.node_types:
            self.batch_norm_dict_pre_conv[node_type] = torch.nn.BatchNorm1d(hidden_channels)

        self.batch_norm_dict_post_conv = torch.nn.ModuleDict()
        for node_type in self.node_types:
            self.batch_norm_dict_post_conv[node_type] = torch.nn.BatchNorm1d(hidden_channels)


        self.lin1 = Linear(-1, hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)


    def forward(self, x_dict, edge_index_dict, batch_dict, slice_dict, training = False, grads = False):


        x_dict = self.forward_core(x_dict, edge_index_dict,training = training, grads = grads)

        # for each node type, aggregate over all nodes of that type
        type_specific_representations = []
        for key, x in x_dict.items():
            rep = self.aggregation_mode(x_dict[key], batch_dict[key])
            type_specific_representations.append(rep)


        x = torch.cat(type_specific_representations, dim=1)  
        x = F.dropout(x, p=self.dropout, training = training)


        return self.lin2(self.lin1(x).relu())


        
    def forward_core(self, x_dict, edge_index_dict, training,  grads):
        # linear layer batchnorm and relu
        # copy the input dict
        x_dict = x_dict.copy()

        for node_type in self.node_types:
            x_dict[node_type] = self.batch_norm_dict_pre_conv[node_type](self.pre_lin_dict[node_type](x_dict[node_type])).relu()

        # convolutions followed by relu
        for i, conv in enumerate(self.han_convs): # vs self.convs
            # apply the conv and then add the skip connections

            x_dict_old = x_dict.copy()

            x_dict = conv(x_dict, edge_index_dict)
            #x_dict = {key: x.relu() for key, x in x_dict.items()} # use with self.convs
            # cat comp
            #x_dict = self.cat_comps[i](x_dict)

            for node_type in self.node_types:
                x_dict[node_type] = x_dict[node_type] + x_dict_old[node_type] #self.skip_lin[i][node_type](x_dict_old[node_type])

        for node_type in self.node_types:
            x_dict[node_type] = self.batch_norm_dict_post_conv[node_type](self.post_lin_dict[node_type](x_dict[node_type]))

        self.final_conv_acts_1 = x_dict["graph_1"]
        self.final_conv_acts_2 = x_dict["graph_2"]

        x_dict = {key: x.relu() for key, x in x_dict.items()}
            
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
