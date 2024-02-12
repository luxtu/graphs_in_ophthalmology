import torch
#from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, GraphConv, HeteroLinear, Linear, GATConv, EdgeConv, HeteroDictLinear, HANConv, HeteroBatchNorm
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool

class Nodewise_GNN(torch.nn.Module):
    def __init__(self, hidden_channels, 
                 out_channels, 
                 num_layers, 
                 dropout, 
                 aggregation_mode ,
                 node_types, 
                 meta_data = None, 
                 num_pre_processing_layers = 3, 
                 num_post_processing_layers = 3, 
                 num_final_lin_layers = 3,
                 batch_norm = True, 
                 conv_aggr = "cat", 
                 hetero_conns = True, 
                 faz_node = False,
                 global_node = False,
                 homogeneous_conv = GCNConv,
                 heterogeneous_conv = GATConv,
                 activation = F.relu,
                 start_rep = False,
                 aggr_faz = False,
                 faz_conns = True,
                 skip_connection = True,
                 ):
        super().__init__()
        torch.manual_seed(1234567)

        self.skip_connection = skip_connection
        self.faz_conns = faz_conns
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.aggregation_mode = aggregation_mode
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.unused_faz = not aggr_faz and not start_rep and not hetero_conns
        self.faz_node = faz_node
        if self.faz_node and not self.unused_faz:
            self.node_types = node_types + ["faz"] #+ ["global"]
        else:
            self.node_types = node_types

        self.global_node = global_node
        if self.global_node:
            self.node_types = self.node_types + ["global"]
        
        self.conv_aggr = conv_aggr
        self.hetero_conns = hetero_conns

        self.homogeneous_conv = homogeneous_conv
        self.heterogeneous_conv = heterogeneous_conv
        self.activation = activation

        self.start_rep = start_rep
        self.aggr_faz = aggr_faz
        if self.aggr_faz:
            self.aggr_keys = ["graph_1", "graph_2", "faz"]
        else:
            self.aggr_keys = ["graph_1", "graph_2"]

        self.num_pre_processing_layers = num_pre_processing_layers
        self.num_post_processing_layers = num_post_processing_layers
        self.num_final_lin_layers = num_final_lin_layers


        self.final_lin_layers = torch.nn.ModuleList()

        if self.num_pre_processing_layers > 0:

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


        if self.num_post_processing_layers > 0:
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

        # handle the case where there is only one layer
        if self.num_final_lin_layers == 1:
            self.final_lin_layers.append(Linear(-1, out_channels))
        else:
            for i in range(self.num_final_lin_layers):

                if i == 0:
                    self.final_lin_layers.append(Linear(-1, hidden_channels))
                elif i == self.num_final_lin_layers - 1:
                    self.final_lin_layers.append(Linear(hidden_channels, out_channels))
                else:
                    self.final_lin_layers.append(Linear(hidden_channels, hidden_channels))




        self.final_conv_acts = {"graph_1": None, "graph_2": None}
        self.final_conv_acts_1 = None
        self.final_conv_acts_2 = None
        self.final_conv_acts_3 = None

        self.final_conv_grads_1 = None
        self.final_conv_grads_2 = None
        self.final_conv_grads_3 = None

        self.start_conv_acts = {"graph_1": None, "graph_2": None}
        self.start_conv_acts_1 = None
        self.start_conv_acts_2 = None
        self.start_conv_acts_3 = None

        self.start_conv_grads_1 = None
        self.start_conv_grads_2 = None
        self.start_conv_grads_3 = None


        
        self.cat_comps = torch.nn.ModuleList()

        self.convs = torch.nn.ModuleList()
        #self.han_convs = torch.nn.ModuleList()
        for i in range(num_layers):
            if self.faz_node:
                if i == num_layers - 1:
                    conv = self.conv_123(self.out_channels, self.conv_aggr, self.homogeneous_conv, self.heterogeneous_conv)
                else:
                    conv = self.conv_12(self.hidden_channels, self.conv_aggr, self.homogeneous_conv, self.heterogeneous_conv)
            else:
                conv = self.conv_12(self.hidden_channels, self.conv_aggr, self.homogeneous_conv, self.heterogeneous_conv)

            self.convs.append(conv)
            if self.conv_aggr == "cat":
                self.cat_comps.append(HeteroDictLinear(-1, hidden_channels, types= self.node_types))




    def forward(self, x_dict, edge_index_dict, batch_dict, grads = False, **kwargs):

        if self.start_rep:

        # for each node type, aggregate over all nodes of that type
            start_representations = []
            for key in x_dict.keys():
                if isinstance(self.aggregation_mode, list):
                    for j in range(len(self.aggregation_mode)):
                        # check if the aggregation mode is global_add_pool
                        if self.aggregation_mode[j] == global_add_pool:
                            start_representations.append(self.aggregation_mode[j](x_dict[key], batch_dict[key]))  #/1000
                        else:
                            start_representations.append(self.aggregation_mode[j](x_dict[key], batch_dict[key]))
                else:
                    start_representations.append(self.aggregation_mode(x_dict[key], batch_dict[key]))
                #start_representations.append(global_mean_pool(x_dict[key], batch_dict[key]))


        x = {}

       ########################################
        #pre processing
        if self.num_pre_processing_layers > 0:
            for i in range(len(self.pre_processing_lin_layers)):
                for node_type in self.node_types:
                    x[node_type] = self.activation(self.pre_processing_batch_norm[i][node_type](self.pre_processing_lin_layers[i][node_type](x_dict[node_type]))) 
        else:
            for node_type in self.node_types:
                x[node_type] = x_dict[node_type]
        #########################################
        #########################################
        # gnn convolutions
        for i, conv in enumerate(self.convs): # vs self.convs
            # apply the conv and then add the skip connections

            if self.skip_connection:
                x_old = x.copy()
            x = conv(x, edge_index_dict)
            if self.conv_aggr == "cat":
                x = self.cat_comps[i](x)

            for node_type in self.node_types:
                try:
                    if self.skip_connection and (i>0 or self.num_pre_processing_layers>0) and i != len(self.convs)-1: # the i larger than 0 is to avoid the first layer
                        x[node_type] = x[node_type] + x_old[node_type] # skip connection
                    #relu after skip connection
                    x[node_type] = self.activation(x[node_type])
                except KeyError:
                    pass

        #########################################
        ########################################
        #post processing
        if self.num_post_processing_layers > 0:
            for i in range(len(self.post_processing_lin_layers)):
                for node_type in self.node_types:
                    try:
                        x[node_type] = self.post_processing_batch_norm[i][node_type](self.post_processing_lin_layers[i][node_type](x[node_type]))
                    except KeyError:
                        pass

                # relu if not last layer
                if i != len(self.post_processing_lin_layers) - 1:
                    for node_type in self.node_types:
                        try:
                            x[node_type] = self.activation(x[node_type])
                        except KeyError:
                            pass
        #########################################

        self.final_conv_acts_1 = x["graph_1"]
        self.final_conv_acts_2 = x["graph_2"]
        if self.faz_node:
            try:
                self.final_conv_acts_3 = x["faz"]
            except KeyError:
                pass

        if grads:
            # register hooks for the gradients
            self.final_conv_acts_1.register_hook(self.activations_hook_1)
            self.final_conv_acts_2.register_hook(self.activations_hook_2)
            if self.faz_node:
                self.final_conv_acts_3.register_hook(self.activations_hook_3)


        # for each node type, aggregate over all nodes of that type
        type_specific_representations = []

        for key in self.aggr_keys:
            type_specific_representations.append(self.aggregation_mode(x[key], batch_dict[key]))

        # mean the type specific representations
        
        x, _ = torch.max(torch.stack(type_specific_representations), dim = 0)


 
        return x

    

    def activations_hook_1(self, grad):
        self.final_conv_grads_1 = grad

    def activations_hook_2(self, grad):
        self.final_conv_grads_2 = grad

    def activations_hook_3(self, grad):
        self.final_conv_grads_3 = grad


    def activations_hook_start_1(self, grad):
        self.start_conv_grads_1 = grad

    def activations_hook_start_2(self, grad):
        self.start_conv_grads_2 = grad



    def conv_12(self, hidden_channels, conv_aggr, homogeneous_conv, heterogeneous_conv):
        if self.hetero_conns:
            if self.global_node:
                conv = HeteroConv({
                    ('graph_1', 'to', 'graph_1'): homogeneous_conv(-1, hidden_channels), # , aggr = ["mean", "std", "max"],
                    ('graph_2', 'to', 'graph_2'): homogeneous_conv(-1, hidden_channels),
                    ('graph_1', 'to', 'graph_2'): heterogeneous_conv((-1, -1), hidden_channels, add_self_loops=False),
                    ('graph_2', 'rev_to', 'graph_1'): heterogeneous_conv((-1, -1), hidden_channels, add_self_loops=False),
                    ('global', 'to', 'graph_1'): heterogeneous_conv((-1,-1), hidden_channels, add_self_loops=False), #
                    ('global', 'to', 'graph_2'): heterogeneous_conv((-1, -1), hidden_channels, add_self_loops=False), #
                    ('global', 'to', 'global'): homogeneous_conv(-1,hidden_channels), 
                }, aggr=conv_aggr)
            else:
                conv = HeteroConv({
                    ('graph_1', 'to', 'graph_1'): homogeneous_conv(-1, hidden_channels), # , aggr = ["mean", "std", "max"],
                    ('graph_2', 'to', 'graph_2'): homogeneous_conv(-1, hidden_channels),
                    ('graph_1', 'to', 'graph_2'): heterogeneous_conv((-1, -1), hidden_channels, add_self_loops=False),
                    ('graph_2', 'rev_to', 'graph_1'): heterogeneous_conv((-1, -1), hidden_channels, add_self_loops=False),
                }, aggr=conv_aggr)                
        else:
            if self.global_node:
                conv = HeteroConv({
                    ('graph_1', 'to', 'graph_1'): homogeneous_conv(-1, hidden_channels), # , aggr = ["mean", "std", "max"],
                    ('graph_2', 'to', 'graph_2'): homogeneous_conv(-1, hidden_channels),
                    ('global', 'to', 'global'): homogeneous_conv(-1,hidden_channels), 
                }, aggr=conv_aggr)
            else:
                conv = HeteroConv({
                    ('graph_1', 'to', 'graph_1'): homogeneous_conv(-1, hidden_channels), # , aggr = ["mean", "std", "max"],
                    ('graph_2', 'to', 'graph_2'): homogeneous_conv(-1, hidden_channels),
                }, aggr=conv_aggr)
        return conv


    def conv_123(self, hidden_channels, conv_aggr, homogeneous_conv, heterogeneous_conv):
        if self.hetero_conns and self.faz_conns:
            conv = HeteroConv({
                ('graph_1', 'to', 'graph_1'): homogeneous_conv(-1, hidden_channels),
                ('graph_2', 'to', 'graph_2'): homogeneous_conv(-1, hidden_channels),
                ('graph_1', 'to', 'graph_2'): heterogeneous_conv((-1, -1), hidden_channels, add_self_loops=False),
                ('graph_1', 'to', 'faz'): heterogeneous_conv((-1, -1), hidden_channels, add_self_loops=False),
                ('faz', 'to', 'graph_2'): heterogeneous_conv((-1, -1), hidden_channels, add_self_loops=False),
                ('graph_2', 'rev_to', 'graph_1'): heterogeneous_conv((-1, -1), hidden_channels, add_self_loops=False),
                ('faz', 'to', 'faz'): homogeneous_conv(-1, hidden_channels), 
                ('faz', 'rev_to', 'graph_1'): heterogeneous_conv((-1, -1), hidden_channels, add_self_loops=False),
                ('graph_2', 'rev_to', 'faz'): heterogeneous_conv((-1, -1), hidden_channels, add_self_loops=False),

            }, aggr=conv_aggr)
        elif self.hetero_conns and not self.faz_conns:
            conv = HeteroConv({
                ('graph_1', 'to', 'graph_1'): homogeneous_conv(-1, hidden_channels),
                ('graph_2', 'to', 'graph_2'): homogeneous_conv(-1, hidden_channels),
                ('faz', 'to', 'faz'): homogeneous_conv(-1, hidden_channels),
                ('graph_1', 'to', 'graph_2'): heterogeneous_conv((-1, -1), hidden_channels, add_self_loops=False),
                ('graph_2', 'rev_to', 'graph_1'): heterogeneous_conv((-1, -1), hidden_channels, add_self_loops=False),
            }, aggr=conv_aggr)

        else:
            if self.unused_faz:
                conv = HeteroConv({
                    ('graph_1', 'to', 'graph_1'): homogeneous_conv(-1, hidden_channels),
                    ('graph_2', 'to', 'graph_2'): homogeneous_conv(-1, hidden_channels),
                }, aggr=conv_aggr)
            else:
                conv = HeteroConv({
                    ('graph_1', 'to', 'graph_1'): homogeneous_conv(-1, hidden_channels),
                    ('graph_2', 'to', 'graph_2'): homogeneous_conv(-1, hidden_channels),
                    ('faz', 'to', 'faz'): homogeneous_conv(-1, hidden_channels),  
                }, aggr=conv_aggr)


        return conv
    

