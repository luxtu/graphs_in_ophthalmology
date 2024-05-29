import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GCNConv, Linear, GATConv, HeteroDictLinear
from torch_geometric.nn import global_add_pool

class Heterogeneous_GNN(torch.nn.Module):
    def __init__(self, hidden_channels, 
                 out_channels, 
                 num_layers, 
                 dropout, 
                 aggregation_mode ,
                 node_types, 
                 num_pre_processing_layers = 3, 
                 num_post_processing_layers = 3, 
                 num_final_lin_layers = 3,
                 batch_norm = True, 
                 conv_aggr = "cat", 
                 hetero_conns = True, 
                 faz_node = False,
                 homogeneous_conv = GCNConv,
                 heterogeneous_conv = GATConv,
                 activation = F.relu,
                 start_rep = False,
                 aggr_faz = False,
                 faz_conns = True
                 ):
        super().__init__()
        torch.manual_seed(1234567)

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
        
        self.conv_aggr = conv_aggr
        self.hetero_conns = hetero_conns

        self.homogeneous_conv = homogeneous_conv
        self.heterogeneous_conv = heterogeneous_conv
        self.activation = activation

        self.start_rep = start_rep
        self.aggr_faz = aggr_faz

        ########################################
        #pre processing MLP block

        self.num_pre_processing_layers = num_pre_processing_layers
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

        ########################################
        #post processing MLP block

        self.num_post_processing_layers = num_post_processing_layers
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

        ########################################
        #classifcation layers at the end

        self.num_final_lin_layers = num_final_lin_layers
        self.final_lin_layers = torch.nn.ModuleList()

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


        ########################################
        #GNN block
        
        self.cat_comps = torch.nn.ModuleList()
        self.convs = torch.nn.ModuleList()

        for _ in range(num_layers):
            if self.faz_node:
                conv = self.conv_gnn_faz(self.hidden_channels, self.conv_aggr, self.homogeneous_conv, self.heterogeneous_conv)
            else:
                conv = self.conv_gnn(self.hidden_channels, self.conv_aggr, self.homogeneous_conv, self.heterogeneous_conv)

            self.convs.append(conv)
            if self.conv_aggr == "cat":
                self.cat_comps.append(HeteroDictLinear(-1, hidden_channels, types= self.node_types))



        ########################################
        #UNUSED BUT NECESSARY TO LOAD CHECKPOINTS

        self.lin1 = Linear(-1, hidden_channels, bias=False) # had *2 before
        self.lin2 = Linear(hidden_channels, hidden_channels, bias=False) # had *2 before
        self.lin3 = Linear(hidden_channels, out_channels, bias=False)

        ######################################## testing attention addon
        # Embedding layer
        self.embedding = Linear(-1, hidden_channels)
        # Attention layer
        self.attention = Linear(hidden_channels, 1)
        # Fully connected layer for classification
        self.fc = Linear(hidden_channels, out_channels)




    def forward(self, x_dict, edge_index_dict, batch_dict, **kwargs):

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
        for i in range(len(self.pre_processing_lin_layers)):
            for node_type in self.node_types:
                x[node_type] = self.activation(self.pre_processing_batch_norm[i][node_type](self.pre_processing_lin_layers[i][node_type](x_dict[node_type]))) 

        #########################################
        #########################################
        # gnn convolutions
        for i, conv in enumerate(self.convs): # vs self.convs
            # apply the conv and then add the skip connections

            x_old = x.copy()
            x = conv(x, edge_index_dict)
            if self.conv_aggr == "cat":
                x = self.cat_comps[i](x)

            for node_type in self.node_types:
                try:
                    x[node_type] = x[node_type] + x_old[node_type] # skip connection
                    #activation after skip connection
                    x[node_type] = self.activation(x[node_type])
                except KeyError:
                    pass

        #########################################
        ########################################
        #post processing
        for i in range(len(self.post_processing_lin_layers)):
            for node_type in self.node_types:
                try:
                    x[node_type] = self.post_processing_batch_norm[i][node_type](self.post_processing_lin_layers[i][node_type](x[node_type]))
                except KeyError:
                    pass

            # activation if not last layer
            if i != len(self.post_processing_lin_layers) - 1:
                for node_type in self.node_types:
                    try:
                        x[node_type] = self.activation(x[node_type])
                    except KeyError:
                        pass

        #########################################
        ########################################
        #aggregation of node types
        type_specific_representations = []
        # no aggregation necessary for the single faz node
        if self.aggr_faz:
            aggr_keys = ["graph_1", "graph_2", "faz"]
        else:
            aggr_keys = ["graph_1", "graph_2", ]

        for key in aggr_keys:
            if isinstance(self.aggregation_mode, list):
                for j in range(len(self.aggregation_mode)):
                    # check if the aggregation mode is global_add_pool
                    if self.aggregation_mode[j] == global_add_pool:
                        type_specific_representations.append(self.aggregation_mode[j](x[key], batch_dict[key]))  # /1000 
                    else:
                        type_specific_representations.append(self.aggregation_mode[j](x[key], batch_dict[key]))
            else:
                type_specific_representations.append(self.aggregation_mode(x[key], batch_dict[key]))

            #rep = self.aggregation_mode(x[key], batch_dict[key])
            #type_specific_representations.append(rep)


        x = torch.cat(type_specific_representations, dim=1)  
        if self.start_rep:
            x = torch.cat([x] + start_representations, dim=1)

        #########################################
        ########################################
        #final linear layers for classification
        for lin in self.final_lin_layers:
            x = lin(x)
            if lin != self.final_lin_layers[-1]:
                x = self.activation(x)
                x = F.dropout(x, p=self.dropout, training = self.training)
        
        return x





    def conv_gnn(self, hidden_channels, conv_aggr, homogeneous_conv, heterogeneous_conv):
        if self.hetero_conns:
            conv = HeteroConv({
                ('graph_1', 'to', 'graph_1'): homogeneous_conv(-1, hidden_channels), # , aggr = ["mean", "std", "max"],
                ('graph_2', 'to', 'graph_2'): homogeneous_conv(-1, hidden_channels),
                ('graph_1', 'to', 'graph_2'): heterogeneous_conv((-1, -1), hidden_channels, add_self_loops=False),
                ('graph_2', 'rev_to', 'graph_1'): heterogeneous_conv((-1, -1), hidden_channels, add_self_loops=False),
            }, aggr=conv_aggr)                
        else:
            conv = HeteroConv({
                ('graph_1', 'to', 'graph_1'): homogeneous_conv(-1, hidden_channels), # , aggr = ["mean", "std", "max"],
                ('graph_2', 'to', 'graph_2'): homogeneous_conv(-1, hidden_channels),
            }, aggr=conv_aggr)
        return conv


    def conv_gnn_faz(self, hidden_channels, conv_aggr, homogeneous_conv, heterogeneous_conv):
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
    