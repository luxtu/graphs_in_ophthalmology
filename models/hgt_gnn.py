import torch
#from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import Linear, HeteroDictLinear, HGTConv
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool

class HGT_GNN(torch.nn.Module):
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
                 num_heads = 2,
                 batch_norm = True, 
                 faz_node = False,
                 global_node = False,
                 activation = F.relu,
                 start_rep = False,
                 aggr_faz = False,
                 ):
        super().__init__()
        torch.manual_seed(1234567)

        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.meta_data = meta_data

        self.dropout = dropout
        self.aggregation_mode = aggregation_mode
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.faz_node = faz_node
        if self.faz_node:
            self.node_types = node_types + ["faz"] #+ ["global"]
        else:
            self.node_types = node_types

        self.global_node = global_node
        if self.global_node:
            self.node_types = self.node_types + ["global"]
        
        self.activation = activation

        self.start_rep = start_rep
        self.aggr_faz = aggr_faz

        self.num_pre_processing_layers = num_pre_processing_layers
        self.pre_processing_lin_layers = torch.nn.ModuleList()
        self.pre_processing_batch_norm = torch.nn.ModuleList()
        self.num_final_lin_layers = num_final_lin_layers


        self.final_lin_layers = torch.nn.ModuleList()



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

        self.num_heads = num_heads
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        #self.han_convs = torch.nn.ModuleList()
        for _ in range(self.num_layers):
            conv = HGTConv(-1, hidden_channels, metadata=self.meta_data, heads = self.num_heads)
            self.convs.append(conv)
            #han_conv = HANConv(-1, hidden_channels, metadata=meta_data, heads = 1)
            #self.han_convs.append(han_conv)





    def forward(self, x_dict, edge_index_dict, batch_dict, grads = False, **kwargs):

        if self.start_rep:

        # for each node type, aggregate over all nodes of that type
            start_representations = []
            for key in x_dict.keys():
                if isinstance(self.aggregation_mode, list):
                    for j in range(len(self.aggregation_mode)):
                        # check if the aggregation mode is global_add_pool
                        if self.aggregation_mode[j] == global_add_pool:
                            start_representations.append(self.aggregation_mode[j](x_dict[key], batch_dict[key])/1000) 
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

            for node_type in self.node_types:
                try:
                    x[node_type] = x[node_type] + x_old[node_type] # skip connection
                    #relu after skip connection
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



        #return self.attention_pooling_classifier(x, batch_dict, kwargs["pos_dict"])



        # for each node type, aggregate over all nodes of that type
        type_specific_representations = []
        # no aggregation for the faz node
        if self.aggr_faz:
            aggr_keys = ["graph_1", "graph_2", "faz"]
        else:
            aggr_keys = ["graph_1", "graph_2", ]

        for key in aggr_keys:
            if isinstance(self.aggregation_mode, list):
                for j in range(len(self.aggregation_mode)):
                    # check if the aggregation mode is global_add_pool
                    if self.aggregation_mode[j] == global_add_pool:
                        type_specific_representations.append(self.aggregation_mode[j](x[key], batch_dict[key])/1000) 
                    else:
                        type_specific_representations.append(self.aggregation_mode[j](x[key], batch_dict[key]))
            else:
                type_specific_representations.append(self.aggregation_mode(x[key], batch_dict[key]))

            #rep = self.aggregation_mode(x[key], batch_dict[key])
            #type_specific_representations.append(rep)


        x = torch.cat(type_specific_representations, dim=1)  
        if self.start_rep:
            x = torch.cat([x] + start_representations, dim=1)
        #x_start = torch.cat(start_representations, dim=1)
        # strong dropout for the start representation, avoid overrelaince on the start representation
        #x_start = F.dropout(x_start, p=0.5, training = self.training)
        #x = x_start
        #x = torch.cat([x_start, x], dim=1)
        #x = F.dropout(x, p=self.dropout, training = self.training)

        for lin in self.final_lin_layers:
            x = lin(x)
            if lin != self.final_lin_layers[-1]:
                x = self.activation(x)
                x = F.dropout(x, p=self.dropout, training = self.training)
        
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

