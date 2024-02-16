import torch
#from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, GraphConv, HeteroLinear, Linear, GATConv, EdgeConv, HeteroDictLinear, HANConv, HeteroBatchNorm
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool

class GNN_global_node(torch.nn.Module):
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


        
        self.cat_comps = torch.nn.ModuleList()

        self.convs = torch.nn.ModuleList()
        #self.han_convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            if self.faz_node:
                conv = self.conv_123(self.hidden_channels, self.conv_aggr, self.homogeneous_conv, self.heterogeneous_conv)
            else:
                conv = self.conv_12(self.hidden_channels, self.conv_aggr, self.homogeneous_conv, self.heterogeneous_conv)

            self.convs.append(conv)
            if self.conv_aggr == "cat":
                self.cat_comps.append(HeteroDictLinear(-1, hidden_channels, types= self.node_types))

            #han_conv = HANConv(-1, hidden_channels, metadata=meta_data, heads = 1)
            #self.han_convs.append(han_conv)

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



    def attention_pooling_classifier(self, x_dict, batch_dict, pos_dict):
        """
        Takes the refined embeedings and pools them into a 6 x 6 grid, the pooling is done with the self.attention_mode function.
        For the spatial pooling the pos_dict information is used, also only nodes from the graph_1 and graph_2 are pooled. The batch dict is used to identify the nodes from the two graphs.
        The pooled embeddings can then attend to each other and the resulting embeddings are used for classification.
        """

        def find_patch_label(x, y, patch_size, num_patches):
            patch_x = (x // patch_size).clamp(0, num_patches-1)
            patch_y = (y // patch_size).clamp(0, num_patches-1)

            # make the patch labels into ints
            patch_x = patch_x.int()
            patch_y = patch_y.int()
            # cannot be larger than 5, and not smaller than 0
            #patch_x = torch.clamp(patch_x, 0, 5)
            #patch_y = torch.clamp(patch_y, 0, 5)
            return patch_x, patch_y

        # Define the size of the larger grid
        grid_size = 1216

        # Define the size of the patch
        patch_size = 202

        # Calculate the number of patches in each dimension
        num_patches = grid_size // patch_size

        labels = torch.zeros((num_patches, num_patches), dtype=torch.long, device=self.device)
        label_counter = 0
        for i in range(num_patches):
            for j in range(num_patches):
                labels[i, j] = label_counter
                label_counter += 1


        pooled_node_type_representations = []
        node_types = ["graph_1", "graph_2"]

        # get the batch size from the unique elements in the batch
        batch_size = torch.unique(batch_dict["graph_1"]).shape[0]
        #print("Batch size: ", batch_size)

        for key in node_types:
            
            node_positions = pos_dict[key].clone().requires_grad_(False)
            node_patch_labels = torch.zeros(node_positions.shape[0], dtype=torch.long, device=self.device)
            # adjust position to -600 to 600 for both x and y axis
            #node_positions -= 600

            x = node_positions[:, 0]
            y = node_positions[:, 1]
            # print the max and min of the x and y coordinates
            #print("Max x: ", torch.max(x))
            #print("Min x: ", torch.min(x))
            #print("Max y: ", torch.max(y))
            #print("Min y: ", torch.min(y))
            patch_x, patch_y = find_patch_label(x, y, patch_size, num_patches)
            node_patch_labels = labels[patch_x, patch_y]

            #for i in range(node_positions.shape[0]):
            #    x = node_positions[i, 0]
            #    y = node_positions[i, 1]
            #    patch_x, patch_y = find_patch_label(x, y, patch_size)
            #    # turn the patche indice into ints
            #    patch_x = patch_x.int()
            #    patch_y = patch_y.int()
            #    node_patch_labels[i] = labels[patch_x, patch_y]
                
            pooled_representations = []
            for patch in range(num_patches**2):
                # print the type of dtype of the index vector
                if isinstance(self.aggregation_mode, list):
                    aggr_list_rep = []
                    for j in range(len(self.aggregation_mode)):
                        # check if the aggregation mode is global_add_pool
                        if self.aggregation_mode[j] == global_add_pool:
                            #pool the nodes from the same patch together
                            aggr_list_rep.append(self.aggregation_mode[j](x_dict[key][node_patch_labels == patch], batch_dict[key][node_patch_labels == patch])) # /1000
                        else:
                            aggr_list_rep.append(self.aggregation_mode[j](x_dict[key][node_patch_labels == patch], batch_dict[key][node_patch_labels == patch]))
                        # concat the pooled representations from the different aggregation modes, axis is the feature dimension
                    pooled_representations.append(torch.cat(aggr_list_rep, dim=1))
                else:
                    pooled_representations.append(self.aggregation_mode(x_dict[key][node_patch_labels == patch], batch_dict[key][node_patch_labels == patch]))
                
                if pooled_representations[-1].shape[0] != batch_size:
                    # identify the sample in the batch that has no node in the patch
                    #print(batch_dict[key][node_patch_labels == patch])
                    #print(batch_dict[key][node_patch_labels == patch].shape)
                    # use unique to find the unique elements in the batch
                    unique_batch = torch.unique(batch_dict[key][node_patch_labels == patch])
                    #print(unique_batch)
                    new_batch = torch.zeros((batch_size, pooled_representations[-1][0].shape[0]), dtype=torch.long, device=self.device)
                    # check which element is missing
                    old_batch_idx = 0
                    for i in range(batch_size):
                        if i not in unique_batch:
                            print("Missing element in batch: ", i)
                            # add a zero tensor at the correct position to the new batch
                            new_batch[i] = torch.zeros_like(pooled_representations[-1][0])
                        else:
                            new_batch[i] = pooled_representations[-1][old_batch_idx]
                            old_batch_idx += 1
                    pooled_representations[-1] = new_batch







                            #if pooled_representations[-1].shape[0] == batch_size - 1:
                            #    # add a zero tensor at the correct position to pooled_representations[-1]
                            #    pooled_representations[-1] = torch.cat([pooled_representations[-1][:i], torch.zeros_like(pooled_representations[-1][0]).unsqueeze(0)], dim=0)
                            #else:
                            #    pooled_representations[-1] = torch.cat([pooled_representations[-1][:i], torch.zeros_like(pooled_representations[-1][0]).unsqueeze(0), pooled_representations[-1][i:]], dim=0)
                            #    print(pooled_representations[-1].shape)
                            # add a zero tensor at the correct position to pooled_representations[-1]
                            #pooled_representations[-1][i] = torch.zeros_like(pooled_representations[-1][i])



                    #raise ValueError("Sth went wrong")


                    #raise ValueError("Sth went wrong")
            # for a pooled representation, the first dimension is the number of patches, the second dimension is the number of patches, the third dimension is the number of features
            # check that for each patch there is an embedding for each sample in the batch
            #print(pooled_representations[0].shape)



            # first dim is the number of patches (must be 36), second dim is the batch size, third dim is the number of features

            pooled_node_type_representations.append(torch.stack(pooled_representations, dim=0))



        
        # concatenate the pooled representations from the two node types
        pooled_node_type_representations = torch.cat(pooled_node_type_representations, dim=2)
        #print(pooled_node_type_representations.shape)

        # flip the second dimension, so that the batch size is the first dimension
        pooled_node_type_representations = pooled_node_type_representations.transpose(0, 1)
        embedded = self.embedding(pooled_node_type_representations)
        #print(embedded.shape)

        # Apply attention mechanism
        attention_weights = F.softmax(self.attention(embedded), dim=1)
        attended = torch.sum(attention_weights * embedded, dim=1)

        # Classification using fully connected layer
        output = self.fc(attended)

        return output


        # allow simple attention between the patches


        
      

        
    def forward_core(self, x_dict, edge_index_dict,  grads):
        # copy the input dict
        x_dict = x_dict.copy()
        # require gradients for the input
        #if grads:
        #    for key in x_dict.keys():
        #        x_dict[key].requires_grad = True
#
        #self.start_conv_acts_1 = x_dict["graph_1"]
        #self.start_conv_acts_2 = x_dict["graph_2"]
#
        #if grads:
        #    # register hooks for the gradients
        #    self.start_conv_acts_1.register_hook(self.activations_hook_start_1)
        #    self.start_conv_acts_2.register_hook(self.activations_hook_start_2)


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

        if self.faz_node:
            self.final_conv_acts_3 = x_dict["faz"]

        if grads:
            # register hooks for the gradients
            self.final_conv_acts_1.register_hook(self.activations_hook_1)
            self.final_conv_acts_2.register_hook(self.activations_hook_2)
            if self.faz_node:
                self.final_conv_acts_3.register_hook(self.activations_hook_3)

        # no relu after the last layer / before the pooling

        return x_dict
    

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
    