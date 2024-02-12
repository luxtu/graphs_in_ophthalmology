import torch
#from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, GraphConv, Linear, GATConv, EdgeConv, TopKPooling
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool

class Baseline_GNN(torch.nn.Module):
    def __init__(self, hidden_channels, 
                 out_channels, 
                 num_layers, 
                 dropout, 
                 aggregation_mode ,
                 num_pre_processing_layers = 3, 
                 num_post_processing_layers = 3, 
                 num_final_lin_layers = 3,
                 batch_norm = True, 
                 homogeneous_conv = GCNConv,
                 activation = F.relu,
                 ):
        super().__init__()
        torch.manual_seed(1234567)

        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.aggregation_mode = aggregation_mode
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


        self.homogeneous_conv = homogeneous_conv
        self.activation = activation

        self.num_pre_processing_layers = num_pre_processing_layers
        self.pre_processing_lin_layers = torch.nn.ModuleList()
        self.pre_processing_batch_norm = torch.nn.ModuleList()

        for _ in range(self.num_pre_processing_layers):
            if batch_norm:
                self.pre_processing_batch_norm.append(torch.nn.BatchNorm1d(hidden_channels))
            else:
                self.pre_processing_batch_norm.append(torch.nn.Identity())

            self.pre_processing_lin_layers.append(Linear(-1, hidden_channels))



        self.num_post_processing_layers = num_post_processing_layers
        self.post_processing_lin_layers = torch.nn.ModuleList()
        self.post_processing_batch_norm = torch.nn.ModuleList()

        for _ in range(self.num_post_processing_layers):
            if batch_norm:
                self.post_processing_batch_norm.append(torch.nn.BatchNorm1d(hidden_channels))
            else:
                self.post_processing_batch_norm.append(torch.nn.Identity())

            self.post_processing_lin_layers.append(Linear(-1, hidden_channels))



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


        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        #self.han_convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = homogeneous_conv(-1, hidden_channels)
            self.convs.append(conv)





    def forward(self, x, edge_index, batch,  **kwargs):



       ########################################
        #pre processing
        for i in range(len(self.pre_processing_lin_layers)):
            x = self.pre_processing_batch_norm[i](self.pre_processing_lin_layers[i](x))
            # relu if not last layer
            if i != len(self.pre_processing_lin_layers) - 1:
                x = self.activation(x)

        #########################################
        #########################################
        # gnn convolutions
        for i in range(len(self.convs)):
            # apply the conv and then add the skip connections
            # copy tensor for skip connection
            x_old = x.clone()
            x = self.convs[i](x, edge_index)
            x = x + x_old
            # relu if not last layer
            if i != len(self.convs) - 1:
                x = self.activation(x)

        #########################################
        ########################################
        #post processing
        for i in range(len(self.post_processing_lin_layers)):
            x = self.post_processing_batch_norm[i](self.post_processing_lin_layers[i](x))
            # relu if not last layer
            if i != len(self.post_processing_lin_layers) - 1:
                x = self.activation(x)

        #aggregate over all nodes
        
        if isinstance(self.aggregation_mode, list):
            x = torch.cat([self.aggregation_mode[i](x, batch) for i in range(len(self.aggregation_mode))], dim = 1)

        else:
            x = self.aggregation_mode(x, batch)


        #########################################
        #########################################
        # final linear layers
        for i in range(len(self.final_lin_layers)):
            x = self.final_lin_layers[i](x)
            if i != len(self.final_lin_layers) - 1:
                x = self.activation(x)
                x = F.dropout(x, p=self.dropout, training = self.training)
        
        return x




class Baseline_GNN_Pooling(torch.nn.Module):
    def __init__(self, hidden_channels, 
                 out_channels, 
                 num_layers, 
                 dropout, 
                 aggregation_mode ,
                 num_pre_processing_layers = 3, 
                 num_post_processing_layers = 3, 
                 num_final_lin_layers = 3,
                 batch_norm = True, 
                 homogeneous_conv = GCNConv,
                 activation = F.relu,
                 ):
        super().__init__()
        torch.manual_seed(1234567)

        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.aggregation_mode = aggregation_mode
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


        self.homogeneous_conv = homogeneous_conv
        self.activation = activation

        self.num_pre_processing_layers = num_pre_processing_layers
        self.pre_processing_lin_layers = torch.nn.ModuleList()
        self.pre_processing_batch_norm = torch.nn.ModuleList()

        for _ in range(self.num_pre_processing_layers):
            if batch_norm:
                self.pre_processing_batch_norm.append(torch.nn.BatchNorm1d(hidden_channels))
            else:
                self.pre_processing_batch_norm.append(torch.nn.Identity())

            self.pre_processing_lin_layers.append(Linear(-1, hidden_channels))



        self.num_post_processing_layers = num_post_processing_layers
        self.post_processing_lin_layers = torch.nn.ModuleList()
        self.post_processing_batch_norm = torch.nn.ModuleList()

        for _ in range(self.num_post_processing_layers):
            if batch_norm:
                self.post_processing_batch_norm.append(torch.nn.BatchNorm1d(hidden_channels))
            else:
                self.post_processing_batch_norm.append(torch.nn.Identity())

            self.post_processing_lin_layers.append(Linear(-1, hidden_channels))



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


        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.poolings = torch.nn.ModuleList()
        #self.han_convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = homogeneous_conv(-1, hidden_channels)
            self.convs.append(conv)
            self.poolings.append(TopKPooling(hidden_channels, ratio = 0.5))





    def forward(self, x, edge_index, batch,  **kwargs):



       ########################################
        #pre processing
        for i in range(len(self.pre_processing_lin_layers)):
            x = self.pre_processing_batch_norm[i](self.pre_processing_lin_layers[i](x))
            # relu if not last layer
            if i != len(self.pre_processing_lin_layers) - 1:
                x = self.activation(x)

        #########################################
        #########################################
        # gnn convolutions
        for i in range(len(self.convs)):
            # apply the conv and then add the skip connections
            # copy tensor for skip connection
            x_old = x.clone()
            x = self.convs[i](x, edge_index)
            x = x + x_old
            # relu if not last layer
            if i != len(self.convs) - 1:
                x = self.activation(x)
            x, edge_index, _, batch, _, _ = self.poolings[i](x, edge_index, batch = batch)

        #########################################
        ########################################
        #post processing
        for i in range(len(self.post_processing_lin_layers)):
            x = self.post_processing_batch_norm[i](self.post_processing_lin_layers[i](x))
            # relu if not last layer
            if i != len(self.post_processing_lin_layers) - 1:
                x = self.activation(x)

        #aggregate over all nodes
        
        if isinstance(self.aggregation_mode, list):
            x = torch.cat([self.aggregation_mode[i](x, batch) for i in range(len(self.aggregation_mode))], dim = 1)

        else:
            x = self.aggregation_mode(x, batch)


        #########################################
        #########################################
        # final linear layers
        for i in range(len(self.final_lin_layers)):
            x = self.final_lin_layers[i](x)
            if i != len(self.final_lin_layers) - 1:
                x = self.activation(x)
                x = F.dropout(x, p=self.dropout, training = self.training)
        
        return x


