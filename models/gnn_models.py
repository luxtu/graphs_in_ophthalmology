import torch
#from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, GraphConv, Linear, GATConv, EdgeConv



class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, aggregation_mode):
        super(GNN, self).__init__()
        torch.manual_seed(1234567)

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.aggregation_mode = aggregation_mode

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # create the conv layers
        self.convLayers = torch.nn.ModuleList()
        for i in range(self.num_layers):
            if i == 0:
                self.convLayers.append(SAGEConv(self.in_channels, self.hidden_channels))
            else:
                self.convLayers.append(SAGEConv(self.hidden_channels, self.hidden_channels))

        #set up final layer
        self.lin = Linear(self.hidden_channels, self.out_channels)
        

    def forward(self, x, edge_index, batch, training = False):

        for conv in self.convLayers:
            x = F.dropout(x, p=self.dropout, training = training)
            x = conv(x, edge_index)
            x = F.relu(x)

        if batch is not None:
            x = self.aggregation_mode(x, batch)

        else:
            x = self.aggregation_mode(x, torch.zeros(x.shape[0], dtype= torch.int64).to(self.device))

        x = F.dropout(x, p=self.dropout, training = training) # dropout is here only applied as the last layer
        x = self.lin(x) # final linear layer/classifier
        
        return x


    def vals_without_aggregation(self, x, edge_index, batch, training = False):

        for conv in self.convLayers:
            x = F.dropout(x, p=self.dropout, training = training)
            x = conv(x, edge_index)
            x = F.relu(x)

        x = self.lin(x) # final linear layer/classifier

        x = F.softmax(x, dim = 1)
        
        return x
    

class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers, dropout, aggregation_mode ,node_types):
        super().__init__()
        torch.manual_seed(1234567)

        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.aggregation_mode = aggregation_mode

        self.pre_lin_dict = torch.nn.ModuleDict()
        for node_type in node_types:
            self.pre_lin_dict[node_type] = Linear(-1, hidden_channels)

        self.post_lin_dict = torch.nn.ModuleDict()
        for node_type in node_types:
            self.post_lin_dict[node_type] = Linear(-1, hidden_channels)

            
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('graph_1', 'to', 'graph_1'): GCNConv(-1, hidden_channels),
                ('graph_2', 'to', 'graph_2'): GCNConv(-1, hidden_channels),
                ('graph_1', 'to', 'graph_2'): GATConv((-1, -1), hidden_channels, add_self_loops=False),
                ('graph_2', 'rev_to', 'graph_1'): GATConv((-1, -1), hidden_channels, add_self_loops=False),
            }, aggr='sum')
            self.convs.append(conv)

        # linear layers for skip connections
        self.skip_lin = torch.nn.ModuleList()
        for _ in range(num_layers):
            lin = torch.nn.ModuleDict()
            for node_type in node_types:
                lin[node_type] = Linear(-1, hidden_channels)
            self.skip_lin.append(lin)

        # createa batch norm layer for each node type
        self.batch_norm_dict_pre_conv = torch.nn.ModuleDict()
        for node_type in node_types:
            self.batch_norm_dict_pre_conv[node_type] = torch.nn.BatchNorm1d(hidden_channels)

        self.batch_norm_dict_post_conv = torch.nn.ModuleDict()
        for node_type in node_types:
            self.batch_norm_dict_post_conv[node_type] = torch.nn.BatchNorm1d(hidden_channels)


        #self.singleLin = Linear(hidden_channels*len(node_types), out_channels)

        self.lin1 = Linear(hidden_channels*len(node_types), hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict, batch_dict, training = False):


        x_dict = self.forward_core(x_dict, edge_index_dict)

        # for each node type, aggregate over all nodes of that type
        if batch_dict is not None:
            type_specific_representations = []
            for key, x in x_dict.items():
                #print(x_dict[key])
                rep = self.aggregation_mode(x_dict[key], batch_dict[key])
                type_specific_representations.append(rep)

        else:
            type_specific_representations = []
            for key, x in x_dict.items():
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                rep = self.aggregation_mode(x_dict[key], torch.zeros(x.shape[0], dtype= torch.int64).to(device))
                type_specific_representations.append(rep)

        x = F.dropout(torch.cat(type_specific_representations, dim=1), p=self.dropout, training = training)
        return self.lin2(self.lin1(x).relu_()) #self.singleLin(x) #self.lin3(self.lin2(self.lin1(x)))


    def forward_core(self, x_dict, edge_index_dict):
        # linear layer batchnorm and relu
        for node_type, x in x_dict.items():
            x_dict[node_type] = self.batch_norm_dict_pre_conv[node_type](self.pre_lin_dict[node_type](x)).relu_()

        # convolutions followed by relu
        for i, conv in enumerate(self.convs):
            # apply the conv and then add the skip connections
            x_dict = conv(x_dict, edge_index_dict) 
            # apply the skip connections
            for node_type, x in x_dict.items():
                x_dict[node_type] = x_dict[node_type] + self.skip_lin[i][node_type](x_dict[node_type])
            x_dict = {key: x.relu() for key, x in x_dict.items()}

        # linear layer batchnorm and relu
        for node_type, x in x_dict.items():
            x_dict[node_type] = self.batch_norm_dict_post_conv[node_type](self.post_lin_dict[node_type](x)).relu_()

        return x_dict

    @torch.no_grad()
    def vals_without_aggregation(self, x_dict, edge_index_dict):


        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        ######################
        # same as before
        ######################

        self.forward_core(x_dict, edge_index_dict)

        ######################
        # end of same as before
        ######################

        
        # for each node type apply the part of the linear layer that corresponds to that node type
        ct = -1
        for node_type, x in x_dict.items():
            ct +=1
            # use only part of the linear layer that corresponds to that node type
            # extract the first half of the weights of the linear layer


            Lin = Linear(self.singleLin.weight.shape[1]//2, self.singleLin.weight.shape[0]).to(device)

            # Save the state dict of the larger linear layer
            larger_state_dict = self.singleLin.state_dict()

            # Extract weights for the first 5 input features from the larger linear layer
            weights_to_transfer = larger_state_dict['weight'][:, ct*self.singleLin.weight.shape[1]//2:(ct+1)*self.singleLin.weight.shape[1]//2]

            # Modify the state dict of the smaller linear layer with the extracted weights
            smaller_state_dict = Lin.state_dict()
            smaller_state_dict['weight'] = weights_to_transfer

            # Load the modified state dict into the smaller linear layer
            Lin.load_state_dict(smaller_state_dict)

            x_dict[node_type] = Lin(x)

        return x_dict