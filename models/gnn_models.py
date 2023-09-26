import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, GraphConv



class GCN_GC(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, aggregation_mode):
        super(GCN_GC, self).__init__()
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
    def __init__(self, hidden_channels, out_channels, num_layers, dropout, aggregation_mode):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.aggregation_mode = aggregation_mode


        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('void', 'to', 'void'): SAGEConv(-1, hidden_channels),
                ('vessel', 'to', 'vessel'): SAGEConv(-1, hidden_channels),
            }, aggr='sum')
            self.convs.append(conv)

        self.lin = Linear(hidden_channels*2, out_channels)

    def forward(self, x_dict, edge_index_dict, batch_dict, training = False):
        for conv in self.convs:
            # conv + relu
            x_dict = conv(x_dict, edge_index_dict) 
            x_dict = {key: x.relu() for key, x in x_dict.items()}


        # for each node type, aggregate over all nodes of that type


        if batch_dict is not None:
            type_specific_representations = []
            for key, x in x_dict.items():

                rep = self.aggregation_mode(x_dict[key], batch_dict[key])
                type_specific_representations.append(rep)

        else:
            type_specific_representations = []
            for key, x in x_dict.items():
                x_dict[key] = self.aggregation_mode(x_dict[key], torch.zeros(x.shape[0], dtype= torch.int64).to(self.device))
                type_specific_representations.append(rep)

        x = F.dropout(torch.cat(type_specific_representations, dim=1), p=self.dropout, training = training)
        return self.lin(x)
