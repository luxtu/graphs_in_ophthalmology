import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GraphConv





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

        # create the conv layers
        self.convLayers = torch.nn.ModuleList()
        for i in range(self.num_layers):
            if i == 0:
                self.convLayers.append(GCNConv(self.in_channels, self.hidden_channels))
            else:
                self.convLayers.append(GCNConv(self.hidden_channels, self.hidden_channels))

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
            x = self.aggregation_mode(x, torch.zeros(x.shape[0], dtype= torch.int64))

        x = F.dropout(x, p=self.dropout, training = training) # dropout is here only applied as the last layer
        x = self.lin(x) # final linear layer/classifier
        
        return x