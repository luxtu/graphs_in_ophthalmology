import torch
#from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, GraphConv, Linear, GATConv, EdgeConv, DenseSAGEConv



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
                self.convLayers.append(DenseSAGEConv(self.in_channels, self.hidden_channels))
            else:
                self.convLayers.append(DenseSAGEConv(self.hidden_channels, self.hidden_channels))

        #set up final layer
        self.lin = Linear(self.hidden_channels, self.out_channels)


    def forward(self, x, edge_index, batch, training = False, agg = True):

        for conv in self.convLayers:
            x = F.dropout(x, p=self.dropout, training = training)
            x = conv(x, edge_index)
            x = F.relu(x)

        if agg:
            
            if batch is None:
                x = x.mean(dim = 1)
            else:
                x = self.aggregation_mode(x, batch)
            x = self.lin(F.dropout(x, p=self.dropout, training = training))
            return x

        else:
            return x
        
