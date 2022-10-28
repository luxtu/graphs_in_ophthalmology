import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GraphConv
from torch_geometric.nn import global_mean_pool



class GCN_GC(torch.nn.Module):
    def __init__(self, hidden_channels, in_features, classes):
        super(GCN_GC, self).__init__()
        torch.manual_seed(1234567)
        self.conv1 = GCNConv(in_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, classes)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
        return x



class GCN_NC(torch.nn.Module):
    def __init__(self, hidden_channels,in_features, classes):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GCNConv(in_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x



class SAGE_NC(torch.nn.Module):
    def __init__(self, hidden_channels,in_features, classes):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = SAGEConv(in_features, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv3(x, edge_index)
        return x


class WC_NC(torch.nn.Module):
    def __init__(self, hidden_channels,in_features, classes):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GraphConv(in_features, hidden_channels, aggr = "add")
        self.conv2 = GraphConv(hidden_channels, hidden_channels, aggr = "mean")
        self.conv3 = GraphConv(hidden_channels, classes, aggr = "max")

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        #x = x.relu()
        x = self.conv2(x, edge_index)
        #x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv3(x, edge_index)
        return x