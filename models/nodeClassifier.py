import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GraphConv
from torch_geometric.nn import global_mean_pool


class nodeClassifier():
    def __init__(self, model, hidden_channels,features, classes, optimizer, lossFunc):
        self.model = model(hidden_channels, len(features), classes)
        self.optimizer= optimizer(self.model.parameters(), lr=0.005, weight_decay=5e-5)
        self.lossFunc = lossFunc()
        self.features = features

    def train(self, nxG, train_mask):
        self.model.train()
        self.optimizer.zero_grad()  # Clear gradients.
        out = self.model(nxG.x[:,self.features], nxG.edge_index)  # Perform a single forward pass.
        loss = self.lossFunc(out[train_mask], nxG.y[train_mask])  # Compute the loss solely based on the training nodes.
        loss.backward()  # Derive gradients.
        self.optimizer.step()  # Update parameters based on gradients.
        return loss

    def test(self, nxG, test_mask):
        self.model.eval()
        out = self.model(nxG.x[:,self.features], nxG.edge_index)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        test_correct = pred[test_mask] == nxG.y[test_mask]  # Check against ground-truth labels.
        test_acc = int(test_correct.sum()) / len(test_mask)  # Derive ratio of correct predictions.
        return test_acc

    def predictions(self, nxG, test_mask):
        self.model.eval()
        out = self.model(nxG.x[:,self.features], nxG.edge_index)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        test_correct = pred[test_mask] == nxG.y[test_mask]  # Check against ground-truth labels.
        return test_correct    






class GCN_NC(torch.nn.Module):
    def __init__(self, hidden_channels, in_features, classes):
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