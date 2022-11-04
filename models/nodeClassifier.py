import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GraphConv
from torch_geometric.nn import global_mean_pool
import wandb
import pandas as pd
from tqdm import tqdm




def embedding_to_wandb(h, color, key="embedding"):
    num_components = h.shape[-1]
    df = pd.DataFrame(data=h.detach().cpu().numpy(),
                        columns=[f"c_{i}" for i in range(num_components)])
    df["target"] = color.detach().cpu().numpy().astype("str")
    cols = df.columns.tolist()
    df = df[cols[-1:] + cols[:-1]]
    wandb.log({key: df})




class GCN_NC(torch.nn.Module):
    def __init__(self, hidden_channels, in_features, classes):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GCNConv(in_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, classes)


    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv4(x, edge_index)
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


class nodeClassifier():
    def __init__(self, model, hidden_channels,features, classes, optimizer, lossFunc):
        self.model = model(hidden_channels, len(features), classes)
        self.optimizer= optimizer(self.model.parameters(), lr=0.005, weight_decay=5e-5)
        self.lossFunc = lossFunc()
        self.features = features

    def train(self, nxG, train_mask):
        self.model.train()
        self.optimizer.zero_grad()  # Clear gradients.
        out = self.model(nxG.x[:,self.features].float(), nxG.edge_index)  # Perform a single forward pass.
        loss = self.lossFunc(out[train_mask].float(), nxG.y[train_mask])  # Compute the loss solely based on the training nodes.
        loss.backward()  # Derive gradients.
        self.optimizer.step()  # Update parameters based on gradients.
        return loss

    def test(self, nxG, test_mask):
        self.model.eval()
        out = self.model(nxG.x[:,self.features].float(), nxG.edge_index)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        test_correct = pred[test_mask] == nxG.y[test_mask]  # Check against ground-truth labels.
        test_acc = int(test_correct.sum()) / len(test_mask)  # Derive ratio of correct predictions.
        return test_acc

    def predictions(self, nxG, test_mask):
        self.model.eval()
        out = self.model(nxG.x[:,self.features].float(), nxG.edge_index)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        test_correct = pred[test_mask] == nxG.y[test_mask]  # Check against ground-truth labels.
        return test_correct    







class SAGE_VS(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super().__init__()
        torch.manual_seed(1234567)
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
        
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x # torch.log_softmax(x, dim=-1) # remove softmax with binary cross entropy


class GCN_VS(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(
            GCNConv(in_channels, hidden_channels, normalize=False))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, normalize=False))
        self.convs.append(
            GCNConv(hidden_channels, out_channels, normalize=False))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training) # maybe remove the dropout from the convolutional layers, or reduce drastically
        x = self.convs[-1](x, adj_t)
        return x # torch.log_softmax(x, dim=-1) # remove softmax with binary cross entropy




class nodeClassifierSweep():
    def __init__(self, features, classes, optimizer, lossFunc, graph, train_mask, test_mask, epochs = 100):
        self.features = features
        self.classes = classes 
        self.optimizer = optimizer
        self.lossFunc = lossFunc
        self.graph = graph
        self.train_mask = train_mask
        self.test_mask = test_mask
        self.epochs = epochs


    def agent_fixed_size_model(self):
        wandb.init()
        if wandb.config.models == "SAGE":
            modelS = SAGE_NC(hidden_channels= wandb.config.hidden_channels, in_features = len(self.features), classes = self.classes)
        elif wandb.config.models == "GCN":
           modelS = GCN_NC(hidden_channels= wandb.config.hidden_channels, in_features = len(self.features), classes = self.classes)
        elif wandb.config.models == "WC":
           modelS = WC_NC(hidden_channels= wandb.config.hidden_channels, in_features = len(self.features), classes = self.classes)


        #modelS = self.model(hidden_channels= wandb.config.hidden_channels, in_features = len(self.features), classes = self.classes) # wandb.config.hidden_channels
        wandb.watch(modelS)
    
        with torch.no_grad():
            out = modelS(self.graph.x.float(), self.graph.edge_index)
            embedding_to_wandb(out, color= self.graph.y, key="gcn/embedding/init")  

        optimizerS = self.optimizer(modelS.parameters(), lr= wandb.config.lr, weight_decay= wandb.config.weight_decay) #wandb.config.lr # wandb.config.weight_decay
        criterionS = self.lossFunc()

        def train():

            modelS.train()
            optimizerS.zero_grad()  # Clear gradients. #
            out = modelS(self.graph.x.float(), self.graph.edge_index)  # Perform a single forward pass.
            loss = criterionS(out[self.train_mask].float(), self.graph.y[self.train_mask])  # Compute the loss solely based on the training nodes.
            loss.backward()  # Derive gradients.
            optimizerS.step()  # Update parameters based on gradients.
 
            return loss

        def test():
            modelS.eval()
            out = modelS(self.graph.x.float(), self.graph.edge_index)
            pred = out.argmax(dim=1)  # Use the class with highest probability.
            test_correct = pred[self.test_mask] == self.graph.y[self.test_mask]  # Check against ground-truth labels.
            test_acc = int(test_correct.sum()) / len(self.test_mask)  # Derive ratio of correct predictions.
            return test_acc


        for self.epoch in tqdm(range(1, self.epochs+1)):
            loss = train()
            wandb.log({"gcn/loss": loss})


        modelS.eval()
        out = modelS(self.graph.x.float(), self.graph.edge_index)
        
        test_acc = test()
        wandb.summary["gcn/accuracy"] = test_acc
        wandb.log({"gcn/accuracy": test_acc})
        embedding_to_wandb(out, color=self.graph.y, key="gcn/embedding/trained")
        wandb.finish()        


    def agent_variable_size_model(self):
        wandb.init()
        if wandb.config.models == "SAGE":
            modelS = SAGE_VS(in_channels = len(self.features), hidden_channels= wandb.config.hidden_channels, 
                out_channels = self.classes, num_layers = wandb.config.num_layers, dropout  = wandb.config.dropout)

        elif wandb.config.models == "GCN":
           modelS = GCN_VS(in_channels = len(self.features), hidden_channels= wandb.config.hidden_channels, 
                out_channels = self.classes, num_layers = wandb.config.num_layers, dropout  = wandb.config.dropout)

        wandb.watch(modelS)
        with torch.no_grad():
            out = modelS(self.graph.x.float(), self.graph.edge_index)
            embedding_to_wandb(out, color= self.graph.y, key="gcn/embedding/init")  
        optimizerS = self.optimizer(modelS.parameters(), lr= wandb.config.lr, weight_decay= wandb.config.weight_decay) #wandb.config.lr # wandb.config.weight_decay
        criterionS = self.lossFunc()


        def train():

            modelS.train()
            optimizerS.zero_grad()  # Clear gradients. #
            out = modelS(self.graph.x.float(), self.graph.edge_index)  # Perform a single forward pass.
            loss = criterionS(out[self.train_mask].float(), self.graph.y[self.train_mask])  # Compute the loss solely based on the training nodes.
            loss.backward()  # Derive gradients.
            optimizerS.step()  # Update parameters based on gradients.
 
            return loss

        def test():
            modelS.eval()
            out = modelS(self.graph.x.float(), self.graph.edge_index)
            pred = out.argmax(dim=1)  # Use the class with highest probability.
            test_correct = pred[self.test_mask] == self.graph.y[self.test_mask]  # Check against ground-truth labels.
            test_acc = int(test_correct.sum()) / len(self.test_mask)  # Derive ratio of correct predictions.
            return test_acc


        for self.epoch in tqdm(range(1, self.epochs+1)):
            loss = train()
            wandb.log({"gcn/loss": loss})


        modelS.eval()
        out = modelS(self.graph.x.float(), self.graph.edge_index)
        
        test_acc = test()
        wandb.summary["gcn/accuracy"] = test_acc
        wandb.log({"gcn/accuracy": test_acc})
        embedding_to_wandb(out, color=self.graph.y, key="gcn/embedding/trained")
        wandb.finish()    