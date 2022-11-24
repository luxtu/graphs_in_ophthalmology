import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GraphConv, GATConv, ClusterGCNConv
from torch_geometric.nn import global_mean_pool
import wandb
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import confusion_matrix




def embedding_to_wandb(h, color, key="embedding"):
    num_components = h.shape[-1]
    df = pd.DataFrame(data=h.detach().cpu().numpy(),
                        columns=[f"c_{i}" for i in range(num_components)])
    df["target"] = color.detach().cpu().numpy().astype("str")
    cols = df.columns.tolist()
    df = df[cols[-1:] + cols[:-1]]
    wandb.log({key: df})



class nodeClassifier():
    def __init__(self, model, hidden_channels, features, classes, num_layers = 4,  lr = 0.005, weight_decay = 5e-5):

        self.model = model(in_channels = len(features), hidden_channels = hidden_channels, out_channels = classes, num_layers = num_layers, dropout = 0.5)
        self.optimizer= torch.optim.Adam(self.model.parameters(), lr= lr, weight_decay= weight_decay)
        self.lossFunc = torch.nn.CrossEntropyLoss()
        self.features = features

    def train(self, nxG, train_mask):
        self.model.train()
        self.optimizer.zero_grad()  # Clear gradients.
        #if isinstance(self.model, GATConv):
        out = self.model(nxG.x[:,self.features].float(), nxG.edge_index)  # Perform a single forward pass.
        loss = self.lossFunc(out[train_mask].float(), nxG.y[train_mask])  # Compute the loss solely based on the training nodes.
        loss.backward()  # Derive gradients.
        self.optimizer.step()  # Update parameters based on gradients.
        return loss


    @torch.no_grad()
    def test(self, nxG, test_mask):
        self.model.eval()
        out = self.model(nxG.x[:,self.features].float(), nxG.edge_index)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        test_correct = pred[test_mask] == nxG.y[test_mask]  # Check against ground-truth labels.
        test_acc = int(test_correct.sum()) / len(test_mask)  # Derive ratio of correct predictions.
        return test_acc


    @torch.no_grad()
    def predictions(self, nxG, mask= None, max_prob = True):
        self.model.eval()

        out = self.model(nxG.x[:,self.features].float(), nxG.edge_index)

        if max_prob:
            pred = out.argmax(dim=1)  # Use the class with highest probability.

        else:
            pred = out
        return pred 







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

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
        
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
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

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training) # maybe remove the dropout from the convolutional layers, or reduce drastically
        x = self.convs[-1](x, edge_index)
        return x # torch.log_softmax(x, dim=-1) # remove softmax with binary cross entropy


class GAT_VS(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(
            GATConv(in_channels, hidden_channels, normalize=False))
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_channels, hidden_channels, normalize=False))
        self.convs.append(
            GATConv(hidden_channels, out_channels, normalize=False))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training) # maybe remove the dropout from the convolutional layers, or reduce drastically
        x = self.convs[-1](x, edge_index)
        return x # torch.log_softmax(x, dim=-1) # remove softmax with binary cross entropy



class CLUST_GCN_VS(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, diag_lambda = 0):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(
            ClusterGCNConv(in_channels, hidden_channels, diag_lambda= diag_lambda))
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_channels, hidden_channels, diag_lambda= diag_lambda))
        self.convs.append(
            GATConv(hidden_channels, out_channels, diag_lambda= diag_lambda))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training) # maybe remove the dropout from the convolutional layers, or reduce drastically
        x = self.convs[-1](x, edge_index)
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


    def handle_model(self, model_str):
        if model_str == "SAGE":
            modelS = SAGE_VS(in_channels = len(self.features), hidden_channels= wandb.config.hidden_channels, 
                out_channels = self.classes, num_layers = wandb.config.num_layers, dropout  = wandb.config.dropout)

        elif model_str == "GCN":
           modelS = GCN_VS(in_channels = len(self.features), hidden_channels= wandb.config.hidden_channels, 
                out_channels = self.classes, num_layers = wandb.config.num_layers, dropout  = wandb.config.dropout)

        elif model_str == "CLUST":
            modelS = GCN_VS(in_channels = len(self.features), hidden_channels= wandb.config.hidden_channels, 
                out_channels = self.classes, num_layers = wandb.config.num_layers, dropout  = wandb.config.dropout) 

        return modelS


    def agent_variable_size_model(self):
        wandb.init()

        modelS = self.handle_model(wandb.config.models)

        wandb.watch(modelS)

        with torch.no_grad():
            out = modelS(self.graph.x.float(), self.graph.edge_index)
            embedding_to_wandb(out, color= self.graph.y, key = "gcn/embedding/init")  
        optimizerS = self.optimizer(modelS.parameters(), lr = wandb.config.lr, weight_decay= wandb.config.weight_decay) #wandb.config.lr # wandb.config.weight_decay
        criterionS = self.lossFunc()


        def train():
            modelS.train()
            optimizerS.zero_grad()  # Clear gradients. #
            out = modelS(self.graph.x.float(), self.graph.edge_index)  # Perform a single forward pass.
            loss = criterionS(out[self.train_mask].float(), self.graph.y[self.train_mask])  # Compute the loss solely based on the training nodes.
            loss.backward()  # Derive gradients.
            optimizerS.step()  # Update parameters based on gradients.
 
            return loss

        @torch.no_grad()
        def test():
            modelS.eval()
            out = modelS(self.graph.x.float(), self.graph.edge_index)
            pred = out.argmax(dim=1)  # Use the class with highest probability.
            test_correct = pred[self.test_mask] == self.graph.y[self.test_mask]  # Check against ground-truth labels.
            test_acc = int(test_correct.sum()) / len(self.test_mask)  # Derive ratio of correct predictions.
            return test_acc


        opt_valacc = 0
        for self.epoch in tqdm(range(1, self.epochs+1)):
            loss = train()
            acc = test()

            if acc > opt_valacc:
                opt_valacc = acc

            wandb.log({"gcn/loss": loss, "gcn/valacc": acc})

        modelS.eval()
        out = modelS(self.graph.x.float(), self.graph.edge_index)
        
        test_acc = test()

        wandb.summary["gcn/accuracy"] = test_acc
        wandb.log({"gcn/max_accuracy": opt_valacc, "gcn/accuracy": test_acc})
        embedding_to_wandb(out, color=self.graph.y, key="gcn/embedding/trained")
        wandb.finish()    