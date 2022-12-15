import torch
#from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, ClusterGCNConv, BatchNorm, Linear, GCN, GAT
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
from torch_geometric.nn import global_mean_pool
import wandb
import pandas as pd
from tqdm import tqdm
#from torch.nn import Linear




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
        out = self.model(nxG.x[:,self.features].float(), nxG.edge_index, training = True)  # Perform a single forward pass.
        loss = self.lossFunc(out[train_mask].float(), nxG.y[train_mask])  # Compute the loss solely based on the training nodes.
        loss.backward()  # Derive gradients.
        self.optimizer.step()  # Update parameters based on gradients.
        return loss


    @torch.no_grad()
    def test(self, nxG, test_mask):
        self.model.eval()
        out = self.model(nxG.x[:,self.features].float(), nxG.edge_index, training = False)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        test_correct = pred[test_mask] == nxG.y[test_mask]  # Check against ground-truth labels.
        test_acc = int(test_correct.sum()) / len(test_mask)  # Derive ratio of correct predictions.
        return test_acc


    @torch.no_grad()
    def predictions(self, nxG, mask= None, max_prob = True):
        self.model.eval()

        out = self.model(nxG.x[:,self.features].float(), nxG.edge_index, training = False)

        if max_prob:
            pred = out.argmax(dim=1)  # Use the class with highest probability.

        else:
            pred = out
        return pred 





class SAGE_VS(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, aggr = "mean", MLP = False, skip = False, norm = False):
        super().__init__()
        torch.manual_seed(1234567)
        self.aggr = aggr
        self.MLP = MLP
        self.skip = skip
        self.norm = norm
        if MLP:
            self.numNormalConvLayers = num_layers
        else:
            self.numNormalConvLayers = num_layers -2

        self.linPre = torch.nn.ModuleList()
        self.linPost = torch.nn.ModuleList()
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()


        if not self.MLP:
            self.convs.append(SAGEConv(in_channels, hidden_channels, aggr = self.aggr))
        else:
            self.linPre.append(Linear(in_channels, hidden_channels, bias = True, weight_initializer = "kaiming_uniform"))
            self.linPre.append(Linear(hidden_channels, hidden_channels, bias = True, weight_initializer = "kaiming_uniform"))


        for _ in range(self.numNormalConvLayers):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr = self.aggr))
            if self.norm:
                self.norms.append(BatchNorm(hidden_channels))

        if not self.MLP:
            self.convs.append(SAGEConv(hidden_channels, out_channels, aggr = self.aggr))
        else:
            self.linPost.append(Linear(hidden_channels, hidden_channels, bias = True, weight_initializer = "kaiming_uniform"))
            self.linPost.append(Linear(hidden_channels, out_channels, bias = True, weight_initializer = "kaiming_uniform"))

        self.dropout = dropout


    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index, training):
        if self.MLP:
            for lin in self.linPre:
                x = lin(x)
                x = F.relu(x)

        for i, conv in enumerate(self.convs[:-1]):
            if self.skip:
                identity = x
            x = conv(x, edge_index)
            if self.norm:
                x = self.norms[i](x)
            if self.skip:
                x += identity
            x = F.relu(x)

        x = F.dropout(x, p=self.dropout, training = training)
        x = self.convs[-1](x, edge_index) # if no mlp after then downsamples to 3 

        if self.MLP:
            for lin in self.linPost:
                x = lin(x)
        #x = F.relu(x)
        

        return x #remove softmax with cross entropy loss - cross entropy applies the softmax on its own


class GCN_VS(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, MLP = False, skip = False):
        super().__init__()
        #torch.manual_seed(1234567)
        self.dropout = dropout
        self.MLP = MLP
        self.skip = skip

        if MLP:
            self.numNormalConvLayers = num_layers
        else:
            self.numNormalConvLayers = num_layers -2
        
        self.linPre = torch.nn.ModuleList()
        self.linPost = torch.nn.ModuleList()
        self.convs = torch.nn.ModuleList()


        if not self.MLP:
            self.convs.append(GCNConv(in_channels, hidden_channels))
        else:
            self.linPre.append(Linear(in_channels, hidden_channels, bias = True, weight_initializer = "kaiming_uniform"))
            self.linPre.append(Linear(hidden_channels, hidden_channels, bias = True, weight_initializer = "kaiming_uniform"))


        for _ in range(self.numNormalConvLayers):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            #if self.norm:
            #    self.norms.append(BatchNorm(hidden_channels))

        if not self.MLP:
            self.convs.append(GCNConv(hidden_channels, out_channels))
        else:
            self.linPost.append(Linear(hidden_channels, hidden_channels, bias = True, weight_initializer = "kaiming_uniform"))
            self.linPost.append(Linear(hidden_channels, out_channels, bias = True, weight_initializer = "kaiming_uniform"))




    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()


    def forward(self, x, edge_index, training):
        if self.MLP:
            for lin in self.linPre:
                x = lin(x)
                x = F.relu(x)

        for i, conv in enumerate(self.convs[:-1]):
            if self.skip:
                identity = x
            x = F.dropout(x, p=self.dropout, training = training)
            x = conv(x, edge_index)
            #if self.norm:
            #    x = self.norms[i](x)
            if self.skip:
                x += identity
            x = F.relu(x)

        x = F.dropout(x, p=self.dropout, training = training)
        x = self.convs[-1](x, edge_index) # if no mlp after then downsamples to 3 

        if self.MLP:
            for lin in self.linPost:
                x = lin(x)
        return x # torch.log_softmax(x, dim=-1) # remove softmax with binary cross entropy


class GAT_VS(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super().__init__()
        torch.manual_seed(1234567)

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

    def forward(self, x, edge_index, training):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=training) # maybe remove the dropout from the convolutional layers, or reduce drastically
        x = self.convs[-1](x, edge_index)
        return x # torch.log_softmax(x, dim=-1) # remove softmax with binary cross entropy



class CLUST_GCN_VS(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, diag_lambda = 0):
        super().__init__()
        torch.manual_seed(1234567)

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

    def forward(self, x, edge_index, training):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=training) # maybe remove the dropout from the convolutional layers, or reduce drastically
        x = self.convs[-1](x, edge_index)
        return x # torch.log_softmax(x, dim=-1) # remove softmax with binary cross entropy






class nodeClassifierSweep():
    
    def __init__(self, features, classes, optimizer, lossFunc, graph, train_mask, val_mask, epochs = 100, test_mask = None):
        self.features = features
        self.classes = classes 
        self.optimizer = optimizer
        self.lossFunc = lossFunc
        self.graph = graph
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.epochs = epochs
        self.test_mask = test_mask


    def handle_model(self, model_str):
        if model_str == "SAGE":
            modelS = SAGE_VS(in_channels = len(self.features), hidden_channels= wandb.config.hidden_channels, 
                out_channels = self.classes, num_layers = wandb.config.num_layers, dropout  = wandb.config.dropout, aggr  = wandb.config.aggr, MLP = wandb.config.MLP, skip = wandb.config.skip, norm = wandb.config.norm)

        elif model_str == "GCN":
           modelS = GCN_VS(in_channels = len(self.features), hidden_channels= wandb.config.hidden_channels, 
                out_channels = self.classes, num_layers = wandb.config.num_layers, dropout  = wandb.config.dropout, MLP = wandb.config.MLP, skip = wandb.config.skip)

        elif model_str == "CLUST":
            modelS = CLUST_GCN_VS(in_channels = len(self.features), hidden_channels= wandb.config.hidden_channels, 
                out_channels = self.classes, num_layers = wandb.config.num_layers, dropout  = wandb.config.dropout) 

        elif model_str == "GAT":
            modelS = GAT_VS(in_channels = len(self.features), hidden_channels= wandb.config.hidden_channels, 
                out_channels = self.classes, num_layers = wandb.config.num_layers, dropout  = wandb.config.dropout) 

        elif model_str == "GCN_PAP":
            modelS = GCN(in_channels = len(self.features), hidden_channels= wandb.config.hidden_channels, 
                num_layers = wandb.config.num_layers, out_channels = self.classes)

        elif model_str == "GAT_PAP":
            modelS = GAT(in_channels = len(self.features), hidden_channels= wandb.config.hidden_channels, 
                num_layers = wandb.config.num_layers, out_channels = self.classes)
        return modelS



    def agent_variable_size_model(self):
        wandb.init()

        modelS = self.handle_model(wandb.config.models)

        wandb.watch(modelS)

        with torch.no_grad():
            out = modelS(self.graph.x.float(), self.graph.edge_index, training = False)
            embedding_to_wandb(out, color= self.graph.y, key = "gcn/embedding/init")  
        optimizerS = self.optimizer(modelS.parameters(), lr = wandb.config.lr, weight_decay= wandb.config.weight_decay) #wandb.config.lr # wandb.config.weight_decay
        criterionS = self.lossFunc()


        def train():
            modelS.train()
            optimizerS.zero_grad()  # Clear gradients. #
            out = modelS(self.graph.x.float(), self.graph.edge_index, training = True)  # Perform a single forward pass.
            loss = criterionS(out[self.train_mask].float(), self.graph.y[self.train_mask])  # Compute the loss solely based on the training nodes.
            loss.backward()  # Derive gradients.
            optimizerS.step()  # Update parameters based on gradients.
 
            return loss

        @torch.no_grad()
        def test(recall = False):
            modelS.eval()
            out = modelS(self.graph.x.float(), self.graph.edge_index, training = False)
            pred = out.argmax(dim=1)  # Use the class with highest probability.
            test_acc = accuracy_score(self.graph.y[self.val_mask], pred[self.val_mask])
            #test_correct = pred[self.val_mask] == self.graph.y[self.val_mask]  # Check against ground-truth labels.
            #test_acc = int(test_correct.sum()) / len(self.val_mask)  # Derive ratio of correct predictions.
            if recall:
                recalls = recall_score(self.graph.y[self.val_mask], pred[self.val_mask],  average=None)
                return test_acc, recalls
            else:
                return test_acc

        @torch.no_grad()
        def geom_test():
            modelS.eval()
            out = modelS(self.graph.x.float(), self.graph.edge_index, training = False)
            pred = out.argmax(dim=1)  # Use the class with highest probability.
            recalls = recall_score(self.graph.y[self.test_mask], pred[self.test_mask],  average=None)
            test_acc = accuracy_score(self.graph.y[self.test_mask], pred[self.test_mask])
            #test_correct = pred == self.test_graph.y  # Check against ground-truth labels.
            #test_acc = int(test_correct.sum()) / self.test_graph.y.shape[0]  # Derive ratio of correct predictions.
            return test_acc, recalls

        opt_valacc = 0
        for self.epoch in tqdm(range(1, self.epochs+1)):
            loss = train()
            acc,recalls = test(recall = True)
            #testacc, recalls = geom_test()

            if acc > opt_valacc:
                opt_valacc = acc

            #wandb.log({"gcn/loss": loss, "gcn/valacc": acc}) # , "gcn/testacc": testacc
            wandb.log({"gcn/loss": loss, "gcn/valacc": acc, "gcn/recall_0": recalls[0], "gcn/recall_1": recalls[1]}) # , "gcn/testacc": testacc

        test_acc = test()
        wandb.summary["gcn/accuracy"] = test_acc

        if self.test_mask is not None:
            test_graph_acc, recalls = geom_test()
            wandb.log({"gcn/max_accuracy": opt_valacc, "gcn/test_graph_accuracy": test_graph_acc, "gcn/recall_0": recalls[0], "gcn/recall_1": recalls[1]}) # , "gcn/recall_2": recalls[2]

        else:
            wandb.log({"gcn/max_accuracy": opt_valacc})

        
        embedding_to_wandb(out, color=self.graph.y, key="gcn/embedding/trained")
        wandb.finish()    