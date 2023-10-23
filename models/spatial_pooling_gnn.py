import torch
#from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, GraphConv, Linear, GATConv, EdgeConv, dense_diff_pool
from math import ceil
from models import global_node_gnn, simple_gnn, hetero_gnn
# import global mean pool
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.utils import to_dense_adj

class GeomPool_GNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers, dropout, aggregation_mode, node_types):
        super().__init__()
        torch.manual_seed(1234567)

        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.aggregation_mode = aggregation_mode
        self.node_types = node_types
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


        # will be used only before pooling
        self.gnn1_embed = global_node_gnn.GNN_global_node(hidden_channels= self.hidden_channels,
                                                         out_channels= self.hidden_channels, # irrelevant
                                                         num_layers= self.num_layers, 
                                                         dropout = self.dropout, 
                                                         aggregation_mode= None, 
                                                         node_types = node_types)
        
        self.lin1 = Linear(-1, hidden_channels*2)
        self.lin2 = Linear(hidden_channels*2, hidden_channels)
        self.lin3 = Linear(hidden_channels, out_channels)
        
        


    def forward(self, x_dict, edge_dict, batch_dict, slice_dict, training = False, grads = False, pos_dict = None):

        x_dict = self.gnn1_embed.forward_core(x_dict, edge_dict, training=training,  grads=grads)

        # pool

        representation_concat = []
        for key in self.node_types:
            # assign the nodes based on position to the superior, inferior, temoral or nasal region

            # get the node positions
            node_positions = pos_dict[key]
            # adjust position to -600 to 600 for both x and y axis
            node_positions -= 600

            signed_distances_diagonal = node_positions[:, 1] - node_positions[:, 0]
            signed_distances_normal_diagonal = node_positions[:, 1] + node_positions[:, 0]

            labels = torch.zeros(node_positions.shape[0], dtype=torch.long, device=self.device)
            labels[signed_distances_diagonal > 0] += 1
            labels[signed_distances_normal_diagonal < 0] += 2
            # make the radius learnable
            # maybe change to 120
            #circ_rad = 100
            #labels[torch.linalg.norm(node_positions, axis=1) < circ_rad] = 4

            # assign to class 0 or 1 by slicing a diagonal through the image
            # diagonal is from top left to bottom right

            for i in range(4):
                rep = self.aggregation_mode(x_dict[key][labels==i], batch_dict[key][labels ==i])
                representation_concat.append(rep)
             
        representation_concat.append(self.aggregation_mode(x_dict["global"], batch_dict["global"]))
        x = torch.cat(representation_concat, dim=1)  
        x = F.dropout(x, p=self.dropout, training = training)

        return self.lin3(self.lin2(self.lin1(x).relu()).relu())


        
