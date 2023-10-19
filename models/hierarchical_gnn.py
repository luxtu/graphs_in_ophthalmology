import torch
#from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, GraphConv, Linear, GATConv, EdgeConv, dense_diff_pool
from math import ceil
from models import global_node_gnn, simple_gnn, hetero_gnn
# import global mean pool
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.utils import to_dense_adj

class DiffPool_GNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers, dropout, node_types, max_nodes):
        super().__init__()
        torch.manual_seed(1234567)

        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.max_nodes = max_nodes
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


        self.num_nodes = ceil(max_nodes *0.1)
        self.gnn1_pool = global_node_gnn.GNN_global_node(hidden_channels= self.num_nodes,
                                                         out_channels= self.num_nodes, # irrelevant
                                                         num_layers= self.num_layers, 
                                                         dropout = self.dropout, 
                                                         aggregation_mode= None, 
                                                         node_types = node_types)
        
        self.gnn1_embed = global_node_gnn.GNN_global_node(hidden_channels= self.hidden_channels,
                                                         out_channels= self.hidden_channels, # irrelevant
                                                         num_layers= self.num_layers, 
                                                         dropout = self.dropout, 
                                                         aggregation_mode= None, 
                                                         node_types = node_types)
        
        self.num_nodes = ceil(self.num_nodes *0.1)
        self.gnn_2_pool = simple_gnn.GNN(in_channels= self.hidden_channels,
                                            hidden_channels= self.num_nodes,
                                            out_channels= self.num_nodes, # irrelevant
                                            num_layers= self.num_layers,
                                            dropout = self.dropout,
                                            aggregation_mode= None)

        self.gnn_2_embed = simple_gnn.GNN(in_channels= self.hidden_channels,
                                            hidden_channels= self.hidden_channels,
                                            out_channels= self.hidden_channels, # irrelevant
                                            num_layers= self.num_layers,
                                            dropout = self.dropout,
                                            aggregation_mode= None)


        self.gnn_3_embed_cls = simple_gnn.GNN(in_channels= self.hidden_channels,
                                    hidden_channels= self.hidden_channels,
                                    out_channels= self.out_channels,
                                    num_layers= self.num_layers,
                                    dropout = self.dropout,
                                    aggregation_mode= None)

        self.gnn1_embed.to(self.device)
        self.gnn1_pool.to(self.device)
        self.gnn_2_pool.to(self.device)
        self.gnn_2_embed.to(self.device)
        self.gnn_3_embed_cls.to(self.device)

    def forward(self, x_dict, edge_dict, batch_dict, slice_dict, training = False, grads = False):
        
        s_dict = self.gnn1_pool.forward_core(x_dict, edge_dict, training=training,  grads=grads)
        x_dict = self.gnn1_embed.forward_core(x_dict, edge_dict, training= training, grads=grads)

        #print(torch.cuda.memory_allocated()/1024/1024/1024)

        s_global = self._merge_graph_nodes(s_dict)
        x_global = self._merge_graph_nodes(x_dict)

        #print(torch.cuda.memory_allocated()/1024/1024/1024)

        ## create the batch vector
        s = self._feature_batch_vector(s_global, slice_dict)
        x = self._feature_batch_vector(x_global, slice_dict)

        #print(torch.cuda.memory_allocated()/1024/1024/1024)

        # create the batch vector for the edge_index_dict
        adj = self._edge_index_batch_vector(batch_dict, edge_dict, slice_dict)

        #print(adj.shape)
        #print(x.shape)
        #print(torch.cuda.memory_allocated()/1024/1024/1024)
        x, adj, _, _ = dense_diff_pool(x, adj, s)

        #print(torch.cuda.memory_allocated()/1024/1024/1024)
        s = self.gnn_2_pool(x, adj, batch = None, agg = False)
        x = self.gnn_2_embed(x, adj, batch = None, agg = False)

        #print(torch.cuda.memory_allocated()/1024/1024/1024)
        x, adj, _, _ = dense_diff_pool(x, adj, s)

        #print(torch.cuda.memory_allocated()/1024/1024/1024)
        x = self.gnn_3_embed_cls(x, adj, batch = None, agg = True)

        return x



    def _edge_index_batch_vector(self, batch_dict, edge_dict, slice_dict):

        batch_size = len(slice_dict["y"]) -1

        adj_graph1 = to_dense_adj(edge_dict[('graph_1', 'to', 'graph_1')], batch_dict["graph_1"])
        adj_graph2 = to_dense_adj(edge_dict[('graph_2', 'to', 'graph_2')], batch_dict["graph_2"])

        adj = torch.zeros((batch_size, self.max_nodes, self.max_nodes), device= self.device)

        # sizes are always differences between the indices
        g1_sizes = torch.diff(slice_dict["graph_1"]["x"], dim=0)
        g2_sizes = torch.diff(slice_dict["graph_2"]["x"], dim=0)

        #print(g1_sizes)
        #print(g2_sizes)

        for batch_idx in range(batch_size):

            adjs_part = torch.block_diag(adj_graph1[batch_idx, :g1_sizes[batch_idx],:g1_sizes[batch_idx]], adj_graph2[batch_idx, :g2_sizes[batch_idx], : g2_sizes[batch_idx]])
            # 
            adj[batch_idx, :g1_sizes[batch_idx] + g2_sizes[batch_idx], :g1_sizes[batch_idx] + g2_sizes[batch_idx]] = adjs_part.to(self.device)

            # add the edges between the graphs
            # the graph_2 indices need to be shifted by the number of nodes in graph_1

            start_hetero_edges = slice_dict[('graph_1', 'to', 'graph_2')]["edge_index"][batch_idx]
            end_hetero_edges = slice_dict[('graph_1', 'to', 'graph_2')]["edge_index"][batch_idx+1]

            # clone is essential here
            relevant_hetero_edges = edge_dict[('graph_1', 'to', 'graph_2')][:, start_hetero_edges:end_hetero_edges].clone()

            # adjust for later position in new adjacency matrix , second graph needs to be shifted by the number of nodes in graph_1
            relevant_hetero_edges[1,:] += g1_sizes[batch_idx]

            # adjust for batch position
            relevant_hetero_edges[0,:] -= slice_dict["graph_1"]["x"][batch_idx] # first graph, offset is induced by other first graphs
            relevant_hetero_edges[1,:] -= slice_dict["graph_2"]["x"][batch_idx] # second graph, offset is induced by other first graphs

            # add these edges to the adjacency matrix

            #for i in range(relevant_hetero_edges.shape[1]):
            #    adj[batch_idx, relevant_hetero_edges[0,i], relevant_hetero_edges[1,i]] = 1

            # causes error
            adj[batch_idx, relevant_hetero_edges[0,:], relevant_hetero_edges[1,:]] = 1
            

        return adj
        
    def _merge_graph_nodes(self, x_dict):
        # combine the two graphs x_values need to be concated
        x_global = torch.cat((x_dict["graph_1"], x_dict["graph_2"]), dim=0)
        # combine the adjacency matrices
       
        return x_global


    def _feature_batch_vector(self, x, slice_dict):

        batch_size = len(slice_dict["y"]) -1
        graph_1_slice = slice_dict["graph_1"]["x"]
        graph_1_end = slice_dict["graph_1"]["x"][-1]
        graph_2_slice = slice_dict["graph_2"]["x"]

        x_batch = torch.zeros((batch_size,self.max_nodes, x.shape[1])).to(self.device)

        # slice is longer than batch size
        for i in range(batch_size):
            end_idx_1 =  graph_1_slice[i+1] - graph_1_slice[i]
            end_idx_2 = graph_2_slice[i+1]- graph_2_slice[i]
            x_batch[i, :end_idx_1, :] = x[graph_1_slice[i]:graph_1_slice[i+1], :]
            x_batch[i, end_idx_1: end_idx_1 +end_idx_2 ,:] = x[graph_1_end + graph_2_slice[i]:graph_1_end + graph_2_slice[i+1], :]

        return x_batch
